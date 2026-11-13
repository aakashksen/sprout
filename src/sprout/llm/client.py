"""Language-model clients behind one call. Standard library only, so serving never needs a heavy import."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sprout.atomic import atomic_write_text

OLLAMA_URL = "http://127.0.0.1:11434"  # the IP, not "localhost", which adds about 2 s per call here
CLAUDE_CALL_CAP = 5000
CLAUDE_CALLS_PATH = Path(__file__).resolve().parents[3] / "results" / "claude_calls.json"


class LLMError(Exception):
    """The model could not answer: outage, timeout, usage limit, bad output or the call cap."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict], *, max_tokens: int, timeout_s: float) -> LLMResult: ...


class FakeLLM:
    """Replays the given replies in order (the last repeats), or calls `reply(messages)`. For tests only."""

    def __init__(self, replies: list[str] | None = None, *, reply=None, model: str = "fake-llm") -> None:
        self.model = model
        self._replies = list(replies or [])
        self._reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], *, max_tokens: int, timeout_s: float) -> LLMResult:
        self.calls.append(messages)
        if self._reply is not None:
            return LLMResult(self._reply(messages), self.model)
        if not self._replies:
            raise LLMError("FakeLLM has no reply")
        return LLMResult(self._replies.pop(0) if len(self._replies) > 1 else self._replies[0], self.model)


class OllamaClient:
    def __init__(self, model: str, *, base_url: str = OLLAMA_URL, keep_alive: str | int = "5m") -> None:
        self.model = model
        self._url = base_url.rstrip("/")
        self._keep_alive = keep_alive

    def _post(self, path: str, body: dict, timeout_s: float) -> dict:
        req = urllib.request.Request(
            self._url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except (OSError, ValueError) as exc:  # URLError, timeouts, refused connections, bad JSON
            raise LLMError(f"Ollama call failed: {exc}") from exc

    def complete(self, messages: list[dict], *, max_tokens: int, timeout_s: float) -> LLMResult:
        start = time.time()
        body = {
            "model": self.model, "messages": messages, "stream": False, "think": False,
            "keep_alive": self._keep_alive, "options": {"num_predict": max_tokens, "temperature": 0},
        }
        data = self._post("/api/chat", body, timeout_s)
        text = (data.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama returned no text")
        return LLMResult(text.strip(), self.model, round(time.time() - start, 3),
                         data.get("prompt_eval_count"), data.get("eval_count"))

    def placement(self) -> dict | None:
        """Where the loaded model sits, from Ollama's process list: bytes in total and in GPU memory. None if not loaded."""
        try:
            with urllib.request.urlopen(self._url + "/api/ps", timeout=30) as resp:
                loaded = json.loads(resp.read()).get("models", [])
        except (OSError, ValueError) as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc
        for m in loaded:
            if m.get("name") == self.model or m.get("model") == self.model:
                return {"size": m.get("size", 0), "size_vram": m.get("size_vram", 0)}
        return None

    def unload(self) -> None:
        """Free the GPU memory this model holds."""
        self._post("/api/generate", {"model": self.model, "keep_alive": 0}, 30)


_COUNTER_LOCK = threading.Lock()


SAFEGUARD_TEXT = "safeguards flagged this message"
_REQUEST_ID = re.compile(r"req_[A-Za-z0-9]+")


class SafeguardRefusal(LLMError):
    """The claude command refused one request because its safeguards flagged the message. Raised only for that error
    text; every other error stays an LLMError. `request_id` is the request id in the error output, or None."""

    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class ClaudeCodeClient:
    """`claude -p` with every tool, setting and session switched off. Counts calls and refuses past the cap."""

    def __init__(self, model: str = "sonnet", *, cap: int = CLAUDE_CALL_CAP, counter_path: Path | None = None) -> None:
        self.model = model
        self.cap = cap
        self._counter = Path(counter_path or CLAUDE_CALLS_PATH)

    def calls_made(self) -> int:
        try:
            return int(json.loads(self._counter.read_text(encoding="utf-8"))["calls"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def _count_call(self) -> None:
        with _COUNTER_LOCK:
            made = self.calls_made()
            if made >= self.cap:
                raise LLMError(f"Claude call cap of {self.cap} reached; no call was made")
            self._counter.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._counter, json.dumps({"calls": made + 1}))

    def complete(self, messages: list[dict], *, max_tokens: int, timeout_s: float) -> LLMResult:
        exe = shutil.which("claude")
        if exe is None:
            raise LLMError("the claude command was not found")
        self._count_call()
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        cmd = [exe, "-p", "--model", self.model, "--output-format", "json", "--tools", "",
               "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
               "--no-session-persistence", "--max-turns", "1"]
        start = time.time()
        try:
            with tempfile.TemporaryDirectory() as cwd:
                done = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                      timeout=timeout_s, cwd=cwd, env={**os.environ})
        except (OSError, subprocess.SubprocessError) as exc:
            raise LLMError(f"claude call failed: {exc}") from exc
        try:
            data = json.loads(done.stdout)
        except ValueError:
            data = {}
        text = data.get("result") if isinstance(data, dict) else None
        if done.returncode != 0 or data.get("is_error") or not isinstance(text, str) or not text.strip():
            detail = text if isinstance(text, str) and text.strip() else (done.stderr or "").strip()[:200]
            seen = " ".join([str(text or ""), done.stdout or "", done.stderr or ""])
            if SAFEGUARD_TEXT in seen.lower():
                found = _REQUEST_ID.search(seen)
                raise SafeguardRefusal(f"claude returned an error (exit {done.returncode}): {detail}",
                                       found.group(0) if found else None)
            raise LLMError(f"claude returned an error (exit {done.returncode}): {detail}")
        usage = data.get("usage") or {}
        return LLMResult(text.strip(), self.model, round(time.time() - start, 3),
                         usage.get("input_tokens"), usage.get("output_tokens"))
