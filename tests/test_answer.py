"""LLM clients and grounded answers, with fakes only: no network, no model, no spending."""
import io
import json
import subprocess
import urllib.request

import pytest

from sprout.llm.client import ClaudeCodeClient, FakeLLM, LLMError, LLMResult, OllamaClient

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_fake_llm_replays_replies_and_records_calls():
    llm = FakeLLM(["one", "two"])
    assert llm.complete(MSGS, max_tokens=10, timeout_s=1).text == "one"
    assert llm.complete(MSGS, max_tokens=10, timeout_s=1).text == "two"
    assert len(llm.calls) == 2 and llm.calls[0] == MSGS


def test_ollama_client_uses_loopback_ip_not_localhost(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["timeout"] = req.full_url, timeout
        seen["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"message": {"content": "ok"}, "eval_count": 3, "prompt_eval_count": 5}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = OllamaClient("qwen3.5:4b-q4_K_M").complete(MSGS, max_tokens=50, timeout_s=7)
    assert seen["url"] == "http://127.0.0.1:11434/api/chat" and "localhost" not in seen["url"]
    assert seen["timeout"] == 7 and seen["body"]["options"]["num_predict"] == 50
    assert seen["body"]["stream"] is False and seen["body"]["think"] is False
    assert out == LLMResult(text="ok", model="qwen3.5:4b-q4_K_M", seconds=out.seconds, prompt_tokens=5, completion_tokens=3)


def test_ollama_failure_is_llm_error(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LLMError):
        OllamaClient("m").complete(MSGS, max_tokens=5, timeout_s=1)


def test_ollama_unload_sends_keep_alive_zero(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OllamaClient("m").unload()
    assert seen["url"].startswith("http://127.0.0.1:11434/") and seen["body"]["keep_alive"] == 0


class Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def claude(tmp_path, monkeypatch, run, cap=5000):
    monkeypatch.setattr("shutil.which", lambda name: "C:/bin/claude.exe")
    monkeypatch.setattr(subprocess, "run", run)
    return ClaudeCodeClient(model="sonnet", cap=cap, counter_path=tmp_path / "claude_calls.json")


def test_claude_command_has_no_tools(tmp_path, monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return Proc(json.dumps({"result": "fine", "is_error": False, "usage": {"input_tokens": 4, "output_tokens": 2}}))

    out = claude(tmp_path, monkeypatch, run).complete(MSGS, max_tokens=50, timeout_s=9)
    cmd = seen["cmd"]
    assert cmd[0] == "C:/bin/claude.exe" and "-p" in cmd and cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json" and cmd[cmd.index("--max-turns") + 1] == "1"
    assert "hi" in seen["input"] and "be brief" in seen["input"]  # the prompt travels on stdin
    assert seen["cwd"] != "." and seen["timeout"] == 9
    assert out.text == "fine" and out.prompt_tokens == 4 and out.completion_tokens == 2
    assert json.loads((tmp_path / "claude_calls.json").read_text())["calls"] == 1


def test_claude_usage_limit_or_nonzero_exit_is_llm_error(tmp_path, monkeypatch):
    with pytest.raises(LLMError):
        claude(tmp_path, monkeypatch, lambda cmd, **kw: Proc("", 1, "boom")).complete(MSGS, max_tokens=5, timeout_s=1)
    limit = json.dumps({"result": "Claude usage limit reached", "is_error": True})
    with pytest.raises(LLMError, match="limit"):
        claude(tmp_path, monkeypatch, lambda cmd, **kw: Proc(limit)).complete(MSGS, max_tokens=5, timeout_s=1)


def test_call_cap_refuses_and_makes_no_call(tmp_path, monkeypatch):
    (tmp_path / "claude_calls.json").write_text(json.dumps({"calls": 3}))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return Proc(json.dumps({"result": "x", "is_error": False}))

    client = claude(tmp_path, monkeypatch, run, cap=3)
    with pytest.raises(LLMError, match="cap"):
        client.complete(MSGS, max_tokens=5, timeout_s=1)
    assert calls == []
    assert json.loads((tmp_path / "claude_calls.json").read_text())["calls"] == 3
