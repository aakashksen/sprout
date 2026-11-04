"""Gunicorn entry point. Builds the app from the environment; never loads a fake model in production."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from sprout.web import create_app
from sprout.web.config import Config

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"  # the IP, not "localhost", which adds about 2 s per call here
DEFAULT_WRITER = "qwen3.5:4b-q4_K_M"  # the writer E7 chose; see docs/DECISIONS.md


def _identifier(env: dict[str, str], cfg: Config):
    """The test double only when SPROUT_FAKE_IDENTIFIER is exactly "1" under SPROUT_ENV=testing; otherwise the real
    ONNX identifier when both files exist; otherwise None, and /readyz says "model missing"."""
    if env.get("SPROUT_FAKE_IDENTIFIER") == "1":
        if cfg.ENV != "testing":
            raise RuntimeError("SPROUT_FAKE_IDENTIFIER=1 is only allowed with SPROUT_ENV=testing")
        from sprout.serve.types import Candidate, FakeIdentifier, Identification

        return FakeIdentifier([Identification(
            "answered", Candidate("tomato_early_blight", 0.9),
            [Candidate("tomato_late_blight", 0.06)], "test double, not a model result")])
    model = Path(env.get("SPROUT_MODEL_PATH", "/models/model.onnx"))
    calibration = Path(env.get("SPROUT_CALIBRATION_PATH", "/models/chosen.json"))
    if model.is_file() and calibration.is_file():
        from sprout.serve.identify import Identifier

        return Identifier(model, calibration)
    return None


def _writer(env: dict[str, str]):
    from sprout.llm.client import OllamaClient

    url = env.get("SPROUT_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    if (urlparse(url).hostname or "").lower() == "localhost":
        raise RuntimeError("SPROUT_OLLAMA_URL must use 127.0.0.1, not localhost, which adds about 2 s per call")
    return OllamaClient(env.get("SPROUT_WRITER_MODEL", DEFAULT_WRITER), base_url=url, keep_alive="30m")


def build(env: dict[str, str]):
    cfg = Config.from_env(env)
    retriever = None
    index = env.get("SPROUT_INDEX_DIR", "")
    if index and (Path(index) / "chunks.jsonl").exists():
        from sprout.rag import serving
        from sprout.rag.retrieve import Retriever

        retriever = Retriever(Path(index), config=serving.retrieval_config())
    # The writer is always built; an unreachable Ollama shows up as "unavailable" in chat, never as an answer.
    return create_app(cfg, identifier=_identifier(env, cfg), retriever=retriever, writer=_writer(env))


app = build(dict(os.environ)) if os.environ.get("SPROUT_NO_AUTOBUILD") != "1" else None
