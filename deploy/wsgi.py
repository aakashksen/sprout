"""Gunicorn entry point. Builds the app from the environment; never loads a fake model in production."""

from __future__ import annotations

import os
from pathlib import Path

from sprout.web import create_app
from sprout.web.config import Config


def build(env: dict[str, str]):
    cfg = Config.from_env(env)
    identifier = None
    if env.get("SPROUT_FAKE_IDENTIFIER") == "1":
        if cfg.ENV != "testing":
            raise RuntimeError("SPROUT_FAKE_IDENTIFIER=1 is only allowed with SPROUT_ENV=testing")
        from sprout.serve.types import Candidate, FakeIdentifier, Identification

        identifier = FakeIdentifier([Identification(
            "answered", Candidate("tomato_early_blight", 0.9),
            [Candidate("tomato_late_blight", 0.06)], "test double, not a model result")])
    # The real Identifier (ONNX) is wired in at T7.4; until then /readyz says "model missing".
    retriever = None
    index = env.get("SPROUT_INDEX_DIR", "")
    if index and (Path(index) / "chunks.jsonl").exists():
        from sprout.rag.retrieve import Retriever

        retriever = Retriever(Path(index))
    return create_app(cfg, identifier=identifier, retriever=retriever)


app = build(dict(os.environ)) if os.environ.get("SPROUT_NO_AUTOBUILD") != "1" else None
