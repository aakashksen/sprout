"""Measurement-only hooks: per-stage times of a request, reported in a `Server-Timing` response header.

Off unless `SPROUT_MEASUREMENT=1` (see `deploy/wsgi.py`). The hooks wrap the real identifier, retriever and writer and
call them unchanged; each adds two clock reads and a dict update around a call that takes milliseconds (checked in
`tests/test_server_timing.py`). Standard library only, so the serving image needs nothing more.
"""

from __future__ import annotations

import contextvars
import time

STAGES = ("preprocess", "inference", "retrieval", "generation")
TIMED_ENDPOINTS = ("analyses.create", "analyses.chat")


def now_ms() -> float:
    return time.perf_counter() * 1000


class StageTimers:
    """Milliseconds per stage for the request in the current context. The web app hands slow calls to pool threads
    with a copy of the request's context, so their stages land in the same request's record."""

    def __init__(self) -> None:
        self._var: contextvars.ContextVar = contextvars.ContextVar("stage_timers", default=None)

    def begin(self) -> None:
        self._var.set({})

    def add(self, name: str, ms: float) -> None:
        stages = self._var.get()
        if stages is not None:
            stages[name] = stages.get(name, 0.0) + ms

    def note(self, name: str, value) -> None:
        stages = self._var.get()
        if stages is not None:
            stages[name] = value

    def end(self) -> dict | None:
        stages = self._var.get()
        self._var.set(None)
        return stages


def timed(timers: StageTimers, name: str, fn):
    def run(*args, **kwargs):
        start = now_ms()
        try:
            return fn(*args, **kwargs)
        finally:
            timers.add(name, now_ms() - start)

    return run


class _TimedSession:
    """The ONNX Runtime session, with `run` timed as inference."""

    def __init__(self, session, timers: StageTimers) -> None:
        self._session = session
        self.run = timed(timers, "inference", session.run)

    def __getattr__(self, name):
        return getattr(self._session, name)


class TimedRetriever:
    def __init__(self, retriever, timers: StageTimers) -> None:
        self._retriever = retriever
        self.search = timed(timers, "retrieval", retriever.search)

    @property
    def crops(self):
        return self._retriever.crops

    @property
    def warmed(self):
        return self._retriever.warmed

    def warm(self, *, rerank: bool = False) -> None:
        self._retriever.warm(rerank=rerank)


class TimedLLM:
    def __init__(self, llm, timers: StageTimers) -> None:
        self._llm = llm
        self.model = llm.model
        self.complete = timed(timers, "generation", llm.complete)

    def __getattr__(self, name):  # placement(), unload() and the like
        return getattr(self._llm, name)


def instrument_identifier(identifier, timers: StageTimers):
    """Time decoding and resizing as preprocessing and the ONNX run as inference, and note the status the identifier
    returns. Patches in place and returns the identifier; there is no undo, the process is a measurement server."""
    identify = identifier.identify

    def identify_and_note(data: bytes):
        ident = identify(data)
        timers.note("_identification", ident.status)
        return ident

    identifier.identify = identify_and_note
    model = getattr(identifier, "model", None)
    if model is not None:  # a real Identifier; a test double has no model to time
        from sprout.serve import export

        export.preprocess_array = timed(timers, "preprocess", export.preprocess_array)
        model.session = _TimedSession(model.session, timers)
        identifier._decode = timed(timers, "preprocess", identifier._decode)
    return identifier


def header_value(stages: dict) -> str:
    parts = [f"{name};dur={stages[name]:.3f}" for name in STAGES if name in stages]
    if "_identification" in stages:
        parts.append(f'ident;desc="{stages["_identification"]}"')
    return ", ".join(parts)


def parse_header(value: str) -> dict:
    """{'preprocess': ms, ..., 'ident': status} from a Server-Timing header; missing stages are absent."""
    out: dict = {}
    for part in value.split(","):
        name, *params = [x.strip() for x in part.split(";")]
        for p in params:
            if p.startswith("dur="):
                out[name] = float(p[4:])
            elif p.startswith("desc="):
                out[name] = p[5:].strip('"')
    return out


def install(app, timers: StageTimers) -> None:
    """Report stage times of the upload and chat routes in `Server-Timing`."""
    from flask import request

    @app.before_request
    def begin() -> None:
        if request.method == "POST" and request.endpoint in TIMED_ENDPOINTS:
            timers.begin()

    @app.after_request
    def report(resp):
        if request.method == "POST" and request.endpoint in TIMED_ENDPOINTS:
            stages = timers.end()
            if stages:
                resp.headers["Server-Timing"] = header_value(stages)
        return resp
