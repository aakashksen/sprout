"""Measurement mode: stage times in a Server-Timing header, rate limits lifted, nothing else changed."""
import io
import re
import time

import pytest
from test_chat_answers import Retriever, model_files, png

from sprout.llm.client import FakeLLM
from sprout.serve import timing
from sprout.web.config import Config


@pytest.fixture
def wsgi(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("SPROUT_NO_AUTOBUILD", "1")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "deploy"))
    import wsgi as module

    return module


class SlowWriter(FakeLLM):
    def complete(self, messages, *, max_tokens, timeout_s):
        time.sleep(0.05)
        return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)


def app_for(wsgi, tmp_path, **extra):
    model, calibration = model_files(tmp_path)
    env = {"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_DATA_DIR": str(tmp_path / "data"), "SPROUT_MODEL_PATH": str(model),
           "SPROUT_CALIBRATION_PATH": str(calibration), **extra}
    app = wsgi.build(env, retriever=Retriever(), writer=SlowWriter(["Remove infected leaves [1]."], model="tiny-writer"))
    deadline = time.time() + 3
    while app.extensions["sprout_chat"]["state"] != "ready" and time.time() < deadline:
        time.sleep(0.01)
    return app


def drive(app):
    c = app.test_client()

    def token():
        c.get("/login")
        with c.session_transaction() as s:
            return s["csrf_token"]

    c.post("/signup", data={"username": "alice", "password": "password123", "csrf_token": token()})
    t = token()
    up = c.post("/analyses", data={"file": (io.BytesIO(png()), "leaf.png"), "csrf_token": t},
                content_type="multipart/form-data")
    aid = re.search(r"/analyses/(\d+)", up.headers["Location"]).group(1)
    chat = c.post(f"/analyses/{aid}/chat", json={"message": "treatment?"}, headers={"X-CSRF-Token": t})
    return c, up, chat


def test_measurement_mode_reports_every_stage_of_the_real_path(wsgi, tmp_path):
    app = app_for(wsgi, tmp_path, SPROUT_MEASUREMENT="1")
    c, up, chat = drive(app)
    first = timing.parse_header(up.headers["Server-Timing"])
    second = timing.parse_header(chat.headers["Server-Timing"])
    assert first["ident"] == "answered" and first["preprocess"] > 0 and first["inference"] > 0
    assert "generation" not in first and "retrieval" not in first
    assert second["retrieval"] > 0 and second["generation"] >= 50
    assert "Server-Timing" not in c.get("/history").headers  # only the two measured routes
    assert c.get("/readyz").get_json()["measurement"] is True


def test_the_header_and_the_limits_are_off_unless_asked_for(wsgi, tmp_path):
    app = app_for(wsgi, tmp_path)
    c, up, chat = drive(app)
    assert "Server-Timing" not in up.headers and "Server-Timing" not in chat.headers
    assert c.get("/readyz").get_json()["measurement"] is False
    assert Config.from_env({"SPROUT_SECRET_KEY": "x" * 32}).RATE_LIMITS["signup"] == (5, 3600)
    lifted = Config.from_env({"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_MEASUREMENT": "1"}).RATE_LIMITS
    assert all(limit[0] >= 10**9 for limit in lifted.values())
    assert not Config.from_env({"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_MEASUREMENT": "true"}).MEASUREMENT


def test_the_answer_is_the_same_with_the_hooks_on(wsgi, tmp_path):
    plain = drive(app_for(wsgi, tmp_path / "a"))[2].get_json()
    hooked = drive(app_for(wsgi, tmp_path / "b", SPROUT_MEASUREMENT="1"))[2].get_json()
    assert plain == hooked


def test_header_round_trip():
    stages = {"preprocess": 1.5, "inference": 2.25, "generation": 300.0, "_identification": "answered"}
    assert timing.parse_header(timing.header_value(stages)) == {
        "preprocess": 1.5, "inference": 2.25, "generation": 300.0, "ident": "answered"}


def test_a_hook_costs_microseconds_against_stages_of_milliseconds():
    """How the hooks are shown to be negligible: the cost per wrapped call, with no work inside, against the
    milliseconds a real stage takes (a resize is about 1 ms, an ONNX run more, a search 10 ms or more)."""
    timers = timing.StageTimers()
    timers.begin()
    calls = 20000
    bare = lambda: None  # noqa: E731
    wrapped = timing.timed(timers, "inference", bare)
    start = time.perf_counter()
    for _ in range(calls):
        wrapped()
    per_call_us = (time.perf_counter() - start) / calls * 1e6
    assert per_call_us < 50, per_call_us
