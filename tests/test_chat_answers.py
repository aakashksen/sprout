"""The chat route calls answer(); the container entry point builds the real Identifier and the writer.
Fakes for the model, the writer and the search only. No Ollama, no network."""
import io
import json
import re
import time

import pytest
import torch
from PIL import Image
from torch import nn

from sprout.llm.client import FakeLLM, LLMError, OllamaClient
from sprout.rag import serving
from sprout.rag.answer import LABEL_NOTE, NO_DOCUMENT, UNAVAILABLE
from sprout.rag.retrieve import Passage
from sprout.serve.export import export_module
from sprout.serve.identify import Identifier
from sprout.serve.types import Candidate, FakeIdentifier, Identification
from sprout.web import create_app
from sprout.web.config import Config
from sprout.web.db import connect

ANSWER = Identification("answered", Candidate("Tomato___Early_blight", 0.9),
                        [Candidate("Potato___Late_blight", 0.06)], "")
LIMITS = {"login": (1000, 60), "signup": (1000, 60), "upload": (1000, 60), "chat": (1000, 60)}
P1 = Passage("Remove infected leaves.", "tomato.pdf", 3, "tomato", 0.9)
P2 = Passage("Water at the base.", "tomato.pdf", 4, "tomato", 0.8)
GENERAL = Passage("Clean your tools.", "general.pdf", 1, "general", 0.8)


class Retriever:
    crops = {"tomato", "potato"}
    warmed = True

    def __init__(self, passages=(P1, P2), error=None):
        self.passages, self.error, self.searches = list(passages), error, []

    def warm(self, *, rerank=False):
        pass

    def search(self, query, **kw):
        self.searches.append({"query": query, **kw})
        if self.error:
            raise self.error
        return self.passages


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 140, 40)).save(buf, "PNG")
    return buf.getvalue()


def make(tmp_path, llm=None, retriever=None, ident=ANSWER, **cfg):
    base = dict(ENV="testing", SECRET_KEY="test-secret", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS)
    app = create_app(Config(**{**base, **cfg}), identifier=FakeIdentifier([ident]),
                     retriever=retriever or Retriever(), writer=llm)
    client = app.test_client()
    def token():
        client.get("/login")
        with client.session_transaction() as s:
            return s["csrf_token"]

    client.post("/signup", data={"username": "alice", "password": "password123", "csrf_token": token()})
    r = client.post("/analyses", data={"file": (io.BytesIO(png()), "leaf.png"), "csrf_token": token()},
                    content_type="multipart/form-data")
    aid = int(re.search(r"/analyses/(\d+)", r.headers["Location"]).group(1))
    deadline = time.time() + 3
    while app.extensions["sprout_chat"]["state"] != "ready" and time.time() < deadline:
        time.sleep(0.01)
    return app, client, aid, {"X-CSRF-Token": token()}


def ask(client, aid, headers, message="How do I treat it?"):
    return client.post(f"/analyses/{aid}/chat", json={"message": message}, headers=headers)


def stored(app, aid):
    con = connect(app.config["DB_PATH"])
    try:
        return [tuple(r) for r in con.execute("select role, content from chat_messages where analysis_id=? order by id", (aid,))]
    finally:
        con.close()


def test_a_generated_answer_carries_text_citations_state_and_the_cited_passages(tmp_path):
    retriever = Retriever()
    app, c, aid, h = make(tmp_path, FakeLLM(["Remove infected leaves [1]."]), retriever)
    r = ask(c, aid, h)
    body = r.get_json()
    assert r.status_code == 200 and body["state"] == "answered" and body["grounded"] is True
    assert body["text"].startswith("Remove infected leaves [1].") and body["text"].endswith(LABEL_NOTE)
    assert [(x["source"], x["page"]) for x in body["citations"]] == [("tomato.pdf", 3)]
    assert [(x["source"], x["page"]) for x in body["passages"]] == [("tomato.pdf", 3)]  # only what was cited
    assert "<p>" in body["html"]
    assert retriever.searches[0]["crop"] == "tomato"
    assert retriever.searches[0]["k"] == serving.K and retriever.searches[0]["rerank"] is serving.RERANK


def test_the_writer_gets_the_stored_diagnosis_the_users_choice_and_recent_turns(tmp_path):
    llm = FakeLLM(["Remove infected leaves [1]."])
    retriever = Retriever()
    app, c, aid, h = make(tmp_path, llm, retriever)
    assert c.post(f"/analyses/{aid}/selection", json={"label": "Potato___Late_blight"}, headers=h).status_code == 200
    ask(c, aid, h, "first question")
    ask(c, aid, h, "second question")
    assert [x["crop"] for x in retriever.searches] == ["potato", "potato"]  # the selected label's plant
    system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert "Potato" in user and "Tomato" in user  # the choice first, the model's own top still shown as another possibility
    assert user.index("Potato") < user.index("Tomato") and "first question" not in user.split("<question>")[0]
    second = llm.calls[1][1]["content"]
    assert "user: first question" in second and "assistant: " in second and "Remove infected leaves [1]." in second
    assert system  # the served prompt


@pytest.mark.parametrize("reply,state,has_note,keeps", [
    ("Remove infected leaves [1].", "answered", True, True),
    ("Remove infected leaves.", "ungrounded", True, True),
    ("NEED: crop\nWhich plant is it?", "clarify", False, True),
    ("NO_SUPPORT", "no_document", False, True),
])
def test_every_answer_state_reaches_the_user_and_is_kept_in_the_history(tmp_path, reply, state, has_note, keeps):
    app, c, aid, h = make(tmp_path, FakeLLM([reply]))
    body = ask(c, aid, h).get_json()
    assert body["state"] == state and body["text"]
    assert body["text"].endswith(LABEL_NOTE) is has_note  # every answer that gives advice carries the note
    if state == "ungrounded":
        assert body["text"].startswith("No supporting document was cited") and body["grounded"] is False
    if state == "no_document":
        assert body["text"] == NO_DOCUMENT and body["citations"] == []
    assert len(stored(app, aid)) == 2


def test_an_empty_search_says_no_document_without_calling_the_writer(tmp_path):
    llm = FakeLLM(["should not be used"])
    app, c, aid, h = make(tmp_path, llm, Retriever(passages=[]))
    body = ask(c, aid, h).get_json()
    assert body["state"] == "no_document" and body["text"] == NO_DOCUMENT and llm.calls == []


def test_general_documents_only_are_said_so(tmp_path):
    app, c, aid, h = make(tmp_path, FakeLLM(["Clean your tools [1]."]), Retriever(passages=[GENERAL]))
    body = ask(c, aid, h).get_json()
    assert body["state"] == "answered" and "no document covers this plant" in body["note"].lower()


class Down(FakeLLM):
    def complete(self, messages, *, max_tokens, timeout_s):
        raise LLMError("Ollama call failed")


@pytest.mark.parametrize("llm,retriever,state", [
    (Down(), None, "unavailable"),
    (FakeLLM(["x"]), Retriever(error=RuntimeError("index unreadable")), "retrieval_failed"),
])
def test_an_outage_is_reported_never_answered_and_leaves_no_trace(tmp_path, llm, retriever, state):
    app, c, aid, h = make(tmp_path, llm, retriever)
    r = ask(c, aid, h)
    body = r.get_json()
    assert r.status_code == 200 and body["state"] == state and body["grounded"] is False and body["citations"] == []
    assert "try again later" in body["text"].lower()
    assert stored(app, aid) == []  # a retry is not a duplicate


def test_a_writer_that_is_too_slow_is_reported_unavailable_within_the_budget(tmp_path):
    class Slow(FakeLLM):
        def complete(self, messages, *, max_tokens, timeout_s):
            time.sleep(0.6)
            return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)

    app, c, aid, h = make(tmp_path, Slow(["Remove infected leaves [1]."]), CHAT_TIMEOUT_SECONDS=0.1,
                          CHAT_LLM_TIMEOUT_SECONDS=0.05)
    start = time.time()
    body = ask(c, aid, h).get_json()
    assert time.time() - start < 0.5
    assert body["state"] == "unavailable" and body["text"] == UNAVAILABLE and stored(app, aid) == []


def test_the_model_call_gets_the_longer_budget_not_the_search_one(tmp_path):
    seen = {}

    class Spy(FakeLLM):
        def complete(self, messages, *, max_tokens, timeout_s):
            seen["timeout_s"] = timeout_s
            return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)

    app, c, aid, h = make(tmp_path, Spy(["Remove infected leaves [1]."]))
    ask(c, aid, h)
    assert seen["timeout_s"] == 40.0 and app.config["API_TIMEOUT_SECONDS"] == 20.0
    assert app.config["CHAT_TIMEOUT_SECONDS"] == 50.0


@pytest.mark.parametrize("payload,status", [({"message": ""}, 400), ({"message": "x" * 5000}, 400), ({}, 400)])
def test_a_rejected_message_leaves_no_trace(tmp_path, payload, status):
    app, c, aid, h = make(tmp_path, FakeLLM(["x"]))
    assert c.post(f"/analyses/{aid}/chat", json=payload, headers=h).status_code == status
    assert stored(app, aid) == []


def test_a_message_sent_before_the_index_is_warm_leaves_no_trace(tmp_path):
    class Cold(Retriever):
        def warm(self, *, rerank=False):
            time.sleep(0.5)

    app, c, aid, h = make(tmp_path, FakeLLM(["x"]), Cold())
    app.extensions["sprout_chat"]["state"] = "warming"
    assert ask(c, aid, h).status_code == 503 and stored(app, aid) == []


def test_readiness_does_not_flip_when_the_writer_is_down_but_says_so(tmp_path):
    app, c, aid, h = make(tmp_path, Down())
    assert c.get("/readyz").get_json()["writer"] == "unknown"
    ask(c, aid, h)
    r = c.get("/readyz")
    assert r.status_code == 200 and r.get_json()["writer"] == "unavailable" and r.get_json()["ready"] is True
    app2, c2, aid2, h2 = make(tmp_path / "ok", FakeLLM(["Remove infected leaves [1]."]))
    ask(c2, aid2, h2)
    assert c2.get("/readyz").get_json()["writer"] == "ok"


def test_without_a_writer_chat_still_returns_cited_passages_only(tmp_path):
    app, c, aid, h = make(tmp_path, None)
    body = ask(c, aid, h).get_json()
    assert "text" not in body and body["passages"][0]["source"] == "tomato.pdf"
    assert c.get("/readyz").get_json()["writer"] == "not configured"


# ---------------------------------------------------------------- the container entry point

CLASS_MAP = {"Tomato___healthy": 0, "Tomato___Early_blight": 1}


def model_files(folder):
    torch.manual_seed(1)
    net = nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Flatten(), nn.Linear(192, 2), nn.Identity()).eval()
    model = export_module(net, CLASS_MAP, folder / "model.onnx", source_sha256="b" * 64)
    calibration = folder / "chosen.json"
    gate = {"kind": "confidence", "confidence_threshold": 0.0}
    calibration.write_text(json.dumps({"temperature": 1.0, "abstain_threshold": 0.0, "model_sha256": "b" * 64,
                                       "classes": list(CLASS_MAP), "gate": {"chosen": "confidence", "settings": gate}}), encoding="utf-8")
    return model, calibration


@pytest.fixture
def wsgi(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("SPROUT_NO_AUTOBUILD", "1")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "deploy"))
    import wsgi as module

    return module


def env(tmp_path, **extra):
    return {"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_DATA_DIR": str(tmp_path / "data"), **extra}


def test_the_real_identifier_is_built_when_the_model_and_calibration_exist(wsgi, tmp_path):
    model, calibration = model_files(tmp_path)
    app = wsgi.build(env(tmp_path, SPROUT_MODEL_PATH=str(model), SPROUT_CALIBRATION_PATH=str(calibration)))
    assert isinstance(app.extensions["sprout_identifier"], Identifier)
    assert app.test_client().get("/readyz").status_code == 200


def test_a_missing_model_or_calibration_reports_model_missing(wsgi, tmp_path):
    model, calibration = model_files(tmp_path)
    for paths in ({}, {"SPROUT_MODEL_PATH": str(model)}, {"SPROUT_MODEL_PATH": str(tmp_path / "none.onnx"),
                                                          "SPROUT_CALIBRATION_PATH": str(calibration)}):
        app = wsgi.build(env(tmp_path, **paths))
        assert app.extensions["sprout_identifier"] is None
        r = app.test_client().get("/readyz")
        assert r.status_code == 503 and "model missing" in r.get_json()["problems"]


def test_the_fake_identifier_is_never_on_by_default_and_never_replaces_a_real_model_silently(wsgi, tmp_path):
    model, calibration = model_files(tmp_path)
    paths = {"SPROUT_MODEL_PATH": str(model), "SPROUT_CALIBRATION_PATH": str(calibration)}
    app = wsgi.build(env(tmp_path, SPROUT_ENV="testing", **paths))  # testing, flag unset: the real one
    assert isinstance(app.extensions["sprout_identifier"], Identifier)
    for flag in ("0", "", "true", "yes"):
        assert not isinstance(wsgi.build(env(tmp_path, SPROUT_ENV="testing", SPROUT_FAKE_IDENTIFIER=flag, **paths))
                              .extensions["sprout_identifier"], FakeIdentifier)
    with pytest.raises(RuntimeError, match="only allowed"):
        wsgi.build(env(tmp_path, SPROUT_FAKE_IDENTIFIER="1", **paths))  # SPROUT_ENV defaults to production


def test_the_writer_is_ollama_on_the_loopback_ip_with_the_served_config(wsgi, tmp_path):
    app = wsgi.build(env(tmp_path))
    writer, cfg = app.extensions["sprout_writer"], app.extensions["sprout_answer_config"]
    assert isinstance(writer, OllamaClient) and writer._url == "http://127.0.0.1:11434"
    assert (cfg.prompt_version, cfg.mode, cfg.rerank, cfg.k) == (serving.PROMPT_VERSION, serving.MODE, serving.RERANK, serving.K)
    assert cfg.timeout_s == 40.0
    other = wsgi.build(env(tmp_path, SPROUT_OLLAMA_URL="http://host.docker.internal:11434", SPROUT_WRITER_MODEL="m:1"))
    assert other.extensions["sprout_writer"]._url == "http://host.docker.internal:11434"
    assert other.extensions["sprout_writer"].model == "m:1"


def test_a_localhost_writer_address_is_refused(wsgi, tmp_path):
    with pytest.raises(RuntimeError, match="127.0.0.1"):
        wsgi.build(env(tmp_path, SPROUT_OLLAMA_URL="http://localhost:11434"))


def test_the_request_context_reaches_the_threads_that_identify_and_write():
    """Timing hooks (E8) and logging read a context variable set by the request; the pool threads must see it."""
    import contextvars
    import tempfile
    from pathlib import Path

    var = contextvars.ContextVar("seen", default=None)
    seen = {}

    class Ident(FakeIdentifier):
        def identify(self, image_bytes):
            seen["identify"] = var.get()
            return super().identify(image_bytes)

    class Writer(FakeLLM):
        def complete(self, messages, *, max_tokens, timeout_s):
            seen["complete"] = var.get()
            return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Config(ENV="testing", SECRET_KEY="s", DATA_DIR=tmp, RATE_LIMITS=LIMITS),
                         identifier=Ident([ANSWER]), retriever=Retriever(), writer=Writer(["Remove infected leaves [1]."]))
        c = app.test_client()
        c.get("/login")
        with c.session_transaction() as s:
            t = s["csrf_token"]
        c.post("/signup", data={"username": "alice", "password": "password123", "csrf_token": t})
        c.get("/login")
        with c.session_transaction() as s:
            t = s["csrf_token"]
        var.set("this request")
        r = c.post("/analyses", data={"file": (io.BytesIO(png()), "leaf.png"), "csrf_token": t},
                   content_type="multipart/form-data")
        aid = int(re.search(r"/analyses/(\d+)", r.headers["Location"]).group(1))
        deadline = time.time() + 3
        while app.extensions["sprout_chat"]["state"] != "ready" and time.time() < deadline:
            time.sleep(0.01)
        c.post(f"/analyses/{aid}/chat", json={"message": "treatment?"}, headers={"X-CSRF-Token": t})
        assert Path(tmp).exists()
    assert seen == {"identify": "this request", "complete": "this request"}


def test_an_unsure_photo_still_searches_the_plant_its_label_names(tmp_path):
    unsure = Identification("unsure", Candidate("Potato___Late_blight", 0.3), [], "retake it")
    retriever = Retriever()
    app, c, aid, h = make(tmp_path, FakeLLM(["Remove infected leaves [1]."]), retriever, ident=unsure)
    ask(c, aid, h, "Is it a tomato problem?")
    assert retriever.searches[0]["crop"] == "potato"  # the photo's plant, not the "tomato" in the words


def test_the_container_path_imports_no_torch_or_pandas():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import os, sys; os.environ['SPROUT_NO_AUTOBUILD']='1'; sys.path.insert(0, 'deploy'); "
        "import wsgi, sprout.serve.identify, sprout.llm.client, sprout.rag.answer, sprout.experiments.e8_perf; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'torchvision', 'pandas', 'scipy', 'sklearn')))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
