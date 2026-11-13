"""A healthy label never promises the plant is fine, at any confidence, on the page or in generated answers."""
import io
import re

import pytest
from PIL import Image

from sprout.llm.client import FakeLLM
from sprout.rag.answer import answer, diagnosis_summary
from sprout.rag.retrieve import Passage
from sprout.serve import wording
from sprout.serve.types import Candidate, FakeIdentifier, Identification
from sprout.web import create_app
from sprout.web.config import Config

LIMITS = {"login": (1000, 60), "signup": (1000, 60), "upload": (1000, 60), "chat": (1000, 60)}
P1 = Passage("Water at the base.", "tomato.pdf", 3, "tomato", 0.9)
PROMISES = re.compile(r"\b(is fine|are fine|no problem|nothing wrong|all clear|safe|disease[- ]free|definitely healthy)\b", re.I)


def healthy(conf):
    return Identification("answered", Candidate("Tomato___healthy", conf), [Candidate("Tomato___Early_blight", 0.01)], "")


def test_the_wording_is_one_sentence_that_rules_nothing_out_and_promises_nothing():
    assert "does not rule out disease" in wording.HEALTHY_CAUTION and "any confidence" in wording.HEALTHY_CAUTION
    assert not PROMISES.search(wording.HEALTHY_CAUTION)
    assert wording.is_healthy("Tomato___healthy") and not wording.is_healthy("Tomato___Early_blight")


def page(tmp_path, ident, path="/analyses/1"):
    app = create_app(Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
                     identifier=FakeIdentifier([ident]))
    c = app.test_client()

    def token():
        c.get("/login")
        with c.session_transaction() as s:
            return s["csrf_token"]

    c.post("/signup", data={"username": "alice", "password": "password123", "csrf_token": token()})
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 140, 40)).save(buf, "PNG")
    c.post("/analyses", data={"file": (io.BytesIO(buf.getvalue()), "a.png"), "csrf_token": token()},
           content_type="multipart/form-data")
    return c, app


@pytest.mark.parametrize("conf", [0.55, 0.9, 0.999])
def test_the_result_page_and_history_say_it_at_every_confidence(tmp_path, conf):
    c, _ = page(tmp_path, healthy(conf))
    for url in ("/analyses/1", "/history"):
        body = c.get(url).get_data(as_text=True)
        assert wording.HEALTHY_CAUTION in body, url
        assert not PROMISES.search(body.replace(wording.HEALTHY_CAUTION, "")), url
    assert "bi-check-circle" not in c.get("/analyses/1").get_data(as_text=True)  # no green tick


def test_a_diagnosis_page_does_not_carry_the_healthy_caution(tmp_path):
    c, _ = page(tmp_path, Identification("answered", Candidate("Tomato___Early_blight", 0.95), [], ""))
    assert wording.HEALTHY_CAUTION not in c.get("/analyses/1").get_data(as_text=True)


class Search:
    crops = {"tomato"}

    def search(self, query, **kw):
        return [P1]


@pytest.mark.parametrize("conf", [0.6, 0.99])
def test_generated_answers_carry_the_caution_when_the_photo_was_read_as_healthy(conf):
    ident = healthy(conf)
    assert wording.HEALTHY_CAUTION in diagnosis_summary(ident)  # the writer is told
    # even if the writer says the plant is fine, the reader still gets the caution, once
    out = answer("Is my plant okay?", identification=ident, history=[], retriever=Search(),
                 llm=FakeLLM(["Your plant looks fine [1]."]))
    assert out.text.count(wording.HEALTHY_CAUTION) == 1 and out.state == "answered"


def test_a_user_who_selects_healthy_gets_the_caution_in_chat(tmp_path):
    from test_chat_answers import Retriever, ask, make  # noqa: PLC0415

    sick = Identification("answered", Candidate("Tomato___Early_blight", 0.9), [Candidate("Tomato___healthy", 0.05)], "")
    app, c, aid, h = make(tmp_path, FakeLLM(["Water at the base [1]."]), Retriever(), ident=sick)
    assert c.post(f"/analyses/{aid}/selection", json={"label": "Tomato___healthy"}, headers=h).status_code == 200
    assert wording.HEALTHY_CAUTION in ask(c, aid, h).get_json()["text"]
    assert wording.HEALTHY_CAUTION in c.get(f"/analyses/{aid}").get_data(as_text=True)


def test_other_diagnoses_and_answers_without_a_photo_are_unchanged():
    sick = Identification("answered", Candidate("Tomato___Early_blight", 0.9), [], "")
    assert wording.HEALTHY_CAUTION not in diagnosis_summary(sick)
    out = answer("Rings on leaves?", identification=sick, history=[], retriever=Search(), llm=FakeLLM(["Remove leaves [1]."]))
    assert wording.HEALTHY_CAUTION not in out.text
    assert wording.HEALTHY_CAUTION not in diagnosis_summary(None)
