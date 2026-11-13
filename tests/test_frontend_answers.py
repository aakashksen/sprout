"""Browser tests for generated chat answers, one per answer state. Fakes for the writer and search; offline."""
import threading
import time

import pytest
from playwright.sync_api import sync_playwright
from test_frontend import ANSWER, LIMITS, ask, open_result
from werkzeug.serving import make_server

from sprout.llm.client import FakeLLM, LLMError
from sprout.rag.retrieve import Passage
from sprout.serve.types import FakeIdentifier
from sprout.web import create_app
from sprout.web.config import Config

P1 = Passage("Remove infected leaves.", "tomato.pdf", 3, "tomato", 0.9)


class Search:
    crops = {"tomato"}
    warmed = True

    def __init__(self, passages=(P1,), error=None):
        self.passages, self.error = list(passages), error

    def warm(self, *, rerank=False):
        pass

    def search(self, query, **kw):
        if self.error:
            raise self.error
        return self.passages


class Flaky(FakeLLM):
    """Fails the first call, then answers."""

    def __init__(self, reply):
        super().__init__([reply])
        self.n = 0

    def complete(self, messages, *, max_tokens, timeout_s):
        self.n += 1
        if self.n == 1:
            raise LLMError("down")
        return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(writer, retriever=None):
        app = create_app(Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
                         identifier=FakeIdentifier([ANSWER]), retriever=retriever or Search(), writer=writer)
        deadline = time.time() + 5
        while app.extensions["sprout_chat"]["state"] == "warming" and time.time() < deadline:
            time.sleep(0.02)
        srv = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"

    yield start
    for s in servers:
        s.shutdown()


def reply(browser, serve, tmp_path, writer, retriever=None):
    page = open_result(browser, serve(writer, retriever), tmp_path)
    ask(page, "treatment?")
    page.wait_for_selector("#chat-log .reply .answer-text")
    return page


def test_an_answer_shows_its_text_label_note_and_a_linked_source(browser, serve, tmp_path):
    page = reply(browser, serve, tmp_path, FakeLLM(["Remove infected leaves [1]."]))
    assert "reply-answered" in page.get_attribute("#chat-log .reply", "class")
    text = page.inner_text("#chat-log .answer-text")
    assert "Remove infected leaves [1]." in text and "product label" in text
    assert "tomato.pdf" in page.inner_text("#chat-log .passage .src")


def test_a_question_back_is_shown_with_a_hint(browser, serve, tmp_path):
    page = reply(browser, serve, tmp_path, FakeLLM(["NEED: crop\nWhich plant is it?"]))
    assert "reply-clarify" in page.get_attribute("#chat-log .reply", "class")
    assert "Which plant is it?" in page.inner_text("#chat-log .answer-text")
    assert "next message" in page.inner_text("#chat-log .reply")


def test_unavailable_offers_a_retry_that_then_answers(browser, serve, tmp_path):
    page = reply(browser, serve, tmp_path, Flaky("Remove infected leaves [1]."))
    assert "reply-unavailable" in page.get_attribute("#chat-log .reply", "class")
    assert "unavailable" in page.inner_text("#chat-log .answer-text").lower()
    page.click("#chat-log .reply button")
    page.wait_for_selector("#chat-log .reply-answered")
    assert page.locator("#chat-log .msg-user").count() == 1


def test_a_failed_search_offers_a_retry(browser, serve, tmp_path):
    page = reply(browser, serve, tmp_path, FakeLLM(["x"]), Search(error=RuntimeError("index unreadable")))
    assert "reply-retrieval_failed" in page.get_attribute("#chat-log .reply", "class")
    assert page.locator("#chat-log .reply button").count() == 1


