"""Browser tests: a real server on a local port and a headless Chromium. Offline."""

import io
import threading
import time

import pytest
from PIL import Image
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from sprout.rag.retrieve import Passage
from sprout.serve.types import Candidate, FakeIdentifier, Identification
from sprout.web import create_app
from sprout.web.config import Config
from sprout.web.db import connect

ANSWER = Identification(
    "answered",
    Candidate("tomato_early_blight", 0.9),
    [Candidate("tomato_late_blight", 0.06), Candidate("tomato_healthy", 0.02)],
    "",
)
LIMITS = {"login": (1000, 60), "signup": (1000, 60), "upload": (1000, 60), "chat": (1000, 60)}
EVIL = '<img src=x onerror="window.pwned=1"><script>window.pwned=1</script>'


class FakeRetriever:
    crops = {"tomato"}
    warmed = True

    def __init__(self, passages=(), delay=0.0, fail_first=0):
        self.passages, self.delay, self.fail_first = list(passages), delay, fail_first
        self.calls = 0

    def warm(self, *, rerank=False):
        pass

    def search(self, query, *, crop=None, disease=None, k=8, mode="hybrid", rerank=True):
        self.calls += 1
        time.sleep(self.delay)
        if self.calls <= self.fail_first:
            raise RuntimeError("boom")
        return self.passages


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 140, 40)).save(buf, "PNG")
    return buf.getvalue()


def _make_app(tmp_path, identifier=None, retriever=None, wait=True, **cfg):
    cfgd = dict(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS)
    cfgd.update(cfg)
    app = create_app(Config(**cfgd), identifier=identifier or FakeIdentifier([ANSWER]), retriever=retriever)
    if retriever is not None and wait:
        deadline = time.time() + 5
        while app.extensions["sprout_chat"]["state"] == "warming" and time.time() < deadline:
            time.sleep(0.02)
    return app


@pytest.fixture(scope="module")
def playwright_():
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="module")
def browser(playwright_):
    b = playwright_.chromium.launch()
    yield b
    b.close()


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(identifier=None, retriever=None, **cfg):
        app = _make_app(tmp_path, identifier, retriever, **cfg)
        srv = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return app, f"http://127.0.0.1:{srv.server_port}"

    yield start
    for s in servers:
        s.shutdown()


def open_result(browser, url, tmp_path):
    photo = tmp_path / "leaf.png"
    photo.write_bytes(_png())
    page = browser.new_page()
    page.goto(url + "/signup")
    page.fill("#username", "alice")
    page.fill("#password", "password123")
    page.click("text=Create account")
    page.goto(url + "/upload")
    page.set_input_files("#file", str(photo))
    page.click("#uploadBtn")
    page.wait_for_selector("#prediction")
    return page


def ask(page, text):
    if not page.locator("#chat-form").is_visible():
        page.click("#open-chat")
        page.wait_for_selector("#chat-form", state="visible")
    page.fill("#chat-message", text)
    page.click("#chat-send")


def test_script_in_user_message_is_not_executed(browser, serve, tmp_path):
    _, url = serve(retriever=FakeRetriever())
    page = open_result(browser, url, tmp_path)
    ask(page, EVIL)
    page.wait_for_selector("#chat-log .reply .note")
    assert page.evaluate("window.pwned") is None
    assert page.inner_text("#chat-log .msg-user") == EVIL
    assert page.locator("#chat-log img, #chat-log script").count() == 0


def test_script_in_model_markdown_is_removed(browser, serve, tmp_path):
    text = f"{EVIL}\n\nUse **clean** tools. [more](javascript:window.pwned=1) [ok](https://example.org/a)"
    _, url = serve(retriever=FakeRetriever([Passage(text, "tomato.pdf", 2, "tomato", 0.9)]))
    page = open_result(browser, url, tmp_path)
    ask(page, "what now?")
    page.wait_for_selector("#chat-log .passage")
    log = page.locator("#chat-log")
    assert log.locator("strong").inner_text() == "clean"
    assert log.locator("script, img, [onerror]").count() == 0
    assert log.locator("a[href^='javascript']").count() == 0
    assert page.evaluate("window.pwned") is None
    assert "<script" not in log.inner_html().lower()


def test_unsure_result_shows_guidance(browser, serve, tmp_path):
    unsure = Identification(
        "unsure", None, [Candidate("tomato_early_blight", 0.4), Candidate("tomato_late_blight", 0.3)],
        "Retake the photo in daylight.",
    )
    _, url = serve(identifier=FakeIdentifier([unsure]))
    page = open_result(browser, url, tmp_path)
    assert "Unsure" in page.inner_text("#prediction")
    assert page.inner_text("#guidance") == "Retake the photo in daylight."
    assert "tomato early blight" in page.inner_text("#alternatives")
    assert page.locator("input[name=choice]").count() == 2
    assert page.locator("progress").count() == 0


def test_not_plant_result_offers_no_chat(browser, serve, tmp_path):
    _, url = serve(identifier=FakeIdentifier([Identification("not_plant", None, [], "No leaf found.")]))
    page = open_result(browser, url, tmp_path)
    assert "Not a plant" in page.inner_text("#prediction")
    assert page.locator("#chat-form").count() == 0


def test_answered_result_shows_confidence_and_choice(browser, serve, tmp_path):
    _, url = serve()
    page = open_result(browser, url, tmp_path)
    assert "90% confidence" in page.inner_text("#prediction")
    page.check("input[value=tomato_late_blight]")
    expect(page.locator("#selection-status")).to_have_text("Saved.")


def test_sources_are_links(browser, serve, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file,crop,diseases,title,publisher,source_url,date\n"
        "tomato.pdf,tomato,early blight,Tomato Guide,Uni,https://example.org/tomato.pdf,2020\n"
        "local.pdf,tomato,,Local Notes,Uni,file:///etc/passwd,2020\n",
        encoding="utf-8",
    )
    passages = [
        Passage("Remove infected leaves.", "tomato.pdf", 3, "tomato", 0.9),
        Passage("Rotate crops.", "local.pdf", 1, "tomato", 0.8),
    ]
    _, url = serve(retriever=FakeRetriever(passages), MANIFEST=str(manifest))
    page = open_result(browser, url, tmp_path)
    ask(page, "treatment?")
    page.wait_for_selector("#chat-log .passage")
    link = page.locator("#chat-log .src a")
    assert link.count() == 1
    assert link.get_attribute("href") == "https://example.org/tomato.pdf"
    assert link.inner_text() == "Tomato Guide"
    assert "Local Notes" in page.locator("#chat-log .src").nth(1).inner_text()
    assert "product label" in page.inner_text("#chat-log .note")


def test_no_supporting_document_is_labelled(browser, serve, tmp_path):
    _, url = serve(retriever=FakeRetriever([]))
    page = open_result(browser, url, tmp_path)
    ask(page, "treatment?")
    page.wait_for_selector("#chat-log .nosrc")
    assert "No supporting document" in page.inner_text("#chat-log .nosrc")


def test_failed_chat_can_be_retried(browser, serve, tmp_path):
    passages = [Passage("Remove infected leaves.", "tomato.pdf", 3, "tomato", 0.9)]
    _, url = serve(retriever=FakeRetriever(passages, fail_first=1))
    page = open_result(browser, url, tmp_path)
    ask(page, "treatment?")
    page.wait_for_selector("#chat-log .reply .error")
    page.click("#chat-log .reply button")
    page.wait_for_selector("#chat-log .passage")
    assert page.locator("#chat-log .msg-user").count() == 1


def test_report_is_stored(browser, serve, tmp_path):
    app, url = serve()
    page = open_result(browser, url, tmp_path)
    page.click("#report-open")
    page.wait_for_selector("#report-note", state="visible")
    page.fill("#report-note", "It is a late blight leaf")
    page.click("#report-send")
    expect(page.locator("#report-status")).to_contain_text("Report stored")
    con = connect(app.config["DB_PATH"])
    rows = con.execute("select note, shown_label from reports").fetchall()
    con.close()
    assert [(r["note"], r["shown_label"]) for r in rows] == [("It is a late blight leaf", "tomato_early_blight")]


def test_double_click_sends_once(browser, serve, tmp_path):
    retriever = FakeRetriever(delay=0.4)
    app, url = serve(retriever=retriever)
    page = open_result(browser, url, tmp_path)
    page.click("#open-chat")
    page.wait_for_selector("#chat-form", state="visible")
    page.fill("#chat-message", "treatment?")
    page.dblclick("#chat-send")
    page.wait_for_selector("#chat-log .reply .note")
    assert retriever.calls == 1
    con = connect(app.config["DB_PATH"])
    n = con.execute("select count(*) from chat_messages").fetchone()[0]
    con.close()
    assert n == 1
    assert page.locator("#chat-log .msg-user").count() == 1


def _signed_in_client(app):
    c = app.test_client()
    c.get("/signup")
    with c.session_transaction() as s:
        t = s["csrf_token"]
    c.post("/signup", data={"username": "alice", "password": "password123", "csrf_token": t})
    c.get("/")
    with c.session_transaction() as s:
        t = s["csrf_token"]
    c.post("/analyses", data={"file": (io.BytesIO(_png()), "a.png"), "csrf_token": t},
           content_type="multipart/form-data")
    return c, {"X-CSRF-Token": t}


def test_rejected_chat_message_is_not_stored(tmp_path):
    class Cold(FakeRetriever):
        def warm(self, *, rerank=False):
            time.sleep(1)

    app = _make_app(tmp_path, retriever=Cold(), wait=False)
    c, h = _signed_in_client(app)
    assert c.post("/analyses/1/chat", json={"message": "hi"}, headers=h).status_code == 503
    con = connect(app.config["DB_PATH"])
    assert con.execute("select count(*) from chat_messages").fetchone()[0] == 0
    con.close()


def test_general_fallback_for_known_crop_says_so(tmp_path):
    general = Passage("Wash tools.", "ipm.pdf", 1, "general", 0.9)
    app = _make_app(tmp_path, retriever=FakeRetriever([general]))
    c, h = _signed_in_client(app)
    body = c.post("/analyses/1/chat", json={"message": "hi"}, headers=h).get_json()
    assert "only general guidance" in body["note"] and "product label" in body["note"]


def watch(page):
    """Collect console errors, failed requests, and every host the page asks for."""
    seen = {"errors": [], "failed": [], "hosts": set()}
    page.on("console", lambda m: m.type == "error" and seen["errors"].append(m.text))
    page.on("pageerror", lambda e: seen["errors"].append(str(e)))
    page.on("requestfailed", lambda r: seen["failed"].append(r.url))
    page.on("request", lambda r: seen["hosts"].add(r.url.split("/")[2] if "://" in r.url else r.url[:5]))
    return seen


def test_pages_render_under_csp_with_no_errors_and_no_network(browser, serve, tmp_path):
    app, url = serve(retriever=FakeRetriever())
    page = browser.new_page()
    seen = watch(page)
    host = url.split("//")[1]
    visited = []
    for path in ("/", "/login", "/signup"):
        page.goto(url + path)
        visited.append(path)
    page.fill("#username", "alice")
    page.fill("#password", "password123")
    with page.expect_navigation(wait_until="load"):  # let the next page's scripts finish before leaving it
        page.click("text=Create account")
    photo = tmp_path / "leaf.png"
    photo.write_bytes(_png())
    page.goto(url + "/upload")
    page.set_input_files("#file", str(photo))
    page.click("#uploadBtn")
    page.wait_for_selector("#prediction")
    for path in ("/", "/upload", "/dashboard", "/history"):
        page.goto(url + path)
    page.goto(url + "/analyses/1")
    page.click("#open-chat")
    page.wait_for_selector("#chat-form", state="visible")
    assert seen["errors"] == [], seen["errors"]
    assert seen["failed"] == [], seen["failed"]
    assert seen["hosts"] <= {host, "data:", "blob:"}, seen["hosts"]
    missing = browser.new_page()
    seen404 = watch(missing)
    assert missing.goto(url + "/nope").status == 404
    assert "Something went wrong" in missing.inner_text("h1")
    assert seen404["errors"] == ["Failed to load resource: the server responded with a status of 404 (NOT FOUND)"]
    assert seen404["failed"] == [] and seen404["hosts"] <= {host}
    # every style, font and script the pages need was served from this app
    assert page.evaluate("document.fonts.check('16px Poppins')") is True
    assert page.locator("i.bi").first.evaluate("e => getComputedStyle(e, '::before').fontFamily").strip('"') == "bootstrap-icons"


def test_dashboard_and_history_show_database_values(browser, serve, tmp_path):
    _, url = serve()
    page = open_result(browser, url, tmp_path)
    page.goto(url + "/dashboard")
    assert "Welcome, alice!" in page.inner_text("h1")
    assert "1 saved analysis" in page.inner_text(".page-header")
    page.goto(url + "/history")
    assert "model said tomato early blight" in page.inner_text(".card-title")
    assert page.locator("img.history-image").count() == 1


def test_camera_page_takes_a_photo_and_analyses_it(playwright_, serve):
    _, url = serve()
    b = playwright_.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
    try:
        page = b.new_context(permissions=["camera"]).new_page()
        seen = watch(page)
        page.goto(url + "/signup")
        page.fill("#username", "alice")
        page.fill("#password", "password123")
        page.click("text=Create account")
        page.goto(url + "/camera")
        expect(page.locator("#captureBtn")).to_be_enabled()
        for _ in range(100):  # wait_for_function evals a string, which the CSP forbids
            if page.evaluate("document.getElementById('video').videoWidth") > 0:
                break
            time.sleep(0.1)
        page.click("#captureBtn")
        page.click("#analyzeBtn")
        page.wait_for_selector("#prediction")
        assert page.url.endswith("/analyses/1")
        assert "90% confidence" in page.inner_text("#prediction")
        assert seen["errors"] == [], seen["errors"]
    finally:
        b.close()


def test_camera_page_without_a_camera_says_so(browser, serve, tmp_path):
    _, url = serve()
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(url + "/signup")
    page.fill("#username", "alice")
    page.fill("#password", "password123")
    page.click("text=Create account")
    page.goto(url + "/camera")
    expect(page.locator("#camera-message")).to_contain_text("Upload a photo instead")


def test_upload_page_needs_login(browser, serve):
    _, url = serve()
    page = browser.new_page()
    page.goto(url + "/upload")
    assert "/login" in page.url


def test_default_manifest_is_used_when_the_variable_is_unset(tmp_path):
    from sprout.web.config import DEFAULT_MANIFEST

    assert DEFAULT_MANIFEST.exists()
    assert Config().MANIFEST == str(DEFAULT_MANIFEST)
    cfg = Config.from_env({"SPROUT_ENV": "testing", "SPROUT_SECRET_KEY": "x"})
    assert cfg.MANIFEST == str(DEFAULT_MANIFEST)
    assert Config.from_env({"SPROUT_ENV": "testing", "SPROUT_SECRET_KEY": "x", "SPROUT_MANIFEST": "other.csv"}).MANIFEST == "other.csv"
    app = create_app(Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path)), identifier=None)
    sources = app.extensions["sprout_sources"]
    assert sources and all(url == "" or url.startswith("http") for _, url in sources.values())
