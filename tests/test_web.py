import io
import re
import time

import pytest
from PIL import Image

from sprout.serve.types import Candidate, FakeIdentifier, Identification
from sprout.web import create_app
from sprout.web.config import Config

ANSWER = Identification(
    "answered",
    Candidate("tomato_early_blight", 0.9),
    [Candidate("tomato_late_blight", 0.06), Candidate("tomato_healthy", 0.02)],
    "see the source",
)

LIMITS = {"login": (1000, 60), "signup": (1000, 60), "upload": (1000, 60), "chat": (1000, 60)}


class Counting(FakeIdentifier):
    def __init__(self, results):
        super().__init__(results)
        self.calls = 0

    def identify(self, image_bytes):
        self.calls += 1
        return super().identify(image_bytes)


def png_bytes(color=(30, 140, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def make_app(tmp_path, identifier="default", **cfg):
    if identifier == "default":
        identifier = Counting([ANSWER])
    base = dict(ENV="testing", SECRET_KEY="test-secret", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS)
    base.update(cfg)
    return create_app(Config(**base), identifier=identifier)


def token(client) -> str:
    client.get("/login")
    with client.session_transaction() as s:
        return s["csrf_token"]


def signup(client, name="alice", password="password123"):
    return client.post("/signup", data={"username": name, "password": password, "csrf_token": token(client)})


def upload(client, data=None, name="leaf.png"):
    return client.post(
        "/analyses",
        data={"file": (io.BytesIO(data or png_bytes()), name), "csrf_token": token(client)},
        content_type="multipart/form-data",
    )


def analysis_id(resp) -> int:
    return int(re.search(r"/analyses/(\d+)", resp.headers["Location"]).group(1))


def rows(app, sql, *args):
    from sprout.web.db import connect

    con = connect(app.config["DB_PATH"])
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def test_prediction_and_user_selection_are_stored_separately(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    r = c.post(f"/analyses/{aid}/selection", json={"label": "tomato_late_blight"}, headers={"X-CSRF-Token": token(c)})
    assert r.status_code == 200
    (row,) = rows(app, "select prediction_json, selected_label from analyses where id=?", aid)
    assert Identification.from_json(row["prediction_json"]).top.label == "tomato_early_blight"
    assert row["selected_label"] == "tomato_late_blight"


def test_selection_must_be_one_of_the_candidates(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    r = c.post(f"/analyses/{aid}/selection", json={"label": "rose_black_spot"}, headers={"X-CSRF-Token": token(c)})
    assert r.status_code == 400
    r = c.post(f"/analyses/{aid}/selection", data="nope", content_type="text/plain", headers={"X-CSRF-Token": token(c)})
    assert r.status_code == 400


def test_refresh_does_not_overwrite_selection(tmp_path):
    ident = Counting([ANSWER])
    app = make_app(tmp_path, identifier=ident)
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    c.post(f"/analyses/{aid}/selection", json={"label": "tomato_healthy"}, headers={"X-CSRF-Token": token(c)})
    for _ in range(3):
        assert c.get(f"/analyses/{aid}").status_code == 200
    assert ident.calls == 1
    (row,) = rows(app, "select selected_label from analyses where id=?", aid)
    assert row["selected_label"] == "tomato_healthy"


def test_user_cannot_see_another_users_upload_or_history(tmp_path):
    app = make_app(tmp_path)
    a, b = app.test_client(), app.test_client()
    signup(a, "alice")
    aid = analysis_id(upload(a))
    signup(b, "bobby")
    assert a.get(f"/analyses/{aid}/image").status_code == 200
    assert b.get(f"/analyses/{aid}").status_code == 404
    assert b.get(f"/analyses/{aid}/image").status_code == 404
    assert f"/analyses/{aid}" not in b.get("/history").get_data(as_text=True)
    assert b.post(f"/analyses/{aid}/delete", data={"csrf_token": token(b)}).status_code == 404
    assert len(rows(app, "select id from analyses")) == 1
    h = {"X-CSRF-Token": token(b)}
    assert b.post(f"/analyses/{aid}/selection", json={"label": "tomato_healthy"}, headers=h).status_code == 404
    assert app.test_client().get(f"/analyses/{aid}/image").status_code in (302, 401)


@pytest.mark.parametrize("target", ["https://evil.example/x", "//evil.example", "/\\evil.example", "javascript:alert(1)"])
def test_open_redirect_is_refused(tmp_path, target):
    app = make_app(tmp_path)
    c = app.test_client()
    signup(c)
    c.post("/logout", data={"csrf_token": token(c)})
    r = c.post(f"/login?next={target}", data={"username": "alice", "password": "password123", "csrf_token": token(c)})
    assert r.status_code == 302
    assert r.headers["Location"] == "/"
    c.post("/logout", data={"csrf_token": token(c)})
    r = c.post("/login?next=/history", data={"username": "alice", "password": "password123", "csrf_token": token(c)})
    assert r.headers["Location"] == "/history"


def test_csrf_required_on_post(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    assert c.post("/signup", data={"username": "alice", "password": "password123"}).status_code == 400
    bad = {"username": "alice", "password": "password123", "csrf_token": "wrong"}
    assert c.post("/signup", data=bad).status_code == 400
    assert rows(app, "select id from users") == []
    assert signup(c).status_code == 302


def test_non_image_upload_rejected(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    signup(c)
    r = upload(c, b"<html>not an image</html>", "evil.png")
    assert r.status_code == 400
    assert rows(app, "select id from analyses") == []
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not [p for p in uploads.rglob("*") if p.is_file()]
    gif = io.BytesIO()
    Image.new("RGB", (8, 8)).save(gif, "GIF")
    assert upload(c, gif.getvalue(), "x.jpg").status_code == 400


def test_oversized_upload_rejected(tmp_path):
    app = make_app(tmp_path, MAX_UPLOAD_BYTES=1000)
    c = app.test_client()
    signup(c)
    assert upload(c, png_bytes() + b"0" * 5000).status_code == 413


def test_rate_limit_applies(tmp_path):
    limits = {**LIMITS, "login": (3, 60)}
    app = make_app(tmp_path, RATE_LIMITS=limits)
    c = app.test_client()
    form = {"username": "nobody", "password": "password123", "csrf_token": token(c)}
    codes = [c.post("/login", data=form).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]


def test_ready_false_when_model_missing(tmp_path):
    app = make_app(tmp_path, identifier=None)
    c = app.test_client()
    assert c.get("/healthz").status_code == 200
    r = c.get("/readyz")
    assert r.status_code == 503 and r.get_json()["ready"] is False
    ok = make_app(tmp_path / "ok").test_client().get("/readyz")
    assert ok.status_code == 200 and ok.get_json()["ready"] is True


def test_two_app_instances_share_state(tmp_path):
    one, two = make_app(tmp_path), make_app(tmp_path)
    a = one.test_client()
    signup(a)
    aid = analysis_id(upload(a))
    b = two.test_client()
    r = b.post("/login", data={"username": "alice", "password": "password123", "csrf_token": token(b)})
    assert r.status_code == 302
    assert b.get(f"/analyses/{aid}").status_code == 200
    assert two.test_client().get("/history").status_code == 302
    b2 = two.test_client()
    b2.set_cookie("session", a.get_cookie("session").value)
    assert b2.get("/history").status_code == 200


def test_delete_removes_row_and_file(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    files = [p for p in (tmp_path / "uploads").rglob("*") if p.is_file()]
    assert len(files) == 1
    assert c.post(f"/analyses/{aid}/delete", data={"csrf_token": token(c)}).status_code == 302
    assert rows(app, "select id from analyses") == []
    assert not files[0].exists()


def test_production_requires_secret_key(tmp_path):
    with pytest.raises(RuntimeError):
        Config.from_env({"SPROUT_ENV": "production"})
    with pytest.raises(RuntimeError):
        Config.from_env({})
    cfg = Config.from_env({"SPROUT_ENV": "production", "SPROUT_SECRET_KEY": "x" * 32, "SPROUT_HTTPS": "1"})
    assert cfg.HTTPS is True
    app = create_app(Config(**{**cfg.__dict__, "DATA_DIR": str(tmp_path)}), identifier=None)
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True


def test_identifier_timeout_returns_504(tmp_path):
    class Slow:
        def identify(self, image_bytes):
            time.sleep(1)
            return ANSWER

    app = make_app(tmp_path, identifier=Slow(), API_TIMEOUT_SECONDS=0.05)
    c = app.test_client()
    signup(c)
    assert upload(c).status_code == 504
    assert rows(app, "select id from analyses") == []


def test_chat_validates_payload_and_is_scoped(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    h = {"X-CSRF-Token": token(c)}
    assert c.post(f"/analyses/{aid}/chat", json={"message": ""}, headers=h).status_code == 400
    assert c.post(f"/analyses/{aid}/chat", json={"message": "x" * 5000}, headers=h).status_code == 400
    r = c.post(f"/analyses/{aid}/chat", json={"message": "how do I treat this?"}, headers=h)
    assert r.status_code == 200
    body = r.get_json()
    assert body["passages"] == [] and "no supporting document" in body["note"].lower()
    assert len(rows(app, "select id from chat_messages where analysis_id=?", aid)) == 1
    b = app.test_client()
    signup(b, "bobby")
    assert b.post(f"/analyses/{aid}/chat", json={"message": "hi"}, headers={"X-CSRF-Token": token(b)}).status_code == 404


def test_chat_with_retriever_returns_cited_passages_and_label_warning(tmp_path):
    from sprout.rag.retrieve import Passage

    seen = {}

    class FakeRetriever:
        crops = {"tomato"}
        warmed = True

        def warm(self, *, rerank=False):
            pass

        def search(self, query, *, crop=None, disease=None, k=8, mode="hybrid", rerank=True):
            seen.update(query=query, crop=crop, rerank=rerank)
            return [Passage("Remove infected leaves.", "tomato.pdf", 3, "tomato", 0.9)]

    app = create_app(
        Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
        identifier=Counting([ANSWER]),
        retriever=FakeRetriever(),
    )
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    assert _wait(lambda: app.extensions["sprout_chat"]["state"] == "ready")
    body = c.post(f"/analyses/{aid}/chat", json={"message": "treatment?"}, headers={"X-CSRF-Token": token(c)}).get_json()
    assert body["passages"][0]["source"] == "tomato.pdf" and body["passages"][0]["page"] == 3
    assert "product label" in body["note"]
    assert seen == {"query": "treatment?", "crop": "tomato", "rerank": True}


def test_lab_class_names_map_to_knowledge_crops():
    import csv
    from pathlib import Path

    from sprout.web.crops import crop_for_label

    root = Path(__file__).resolve().parents[1]
    with (root / "knowledge" / "manifest.csv").open(encoding="utf-8") as f:
        known = {r["crop"].strip().lower() for r in csv.DictReader(f)} - {"general"}
    with (root / "datasets" / "manifests" / "lab.csv").open(encoding="utf-8") as f:
        labels = sorted({r["label"] for r in csv.DictReader(f)})
    assert len(labels) == 38
    got = {label: crop_for_label(label, known) for label in labels}
    assert got["Pepper,_bell___Bacterial_spot"] == "pepper"
    assert got["Cherry_(including_sour)___healthy"] == "cherry"
    assert got["Corn_(maize)___Common_rust_"] == "corn"
    assert got["Orange___Haunglongbing_(Citrus_greening)"] == "orange"
    assert got["Tomato___Tomato_mosaic_virus"] == "tomato"
    # every plant now has documents of its own (T6.5 added raspberry, soybean and squash)
    assert {label.split("___")[0] for label, crop in got.items() if crop is None} == set()
    assert {c for c in got.values() if c} <= known


def test_unknown_plant_gets_general_documents_only(tmp_path):
    from sprout.rag.retrieve import Passage

    seen = {}

    class FakeRetriever:
        crops = {"tomato"}
        warmed = True

        def warm(self, *, rerank=False):
            pass

        def search(self, query, *, crop=None, disease=None, k=8, mode="hybrid", rerank=True):
            seen["crop"] = crop
            return [Passage("Clean your tools.", "general.pdf", 1, "general", 0.9)]

    soy = Identification("answered", Candidate("Soybean___healthy", 0.9), [], "")
    app = create_app(
        Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
        identifier=Counting([soy]),
        retriever=FakeRetriever(),
    )
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    assert _wait(lambda: app.extensions["sprout_chat"]["state"] == "ready")
    body = c.post(f"/analyses/{aid}/chat", json={"message": "care?"}, headers={"X-CSRF-Token": token(c)}).get_json()
    assert seen["crop"] == "general"
    assert "no document covers this plant" in body["note"].lower()


def _wait(pred, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end and not pred():
        time.sleep(0.01)
    return pred()


def test_readiness_tells_the_truth_about_chat(tmp_path):
    import threading

    gate = threading.Event()

    class SlowRetriever:
        crops = {"tomato"}

        def __init__(self):
            self.warmed = False

        def warm(self, *, rerank=False):
            gate.wait(5)
            self.warmed = True

        def search(self, query, **kw):
            return []

    app = create_app(
        Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
        identifier=Counting([ANSWER]),
        retriever=SlowRetriever(),
    )
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    assert c.get("/readyz").get_json()["chat"] == "warming"
    r = c.post(f"/analyses/{aid}/chat", json={"message": "hi"}, headers={"X-CSRF-Token": token(c)})
    assert r.status_code == 503  # not a timeout, and no second embed started
    gate.set()
    assert _wait(lambda: c.get("/readyz").get_json()["chat"] == "ready")
    assert c.post(f"/analyses/{aid}/chat", json={"message": "hi"}, headers={"X-CSRF-Token": token(c)}).status_code == 200


def test_readiness_reports_failed_warmup_and_missing_retriever(tmp_path):
    class Broken:
        crops = set()
        warmed = False

        def warm(self, *, rerank=False):
            raise RuntimeError("index unreadable")

    app = create_app(
        Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
        identifier=Counting([ANSWER]),
        retriever=Broken(),
    )
    c = app.test_client()
    assert _wait(lambda: c.get("/readyz").get_json()["chat"] == "failed")
    assert make_app(tmp_path / "none").test_client().get("/readyz").get_json()["chat"] == "unavailable"


def _app_with_retriever(tmp_path, retriever, ident=ANSWER):
    app = create_app(
        Config(ENV="testing", SECRET_KEY="s", DATA_DIR=str(tmp_path), RATE_LIMITS=LIMITS),
        identifier=Counting([ident]),
        retriever=retriever,
    )
    app.config["TESTING"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = False  # a failing chat call must answer 500, not raise into the test
    return app


class _BrokenRetriever:
    crops = {"tomato"}
    warmed = True

    def warm(self, *, rerank=False):
        pass

    def search(self, query, **kw):
        raise RuntimeError("index unreadable")


def test_failed_chat_call_stores_no_message_so_a_retry_stores_one(tmp_path):
    app = _app_with_retriever(tmp_path, _BrokenRetriever())
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    assert _wait(lambda: app.extensions["sprout_chat"]["state"] == "ready")
    h = {"X-CSRF-Token": token(c)}
    assert c.post(f"/analyses/{aid}/chat", json={"message": "treatment?"}, headers=h).status_code == 500
    assert rows(app, "select id from chat_messages where analysis_id=?", aid) == []


@pytest.mark.parametrize("status", ["not_plant", "invalid"])
def test_chat_refused_when_the_image_was_not_identified_as_a_plant(tmp_path, status):
    app = make_app(tmp_path, identifier=Counting([Identification(status, None, [], "try again")]))
    c = app.test_client()
    signup(c)
    aid = analysis_id(upload(c))
    r = c.post(f"/analyses/{aid}/chat", json={"message": "hello"}, headers={"X-CSRF-Token": token(c)})
    assert r.status_code == 409
    assert rows(app, "select id from chat_messages where analysis_id=?", aid) == []


@pytest.mark.parametrize("secret", ["", "   ", "x", "short-secret", " " * 40])
def test_short_or_blank_secret_key_refused_outside_testing(secret):
    for env in ("production", "development"):
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            Config.from_env({"SPROUT_ENV": env, "SPROUT_SECRET_KEY": secret})
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        Config.from_env({"SPROUT_ENV": "testing", "SPROUT_SECRET_KEY": " "})


def test_a_long_secret_key_is_accepted():
    assert Config.from_env({"SPROUT_ENV": "production", "SPROUT_SECRET_KEY": "k" * 16}).SECRET_KEY == "k" * 16


BS = chr(92)


@pytest.mark.parametrize(
    "href",
    ["//evil.example/x", BS * 2 + "evil.example/x", "javascript:alert(1)", "ftp://x.example/f"],
)
def test_markdown_links_to_other_origins_by_odd_forms_lose_their_href(href):
    from sprout.web.markup import render_markdown

    html = render_markdown(f"[go]({href})")
    assert "href" not in html and "go" in html


def test_markdown_links_open_safely():
    from sprout.web.markup import render_markdown

    html = render_markdown("[ok](https://example.org/a) and [mail](mailto:a@example.org)")
    assert html.count('rel="noopener noreferrer nofollow"') == 2
    assert 'href="https://example.org/a"' in html
