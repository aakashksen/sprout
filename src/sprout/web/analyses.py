"""Uploads, results, the user's own selection, history, delete, and chat sources.

The model's prediction is written once, when the image is analysed. The user's choice is a
separate column, and viewing a result never runs the model again.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from sprout.rag.answer import UNAVAILABLE, Answer, answer
from sprout.rag.retrieve import GENERAL_CROP
from sprout.serve.types import Identification

from .crops import crop_for_label
from .markup import render_markdown
from .security import call_with_timeout, get_db, login_required, rate_limit
from .uploads import BadImage, check_image, save_private

bp = Blueprint("analyses", __name__)
MAX_MESSAGE = 1000
MAX_REPORT = 1000
CHATTABLE = ("answered", "unsure")
NO_SOURCE = "No supporting document was found for this question, so no advice is given."
GENERAL_ONLY = "No document covers this plant, so only general guidance is shown, which may not apply to it."
LABEL_NOTE = "Check the product label and local rules before using any chemical or disposal advice."


def _owned(analysis_id: int):
    row = get_db().execute(
        "select * from analyses where id = ? and user_id = ?", (analysis_id, g.user["id"])
    ).fetchone()
    if row is None:
        abort(404)
    return row


def _candidates(ident: Identification) -> list[str]:
    return ([ident.top.label] if ident.top else []) + [a.label for a in ident.alternatives]


def _shown_label(row, ident: Identification) -> str | None:
    return row["selected_label"] or (ident.top.label if ident.top else None)


def pretty(label: str) -> str:
    return label.replace("_", " ")


@bp.post("/analyses")
@login_required
def create():
    rate_limit("upload", str(g.user["id"]))
    file = request.files.get("file")
    if file is None or not file.filename:
        abort(400, "choose an image file")
    limit = current_app.config["MAX_UPLOAD_BYTES"]
    data = file.read(limit + 1)
    if len(data) > limit:
        abort(413, "image is too large")
    try:
        ext = check_image(data)
    except BadImage as e:
        abort(400, str(e))
    identifier = current_app.extensions["sprout_identifier"]
    if identifier is None:
        abort(503, "the identification model is not available")
    ident = call_with_timeout(identifier.identify, data)
    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    stored = save_private(upload_dir, g.user["id"], data, ext)
    db = get_db()
    try:
        cur = db.execute(
            "insert into analyses (user_id, stored_name, prediction_json) values (?, ?, ?)",
            (g.user["id"], stored, ident.to_json()),
        )
        db.commit()
    except Exception:
        (upload_dir / stored).unlink(missing_ok=True)
        raise
    return redirect(url_for("analyses.show", analysis_id=cur.lastrowid))


@bp.get("/analyses/<int:analysis_id>")
@login_required
def show(analysis_id: int):
    row = _owned(analysis_id)
    ident = Identification.from_json(row["prediction_json"])
    return render_template(
        "result.html",
        analysis=row,
        ident=ident,
        candidates=_candidates(ident),
        pretty=pretty,
    )


@bp.get("/analyses/<int:analysis_id>/image")
@login_required
def image(analysis_id: int):
    row = _owned(analysis_id)
    resp = send_from_directory(current_app.config["UPLOAD_DIR"], row["stored_name"])
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@bp.post("/analyses/<int:analysis_id>/selection")
@login_required
def select(analysis_id: int):
    row = _owned(analysis_id)
    body = request.get_json(silent=True)
    label = body.get("label") if isinstance(body, dict) else None
    if not isinstance(label, str) or label not in _candidates(Identification.from_json(row["prediction_json"])):
        abort(400, "label must be one of the candidates for this image")
    db = get_db()
    db.execute("update analyses set selected_label = ? where id = ?", (label, analysis_id))
    db.commit()
    return jsonify(selected=label)


@bp.post("/analyses/<int:analysis_id>/delete")
@login_required
def delete(analysis_id: int):
    row = _owned(analysis_id)
    db = get_db()
    db.execute("delete from analyses where id = ?", (analysis_id,))
    db.commit()
    (Path(current_app.config["UPLOAD_DIR"]) / row["stored_name"]).unlink(missing_ok=True)
    return redirect(url_for("analyses.history"))


@bp.get("/history")
@login_required
def history():
    rows = get_db().execute(
        "select id, prediction_json, selected_label, created_at from analyses where user_id = ? order by id desc",
        (g.user["id"],),
    ).fetchall()
    items = [
        dict(id=r["id"], created_at=r["created_at"], ident=Identification.from_json(r["prediction_json"]),
             selected=r["selected_label"])
        for r in rows
    ]
    return render_template("history.html", items=items, pretty=pretty)


@bp.post("/analyses/<int:analysis_id>/chat")
@login_required
def chat(analysis_id: int):
    row = _owned(analysis_id)
    ident = Identification.from_json(row["prediction_json"])
    if ident.status not in CHATTABLE:
        abort(409, "this image was not identified as a plant, so there is nothing to ask about")
    rate_limit("chat", str(g.user["id"]))
    body = request.get_json(silent=True)
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, str) or not 1 <= len(message.strip()) <= MAX_MESSAGE:
        abort(400, f"message must be 1-{MAX_MESSAGE} characters")
    message = message.strip()
    retriever = current_app.extensions["sprout_retriever"]
    writer = current_app.extensions["sprout_writer"]
    if writer is not None and retriever is not None and current_app.extensions["sprout_chat"]["state"] == "ready":
        return _generated_reply(analysis_id, row, ident, message, retriever, writer)
    if retriever is not None and current_app.extensions["sprout_chat"]["state"] != "ready":
        abort(503, "chat is not ready yet")  # before anything is stored: a rejected message leaves no trace
    passages = []
    general_only = False
    label = _shown_label(row, ident)
    if retriever is not None and label:
        crop = crop_for_label(label, retriever.crops)
        # the score floor only works on reranked scores, so reranking stays on here
        found = call_with_timeout(lambda: retriever.search(message, crop=crop or GENERAL_CROP, rerank=True))
        # search falls back to general documents when the plant's own are all filtered out
        general_only = crop is None or (bool(found) and all(p.crop == GENERAL_CROP for p in found))
        sources = current_app.extensions["sprout_sources"]
        passages = [_passage(p, sources) for p in found]
    # stored only once retrieval worked: a failed call leaves no trace, so a retry is not a duplicate
    db = get_db()
    db.execute(
        "insert into chat_messages (user_id, analysis_id, role, content) values (?, ?, 'user', ?)",
        (g.user["id"], analysis_id, message),
    )
    db.commit()
    # generated answers arrive with sprout.rag.answer; until then the reply is the cited passages
    note = LABEL_NOTE if passages else NO_SOURCE
    if passages and general_only:
        note = f"{GENERAL_ONLY} {LABEL_NOTE}"
    return jsonify(passages=passages, note=note)


FAILED_STATES = ("unavailable", "retrieval_failed")  # nothing is stored for these, so a retry is not a duplicate
HISTORY_ROWS = 8


def _as_selected(ident: Identification, selected: str | None) -> Identification:
    """The diagnosis the writer is told about: the user's own choice first, the model's other guesses after it,
    each with the model's own confidence. Nothing is made up about how sure the model was."""
    if not selected or (ident.top and ident.top.label == selected):
        return ident
    pool = ([ident.top] if ident.top else []) + list(ident.alternatives)
    chosen = next((c for c in pool if c.label == selected), None)
    if chosen is None:
        return ident
    return Identification(ident.status, chosen, [c for c in pool if c is not chosen], ident.guidance)


def _generated_reply(analysis_id: int, row, ident: Identification, message: str, retriever, writer):
    cfg = current_app.extensions["sprout_answer_config"]
    label = _shown_label(row, ident)
    crop = (crop_for_label(label, retriever.crops) or GENERAL_CROP) if label else None
    db = get_db()
    recent = db.execute(
        "select role, content from chat_messages where analysis_id = ? order by id desc limit ?",
        (analysis_id, HISTORY_ROWS),
    ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(recent)]
    shown = _as_selected(ident, row["selected_label"])
    future = current_app.extensions["sprout_pool"].submit(
        contextvars.copy_context().run,
        lambda: answer(message, identification=shown, history=history, retriever=retriever, llm=writer,
                       config=cfg, crop=crop))
    try:
        ans = future.result(timeout=current_app.config["CHAT_TIMEOUT_SECONDS"])
    except FutureTimeout:
        # the thread ends by itself: the model call has its own, shorter timeout
        ans = Answer(UNAVAILABLE, [], False, [], "unavailable")
    writer_state = current_app.extensions["sprout_writer_state"]
    if ans.state == "unavailable":
        writer_state["state"] = "unavailable"
    elif ans.passages:
        writer_state["state"] = "ok"
    if ans.state not in FAILED_STATES:
        for role, content in (("user", message), ("assistant", ans.text)):
            db.execute("insert into chat_messages (user_id, analysis_id, role, content) values (?, ?, ?, ?)",
                       (g.user["id"], analysis_id, role, content))
        db.commit()
    sources = current_app.extensions["sprout_sources"]
    cited = {(c.source, c.page) for c in ans.citations}
    shown_passages = [_passage(p, sources) for p in ans.passages if (p.source, p.page) in cited]
    only_general = bool(ans.passages) and all(p.crop == GENERAL_CROP for p in ans.passages) and crop != GENERAL_CROP
    return jsonify(
        state=ans.state, text=ans.text, html=render_markdown(ans.text), grounded=ans.grounded, asks_for=ans.asks_for,
        citations=[dict(source=c.source, page=c.page, title=c.title, url=c.url, note=c.note, quote=c.quote)
                   for c in ans.citations],
        passages=shown_passages, note=GENERAL_ONLY if only_general else "",
    )


def _passage(p, sources: dict) -> dict:
    title, url = sources.get(p.source, (p.source, ""))
    return dict(text=p.text, html=render_markdown(p.text), source=p.source, page=p.page, score=p.score,
                title=title, url=url)


@bp.post("/analyses/<int:analysis_id>/report")
@login_required
def report(analysis_id: int):
    row = _owned(analysis_id)
    body = request.get_json(silent=True)
    note = body.get("note") if isinstance(body, dict) else None
    if not isinstance(note, str) or not 1 <= len(note.strip()) <= MAX_REPORT:
        abort(400, f"describe the problem in 1-{MAX_REPORT} characters")
    shown = _shown_label(row, Identification.from_json(row["prediction_json"]))
    db = get_db()
    cur = db.execute(
        "insert into reports (user_id, analysis_id, shown_label, note) values (?, ?, ?, ?)",
        (g.user["id"], analysis_id, shown, note.strip()),
    )
    db.commit()
    return jsonify(stored=True, id=cur.lastrowid), 201
