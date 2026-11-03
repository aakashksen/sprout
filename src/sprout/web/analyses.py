"""Uploads, results, the user's own selection, history, delete, and chat sources.

The model's prediction is written once, when the image is analysed. The user's choice is a
separate column, and viewing a result never runs the model again.
"""

from __future__ import annotations

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

from sprout.serve.types import Identification

from .security import call_with_timeout, get_db, login_required, rate_limit
from .uploads import BadImage, check_image, save_private

bp = Blueprint("analyses", __name__)
MAX_MESSAGE = 1000
NO_SOURCE = "No supporting document was found for this question, so no advice is given."
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
    rate_limit("chat", str(g.user["id"]))
    body = request.get_json(silent=True)
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, str) or not 1 <= len(message.strip()) <= MAX_MESSAGE:
        abort(400, f"message must be 1-{MAX_MESSAGE} characters")
    message = message.strip()
    db = get_db()
    db.execute(
        "insert into chat_messages (user_id, analysis_id, role, content) values (?, ?, 'user', ?)",
        (g.user["id"], analysis_id, message),
    )
    db.commit()
    passages = []
    retriever = current_app.extensions["sprout_retriever"]
    top = Identification.from_json(row["prediction_json"]).top
    label = row["selected_label"] or (top.label if top else None)
    if retriever is not None and label:
        crop = label.split("_", 1)[0]
        # the score floor only works on reranked scores, so reranking stays on here
        found = call_with_timeout(lambda: retriever.search(message, crop=crop, rerank=True))
        passages = [dict(text=p.text, source=p.source, page=p.page, score=p.score) for p in found]
    # generated answers arrive with sprout.rag.answer; until then the reply is the cited passages
    note = LABEL_NOTE if passages else NO_SOURCE
    return jsonify(passages=passages, note=note)
