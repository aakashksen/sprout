"""Home page plus liveness and readiness."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template

from .security import get_db

bp = Blueprint("pages", __name__)


@bp.get("/")
def home():
    return render_template("home.html")


@bp.get("/healthz")
def healthz():
    """Liveness: the process answers."""
    return jsonify(alive=True)


@bp.get("/readyz")
def readyz():
    """Readiness: the database opens and a model is loaded."""
    problems = []
    try:
        get_db().execute("select 1").fetchone()
    except Exception:
        problems.append("database unavailable")
    if current_app.extensions["sprout_identifier"] is None:
        problems.append("model missing")
    body = {"ready": not problems, "problems": problems, "chat": current_app.extensions["sprout_retriever"] is not None}
    return jsonify(body), (200 if not problems else 503)
