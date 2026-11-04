"""Home page plus liveness and readiness."""

from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, render_template

from .security import get_db, login_required

bp = Blueprint("pages", __name__)


@bp.get("/")
def home():
    return render_template("home.html")


@bp.get("/upload")
@login_required
def upload():
    return render_template("upload.html")


@bp.get("/camera")
@login_required
def camera():
    return render_template("camera.html")


@bp.get("/dashboard")
@login_required
def dashboard():
    db = get_db()
    (count,) = db.execute("select count(*) from analyses where user_id = ?", (g.user["id"],)).fetchone()
    latest = db.execute(
        "select id, created_at from analyses where user_id = ? order by id desc limit 1", (g.user["id"],)
    ).fetchone()
    return render_template("dashboard.html", count=count, latest=latest)


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
    # The writer's last known state is reported but never makes the app unready: the platform would restart a
    # container for an outage it cannot fix, and chat already says "unavailable" to the user.
    body = {"ready": not problems, "problems": problems, "chat": current_app.extensions["sprout_chat"]["state"],
            "writer": current_app.extensions["sprout_writer_state"]["state"],
            "measurement": current_app.config["MEASUREMENT"]}
    return jsonify(body), (200 if not problems else 503)
