"""Flask app factory. The identifier and retriever are injected, so the app never loads a model itself."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, g, jsonify, render_template, request, session
from werkzeug.exceptions import HTTPException

from sprout.serve.types import IdentifierProtocol

from . import analyses, auth, pages
from .config import Config
from .db import init_db
from .security import check_csrf, close_db, csrf_token, get_db


def create_app(config: Config | None = None, *, identifier: IdentifierProtocol | None = None, retriever=None) -> Flask:
    """`identifier=None` means the model is missing: the app starts but /readyz says not ready.

    `retriever` is a `sprout.rag.retrieve.Retriever`; without one, chat returns no passages.
    """
    cfg = config or Config.from_env(dict(os.environ))
    cfg.validate()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=cfg.SECRET_KEY,
        TESTING=cfg.ENV == "testing",
        MAX_CONTENT_LENGTH=cfg.MAX_UPLOAD_BYTES + 4096,
        MAX_UPLOAD_BYTES=cfg.MAX_UPLOAD_BYTES,
        SESSION_COOKIE_SECURE=cfg.HTTPS,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=7 * 24 * 3600,
        DB_PATH=str(cfg.db_path),
        UPLOAD_DIR=str(cfg.upload_dir),
        RATE_LIMITS=cfg.RATE_LIMITS,
        API_TIMEOUT_SECONDS=cfg.API_TIMEOUT_SECONDS,
    )
    init_db(cfg.db_path)
    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    app.extensions["sprout_identifier"] = identifier
    app.extensions["sprout_retriever"] = retriever
    app.extensions["sprout_pool"] = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sprout")

    app.teardown_appcontext(close_db)
    app.context_processor(lambda: {"csrf_token": csrf_token})

    @app.before_request
    def load_user() -> None:
        check_csrf()
        uid = session.get("uid")
        g.user = None
        if uid is not None:
            g.user = get_db().execute("select id, username from users where id = ?", (uid,)).fetchone()

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        wants_json = request.is_json or request.path.startswith(("/healthz", "/readyz"))
        if wants_json:
            return jsonify(error=e.description), e.code
        return render_template("error.html", code=e.code, message=e.description), e.code

    app.register_blueprint(pages.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(analyses.bp)
    return app
