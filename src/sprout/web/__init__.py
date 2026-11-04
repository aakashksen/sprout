"""Flask app factory. The identifier and retriever are injected, so the app never loads a model itself."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request, session
from werkzeug.exceptions import HTTPException

from sprout.serve.types import IdentifierProtocol
from sprout.serve.wording import HEALTHY_CAUTION, is_healthy

from . import analyses, auth, pages
from .config import Config
from .db import init_db
from .security import check_csrf, close_db, csrf_token, get_db


def create_app(
    config: Config | None = None,
    *,
    identifier: IdentifierProtocol | None = None,
    retriever=None,
    writer=None,
    answer_config=None,
) -> Flask:
    """`identifier=None` means the model is missing: the app starts but /readyz says not ready.

    `retriever` is a `sprout.rag.retrieve.Retriever`; without one, chat returns no passages.
    `writer` is an LLM client (`sprout.llm.client`); with one, chat answers through `sprout.rag.answer.answer`,
    configured by `answer_config` (default: the served setup in `sprout.rag.serving`). Without one, chat returns
    the cited passages only.
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
        CHAT_TIMEOUT_SECONDS=cfg.CHAT_TIMEOUT_SECONDS,
        MEASUREMENT=cfg.MEASUREMENT,
    )
    init_db(cfg.db_path)
    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    app.extensions["sprout_identifier"] = identifier
    app.extensions["sprout_retriever"] = retriever
    app.extensions["sprout_chat"] = {"state": "unavailable" if retriever is None else "warming"}
    app.extensions["sprout_sources"] = _load_sources(cfg.MANIFEST)
    app.extensions["sprout_writer"] = writer
    app.extensions["sprout_writer_state"] = {"state": "not configured" if writer is None else "unknown"}
    app.extensions["sprout_answer_config"] = (answer_config or _answer_config(cfg)) if writer is not None else None
    app.extensions["sprout_pool"] = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sprout")

    if retriever is not None:
        threading.Thread(target=_warm, args=(app.extensions["sprout_chat"], retriever), daemon=True).start()

    app.teardown_appcontext(close_db)
    app.context_processor(lambda: {"csrf_token": csrf_token, "healthy_caution": HEALTHY_CAUTION, "is_healthy": is_healthy})

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
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        return resp

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        wants_json = request.is_json or request.accept_mimetypes.best == "application/json" or request.path.startswith(("/healthz", "/readyz"))
        if wants_json:
            return jsonify(error=e.description), e.code
        return render_template("error.html", code=e.code, message=e.description), e.code

    app.register_blueprint(pages.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(analyses.bp)
    return app


def _answer_config(cfg: Config):
    from sprout.rag.answer import AnswerConfig, load_sources

    sources = load_sources(Path(cfg.MANIFEST)) if cfg.MANIFEST else None
    return AnswerConfig(timeout_s=cfg.CHAT_LLM_TIMEOUT_SECONDS, sources=sources)


def _load_sources(manifest: str) -> dict[str, tuple[str, str]]:
    """file name -> (title, link). Only http(s) addresses become links."""
    if not manifest:
        return {}
    from sprout.rag.ingest import load_manifest

    return {
        e.file: (e.title, e.source_url if e.source_url.startswith(("http://", "https://")) else "")
        for e in load_manifest(Path(manifest))
    }


def _warm(status: dict, retriever) -> None:
    """Embed the index and load the reranker in the background; chat opens when this finishes."""
    try:
        retriever.warm(rerank=True)
        status["state"] = "ready"
    except Exception:
        status["state"] = "failed"
