"""CSRF, safe redirects, rate limits, login guard, and a timeout for slow calls."""

from __future__ import annotations

import contextvars
import hmac
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from functools import wraps
from urllib.parse import urlsplit

from flask import abort, current_app, g, jsonify, redirect, request, session, url_for

from .db import connect

SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


def get_db():
    if "db" not in g:
        g.db = connect(current_app.config["DB_PATH"])
    return g.db


def close_db(_exc=None) -> None:
    con = g.pop("db", None)
    if con is not None:
        con.close()


def csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def check_csrf() -> None:
    csrf_token()
    if request.method in SAFE_METHODS:
        return
    sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
    if not hmac.compare_digest(sent.encode(), session["csrf_token"].encode()):
        abort(400, "missing or invalid CSRF token")


def safe_next(target: str | None) -> str:
    """Only same-site relative paths are followed; anything else goes home."""
    home = url_for("pages.home")
    if not target or "\\" in target or any(ord(ch) < 32 for ch in target):
        return home
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return home
    return target


def rate_limit(name: str, who: str) -> None:
    limit, window = current_app.config["RATE_LIMITS"][name]
    key, now = f"{name}:{who}", time.time()
    db = get_db()
    db.execute("delete from rate_events where at < ?", (now - 3600 * 24,))
    db.execute("insert into rate_events (key, at) values (?, ?)", (key, now))
    (n,) = db.execute("select count(*) from rate_events where key = ? and at > ?", (key, now - window)).fetchone()
    db.commit()
    if n > limit:
        abort(429, "too many requests, try again later")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.method in SAFE_METHODS and not request.is_json:
                return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
            return jsonify(error="login required"), 401
        return view(*args, **kwargs)

    return wrapped


def call_with_timeout(fn, *args):
    """Run a slow call (model, retrieval, remote API) and give up after the configured time."""
    pool: ThreadPoolExecutor = current_app.extensions["sprout_pool"]
    future = pool.submit(contextvars.copy_context().run, fn, *args)  # the thread sees the request's context
    try:
        return future.result(timeout=current_app.config["API_TIMEOUT_SECONDS"])
    except FutureTimeout:
        future.cancel()
        abort(504, "the request took too long")
