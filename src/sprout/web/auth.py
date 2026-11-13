"""Sign up, log in, log out. Identity is a user id in the signed session cookie."""

from __future__ import annotations

import re
import sqlite3

from flask import Blueprint, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .security import get_db, rate_limit, safe_next

bp = Blueprint("auth", __name__)
USERNAME = re.compile(r"[A-Za-z0-9_.-]{4,20}")


def _start_session(user_id: int) -> None:
    session.clear()  # new session, so a pre-login session id is never reused
    session["uid"] = user_id
    session.permanent = True


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        rate_limit("signup", request.remote_addr or "?")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not USERNAME.fullmatch(username):
            error = "Username must be 4-20 letters, digits, dot, dash or underscore."
        elif not 8 <= len(password) <= 128:
            error = "Password must be 8-128 characters."
        else:
            db = get_db()
            try:
                cur = db.execute(
                    "insert into users (username, password_hash) values (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()
            except sqlite3.IntegrityError:
                error = "That username is taken."
            else:
                _start_session(cur.lastrowid)
                return redirect(url_for("pages.home"))
    return render_template("signup.html", error=error), (400 if error else 200)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None and request.method == "GET":
        return redirect(url_for("pages.home"))
    error = None
    if request.method == "POST":
        rate_limit("login", request.remote_addr or "?")
        row = get_db().execute(
            "select id, password_hash from users where username = ?", (request.form.get("username", ""),)
        ).fetchone()
        if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
            _start_session(row["id"])
            return redirect(safe_next(request.args.get("next")))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("pages.home"))
