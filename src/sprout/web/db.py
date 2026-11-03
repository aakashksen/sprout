"""One explicit database setup. State lives here, never in process memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

BASE = """
create table users (
    id integer primary key,
    username text not null unique,
    password_hash text not null,
    created_at text not null default current_timestamp
);
create table analyses (
    id integer primary key,
    user_id integer not null references users(id) on delete cascade,
    stored_name text not null,
    prediction_json text not null,
    selected_label text,
    created_at text not null default current_timestamp
);
create index analyses_user on analyses(user_id, id);
create table chat_messages (
    id integer primary key,
    user_id integer not null references users(id) on delete cascade,
    analysis_id integer not null references analyses(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at text not null default current_timestamp
);
create index chat_by_analysis on chat_messages(analysis_id, id);
create table rate_events (
    key text not null,
    at real not null
);
create index rate_by_key on rate_events(key, at);
"""

REPORTS = """
create table reports (
    id integer primary key,
    user_id integer not null references users(id) on delete cascade,
    analysis_id integer not null references analyses(id) on delete cascade,
    shown_label text,
    note text not null,
    created_at text not null default current_timestamp
);
create index reports_by_analysis on reports(analysis_id, id);
"""
SCHEMA = BASE + REPORTS



def connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("pragma foreign_keys = on")
    return con


def init_db(path: Path) -> None:
    """Create the schema on a new database; refuse a database from a different version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(path)
    try:
        con.execute("pragma journal_mode = wal")
        version = con.execute("pragma user_version").fetchone()[0]
        if version == 0:
            con.executescript(SCHEMA)
            con.execute(f"pragma user_version = {SCHEMA_VERSION}")
            con.commit()
        elif version == 1:  # version 1 had no reports table
            con.executescript(REPORTS)
            con.execute(f"pragma user_version = {SCHEMA_VERSION}")
            con.commit()
        elif version != SCHEMA_VERSION:
            raise RuntimeError(f"database schema version {version}, expected {SCHEMA_VERSION}")
    finally:
        con.close()
