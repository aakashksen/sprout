"""Crash-safe file writes: a reader sees the old file or the complete new one, never a partial one."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import IO, Callable

REPLACE_TRIES = 20
REPLACE_WAIT = 0.1


def temp_path(path: Path) -> Path:
    """Where a write to `path` is staged: same folder, and a name no record, checkpoint or glob in this
    project matches (it ends in .tmp, which .gitignore also covers)."""
    return path.with_name(f".{path.name}.tmp")


def _replace(src: Path, dst: Path) -> None:
    # Windows refuses to replace a file another process has open; wait briefly, then fail clearly.
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if attempt == REPLACE_TRIES - 1:
                raise OSError(f"cannot replace {dst}: still open elsewhere after "
                              f"{REPLACE_TRIES * REPLACE_WAIT:.0f}s ({exc})") from exc
            time.sleep(REPLACE_WAIT)


def atomic_write(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Stage `write(file)` in a temp file beside `path`, fsync it, then move it into place."""
    path = Path(path)
    tmp = temp_path(path)
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, lambda f: f.write(text.encode("utf-8")))
