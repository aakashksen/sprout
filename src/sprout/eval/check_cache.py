"""Remembers which run records fully passed check_run, so a land rechecks only new or changed ones.

A record is skipped only when the checker and everything it reads are byte-identical to when it passed:
the sprout modules check_run imports (followed through the import graph), the versions of the third-party
libraries those modules import, the manifests and the RAG and perf lists the splits come from, the record,
and the prediction and sample files it points to. Failures are never stored. The file lives in git's own
directory, so it is never tracked and every worktree of a repository shares it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sprout.atomic import atomic_write_text

CACHE_NAME = "sprout-check-run-cache.json"
MAX_ENTRIES = 5000
SPROUT = Path(__file__).resolve().parents[1]
ENTRY_MODULE = "sprout.eval.check_run"


def _module_file(name: str) -> Path | None:
    parts = name.split(".")
    if parts[0] != "sprout":
        return None
    base = SPROUT.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(path: Path) -> set[str]:
    """Every module name a file imports, including `from pkg import submodule`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def _closure(entry: str) -> tuple[list[Path], set[str]]:
    """The sprout files reachable from `entry` (parent packages included) and the third-party top-level names."""
    files: dict[Path, None] = {}
    outside: set[str] = set()
    todo = [entry]
    while todo:
        name = todo.pop()
        path = _module_file(name)
        if path is None:
            top = name.split(".")[0]
            if top != "sprout" and top not in sys.stdlib_module_names:
                outside.add(top)
            continue
        if path in files:
            continue
        files[path] = None
        todo += _imports(path)
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
    return sorted(files), outside


def _digest(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()


def code_fingerprint() -> str:
    files, outside = _closure(ENTRY_MODULE)
    parts = [p.relative_to(SPROUT).as_posix().encode() + p.read_bytes() for p in files]
    for top in sorted(outside):
        parts.append(f"{top}=={getattr(sys.modules.get(top), '__version__', '?')}".encode())
    return _digest(parts)


def data_fingerprint(root: Path) -> str:
    """The files every split is read from: a change to any of them forces a full recheck."""
    paths = sorted((root / "datasets" / "manifests").glob("*.csv"))
    paths += [root / "datasets/rag/questions.jsonl", root / "datasets/perf/images.txt", root / "datasets/label_exclusions.csv"]
    return _digest([p.name.encode() + (p.read_bytes() if p.is_file() else b"<missing>") for p in paths])


def shared_fingerprint(root: Path) -> str:
    return code_fingerprint() + data_fingerprint(root)


def _referenced(rec: Any) -> list[str]:
    if not isinstance(rec, dict):
        return []
    preds = rec.get("predictions")
    files = list(preds.values()) if isinstance(preds, dict) else []
    for kind in ("measurements", "judged"):
        block = rec.get(kind)
        if isinstance(block, dict):
            files.append(block.get("file"))
    return sorted(f for f in files if isinstance(f, str))


def record_key(path: Path, rec: Any, root: Path, min_n: int, shared: str) -> str:
    """One record's checkable state: the checker and data (`shared`), --min-n, the record and the files it names."""
    parts = [shared.encode(), str(min_n).encode(), path.name.encode(), path.read_bytes()]
    for rel in _referenced(rec):
        f = root / rel
        parts.append(rel.encode() + (f.read_bytes() if f.is_file() else b"<missing>"))
    return _digest(parts)


def cache_path(root: Path) -> Path:
    out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cannot find the git directory: {out.stderr.strip()}")
    return (root / out.stdout.strip()).resolve() / CACHE_NAME


def load(root: Path) -> list[str]:
    try:
        data = json.loads(cache_path(root).read_text(encoding="utf-8"))
        return [k for k in data["passed"] if isinstance(k, str)]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def save(root: Path, keys: list[str]) -> None:
    atomic_write_text(cache_path(root), json.dumps({"passed": list(dict.fromkeys(keys))[-MAX_ENTRIES:]}))
