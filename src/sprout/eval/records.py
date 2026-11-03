"""Run records: one JSON file per experiment and split, mapping each system to its saved predictions."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sprout.data.splits import COLUMNS, dataset_hash
from sprout.eval.predict import Predictions
from sprout.eval.stats import percentile

ROOT = Path(__file__).resolve().parents[3]
RUNS_REL = Path("results/runs")
RAG_REL = Path("datasets/rag/questions.jsonl")
PERF_REL = Path("datasets/perf/images.txt")
SYSTEM_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
MEASURE_QUANTILES = (50, 95)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 13
IMAGE_MANIFESTS = ("lab", "field", "not_plant")


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def is_dirty(root: Path) -> bool:
    """Uncommitted changes, ignoring orchestration/ (as taskctl does) and new untracked run files."""
    out = git(root, "status", "--porcelain", "-uall", "--", ".", ":!orchestration")
    if out.returncode != 0:
        raise RuntimeError(f"git status failed: {out.stderr.strip()}")
    return any(ln and not ln.startswith(f"?? {RUNS_REL.as_posix()}/") for ln in out.stdout.splitlines())


def head_commit(root: Path) -> str:
    out = git(root, "rev-parse", "HEAD")
    if out.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {out.stderr.strip()}")
    return out.stdout.strip()


@dataclass
class SplitInfo:
    """What a split is made of: the ids a run must cover, their near-duplicate groups, labels, and a hash."""

    hash: str
    ids: list[str]
    groups: list[str]
    labels: list[str] | None  # class names; None when the split has none (RAG, perf)


def _manifest(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "datasets" / "manifests" / f"{name}.csv")


def split_info(split: str, root: Path = ROOT) -> SplitInfo:
    """Read a split from its source: manifest rows, RAG questions, or the fixed perf image list."""
    if split in ("rag_dev", "rag_test"):
        lines = (root / RAG_REL).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in lines if x.strip()]
        rows = [r for r in rows if r["split"] == split.removeprefix("rag_")]
        digest = hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest()
        ids = [r["id"] for r in rows]
        return SplitInfo(digest, ids, ids, None)
    if split == "perf":
        data = (root / PERF_REL).read_bytes()
        ids = [x.strip() for x in data.decode("utf-8").splitlines() if x.strip()]
        return SplitInfo(hashlib.sha256(data).hexdigest(), ids, ids, None)
    name = next((m for m in IMAGE_MANIFESTS if split.startswith(m + "_")), None)
    if name is None:
        raise ValueError(f"unknown split {split!r}")
    df = _manifest(root, name)
    part = df[df["split"] == split]
    return SplitInfo(dataset_hash(df[COLUMNS]), list(part["path"]), list(part["group"]), list(part["label"]))


def subset_positions(name: str, split: str, info: SplitInfo, root: Path = ROOT) -> list[int]:
    """Positions within the split of a declared subset: `first_<N>` rows, or `shared_classes`
    (labels present in both lab.csv and field.csv)."""
    first = re.fullmatch(r"first_(\d+)", name)
    if first:
        return list(range(min(int(first.group(1)), len(info.ids))))
    if name == "shared_classes" and info.labels is not None:
        shared = set(_manifest(root, "lab")["label"]) & set(_manifest(root, "field")["label"])
        return [i for i, label in enumerate(info.labels) if label in shared]
    raise ValueError(f"unknown subset {name!r} for split {split}")


def _rows(system_rows: Any) -> list[dict[str, Any]]:
    if isinstance(system_rows, Predictions):
        return [
            {"path": p, "label": int(y), "probs": [float(v) for v in pr]}
            for p, y, pr in zip(system_rows.path, system_rows.label, system_rows.probs)
        ]
    return list(system_rows)


def _write_samples(runs: Path, run_id: str, kind: str, samples: dict[str, list]) -> str:
    path = runs / f"{run_id}.{kind}.json"
    path.write_text(json.dumps(samples, indent=1) + "\n", encoding="utf-8")
    return path.relative_to(runs.parents[1]).as_posix()


def write_run(
    experiment: str,
    split: str,
    config: dict[str, Any],
    models: dict[str, str],
    metrics: dict[str, dict[str, float | None]],
    intervals: dict[str, dict[str, tuple[float, float]]],
    attempts: list[dict[str, Any]],
    predictions: dict[str, Any],
    dataset_hash: str,
    *,
    root: Path = ROOT,
    allow_dirty: bool = False,
    notes: str = "",
    subset: str | None = None,
    comparisons: list[dict[str, Any]] | None = None,
    measurements: dict[str, list[float]] | None = None,
    judged: dict[str, list[bool]] | None = None,
) -> Path:
    """Write results/runs/<experiment>-<timestamp>.json plus saved predictions, and return its path.

    `predictions` maps each system name to a `Predictions` or to a list of row dicts (RAG rows hold
    `id` and `rank`; perf rows hold `path`). `measurements` and `judged` hold the raw samples; the
    summary values are computed here and the samples are saved beside the record.
    """
    bad = [name for name in predictions if not SYSTEM_NAME.match(name)]
    if not predictions or bad:
        raise ValueError(f"predictions must map system names (letters, digits, _ and -) to rows: {bad}")
    dirty = is_dirty(root)
    if dirty and not allow_dirty:
        raise RuntimeError("working tree has uncommitted changes; commit first")
    now = datetime.now(timezone.utc)
    run_id = f"{experiment}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    runs = root / RUNS_REL
    runs.mkdir(parents=True, exist_ok=True)
    rows = {name: _rows(r) for name, r in predictions.items()}
    pred_paths = {}
    for name, system_rows in rows.items():
        p = runs / f"{run_id}.{name}.predictions.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in system_rows), encoding="utf-8")
        pred_paths[name] = p.relative_to(root).as_posix()
    measured: dict[str, Any] = {}
    if measurements:
        values = {f"{k}_p{q}": percentile(v, q) for k, v in measurements.items() for q in MEASURE_QUANTILES}
        measured = {"file": _write_samples(runs, run_id, "measurements", measurements), "values": values}
    judgments: dict[str, Any] = {}
    if judged:
        values = {k: sum(map(bool, v)) / len(v) for k, v in judged.items()}
        judgments = {"file": _write_samples(runs, run_id, "judged", judged), "values": values}
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "git_commit": head_commit(root),
        "dirty": dirty,
        "dataset_hash": dataset_hash,
        "split": split,
        "subset": subset,
        "config": config,
        "models": models,
        "n": len(next(iter(rows.values()))),
        "metrics": metrics,
        "measurements": measured,
        "judged": judgments,
        "intervals": {s: {k: list(v) for k, v in d.items()} for s, d in intervals.items()},
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "comparisons": comparisons or [],
        "attempts": attempts,
        "claude_calls": config.get("claude_calls", 0),
        "labels_reviewed": config.get("labels_reviewed", False),
        "predictions": pred_paths,
        "started_at": config.get("started_at", now.isoformat()),
        "finished_at": now.isoformat(),
        "notes": notes,
    }
    path = runs / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path
