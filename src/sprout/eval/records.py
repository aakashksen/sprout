"""Run records: one JSON file per experiment and split, mapping each system to its saved predictions."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprout.atomic import atomic_write_text
from sprout.data.label_review import exclusion_paths, read_log
from sprout.data.splits import COLUMNS, dataset_hash
from sprout.eval import metrics as M
from sprout.eval.mlflow_log import log_record
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
IMAGE_MANIFESTS = ("lab", "field", "not_plant", "seg")


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def is_dirty(root: Path) -> bool:
    """Uncommitted changes, ignoring orchestration/ (as taskctl does), mlruns/ (logging output) and new untracked run files."""
    out = git(root, "--no-optional-locks", "status", "--porcelain", "-uall", "--", ".", ":!orchestration", ":!mlruns")
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
    expected: list[str] | None = None  # RAG gold behaviour per question: answer, abstain, ...


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
        return SplitInfo(digest, ids, ids, None, [r.get("expected", "answer") for r in rows])
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


def scoring_part(groups: list[str], seed: int, fit_percent: int) -> np.ndarray:
    """True for rows in the scoring part of a group-wise split: the sorted distinct groups are shuffled with
    `seed`, the first `fit_percent` percent of them form the fitting part, and the rest are scored. Every
    group lands wholly on one side."""
    distinct = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(distinct))
    fit = {distinct[i] for i in order[: round(len(distinct) * fit_percent / 100)]}
    return np.array([g not in fit for g in groups], dtype=bool)


def subset_positions(name: str, split: str, info: SplitInfo, root: Path = ROOT, commit: str | None = None) -> list[int]:
    """Positions within the split of a declared subset: `first_<N>` rows, `source:<name>` (field rows whose
    `source` column in field.csv is <name>), `shared_classes` (labels present in both lab.csv and field.csv),
    `final_classes` (labels in `classes` of configs/e15.json), `answerable` (RAG questions whose gold behaviour is to answer), `clean:<N>` (field rows not in datasets/label_exclusions.csv up to version N),
    `classes_in:<json path>` (rows whose label is in that file's `classes`, read at `commit` when one is given),
    `listed:<json path>` / `unlisted:<json path>` (rows whose path is, or is not, in that file's `paths`), or
    `group_part:<seed>:<fit_percent>` (the scoring part of `scoring_part`, for settings fitted on the rest).
    Parts joined with `+` must all hold."""
    if "+" in name:
        keep = None
        for part in name.split("+"):
            found = set(subset_positions(part, split, info, root, commit))
            keep = found if keep is None else keep & found
        return sorted(keep)
    clean = re.fullmatch(r"clean:(\d+)", name)
    if clean and split.startswith("field_"):
        version = int(clean.group(1))
        log = read_log(root / "datasets" / "label_exclusions.csv")
        if version > (int(log["version"].max()) if len(log) else 0):
            raise ValueError(f"exclusion version {version} is not in label_exclusions.csv")
        gone = exclusion_paths(log, version)
        return [i for i, path in enumerate(info.ids) if path not in gone]
    part = re.fullmatch(r"group_part:(\d+):(\d+)", name)
    if part:
        return [int(i) for i in np.flatnonzero(scoring_part(info.groups, int(part.group(1)), int(part.group(2))))]
    listed = re.fullmatch(r"classes_in:(.+)", name)
    if listed and info.labels is not None:
        if commit is None:
            text = (root / listed.group(1)).read_text(encoding="utf-8")
        else:  # the file as it was at the record's commit
            shown = git(root, "show", f"{commit}:{listed.group(1)}")
            if shown.returncode != 0:
                raise ValueError(f"{listed.group(1)} cannot be read at commit {commit}")
            text = shown.stdout
        names = set(json.loads(text)["classes"])
        return [i for i, label in enumerate(info.labels) if label in names]
    paths = re.fullmatch(r"(listed|unlisted):(.+)", name)
    if paths:
        if commit is None:
            text = (root / paths.group(2)).read_text(encoding="utf-8")
        else:
            shown = git(root, "show", f"{commit}:{paths.group(2)}")
            if shown.returncode != 0:
                raise ValueError(f"{paths.group(2)} cannot be read at commit {commit}")
            text = shown.stdout
        listed = set(json.loads(text)["paths"])
        want = paths.group(1) == "listed"
        return [i for i, path in enumerate(info.ids) if (path in listed) == want]
    first = re.fullmatch(r"first_(\d+)", name)
    if first:
        return list(range(min(int(first.group(1)), len(info.ids))))
    source = re.fullmatch(r"source:(.+)", name)
    if source and split.startswith("field_"):
        part = _manifest(root, "field")
        part = part[part["split"] == split]
        return [i for i, s in enumerate(part["source"]) if s == source.group(1)]
    if name == "answerable" and info.expected is not None:
        return [i for i, e in enumerate(info.expected) if e == "answer"]
    if name == "final_classes" and info.labels is not None:
        keep = set(json.loads((root / "configs" / "e15.json").read_text(encoding="utf-8"))["classes"])
        return [i for i, label in enumerate(info.labels) if label in keep]
    if name == "shared_classes" and info.labels is not None:
        shared = set(_manifest(root, "lab")["label"]) & set(_manifest(root, "field")["label"])
        return [i for i, label in enumerate(info.labels) if label in shared]
    raise ValueError(f"unknown subset {name!r} for split {split}")


def _rows(system_rows: Any) -> list[dict[str, Any]]:
    if isinstance(system_rows, Predictions):
        return [
            {"path": p, "label": int(y), "probs": [float(v) for v in pr]}
            for p, y, pr in zip(system_rows.path, system_rows.label, system_rows.probs, strict=True)
        ]
    return list(system_rows)


def standard_image_metrics(probs, labels, class_names: list[str]) -> dict[str, float | None]:
    """The metrics every image evaluation reports, under the names check_run recomputes. Plant-given
    numbers are None when undefined."""
    probs = np.asarray(probs, dtype=float)
    out: dict[str, float | None] = {
        "accuracy": M.top_k_accuracy(probs, labels, 1),
        "macro_f1": M.macro_f1(labels, probs.argmax(axis=1)),
        "ece": M.ece(probs, labels),
    }
    for k in (1, 3, 5):
        out[f"top_{k}_accuracy"] = M.top_k_accuracy(probs, labels, k)
    for k in (1, 3):
        v = M.plant_given_top_k(probs, labels, class_names, k)
        out[f"plant_given_top_{k}"] = None if v != v else v
    return out


def classification_tables(rows: list[dict[str, Any]], class_names: list[str]) -> tuple[dict, list]:
    """Per-class precision, recall, F1 and support, and the confusion matrix, from saved rows."""
    labels = np.array([r["label"] for r in rows], dtype=int)
    pred = np.array([r["probs"] for r in rows], dtype=float).argmax(axis=1)
    return M.per_class_report(labels, pred, class_names), M.confusion(labels, pred, len(class_names)).tolist()


def _write_samples(runs: Path, run_id: str, kind: str, samples: dict[str, list]) -> str:
    path = runs / f"{run_id}.{kind}.json"
    atomic_write_text(path, json.dumps(samples, indent=1) + "\n")
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
    if any(m.startswith("plant_given_top_") for v in metrics.values() for m in v) and isinstance(
        config.get("class_names"), list
    ):
        config = {**config, "options_per_plant": M.options_per_plant(config["class_names"])}
    dirty = is_dirty(root)
    if dirty and not allow_dirty:
        raise RuntimeError("working tree has uncommitted changes; commit first")
    now = datetime.now(timezone.utc)
    run_id = f"{experiment}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    runs = root / RUNS_REL
    runs.mkdir(parents=True, exist_ok=True)
    rows = {name: _rows(r) for name, r in predictions.items()}
    tables = {}
    for name, system_rows in rows.items():
        if system_rows and "probs" in system_rows[0] and "label" in system_rows[0]:
            names = config.get("class_names")
            if not isinstance(names, list):
                raise ValueError("config.class_names is required for classification predictions")
            tables[name] = classification_tables(system_rows, names)
    pred_paths = {}
    for name, system_rows in rows.items():
        p = runs / f"{run_id}.{name}.predictions.jsonl"
        atomic_write_text(p, "".join(json.dumps(r) + "\n" for r in system_rows))
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
        "per_class": {k: v[0] for k, v in tables.items()},
        "confusion": {k: v[1] for k, v in tables.items()},
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
    atomic_write_text(path, json.dumps(record, indent=1) + "\n")  # last: a record implies its files
    try:
        log_record(record, root)
    except Exception as exc:  # the record is already saved and stays the source of truth
        warnings.warn(f"{run_id}: the record was saved but not logged to MLflow: {exc!r}", stacklevel=2)
    return path
