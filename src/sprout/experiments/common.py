"""Pieces E1, E2 and E3 all need: file hashes, the standard scoring, and the E1 checkpoint check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sprout.eval.check_run import Scored, recompute_interval, recompute_metric
from sprout.eval.predict import Predictions
from sprout.eval.records import RUNS_REL
from sprout.train.checkpoint import load_checkpoint

METRICS = ("accuracy", "macro_f1", "ece")
INTERVALS = ("accuracy", "macro_f1")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_systems(split_rows: pd.DataFrame, systems: dict[str, Predictions]):
    """Metrics and intervals per system, recomputed the way check_run does."""
    groups = list(split_rows["group"])
    scored = {n: Scored(groups=groups, labels=p.label, probs=p.probs) for n, p in systems.items()}
    metrics = {n: {m: recompute_metric(m, s) for m in METRICS} for n, s in scored.items()}
    intervals = {n: {m: recompute_interval(m, s) for m in INTERVALS} for n, s in scored.items()}
    return metrics, intervals


def load_e1(cfg: dict[str, Any], root: Path, who: str):
    """The E1 checkpoint, after checking it is the one the named E1 record produced.
    Returns the checkpoint, that record's first attempt, and the checkpoint's sha256."""
    ckpt_path = root / cfg["checkpoint"]
    if not ckpt_path.is_file():
        raise RuntimeError(f"E1 checkpoint not found at {ckpt_path}; {who} cannot run without it")
    e1_path = root / RUNS_REL / f"{cfg['e1_run_id']}.json"
    if not e1_path.is_file():
        raise RuntimeError(f"E1 record {cfg['e1_run_id']} not found at {e1_path}")
    ckpt = load_checkpoint(ckpt_path)
    attempt = json.loads(e1_path.read_text(encoding="utf-8"))["attempts"][0]
    if abs(ckpt["best_metric"] - attempt["best_dev_accuracy"]) > 1e-9:
        raise RuntimeError(
            f"checkpoint best lab_dev accuracy {ckpt['best_metric']} differs from E1 record's "
            f"{attempt['best_dev_accuracy']}; it was not produced by that run"
        )
    return ckpt, attempt, sha256_file(ckpt_path)
