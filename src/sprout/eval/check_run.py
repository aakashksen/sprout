"""Check run records against docs/TESTING.md: every saved number must recompute from saved files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sprout.eval import agreement, answer_checks, check_cache
from sprout.eval import metrics as M
from sprout.eval.records import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    ROOT,
    RUNS_REL,
    SplitInfo,
    git,
    split_info,
    subset_positions,
)
from sprout.eval.stats import (
    bootstrap_ci,
    macro_f1_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_macro_f1_diff,
    percentile,
)

REQUIRED = (
    "run_id experiment git_commit dirty dataset_hash split subset config models n metrics measurements judged "
    "intervals bootstrap per_class confusion comparisons attempts claude_calls labels_reviewed predictions started_at finished_at notes"
).split()
TOLERANCE = 1e-9
NOT_PLANT = "not_plant"
SIDECAR = re.compile(r".+\.(measurements|judged)\.json")
FAKE = re.compile(r"fake|stub", re.I)
TIMESTAMP = re.compile(r"\d{8}T\d{12}Z")
# Every labeled image-classification system must report these. E1, E2 and E3 are exempt by name: their
# records predate the rule or were running without it, and results/topk-baselines.json covers them.
REQUIRED_IMAGE_METRICS = (
    "accuracy macro_f1 ece top_1_accuracy top_3_accuracy top_5_accuracy plant_given_top_1 plant_given_top_3"
).split()
EXEMPT_EXPERIMENTS = ("E1", "E2", "E3")
THRESHOLDED = ("coverage", "selective_accuracy", "confident_error_rate")


@dataclass
class Scored:
    """One system's saved rows, aligned to the split (or declared subset) in manifest order."""

    groups: list[str]
    labels: np.ndarray | None = None
    class_names: list[str] | None = None
    probs: np.ndarray | None = None
    ranks: list[int | None] | None = None
    accepted: np.ndarray | None = None  # gate decision per image
    is_plant: np.ndarray | None = None  # from the manifest's labels
    answered: np.ndarray | None = None  # RAG: answered rather than declined
    expected: list[str] | None = None  # RAG gold behaviour from the questions file
    iou: np.ndarray | None = None  # segmentation: per-image IoU, recomputed from the saved masks


def _num(x: float) -> float | None:
    return None if x != x else float(x)


def _classification_series(name: str, s: Scored) -> np.ndarray | None:
    correct = (s.probs.argmax(axis=1) == s.labels).astype(float)
    if name == "accuracy":
        return correct
    if name == "end_to_end_accuracy" and s.accepted is not None:
        return (correct.astype(bool) & s.accepted).astype(float)
    m = re.fullmatch(r"coverage@([\d.]+)(?:~([\d.]+))?", name)
    if m:
        if m.group(2) is not None:
            from sprout.serve.calibrate import answered_mask

            return answered_mask(s.probs, s.class_names, float(m.group(1)), float(m.group(2))).astype(float)
        return (s.probs.max(axis=1) >= float(m.group(1))).astype(float)
    return None


def _retrieval_series(name: str, ranks: list[int | None]) -> np.ndarray | None:
    m = re.fullmatch(r"recall@(\d+)", name)
    if m:
        return np.array([r is not None and r <= int(m.group(1)) for r in ranks], dtype=float)
    if name == "mrr":
        return np.array([0.0 if r is None else 1.0 / r for r in ranks])
    return None


def item_series(name: str, s: Scored) -> np.ndarray | None:
    """Per-item values whose mean is the metric, for metrics that have an interval or comparison."""
    if s.probs is not None and s.labels is not None:
        found = _classification_series(name, s)
        if found is not None:
            return found
    if s.iou is not None:
        if name == "mean_iou":
            return s.iou
        if name == "iou_ge_0.5":
            return (s.iou >= 0.5).astype(float)
    if s.ranks is not None:
        return _retrieval_series(name, s.ranks)
    return None


def _rate(flags: np.ndarray, among: np.ndarray) -> float | None:
    return float(flags[among].mean()) if among.any() else None


_OTHER = object()


def _gate_or_rag_metric(name: str, s: Scored) -> object:
    """Gate, RAG and segmentation rates; _OTHER when the name is not one of them."""
    if name == "plant_rejection_rate" and s.accepted is not None and s.is_plant is not None:
        return _rate(~s.accepted, s.is_plant)
    if name == "not_plant_acceptance_rate" and s.accepted is not None and s.is_plant is not None:
        return _rate(s.accepted, ~s.is_plant)
    declines = {"declines_when_unanswerable": "abstain", "declines_when_crop_mismatch": "crop_mismatch",
                "declines_when_ambiguous": "clarify"}
    if (name == "answers_when_answerable" or name in declines) and s.answered is not None:
        gold = np.array(s.expected or [])
        if name == "answers_when_answerable":
            return _rate(s.answered, gold == "answer")
        return _rate(~s.answered, gold == declines[name])
    if name == "median_iou" and s.iou is not None:
        return float(np.median(s.iou))
    return _OTHER


def _probability_metric(name: str, s: Scored) -> object:
    """Metrics that need probabilities and labels; _OTHER when the name is not one of them."""
    if name == "macro_f1":
        return M.macro_f1(s.labels, s.probs.argmax(axis=1))
    if name == "ece":
        return M.ece(s.probs, s.labels)
    m = re.fullmatch(r"top_(\d+)_accuracy", name)
    if m and int(m.group(1)) >= 1:
        return M.top_k_accuracy(s.probs, s.labels, int(m.group(1)))
    m = re.fullmatch(r"plant_given_top_(\d+)", name)
    if m and int(m.group(1)) >= 1:
        if s.class_names is None:
            return None
        return _num(M.plant_given_top_k(s.probs, s.labels, s.class_names, int(m.group(1))))
    m = re.fullmatch(r"(selective_accuracy|confident_error_rate)@([\d.]+)(?:~([\d.]+))?", name)
    if m:
        healthy = float(m.group(3)) if m.group(3) is not None else None
        return _num(M.selective_metrics(s.probs, s.labels, float(m.group(2)), s.class_names, healthy)[m.group(1)])
    return _OTHER


def recompute_metric(name: str, s: Scored, others: dict[str, Scored] | None = None) -> float | None:
    """The metric from saved rows, using sprout.eval.metrics. None when it is undefined for these rows
    (nothing to divide by, or the rows lack the needed fields). Raises KeyError for an unknown metric."""
    found = _gate_or_rag_metric(name, s)
    if found is not _OTHER:
        return found
    changed = re.fullmatch(r"changed_vs@([A-Za-z0-9_-]+)", name)
    if changed:
        other = (others or {}).get(changed.group(1))
        if other is None:
            raise KeyError(name)
        if s.probs is None or other.probs is None:
            return None
        return float((s.probs.argmax(axis=1) != other.probs.argmax(axis=1)).mean())
    series = item_series(name, s)
    if series is not None:
        return float(series.mean())
    if s.probs is not None and s.labels is not None:
        found = _probability_metric(name, s)
        if found is not _OTHER:
            return found
    raise KeyError(name)


def recompute_interval(
    name: str, s: Scored, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float] | None:
    """95% interval resampling whole near-duplicate groups; macro_f1 is recomputed inside each resample."""
    if name == "macro_f1" and s.probs is not None and s.labels is not None:
        return macro_f1_ci(s.labels, s.probs.argmax(axis=1), s.groups, s.probs.shape[1], n, seed)
    series = item_series(name, s)
    return None if series is None else bootstrap_ci(series, s.groups, n, seed)


def recompute_comparison(
    a: Scored, b: Scored, metric: str, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any] | None:
    """System a minus system b over the same resampled groups, as saved in a record's `comparisons`.
    McNemar is always on per-image correctness (accuracy, or recall@k for retrieval; for mrr, first-rank hits)."""
    if a.groups != b.groups:
        return None
    if metric == "macro_f1":
        if a.probs is None or b.probs is None or a.labels is None or not np.array_equal(a.labels, b.labels):
            return None
        pa, pb = a.probs.argmax(axis=1), b.probs.argmax(axis=1)
        diff, lo, hi = paired_macro_f1_diff(a.labels, pa, pb, a.groups, a.probs.shape[1], n, seed)
        p = mcnemar_exact(pa == a.labels, pb == b.labels)
        return {"diff": diff, "ci": [lo, hi], "mcnemar_p": p}
    if metric not in ("accuracy", "mrr") and not re.fullmatch(r"recall@\d+", metric):
        return None
    sa, sb = item_series(metric, a), item_series(metric, b)
    if sa is None or sb is None or len(sa) != len(sb):
        return None
    diff, lo, hi = paired_bootstrap_diff(sa, sb, a.groups, n, seed)
    return {"diff": diff, "ci": [lo, hi], "mcnemar_p": mcnemar_exact(sa > 0.5, sb > 0.5)}


def _same(fresh: float | None, saved: object) -> bool:
    if fresh is None or saved is None:
        return fresh is None and saved is None
    return isinstance(saved, (int, float)) and abs(fresh - saved) <= TOLERANCE


def _same_pair(fresh, saved: object) -> bool:
    return isinstance(saved, (list, tuple)) and len(saved) == 2 and all(_same(f, v) for f, v in zip(fresh, saved, strict=True))


def _strings(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def _timestamp(name: str) -> str:
    found = TIMESTAMP.findall(name)
    return found[-1] if found else name


def records_for(experiment: str, root: Path = ROOT) -> list[Path]:
    """Every record of the experiment, oldest first."""
    pat = re.compile(rf"{re.escape(experiment)}-{TIMESTAMP.pattern}\.json")
    found = [p for p in (root / RUNS_REL).glob(f"{experiment}-*.json") if pat.fullmatch(p.name)]
    return sorted(found, key=lambda p: (_timestamp(p.name), p.name))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# A gate row that saves the numbers its decision came from must agree with them: accepted is the AND of
# each saved (score, threshold) pair. Rows without any of these fields (older records) are not checked.
GATE_RULES = (("confidence", "threshold", np.greater_equal), ("ood_score", "ood_threshold", np.less_equal),
              ("green_ratio", "green_threshold", np.greater_equal))


def _gate_problems(rows: list[dict[str, Any]], accepted: np.ndarray, tag: str) -> list[str]:
    pairs = [(s, t, op) for s, t, op in GATE_RULES if s in rows[0] or t in rows[0]]
    if not pairs:
        return []
    expected = np.ones(len(rows), dtype=bool)
    for score, threshold, op in pairs:
        try:
            values = np.array([r[score] for r in rows], dtype=float)
            limits = np.array([r[threshold] for r in rows], dtype=float)
        except (KeyError, TypeError, ValueError):
            return [tag + f"gate rows must hold numeric {score} and {threshold} together"]
        if not (np.isfinite(values).all() and np.isfinite(limits).all()):
            return [tag + f"{score} and {threshold} must be finite in every row"]
        expected &= op(values, limits)
    wrong = int((expected != accepted).sum())
    return [tag + f"accepted does not follow from the saved scores and thresholds in {wrong} rows"] if wrong else []


def _score_gate(ordered, info, positions, scored, tag, out) -> bool:
    flags = [r.get("accepted") for r in ordered]
    if not all(isinstance(f, bool) for f in flags):
        out.append(tag + "accepted must be true or false in every row")
        return False
    scored.accepted = np.array(flags)
    out.extend(_gate_problems(ordered, scored.accepted, tag))
    if info.labels is not None:
        scored.is_plant = np.array([info.labels[i] != NOT_PLANT for i in positions])
    return True


def _score_rag(ordered, info, positions, scored, tag, out) -> bool:
    flags = [r.get("answered") for r in ordered]
    if not all(isinstance(f, bool) for f in flags) or info.expected is None:
        out.append(tag + "answered must be true or false in every row of a RAG split")
        return False
    scored.answered = np.array(flags)
    scored.expected = [info.expected[i] for i in positions]
    return True


def _score_probs(ordered, info, positions, rec, scored, tag, out) -> bool:
    names = rec["config"].get("class_names") if isinstance(rec["config"], dict) else None
    probs = np.array([r["probs"] for r in ordered], dtype=float)
    labels = np.array([r["label"] for r in ordered], dtype=int)
    if not isinstance(names, list) or probs.ndim != 2 or probs.shape[1] != len(names):
        out.append(tag + "config.class_names must list the classes, one per probability column")
        return False
    if info.labels is not None and [names[x] for x in labels] != [info.labels[i] for i in positions]:
        out.append(tag + "saved labels do not match the manifest's labels")
        return False
    scored.probs, scored.labels, scored.class_names = probs, labels, names
    return True


def _score_iou(ordered, scored, tag, out) -> bool:
    from sprout.segment.data import rle_iou

    try:
        fresh = np.array([rle_iou(r["pred"], r["gt"]) for r in ordered])
    except (KeyError, ValueError, TypeError) as exc:
        out.append(tag + f"masks cannot be decoded to recompute IoU: {exc}")
        return False
    saved = [r["iou"] for r in ordered]
    if not all(isinstance(v, (int, float)) and abs(v - f) <= TOLERANCE for v, f in zip(saved, fresh, strict=True)):
        out.append(tag + "a row's iou does not recompute from its saved masks")
    scored.iou = fresh
    return True


def _score(
    rows: list[dict[str, Any]], info: SplitInfo, positions: list[int], rec: dict[str, Any], tag: str, out: list[str]
) -> Scored | None:
    """Align a system's rows to the split in manifest order; None (with problems added) when they don't fit."""
    if len({frozenset(r) for r in rows}) != 1:
        out.append(tag + "every row must have the same fields")
        return None
    key = "path" if "path" in rows[0] else "id"
    ids = [r.get(key) for r in rows]
    wanted = [info.ids[i] for i in positions]
    if len(set(ids)) != len(ids) or set(ids) != set(wanted):
        out.append(tag + "prediction ids do not match the split's ids exactly once each")
        return None
    by_id = dict(zip(ids, rows, strict=True))
    ordered = [by_id[i] for i in wanted]
    scored = Scored(groups=[info.groups[i] for i in positions])
    steps = (
        ("accepted", lambda: _score_gate(ordered, info, positions, scored, tag, out)),
        ("answered", lambda: _score_rag(ordered, info, positions, scored, tag, out)),
        ("probs", lambda: _score_probs(ordered, info, positions, rec, scored, tag, out)),
        ("iou", lambda: _score_iou(ordered, scored, tag, out)),
    )
    for field, step in steps:
        if field in rows[0] and not step():
            return None
    if "rank" in rows[0]:
        scored.ranks = [r["rank"] for r in ordered]
    return scored


def _measurement_problems(block: Any, root: Path, kind: str) -> list[str]:
    """Saved summary values must recompute from the saved sample file."""
    if not block:
        return []
    path = root / block["file"] if isinstance(block, dict) and "file" in block else None
    if path is None or not path.is_file():
        return [f"{kind} file is missing"]
    samples = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for name, saved in block.get("values", {}).items():
        if kind == "judged":
            fresh = float(np.mean([bool(x) for x in samples[name]])) if name in samples and samples[name] else None
        else:
            m = re.fullmatch(r"(.+)_p(\d+)", name)
            fresh = percentile(samples[m.group(1)], int(m.group(2))) if m and m.group(1) in samples else None
        if fresh is None:
            out.append(f"{kind} value {name} has no samples in {block['file']}")
        elif not _same(fresh, saved):
            out.append(f"{kind} value {name} does not match its saved file")
    return out


def _table_problems(rec: dict[str, Any], name: str, s: Scored, tag: str) -> list[str]:
    """Per-class report and confusion matrix: saved for every classification system, and recomputed."""
    saved_pc, saved_cm = rec["per_class"].get(name), rec["confusion"].get(name)
    if s.probs is None or s.labels is None:
        return [tag + f"{k} given for rows that are not classifications" for k, v in
                (("per_class", saved_pc), ("confusion", saved_cm)) if v]
    names, pred = rec["config"]["class_names"], s.probs.argmax(axis=1)
    out = []
    fresh_pc = M.per_class_report(s.labels, pred, names)
    if not isinstance(saved_pc, dict) or set(saved_pc) != set(fresh_pc) or not all(
        set(saved_pc[c]) == set(v) and all(_same(float(v[k]), saved_pc[c][k]) for k in v)
        for c, v in fresh_pc.items()
    ):
        out.append(tag + "per_class does not recompute from predictions")
    if saved_cm != M.confusion(s.labels, pred, len(names)).tolist():
        out.append(tag + "confusion does not recompute from predictions")
    return out


def _judge_problems(config: Any, models: Any) -> list[str]:
    judge = config.get("judge") if isinstance(config, dict) else None
    model = judge.get("model") if isinstance(judge, dict) else None
    if not (isinstance(model, str) and model.strip()):
        return ["config.judge.model is missing: judged values must name their judge"]
    out = []
    if FAKE.search(model):
        out.append("the judge model matches fake or stub")
    if isinstance(models, dict) and model == models.get("writer"):
        out.append("the judge must be a different model from the writer")
    return out


def _nonfinite(x: Any, path: str = "", loss_only: bool = False) -> list[str]:
    """Paths of NaN or infinite numbers inside nested JSON values (json.load accepts NaN and Infinity)."""
    if isinstance(x, float):
        return [] if math.isfinite(x) or (loss_only and "loss" not in path.rsplit(".", 1)[-1]) else [path]
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in _nonfinite(v, f"{path}.{k}" if path else str(k), loss_only)]
    if isinstance(x, list):
        return [p for i, v in enumerate(x) for p in _nonfinite(v, f"{path}[{i}]" if path else f"[{i}]", loss_only)]
    return []


def _finite_problems(rec: dict[str, Any]) -> list[str]:
    out = []
    for key in ("metrics", "intervals", "comparisons"):
        bad = _nonfinite(rec.get(key), key)
        if bad:
            out.append(f"non-finite value in {key}: {', '.join(bad[:3])}")
    bad = _nonfinite(rec.get("attempts"), "attempts", loss_only=True)
    if bad:
        out.append(f"attempts history holds a non-finite loss: {', '.join(bad[:3])}")
    return out


def _stamp_problems(rec: dict[str, Any]) -> list[str]:
    """finished_at must be the write time that run_id carries."""
    found = TIMESTAMP.findall(str(rec.get("run_id")))
    try:
        stamp = datetime.strptime(found[-1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        ok = datetime.fromisoformat(rec["finished_at"]) == stamp
    except (IndexError, ValueError, TypeError):
        ok = False
    return [] if ok else ["finished_at disagrees with the timestamp in run_id"]


def _ancestry_problems(rec: dict[str, Any], root: Path) -> list[str]:
    if git(root, "merge-base", "--is-ancestor", rec["git_commit"], "HEAD").returncode != 0:
        return ["git_commit is not an ancestor of HEAD"]
    return []


def _header_problems(rec: dict[str, Any], root: Path, min_n: int) -> list[str]:
    problems = []
    if rec["dirty"]:
        problems.append("record was made on a dirty tree")
    problems += _stamp_problems(rec)
    if rec["experiment"] == "E9":
        saved = (rec["config"] or {}).get("labels_reviewed")
        if rec["labels_reviewed"] is not saved:
            problems.append("labels_reviewed differs from config.labels_reviewed")
    problems += _finite_problems(rec)
    if any(FAKE.search(s) for s in _strings(rec["models"])):
        problems.append("a model name matches fake or stub")
    if not rec["attempts"]:
        problems.append("attempts is empty")
    problems += _ancestry_problems(rec, root)
    if min_n and rec["n"] < min_n:
        problems.append(f"n={rec['n']} is below --min-n {min_n}")
    return problems


def _scope(rec: dict[str, Any], root: Path, problems: list[str]) -> tuple[SplitInfo, list[int]] | None:
    """The split and the positions scored; None (with the reason added) when they can't be read."""
    try:
        info = split_info(rec["split"], root)
    except (ValueError, OSError) as exc:
        problems.append(f"cannot read split {rec['split']}: {exc}")
        return None
    if rec["dataset_hash"] != info.hash:
        problems.append("dataset_hash does not match the manifest")
    positions = list(range(len(info.ids)))
    if rec["subset"] is not None:
        try:
            positions = subset_positions(rec["subset"], rec["split"], info, root, rec.get("git_commit"))
        except ValueError as exc:
            problems.append(str(exc))
            return None
    if not positions or rec["n"] == 0:
        problems.append("no scored rows: the split or subset is empty")
    what = f"subset {rec['subset']}" if rec["subset"] else "split"
    if rec["n"] != len(positions):
        problems.append(f"n={rec['n']} differs from the {what}'s {len(positions)} items")
    return info, positions


def _score_systems(
    rec: dict[str, Any], root: Path, info: SplitInfo, positions: list[int], problems: list[str]
) -> dict[str, Scored]:
    paths, metrics = rec["predictions"], rec["metrics"]
    usable: dict[str, Scored] = {}
    for name, rel in paths.items():
        tag = f"system {name}: "
        if not (root / rel).is_file():
            problems.append(tag + "predictions file is missing")
            continue
        rows = _read_rows(root / rel)
        if len(rows) != rec["n"]:
            problems.append(tag + f"predictions hold {len(rows)} rows but n={rec['n']}")
        if not metrics[name] and rec["split"] != "perf":
            problems.append(tag + "metrics is empty")
        if not rows:
            problems.append(tag + "no scored rows: the predictions file is empty")
            continue
        scored = _score(rows, info, positions, rec, tag, problems)
        if scored is not None:
            usable[name] = scored

    for name in set(paths) - set(usable):
        if f"system {name}: " not in " ".join(problems):
            problems.append(f"system {name}: was not scored")
    return usable


def _metric_set_problems(rec: dict[str, Any], usable: dict[str, Scored]) -> list[str]:
    metrics, problems = rec["metrics"], []
    # Exempt by reason: not_plant splits have no class labels, and systems without probabilities (a gate's
    # accept/reject rows) are not classifiers. seg_dev, RAG and perf splits are not image classification.
    if rec["experiment"] not in EXEMPT_EXPERIMENTS and not rec["split"].startswith(NOT_PLANT):
        for name, scored in usable.items():
            if scored.probs is not None and scored.labels is not None:
                missing = [m for m in REQUIRED_IMAGE_METRICS if m not in metrics[name]]
                if missing:
                    problems.append(f"system {name}: missing required metric(s) {', '.join(missing)}")

    if any(m.startswith("plant_given_top_") for v in metrics.values() for m in v):
        names = rec["config"].get("class_names")
        stated = rec["config"].get("options_per_plant")
        if not isinstance(names, list) or stated != M.options_per_plant(names):
            problems.append("a record with plant-given metrics must state config.options_per_plant, matching class_names")
    return problems


def _system_problems(
    rec: dict[str, Any], usable: dict[str, Scored], intervals: dict[str, Any], n_boot: int, seed: int
) -> list[str]:
    problems = []
    for name, scored in usable.items():
        tag = f"system {name}: "
        for metric, saved in rec["metrics"][name].items():
            try:
                fresh = recompute_metric(metric, scored, usable)
            except KeyError:
                problems.append(tag + f"metric {metric} cannot be recomputed from predictions")
                continue
            if not _same(fresh, saved):
                problems.append(tag + f"metric {metric} does not recompute from predictions")
        for metric, saved in (intervals.get(name) or {}).items():
            got = recompute_interval(metric, scored, n_boot, seed)
            if got is None:
                problems.append(tag + f"interval {metric} cannot be recomputed from predictions")
            elif not _same_pair(got, saved):
                problems.append(tag + f"interval {metric} does not recompute from predictions")
        problems += _table_problems(rec, name, scored, tag)
    return problems


def _comparison_problems(rec: dict[str, Any], usable: dict[str, Scored], n_boot: int, seed: int) -> list[str]:
    problems = []
    for i, c in enumerate(rec["comparisons"]):
        tag = f"comparison {i}: "
        if not (isinstance(c, dict) and {"a", "b", "metric", "diff", "ci", "mcnemar_p"} <= set(c)):
            problems.append(tag + "needs a, b, metric, diff, ci and mcnemar_p")
        elif c["a"] == c["b"] or c["a"] not in usable or c["b"] not in usable:
            problems.append(tag + "a and b must be two different systems with usable predictions")
        else:
            fresh = recompute_comparison(usable[c["a"]], usable[c["b"]], c["metric"], n_boot, seed)
            if fresh is None:
                problems.append(tag + f"{c['metric']} cannot be recomputed for a comparison")
            elif not (_same(fresh["diff"], c["diff"]) and _same_pair(fresh["ci"], c["ci"])
                      and _same(fresh["mcnemar_p"], c["mcnemar_p"])):
                problems.append(tag + f"{c['metric']} for {c['a']} vs {c['b']} does not recompute from predictions")
    return problems


def _shape_problem(rec: dict[str, Any]) -> str | None:
    """Why the saved predictions, metrics or bootstrap can't be read system by system, if they can't."""
    paths, metrics, boot = rec["predictions"], rec["metrics"], rec["bootstrap"]
    if not isinstance(paths, dict) or not paths:
        return "predictions must map each system to its predictions file"
    if not isinstance(metrics, dict) or set(metrics) != set(paths):
        return "metrics must hold one entry per system, matching predictions"
    if not (isinstance(boot, dict) and isinstance(boot.get("n"), int) and isinstance(boot.get("seed"), int)):
        return "bootstrap must record n and seed"
    return None


def _same_source(cfg: Any, rows: list[dict[str, Any]], root: Path) -> list[str]:
    """E14's local grades must be the ones in the judgment file it names, unchanged."""
    src = cfg.get("source") if isinstance(cfg, dict) else None
    path = root / RUNS_REL / src["judgments"] if isinstance(src, dict) and "judgments" in src else None
    if path is None or not path.is_file():
        return ["the source judgments file named in config.source is missing"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != src.get("judgments_sha256"):
        return ["the source judgments file differs from the sha256 in config.source"]
    saved = {(r["system"], r["id"]): ([c["claim"] for c in r["claims"]], [bool(c["verdict"]) for c in r["claims"]])
             for r in _read_rows(path)}
    bad = sum(saved.get((r["system"], r["id"])) != (r["claims"], r["local"]) for r in rows)
    return [f"{bad} rows' claims or local grades differ from the source judgments"] if bad else []


def _judge_check_problems(rec: dict[str, Any], root: Path, min_n: int) -> list[str]:
    """E14 grades answers, not split items: rows are one per sampled (question, system) answer, all from rag_dev.
    Every metric and interval is recomputed from them with sprout.eval.agreement."""
    problems = _header_problems(rec, root, min_n)
    if rec["split"] != "rag_dev":
        return problems + ["E14 grades rag_dev answers only"]
    info = split_info(rec["split"], root)
    if rec["dataset_hash"] != info.hash:
        problems.append("dataset_hash does not match the manifest")
    paths, metrics, boot, cfg = rec["predictions"], rec["metrics"], rec["bootstrap"], rec["config"]
    if not (isinstance(paths, dict) and set(paths) == {"sample"} and isinstance(metrics, dict) and set(metrics) == {"sample"}):
        return problems + ["E14 holds one system, sample, in predictions and metrics"]
    if not (isinstance(boot, dict) and boot.get("n") == BOOTSTRAP_N and isinstance(boot.get("seed"), int)):
        return problems + [f"bootstrap must record n={BOOTSTRAP_N:,} and a seed"]
    served = cfg.get("served_system") if isinstance(cfg, dict) else None
    if not isinstance(served, str) or not (root / paths["sample"]).is_file():
        return problems + ["config.served_system and the sample predictions file are required"]
    rows = _read_rows(root / paths["sample"])
    if len(rows) != rec["n"]:
        problems.append(f"predictions hold {len(rows)} rows but n={rec['n']}")
    shape = agreement.row_problems(rows, set(info.ids))
    if shape:
        return problems + shape
    if rec["claude_calls"] != sum(r["calls"] for r in rows):
        problems.append("claude_calls differs from the calls the rows record")
    problems += _same_source(cfg, rows, root)
    fresh = agreement.metrics_from_rows(rows, served)
    for name, saved in metrics["sample"].items():
        if name not in fresh:
            problems.append(f"metric {name} cannot be recomputed from predictions")
        elif not _same(fresh[name], saved):
            problems.append(f"metric {name} does not recompute from predictions")
    fresh_ci = agreement.intervals_from_rows(rows, served, boot["n"], boot["seed"])
    for name, saved in (rec["intervals"].get("sample") or {}).items():
        if name not in fresh_ci:
            problems.append(f"interval {name} cannot be recomputed from predictions")
        elif not _same_pair(fresh_ci[name], saved):
            problems.append(f"interval {name} does not recompute from predictions")
    if set(rec["intervals"]) - {"sample"}:
        problems.append("intervals name systems not in predictions")
    return problems


def _prompt_problems(rec: dict[str, Any], root: Path) -> list[str]:
    """E9: the wrong-disease prompt the record names must be the file's text at the commit the record was made on."""
    cfg = rec["config"]
    rel, want = cfg.get("prompt_file"), cfg.get("prompt_sha256")
    if not (isinstance(rel, str) and isinstance(want, str)):
        return ["config must name prompt_file and prompt_sha256"]
    shown = subprocess.run(["git", "show", f"{rec['git_commit']}:{rel}"], cwd=root, capture_output=True)
    if shown.returncode != 0 or hashlib.sha256(shown.stdout).hexdigest() != want:
        return [f"prompt_sha256 is not the hash of {rel} at the record's commit"]
    return []


def _e9_dev_rows(rec: dict[str, Any], info: SplitInfo, rows: list[dict[str, Any]], root: Path) -> list[str]:
    """Rows of the wrong-disease dev stage: one per checked answer, from the sha-checked E7 judgments file."""
    cfg, problems = rec["config"], []
    if any(set(r) != set(answer_checks.DEV_ROW_KEYS) for r in rows):
        return ["every row must hold exactly " + ", ".join(answer_checks.DEV_ROW_KEYS)]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(info.ids):
        problems.append("rows must be distinct rag_dev questions")
    if any(r["call"] != answer_checks.parse_call(str(r["reply"])) for r in rows):
        problems.append("a row's call is not what its reply says")
    src = cfg.get("source")
    path = root / RUNS_REL / src["judgments"] if isinstance(src, dict) and "judgments" in src else None
    if path is None or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != src.get("judgments_sha256"):
        return problems + ["the source judgments file named in config.source is missing or differs from its sha256"]
    saved = {(r["system"], r["id"]): (r["question"], r["answer"]) for r in _read_rows(path)}
    bad = sum(saved.get((cfg.get("served_system"), r["id"])) != (r["question"], r["answer"]) for r in rows)
    if bad:
        problems.append(f"{bad} rows' question or answer differ from the source judgments")
    if rec["claude_calls"] != len(rows):
        problems.append("claude_calls must be one per checked answer")
    return problems


def _e9_dev_problems(rec: dict[str, Any], root: Path, min_n: int) -> list[str]:
    """E9 stage records on rag_dev (the wrong-disease dev run). Rows are not one per split item, so
    they are checked on their own and every number is recomputed with sprout.eval.answer_checks."""
    problems = _header_problems(rec, root, min_n)
    info = split_info(rec["split"], root)
    if rec["dataset_hash"] != info.hash:
        problems.append("dataset_hash does not match the manifest")
    paths, metrics, cfg = rec["predictions"], rec["metrics"], rec["config"]
    if not (isinstance(paths, dict) and set(paths) == {"served"} and isinstance(metrics, dict) and set(metrics) == {"served"}):
        return problems + ["an E9 stage record holds one system, served, in predictions and metrics"]
    if not (root / paths["served"]).is_file():
        return problems + ["the predictions file is missing"]
    rows = _read_rows(root / paths["served"])
    if len(rows) != rec["n"]:
        problems.append(f"predictions hold {len(rows)} rows but n={rec['n']}")
    stage = cfg.get("stage") if isinstance(cfg, dict) else None
    if stage == "wrong_disease_dev":
        found, fresh, fresh_ci = _e9_dev_rows(rec, info, rows, root), None, None
        if not found:
            fresh, fresh_ci = answer_checks.dev_metrics(rows), answer_checks.dev_intervals(rows)
    else:
        return problems + [f"unknown E9 stage {stage!r} on rag_dev"]
    problems += found + _prompt_problems(rec, root)
    if fresh is not None:
        saved = metrics["served"]
        if set(saved) != set(fresh) or not all(_same(fresh[k], saved[k]) for k in fresh):
            problems.append("metrics do not recompute from predictions")
        got = (rec["intervals"].get("served") or {})
        if set(got) != set(fresh_ci) or not all(_same_pair(fresh_ci[k], got[k]) for k in fresh_ci):
            problems.append("intervals do not recompute from predictions")
    if set(rec["intervals"]) - {"served"}:
        problems.append("intervals name systems not in predictions")
    return problems


def _e9_final_problems(rec: dict[str, Any], root: Path) -> list[str]:
    """E9 rag_test: the saved claims must be the answer's own claims, each verdict must follow from its reply, the judged
    samples must be the rows' own claims, the first judge must be the one the committed Haiku check names, and the prompt
    must be the frozen one. Neither judge's figure may be called accuracy, upper or conservative."""
    from sprout.experiments.e14_judge_check import parse_verdict
    from sprout.rag.judge import split_claims

    cfg, problems = rec["config"], _prompt_problems(rec, root)
    shown = subprocess.run(["git", "show", f"{rec['git_commit']}:configs/e9_wrong_disease.json"], cwd=root,
                           capture_output=True)
    frozen = json.loads(shown.stdout) if shown.returncode == 0 else {}
    if frozen.get("prompt_sha256") != cfg.get("prompt_sha256"):
        problems.append("config.prompt_sha256 is not the frozen prompt's hash at the record's commit")
    problems += _frozen_record_problems(frozen, root)
    rows = _read_rows(root / rec["predictions"]["served"])
    if any(set(r) != set(answer_checks.FINAL_ROW_KEYS) for r in rows) or any(
            set(c) != set(answer_checks.CLAIM_KEYS) for r in rows for c in r["claims"]):
        return problems + ["every row and claim must hold exactly the E9 fields"]
    bad = {"claims": 0, "haiku": 0, "sonnet": 0, "check": 0, "calls": 0}
    for r in rows:
        want = [c.text for c in split_claims(r["text"])] if r["state"] in ("answered", "ungrounded") else []
        bad["claims"] += [c["claim"] for c in r["claims"]] != want
        bad["haiku"] += any(c["haiku"] != parse_verdict(str(c["haiku_reply"])) for c in r["claims"])
        bad["sonnet"] += any(c["sonnet"] != parse_verdict(str(c["sonnet_reply"])) for c in r["claims"])
        reply, status = r["wrong_disease_reply"], r["wrong_disease_status"]
        bad["check"] += (r["wrong_disease"] != answer_checks.parse_call(str(reply or ""))
                         or status not in answer_checks.WD_STATUSES or (status == "no_claims") != (not r["claims"])
                         or (status == "graded") != (reply is not None)
                         or (status != "refused_by_safeguard" and r["wrong_disease_request_id"] is not None))
        bad["calls"] += (r["haiku_calls"] != len(r["claims"])
                         or r["sonnet_calls"] != len(r["claims"]) + (status in ("graded", "refused_by_safeguard")))
    for key, what in (("claims", "claims do not follow from the answer text"),
                      ("haiku", "a Haiku verdict does not match its reply"),
                      ("sonnet", "a Sonnet verdict does not match its reply"),
                      ("check", "a wrong-disease call does not match its reply or its answer has no claim to check"),
                      ("calls", "haiku_calls or sonnet_calls is not one per claim (plus one per check for Sonnet)")):
        if bad[key]:
            problems.append(f"{bad[key]} rows: {what}")
    calls = {"haiku": sum(r["haiku_calls"] for r in rows), "sonnet": sum(r["sonnet_calls"] for r in rows)}
    if cfg.get("calls") != calls or rec["claude_calls"] != calls["haiku"] + calls["sonnet"]:
        problems.append("claude_calls or config.calls differ from the calls the rows record")
    counts = answer_checks.wd_counts(rows)
    held = cfg.get("wrong_disease_check")
    if not isinstance(held, dict) or {k: held.get(k) for k in counts} != counts:
        problems.append("config.wrong_disease_check does not recompute from the rows")
    if not counts["complete"] and "partial" not in json.dumps([rec["notes"], cfg.get("caveat")]).lower():
        problems.append("the wrong-disease check is partial (refused or unsent rows) but the record does not say so")
    if counts["complete"] and "partial" in str(rec["notes"]).lower():
        problems.append("the record calls a complete wrong-disease check partial")
    cov = cfg.get("generic_followup_coverage")
    if not (isinstance(cov, dict) and cov.get("questions") == len(rows)
            and cov.get("generic_followups") == sum(bool(r["generic_followup"]) for r in rows)
            and cov.get("fallback_fired") == sum(bool(r["fallback_fired"]) for r in rows)):
        problems.append("config.generic_followup_coverage does not recompute from the rows")
    side = root / rec["judged"]["file"] if isinstance(rec["judged"], dict) and "file" in rec["judged"] else None
    if side is None or not side.is_file() or json.loads(side.read_text(encoding="utf-8")) != answer_checks.final_lists(rows):
        problems.append("the judged samples are not the rows' own claim verdicts and checks")
    problems += _first_judge_problems(rec, root)
    problems += _account_problems(rec, rows, root)
    if re.search(r"accuracy|\bupper\b|conservative", json.dumps({k: rec[k] for k in ("config", "models", "notes", "judged", "metrics")}), re.I):
        problems.append("a judge's figure is called accuracy, upper or conservative; it is a share of claims judged supported")
    return problems


def superseded_by(root: Path, run_id: str) -> str | None:
    """The E9 rag_test record that names `run_id` in config.supersedes, if any. A superseded record stays committed as
    history; claims must cite the one that replaces it."""
    for path in sorted((Path(root) / RUNS_REL).glob("E9-*.json")):
        if SIDECAR.fullmatch(path.name) or path.stem == run_id:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        held = (rec.get("config") or {}).get("supersedes")
        if rec.get("split") == "rag_test" and isinstance(held, dict) and held.get("record") == run_id:
            return path.stem
    return None


def _account_problems(rec: dict[str, Any], rows: list[dict[str, Any]], root: Path) -> list[str]:
    """E9 rag_test: a record whose wrong-disease check was refused by the safeguard says so in config.run_account and in
    its notes, unless a later record supersedes it (then it is history). The account must agree with the rows."""
    cfg, out = rec["config"], []
    counts = answer_checks.wd_counts(rows)
    account = cfg.get("run_account")
    interrupted = counts["refused_by_safeguard"] > 0
    history = superseded_by(root, rec["run_id"]) is not None
    if isinstance(account, dict):
        i = account.get("interruption") or {}
        if bool(i.get("occurred")) != interrupted or [(r.get("id"), r.get("request_id")) for r in i.get("refused", [])] != [
                (r["id"], r["request_id"]) for r in counts["refused"]]:
            out.append("config.run_account.interruption does not match the refused rows")
        if (account.get("window") or {}).get("record_written") != cfg.get("started_at"):
            out.append("config.run_account.window.record_written is not the record's own started_at")
        if account.get("model_calls_by_rag_record") != 0:
            out.append("config.run_account must say the record made no model call")
        if interrupted and not str(i.get("note") or "").strip():
            out.append("config.run_account.interruption has refused rows but no interruption note")
        if interrupted and "interrupt" not in str(rec.get("notes", "")).lower():
            out.append("the notes do not mention the interruption")
    elif interrupted and not history:
        out.append("the wrong-disease check was interrupted by a safeguard refusal but the record has no config.run_account")
    if interrupted and not history and re.search(r"from one run with everything frozen", json.dumps([rec.get("notes"), cfg.get("caveat")])):
        out.append("the record calls itself one run with everything frozen though the stages were interrupted")
    held = cfg.get("supersedes")
    if held is not None:
        path = root / RUNS_REL / f"{held.get('record')}.json" if isinstance(held, dict) else None
        if path is None or not path.is_file():
            out.append("config.supersedes must name a record that exists")
        else:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_judged = json.loads((root / old["judged"]["file"]).read_text(encoding="utf-8"))
            same = old["metrics"] == rec["metrics"] and old_judged == json.loads((root / rec["judged"]["file"]).read_text(encoding="utf-8"))
            if old.get("experiment") != "E9" or old.get("split") != "rag_test":
                out.append("config.supersedes names a record that is not an E9 rag_test record")
            if held.get("same_numbers") is not same or not same:
                out.append("a superseding record must carry the superseded record's numbers, and same_numbers must say so truly")
            if not _valid_note(rec):
                out.append("a superseding record is a second rag_test record and needs an override_note")
    return out


def _first_judge_problems(rec: dict[str, Any], root: Path) -> list[str]:
    """The Haiku judge is the first judge the committed Haiku check names, at the commit the record was made on."""
    judges = rec["config"].get("judges")
    if not (isinstance(judges, dict) and isinstance(judges.get("haiku"), dict) and isinstance(judges.get("sonnet"), dict)
            and judges.get("same_claims_and_rubric") is True):
        return ["config.judges must name the Haiku judge and the Sonnet judge, reading the same claims and rubric"]
    shown = git(root, "show", f"{rec['git_commit']}:results/e14/haiku_decision.json")
    decision = json.loads(shown.stdout) if shown.returncode == 0 else {}
    if decision.get("first_judge") != "haiku" or judges["haiku"].get("model") != decision.get("model") \
            or rec["models"].get("judge_first") != decision.get("model")             or (rec["config"].get("judge") or {}).get("model") != decision.get("model"):
        return ["the first judge is not the model the committed Haiku decision (results/e14/haiku_decision.json) names"]
    return []


def _e9_roles_problems(rec: dict[str, Any], root: Path) -> list[str]:
    """E9 images: config.primary and the alternates are the candidates the committed decision picks. The frozen config
    and the decision are read at the commit the record names, so a later decision does not change an earlier record.
    An alternate whose checkpoint was not on disk is named in config.alternates_not_scored instead of being scored."""
    from sprout.eval import e9_roles

    cfg = rec["config"]
    frozen = git(root, "show", f"{rec['git_commit']}:configs/e9.json")
    if frozen.returncode != 0:
        return []  # no E9 config was committed where this record was made (a bare fixture, not a real E9 run)
    read = {"frozen config": json.loads(frozen.stdout)}
    got = git(root, "show", f"{rec['git_commit']}:{read['frozen config'].get('decision_file')}")
    if got.returncode != 0:
        return ["the decision cannot be read at the record's commit, so primary and alternate cannot be re-derived"]
    e15 = git(root, "show", f"{rec['git_commit']}:{read['frozen config'].get('e15_config')}")
    try:
        decision = e9_roles.normalized(json.loads(got.stdout), json.loads(e15.stdout) if e15.returncode == 0 else {})
        specs = e9_roles.candidate_specs(read["frozen config"].get("candidates", {}), decision)
        primary, others = e9_roles.roles(read["frozen config"].get("candidates", {}), decision)
    except (ValueError, StopIteration) as exc:
        return [f"the committed decision cannot be matched to a candidate: {exc}"]
    out = []
    wants = [("primary", primary)] + [(e9_roles.system_of(i), k) for i, k in enumerate(others)]
    left = {x.get("candidate") for x in cfg.get("alternates_not_scored") or [] if isinstance(x, dict)}
    for role, key in wants:
        have = cfg.get(role)
        if have is None and key in left and role != "primary":
            continue
        want = {"candidate": key, "checkpoint": specs[key]["checkpoint"]}
        if role == "primary":
            want["sha256"] = decision.get("chosen_model_sha256")
        if not isinstance(have, dict) or any(have.get(k) != v for k, v in want.items()):
            out.append(f"config.{role} differs from the committed decision (it should be {key}, {want['checkpoint']})")
    named = {r for r, _ in wants}
    if any(k.startswith("alternate") and k not in named for k in cfg if k not in ("alternates_not_scored",)):
        out.append("config names an alternate the committed decision does not")
    if cfg.get("chosen_model") != decision.get("chosen_model"):
        out.append("config.chosen_model differs from the committed decision")
    return out


def _e9_exif_problems(rec: dict[str, Any], root: Path, info: SplitInfo) -> list[str]:
    """config.exif counts the photos of the whole scored set, which the shared-classes view of a lab record shares, so
    it is counted again from the headers of the set the subset names without that view."""
    parts = [x for x in (rec["subset"] or "").split("+") if x and x != "shared_classes"]
    try:
        positions = subset_positions("+".join(parts), rec["split"], info, root, rec.get("git_commit")) if parts else range(len(info.ids))
        fresh = answer_checks.exif_counts([root / info.ids[i] for i in positions])
    except (OSError, ValueError):
        return ["config.exif cannot be recomputed: some scored photos are not on disk"]
    return [] if rec["config"].get("exif") == fresh else ["config.exif does not recompute from the photos' headers"]


def _e9_thresholds_problems(rec: dict[str, Any], root: Path) -> list[str]:
    """config.quantized_thresholds is E6's own account, read again from the E6 record it names."""
    cfg = rec["config"]
    saved = cfg.get("quantized_thresholds")
    path = root / RUNS_REL / f"{saved.get('e6_record')}.json" if isinstance(saved, dict) else None
    if path is None or not path.is_file():
        return ["config.quantized_thresholds must name an E6 record that exists"]
    e6 = json.loads(path.read_text(encoding="utf-8"))
    held = (e6.get("config") or {}).get("thresholds") or {}
    row = ((held.get("systems") or {}).get(saved.get("variant")) or {}).get("scoring_part")
    if e6.get("experiment") != "E6" or e6["config"].get("checkpoint_sha256") != cfg.get("chosen_model_sha256") or row is None:
        return ["config.quantized_thresholds: the named record is not an E6 record of the chosen checkpoint for this variant"]
    want = {"e6_record": e6["run_id"], "variant": saved["variant"], "e6_split": e6["split"], "file": held["file"],
            "temperature": held["temperature"], "abstain_threshold": held["abstain_threshold"],
            "names_this_checkpoint": held["names_this_checkpoint"], "on_e6_rows": row}
    return [] if saved == want else ["config.quantized_thresholds does not match the E6 record it names"]


def _e9_image_problems(rec: dict[str, Any], root: Path, info: SplitInfo, positions: list[int]) -> list[str]:
    """E9 lab_test and field_test: the cleaned-row counts recompute from the exclusion log, and each calibrated system is its plain system's saved probabilities at E5's temperature,
    with E5's threshold in its metric names."""
    from sprout.data.label_review import read_log
    from sprout.serve.calibrate import apply_temperature

    cfg, problems = rec["config"], _e9_roles_problems(rec, root)
    if rec["split"] == "field_test":
        saved = cfg.get("cleaned_rows")
        version = saved.get("version") if isinstance(saved, dict) else None
        parts = str(rec["subset"] or "").split("+")
        known = all(re.fullmatch(r"clean:\d+|classes_in:.+", x) for x in parts)  # nothing else may ride along
        if not isinstance(version, int) or not known or parts.count(f"clean:{version}") != 1:
            problems.append("an E9 field_test record must be scored on its cleaned rows (subset clean:N) and state config.cleaned_rows")
        elif saved != answer_checks.cleaned_rows(read_log(root / "datasets" / "label_exclusions.csv"), version, info.ids,
                                                 [info.ids[i] for i in subset_positions(f"clean:{version}", "field_test", info, root)]):
            problems.append("config.cleaned_rows does not recompute from the exclusion log")
    real = git(root, "cat-file", "-e", f"{rec['git_commit']}:configs/e9.json").returncode == 0  # else a bare fixture
    if real:
        problems += _e9_exif_problems(rec, root, info) + _e9_thresholds_problems(rec, root)
    cal = cfg.get("calibration")
    bound = cfg.get("calibration_binding")
    for system in ("improved", *[k for k in ("alternate", "alternate_2") if k in cfg], "quantized") if real else ():
        role = cfg.get(system if system.startswith("alternate") else "primary")
        have = bound.get(system) if isinstance(bound, dict) else None
        fitted = cal.get(system) if isinstance(cal, dict) else None
        if not (isinstance(have, dict) and isinstance(role, dict)):
            problems.append(f"config.calibration_binding must say which checkpoint {system}'s calibration is bound to")
        elif have.get("sha256") != role.get("sha256") or have.get("calibration_file") != (fitted or {}).get("file"):
            problems.append(f"config.calibration_binding for {system} does not match the checkpoint it is scored on "
                            "and the calibration file used")
    for name, rel in rec["predictions"].items():
        base = name.removesuffix("_calibrated")
        if base == name:
            continue
        settings = cal.get(base) if isinstance(cal, dict) else None
        if not (isinstance(settings, dict) and {"temperature", "abstain_threshold"} <= set(settings)):
            problems.append(f"{name}: config.calibration must hold temperature and abstain_threshold for {base}")
            continue
        t = f"{settings['abstain_threshold']:.6f}".rstrip("0").rstrip(".")
        if base not in rec["predictions"] or not (root / rel).is_file() or not (root / rec["predictions"][base]).is_file():
            problems.append(f"{name}: calibrated system without its plain system's predictions")
            continue
        raw = {r["path"]: r["probs"] for r in _read_rows(root / rec["predictions"][base])}
        got = {r["path"]: r["probs"] for r in _read_rows(root / rel)}
        fresh = {k: apply_temperature(np.log(np.maximum(np.array(v, dtype=float), 1e-12)), settings["temperature"])
                 for k, v in raw.items()}
        if set(got) != set(fresh) or not all(np.allclose(got[k], fresh[k], atol=1e-9, rtol=0) for k in got):
            problems.append(f"{name}: probabilities are not the plain system's at the calibration temperature")
        if f"coverage@{t}" not in rec["metrics"].get(name, {}):
            problems.append(f"{name}: metrics lack coverage@{t}, the calibration's abstention threshold")
        if settings.get("healthy_threshold") is not None:
            h = f"{settings['healthy_threshold']:.6f}".rstrip("0").rstrip(".")
            if f"coverage@{t}~{h}" not in rec["metrics"].get(name, {}):
                problems.append(f"{name}: metrics lack coverage@{t}~{h}, abstention as served (healthy bar {h})")
    return problems


def _frozen_record_problems(frozen: dict[str, Any], root: Path) -> list[str]:
    """The frozen config's dev run must exist, have been made before the freeze, and name the frozen prompt's hash; the
    frozen config must say the check was not validated."""
    out = []
    if frozen.get("validated") is not False:
        out.append("the frozen config must say validated: false (the check was not read blind)")
    try:
        frozen_at = datetime.fromisoformat(frozen["frozen_at"])
    except (KeyError, TypeError, ValueError):
        return out + ["the frozen config has no valid frozen_at"]
    rid = frozen.get("dev_record")
    path = root / RUNS_REL / f"{rid}.json"
    if not path.is_file():
        return out + [f"frozen dev_record {rid} does not exist"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    cfg = rec.get("config") or {}
    if not (rec.get("experiment") == "E9" and rec.get("split") == "rag_dev" and cfg.get("stage") == "wrong_disease_dev"):
        out.append(f"dev_record {rid} is not an E9 wrong_disease_dev record on rag_dev")
    if cfg.get("prompt_sha256") != frozen.get("prompt_sha256"):
        out.append(f"dev_record {rid}: its prompt_sha256 is not the frozen prompt's")
    try:
        if datetime.fromisoformat(rec["finished_at"]) >= frozen_at:
            out.append(f"dev_record {rid}: finished at or after the freeze; it must be made before the freeze")
    except (KeyError, TypeError, ValueError):
        out.append(f"dev_record {rid}: has no valid finished_at to compare with the freeze")
    return out


def check_record(rec: dict[str, Any], root: Path, min_n: int = 0) -> list[str]:
    problems = [f"missing field {k}" for k in REQUIRED if k not in rec]
    if problems:
        return problems
    if rec["experiment"] == "E14":
        return _judge_check_problems(rec, root, min_n)
    if rec["experiment"] == "E9" and rec["split"] == "rag_dev":
        return _e9_dev_problems(rec, root, min_n)
    if rec["experiment"] == "E14H":
        from sprout.experiments.e14h_haiku_check import problems as haiku_check_problems

        return haiku_check_problems(rec, root, min_n)
    problems = _header_problems(rec, root, min_n)
    scope = _scope(rec, root, problems)
    if scope is None:
        return problems
    info, positions = scope
    if rec["experiment"] == "E9" and rec["split"] in ("lab_test", "field_test"):
        problems += _e9_image_problems(rec, root, info, positions)

    shape = _shape_problem(rec)
    if shape and not shape.startswith("bootstrap"):
        return problems + [shape]
    intervals = rec["intervals"]
    if not set(intervals) <= set(rec["predictions"]):
        problems.append(f"intervals name systems not in predictions: {sorted(set(intervals) - set(rec['predictions']))}")
        intervals = {}
    if shape:
        return problems + [shape]
    boot = rec["bootstrap"]
    if boot["n"] != BOOTSTRAP_N:
        problems.append(f"bootstrap n={boot['n']}, but intervals need {BOOTSTRAP_N:,} resamples")

    usable = _score_systems(rec, root, info, positions, problems)
    problems += _metric_set_problems(rec, usable)
    problems += _system_problems(rec, usable, intervals, boot["n"], boot["seed"])
    problems += _comparison_problems(rec, usable, boot["n"], boot["seed"])
    problems += _measurement_problems(rec["measurements"], root, "measurements")
    problems += _measurement_problems(rec["judged"], root, "judged")
    if rec["judged"]:
        problems += _judge_problems(rec["config"], rec["models"])
    if rec["experiment"] == "E9" and rec["split"] == "rag_test":
        problems += (_e9_final_problems(rec, root) if set(rec["predictions"]) == {"served"}
                     else ["an E9 rag_test record holds one system, served"])
    return problems


def _load_records(experiment: str, root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in records_for(experiment, root)]


def _row_map(root: Path, rel: str) -> dict[str, str]:
    rows = _read_rows(root / rel) if (root / rel).is_file() else []
    return {str(r.get("path", r.get("id"))): json.dumps(r, sort_keys=True) for r in rows}


def _is_view_of(view: dict[str, Any], full: dict[str, Any], root: Path) -> bool:
    """A declared subset drawn from the same commit, models, config and predictions as a full record."""
    parts = lambda r: set((r.get("subset") or "").split("+")) - {""}  # noqa: E731
    if view.get("subset") is None or parts(view) == parts(full) or not parts(full) <= parts(view):
        return False  # a view narrows its full record: it holds every part of the full record's subset and more
    drop = lambda c: {k: v for k, v in (c or {}).items() if k != "override_note"}  # noqa: E731
    if any(view.get(k) != full.get(k) for k in ("git_commit", "models", "split")) or drop(view.get("config")) != drop(
        full.get("config")
    ):
        return False
    for system, rel in (view.get("predictions") or {}).items():
        if system not in (full.get("predictions") or {}):
            return False
        theirs = _row_map(root, full["predictions"][system])
        if not all(theirs.get(k) == v for k, v in _row_map(root, rel).items()):
            return False
    return True


def _valid_note(rec: dict[str, Any]) -> bool:
    note = (rec.get("config") or {}).get("override_note")
    return isinstance(note, str) and bool(note.strip())


def _history(root: Path) -> list[tuple[str, str]]:
    """(commit, path) for every file ever added under results/runs, oldest first. Raises when git can't say."""
    out = git(root, "log", "--reverse", "--no-renames", "--diff-filter=A", "--name-only",
              "--format=commit %H", "--", RUNS_REL.as_posix())
    if out.returncode != 0:
        raise RuntimeError(f"git log failed: {out.stderr.strip()}")
    pairs, commit = [], ""
    for line in out.stdout.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.strip():
            pairs.append((commit, line.strip()))
    return pairs


def _uses(root: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every test-split record that has ever existed (committed history plus working tree), in order of use:
    committed ones by the commit that added them, then uncommitted ones by the timestamp in their name."""
    recs: dict[str, tuple[tuple, dict[str, Any]]] = {}
    for index, (commit, rel) in enumerate(_history(root)):
        name = rel.rsplit("/", 1)[-1]
        if not name.endswith(".json") or SIDECAR.fullmatch(name):
            continue
        if name[:-5] in recs:
            continue  # the first add decides the order
        path = root / rel
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            shown = git(root, "show", f"{commit}:{rel}")
            if shown.returncode != 0:
                raise RuntimeError(f"git show failed for {rel}")
            text = shown.stdout
        recs[name[:-5]] = ((0, index, _timestamp(name)), json.loads(text))
    for p in (root / RUNS_REL).glob("*.json"):
        if p.stem not in recs and not SIDECAR.fullmatch(p.name):
            recs[p.stem] = ((1, 0, _timestamp(p.name)), json.loads(p.read_text(encoding="utf-8")))
    return [(stem, rec) for stem, (_, rec) in sorted(recs.items(), key=lambda kv: (kv[1][0], kv[0]))
            if isinstance(rec, dict) and str(rec.get("split", "")).endswith("_test")]


def test_once_problems(root: Path) -> list[str]:
    """A test split is scored once. Every record that has ever existed on it counts, whatever its file name or
    experiment, deleted or not; a later one needs a non-empty string config.override_note. A declared subset
    of a full record's own predictions (same commit, models, config) is a view of that use, not another."""
    by_split: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for stem, rec in _uses(root):
        by_split.setdefault(rec["split"], []).append((stem, rec))
    out = []
    for split, recs in by_split.items():
        uses = [(s, r) for s, r in recs if not any(_is_view_of(r, f, root) for t, f in recs if t != s)]
        out += [f"{stem}: more than one {split} record and no override_note in config"
                for stem, r in uses[1:] if not _valid_note(r)]
    return out


def check_file(path: Path, root: Path, min_n: int = 0, once: list[str] | None = None) -> list[str]:
    """Every problem with one record file. check_run and claims both call this, so they agree."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if rec.get("run_id") != path.stem or not TIMESTAMP.search(path.stem):
        problems.append("the record's run_id must equal its file name and end in a write timestamp")
    problems += check_record(rec, root, min_n)
    once = test_once_problems(root) if once is None else once
    return problems + [m.split(": ", 1)[1] for m in once if m.startswith(f"{path.stem}: ")]


def check(
    experiment: str, *, root: Path = ROOT, min_n: int = 0, split: str | None = None, use_cache: bool = False
) -> list[str]:
    """With use_cache, a record that fully passed before, with the same checker, data and files, is not
    rechecked in full. What depends on HEAD or on other records (ancestry, the test-once rule) is still checked."""
    recs = _load_records(experiment, root)
    if split:
        recs = [(p, r) for p, r in recs if r.get("split") == split]
    if not recs:
        return [f"no run record for {experiment}"]
    once = test_once_problems(root)
    passed = check_cache.load(root) if use_cache else []
    known, shared, problems, fresh = set(passed), check_cache.shared_fingerprint(root) if use_cache else "", [], []
    for p, rec in recs:
        key = check_cache.record_key(p, rec, root, min_n, shared) if use_cache else ""
        if key in known:
            found = _ancestry_problems(rec, root) if "git_commit" in rec else []
            found += [m.split(": ", 1)[1] for m in once if m.startswith(f"{p.stem}: ")]
        else:
            found = check_file(p, root, min_n, once)
            if use_cache and not found:
                fresh.append(key)
        problems += [f"{p.name}: {m}" for m in found]
    if fresh:
        check_cache.save(root, passed + fresh)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--split")
    ap.add_argument("--all", action="store_true", help="recheck every record, ignoring the cache of passed ones")
    a = ap.parse_args(argv)
    problems = check(a.experiment, root=ROOT, min_n=a.min_n, split=a.split, use_cache=not a.all)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"{a.experiment} ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
