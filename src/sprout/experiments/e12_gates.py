"""The `gates` stage of E12: G1, G2 and G3 of docs/DECISIONS.md, computed by configs/e12_gates.json's thresholds from the
gate runs' own records. It writes results/e12/gates.json and nothing else; choosing the model is left to the person, and
results/e12/decision.json is not touched. Run through
    .venv/Scripts/python -m sprout.experiments.e12_more_sources --config configs/e12_gates.json gates
Reads field_dev records and probabilities only; field_test is never opened.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sprout.atomic import atomic_write_text
from sprout.eval.check_run import (
    Scored,
    check_file,
    recompute_comparison,
    recompute_interval,
    recompute_metric,
)
from sprout.eval.records import ROOT, head_commit, split_info
from sprout.experiments import e5_confidence as e5
from sprout.experiments import e12_more_sources as e12


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([r["label"] for r in rows], dtype=int), np.array([r["probs"] for r in rows], dtype=float)


def healthy_flags(class_names: list[str], suffix: str) -> np.ndarray:
    return np.array([c.endswith(suffix) for c in class_names])


def _part(s: Scored, mask: np.ndarray) -> Scored:
    return Scored(groups=[g for g, k in zip(s.groups, mask, strict=True) if k], labels=s.labels[mask], probs=s.probs[mask])


def _interval(metric: str, s: Scored) -> list[float]:
    return [float(x) for x in recompute_interval(metric, s)]


def sick_healthy_split(new: Scored, old: Scored, healthy: np.ndarray) -> dict[str, Any]:
    """Macro-F1 on the rows whose true class is sick and accuracy on those whose true class is healthy, for both
    systems, with the paired difference (new minus old) and its interval. See configs/e12_gates.json `sick_only`."""
    is_healthy = healthy[new.labels]
    out: dict[str, Any] = {"n_sick": int((~is_healthy).sum()), "n_healthy": int(is_healthy.sum())}
    for part, mask, metric in (("sick_only_macro_f1", ~is_healthy, "macro_f1"), ("healthy_accuracy", is_healthy, "accuracy")):
        if not mask.any():  # PlantSeg's rows are all sick: there is no healthy accuracy to report
            out[part] = None
            continue
        a, b = _part(new, mask), _part(old, mask)
        got = recompute_comparison(a, b, metric)
        out[part] = {"new": float(recompute_metric(metric, a)), "old": float(recompute_metric(metric, b)),
                     "new_ci": _interval(metric, a), "old_ci": _interval(metric, b),
                     "diff": got["diff"], "ci": got["ci"], "ci_touches_zero": got["ci"][0] <= 0 <= got["ci"][1]}
    return out


# ----------------------------------------------------------------------------------------------------------------
# the three rules, on numbers only

def evaluate_g1(rule: dict[str, Any], comparison: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    gain, (lo, hi) = comparison["diff"], comparison["ci"]
    checks = {"gain_at_least_min": gain >= rule["min_gain"], "interval_above_zero": lo > rule["ci_lower_above"]}
    sick = split["sick_only_macro_f1"]
    return {
        "pass": all(checks.values()), "checks": checks, "gain": gain, "ci": [lo, hi], "min_gain": rule["min_gain"],
        "reference_gain": rule["reference_gain"], "split": split,
        "sick_only_interval_touches_zero": sick["ci_touches_zero"],
        "statement": ("The sick-only gain's interval touches zero, so the PlantDoc gain is not described as better "
                      "disease detection." if sick["ci_touches_zero"] else
                      "The sick-only gain's interval is clear of zero."),
    }


def evaluate_g2(rule: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Each pair holds `kind` ('seen' or a held-out source) and the paired macro-F1 comparison, E12 minus E4."""
    results = []
    for p in pairs:
        gain, (lo, hi) = p["comparison"]["diff"], p["comparison"]["ci"]
        if p["kind"] == "seen":
            ok = gain >= rule["seen_point_at_least"]
        else:
            ok = lo > rule["heldout_ci_lower_above"]
        results.append({**p, "pass": bool(ok), "gain": gain, "ci": [lo, hi]})
    return {"pass": all(r["pass"] for r in results), "pairs": results}


def evaluate_g3(rule: dict[str, Any], e4_count: int, e12_count: int) -> dict[str, Any]:
    return {"pass": e12_count <= e4_count + rule["max_more_confident_errors_than_e4"], "e4_confident": e4_count,
            "e12_confident": e12_count, "tie": e4_count == e12_count,
            "e4_count_too_low_to_mean_much": e4_count < rule["low_count_below"]}


# ----------------------------------------------------------------------------------------------------------------
# counting for G3

def confident_sick_called_healthy(probs: np.ndarray, labels: np.ndarray, healthy: np.ndarray, temperature: float,
                                  threshold: float) -> int:
    cal = e5.calibrate(probs, temperature)
    answered = cal.max(axis=1) >= threshold
    return int((~healthy[labels] & healthy[cal.argmax(axis=1)] & answered).sum())


def crossover_rates(probs: np.ndarray, labels: np.ndarray, healthy: np.ndarray) -> dict[str, Any]:
    """Uncalibrated, unthresholded: sick photos whose top class is a healthy class, and healthy photos whose top class
    is a sick one."""
    pred, is_healthy = probs.argmax(axis=1), healthy[labels]
    sick_n, healthy_n = int((~is_healthy).sum()), int(is_healthy.sum())
    s2h, h2s = int((~is_healthy & healthy[pred]).sum()), int((is_healthy & ~healthy[pred]).sum())
    return {"n": len(labels), "sick_rows": sick_n, "sick_called_healthy": s2h,
            "sick_called_healthy_rate": s2h / sick_n if sick_n else None,
            "healthy_rows": healthy_n, "healthy_called_sick": h2s,
            "healthy_called_sick_rate": h2s / healthy_n if healthy_n else None}


# ----------------------------------------------------------------------------------------------------------------
# the stage

def _records(cfg: dict[str, Any], root: Path, names: list[str]) -> dict[str, tuple[Path, dict[str, Any]]]:
    found = e12.newest_records({**cfg, "runs": [{"name": n} for n in names]}, root)
    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(f"no record yet for {missing}; run them first")
    for name, (path, _) in found.items():
        problems = check_file(path, root, min_n=cfg['gates']['min_n'])
        if problems:
            raise RuntimeError(f"{path.name} ({name}) does not pass check_run --min-n {cfg['gates']['min_n']}: {problems[0]}")
    return found


def _scored(rec: dict[str, Any], system: str, groups: dict[str, str], root: Path) -> tuple[Scored, list[str]]:
    rows = _read(root / rec["predictions"][system])
    labels, probs = _arrays(rows)
    paths = [r["path"] for r in rows]
    return Scored(groups=[groups[p] for p in paths], labels=labels, probs=probs), paths


def _comparison(rec: dict[str, Any], a: str, b: str, metric: str) -> dict[str, Any]:
    return e12._comparison(rec, a, b, metric)


def _minutes(rec: dict[str, Any]) -> float:
    return sum(h["seconds"] for a in rec["attempts"] for h in a.get("history", [])) / 60


def gate_g1(cfg: dict[str, Any], root: Path, recs: dict, groups: dict[str, str], healthy: np.ndarray) -> dict[str, Any]:
    rule = cfg["gates"]["g1"]
    rec = recs[rule["run"]][1]
    c = _comparison(rec, rule["run"], rule["baseline"], rule["metric"])
    new, new_paths = _scored(rec, rule["run"], groups, root)
    old, old_paths = _scored(rec, rule["baseline"], groups, root)
    if new_paths != old_paths or rec["config"]["trained_on_plantdoc"]:
        raise RuntimeError("the trimmed run and E4's held-out run must cover the same photos and not train on PlantDoc")
    out = evaluate_g1(rule, c, sick_healthy_split(new, old, healthy))
    listed = json.loads((root / rule["trim"]["list"]).read_text(encoding="utf-8"))
    return {**out, "n": rec["n"], "record": rec["run_id"], "baseline": rule["baseline"],
            "extras_removed": listed["union"], "extras_kept": listed["trimmed_rows"],
            "removed_against_evaluated": listed["against_evaluated"], "removed_against_training": listed["against_training"]}


def gate_g2(cfg: dict[str, Any], root: Path, recs: dict, groups: dict[str, str], healthy: np.ndarray) -> dict[str, Any]:
    rule, workers = cfg["gates"]["g2"], cfg["num_workers"]["default"]
    pairs, minutes = [], {"e12": 0.0, "e4": 0.0}
    for p in rule["pairs"]:
        rec, other = recs[p["e12"]][1], recs[p["e4"]][1]
        c = _comparison(rec, p["e12"], p["e4"], rule["metric"])
        acc = _comparison(rec, p["e12"], p["e4"], "accuracy")
        new, new_paths = _scored(rec, p["e12"], groups, root)
        old, old_paths = _scored(rec, p["e4"], groups, root)
        theirs = [r["path"] for r in _read(root / other["predictions"][p["e4"]])]
        mine = [r["config"]["train"] for r in (rec, other)]
        if not (new_paths == old_paths == theirs):
            raise RuntimeError(f"{p['kind']}: the pair does not cover the same photos in the same order")
        if any(t["seed"] != rule["seed"] or t["num_workers"] != workers for t in mine):
            raise RuntimeError(f"{p['kind']}: both runs must use seed {rule['seed']} and {workers} workers")
        minutes["e12"] += _minutes(rec)
        minutes["e4"] += _minutes(other)
        pairs.append({"kind": p["kind"], "e12": p["e12"], "e4": p["e4"], "n": rec["n"], "records": [rec["run_id"], other["run_id"]],
                      "comparison": c, "accuracy_comparison": acc,
                      "macro_f1": {"e12": rec["metrics"][p["e12"]]["macro_f1"], "e4": rec["metrics"][p["e4"]]["macro_f1"]},
                      "split": sick_healthy_split(new, old, healthy)})
    out = evaluate_g2(rule, pairs)
    out["gpu_minutes"] = {k: round(v, 1) for k, v in minutes.items()} | {"total": round(sum(minutes.values()), 1)}
    out["gpu_minutes_limit"] = rule["gpu_minutes_limit"]
    out["seed"] = rule["seed"]
    return out


def gate_g3(cfg: dict[str, Any], root: Path, recs: dict, healthy: np.ndarray) -> dict[str, Any]:
    rule = cfg["gates"]["g3"]
    chosen = {k: json.loads((root / rule[k]).read_text(encoding="utf-8")) for k in ("e4_chosen", "e12_chosen")}
    sha = hashlib.sha256((root / rule["e12_model"]).read_bytes()).hexdigest()
    if chosen["e12_chosen"].get("model_sha256") != sha:
        raise RuntimeError(f"{rule['e12_chosen']} was fitted for different weights from {rule['e12_model']}")
    seen = recs[rule["seen_record"]][1]
    e4_rows = _read(root / seen["predictions"]["convnext_joint"])
    e12_rows = _read(root / seen["predictions"][rule["seen_record"]])
    saved = {r["path"]: r for r in _read(root / rule["e12_probabilities"])}
    if any(saved[r["path"]]["probs"] != r["probs"] for r in e12_rows):
        raise RuntimeError("the saved field_dev probabilities differ from the E12 record's rows")
    (y4, p4), (y12, p12) = _arrays(e4_rows), _arrays(e12_rows)
    if [r["path"] for r in e4_rows] != [r["path"] for r in e12_rows] or not np.array_equal(y4, y12):
        raise RuntimeError("E4 and E12 rows for G3 are not the same photos")
    counts = {name: confident_sick_called_healthy(p, y4, healthy, c["temperature"], c["abstain_threshold"])
              for name, p, c in (("e4", p4, chosen["e4_chosen"]), ("e12", p12, chosen["e12_chosen"]))}
    out = evaluate_g3(rule, counts["e4"], counts["e12"])
    out["rows"] = len(y4)
    out["e5"] = {"e4": {k: chosen["e4_chosen"][k] for k in ("temperature", "abstain_threshold")},
                 "e12": {k: chosen["e12_chosen"][k] for k in ("temperature", "abstain_threshold", "model_sha256")}}
    rates = {"seen": {"e4": crossover_rates(p4, y4, healthy), "e12": crossover_rates(p12, y12, healthy)}}
    for name in rule["heldout_records"]:
        rec = recs[name][1]
        base = name.removesuffix("_extra")
        rates[base.removeprefix("heldout_")] = {
            side: crossover_rates(*reversed(_arrays(_read(root / rec["predictions"][system]))), healthy)
            for side, system in (("e4", base), ("e12", name))}
    out["unthresholded"] = rates
    return out


def g3_only(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    """G3 alone, from the first E12 records and the E5 refit, for when G1 and G2 aren't run. It needs no GPU, and it
    writes its own file with no all_pass, so a single gate can't read as all three."""
    g = cfg["gates"]
    recs = _records(cfg, root, [g["g3"]["seen_record"], *g["g3"]["heldout_records"]])
    healthy = healthy_flags(recs[g["g3"]["seen_record"]][1]["config"]["class_names"], g["healthy_suffix"])
    if not healthy.any() or healthy.all():
        raise RuntimeError("the class names hold no healthy/sick split")
    result = {
        "experiment": cfg["experiment"], "computed_at": datetime.now(timezone.utc).isoformat(), "git_commit": head_commit(root),
        "config": "configs/e12_gates.json", "records": {n: r["run_id"] for n, (_, r) in recs.items()},
        "g3": gate_g3(cfg, root, recs, healthy), "g1_g2": "not run",
        "field_test_read": False,
        "label": "development numbers, field_dev only; web photos, not phone photos",
    }
    path = root / Path(g["gates_file"]).with_name("g3.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(result, indent=1) + "\n")
    return path


def gates(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    g = cfg["gates"]
    first = ["convnext_joint_extra", *g["g3"]["heldout_records"]]
    names = [g["g1"]["run"], *(n for p in g["g2"]["pairs"] for n in (p["e12"], p["e4"]))]
    recs = {**_records(cfg, root, first), **_records(cfg, root, names)}
    info = split_info(e12.SPLIT, root)
    groups = dict(zip(info.ids, info.groups, strict=True))
    healthy = healthy_flags(recs[g["g1"]["run"]][1]["config"]["class_names"], g["healthy_suffix"])
    if not healthy.any() or healthy.all():
        raise RuntimeError("the class names hold no healthy/sick split")
    g1, g2, g3 = gate_g1(cfg, root, recs, groups, healthy), gate_g2(cfg, root, recs, groups, healthy), gate_g3(cfg, root, recs, healthy)
    result = {
        "experiment": cfg["experiment"], "computed_at": datetime.now(timezone.utc).isoformat(), "git_commit": head_commit(root),
        "config": "configs/e12_gates.json", "records": {n: r["run_id"] for n, (_, r) in recs.items()},
        "g1": g1, "g2": g2, "g3": g3, "all_pass": bool(g1["pass"] and g2["pass"] and g3["pass"]),
        "decision_unchanged": "results/e12/decision.json is not read or written here; the choice is the person's.",
        "field_test_read": False,
        "label": "development numbers, field_dev only; web photos, not phone photos",
    }
    path = root / g["gates_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(result, indent=1) + "\n")
    return path
