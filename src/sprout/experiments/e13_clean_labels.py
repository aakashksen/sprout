"""E13: the earlier real-photo models scored again on cleaned field_dev, beside their old numbers.

Nothing is trained or run here. Every system's rows come from its saved predictions in an earlier record on the
same field_dev rows, filtered to the rows that survive the label exclusions (so the comparison is exact and
paired, and needs no GPU). A system with no saved predictions would need its model run on the GPU; that path
does not exist, and such a system is refused rather than scored some other way. No image is opened, and
field_test is never scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprout.data.label_review import apply_exclusions, read_log
from sprout.data.splits import COLUMNS, dataset_hash
from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)

SPLIT = "field_dev"
INTERVALS = ("accuracy", "macro_f1")
COMPARED = ("accuracy", "macro_f1")
DEVICE = "none: saved predictions were filtered and no model was run"
CAVEAT = (
    "Development numbers on field_dev only. The models were chosen or tuned on field_dev before the labels were "
    "cleaned, and the cleaning looked only at each photo, its label and its file name. field_test is not scored."
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["split"] != SPLIT:
        raise ValueError(f"E13 scores {SPLIT} only, not {cfg['split']!r}")
    return cfg


def _record(root: Path, run_id: str) -> dict[str, Any]:
    return json.loads((root / RUNS_REL / f"{run_id}.json").read_text(encoding="utf-8"))


def _saved(root: Path, rec: dict[str, Any], system: str, paths: list[str]) -> list[dict[str, Any]]:
    """The system's saved rows for exactly these photos, in this order."""
    if system not in rec["predictions"]:
        raise ValueError(f"{rec['run_id']} has no saved predictions for {system}; E13 does not run models")
    lines = (root / rec["predictions"][system]).read_text(encoding="utf-8").splitlines()
    by_path = {r["path"]: r for r in map(json.loads, (x for x in lines if x.strip()))}
    missing = [p for p in paths if p not in by_path]
    if missing:
        raise ValueError(f"{rec['run_id']} {system} lacks {len(missing)} of the rows being scored, e.g. {missing[0]}")
    return [by_path[p] for p in paths]


def _scored(rows: list[dict[str, Any]], groups: list[str]) -> tuple[Scored, np.ndarray, np.ndarray]:
    y = np.array([r["label"] for r in rows], dtype=int)
    p = np.array([r["probs"] for r in rows], dtype=float)
    return Scored(groups=groups, labels=y, probs=p), y, p


def _comparisons(scored: dict[str, Scored], reference: str) -> list[dict[str, Any]]:
    out = []
    for other in scored:
        if other == reference:
            continue
        for metric in COMPARED:
            got = recompute_comparison(scored[reference], scored[other], metric)
            out.append({"a": reference, "b": other, "metric": metric, "diff": got["diff"], "ci": got["ci"],
                        "mcnemar_p": got["mcnemar_p"]})
    return out


def verdict(comp: dict[str, Any]) -> str:
    lo, hi = comp["ci"]
    word = "better" if lo > 0 else "worse" if hi < 0 else "no measurable difference"
    return f"{comp['a']} vs {comp['b']} {comp['metric']}: {comp['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}], {word}"


def run_view(cfg: dict[str, Any], view: dict[str, Any], root: Path = ROOT, *, note: str = "") -> Path:
    """Score one view (one old subset, several systems) on the cleaned rows and write its record."""
    version = cfg["exclusion_version"]
    info = split_info(SPLIT, root)
    old_pos = subset_positions(view["subset"], SPLIT, info, root)
    subset = f"{view['subset']}+clean:{version}"
    new_pos = subset_positions(subset, SPLIT, info, root)
    old_paths, new_paths = [info.ids[i] for i in old_pos], [info.ids[i] for i in new_pos]
    log = read_log(root / "datasets" / "label_exclusions.csv")
    gone = log[log["version"] <= version]
    removed = sorted(set(old_paths) - set(new_paths))
    old_groups, new_groups = [info.groups[i] for i in old_pos], [info.groups[i] for i in new_pos]

    records = {s: _record(root, rid) for s, rid in view["systems"].items()}
    names = None
    for system, rec in records.items():
        if rec["split"] != SPLIT or rec["subset"] != view["subset"] or rec["dataset_hash"] != info.hash:
            raise ValueError(f"{rec['run_id']} is not a {SPLIT} {view['subset']} record on the current field.csv")
        names = names or rec["config"]["class_names"]
        if rec["config"]["class_names"] != names:
            raise ValueError("the source records disagree on the class names")

    old_rows = {s: _saved(root, r, s, old_paths) for s, r in records.items()}
    new_rows = {s: _saved(root, r, s, new_paths) for s, r in records.items()}
    old = {s: _scored(rows, old_groups) for s, rows in old_rows.items()}
    new = {s: _scored(rows, new_groups) for s, rows in new_rows.items()}
    reference = cfg["reference"]
    if reference not in new:
        raise ValueError(f"the reference system {reference} is not in view {view['name']}")

    metrics = {s: standard_image_metrics(p, y, names) for s, (_, y, p) in new.items()}
    old_metrics = {s: standard_image_metrics(p, y, names) for s, (_, y, p) in old.items()}
    intervals = {s: {m: recompute_interval(m, sc) for m in INTERVALS} for s, (sc, _, _) in new.items()}
    comparisons = _comparisons({s: v[0] for s, v in new.items()}, reference)
    old_comparisons = _comparisons({s: v[0] for s, v in old.items()}, reference)
    clean = apply_exclusions(pd.read_csv(root / "datasets" / "manifests" / "field.csv"), log)
    config = {
        "class_names": names,
        "view": view["name"],
        "source_records": view["systems"],
        "exclusion_version": version,
        "rows": {"before": len(old_paths), "after": len(new_paths), "removed": len(removed)},
        "removed_by_reason": gone[gone["path"].isin(removed)]["reason_code"].value_counts().to_dict(),
        "old_metrics_same_systems_all_rows": old_metrics,
        "old_comparisons_all_rows": old_comparisons,
        "clean_manifest_hash": dataset_hash(clean[COLUMNS]),
        "device": DEVICE,
        "label": "development number, field_dev only",
    }
    models = {s: f"{s}: saved predictions of {rid}, filtered to cleaned rows" for s, rid in view["systems"].items()}
    attempts = [{"setting": "filter saved predictions", "result": f"{len(removed)} of {len(old_paths)} rows removed",
                 "device": DEVICE}]
    shift = "; ".join(
        f"{s} accuracy {old_metrics[s]['accuracy']:.4f} -> {metrics[s]['accuracy']:.4f}, macro-F1 "
        f"{old_metrics[s]['macro_f1']:.4f} -> {metrics[s]['macro_f1']:.4f}" for s in metrics
    )
    notes = (f"{CAVEAT} Exclusion version {version}: {len(removed)} of {len(old_paths)} {view['subset']} rows removed. "
             f"Old (all rows) -> new (cleaned rows): {shift}. Paired on cleaned rows: "
             + "; ".join(verdict(c) for c in comparisons))
    if note:
        notes += f" {note}"
    return write_run("E13", SPLIT, config, models, metrics, intervals, attempts,
                     {s: rows for s, rows in new_rows.items()}, info.hash, root=root, notes=notes, subset=subset,
                     comparisons=comparisons)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e13.json")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        for view in cfg["views"]:
            print(run_view(cfg, view, ROOT, note=a.note))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
