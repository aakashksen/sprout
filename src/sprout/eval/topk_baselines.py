"""Top-k and plant-given scores for the E1, E2 and E3 records, recomputed from their saved predictions.

    python -m sprout.eval.topk_baselines

Writes results/topk-baselines.json and never edits a record. Run it again after a new E1, E2 or E3 record lands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sprout.eval import metrics as M
from sprout.eval.check_run import _read_rows, records_for
from sprout.eval.records import ROOT, split_info, subset_positions

EXPERIMENTS = ("E1", "E2", "E3")
OUT_REL = Path("results/topk-baselines.json")
NOTE = (
    "Every output is kept: top-k counts a class outside the shared list as wrong. Plant-given restricts each "
    "image's outputs to the classes of its true plant, so it is easy: plant-given top-3 is 1.0 by construction "
    "for any plant with three or fewer classes (see plants_scoring_one_at_top_3 and share_of_images_trivial_at_top_3)."
)


def score_system(rows: list[dict], class_names: list[str]) -> dict:
    probs = np.array([r["probs"] for r in rows], dtype=float)
    labels = np.array([r["label"] for r in rows], dtype=int)
    options = M.options_per_plant(class_names)
    plant = [M.plant_of(class_names[y]) for y in labels]
    trivial = sorted(p for p, n in options.items() if n <= 3)
    return {
        "n": len(rows),
        "top_1_accuracy": M.top_k_accuracy(probs, labels, 1),
        "top_3_accuracy": M.top_k_accuracy(probs, labels, 3),
        "top_5_accuracy": M.top_k_accuracy(probs, labels, 5),
        "plant_given_top_1": M.plant_given_top_k(probs, labels, class_names, 1),
        "plant_given_top_3": M.plant_given_top_k(probs, labels, class_names, 3),
        "share_of_images_trivial_at_top_3": float(np.mean([p in trivial for p in plant])),
        "plants_scoring_one_at_top_3": trivial,
    }


def build(root: Path = ROOT) -> dict:
    out: dict = {"note": NOTE, "records": []}
    for exp in EXPERIMENTS:
        for path in records_for(exp, root):
            rec = json.loads(path.read_text(encoding="utf-8"))
            names = rec["config"]["class_names"]
            info = split_info(rec["split"], root)
            keep = set(info.ids[i] for i in (
                subset_positions(rec["subset"], rec["split"], info, root) if rec["subset"] else range(len(info.ids))))
            systems = {}
            for system, rel in rec["predictions"].items():
                rows = [r for r in _read_rows(root / rel) if "probs" in r and r["path"] in keep]
                if not rows:
                    continue
                scores = score_system(rows, names)
                saved = rec["metrics"].get(system, {}).get("accuracy")
                if saved is not None and abs(saved - scores["top_1_accuracy"]) > 1e-9:
                    raise RuntimeError(f"{rec['run_id']} {system}: top-1 differs from the record's saved accuracy")
                systems[system] = scores
            out["records"].append({
                "run_id": rec["run_id"],
                "experiment": exp,
                "split": rec["split"],
                "subset": rec["subset"],
                "predictions": {s: rec["predictions"][s] for s in systems},
                "options_per_plant": M.options_per_plant(names),
                "systems": systems,
            })
    return out


def main() -> int:
    data = build()
    path = ROOT / OUT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_REL.as_posix()} for {[r['run_id'] for r in data['records']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
