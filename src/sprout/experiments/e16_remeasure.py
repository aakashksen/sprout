"""T8.2's two development remeasures of the final model, on the CPU, from saved predictions. No model is trained,
no test split is read, and no Claude, Ollama or GPU call is made.

E16 `gate`: how often the photo gate refuses real plant photos. The settings are T4.9's refit (results/e15/refit/
chosen.json) applied unchanged to the final model's saved field_dev probabilities and the saved green ratios: no
tuning. The rate is stated on the whole cleaned field_dev, on the part the refit fitted the gate on, and on the part
it did not.

E17 `neighbours` then `records`: the effect of same-scene field_train look-alikes. Every scored field_dev photo gets
its highest DINOv2 ViT-S/14 class-token cosine to a field_train photo of the 29 classes (the higher of the photo and
its mirror, as sprout.data.near_extras does). Photos at or above the threshold are listed in results/t82/
lookalikes.json, committed before the records are written, because the records' subsets `listed:` and `unlisted:` read
that file at the record's commit. `records` also scores PlantDoc and PlantSeg separately (secondary numbers).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sprout.atomic import atomic_write_text
from sprout.eval.check_run import Scored, recompute_interval, recompute_metric
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    scoring_part,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)
from sprout.serve.gate import GateSettings

SPLIT = "field_dev"
SYSTEM = "convnextv2_final"
NEIGHBOURS_REL = Path("results/t82/lookalikes.json")
THRESHOLD = 0.90
STRICT_THRESHOLD = 0.95
GATE_LABEL = "development number, field_dev only"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def final_rows(root: Path, e15_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], Any]:
    """The E15 record, the final model's saved rows, the class names and the field_dev split info. The rows must be
    exactly the record's declared subset, in order."""
    rec = json.loads((root / RUNS_REL / f"{e15_id}.json").read_text(encoding="utf-8"))
    info = split_info(SPLIT, root)
    if rec["experiment"] != "E15" or rec["split"] != SPLIT or rec["dataset_hash"] != info.hash:
        raise ValueError(f"{e15_id} is not an E15 field_dev record on the current manifest")
    rows = read_jsonl(root / rec["predictions"][SYSTEM])
    pos = subset_positions(rec["subset"], SPLIT, info, root)
    if [r["path"] for r in rows] != [info.ids[i] for i in pos]:
        raise ValueError("the saved rows are not the record's subset in order")
    return rec, rows, rec["config"]["class_names"], info


# ---- E16: the gate ----

def rejection(accepted: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    refused = int((~accepted & mask).sum())
    return {"n": n, "refused": refused, "rate": refused / n if n else None}


def gate_record(root: Path, e15_id: str, note: str = "") -> Path:
    from sprout.experiments import e5_confidence as e5
    from sprout.serve.gate import KINDS

    rec, rows, names, info = final_rows(root, e15_id)
    chosen = json.loads((root / "results/e15/refit/chosen.json").read_text(encoding="utf-8"))
    if chosen["e15_record"] != e15_id or chosen["classes"] != names:
        raise ValueError("results/e15/refit/chosen.json was not fitted for this E15 record")
    pos = subset_positions(rec["subset"], SPLIT, info, root)
    paths, groups = [info.ids[i] for i in pos], [info.groups[i] for i in pos]
    green_by = {r["path"]: r["green_ratio"] for r in read_jsonl(root / "results/e5/green_ratio.jsonl")}
    green = np.array([green_by[p] for p in paths], dtype=float)
    labels = np.array([r["label"] for r in rows], dtype=int)
    cal = e5.calibrate(np.array([r["probs"] for r in rows], dtype=float), chosen["temperature"])
    scoring = scoring_part(groups, chosen["seed"], chosen["fit_percent"])
    base = standard_image_metrics(cal, labels, names)

    metrics: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for kind in KINDS:
        s = GateSettings.from_dict(chosen["gates"][kind]["settings"])
        gate = e5.gate_rows(s, cal, green)
        accepted = np.array([g["accepted"] for g in gate])
        sc = Scored(groups=groups, labels=labels, probs=cal, accepted=accepted, is_plant=np.ones(len(rows), dtype=bool))
        metrics[kind] = {**base, "plant_rejection_rate": recompute_metric("plant_rejection_rate", sc),
                         "end_to_end_accuracy": recompute_metric("end_to_end_accuracy", sc)}
        intervals[kind] = {"end_to_end_accuracy": list(recompute_interval("end_to_end_accuracy", sc))}
        predictions[kind] = [{"path": p, "label": int(y), "probs": [float(v) for v in pr], **g}
                             for p, y, pr, g in zip(paths, labels, cal, gate, strict=True)]
        attempts.append({"gate": kind, "settings": chosen["gates"][kind]["settings"],
                         "fit_plant_rejection_in_the_refit": chosen["gates"][kind]["fit_plant_rejection"]})
        summary[kind] = {"all_rows": rejection(accepted, np.ones(len(rows), dtype=bool)),
                         "fitted_on": rejection(accepted, ~scoring), "not_fitted_on": rejection(accepted, scoring)}
    picked = chosen["gate"]["chosen"]
    accepted = np.array([g["accepted"] for g in predictions[picked]])
    per_class: dict[str, dict[str, int]] = {}
    for y, ok in zip(labels, accepted, strict=True):
        c = per_class.setdefault(names[y], {"photos": 0, "refused": 0})
        c["photos"] += 1
        c["refused"] += int(not ok)
    refused_paths = [p for p, ok in zip(paths, accepted, strict=True) if not ok]
    config = {
        "class_names": names, "stage": "gate", "model": SYSTEM, "e15_record": e15_id,
        "settings_file": "results/e15/refit/chosen.json", "temperature": chosen["temperature"],
        "chosen_gate": picked, "gate_settings": chosen["gates"][picked]["settings"],
        "green_file": "results/e5/green_ratio.jsonl",
        "rows": summary, "refused_by_class": per_class, "refused_paths": refused_paths,
        "parts": ("fitted_on: the refit's fitting half (seed %d, %d%% of the groups), where the gate's settings were chosen; "
                  "not_fitted_on: the other half" % (chosen["seed"], chosen["fit_percent"])),
        "no_tuning": "T4.9's settings applied unchanged; nothing was fitted or chosen here",
        "label": GATE_LABEL, "labels_reviewed": False,
    }
    shown = summary[picked]
    text = (f"The photo gate T4.9 chose ({picked}) applied unchanged to the final model's saved field_dev probabilities "
            f"on the {len(rows)} cleaned photos, all of which are plant photos: it refused {shown['all_rows']['refused']} "
            f"({shown['all_rows']['rate']:.4f}); {shown['not_fitted_on']['refused']} of {shown['not_fitted_on']['n']} "
            f"({shown['not_fitted_on']['rate']:.4f}) on the half its settings were not fitted on. Development numbers; "
            "the model and the gate were chosen on field_dev. A refused real plant photo is a false rejection.")
    models = {k: f"{k}: {SYSTEM} from E15 record {e15_id}, calibrated, gate settings from the T4.9 refit" for k in KINDS}
    return write_run("E16", SPLIT, config, models, metrics, intervals, attempts, predictions, info.hash, root=root,
                     notes=(text + " " + note).strip(), subset=rec["subset"])


# ---- E17: look-alikes ----

def best_neighbours(dev: np.ndarray, dev_mirror: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(highest cosine to a training photo, index of that photo) per development photo, taking the higher of the
    photo and its mirror."""
    both = np.maximum(dev @ train.T, dev_mirror @ train.T)
    return both.max(axis=1), both.argmax(axis=1)


def neighbours_file(paths: list[str], cos: np.ndarray, nearest: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    close = [p for p, c in zip(paths, cos, strict=True) if c >= THRESHOLD]
    return {
        **extra, "threshold": THRESHOLD, "strict_threshold": STRICT_THRESHOLD,
        "measure": "DINOv2 ViT-S/14 class token, unit length, 224x224 bicubic resize of the whole photo, the higher of "
                   "the photo and its mirror, against every field_train photo of the 29 classes",
        "rows": len(paths), "at_or_above_threshold": len(close),
        "at_or_above_strict_threshold": int((cos >= STRICT_THRESHOLD).sum()),
        "paths": close,
        "closest": [{"path": p, "cosine": round(float(c), 6), "nearest_train": n}
                    for p, c, n in sorted(zip(paths, cos, nearest, strict=True), key=lambda t: -t[1])[:len(close) + 40]],
    }


def neighbours(root: Path, e15_id: str, workers: int = 4) -> Path:
    import pandas as pd
    import torch

    from sprout.baselines.dino import load_dino_encoder
    from sprout.data import near_extras as ne

    rec, rows, names, info = final_rows(root, e15_id)
    paths = [r["path"] for r in rows]
    field = pd.read_csv(root / "datasets" / "manifests" / "field.csv")
    train = field[(field["split"] == "field_train") & field["label"].isin(names)]
    train_paths = list(train["path"])
    model, device = load_dino_encoder(), torch.device("cpu")
    dev = ne.embed_paths(paths, model, device, root, workers=workers)
    dev_mirror = ne.embed_paths(paths, model, device, root, mirror=True, workers=workers)
    tr = ne.embed_paths(train_paths, model, device, root, workers=workers)
    cos, idx = best_neighbours(dev, dev_mirror, tr)
    body = neighbours_file(paths, cos, [train_paths[i] for i in idx],
                           {"e15_record": e15_id, "train_rows": len(train_paths), "device": "cpu"})
    out = root / NEIGHBOURS_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(body, indent=1) + "\n")
    return out


def subset_record(root: Path, e15_id: str, subset: str, stage: str, config: dict[str, Any], note: str) -> Path:
    rec, _, names, info = final_rows(root, e15_id)
    saved = {r["path"]: r for r in read_jsonl(root / rec["predictions"][SYSTEM])}
    pos = subset_positions(subset, SPLIT, info, root)
    paths, groups = [info.ids[i] for i in pos], [info.groups[i] for i in pos]
    rows = [saved[p] for p in paths]
    labels = np.array([r["label"] for r in rows], dtype=int)
    probs = np.array([r["probs"] for r in rows], dtype=float)
    sc = Scored(groups=groups, labels=labels, probs=probs)
    metrics = {SYSTEM: standard_image_metrics(probs, labels, names)}
    intervals = {SYSTEM: {m: list(recompute_interval(m, sc)) for m in ("accuracy", "macro_f1")}}
    cfg = {"class_names": names, "stage": stage, "model": SYSTEM, "e15_record": e15_id, "label": GATE_LABEL,
           "labels_reviewed": False, **config}
    return write_run("E17", SPLIT, cfg, {SYSTEM: f"{SYSTEM} from E15 record {e15_id}: saved rows on this subset"},
                     metrics, intervals, [{"stage": stage, "subset": subset, "rows": len(paths)}],
                     {SYSTEM: rows}, info.hash, root=root, notes=note, subset=subset)


def lookalike_records(root: Path, e15_id: str) -> list[Path]:
    body = json.loads((root / NEIGHBOURS_REL).read_text(encoding="utf-8"))
    if body["e15_record"] != e15_id:
        raise ValueError("results/t82/lookalikes.json was made for another E15 record")
    rec, rows, names, info = final_rows(root, e15_id)
    base = rec["subset"]
    cos = {c["path"]: c["cosine"] for c in body["closest"]}
    strict = [r for r in rows if cos.get(r["path"], 0.0) >= body["strict_threshold"]]
    accuracy_strict = float(np.mean([int(np.argmax(r["probs"]) == r["label"]) for r in strict])) if strict else None
    common = {"neighbours_file": NEIGHBOURS_REL.as_posix(), "threshold": body["threshold"],
              "at_or_above_threshold": body["at_or_above_threshold"],
              "at_or_above_strict_threshold": body["at_or_above_strict_threshold"],
              "accuracy_at_or_above_strict_threshold": accuracy_strict,
              "train_rows_compared": body["train_rows"]}
    caveat = ("Development numbers on field_dev only. A look-alike here is a field_dev photo whose highest DINOv2 class-token "
              f"cosine to a field_train photo of the 29 classes is at least {body['threshold']}; the closest are mostly the same "
              "photo or scene, the rest are different photos of similar subjects. Extras (PlantWild, Plant Pathology) were not "
              "searched. Accuracy is higher on photos with a close neighbour because they are also ordinarily easier, so this is "
              "not a pure leak measure.")
    out = []
    for word in ("listed", "unlisted"):
        subset = f"{base}+{word}:{NEIGHBOURS_REL.as_posix()}"
        what = "with a close field_train neighbour" if word == "listed" else "with no close field_train neighbour"
        out.append(subset_record(root, e15_id, subset, f"lookalike_{word}", common, f"Final model on the field_dev photos {what}. {caveat}"))
    for source in ("plantdoc", "plantseg"):
        subset = f"source:{source}+{base}"
        out.append(subset_record(root, e15_id, subset, f"source_{source}", {},
                                 f"Final model on the {source} photos of field_dev (a secondary number; the headline is all of "
                                 "the split, in the E15 record). Development numbers. No held-out-source run exists for this "
                                 "model: the held-out-source numbers are E4's and E12's."))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gate", "neighbours", "records"))
    ap.add_argument("--e15-record", default="E15-20261104T220052184613Z")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        if a.stage == "gate":
            print(gate_record(ROOT, a.e15_record, a.note))
        elif a.stage == "neighbours":
            print(neighbours(ROOT, a.e15_record, a.workers))
        else:
            for p in lookalike_records(ROOT, a.e15_record):
                print(p)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
