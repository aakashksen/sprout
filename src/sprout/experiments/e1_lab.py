"""E1: ConvNeXt-Tiny fine-tuned on the lab data, scored on lab_dev next to the legacy checkpoint.

Run from the repo root:  python -m sprout.experiments.e1_lab [--config configs/e1.json] [--resume PATH] [--note TEXT]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sprout.data.splits import dataset_hash, load_manifest
from sprout.eval.predict import Predictions, predict
from sprout.eval.records import ROOT, is_dirty, write_run
from sprout.experiments.common import score_systems, sha256_file
from sprout.train.checkpoint import load_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import make_class_map
from sprout.train.loop import build_model, train

BACKGROUND = "Background_without_leaves"
LEGACY_LABEL = "legacy, may have seen these images; 39-class checkpoint mapped onto 38 shared classes"


def legacy_class_names(application: Path) -> list[str]:
    """The `self.class_names = [...]` list in the old app, read from its source without importing it."""
    for node in ast.walk(ast.parse(Path(application).read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "class_names" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise ValueError(f"no class_names list found in {application}")


def _key(name: str) -> str:
    """Official and legacy names differ only in a parenthetical on the plant and a trailing underscore."""
    plant, sep, disease = name.partition("___")
    return re.sub(r"_\(.*?\)", "", plant) + sep + disease.rstrip("_")


def map_legacy(legacy: list[str], official: list[str]) -> tuple[dict[str, str], list[str]]:
    """Legacy name -> official name, and the legacy names with no official class (dropped)."""
    by_key: dict[str, str] = {}
    for name in official:
        if _key(name) in by_key:
            raise ValueError(f"two official classes share the key {_key(name)}")
        by_key[_key(name)] = name
    mapping = {n: by_key[_key(n)] for n in legacy if _key(n) in by_key}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("two legacy classes map onto one official class")
    missing = sorted(set(official) - set(mapping.values()))
    if missing:
        raise ValueError(f"official classes with no legacy output: {missing}")
    return mapping, [n for n in legacy if n not in mapping]


def legacy_to_shared(
    preds: Predictions, legacy: list[str], mapping: dict[str, str], official: list[str]
) -> tuple[Predictions, float]:
    """Keep the mapped outputs in official order and renormalise the rows over them.
    Also returns the share of images whose top legacy output was an unmapped one (the background)."""
    index = {mapping[n]: i for i, n in enumerate(legacy) if n in mapping}
    cols = [index[c] for c in official]
    top_unmapped = float(np.mean([legacy[i] not in mapping for i in preds.probs.argmax(axis=1)]))
    kept = preds.probs[:, cols]
    return Predictions(preds.path, preds.label, kept / kept.sum(axis=1, keepdims=True)), top_unmapped


def run(cfg: dict[str, Any], root: Path = ROOT, resume: Path | None = None, note: str = "") -> Path:
    """Train (or resume), score e1 and the legacy checkpoint on cfg['split'], write the record, return its path."""
    if is_dirty(root):
        raise RuntimeError("working tree has uncommitted changes; commit first")
    started = datetime.now(timezone.utc).isoformat()
    split, leg = cfg["split"], cfg["legacy"]
    train_cfg = TrainConfig(manifest=cfg["manifest"], out_dir=str(root / cfg["out_dir"]), **cfg["train"])
    out = train(train_cfg, resume=resume)

    manifest = load_manifest(cfg["manifest"])
    part = manifest[manifest["split"] == split]
    ckpt = load_checkpoint(out.best_path)
    class_map: dict[str, int] = ckpt["class_map"]
    assert class_map == make_class_map(manifest)
    official = sorted(class_map, key=class_map.get)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(train_cfg.model, len(class_map), pretrained=False)
    model.load_state_dict(ckpt["model"])
    e1 = predict(model.to(device), part, class_map, train_cfg.batch_size)

    names39 = legacy_class_names(root / leg["application"])
    mapping, dropped = map_legacy(names39, official)
    leg_path = Path(leg["checkpoint"])
    old = build_model(leg["model"], len(names39), pretrained=False)
    old.load_state_dict(torch.load(leg_path, map_location="cpu", weights_only=True))
    raw = predict(old.to(device), part, class_map, train_cfg.batch_size)
    legacy, top_dropped = legacy_to_shared(raw, names39, mapping, official)

    # float32 outputs score differently from their saved float64 copies, so score what is saved
    e1, legacy = (Predictions(p.path, p.label, p.probs.astype(np.float64)) for p in (e1, legacy))
    metrics, intervals = score_systems(part, {"e1": e1, "legacy": legacy})

    legacy_acc = metrics["legacy"]["accuracy"]
    order_check = {
        "list_is_sorted": names39 == sorted(names39),
        "accuracy": legacy_acc,
        "chance": 1 / len(official),
        "threshold": leg["sensible_accuracy"],
        "established": legacy_acc >= leg["sensible_accuracy"],
    }
    legacy_info = {
        "label": LEGACY_LABEL,
        "checkpoint": leg_path.name,
        "checkpoint_sha256": sha256_file(leg_path),
        "class_order_source": leg["application"],
        "class_order": names39,
        "mapping": mapping,
        "unmapped_dropped": dropped,
        "background_handling": f"{BACKGROUND} output dropped; probabilities renormalised over the mapped outputs",
        "share_top_output_unmapped": top_dropped,
        "order_check": order_check,
    }
    config = {
        "class_names": official,
        "train": train_cfg.to_dict() | {"manifest_hash": dataset_hash(manifest)},
        "legacy": legacy_info,
        "started_at": started,
    }
    attempts = [
        {
            "setting": dict(cfg["train"]),
            "best_dev_accuracy": out.best_metric,
            "epochs_done": out.epochs_done,
            "history": out.history,
        }
    ]
    state = "confirmed" if order_check["established"] else "NOT established"
    notes = f"{LEGACY_LABEL}. Its class order was {state} by its score on {split}. The lab_test split was not read."
    if note:
        notes = f"{notes} {note}"
    path = write_run(
        cfg["experiment"],
        split,
        config,
        {"e1": f"{train_cfg.model} fine-tuned from ImageNet weights", "legacy": LEGACY_LABEL},
        metrics,
        intervals,
        attempts,
        {"e1": e1, "legacy": legacy},
        dataset_hash(manifest),
        root=root,
        notes=notes,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e1.json")
    ap.add_argument("--resume", type=Path, help="checkpoint to continue from (latest.pt of an interrupted run)")
    ap.add_argument("--note", default="", help="text appended to the record's notes")
    a = ap.parse_args(argv)
    cfg = json.loads((ROOT / a.config).read_text(encoding="utf-8"))
    try:
        path = run(cfg, ROOT, a.resume, a.note)
    except RuntimeError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
