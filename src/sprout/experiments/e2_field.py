"""E2: the E1 model on real photos (field_dev) and on lab_dev, both over the classes the two sets share,
with Grad-CAM and the share of its energy that falls outside the leaf.

Run from the repo root:  python -m sprout.experiments.e2_field [--config configs/e2.json] [--report]

Needs the E1 checkpoint named in the config. Grad-CAM keeps a graph only through the layers after the
target layer, so it needs about the memory of inference at the batch size; the measured peak is saved
in the record (`peak_gpu_mem_bytes`) when a GPU is used. Only field_dev and lab_dev are ever read.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sprout.data.masks import energy_outside, hsv_mask, iou, segmented_mask
from sprout.data.splits import dataset_hash, load_manifest
from sprout.eval.predict import Predictions
from sprout.eval.records import ROOT, is_dirty, write_run
from sprout.experiments import e2_review as review
from sprout.experiments.common import load_e1, score_systems
from sprout.explain.gradcam import grad_cam, overlay, target_layer
from sprout.preprocess import IMAGE_SIZE, preprocess
from sprout.train.loop import build_model

KINDS = {"field_dev": "field", "lab_dev": "lab"}  # the only splits this experiment may read
MASKED_LABEL = "E1 with its outputs for classes outside the shared list masked (secondary number)"


def shared_classes() -> set[str]:
    """Labels present in both lab.csv and field.csv, the same rule check_run uses for `shared_classes`."""
    return set(load_manifest("lab")["label"]) & set(load_manifest("field")["label"])


def load_rows(split: str, shared: set[str]) -> pd.DataFrame:
    """Manifest rows of one allowed split whose class is shared, in manifest order."""
    if split not in KINDS:
        raise ValueError(f"E2 reads only {sorted(KINDS)}; refusing {split!r}")
    df = load_manifest(KINDS[split])
    part = df[df["split"] == split]
    return part[part["label"].isin(shared)].reset_index(drop=True)


def mask_shared_outputs(probs: np.ndarray, class_names: list[str], shared: set[str]) -> np.ndarray:
    keep = np.array([c in shared for c in class_names])
    out = np.where(keep, probs, 0.0)
    return out / out.sum(axis=1, keepdims=True)


class CamDataset(Dataset):
    """Model input, label, the leaf mask used for the energy, and the colour-threshold mask.
    Lab images use the official segmented copy for the leaf mask; real photos use the colour threshold."""

    def __init__(self, rows: pd.DataFrame, class_map: dict[str, int], kind: str):
        self.paths, self.kind = list(rows["path"]), kind
        self.targets = [class_map[x] for x in rows["label"]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            x, rough = preprocess(im), hsv_mask(im, IMAGE_SIZE)
        leaf = segmented_mask(self.paths[i], IMAGE_SIZE) if self.kind == "lab" else rough
        return x, self.targets[i], torch.from_numpy(leaf), torch.from_numpy(rough)


@dataclass
class Scoring:
    preds: Predictions
    energy: list[float | None]
    leaf_fraction: list[float]
    hsv_iou: list[float]
    seconds: list[float] = field(default_factory=list)
    peak_mem: list[int] = field(default_factory=list)


def score_split(model, layer, rows: pd.DataFrame, class_map: dict[str, int], kind: str, batch_size: int,
                num_workers: int) -> Scoring:
    device = next(model.parameters()).device
    loader = DataLoader(CamDataset(rows, class_map, kind), batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    probs, labels = [], []
    out = Scoring(Predictions([], np.zeros(0, int), np.zeros((0, len(class_map)))), [], [], [])
    for x, y, leaf, rough in loader:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        cams, logits = grad_cam(model, x.to(device), layer)
        if device.type == "cuda":
            torch.cuda.synchronize()
            out.peak_mem.append(torch.cuda.max_memory_allocated())
        out.seconds.append(time.perf_counter() - began)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
        for cam, a, b in zip(cams, leaf.numpy(), rough.numpy(), strict=True):
            out.energy.append(energy_outside(cam, a))
            out.leaf_fraction.append(float(a.mean()))
            if kind == "lab":
                out.hsv_iou.append(iou(a, b))
    out.preds = Predictions(list(rows["path"]), np.concatenate(labels).astype(int), np.concatenate(probs))
    return out


def _compose(left: Image.Image, right: Image.Image) -> Image.Image:
    both = Image.new("RGB", (left.width + right.width, left.height))
    both.paste(left, (0, 0))
    both.paste(right, (left.width, 0))
    return both


def write_overlays(model, layer, split: str, rows: pd.DataFrame, scoring: Scoring, class_names: list[str],
                   out_dir: Path, per_cell: int, seed: int) -> list[Path]:
    """A seeded few correct and wrong predictions of one split: photo and Grad-CAM overlay side by side."""
    device = next(model.parameters()).device
    guess = scoring.preds.probs.argmax(axis=1)
    right = guess == scoring.preds.label
    rng, written = random.Random(seed), []
    for name, flag in (("correct", True), ("wrong", False)):
        pool = [i for i in range(len(rows)) if right[i] == flag]
        for rank, i in enumerate(sorted(rng.sample(pool, min(per_cell, len(pool))))):
            with Image.open(rows["path"][i]) as im:
                photo = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                cam, _ = grad_cam(model, preprocess(im)[None].to(device), layer)
            true, pred = class_names[scoring.preds.label[i]], class_names[guess[i]]
            dest = out_dir / KINDS[split].split("_")[0] / name / f"{rank:02d}_{true}__as__{pred}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            _compose(photo, overlay(photo, cam[0])).save(dest, quality=80)
            written.append(dest)
    return written


def _measurements(sc: Scoring, report_energy: bool) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"batch_seconds": sc.seconds}
    if sc.peak_mem:
        out["peak_gpu_mem_bytes"] = [float(v) for v in sc.peak_mem]
    if report_energy:
        out["energy_outside_leaf"] = [e for e in sc.energy if e is not None]
        out["leaf_fraction"] = sc.leaf_fraction
    if sc.hsv_iou:
        out["hsv_vs_segmented_iou"] = sc.hsv_iou
    return out


def _write_sheets(cfg: dict[str, Any], root: Path, field_rows: pd.DataFrame, field_scoring: Scoring,
                  class_names: list[str]) -> None:
    """The two blank hand-work sheets, never overwriting one a person may have started."""
    mistakes, masks = root / cfg["mistakes_sheet"], root / cfg["mask_sheet"]
    if not mistakes.exists():
        sheet = review.select_mistakes(field_scoring.preds, class_names, cfg["mistakes_n"], cfg["seed"])
        mistakes.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(mistakes, index=False)
    if not masks.exists():
        sheet = review.select_mask_sample(list(field_rows["path"]), cfg["mask_sample_n"], cfg["seed"])
        masks.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(masks, index=False)
        review.write_mask_images(sheet, root / cfg["out_dir"] / "field" / "mask-check")


def _energy_note(is_field: bool, report_energy: bool) -> str:
    if not is_field:
        return "Grad-CAM energy outside the leaf uses the official segmented images as masks."
    if report_energy:
        return "Grad-CAM energy outside the leaf is reported: field masks were judged reliable."
    return (
        "Grad-CAM energy outside the leaf is not reported for real photos: their masks are not "
        "judged reliable (see results/e2-mask-check.csv)."
    )


def _split_notes(split: str, sha: str, cfg: dict[str, Any], n_shared: int, n_classes: int, energy_note: str) -> str:
    return (
        f"E1 checkpoint {sha[:12]} from {cfg['e1_run_id']}, scored on {split} over the {n_shared} shared "
        f"classes. The model keeps all {n_classes} outputs, so a prediction outside the shared list is "
        f"wrong; e1_masked is the secondary number with those outputs masked. {energy_note} "
        "field_test and lab_test were not read."
    )


def run(cfg: dict[str, Any], root: Path = ROOT) -> list[Path]:
    """Score E1 on each split in cfg['splits'], write one record per split, then the overlays and sheets."""
    if is_dirty(root):
        raise RuntimeError("working tree has uncommitted changes; commit first")
    started = datetime.now(timezone.utc).isoformat()
    ckpt, _, sha = load_e1(cfg, root, "E2")
    class_map: dict[str, int] = ckpt["class_map"]
    class_names = sorted(class_map, key=class_map.get)
    train_cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(train_cfg.model, len(class_map), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    layer = target_layer(model, train_cfg.model)

    shared = shared_classes()
    done: dict[str, tuple[pd.DataFrame, Scoring]] = {}
    paths = []
    for split in cfg["splits"]:
        rows = load_rows(split, shared)
        if rows.empty:
            raise RuntimeError(f"{split} has no rows in the shared classes")
        sc = score_split(model, layer, rows, class_map, KINDS[split], cfg["batch_size"], cfg["num_workers"])
        sc.preds = Predictions(sc.preds.path, sc.preds.label, sc.preds.probs.astype(np.float64))
        masked = Predictions(sc.preds.path, sc.preds.label, mask_shared_outputs(sc.preds.probs, class_names, shared))
        systems = {"e1": sc.preds, "e1_masked": masked}
        metrics, intervals = score_systems(rows, systems)
        is_field = KINDS[split] == "field"
        report_energy = not is_field or bool(cfg["field_masks_reliable"])
        energy_note = _energy_note(is_field, report_energy)
        config = {
            "class_names": class_names,
            "e2": dict(cfg),
            "checkpoint": cfg["checkpoint"],
            "checkpoint_sha256": sha,
            "e1_run_id": cfg["e1_run_id"],
            "model": train_cfg.model,
            "shared_classes": sorted(shared),
            "outputs_without_shared_class": sorted(set(class_names) - shared),
            "gradcam_target": "the class the model predicts, at the last stage of the backbone",
            "started_at": started,
        }
        attempts = [{"setting": {"checkpoint_sha256": sha, "split": split, "subset": "shared_classes"}}]
        notes = _split_notes(split, sha, cfg, len(shared), len(class_map), energy_note)
        paths.append(write_run(
            cfg["experiment"], split, config,
            {"e1": f"{train_cfg.model} fine-tuned on lab data, checkpoint {sha[:12]}", "e1_masked": MASKED_LABEL},
            metrics, intervals, attempts, systems, dataset_hash(load_manifest(KINDS[split])),
            root=root, notes=notes, subset="shared_classes",
            measurements=_measurements(sc, report_energy),
        ))
        done[split] = (rows, sc)
        if is_field:  # raw numbers for the mask check; the folder is git-ignored
            side = root / cfg["out_dir"] / "field"
            side.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"path": rows["path"], "energy_outside_leaf": sc.energy,
                          "leaf_fraction": sc.leaf_fraction}).to_csv(side / "energy.csv", index=False)
        print(f"{split}: n={len(rows)} accuracy={metrics['e1']['accuracy']:.4f} "
              f"seconds={sum(sc.seconds):.1f} peak_gpu_mem={max(sc.peak_mem, default=None)}", flush=True)

    for split, (rows, sc) in done.items():
        write_overlays(model, layer, split, rows, sc, class_names, root / cfg["out_dir"],
                       cfg["overlays_per_cell"], cfg["seed"])
    if "field_dev" in done:
        _write_sheets(cfg, root, *done["field_dev"], class_names)
    return paths


def report(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    """Tally the filled-in mistake sheet into the error analysis, and say whether the mask check passed."""
    sheet = review.read_sheet(root / cfg["mistakes_sheet"], review.MISTAKE_COLUMNS)
    masks = review.read_sheet(root / cfg["mask_sheet"], review.MASK_COLUMNS)
    verdict = review.mask_verdict(masks, cfg["mask_min_good"])
    state = (
        f"{verdict['good']} of {verdict['judged']} sampled real-photo masks judged good "
        f"(reliable needs all {verdict['sampled']} judged and at least {cfg['mask_min_good']:.0%} good): "
        + ("reliable." if verdict["reliable"] else "NOT shown to be reliable, so the energy metric is lab only.")
    )
    header = (
        f"{len(sheet)} wrong predictions on field_dev, seed {cfg['seed']}, selected by "
        f"`sprout.experiments.e2_review.select_mistakes` from the E2 field_dev record. Categories were assigned by "
        "viewing each photo; they describe what was seen in this sample and are not a proven cause of the errors. "
        "Where a category was a judgment call or the label looks wrong, the note says so; these notes are input for "
        "the H2 label review.\n\n"
        "Scope: lab and real-photo scores are each computed on the classes the two datasets share and are not "
        "merged. field_dev is split from field_train by near-duplicate group, but a few same-scene images could "
        "remain across the field splits. The Grad-CAM overlays kept for field_dev are a different random sample "
        "from these mistakes, so no overlay was used for these categories.\n\n"
        "`blur` here includes very small low-resolution images. `unsupported_condition` here covers fruit photos and a nutrient deficiency, which the "
        "leaf-trained classes do not cover.\n\n"
        f"Mask check: {state}"
    )
    out = root / cfg["mistakes_report"]
    out.write_text(review.mistakes_markdown(sheet, header), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2.json")
    ap.add_argument("--report", action="store_true", help="write the error analysis from the filled-in sheets")
    a = ap.parse_args(argv)
    cfg = json.loads((ROOT / a.config).read_text(encoding="utf-8"))
    try:
        if a.report:
            print(report(cfg, ROOT))
        else:
            for path in run(cfg, ROOT):
                print(path)
    except (RuntimeError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
