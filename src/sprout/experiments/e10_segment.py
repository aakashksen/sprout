"""E10: how well does a small segmentation model, trained on PlantSeg's masks, find the disease area?

Run from the repo root with PYTHONPATH set to its src folder:
    .venv/Scripts/python -m sprout.experiments.e10_segment --stage train [--config configs/e10.json] [--resume]
    .venv/Scripts/python -m sprout.experiments.e10_segment --stage records

Stages, and what each needs:
  train    GPU if present (CPU works but is slow): fine-tunes on seg_train; seg_dev picks the best epoch.
           Keeps `<out_dir>/best.pt` and `latest.pt` only, and flushes seconds and peak GPU memory per epoch to
           `<out_dir>/epochs.log`. A non-finite loss stops the run. `--resume` continues from latest.pt and
           refuses if the settings or the manifest changed; without it, existing checkpoints are an error.
  records  CPU is enough (GPU used if present): best.pt on every seg_dev image at the photo's own size, saved as
           run-length-encoded masks with per-image IoU, then one E10 record on split seg_dev.
The manifest and mask extraction are a separate CPU step: `python -m sprout.segment.data`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from sprout.atomic import atomic_write, atomic_write_text
from sprout.data.splits import dataset_hash
from sprout.eval.records import ROOT, write_run
from sprout.segment.data import disease_mask, mask_iou, rle_encode, rle_iou
from sprout.segment.model import MODEL_SOURCE, build_model, predict_mask, to_tensor

MAX_WORKERS = 8
NOT_SETTINGS = ("num_workers", "out_dir", "min_free_disk_gb")
CAVEAT = (
    "PlantSeg masks mark disease on PlantSeg-style web photos; the model may not carry over to PlantDoc or other "
    "photos. seg_dev both picks the best epoch and scores it, so its IoU is slightly optimistic; there is no "
    "separate seg test split. Masks are PlantSeg's own and were not reviewed."
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["num_workers"] > MAX_WORKERS:
        raise ValueError(f"num_workers is {cfg['num_workers']}; at most {MAX_WORKERS}")
    return cfg


def load_seg_manifest(root: Path, name: str = "seg") -> pd.DataFrame:
    return pd.read_csv(root / "datasets" / "manifests" / f"{name}.csv")


class SegDataset:
    """Image tensor and a long mask tensor, both at size x size. Training adds a random flip and brightness."""

    def __init__(self, rows: pd.DataFrame, root: Path, size: int, train: bool):
        self.images = [root / p for p in rows["path"]]
        self.masks = [root / p for p in rows["mask"]]
        self.size, self.train = size, train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(self.images[i]) as im:
            image = im.convert("RGB")
        mask = Image.fromarray(disease_mask(self.masks[i]).astype(np.uint8)).resize(
            (self.size, self.size), Image.NEAREST
        )
        if self.train and random.random() < 0.5:
            image, mask = image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        x = to_tensor(image, self.size)
        if self.train:
            x = x * random.uniform(0.85, 1.15) + random.uniform(-0.15, 0.15)
        return x, torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())


def _batch_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    inter = (pred & gt).flatten(1).sum(1).float()
    union = (pred | gt).flatten(1).sum(1).float()
    return torch.where(union == 0, torch.ones_like(union), inter / union.clamp(min=1))


def _dev_iou(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))["out"].argmax(1) == 1
            scores.append(_batch_iou(pred, (y == 1).to(device)).cpu())
    return float(torch.cat(scores).mean())


def _settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in NOT_SETTINGS}


def _save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda f: torch.save(state, f))


def _check_can_start(cfg: dict[str, Any], out: Path, latest: Path, best: Path, root: Path, resume: bool) -> None:
    if not resume and (latest.exists() or best.exists()):
        raise FileExistsError(f"{out} has checkpoints; pass --resume or use a new out_dir")
    if resume and not latest.is_file():
        raise FileNotFoundError(latest)
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < cfg["min_free_disk_gb"]:
        raise RuntimeError(f"only {free_gb:.0f} GB free; the config needs {cfg['min_free_disk_gb']} GB")


def _restore(cfg, latest: Path, hist_path: Path, mhash: str, settings, model, optimizer, scheduler, scaler):
    """Load the last checkpoint into the training objects; returns (epochs done, best IoU, history so far)."""
    ckpt = torch.load(latest, map_location="cpu", weights_only=True)
    if ckpt["manifest_hash"] != mhash:
        raise ValueError("checkpoint was trained on a different manifest")
    if ckpt["settings"] != settings:
        raise ValueError("training settings differ from the checkpoint's; resume with the same settings")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start = ckpt["epoch"]
    history = json.loads(hist_path.read_text(encoding="utf-8"))[:start]
    torch.manual_seed(cfg["seed"] + start)
    random.seed(cfg["seed"] + start)
    return start, ckpt["best_iou"], history


def _train_epoch(model, loader, optimizer, scaler, loss_fn, device: torch.device, epoch: int) -> float:
    """One pass over the training loader; returns the mean loss per image."""
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x)["out"], y)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"epoch {epoch + 1}: train loss is {loss.item()}, not finite; run stopped")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / max(n, 1)


def train_seg(cfg: dict[str, Any], manifest: pd.DataFrame, root: Path = ROOT, *, resume: bool = False,
              device: torch.device | None = None) -> dict[str, Any]:
    """Fine-tune on seg_train, pick the best epoch on seg_dev. Returns the history and checkpoint paths."""
    out = root / cfg["out_dir"]
    latest, best, hist_path = out / "latest.pt", out / "best.pt", out / "history.json"
    _check_can_start(cfg, out, latest, best, root, resume)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mhash = dataset_hash(manifest)
    settings = _settings(cfg)

    parts = {s: manifest[manifest["split"] == s] for s in ("seg_train", "seg_dev")}
    loaders = {
        s: DataLoader(SegDataset(parts[s], root, cfg["size"], train=(s == "seg_train")),
                      batch_size=cfg["batch_size"], shuffle=(s == "seg_train"), num_workers=cfg["num_workers"])
        for s in parts
    }
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    model = build_model(cfg["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg["epochs"], 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    start, best_iou, history = 0, float("-inf"), []
    if resume:
        start, best_iou, history = _restore(cfg, latest, hist_path, mhash, settings, model, optimizer, scheduler,
                                            scaler)

    for epoch in range(start, cfg["epochs"]):
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_loss = _train_epoch(model, loaders["seg_train"], optimizer, scaler, loss_fn, device, epoch)
        scheduler.step()
        dev_iou = _dev_iou(model, loaders["seg_dev"], device)
        if not math.isfinite(dev_iou):
            raise RuntimeError(f"epoch {epoch + 1}: seg_dev IoU is {dev_iou}, not finite; run stopped")
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_iou": dev_iou,
            "seconds": time.perf_counter() - began,
            "peak_mem_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "peak_mem_reserved": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        }
        history.append(entry)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "epochs.log", "a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
            log.flush()
        print(f"epoch {epoch + 1}/{cfg['epochs']} {entry}", flush=True)
        improved = dev_iou > best_iou
        if improved:
            best_iou = dev_iou
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_iou": best_iou, "settings": settings,
                 "manifest_hash": mhash}
        _save(latest, state)
        if improved:
            _save(best, state)
        atomic_write_text(hist_path, json.dumps(history, indent=1) + "\n")
    return {"best": best, "latest": latest, "history": history}


def segment_rows(model: nn.Module, rows: pd.DataFrame, root: Path, size: int) -> list[dict[str, Any]]:
    """Prediction rows at each photo's own size: path, per-image IoU, and the predicted and true masks."""
    out = []
    for path, mask_path in zip(rows["path"], rows["mask"], strict=True):
        with Image.open(root / path) as im:
            image = im.convert("RGB")
        gt = disease_mask(root / mask_path)
        if gt.shape != (image.height, image.width):
            raise ValueError(f"{mask_path}: mask size {gt.shape} differs from the photo's {image.size}")
        pred = predict_mask(model, image, size)
        out.append({"path": path, "iou": mask_iou(pred, gt), "pred": rle_encode(pred), "gt": rle_encode(gt)})
    return out


def whole_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline: call every pixel diseased."""
    out = []
    for r in rows:
        full = rle_encode(np.ones(r["gt"]["size"], dtype=bool))
        out.append({"path": r["path"], "iou": rle_iou(full, r["gt"]), "pred": full, "gt": r["gt"]})
    return out


def summary_metrics(ious: np.ndarray) -> dict[str, float]:
    return {"mean_iou": float(ious.mean()), "median_iou": float(np.median(ious)),
            "iou_ge_0.5": float((ious >= 0.5).mean())}


def write_records(cfg: dict[str, Any], manifest: pd.DataFrame, root: Path = ROOT, *,
                  device: torch.device | None = None, allow_dirty: bool = False) -> Path:
    from sprout.eval.check_run import Scored, recompute_interval

    out = root / cfg["out_dir"]
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    mhash = dataset_hash(manifest)
    if ckpt["manifest_hash"] != mhash or ckpt["settings"] != _settings(cfg):
        raise ValueError("best.pt was trained with a different manifest or settings than this config")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    dev = manifest[manifest["split"] == "seg_dev"]
    systems = {"e10": segment_rows(model, dev, root, cfg["size"])}
    systems["whole_image"] = whole_image_rows(systems["e10"])
    metrics, intervals = {}, {}
    for name, rows in systems.items():
        ious = np.array([r["iou"] for r in rows])
        metrics[name] = summary_metrics(ious)
        s = Scored(groups=list(dev["group"]), iou=ious)
        intervals[name] = {k: list(recompute_interval(k, s)) for k in ("mean_iou", "iou_ge_0.5")}
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    attempts = [{"setting": _settings(cfg), "best_epoch": max(history, key=lambda h: h["dev_iou"])["epoch"],
                 "history": history}]
    config = {**cfg, "model_source": MODEL_SOURCE, "labels_reviewed": False}
    return write_run(
        "E10", "seg_dev", config,
        {"e10": f"{MODEL_SOURCE['name']} fine-tuned on seg_train", "whole_image": "baseline: every pixel diseased"},
        metrics, intervals, attempts, systems, mhash, root=root, allow_dirty=allow_dirty, notes=CAVEAT,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e10.json")
    ap.add_argument("--stage", choices=["train", "records"], required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(ROOT / a.config)
    manifest = load_seg_manifest(ROOT, cfg["manifest"])
    if a.stage == "train":
        train_seg(cfg, manifest, ROOT, resume=a.resume)
    else:
        print(write_records(cfg, manifest, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
