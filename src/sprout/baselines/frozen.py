"""Frozen image encoders with a trained linear head, and the feature cache they share with the other baselines.

Features are computed once, written in shards under the cache folder, and reused on a rerun. A shard folder's
name carries a fingerprint of the feature settings and the image list, so changed settings never reuse old files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sprout.atomic import atomic_write
from sprout.train.loop import build_model


def freeze(module: nn.Module) -> nn.Module:
    """Evaluation mode, no gradients: nothing in it can be trained or updated by batch statistics."""
    module.eval()
    module.requires_grad_(False)
    return module


class ConvNextFeatures(nn.Module):
    """ConvNeXt up to its pooled, normalised feature vector (768 values for Tiny); no classifier."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.features, self.avgpool, self.norm = model.features, model.avgpool, model.classifier[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.avgpool(self.features(x))).flatten(1)


def load_convnext_encoder(name: str = "convnext_tiny") -> nn.Module:
    """The ImageNet-pretrained backbone, frozen. These are not the E1 weights."""
    if name != "convnext_tiny":
        raise ValueError(f"unknown frozen backbone: {name}")
    return freeze(ConvNextFeatures(build_model(name, 1, pretrained=True)))


class FileDataset(Dataset):
    """One transformed item per image path; `fn` must be a module-level function or a partial of one."""

    def __init__(self, paths: list[str], fn: Callable[[Image.Image], Any]):
        self.paths, self.fn = paths, fn

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.fn(im)


def image_batches(rows: pd.DataFrame, fn: Callable[[Image.Image], Any], batch_size: int,
                  num_workers: int) -> Iterable[torch.Tensor]:
    return DataLoader(FileDataset(list(rows["path"]), fn), batch_size=batch_size, num_workers=num_workers)


def encode_batches(encoder: nn.Module, batches: Iterable[torch.Tensor], device: torch.device) -> Iterable[np.ndarray]:
    with torch.no_grad():
        for x in batches:
            yield encoder(x.to(device)).float().cpu().numpy()


@dataclass
class Features:
    x: np.ndarray
    seconds: list[float] = field(default_factory=list)  # one per shard, whichever session made it
    peak_mem: list[int] = field(default_factory=list)  # peak GPU bytes per shard; empty without a GPU
    fingerprint: str = ""  # of the settings and image list these features came from


def cached_features(cache_dir: Path, name: str, key: str, rows: pd.DataFrame,
                    make_batches: Callable[[pd.DataFrame], Iterable[np.ndarray]], shard_rows: int,
                    log: Callable[[str], None] = lambda _: None) -> Features:
    """Features for `rows` in order. Finished shards are read back; the rest are computed and saved atomically."""
    fingerprint = hashlib.sha256((key + "\n" + "\n".join(rows["path"])).encode()).hexdigest()[:12]
    folder = Path(cache_dir) / f"{name}-{fingerprint}"
    folder.mkdir(parents=True, exist_ok=True)
    parts: list[np.ndarray] = []
    out = Features(np.zeros((0, 0), np.float32), fingerprint=fingerprint)
    starts = range(0, len(rows), shard_rows)
    for k, start in enumerate(starts):
        piece = rows.iloc[start:start + shard_rows]
        shard = folder / f"shard-{k:05d}.npz"
        saved = None
        if shard.is_file():
            with np.load(shard) as z:
                if z["x"].shape[0] == len(piece):
                    saved = {item: z[item] for item in z.files}
        if saved is None:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            x = np.concatenate(list(make_batches(piece))).astype(np.float32)
            seconds = time.perf_counter() - began
            if x.shape[0] != len(piece) or not np.isfinite(x).all():
                raise RuntimeError(f"{name}: features for shard {k} are missing rows or not finite")
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else -1
            saved = {"x": x, "seconds": np.array(seconds), "peak": np.array(peak)}
            tmp = folder / f"shard-{k:05d}.tmp.npz"
            np.savez(tmp, **saved)
            os.replace(tmp, shard)
            log(f"{name} shard {k + 1}/{len(starts)} done seconds={seconds:.1f} peak_gpu_mem={peak}")
        else:
            log(f"{name} shard {k + 1}/{len(starts)} reused from cache")
        parts.append(saved["x"])
        out.seconds.append(float(saved["seconds"]))
        if int(saved["peak"]) >= 0:
            out.peak_mem.append(int(saved["peak"]))
    if parts:
        out.x = np.concatenate(parts)
    return out


@dataclass
class HeadFit:
    model: nn.Module
    best_dev_accuracy: float
    history: list[dict[str, float]]


def _loss_and_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train_head(x: np.ndarray, y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray, num_classes: int,
               settings: dict[str, Any], out_dir: Path, seed: int) -> HeadFit:
    """Linear head on fixed features. Keeps only latest.pt and best.pt (best dev accuracy) in `out_dir`, and
    continues from latest.pt when it matches these settings. A non-finite loss stops the run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest, best = out_dir / "latest.pt", out_dir / "best.pt"
    tag = hashlib.sha256(json.dumps([settings, seed, list(x.shape), list(dev_x.shape)]).encode()).hexdigest()
    tx, ty = torch.from_numpy(x), torch.from_numpy(y).long()
    dx, dy = torch.from_numpy(dev_x), torch.from_numpy(dev_y).long()
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    start, best_acc, history = 0, float("-inf"), []
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["tag"] != tag:
            raise ValueError(f"{latest} was trained with different settings or features; use a new folder")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best_acc, history = state["epoch"], state["best_metric"], state["history"]
    for epoch in range(start, settings["epochs"]):
        began = time.perf_counter()
        model.train()
        order = torch.randperm(len(tx), generator=torch.Generator().manual_seed(seed * 100003 + epoch))
        loss_sum = 0.0
        for i in range(0, len(order), settings["batch_size"]):
            idx = order[i:i + settings["batch_size"]]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(tx[idx]), ty[idx])
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"epoch {epoch + 1}: head train loss is {loss.item()}, not finite; run stopped")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
        val_loss, val_acc = _loss_and_accuracy(model, dx, dy)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"epoch {epoch + 1}: head validation loss is {val_loss}, not finite; run stopped")
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(tx), "val_loss": val_loss,
                        "val_acc": val_acc, "seconds": time.perf_counter() - began})
        print(f"head epoch {epoch + 1}/{settings['epochs']} {history[-1]}", flush=True)
        improved = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "best_metric": best_acc, "history": history, "tag": tag}
        for target in (latest, best) if improved else (latest,):
            atomic_write(target, lambda f: torch.save(state, f))  # noqa: B023 (runs at once)
    model.load_state_dict(torch.load(best, map_location="cpu", weights_only=True)["model"])
    return HeadFit(model.eval(), best_acc, history)


def head_probs(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(x)).float(), dim=1).numpy().astype(np.float64)
