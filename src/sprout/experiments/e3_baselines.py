"""E3: three simpler systems next to the E1 model, on field_dev and lab_dev over the classes the two sets share.

Run from the repo root with PYTHONPATH set to its src folder:
    .venv/Scripts/python -m sprout.experiments.e3_baselines [--config configs/e3.json]

Stages, in order, and what each needs from the machine (seconds and peak GPU memory are logged for each to
`<cache_dir>/stages.log` and the console, flushed as they happen):
  1. classic_features   CPU workers: HOG and colour histograms of lab_train, lab_dev, field_dev (cached shards)
  2. classic_fit        CPU: XGBoost (tree_method hist, device cpu; no GPU is used for it)
  3. frozen_features    GPU if present: ImageNet ConvNeXt-Tiny, frozen, features of the same three sets (cached)
  4. frozen_head        CPU: linear head on those features (latest.pt and best.pt only)
  5. dino_features      GPU if present: DINOv2 small, frozen, class-token features (cached); first use
                        downloads the pinned hub code and an 88 MB weight file into the torch hub cache
  6. dino_head          CPU: linear head on those features
  7. e1_predict         GPU if present: the E1 checkpoint on lab_dev and field_dev
  8. records            one run record per split
Everything trains on lab_train; lab_dev picks each head's best epoch. lab_test, field_test and field_train
are never read. A rerun reuses finished feature shards, a finished XGBoost model and a finished head.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader

from sprout.baselines import dino
from sprout.baselines.classic import classic_features, fit_xgboost, xgboost_probs
from sprout.baselines.dino import load_dino_encoder
from sprout.baselines.frozen import (
    Features,
    cached_features,
    encode_batches,
    head_probs,
    image_batches,
    load_convnext_encoder,
    train_head,
)
from sprout.data.splits import dataset_hash, load_manifest
from sprout.eval.check_run import Scored, recompute_comparison
from sprout.eval.predict import Predictions
from sprout.eval.records import ROOT, is_dirty, write_run
from sprout.experiments.common import load_e1, score_systems
from sprout.experiments.e2_field import KINDS, load_rows, shared_classes
from sprout.preprocess import preprocess
from sprout.train.dataset import ManifestDataset, make_class_map, val_transform
from sprout.train.loop import build_model

SYSTEMS = ("e1", "dinov2_small", "frozen_convnext", "xgboost")
COMPARED = ("accuracy", "macro_f1")
# lab_train is read to train and lab_dev to pick epochs; nothing else is ever opened.
READABLE = {"lab_train": "lab", "lab_dev": "lab", "field_dev": "field"}
GIB = 1024**3


class Log:
    """Lines to the console and to a file, flushed so a person can read the file while a stage runs."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


@contextmanager
def stage(log: Log, name: str):
    log(f"stage {name} start")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    yield
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    log(f"stage {name} done seconds={time.perf_counter() - began:.1f} peak_gpu_mem_bytes={peak}")


def load_split(split: str) -> pd.DataFrame:
    if split not in READABLE:
        raise ValueError(f"E3 reads only {sorted(READABLE)}; refusing {split!r}")
    df = load_manifest(READABLE[split])
    return df[df["split"] == split].reset_index(drop=True)


def _digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def tensor_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


def e1_probs(ckpt: dict, rows: pd.DataFrame, batch_size: int, num_workers: int, device: torch.device) -> np.ndarray:
    model = build_model(ckpt["config"].model, len(ckpt["class_map"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    loader = DataLoader(ManifestDataset(rows, ckpt["class_map"], val_transform), batch_size=batch_size,
                        num_workers=num_workers)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


class Result:
    """What one system leaves behind: probabilities per evaluated split and the facts about training it."""

    def __init__(self, setting: dict[str, Any]):
        self.setting = setting
        self.probs: dict[str, np.ndarray] = {}
        self.train_seconds: float = 0.0
        self.fit_seconds: list[float] = []
        self.extract_seconds: list[float] = []
        self.peak_mem: list[int] = []
        self.size_bytes: int = 0
        self.best_dev_accuracy: float | None = None
        self.history: Any = None

    def attempt(self, name: str) -> dict[str, Any]:
        return {"system": name, "setting": self.setting, "best_dev_accuracy": self.best_dev_accuracy,
                "history": self.history}


def _eval_probs(all_probs: dict[str, np.ndarray], dev_mask: np.ndarray, splits: list[str]) -> dict[str, np.ndarray]:
    """Rows of each evaluated split that fall in the shared classes (lab_dev is scored whole for training)."""
    return {s: (all_probs[s][dev_mask] if s == "lab_dev" else all_probs[s]) for s in splits}


def extract(name: str, key: str, make_batches, needs: dict[str, pd.DataFrame], cfg: dict[str, Any],
            cache: Path, log: Log) -> dict[str, Features]:
    return {
        split: cached_features(cache / "features", f"{name}-{split}", key, rows, make_batches,
                               cfg["shard_rows"], log)
        for split, rows in needs.items()
    }


def run_classic(cfg, needs, y_train, y_dev, num_classes, dev_mask, splits, cache: Path, log: Log) -> Result:
    spec = cfg["classic"]
    result = Result({"features": spec["features"], "xgboost": spec["xgboost"], "seed": cfg["seed"]})
    fn = partial(classic_features, **spec["features"])

    def batches(piece):
        return (x.numpy() for x in image_batches(piece, fn, cfg["batch_size"], cfg["num_workers"]))

    with stage(log, "classic_features"):
        feats = extract("classic", json.dumps(spec["features"], sort_keys=True), batches, needs, cfg, cache, log)
    folder = cache / "xgboost"
    folder.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = folder / "model.ubj", folder / "meta.json"
    tag = _digest(spec, cfg["seed"], feats["lab_train"].fingerprint, feats["lab_dev"].fingerprint)
    booster = xgb.Booster()
    if meta_path.is_file() and model_path.is_file() and json.loads(meta_path.read_text())["tag"] == tag:
        meta = json.loads(meta_path.read_text())
        booster.load_model(model_path)
        log("stage classic_fit reused the finished XGBoost model")
    else:
        with stage(log, "classic_fit"):
            booster, history, seconds = fit_xgboost(feats["lab_train"].x, y_train, feats["lab_dev"].x, y_dev,
                                                    num_classes, spec["xgboost"], cfg["seed"])
        booster.save_model(model_path)
        meta = {"tag": tag, "history": history, "fit_seconds": seconds}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    probs = {s: xgboost_probs(booster, f.x) for s, f in feats.items() if s != "lab_train"}
    result.probs = _eval_probs(probs, dev_mask, splits)
    result.best_dev_accuracy = float((probs["lab_dev"].argmax(1) == y_dev).mean())
    result.history = meta["history"]
    result.fit_seconds = [meta["fit_seconds"]]
    result.extract_seconds = [x for f in feats.values() for x in f.seconds]
    result.train_seconds = sum(feats["lab_train"].seconds) + meta["fit_seconds"]
    result.size_bytes = model_path.stat().st_size
    return result


def run_encoder(name: str, spec: dict[str, Any], encoder, needs, y_train, y_dev, num_classes, dev_mask, splits,
                cfg, cache: Path, log: Log) -> Result:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    result = Result({"head": spec["head"], "seed": cfg["seed"], **{k: v for k, v in spec.items() if k != "head"}})

    def batches(piece):
        return encode_batches(encoder, image_batches(piece, preprocess, cfg["batch_size"], cfg["num_workers"]),
                              device)

    with stage(log, f"{name}_features"):
        feats = extract(name, f"{name}:{json.dumps(spec, sort_keys=True)}", batches, needs, cfg, cache, log)
    settings = {**spec["head"], "features": feats["lab_train"].fingerprint + feats["lab_dev"].fingerprint}
    with stage(log, f"{name}_head"):
        fit = train_head(feats["lab_train"].x, y_train, feats["lab_dev"].x, y_dev, num_classes, settings,
                         cache / f"{name}-head", cfg["seed"])
    probs = {s: head_probs(fit.model, f.x) for s, f in feats.items() if s != "lab_train"}
    result.probs = _eval_probs(probs, dev_mask, splits)
    result.best_dev_accuracy = fit.best_dev_accuracy
    result.history = fit.history
    result.fit_seconds = [h["seconds"] for h in fit.history]
    result.extract_seconds = [x for f in feats.values() for x in f.seconds]
    result.peak_mem = [m for f in feats.values() for m in f.peak_mem]
    result.train_seconds = sum(feats["lab_train"].seconds) + sum(result.fit_seconds)
    result.size_bytes = tensor_bytes(encoder.state_dict()) + tensor_bytes(fit.model.state_dict())
    return result


def run_e1(cfg, ckpt, sha: str, e1_seconds: list[float], eval_rows, splits, log: Log) -> Result:
    e1 = Result({"checkpoint_sha256": sha, "trained_by": cfg["e1_run_id"]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with stage(log, "e1_predict"):
        e1.probs = {s: e1_probs(ckpt, eval_rows[s], cfg["batch_size"], cfg["num_workers"], device) for s in splits}
    if torch.cuda.is_available():
        e1.peak_mem = [torch.cuda.max_memory_allocated()]
    e1.best_dev_accuracy = ckpt["best_metric"]
    e1.history = f"see {cfg['e1_run_id']}"
    e1.fit_seconds = e1_seconds
    e1.train_seconds = sum(e1_seconds)
    e1.size_bytes = tensor_bytes(ckpt["model"])
    return e1


def _measurements(results: dict[str, Result]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, r in results.items():
        out[f"train_seconds_{name}"] = [r.train_seconds]
        out[f"fit_seconds_{name}"] = r.fit_seconds
        out[f"model_size_bytes_{name}"] = [float(r.size_bytes)]
        if r.extract_seconds:
            out[f"extract_shard_seconds_{name}"] = r.extract_seconds
        if r.peak_mem:
            out[f"peak_gpu_mem_bytes_{name}"] = [float(m) for m in r.peak_mem]
    return out


def _comparisons(rows: pd.DataFrame, systems: dict[str, Predictions]) -> list[dict[str, Any]]:
    groups = list(rows["group"])
    scored = {s: Scored(groups=groups, labels=p.label, probs=p.probs) for s, p in systems.items()}
    out = []
    for a, b in itertools.combinations(SYSTEMS, 2):
        for metric in COMPARED:
            got = recompute_comparison(scored[a], scored[b], metric)
            out.append({"a": a, "b": b, "metric": metric, "diff": got["diff"], "ci": got["ci"],
                        "mcnemar_p": got["mcnemar_p"]})
    return out


def run(cfg: dict[str, Any], root: Path = ROOT) -> list[Path]:
    """Train the three baselines on lab_train, score all four systems on each split in cfg['splits'], and write
    one record per split."""
    if is_dirty(root):
        raise RuntimeError("working tree has uncommitted changes; commit first")
    started = datetime.now(timezone.utc).isoformat()
    cache = root / cfg["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if free < cfg["min_free_disk_gb"] * GIB:
        raise RuntimeError(f"only {free / GIB:.1f} GiB free; E3 stops below {cfg['min_free_disk_gb']} GiB")
    splits = list(cfg["splits"])
    for split in splits:
        if split not in KINDS:
            raise ValueError(f"E3 scores only {sorted(KINDS)}; refusing {split!r}")
    log = Log(cache / "stages.log")
    ckpt, e1_attempt, sha = load_e1(cfg, root, "E3")
    e1_seconds = [h["seconds"] for h in e1_attempt["history"]]
    class_map: dict[str, int] = ckpt["class_map"]
    class_names = sorted(class_map, key=class_map.get)
    if class_map != make_class_map(load_manifest("lab")):
        raise RuntimeError("the E1 checkpoint's classes differ from the lab manifest's")
    shared = shared_classes()
    train_rows, dev_rows = load_split("lab_train"), load_split("lab_dev")
    dev_mask = dev_rows["label"].isin(shared).to_numpy()
    eval_rows = {s: load_rows(s, shared) for s in splits}
    for s, rows in eval_rows.items():
        if rows.empty:
            raise RuntimeError(f"{s} has no rows in the shared classes")
    needs = {"lab_train": train_rows, "lab_dev": dev_rows,
             **({"field_dev": eval_rows["field_dev"]} if "field_dev" in splits else {})}
    y_train = np.array([class_map[x] for x in train_rows["label"]], dtype=np.int64)
    y_dev = np.array([class_map[x] for x in dev_rows["label"]], dtype=np.int64)
    n = len(class_map)
    log(f"E3 start: train={len(train_rows)} dev={len(dev_rows)} splits={splits} classes={n} shared={len(shared)}")

    results: dict[str, Result] = {}
    results["xgboost"] = run_classic(cfg, needs, y_train, y_dev, n, dev_mask, splits, cache, log)
    convnext = cfg["frozen_convnext"]
    results["frozen_convnext"] = run_encoder(
        "frozen_convnext", convnext, load_convnext_encoder(convnext["model"]), needs, y_train, y_dev, n,
        dev_mask, splits, cfg, cache, log)
    results["dinov2_small"] = run_encoder(
        "dinov2_small", cfg["dinov2_small"], load_dino_encoder(), needs, y_train, y_dev, n, dev_mask, splits,
        cfg, cache, log)

    results["e1"] = run_e1(cfg, ckpt, sha, e1_seconds, eval_rows, splits, log)

    measurements = _measurements(results)
    labels = {
        "e1": f"{ckpt['config'].model} fine-tuned on lab data, checkpoint {sha[:12]}",
        "dinov2_small": f"DINOv2 small ({dino.SOURCE['model_id']}), frozen, linear head trained on lab_train",
        "frozen_convnext": f"{convnext['model']} with ImageNet weights, frozen, linear head trained on lab_train",
        "xgboost": "HOG and colour histograms with XGBoost, trained on lab_train (CPU)",
    }
    attempts = [results[s].attempt(s) for s in SYSTEMS]
    paths = []
    for split in splits:
        rows = eval_rows[split]
        labels_idx = np.array([class_map[x] for x in rows["label"]], dtype=int)
        systems = {s: Predictions(list(rows["path"]), labels_idx, results[s].probs[split]) for s in SYSTEMS}
        metrics, intervals = score_systems(rows, systems)
        comparisons = _comparisons(rows, systems)
        config = {
            "class_names": class_names,
            "e3": dict(cfg),
            "checkpoint": cfg["checkpoint"],
            "checkpoint_sha256": sha,
            "e1_run_id": cfg["e1_run_id"],
            "shared_classes": sorted(shared),
            "outputs_without_shared_class": sorted(set(class_names) - shared),
            "trained_on": "lab_train",
            "xgboost_device": cfg["classic"]["xgboost"]["device"],
            "dinov2": dino.SOURCE,
            "started_at": started,
        }
        notes = (
            f"Baselines trained on lab_train only; lab_dev chose each head's best epoch. Scored on {split} over "
            f"the {len(shared)} shared classes with all {n} outputs kept, so a prediction outside the shared list "
            "is wrong. Pairs are compared with the paired bootstrap by near-duplicate group and exact McNemar. "
            "Training time is feature extraction on lab_train plus fitting; model size is the weights needed at "
            f"inference in bytes (XGBoost: its saved file). DINOv2 small: {dino.SOURCE['license']}, not gated, per "
            "the README of facebookresearch/dinov2. lab_test, field_test and field_train were not read."
        )
        paths.append(write_run(
            cfg["experiment"], split, config, labels, metrics, intervals, attempts, systems,
            dataset_hash(load_manifest(KINDS[split])), root=root, notes=notes, subset="shared_classes",
            comparisons=comparisons, measurements=measurements,
        ))
        log(f"{split}: n={len(rows)} " + " ".join(f"{s}={metrics[s]['accuracy']:.4f}" for s in SYSTEMS))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e3.json")
    a = ap.parse_args(argv)
    cfg = json.loads((ROOT / a.config).read_text(encoding="utf-8"))
    try:
        for path in run(cfg, ROOT):
            print(path)
    except (RuntimeError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
