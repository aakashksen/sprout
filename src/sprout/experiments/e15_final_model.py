"""E15: the final model. ConvNeXt V2 Tiny on 29 classes, trained on every usable lab and real photo, drawn per pass.

Settings and the decision rules are in configs/e15.json. Run from the repo root with PYTHONPATH set to its src folder,
one stage per command (each is safe to run alone, and says what it needs):
  classes   CPU  the 80/10/10 class rule from the manifests -> results/e15/classes.json; refuses if it differs from the config
  extra     CPU or GPU (--device cuda) rebuild extra_train without the cap (same copy checks; the 2026-11-04 attempt on
                 the CPU was still running after 31 minutes, the T4.8 build took 123 minutes on a GPU share)
                 -> datasets/manifests/extra_train_full.csv
                 and results/e15/extra_full.json (rows, hash, the rows the cap used to cut)
  draws     CPU  every pass's draw, and how many different photos training will see -> results/e15/draws.json
  train     GPU  10 passes, best pass chosen on field_dev accuracy -> models/e15/final/{best,latest}.pt (--resume continues)
  score     GPU* the new model on field_dev, E4/E12/E1 rescored from saved predictions on the same 29 classes and rows
                 -> one E15 record (needs a clean committed tree)      (*CPU works, slower)
  decide    CPU  the fixed rule -> results/e15/decision.json
  refit     CPU  E5's calibration and gate settings on the chosen model, plus the stricter healthy threshold
                 -> results/e15/refit/chosen.json and report.json
field_test and lab_test photos are never opened; field_test rows are only counted per class for the class rule.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprout.atomic import atomic_write_text
from sprout.data.splits import MANIFEST_DIR, dataset_hash, load_manifest
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)

SPLIT = "field_dev"
NEW = "convnextv2_final"
MAX_WORKERS = 4


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["split"] != SPLIT or cfg["settings"]["select_split"] != SPLIT:
        raise ValueError(f"E15 selects and scores on {SPLIT} only")
    if len(set(cfg["classes"])) != len(cfg["classes"]) or cfg["classes"] != sorted(cfg["classes"]):
        raise ValueError("classes must be sorted and unique")
    return cfg


def require_not_test(split: str) -> None:
    if split.endswith("_test"):
        raise ValueError(f"E15 never reads a test split, not {split!r}")


def resolve_workers(cfg: dict[str, Any], flag: int | None = None) -> int:
    spec = cfg["num_workers"]
    value = flag if flag is not None else int(os.environ.get(spec["env"], spec["default"]))
    if not 0 <= value <= min(spec["max"], MAX_WORKERS):
        raise ValueError(f"num_workers is {value}; at most {spec['max']}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---- the class rule ----

def derive_classes(lab: pd.DataFrame, field: pd.DataFrame, extra: pd.DataFrame, rule: dict[str, Any]):
    """(kept, dropped, table). A class stays with at least min_real_train real training photos (field_train plus
    all extra_train rows) and min_dev and min_test photos in field_dev and field_test. The test column is a row
    count; no field_test photo is opened."""
    classes = sorted(set(lab["label"]) | set(field["label"]) | set(extra["label"]))
    count = lambda df: df.groupby("label").size().reindex(classes, fill_value=0)  # noqa: E731
    table = pd.DataFrame({
        "real_train": count(field[field["split"] == "field_train"]) + count(extra),
        "dev": count(field[field["split"] == "field_dev"]),
        "test": count(field[field["split"] == "field_test"]),
        "lab_train": count(lab[lab["split"] == "lab_train"]),
    })
    ok = ((table["real_train"] >= rule["min_real_train"]) & (table["dev"] >= rule["min_dev"])
          & (table["test"] >= rule["min_test"]))
    return list(table.index[ok]), list(table.index[~ok]), table


# ---- the photos and the draws ----

def build_pools(lab: pd.DataFrame, field: pd.DataFrame, extra: pd.DataFrame, classes: list[str]):
    """(real pool, lab pool, selection rows) for these classes. Only lab_train, field_train and extra_train train;
    only field_dev selects. Test rows are never part of any of them."""
    keep = set(classes)
    real = pd.concat([field[field["split"] == "field_train"], extra[extra["split"] == "extra_train"]], ignore_index=True)
    labp = lab[lab["split"] == "lab_train"].assign(source="lab")
    sel = field[field["split"] == SPLIT]
    out = tuple(d[d["label"].isin(keep)].sort_values("path").reset_index(drop=True) for d in (real, labp, sel))
    for d in out:
        if d["split"].str.endswith("_test").any():
            raise RuntimeError("a test row reached the pools")
    return out


def _cycle(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n positions in range(m): shuffled passes over all m, so every photo is used floor(n/m) or ceil(n/m) times."""
    out: list[int] = []
    while len(out) < n:
        out.extend(rng.permutation(m).tolist())
    return np.array(out[:n], dtype=int)


def draw_pass(real: pd.DataFrame, lab: pd.DataFrame, settings: dict[str, Any], pass_index: int) -> pd.DataFrame:
    """One pass's training rows: real_per_class real photos per class split evenly across that class's sources,
    lab_per_class lab photos per class. A function of (seed, pass_index) alone, so a resumed run draws the same."""
    seed, per_real, per_lab = settings["seed"], settings["real_per_class"], settings["lab_per_class"]
    rng_real = np.random.default_rng([seed, pass_index, 1])
    rng_lab = np.random.default_rng([seed, pass_index, 2])
    parts = []
    for label, part in real.groupby("label", sort=True):
        sources = sorted(part["source"].unique())
        base, rem = divmod(per_real, len(sources))
        bonus = set(rng_real.permutation(len(sources))[:rem].tolist())
        for i, src in enumerate(sources):
            mine = part[part["source"] == src].sort_values("path")
            parts.append(mine.iloc[_cycle(len(mine), base + (i in bonus), rng_real)])
    for label, part in lab.groupby("label", sort=True):
        part = part.sort_values("path")
        parts.append(part.iloc[_cycle(len(part), per_lab, rng_lab)] if len(part) < per_lab
                     else part.iloc[rng_lab.permutation(len(part))[:per_lab]])
    return pd.concat(parts, ignore_index=True)


def summarize_draws(real: pd.DataFrame, lab: pd.DataFrame, settings: dict[str, Any], passes: int) -> dict[str, Any]:
    """How many photos each pass holds and how many different ones training sees over all passes."""
    seen_real: set[str] = set()
    seen_lab: set[str] = set()
    per_pass = []
    for p in range(passes):
        d = draw_pass(real, lab, settings, p)
        is_lab = d["source"] == "lab"
        seen_real |= set(d.loc[~is_lab, "path"])
        seen_lab |= set(d.loc[is_lab, "path"])
        per_pass.append({"pass": p + 1, "real": int((~is_lab).sum()), "lab": int(is_lab.sum())})
    src = real[real["path"].isin(seen_real)]["source"].value_counts().sort_index()
    return {
        "passes": per_pass, "draws_total": sum(x["real"] + x["lab"] for x in per_pass),
        "different_real_photos_seen": len(seen_real), "different_lab_photos_seen": len(seen_lab),
        "different_photos_seen": len(seen_real) + len(seen_lab),
        "different_real_by_source": {k: int(v) for k, v in src.items()},
        "real_pool": len(real), "lab_pool": len(lab),
    }


# ---- rescoring saved predictions on the final classes ----

def restrict(probs: np.ndarray, labels: np.ndarray, names: list[str], keep: list[str]):
    """Rows whose true class is kept; outputs of the dropped classes set to zero and each row renormalized; then
    only the kept columns, in `keep` order, with labels re-indexed. Returns (probs, labels, rows_kept)."""
    labels = np.asarray(labels, dtype=int)
    cols = [names.index(c) for c in keep]
    rows_kept = np.isin(labels, cols)
    remap = {c: i for i, c in enumerate(cols)}
    return (mask_columns(probs, names, keep)[rows_kept], np.array([remap[x] for x in labels[rows_kept]], dtype=int),
            rows_kept)


def mask_columns(probs: np.ndarray, names: list[str], keep: list[str]) -> np.ndarray:
    """Outputs of classes outside `keep` set to zero, each row renormalized, only the kept columns left."""
    probs = np.asarray(probs, dtype=float)
    cols = [names.index(c) for c in keep]
    kept = probs[:, cols]
    total = kept.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise RuntimeError("a row has no probability left on the kept classes")
    return kept / total


def rescored_rows(rows: list[dict[str, Any]], names: list[str], keep: list[str]) -> list[dict[str, Any]]:
    probs, labels, kept = restrict(np.array([r["probs"] for r in rows]), np.array([r["label"] for r in rows]), names, keep)
    paths = [r["path"] for r, k in zip(rows, kept, strict=True) if k]
    return [{"path": p, "label": int(y), "probs": [float(v) for v in pr]} for p, y, pr in zip(paths, labels, probs)]


# ---- the healthy threshold ----

def healthy_threshold(abstain: float) -> float:
    """Halfway between the abstain threshold and 1 (configs/e15.json refit.healthy_rule)."""
    return round(abstain + (1 - abstain) / 2, 6)


def answered(cal: np.ndarray, names: list[str], abstain: float, healthy: float, suffix: str) -> np.ndarray:
    """Which rows are answered: top probability at or above the abstain threshold, or the healthy one when the top
    label ends in the suffix."""
    top, conf = cal.argmax(axis=1), cal.max(axis=1)
    is_healthy = np.array([n.endswith(suffix) for n in names])[top]
    return conf >= np.where(is_healthy, max(healthy, abstain), abstain)


# ---- stages ----

def results_dir(cfg: dict[str, Any], root: Path) -> Path:
    d = root / cfg["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(obj, indent=1) + "\n")
    return path


def stage_classes(cfg: dict[str, Any], root: Path) -> Path:
    kept, dropped, table = derive_classes(load_manifest(cfg["lab_manifest"]), load_manifest(cfg["field_manifest"]),
                                          _extra(cfg), cfg["class_rule"])
    out = {"rule": cfg["class_rule"], "kept": kept, "dropped": dropped, "matches_config": kept == cfg["classes"],
           "table": json.loads(table.to_json(orient="index")), "extra_manifest": cfg["extra"]["manifest"]}
    path = write_json(results_dir(cfg, root) / "classes.json", out)
    if kept != cfg["classes"]:
        raise RuntimeError(f"the rule gives {len(kept)} classes, not the config's {len(cfg['classes'])}; see {path}")
    return path


def _extra(cfg: dict[str, Any]) -> pd.DataFrame:
    """The uncapped extras when built, else (for the class rule's first look) the capped ones."""
    name = cfg["extra"]["manifest"]
    return load_manifest(name if (MANIFEST_DIR / f"{name}.csv").is_file() else "extra_train")


def stage_extra(cfg: dict[str, Any], root: Path, summary: dict[str, Any] | None = None, device: str = "cpu") -> Path:
    """Build the uncapped manifest and describe it. `summary` is the dict extra_sources.main returned (or printed)
    for a build that already ran, so a finished build is not repeated."""
    from sprout.data import extra_sources

    summary = summary or extra_sources.main(lambda: contextlib.nullcontext(device), cap=10**9,
                                            out_name=cfg["extra"]["manifest"])
    new = load_manifest(cfg["extra"]["manifest"])
    old = load_manifest("extra_train")
    cut = summary["removed"].get("over the per-class cap", 0)
    out = {"rows": len(new), "dataset_hash": dataset_hash(new), "capped_rows": len(old),
           "removed": summary["removed"], "over_the_cap_now": cut,
           "capped_plus_expected_cut": len(old) + cfg["extra"]["expected_cap_cut"],
           "rows_not_in_capped": int((~new["path"].isin(set(old["path"]))).sum()),
           "capped_rows_missing_now": int((~old["path"].isin(set(new["path"]))).sum()),
           "by_source": {k: int(v) for k, v in new["source"].value_counts().sort_index().items()},
           "matches_7303": len(new) - len(old) == cfg["extra"]["expected_cap_cut"] and cut == 0}
    return write_json(results_dir(cfg, root) / "extra_full.json", out)


def check_extra(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    f = results_dir(cfg, root) / "extra_full.json"
    if not f.is_file():
        raise RuntimeError("results/e15/extra_full.json is missing; run the extra stage first")
    frozen = json.loads(f.read_text(encoding="utf-8"))
    extra = load_manifest(cfg["extra"]["manifest"])
    if len(extra) != frozen["rows"] or dataset_hash(extra) != frozen["dataset_hash"]:
        raise RuntimeError("the extra manifest differs from results/e15/extra_full.json")
    return extra


def stage_draws(cfg: dict[str, Any], root: Path) -> Path:
    kept = cfg["classes"]
    real, lab, _ = build_pools(load_manifest(cfg["lab_manifest"]), load_manifest(cfg["field_manifest"]),
                               check_extra(cfg, root), kept)
    out = summarize_draws(real, lab, cfg["settings"], cfg["settings"]["passes"])
    out["classes"] = len(kept)
    out["real_pool_by_source"] = {k: int(v) for k, v in real["source"].value_counts().sort_index().items()}
    return write_json(results_dir(cfg, root) / "draws.json", out)


def train_config(cfg: dict[str, Any], num_workers: int, root: Path, manifest_hash: str):
    from sprout.train.config import TrainConfig

    s = cfg["settings"]
    return TrainConfig(
        manifest=cfg["lab_manifest"], out_dir=str(root / cfg["out_dir"]), model=cfg["model"]["name"], seed=s["seed"],
        lr=s["lr"], batch_size=s["batch_size"], epochs=s["passes"], transforms=tuple(s["transforms"]),
        manifest_hash=manifest_hash, num_workers=num_workers, field_manifest=cfg["field_manifest"],
        train_splits=("lab_train", "field_train", "extra_train"), select_split=s["select_split"], lab_cap=0,
        extra_manifest=cfg["extra"]["manifest"])


def stage_train(cfg: dict[str, Any], root: Path, *, resume: bool = False, num_workers: int | None = None) -> Path:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from sprout.train.checkpoint import load_checkpoint, save_checkpoint
    from sprout.train.joint import JointDataset
    from sprout.train.loop import _evaluate, _train_epoch, build_model

    workers = resolve_workers(cfg, num_workers)
    out_dir = root / cfg["out_dir"]
    latest, best = out_dir / "latest.pt", out_dir / "best.pt"
    if not resume and (latest.exists() or best.exists()):
        raise FileExistsError(f"{out_dir} has checkpoints; pass --resume or use a new out_dir")
    if resume and not latest.is_file():
        raise FileNotFoundError(latest)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["classes"]
    class_map = {c: i for i, c in enumerate(classes)}
    real, lab, sel = build_pools(load_manifest(cfg["lab_manifest"]), load_manifest(cfg["field_manifest"]),
                                 check_extra(cfg, root), classes)
    tc = train_config(cfg, workers, root, dataset_hash(pd.concat([real, lab, sel], ignore_index=True)))
    seen = summarize_draws(real, lab, cfg["settings"], tc.epochs)
    write_json(out_dir / "photos_seen.json", seen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(tc.model, len(classes)).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    start, best_metric, history = 0, float("-inf"), []
    if resume:
        ck = load_checkpoint(latest)
        if ck["class_map"] != class_map or ck["config"].manifest_hash != tc.manifest_hash:
            raise ValueError("the checkpoint is from other classes or other photos")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best_metric = ck["epoch"], ck["best_metric"]
        hist_file = out_dir / "history.json"
        history = json.loads(hist_file.read_text(encoding="utf-8")) if hist_file.is_file() else []
    val_dl = DataLoader(JointDataset(sel, class_map, train=False), batch_size=tc.batch_size, num_workers=workers)
    for epoch in range(start, tc.epochs):
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        random.seed(tc.seed * 1000 + epoch)
        torch.manual_seed(tc.seed * 1000 + epoch)
        rows = draw_pass(real, lab, cfg["settings"], epoch)
        gen = torch.Generator().manual_seed(tc.seed * 1000 + epoch)
        train_dl = DataLoader(JointDataset(rows, class_map, train=True, transforms=tc.transforms), shuffle=True,
                              generator=gen, batch_size=tc.batch_size, num_workers=workers,
                              pin_memory=device.type == "cuda")
        train_loss = _train_epoch(model, train_dl, optimizer, scaler, loss_fn, device, epoch)
        scheduler.step()
        val_loss, val_acc = _evaluate(model, val_dl, loss_fn, device)
        history.append({"pass": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                        "rows": len(rows), "seconds": time.perf_counter() - began,
                        "peak_mem_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
        print(f"pass {epoch + 1}/{tc.epochs} {history[-1]}", flush=True)
        improved = val_acc > best_metric
        best_metric = max(best_metric, val_acc)
        state = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1,
                     best_metric=best_metric, class_map=class_map, config=tc)
        write_json(out_dir / "history.json", history)
        save_checkpoint(latest, **state)
        if improved:
            save_checkpoint(best, **state)
    return best


def load_model(path: Path, classes: list[str]):
    """The model in a checkpoint, with its class names checked, ready on the CPU."""
    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import build_model

    ck = load_checkpoint(path)
    names = [k for k, _ in sorted(ck["class_map"].items(), key=lambda kv: kv[1])]
    model = build_model(ck["config"].model, len(names), pretrained=False)
    model.load_state_dict(ck["model"])
    return model, names


def predict(model, paths: list[str], root: Path, batch_size: int, num_workers: int, device) -> np.ndarray:
    from sprout.experiments.e5_confidence import score_images

    return score_images(model, paths, root, batch_size, num_workers, device)


def compared_rows(cfg: dict[str, Any], root: Path):
    """The explicit comparison row set: field_dev rows in clean version 1 whose class is one of the 29, in the
    order check_run reads them. Returns (info, ids, labels, groups)."""
    info = split_info(SPLIT, root)
    pos = subset_positions(cfg["subset"], SPLIT, info, root)
    return info, [info.ids[i] for i in pos], [info.labels[i] for i in pos], [info.groups[i] for i in pos]


def saved_systems(cfg: dict[str, Any], root: Path, ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """E1, E4 and E12 from the saved predictions of the E12 comparison record, rescored on the 29 classes and
    restricted to these rows."""
    rec = json.loads((root / RUNS_REL / f"{cfg['compare']['record']}.json").read_text(encoding="utf-8"))
    if rec["split"] != SPLIT or rec["dataset_hash"] != split_info(SPLIT, root).hash:
        raise RuntimeError("the comparison record is not on the current field_dev manifest")
    names = rec["config"]["class_names"]
    out = {}
    for ours, theirs in cfg["compare"]["systems"].items():
        by_path = {r["path"]: r for r in read_jsonl(root / rec["predictions"][theirs])}
        missing = [p for p in ids if p not in by_path]
        if missing:
            raise RuntimeError(f"{theirs} lacks {len(missing)} of the compared rows, e.g. {missing[0]}")
        out[ours] = rescored_rows([by_path[p] for p in ids], names, cfg["classes"])
    return out


def stage_score(cfg: dict[str, Any], root: Path, *, num_workers: int | None = None, note: str = "") -> Path:
    import torch

    from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval
    from sprout.experiments.e4_improve import verdict

    workers = resolve_workers(cfg, num_workers)
    classes = cfg["classes"]
    class_map = {c: i for i, c in enumerate(classes)}
    out_dir = root / cfg["out_dir"]
    model, names = load_model(out_dir / "best.pt", classes)
    if names != classes:
        raise RuntimeError("the checkpoint's classes differ from the config's")
    field = load_manifest(cfg["field_manifest"])
    all_rows = field[(field["split"] == SPLIT) & field["label"].isin(set(classes))].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_probs = predict(model, list(all_rows["path"]), root, cfg["settings"]["batch_size"], workers, device)
    atomic_write_text(out_dir / "field_dev.predictions.jsonl", "".join(
        json.dumps({"path": p, "label": class_map[y], "probs": [float(v) for v in pr]}) + "\n"
        for p, y, pr in zip(all_rows["path"], all_rows["label"], all_probs, strict=True)))

    info, ids, labels, groups = compared_rows(cfg, root)
    index = {p: i for i, p in enumerate(all_rows["path"])}
    systems = saved_systems(cfg, root, ids)
    systems[NEW] = [{"path": p, "label": class_map[y], "probs": [float(v) for v in all_probs[index[p]]]}
                    for p, y in zip(ids, labels, strict=True)]
    arrays = {s: (np.array([r["label"] for r in rs]), np.array([r["probs"] for r in rs])) for s, rs in systems.items()}
    for s, (y, _) in arrays.items():
        if [classes[i] for i in y] != labels:
            raise RuntimeError(f"{s}: labels differ from the manifest's for the compared rows")
    scored = {s: Scored(groups=groups, labels=y, probs=p) for s, (y, p) in arrays.items()}
    metrics = {s: standard_image_metrics(p, y, classes) for s, (y, p) in arrays.items()}
    intervals = {s: {m: recompute_interval(m, sc) for m in ("accuracy", "macro_f1")} for s, sc in scored.items()}
    comparisons = [{"a": NEW, "b": other, "metric": m, **{k: v for k, v in recompute_comparison(
        scored[NEW], scored[other], m).items() if k in ("diff", "ci", "mcnemar_p")}}
        for other in cfg["compare"]["systems"] for m in ("accuracy", "macro_f1")]
    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    seen = json.loads((out_dir / "photos_seen.json").read_text(encoding="utf-8"))
    ck = out_dir / "best.pt"
    config = {
        "run": NEW, "class_names": classes, "e15": cfg, "rows_rule": cfg["compare"]["rows"],
        "rescore_rule": cfg["compare"]["rescore"], "photos_seen": seen, "history": history,
        "checkpoint": cfg["out_dir"] + "/best.pt", "checkpoint_sha256": sha256_file(ck),
        "weights_license": cfg["model"]["weights_license"], "comparison_record": cfg["compare"]["record"],
        "field_test_read": False, "label": "development number, field_dev only", "selected_on": "field_dev (optimistic)",
        "device": str(device),
    }
    models = {s: f"{s}: saved predictions of {cfg['compare']['record']}, rescored on the 29 classes" for s in cfg["compare"]["systems"]}
    models[NEW] = f"{NEW}: {cfg['model']['timm_name']} fine-tuned here on the 29 classes"
    notes = (f"Development numbers on field_dev only, which also picked the best pass. Scored on {cfg['subset']} "
             f"({len(ids)} photos). E1, E4 and E12 are their saved predictions with the other classes masked and "
             f"renormalized. Different photos training saw: {seen['different_photos_seen']} "
             f"({seen['different_real_photos_seen']} real, {seen['different_lab_photos_seen']} lab). "
             f"Paired: {'; '.join(verdict(c) for c in comparisons)}. {note}").strip()
    return write_run(cfg["experiment"], SPLIT, config, models, metrics, intervals,
                     [{"step": "train", "passes": len(history), "best_val_acc": max(h["val_acc"] for h in history)}],
                     systems, info.hash, root=root, notes=notes, subset=cfg["subset"], comparisons=comparisons)


def newest_record(cfg: dict[str, Any], root: Path) -> tuple[Path, dict[str, Any]]:
    from sprout.eval.check_run import records_for

    found = records_for(cfg["experiment"], root)
    found = [p for p in found if json.loads(p.read_text(encoding="utf-8"))["split"] == SPLIT]
    if not found:
        raise RuntimeError("no E15 field_dev record yet; run the score stage")
    return found[-1], json.loads(found[-1].read_text(encoding="utf-8"))


def choose(f1: dict[str, float], current: str) -> tuple[str, str]:
    """(chosen system, why): the highest macro-F1 of the new model, E4 and E12; a tie goes to the new model, then
    to `current` (E4), then E12."""
    order = [NEW, current, *(s for s in f1 if s not in (current, NEW, "e1"))]
    best = max(order, key=lambda s: (f1[s], -order.index(s)))
    return best, f"highest macro-F1 of {order}: {best} at {f1[best]:.4f} ({NEW} {f1[NEW]:.4f}, {current} {f1[current]:.4f})"


def stage_decide(cfg: dict[str, Any], root: Path) -> Path:
    from sprout.eval.check_run import check_file

    path, rec = newest_record(cfg, root)
    problems = check_file(path, root)
    if problems:
        raise RuntimeError(f"{path.name} does not pass check_run: {problems[0]}")
    f1 = {s: m["macro_f1"] for s, m in rec["metrics"].items()}
    current = cfg["decision"]["current"]
    chosen, why = choose(f1, current)
    ckpt = rec["config"]["checkpoint"] if chosen == NEW else cfg["compare"]["checkpoints"][chosen]
    out = {"experiment": cfg["experiment"], "decided_at": datetime.now(timezone.utc).isoformat(), "record": rec["run_id"],
           "rule": cfg["decision"]["rule"], "macro_f1": f1, "current": current, "chosen": chosen, "why": why,
           "chosen_checkpoint": ckpt, "chosen_checkpoint_sha256": sha256_file(root / ckpt), "classes": cfg["classes"],
           "n": rec["n"], "label": "development numbers, field_dev only"}
    return write_json(root / cfg["decision"]["file"], out)


# ---- refit of E5's calibration and thresholds on the chosen model ----

def stage_refit(cfg: dict[str, Any], root: Path, *, device_name: str = "cpu") -> Path:
    import torch

    from sprout.experiments import e5_confidence as e5

    r = cfg["refit"]
    decision = json.loads((root / cfg["decision"]["file"]).read_text(encoding="utf-8"))
    classes, chosen = cfg["classes"], decision["chosen"]
    _, rec = newest_record(cfg, root)
    if rec["run_id"] != decision["record"]:
        raise RuntimeError("decision.json is not for the newest E15 record")
    info, ids, labels, groups = compared_rows(cfg, root)
    rows = {s: read_jsonl(root / rec["predictions"][s]) for s in rec["predictions"]}
    y = np.array([row["label"] for row in rows[chosen]], dtype=int)
    green = {x["path"]: x["green_ratio"] for x in e5.read_jsonl(root / r["green_file"])}
    np_info = split_info(r["not_plant_split"], root)
    ckpt = root / decision["chosen_checkpoint"]
    model, names = load_model(ckpt, classes)
    began = time.perf_counter()
    raw_np = predict(model, np_info.ids, root, r["batch_size"], r["num_workers"], torch.device(device_name))
    raw_np = mask_columns(raw_np, names, classes)
    probs = np.array([row["probs"] for row in rows[chosen]], dtype=float)
    field = e5.Dev(SPLIT, info.hash, ids, groups, probs, np.array([green[p] for p in ids]), y, classes)
    nonplant = e5.Dev(r["not_plant_split"], np_info.hash, np_info.ids, np_info.groups, raw_np,
                      np.array([green[p] for p in np_info.ids]), None, None)
    e5cfg = {"seed": r["seed"], "fit_percent": r["fit_percent"], "abstention": r["abstention"], "gates": r["gates"]}
    fitted = e5.fit(e5cfg, field, nonplant)
    t, suffix = fitted["abstain_threshold"], r["healthy_suffix"]
    t_h = healthy_threshold(t)
    fitted |= {"classes": classes, "healthy_threshold": t_h, "healthy_suffix": suffix, "model": chosen,
               "model_checkpoint": decision["chosen_checkpoint"], "model_sha256": decision["chosen_checkpoint_sha256"],
               "e15_record": rec["run_id"], "healthy_rule": r["healthy_rule"], "fit_seconds": time.perf_counter() - began}
    out_dir = root / r["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "chosen.json", fitted)

    # the cost, on the scoring part, beside E4's and E12's saved settings applied to their rescored outputs
    score = field.scoring(e5cfg)
    flags = np.array([c.endswith(suffix) for c in classes])
    settings = {chosen: (fitted["temperature"], t, t_h)}
    for system, f in (("convnext_joint", r["e4_settings"]), ("convnext_joint_extra", r["e12_settings"])):
        s = json.loads((root / f).read_text(encoding="utf-8"))
        settings.setdefault(system, (s["temperature"], s["abstain_threshold"], None))
    report: dict[str, Any] = {"rows_scored": int(score.sum()), "classes": len(classes), "systems": {}}
    for system, (temp, thr, healthy_thr) in settings.items():
        sp = np.array([row["probs"] for row in rows[system]], dtype=float)[score]
        sy = np.array([row["label"] for row in rows[system]], dtype=int)[score]
        cal = e5.calibrate(sp, temp)
        entry: dict[str, Any] = {"temperature": temp, "abstain_threshold": thr}
        for label, hthr in (("abstain_only", thr), ("with_healthy_threshold", healthy_thr)):
            if hthr is None:
                continue
            ok = answered(cal, classes, thr, hthr, suffix)
            wrong = ok & (cal.argmax(axis=1) != sy)
            entry[label] = {
                "healthy_threshold": hthr, "answered": int(ok.sum()), "unsure": int((~ok).sum()),
                "unsure_share": float((~ok).mean()),
                "healthy_answers": int((ok & flags[cal.argmax(axis=1)]).sum()),
                "confident_sick_called_healthy": int((ok & ~flags[sy] & flags[cal.argmax(axis=1)]).sum()),
                "confident_wrong": int(wrong.sum())}
        report["systems"][system] = entry
    report["chosen"] = chosen
    report["note"] = ("Counts are on the scoring part of the compared rows (seed 0, 50% of groups). E4 and E12 use their saved "
                      "temperature and threshold on outputs masked to the 29 classes and renormalized.")
    return write_json(out_dir / "report.json", report)


STAGES = ("classes", "extra", "draws", "train", "score", "decide", "refit")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e15.json")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="extra stage only: where the DINOv2 copy checks run")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        if a.stage == "train":
            result: Any = stage_train(cfg, ROOT, resume=a.resume, num_workers=a.num_workers)
        elif a.stage == "extra":
            result = stage_extra(cfg, ROOT, device=a.device)
        elif a.stage == "score":
            result = stage_score(cfg, ROOT, num_workers=a.num_workers, note=a.note)
        else:
            result = {"classes": stage_classes, "draws": stage_draws, "decide": stage_decide,
                      "refit": stage_refit}[a.stage](cfg, ROOT)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
