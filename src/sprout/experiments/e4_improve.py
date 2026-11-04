"""E4: what raises real-photo performance? Joint training on real and lab photos, then one change at a time.

Run from the repo root with PYTHONPATH set to its src folder, one arm per command so each takes its own GPU turn:
    .venv/Scripts/python -m sprout.experiments.e4_improve --arm <name> [--resume] [--note TEXT]

Arms are listed, with the rules for choosing among them, in configs/e4.json (committed before any run). Suggested
order, and what each command needs:
  convnext_joint, convnext_strong_aug, convnext_background, dinov2_head   GPU: train, score field_dev, Grad-CAM on lab
  heldout_plantdoc, heldout_plantseg                                      GPU: train on one real source, score the other
  dinov2_finetune     decides from the records of dinov2_head and convnext_joint whether to run; if not, it records
                      "not run" (CPU only, no training)
  best_with_crop      decides which arm is best from the records of arms 1 to 5; then CPU or GPU to cut masks for the
                      photos it needs with models/e10/best.pt (cached), then GPU: train and score with the crop
Run the held-out arms before best_with_crop, so the last record lists every arm. Every run writes one record on
field_dev (subset shared_classes, or source:<name> for a held-out run) with E1 and the three E3 baselines read from
E3's saved rows. A run that stops early (a non-finite loss, an interruption) writes no record; continue it with
--resume, which restarts from models/e4/<arm>/latest.pt and lists both attempts. Only best.pt and latest.pt are kept.
Per-epoch seconds and peak GPU memory go to models/e4/<arm>/epochs.log, flushed. field_test and lab_test are never read.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sprout.atomic import atomic_write_text
from sprout.data.splits import dataset_hash, load_manifest
from sprout.eval.check_run import (
    Scored,
    recompute_comparison,
    recompute_interval,
    records_for,
)
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    is_dirty,
    standard_image_metrics,
    write_run,
)
from sprout.experiments.e2_field import load_rows, score_split, shared_classes
from sprout.explain.gradcam import target_layer
from sprout.segment.model import build_model as build_segmenter
from sprout.segment.model import predict_mask
from sprout.train.checkpoint import load_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import make_class_map
from sprout.train.joint import (
    MASK_SIZE,
    MaskStore,
    eval_dataset,
    leaf_mask,
    load_joint_manifest,
    select_rows,
)
from sprout.train.loop import build_model, train

GIB = 1024**3
MAX_WORKERS = 8
MAX_TRAIN_ARMS = 6
ENERGY_MODELS = ("convnext_tiny", "tiny_cnn")  # models with a convolutional Grad-CAM layer
INTERVALS = ("accuracy", "macro_f1")
COMPARED = ("accuracy", "macro_f1")
CAVEAT_DEV = (
    "The arm was chosen on field_dev (its best epoch by field_dev accuracy, and the best arm by field_dev macro-F1), "
    "so its field_dev numbers are optimistic. These are development numbers, not claims."
)


class Blocked(RuntimeError):
    """An arm cannot be decided yet because an arm it reads from has no result."""


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["num_workers"] > MAX_WORKERS:
        raise ValueError(f"num_workers is {cfg['num_workers']}; at most {MAX_WORKERS}")
    names = [a["name"] for a in cfg["arms"]]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    if sum(a["kind"] != "heldout" for a in cfg["arms"]) > MAX_TRAIN_ARMS:
        raise ValueError(f"at most {MAX_TRAIN_ARMS} training arms besides the held-out runs")
    return cfg


def arm_spec(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for a in cfg["arms"]:
        if a["name"] == name:
            return a
    raise ValueError(f"unknown arm {name!r}; arms are {[a['name'] for a in cfg['arms']]}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def check_disk(path: Path, minimum_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum_gb * GIB:
        raise RuntimeError(f"only {free / GIB:.1f} GiB free; E4 stops below {minimum_gb} GiB")


# ----------------------------------------------------------------------------------------------------------------
# results of earlier arms, read from their records and from not-run entries


def arm_results(cfg: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    """The newest field_dev record of each arm: its run id, predictions file, and field_dev macro-F1."""
    names = {a["name"] for a in cfg["arms"]}
    out: dict[str, dict[str, Any]] = {}
    for path in records_for(cfg["experiment"], root):  # oldest first, so a later record wins
        rec = json.loads(path.read_text(encoding="utf-8"))
        arm = (rec.get("config") or {}).get("arm")
        if arm in names and arm in rec["metrics"] and rec["split"] == cfg["split"]:
            out[arm] = {"run_id": rec["run_id"], "macro_f1": rec["metrics"][arm]["macro_f1"],
                        "accuracy": rec["metrics"][arm]["accuracy"], "predictions": rec["predictions"][arm],
                        "base_arm": rec["config"].get("base_arm"), "subset": rec["subset"]}
    return out


def not_run_dir(cfg: dict[str, Any], root: Path) -> Path:
    return root / cfg["out_dir"] / "not_run"


def not_run_entries(cfg: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    d = not_run_dir(cfg, root)
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


def pick_best(cfg: dict[str, Any], results: dict[str, dict[str, Any]]) -> str:
    """Highest field_dev macro-F1 among arms 1 to 5 that ran; a tie goes to the lower arm number."""
    pool = [a for a in cfg["arms"] if a["kind"] == "train" and a["name"] in results]
    if not pool:
        raise Blocked("no arm among 1 to 5 has a record yet")
    return max(pool, key=lambda a: (results[a["name"]]["macro_f1"], -a["number"]))["name"]


def arm5_rule(results: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    for needed in ("dinov2_head", "convnext_joint"):
        if needed not in results:
            raise Blocked(f"arm {needed} has no record yet; dinov2_finetune reads it")
    head, joint = results["dinov2_head"]["macro_f1"], results["convnext_joint"]["macro_f1"]
    if head > joint:
        return True, f"dinov2_head field_dev macro-F1 {head:.4f} > convnext_joint {joint:.4f}"
    return False, f"dinov2_head field_dev macro-F1 {head:.4f} is not above convnext_joint {joint:.4f}"


def mask_share(rows: pd.DataFrame) -> tuple[float, dict[str, int]]:
    """Share of lab_train photos with a believable leaf mask, and how many came from each kind of mask."""
    kinds = {"official": 0, "hsv": 0, "none": 0}
    for path in rows["path"]:
        with Image.open(path) as im:
            mask, kind = leaf_mask(path, im.convert("RGB"))
        kinds[kind if mask is not None else "none"] += 1
    return (1 - kinds["none"] / max(len(rows), 1)), kinds


# ----------------------------------------------------------------------------------------------------------------
# settings


def resolved_spec(cfg: dict[str, Any], name: str, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The true settings of an arm: held-out arms copy their `same_as` arm, the crop arm copies the best arm."""
    spec = dict(arm_spec(cfg, name))
    if spec["kind"] == "heldout":
        base = {k: v for k, v in arm_spec(cfg, spec["same_as"]).items() if k not in ("number", "name", "change", "reference")}
        return {**base, **spec, "base_arm": spec["same_as"]}
    if spec["kind"] == "crop_of_best":
        best = pick_best(cfg, results)
        base = {k: v for k, v in arm_spec(cfg, best).items() if k not in ("number", "name", "change", "reference", "requires")}
        return {**base, **spec, "base_arm": best, "reference": best, "crop": True}
    return {**spec, "base_arm": None}


def train_config(cfg: dict[str, Any], spec: dict[str, Any], root: Path, crop_masks: str = "") -> TrainConfig:
    return TrainConfig(
        manifest=cfg["lab_manifest"],
        out_dir=str(root / cfg["out_dir"] / spec["name"]),
        model=spec["model"],
        seed=cfg["seed"],
        lr=spec["lr"],
        batch_size=spec["batch_size"],
        epochs=spec["epochs"],
        transforms=tuple(spec["transforms"]),
        num_workers=cfg["num_workers"],
        field_manifest=cfg["field_manifest"],
        train_splits=("lab_train", "field_train"),
        select_split="field_dev",
        lab_cap=cfg["lab_cap"],
        held_out_source=spec.get("held_out_source", ""),
        crop_masks=crop_masks,
        unfreeze_blocks=spec.get("unfreeze_blocks", 0),
        backbone_lr=spec.get("backbone_lr", 0.0),
    )


# ----------------------------------------------------------------------------------------------------------------
# crop masks


def load_segmenter(path: Path, device: torch.device):
    model = build_segmenter(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.eval()


def ensure_crop_masks(paths: list[str], store_path: Path, e10_path: Path, log) -> Path:
    """Predict (with E10) the disease mask of every photo not yet in the cache, at MASK_SIZE, and save the cache."""
    have = MaskStore.load(store_path).masks if store_path.is_file() else {}
    todo = [p for p in paths if p not in have]
    if todo:
        if not e10_path.is_file():
            raise RuntimeError(f"E10 segmentation checkpoint not found at {e10_path}; the crop arm cannot run")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_segmenter(e10_path, device)
        began = time.perf_counter()
        for i, p in enumerate(todo, 1):
            with Image.open(p) as im:
                mask = predict_mask(model, im.convert("RGB"))
            small = Image.fromarray(mask.astype(np.uint8)).resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
            have[p] = np.asarray(small, dtype=bool)
            if i % 500 == 0:
                log(f"crop masks {i}/{len(todo)} seconds={time.perf_counter() - began:.1f}")
        store_path.parent.mkdir(parents=True, exist_ok=True)
        MaskStore(have).save(store_path)
        log(f"crop masks: {len(todo)} predicted, {len(have)} cached, seconds={time.perf_counter() - began:.1f}")
    return store_path


# ----------------------------------------------------------------------------------------------------------------
# attempts of one arm (an interrupted run and its resumption are both kept)


def attempts_file(cfg: dict[str, Any], root: Path, name: str) -> Path:
    return root / cfg["out_dir"] / name / "attempts.json"


def start_attempt(cfg: dict[str, Any], root: Path, name: str, resume: bool) -> list[dict[str, Any]]:
    path = attempts_file(cfg, root, name)
    entries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if resume and not entries:
        raise RuntimeError("--resume given but this arm has no earlier attempt")
    if entries and entries[-1]["status"] == "started":
        entries[-1]["status"] = "interrupted"
        latest = path.parent / "latest.pt"
        if latest.is_file():
            entries[-1]["epochs_in_latest_checkpoint"] = torch.load(latest, map_location="cpu", weights_only=True)["epoch"]
    entries.append({"attempt": len(entries) + 1, "resumed_from_latest": resume, "status": "started",
                    "started_at": datetime.now(timezone.utc).isoformat(), "history": []})
    atomic_write_text(path, json.dumps(entries, indent=1) + "\n")
    return entries


def finish_attempt(cfg: dict[str, Any], root: Path, name: str, entries: list[dict[str, Any]], history: list[dict],
                   status: str) -> None:
    entries[-1].update(status=status, history=history, finished_at=datetime.now(timezone.utc).isoformat())
    atomic_write_text(attempts_file(cfg, root, name), json.dumps(entries, indent=1) + "\n")


# ----------------------------------------------------------------------------------------------------------------
# scoring


def predict_probs(model, rows: pd.DataFrame, class_map: dict[str, int], config: TrainConfig,
                  device: torch.device) -> np.ndarray:
    """Softmax outputs for the rows, through the same input path as training's selection (crop included)."""
    loader = DataLoader(eval_dataset(rows, class_map, config), batch_size=config.batch_size,
                        num_workers=config.num_workers)
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def saved_rows(root: Path, rel: str, paths: list[str]) -> list[dict[str, Any]]:
    """A system's saved prediction rows for exactly these photos, in this order."""
    by_path = {}
    for line in (root / rel).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_path[row["path"]] = row
    missing = [p for p in paths if p not in by_path]
    if missing:
        raise RuntimeError(f"{rel} lacks {len(missing)} of the photos being scored, e.g. {missing[0]}")
    return [by_path[p] for p in paths]


def baseline_rows(cfg: dict[str, Any], root: Path, paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    rec = json.loads((root / RUNS_REL / f"{cfg['baselines']['record']}.json").read_text(encoding="utf-8"))
    return {s: saved_rows(root, rec["predictions"][s], paths) for s in cfg["baselines"]["systems"]}


def verdict(comp: dict[str, Any]) -> str:
    lo, hi = comp["ci"]
    word = "improvement" if lo > 0 else "worse" if hi < 0 else "no measurable difference"
    return f"{comp['a']} vs {comp['b']} {comp['metric']}: {comp['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}], {word}"


def lab_energy(model, model_name: str, class_map: dict[str, int], cfg: dict[str, Any], config: TrainConfig,
               device: torch.device, rows: pd.DataFrame) -> list[float]:
    """Share of Grad-CAM energy outside the leaf on lab_dev photos (the official segmented copy is the leaf)."""
    model.to(device)
    scoring = score_split(model, target_layer(model, model_name), rows, class_map, "lab", config.batch_size,
                          config.num_workers)
    return [e for e in scoring.energy if e is not None]


# ----------------------------------------------------------------------------------------------------------------
# records


def arm_entries(cfg: dict[str, Any], root: Path, results: dict[str, dict[str, Any]], notrun: dict[str, Any],
                current: dict[str, Any]) -> list[dict[str, Any]]:
    """Every arm of the config as it stands, plus the dropped ones. `current` is the arm this record is for."""
    out = []
    for a in cfg["arms"]:
        entry: dict[str, Any] = {"arm": a["name"], "number": a["number"], "change": a["change"]}
        if a["name"] == current["arm"]:
            entry.update(current)
        elif a["name"] in results:
            r = results[a["name"]]
            entry.update(status="run", record=r["run_id"], field_dev_macro_f1=r["macro_f1"], base_arm=r["base_arm"])
        elif a["name"] in notrun:
            entry.update(status="not_run", reason=notrun[a["name"]]["reason"], inputs=notrun[a["name"]].get("inputs"))
        else:
            entry.update(status="pending", reason="not run at the time of this record")
        out.append(entry)
    out += [{"arm": d["name"], "status": "dropped", "reason": d["reason"]} for d in cfg["dropped"]]
    return out


def _plan_arm(cfg: dict[str, Any], root: Path, name: str, joint: pd.DataFrame, class_names: list[str],
              results: dict[str, dict[str, Any]], notrun: dict[str, Any]):
    """Apply the arm's stated rules to earlier results. Returns (settings, None) to train, or
    (None, (reason, inputs)) when the arm is not run. Raises Blocked when an earlier arm it reads is missing."""
    spec0 = arm_spec(cfg, name)
    if spec0.get("requires") == "arm_5":
        go, why = arm5_rule(results)
        if not go:
            return None, (why, {n: results[n]["macro_f1"] for n in ("dinov2_head", "convnext_joint")})
    if spec0["kind"] == "crop_of_best":
        for a in cfg["arms"]:
            if a["kind"] == "train" and a["name"] not in results and a["name"] not in notrun:
                raise Blocked(f"arm {a['name']} has neither a record nor a not-run entry; run it before best_with_crop")
    spec = resolved_spec(cfg, name, results)
    baseline = json.loads((root / RUNS_REL / f"{cfg['baselines']['record']}.json").read_text(encoding="utf-8"))
    if baseline["config"]["class_names"] != class_names:
        raise RuntimeError("the baseline record's classes differ from the joint manifest's")
    if spec0.get("requires") == "arm_3":
        lab_rows, _ = select_rows(joint, train_config(cfg, spec, root))
        share, kinds = mask_share(lab_rows[lab_rows["source"] == "lab"])
        need = cfg["rules"]["min_good_mask_share"]
        if share < need:
            return None, (f"only {share:.3f} of the capped lab_train photos have a believable "
                          f"leaf mask; the rule needs {need}", {"mask_kinds": kinds})
    reference = spec.get("reference")
    if reference and reference not in results:
        raise Blocked(f"reference arm {reference} has no record yet; run it first")
    return spec, None


def _train_arm(cfg: dict[str, Any], root: Path, name: str, spec: dict[str, Any], joint: pd.DataFrame,
               resume: bool, log_path: Path):
    """Train one arm (or resume it). Returns the train config, the rows used, the run output, the attempts and
    the history across all attempts."""
    crop_masks = ""
    if spec.get("crop"):
        plain = train_config(cfg, spec, root)
        train_rows, sel_rows = select_rows(joint, plain)
        crop_masks = str(ensure_crop_masks(list(train_rows["path"]) + list(sel_rows["path"]),
                                           root / cfg["e10"]["crop_mask_cache"], root / cfg["e10"]["checkpoint"],
                                           lambda m: print(m, flush=True)))
    tc = train_config(cfg, spec, root, crop_masks)
    train_rows, sel_rows = select_rows(joint, tc)

    latest = Path(tc.out_dir) / "latest.pt"
    if not resume and (latest.exists() or (Path(tc.out_dir) / "best.pt").exists()):
        raise FileExistsError(f"{tc.out_dir} has checkpoints from an earlier attempt; pass --resume to continue it")
    if resume and not latest.is_file():
        raise FileNotFoundError(f"--resume given but {latest} does not exist")
    attempts = start_attempt(cfg, root, name, resume)
    with log_path.open("a", encoding="utf-8") as logfile, contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
        print(f"arm {name}: train={len(train_rows)} select={len(sel_rows)} resume={resume}", flush=True)
        # an error leaves the attempt "started"; --resume marks it interrupted
        out = train(tc, resume=latest if resume else None)
    finish_attempt(cfg, root, name, attempts, out.history, "complete")
    return tc, train_rows, sel_rows, out, attempts, [h for a in attempts for h in a["history"]]


def _field_rows(cfg: dict[str, Any], spec: dict[str, Any], shared: set[str]):
    """The field_dev rows an arm is scored on, and the subset name for the record."""
    field = load_rows(cfg["split"], shared)
    held = spec.get("held_out_source")
    if not held:
        return field, cfg["subset"]
    manifest = load_manifest(cfg["field_manifest"])
    source = dict(zip(manifest["path"], manifest["source"], strict=True))
    return field[field["path"].map(source) == held].reset_index(drop=True), f"source:{held}"


def _measure_energy(cfg, name, spec, tc, model, class_map, device, shared, root, measurements) -> str:
    """Add Grad-CAM energy samples for the arm and E1 to `measurements`; returns the sentence for the notes."""
    if tc.model not in ENERGY_MODELS or spec.get("crop"):
        why = ("the input is cropped, so the leaf mask would not line up" if spec.get("crop")
               else "a ViT has no convolutional Grad-CAM layer")
        return f" Grad-CAM energy outside the leaf was not measured for this arm: {why}."
    lab = load_rows("lab_dev", shared)
    lab = lab.sample(n=min(cfg["cam_sample"], len(lab)), random_state=cfg["seed"]).reset_index(drop=True)
    e1 = load_checkpoint(root / cfg["e1_checkpoint"])
    e1_model = build_model(e1["config"].model, len(e1["class_map"]), pretrained=False)
    e1_model.load_state_dict(e1["model"])
    for label, m, mname, mmap in ((name, model, tc.model, class_map), ("e1", e1_model, e1["config"].model, e1["class_map"])):
        vals = lab_energy(m, mname, mmap, cfg, tc, device, lab)
        if vals:
            measurements[f"energy_outside_leaf_{label}"] = vals
    return (f" Grad-CAM energy outside the leaf: {len(lab)} lab_dev photos sampled with seed {cfg['seed']}, "
            f"the same for e1 and {name}.")


def _crop_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"masks": "sprout.segment.model.predict_mask with models/e10/best.pt, cached at 320 px",
            "e10_record": cfg["e10"]["record"],
            "caveat": "E10 was trained on PlantSeg-style masks only; how well it carries over to PlantDoc "
                      "and lab photos is untested"}


def _score_arm(cfg, root, name, reference, results, field, class_map, class_names, probs):
    """Rows of the arm, its baselines and its reference; their metrics and intervals; the paired comparisons."""
    paths = list(field["path"])
    labels = np.array([class_map[x] for x in field["label"]], dtype=int)
    systems: dict[str, list[dict[str, Any]]] = baseline_rows(cfg, root, paths)
    if reference:
        systems[reference] = saved_rows(root, results[reference]["predictions"], paths)
    systems[name] = [{"path": p, "label": int(y), "probs": [float(v) for v in pr]}
                     for p, y, pr in zip(paths, labels, probs, strict=True)]
    groups = list(field["group"])
    arrays = {s: (np.array([r["label"] for r in rows], dtype=int), np.array([r["probs"] for r in rows], dtype=float))
              for s, rows in systems.items()}
    scored = {s: Scored(groups=groups, labels=y, probs=p) for s, (y, p) in arrays.items()}
    metrics = {s: standard_image_metrics(p, y, class_names) for s, (y, p) in arrays.items()}
    intervals = {s: {m: recompute_interval(m, sc) for m in INTERVALS} for s, sc in scored.items()}
    comparisons = []
    for other in systems:
        if other == name:
            continue
        for metric in COMPARED:
            got = recompute_comparison(scored[name], scored[other], metric)
            comparisons.append({"a": name, "b": other, "metric": metric, "diff": got["diff"], "ci": got["ci"],
                                "mcnemar_p": got["mcnemar_p"]})
    return systems, metrics, intervals, comparisons


def run_arm(cfg: dict[str, Any], name: str, root: Path = ROOT, *, resume: bool = False, note: str = "") -> Path | None:
    """Decide, train, score and write the record of one arm; None when the arm is recorded as not run."""
    if is_dirty(root):
        raise RuntimeError("working tree has uncommitted changes; commit first")
    started = datetime.now(timezone.utc).isoformat()
    out_root = root / cfg["out_dir"]
    check_disk(out_root, cfg["min_free_disk_gb"])
    arm_spec(cfg, name)
    results, notrun = arm_results(cfg, root), not_run_entries(cfg, root)
    log_path = out_root / name / "epochs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    joint = load_joint_manifest(TrainConfig(
        manifest=cfg["lab_manifest"], out_dir="", field_manifest=cfg["field_manifest"],
        train_splits=("lab_train", "field_train"), select_split="field_dev"))
    class_map = make_class_map(joint)
    class_names = sorted(class_map, key=class_map.get)

    spec, skipped = _plan_arm(cfg, root, name, joint, class_names, results, notrun)
    if skipped:
        return _record_not_run(cfg, root, name, *skipped)
    tc, train_rows, sel_rows, out, attempts, history = _train_arm(cfg, root, name, spec, joint, resume, log_path)
    on_plantdoc = bool((train_rows["source"] == "plantdoc").any())

    # score on field_dev
    ckpt = load_checkpoint(out.best_path)
    assert ckpt["class_map"] == class_map
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra = {"unfreeze_blocks": tc.unfreeze_blocks} if tc.unfreeze_blocks else {}
    model = build_model(tc.model, len(class_map), pretrained=False, **extra)
    model.load_state_dict(ckpt["model"])

    shared = shared_classes()
    field, subset = _field_rows(cfg, spec, shared)
    probs = predict_probs(model, field, class_map, tc, device)

    reference = spec.get("reference")
    systems, metrics, intervals, comparisons = _score_arm(cfg, root, name, reference, results, field, class_map,
                                                          class_names, probs)

    measurements: dict[str, list[float]] = {f"epoch_seconds_{name}": [h["seconds"] for h in history]}
    peaks = [float(h["peak_mem_allocated"]) for h in history if h.get("peak_mem_allocated") is not None]
    if peaks:
        measurements[f"peak_gpu_mem_bytes_{name}"] = peaks
    energy_note = _measure_energy(cfg, name, spec, tc, model, class_map, device, shared, root, measurements)

    current = {
        "arm": name, "status": "run", "setting": {k: v for k, v in spec.items()}, "train_config": tc.to_dict(),
        "best_dev_accuracy": out.best_metric, "epochs_done": out.epochs_done, "history": history,
        "attempts": attempts, "trained_on_plantdoc": on_plantdoc,
        "train_rows": {"total": len(train_rows), "lab": int((train_rows["source"] == "lab").sum()),
                       "plantdoc": int((train_rows["source"] == "plantdoc").sum()),
                       "plantseg": int((train_rows["source"] == "plantseg").sum())},
        "selection_rows": len(sel_rows),
    }
    attempts_list = arm_entries(cfg, root, results, notrun, current)
    config = {
        "arm": name, "base_arm": spec.get("base_arm"), "class_names": class_names, "e4": cfg, "train": tc.to_dict(),
        "trained_on_plantdoc": on_plantdoc, "lab_cap": cfg["lab_cap"], "lab_cap_reason": cfg["lab_cap_reason"],
        "baseline_record": cfg["baselines"]["record"], "reference": reference, "subset_rule": subset,
        "shared_classes": sorted(shared), "label": "development number, field_dev only",
        "selected_on": "field_dev (optimistic)", "started_at": started,
    }
    if spec.get("crop"):
        config["crop"] = _crop_config(cfg)
    models = {s: f"{s}: saved predictions of E3 record {cfg['baselines']['record']}" for s in cfg["baselines"]["systems"]}
    if reference:
        models[reference] = f"{reference}: arm record {results[reference]['run_id']}"
    models[name] = f"{name}: {tc.model} trained here ({spec['change']})"
    held = spec.get("held_out_source")
    notes = (
        f"{CAVEAT_DEV} PlantDoc is cleared for local use only; weights trained on it: {on_plantdoc}. "
        f"Lab photos are capped at {cfg['lab_cap']} per class. field_test and lab_test were not read."
        + energy_note + " Paired comparisons: " + "; ".join(verdict(c) for c in comparisons)
        + (f" Held-out run: {held} was left out of training and selection; scored on its field_dev part." if held else "")
        + (f" {note}" if note else "")
    )
    return write_run(
        cfg["experiment"], cfg["split"], config, models, metrics, intervals, attempts_list, systems,
        dataset_hash(load_manifest(cfg["field_manifest"])), root=root, notes=notes, subset=subset,
        comparisons=comparisons, measurements=measurements,
    )


def _record_not_run(cfg: dict[str, Any], root: Path, name: str, reason: str, inputs: dict[str, Any]) -> None:
    d = not_run_dir(cfg, root)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / f"{name}.json", json.dumps({"arm": name, "reason": reason, "inputs": inputs,
                                                      "decided_at": datetime.now(timezone.utc).isoformat()}, indent=1) + "\n")
    print(f"arm {name} NOT RUN: {reason}", flush=True)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e4.json")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from models/e4/<arm>/latest.pt")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        path = run_arm(cfg, a.arm, ROOT, resume=a.resume, note=a.note)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(path if path else "no record (arm not run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
