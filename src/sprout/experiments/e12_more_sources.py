"""E12: E4's best configuration (convnext_joint) trained again with the extra real photos added.

Run from the repo root with PYTHONPATH set to its src folder. Settings and the decision rule are in configs/e12.json,
committed before any run. One command per run, each on the GPU (train, then score field_dev):
    .venv/Scripts/python -m sprout.experiments.e12_more_sources run --run convnext_joint_extra [--resume]
    .venv/Scripts/python -m sprout.experiments.e12_more_sources run --run heldout_plantdoc_extra [--resume]
    .venv/Scripts/python -m sprout.experiments.e12_more_sources run --run heldout_plantseg_extra [--resume]
    .venv/Scripts/python -m sprout.experiments.e12_more_sources decide          # CPU: applies the rule
The gate runs and the gates stage (configs/e12_gates.json, docs/DECISIONS.md) take the same commands with
`--config configs/e12_gates.json`: `run --run <name>` for each of its seven runs, then `gates`.
`--num-workers N` or SPROUT_NUM_WORKERS sets the loader workers (default 2, at most 4). A run that stops early
writes no record; continue it with --resume (same worker count), which restarts from latest.pt and lists both
attempts. Each run writes one record on field_dev scored on the rows kept by the label cleaning (clean:1), beside
E4's saved rows for the same photos. field_test and lab_test are never read.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sprout.atomic import atomic_write_text
from sprout.data.splits import dataset_hash, load_manifest
from sprout.eval.check_run import (
    Scored,
    check_file,
    recompute_comparison,
    recompute_interval,
    records_for,
)
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    is_dirty,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)
from sprout.experiments import e4_improve as e4
from sprout.train.checkpoint import load_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import make_class_map
from sprout.train.joint import load_joint_manifest, select_rows
from sprout.train.loop import build_model, train

SPLIT = "field_dev"
INTERVALS = ("accuracy", "macro_f1")
COMPARED = ("accuracy", "macro_f1")
PREDICTIONS_FILE = "field_dev.predictions.jsonl"
CAVEAT = (
    "Development numbers on field_dev only. field_dev also chose E4's model and its best epoch is picked on field_dev "
    "(for held-out runs, on the other source's part only), so every number here is optimistic. One run per row, one seed."
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "base" in cfg:
        return load_gates_config(Path(path), cfg)
    if cfg["split"] != SPLIT:
        raise ValueError(f"E12 scores {SPLIT} only, not {cfg['split']!r}")
    if any(s.endswith("_test") for s in (*cfg["settings"]["train_splits"], cfg["settings"]["select_split"])):
        raise ValueError("E12 never reads a test split")
    names = [r["name"] for r in cfg["runs"]]
    if len(set(names)) != len(names) or set(names) != set(cfg["e4"]["records"]):
        raise ValueError("run names must be unique and each needs an E4 record")
    if not 0 < cfg["num_workers"]["default"] <= cfg["num_workers"]["max"]:
        raise ValueError("num_workers default must be between 1 and max")
    return cfg


def load_gates_config(path: Path, gates: dict[str, Any]) -> dict[str, Any]:
    """configs/e12_gates.json on top of its base (configs/e12.json): the same settings and rules, the gate runs in
    place of the three first runs, its loader worker rule, and the whole gate file under `gates`."""
    cfg = load_config(path.with_name(gates["base"]))
    names = [r["name"] for r in gates["runs"]]
    if len(set(names)) != len(names):
        raise ValueError("gate run names must be unique")
    for r in gates["runs"]:
        if r["compare"] and "e4_record" not in r:
            raise ValueError(f"{r['name']} compares with E4's saved rows and needs an e4_record")
        if r.get("paired_with") and r["paired_with"] not in names:
            raise ValueError(f"{r['name']} is paired with {r['paired_with']!r}, which is not a gate run")
    spec = gates["num_workers"]
    if not spec["min"] <= spec["default"] <= spec["max"]:
        raise ValueError("num_workers default must be between min and max")
    return {**cfg, "runs": gates["runs"], "num_workers": spec, "gates": gates}


def resolve_workers(cfg: dict[str, Any], flag: int | None = None, env: dict[str, str] | None = None) -> int:
    """The flag, else the environment variable, else the config default; never above the config's maximum."""
    spec = cfg["num_workers"]
    env = os.environ if env is None else env
    value = flag if flag is not None else int(env[spec["env"]]) if spec["env"] in env else spec["default"]
    low = spec.get("min", 0)
    if not low <= value <= spec["max"]:
        raise ValueError(f"num_workers is {value}; it must be between {low} and {spec['max']}")
    return value


def run_spec(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for r in cfg["runs"]:
        if r["name"] == name:
            return r
    raise ValueError(f"unknown run {name!r}; runs are {[r['name'] for r in cfg['runs']]}")


def run_dir(cfg: dict[str, Any], spec: dict[str, Any], root: Path) -> Path:
    return root / spec.get("out_dir", cfg["out_dir"]) / spec["name"]


def train_config(cfg: dict[str, Any], spec: dict[str, Any], root: Path, num_workers: int) -> TrainConfig:
    """A run may set its own seed, out_dir and extra manifest, or train with no extras (E4's arm again)."""
    s = cfg["settings"]
    extras = spec.get("extras", True)
    return TrainConfig(
        manifest=cfg["lab_manifest"],
        out_dir=str(run_dir(cfg, spec, root)),
        model=s["model"],
        seed=spec.get("seed", s["seed"]),
        lr=s["lr"],
        batch_size=s["batch_size"],
        epochs=s["epochs"],
        transforms=tuple(s["transforms"]),
        num_workers=num_workers,
        field_manifest=cfg["field_manifest"],
        extra_manifest=spec.get("extra", cfg["extra"])["manifest"] if extras else "",
        train_splits=tuple(s["train_splits"]) if extras else tuple(x for x in s["train_splits"] if x != "extra_train"),
        select_split=s["select_split"],
        lab_cap=s["lab_cap"],
        held_out_source=spec["held_out_source"],
    )


def check_extra_frozen(cfg: dict[str, Any]) -> None:
    extra = load_manifest(cfg["extra"]["manifest"])
    if len(extra) != cfg["extra"]["rows"] or dataset_hash(extra) != cfg["extra"]["dataset_hash"]:
        raise RuntimeError(f"{cfg['extra']['manifest']}.csv differs from the frozen extra rows in configs/e12.json")


def check_run_extras(cfg: dict[str, Any], spec: dict[str, Any], root: Path) -> None:
    """The frozen extras for a run that trains with them; for a trimmed run, also that its manifest is those extras
    without the committed removal list, and that the list reproduced the verifier's counts."""
    if not spec.get("extras", True):
        return
    check_extra_frozen(cfg)
    if "extra" not in spec:
        return
    if not (root / spec["extra"]["list"]).is_file():
        raise RuntimeError(f"{spec['extra']['list']} does not exist; make and commit it with tools/extra_trim.py first")
    listed = json.loads((root / spec["extra"]["list"]).read_text(encoding="utf-8"))
    trimmed = load_manifest(spec["extra"]["manifest"])
    gone = {e["path"] for e in listed["remove"]}
    whole = load_manifest(cfg["extra"]["manifest"])
    if listed["extra_dataset_hash"] != cfg["extra"]["dataset_hash"] or not listed["reproduces_verifier"]:
        raise RuntimeError(f"{spec['extra']['list']} was not made from the frozen extras or does not reproduce the verifier's counts")
    if (len(trimmed) != listed["trimmed_rows"] or dataset_hash(trimmed) != listed["trimmed_dataset_hash"]
            or set(whole["path"]) - gone != set(trimmed["path"])):
        raise RuntimeError(f"{spec['extra']['manifest']}.csv is not the extras minus {spec['extra']['list']}")


# ----------------------------------------------------------------------------------------------------------------
# the rows scored

def dev_rows(cfg: dict[str, Any], spec: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """The field_dev rows predicted (the run's subset, every photo) and those scored (minus the cleaning's
    exclusions), both in the order check_run reads them, and the two subset names."""
    info = split_info(SPLIT, root)
    all_name = spec["subset"]
    clean_name = f"{all_name}+clean:{cfg['clean_version']}"
    dev = pd.read_csv(root / "datasets" / "manifests" / "field.csv")
    dev = dev[dev["split"] == SPLIT].set_index("path", drop=False)
    pick = lambda name: dev.loc[[info.ids[i] for i in subset_positions(name, SPLIT, info, root)]].reset_index(drop=True)  # noqa: E731
    return pick(all_name), pick(clean_name), all_name, clean_name


def saved_e4_rows(cfg: dict[str, Any], root: Path, spec: dict[str, Any], paths: list[str],
                  class_names: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """E4's saved rows for exactly these photos: the systems this run is compared with."""
    record = spec.get("e4_record") or cfg["e4"]["records"][spec["name"]]
    rec = json.loads((root / RUNS_REL / f"{record}.json").read_text(encoding="utf-8"))
    if rec["split"] != SPLIT or rec["dataset_hash"] != split_info(SPLIT, root).hash:
        raise RuntimeError(f"{rec['run_id']} is not a {SPLIT} record on the current field.csv")
    if rec["config"]["class_names"] != class_names:
        raise RuntimeError(f"{rec['run_id']} has different classes from the joint manifest with the extra rows")
    return {s: e4.saved_rows(root, rec["predictions"][s], paths) for s in spec["compare"]}, rec


def paired_rows(cfg: dict[str, Any], root: Path, spec: dict[str, Any], paths: list[str], class_names: list[str],
                tc: TrainConfig) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """The saved rows of the E4 run this run is paired with (same seed, same photos), from its own record."""
    other = spec["paired_with"]
    found = newest_records(cfg, root).get(other)
    if found is None:
        raise RuntimeError(f"{other} has no record yet; run it before {spec['name']}")
    rec = found[1]
    if rec["config"]["class_names"] != class_names:
        raise RuntimeError(f"{rec['run_id']} has different classes from this run")
    theirs = rec["config"]["train"]
    if (theirs["seed"], theirs["num_workers"]) != (tc.seed, tc.num_workers):
        raise RuntimeError(f"{rec['run_id']} trained with seed {theirs['seed']} and {theirs['num_workers']} workers, "
                           f"not {tc.seed} and {tc.num_workers}; a pair must match")
    return {other: e4.saved_rows(root, rec["predictions"][other], paths)}, rec


def score_run(name: str, spec: dict[str, Any], rows: pd.DataFrame, probs: np.ndarray, others: dict[str, list],
              class_map: dict[str, int], class_names: list[str]):
    """Rows, metrics, intervals and paired comparisons of the new run and the E4 systems on the same photos."""
    labels = np.array([class_map[x] for x in rows["label"]], dtype=int)
    systems: dict[str, list[dict[str, Any]]] = dict(others)
    systems[name] = [{"path": p, "label": int(y), "probs": [float(v) for v in pr]}
                     for p, y, pr in zip(rows["path"], labels, probs, strict=True)]
    groups = list(rows["group"])
    arrays = {s: (np.array([r["label"] for r in rs], dtype=int), np.array([r["probs"] for r in rs], dtype=float))
              for s, rs in systems.items()}
    for s, (y, _) in arrays.items():
        if not np.array_equal(y, labels):
            raise RuntimeError(f"{s}: saved labels differ from the manifest's for these photos")
    scored = {s: Scored(groups=groups, labels=y, probs=p) for s, (y, p) in arrays.items()}
    metrics = {s: standard_image_metrics(p, y, class_names) for s, (y, p) in arrays.items()}
    intervals = {s: {m: recompute_interval(m, sc) for m in INTERVALS} for s, sc in scored.items()}
    comparisons = []
    for other in others:
        for metric in COMPARED:
            got = recompute_comparison(scored[name], scored[other], metric)
            comparisons.append({"a": name, "b": other, "metric": metric, "diff": got["diff"], "ci": got["ci"],
                                "mcnemar_p": got["mcnemar_p"]})
    return systems, metrics, intervals, comparisons


def device_info(device: torch.device, num_workers: int, e4_workers: int) -> dict[str, Any]:
    on_gpu = device.type == "cuda"
    return {"type": device.type, "name": torch.cuda.get_device_name(0) if on_gpu else "cpu",
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory if on_gpu else None,
            "amp": on_gpu, "num_workers": num_workers, "e4_num_workers": e4_workers}


# ----------------------------------------------------------------------------------------------------------------
# one run


def _train(cfg: dict[str, Any], root: Path, name: str, tc: TrainConfig, joint: pd.DataFrame, resume: bool):
    latest, best = Path(tc.out_dir) / "latest.pt", Path(tc.out_dir) / "best.pt"
    if not resume and (latest.exists() or best.exists()):
        raise FileExistsError(f"{tc.out_dir} has checkpoints from an earlier attempt; pass --resume to continue it")
    if resume:
        if not latest.is_file():
            raise FileNotFoundError(f"--resume given but {latest} does not exist")
        started_with = load_checkpoint(latest)["config"].num_workers
        if started_with != tc.num_workers:
            raise ValueError(f"the run started with {started_with} workers; resume with the same count "
                             "(the augmentation draws depend on it)")
    train_rows, sel_rows = select_rows(joint, tc)
    attempts = e4.start_attempt(cfg, root, name, resume)
    log_path = Path(tc.out_dir) / "epochs.log"
    with log_path.open("a", encoding="utf-8") as logfile, contextlib.redirect_stdout(e4.Tee(sys.stdout, logfile)):
        print(f"run {name}: train={len(train_rows)} select={len(sel_rows)} workers={tc.num_workers} resume={resume} "
              f"device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
        out = train(tc, resume=latest if resume else None)  # an error leaves the attempt "started"
    e4.finish_attempt(cfg, root, name, attempts, out.history, "complete")
    return train_rows, sel_rows, out, attempts, [h for a in attempts for h in a["history"]]


def run(cfg: dict[str, Any], name: str, root: Path = ROOT, *, resume: bool = False, num_workers: int | None = None,
        note: str = "") -> Path:
    """Train one run (or resume it), score it on field_dev, and write its record."""
    if is_dirty(root):
        raise RuntimeError("working tree has uncommitted changes; commit first")
    started = datetime.now(timezone.utc).isoformat()
    spec = run_spec(cfg, name)
    workers = resolve_workers(cfg, num_workers)
    check_run_extras(cfg, spec, root)
    out_root = root / spec.get("out_dir", cfg["out_dir"])
    e4.check_disk(out_root, cfg["min_free_disk_gb"])
    (out_root / name).mkdir(parents=True, exist_ok=True)

    tc = train_config(cfg, spec, root, workers)
    joint = load_joint_manifest(tc)
    class_map = make_class_map(joint)
    class_names = sorted(class_map, key=class_map.get)

    all_rows, rows, all_name, subset = dev_rows(cfg, spec, root)
    others, e4_rec = {}, None
    if spec["compare"]:
        others, e4_rec = saved_e4_rows(cfg, root, spec, list(rows["path"]), class_names)
    pair = None
    if spec.get("paired_with"):
        paired, pair = paired_rows(cfg, root, spec, list(rows["path"]), class_names, tc)
        others |= paired
    train_rows, sel_rows, out, attempts, history = _train({**cfg, "out_dir": str(out_root.relative_to(root))}, root, name,
                                                          tc, joint, resume)

    ckpt = load_checkpoint(out.best_path)
    if ckpt["class_map"] != class_map:
        raise RuntimeError("the checkpoint's classes differ from the joint manifest's")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(tc.model, len(class_map), pretrained=False)
    model.load_state_dict(ckpt["model"])
    all_probs = e4.predict_probs(model, all_rows, class_map, tc, device)
    atomic_write_text(Path(tc.out_dir) / PREDICTIONS_FILE, "".join(
        json.dumps({"path": p, "label": class_map[y], "probs": [float(v) for v in pr]}) + "\n"
        for p, y, pr in zip(all_rows["path"], all_rows["label"], all_probs, strict=True)))
    keep = all_rows["path"].isin(set(rows["path"])).to_numpy()
    probs = all_probs[keep]
    if list(all_rows["path"][keep]) != list(rows["path"]):
        raise RuntimeError("the scored rows are not in the order check_run reads them")

    systems, metrics, intervals, comparisons = score_run(name, spec, rows, probs, others, class_map, class_names)
    measurements: dict[str, list[float]] = {f"epoch_seconds_{name}": [h["seconds"] for h in history]}
    for key, label in (("peak_mem_allocated", "peak_gpu_mem_bytes"), ("peak_mem_reserved", "peak_gpu_reserved_bytes")):
        vals = [float(h[key]) for h in history if h.get(key) is not None]
        if vals:
            measurements[f"{label}_{name}"] = vals

    train_by_source = train_rows["source"].value_counts().to_dict()
    current = {
        "run": name, "status": "run", "setting": {**cfg["settings"], "held_out_source": spec["held_out_source"]},
        "train_config": tc.to_dict(), "best_dev_accuracy": out.best_metric, "epochs_done": out.epochs_done,
        "history": history, "attempts": attempts, "selection_rows": len(sel_rows),
        "train_rows": {"total": len(train_rows), **{k: int(v) for k, v in sorted(train_by_source.items())}},
    }
    held = spec["held_out_source"]
    on_plantdoc = bool((train_rows["source"] == "plantdoc").any())
    config = {
        "run": name, "class_names": class_names, "e12": cfg, "train": tc.to_dict(), "subset_rule": subset,
        "predicted_subset": all_name, "e4_record": e4_rec["run_id"] if e4_rec else None, "e4_compared": spec["compare"],
        "paired_record": pair["run_id"] if pair else None, "seed": tc.seed,
        "extra_dataset_hash": cfg["extra"]["dataset_hash"] if tc.extra_manifest else None,
        "extra_manifest": tc.extra_manifest,
        "extra_rows_in_training": int((train_rows["split"] == "extra_train").sum()),
        "trained_on_plantdoc": on_plantdoc, "device": device_info(device, workers, cfg["num_workers"]["e4"]),
        "field_test_read": False, "label": "development number, field_dev only", "selected_on": "field_dev (optimistic)",
        "started_at": started,
    }
    models = {s: f"{s}: saved predictions of E4 record {e4_rec['run_id']}, filtered to the cleaned rows" for s in spec["compare"]}
    if pair:
        models[spec["paired_with"]] = f"{spec['paired_with']}: seed {tc.seed} run of record {pair['run_id']}"
    models[name] = f"{name}: {tc.model} trained here ({spec['change']})"
    verdicts = "; ".join(e4.verdict(c) for c in comparisons)
    notes = (f"{CAVEAT} PlantDoc is cleared for local use only; weights trained on it: {on_plantdoc}. Scored on {subset} "
             f"({len(rows)} photos); E4's systems are its saved rows for the same photos. Loader workers: {workers} "
             f"(E4 used {cfg['num_workers']['e4']}). field_test and lab_test were not read. Paired: {verdicts}."
             + (f" Held-out run: {held} left out of training and selection." if held else "")
             + (f" {note}" if note else ""))
    return write_run(cfg["experiment"], SPLIT, config, models, metrics, intervals, [current], systems, split_info(SPLIT, root).hash,
                     root=root, notes=notes, subset=subset, comparisons=comparisons, measurements=measurements)


# ----------------------------------------------------------------------------------------------------------------
# the decision


def newest_records(cfg: dict[str, Any], root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    found = {}
    for path in records_for(cfg["experiment"], root):  # oldest first, so a later record wins
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["config"].get("run") in {r["name"] for r in cfg["runs"]}:
            found[rec["config"]["run"]] = (path, rec)
    return found


def _comparison(rec: dict[str, Any], a: str, b: str, metric: str) -> dict[str, Any]:
    for c in rec["comparisons"]:
        if (c["a"], c["b"], c["metric"]) == (a, b, metric):
            return c
    raise RuntimeError(f"{rec['run_id']} has no {metric} comparison of {a} with {b}")


def apply_rule(cfg: dict[str, Any], seen: dict[str, Any], held: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The fixed rule of configs/e12.json applied to the records' own paired comparisons."""
    rule, metric = cfg["decision"], cfg["decision"]["metric"]
    c = _comparison(seen, "convnext_joint_extra", "convnext_joint", metric)
    seen_ok = c["ci"][0] > 0
    margin = rule["heldout_margin"]
    sources = {}
    for name, rec in held.items():
        h = _comparison(rec, name, name.removesuffix("_extra"), metric)
        sources[name] = {"diff": h["diff"], "ci": h["ci"], "ok": h["diff"] >= -margin}
    helps = seen_ok and all(v["ok"] for v in sources.values())
    return {"metric": metric, "seen": {"diff": c["diff"], "ci": c["ci"], "ok": seen_ok}, "heldout": sources,
            "heldout_margin": margin, "clearly_helps": helps}


def _metrics_by_source(rows: list[dict[str, Any]], sources: pd.Series, class_names: list[str]) -> dict[str, dict[str, Any]]:
    """Accuracy and macro-F1 of saved rows, on all of them and on each source's part."""
    out = {}
    for label, mask in (("all", np.ones(len(rows), bool)), *((s, (sources == s).to_numpy()) for s in sorted(set(sources)))):
        if mask.any():
            y = np.array([r["label"] for r in rows], dtype=int)[mask]
            p = np.array([r["probs"] for r in rows], dtype=float)[mask]
            m = standard_image_metrics(p, y, class_names)
            out[label] = {"n": int(mask.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    return out


def side_by_side(cfg: dict[str, Any], recs: dict[str, dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """E4 and E12 on the same cleaned rows: seen-source (all, then each source's part) and held-out."""
    field = pd.read_csv(root / "datasets" / "manifests" / "field.csv").drop_duplicates("path").set_index("path")["source"]
    table = []
    seen = recs["convnext_joint_extra"]
    names = seen["config"]["class_names"]
    for system, label in (("convnext_joint", "E4 seen-source"), ("convnext_joint_extra", "E12 seen-source")):
        rows = [json.loads(x) for x in (root / seen["predictions"][system]).read_text(encoding="utf-8").splitlines() if x.strip()]
        sources = pd.Series([field[r["path"]] for r in rows])
        for part, m in _metrics_by_source(rows, sources, names).items():
            table.append({"model": label, "scored_on": part, **m})
    for name, rec in recs.items():
        if name == "convnext_joint_extra":
            continue
        base = name.removesuffix("_extra")
        source = base.removeprefix("heldout_")
        for system, label in ((base, "E4 held-out"), (name, "E12 held-out")):
            rows = [json.loads(x) for x in (root / rec["predictions"][system]).read_text(encoding="utf-8").splitlines() if x.strip()]
            m = _metrics_by_source(rows, pd.Series([source] * len(rows)), names)["all"]
            table.append({"model": label, "scored_on": source, **m})
    return table


def per_class_focus(cfg: dict[str, Any], seen: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Apple and corn classes: E4 against E12 on the seen-source record, with the extras each class received."""
    extra = load_manifest(cfg["extra"]["manifest"])
    train = load_manifest(cfg["field_manifest"])
    train = train[train["split"] == "field_train"]
    out = []
    for cls, new in seen["per_class"]["convnext_joint_extra"].items():
        if not cls.startswith(tuple(cfg["focus_classes"])):
            continue
        old = seen["per_class"]["convnext_joint"][cls]
        mine = extra[extra["label"] == cls]
        out.append({"class": cls, "support": new["support"],
                    "e4": {k: old[k] for k in ("precision", "recall", "f1")},
                    "e12": {k: new[k] for k in ("precision", "recall", "f1")},
                    "f1_change": new["f1"] - old["f1"], "extra_rows": len(mine),
                    "extra_by_source": {k: int(v) for k, v in mine["source"].value_counts().items()},
                    "field_train_rows": int((train["label"] == cls).sum())})
    return out


def decide(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    """Apply the rule to the three newest records, each of which must pass check_run, and write the decision."""
    recs = newest_records(cfg, root)
    missing = [r["name"] for r in cfg["runs"] if r["name"] not in recs]
    if missing:
        raise RuntimeError(f"no record yet for {missing}; run them first")
    for name, (path, _) in recs.items():
        problems = check_file(path, root)
        if problems:
            raise RuntimeError(f"{path.name} does not pass check_run, so it cannot decide anything: {problems[0]}")
    plain = {n: r for n, (_, r) in recs.items()}
    outcome = apply_rule(cfg, plain["convnext_joint_extra"], {n: r for n, r in plain.items() if n != "convnext_joint_extra"})
    chosen = (f"{cfg['out_dir']}/convnext_joint_extra/best.pt" if outcome["clearly_helps"]
              else cfg["e4"]["model"])
    if not (root / chosen).is_file():
        raise RuntimeError(f"the chosen model file {chosen} does not exist")
    decision = {
        "experiment": cfg["experiment"], "decided_at": datetime.now(timezone.utc).isoformat(),
        "records": {n: r["run_id"] for n, r in plain.items()}, "rule": cfg["decision"]["rule"], **outcome,
        "chosen_model": chosen, "chosen_model_sha256": hashlib.sha256((root / chosen).read_bytes()).hexdigest(),
        "class_names": plain["convnext_joint_extra"]["config"]["class_names"],
        "refit_e5": outcome["clearly_helps"],
        "refit_e5_how": cfg["decision"]["if_helps"] if outcome["clearly_helps"] else cfg["decision"]["otherwise"],
        "e12_field_dev_probabilities": f"{cfg['out_dir']}/convnext_joint_extra/{PREDICTIONS_FILE}",
        "side_by_side": side_by_side(cfg, plain, root),
        "per_class_focus": per_class_focus(cfg, plain["convnext_joint_extra"], root),
        "label": "development numbers, field_dev only",
    }
    path = root / cfg["decision_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(decision, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e12.json")
    sub = ap.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--resume", action="store_true", help="continue from models/e12/<run>/latest.pt")
    r.add_argument("--num-workers", type=int, default=None)
    r.add_argument("--note", default="")
    sub.add_parser("decide")
    sub.add_parser("gates", help="with --config configs/e12_gates.json: writes results/e12/gates.json")
    sub.add_parser("g3", help="with --config configs/e12_gates.json: G3 alone, writes results/e12/g3.json")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        if a.stage == "run":
            print(run(cfg, a.run, ROOT, resume=a.resume, num_workers=a.num_workers, note=a.note))
        elif a.stage == "g3":
            if "gates" not in cfg:
                raise ValueError("the g3 stage needs --config configs/e12_gates.json")
            from sprout.experiments.e12_gates import g3_only

            print(g3_only(cfg, ROOT))
        elif a.stage == "gates":
            if "gates" not in cfg:
                raise ValueError("the gates stage needs --config configs/e12_gates.json")
            from sprout.experiments.e12_gates import gates

            print(gates(cfg, ROOT))
        else:
            print(decide(cfg, ROOT))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
