"""E5: does calibration plus abstention cut confident errors, and which input check works best?

Run from the repo root with PYTHONPATH set to its src folder, one stage per command (configs/e5.json, committed
before any run, holds every setting and rule):
    .venv/Scripts/python -m sprout.experiments.e5_confidence --stage <stage>
  probs-not-plant   GPU   convnext_joint on not_plant_dev (1207 photos) -> results/e5/not_plant_dev.probs.jsonl
  green             CPU   the old green-pixel share of every field_dev and not_plant_dev photo
                          -> results/e5/green_ratio.jsonl
  fit               CPU   temperature, abstention threshold and gate settings from the fitting part of each dev
                          split -> results/e5/chosen.json
  record            CPU   scores the scoring part and writes one E5 record per split (field_dev, not_plant_dev)
Only probs-not-plant needs the GPU, so only it needs a GPU turn. field_dev probabilities are not recomputed: they are
read from the E4 record named in the config. Commit results/e5 after each stage; `record` refuses a dirty tree.
Only dev splits are ever read. Each stage logs seconds (and peak GPU memory) to results/e5/stages.log, flushed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sprout.atomic import atomic_write_text
from sprout.eval import check_run
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    scoring_part,
    split_info,
    standard_image_metrics,
    write_run,
)
from sprout.serve.calibrate import apply_temperature, choose_threshold, fit_temperature
from sprout.serve.gate import (
    GREEN_THRESHOLD,
    KINDS,
    GateSettings,
    green_ratio,
    normalized_entropy,
)

MAX_WORKERS = 8
DEV_SPLITS = ("field_dev", "not_plant_dev")
FLOOR = 1e-12


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["num_workers"] > MAX_WORKERS:
        raise ValueError(f"num_workers is {cfg['num_workers']}; at most {MAX_WORKERS}")
    for split in (cfg["field_split"], cfg["not_plant_split"]):
        require_dev(split)
    return cfg


def require_dev(split: str) -> None:
    """E5 chooses settings; only dev splits may be read. Test and train splits are refused."""
    if split not in DEV_SPLITS:
        raise ValueError(f"E5 reads only {DEV_SPLITS}, not {split!r}")


def out_dir(cfg: dict[str, Any], root: Path) -> Path:
    return root / cfg["out_dir"]


class StageLog:
    """Prints and appends to results/e5/stages.log, flushing every line."""

    def __init__(self, cfg: dict[str, Any], root: Path):
        self.path = out_dir(cfg, root) / "stages.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows))


@dataclass
class Dev:
    """One dev split in manifest order. `labels` and `class_names` are None for not_plant_dev."""

    split: str
    hash: str
    paths: list[str]
    groups: list[str]
    probs: np.ndarray  # uncalibrated model outputs
    green: np.ndarray
    labels: np.ndarray | None = None
    class_names: list[str] | None = None

    def scoring(self, cfg: dict[str, Any]) -> np.ndarray:
        return scoring_part(self.groups, cfg["seed"], cfg["fit_percent"])


def _by_path(rows: list[dict[str, Any]], paths: list[str], what: str) -> list[dict[str, Any]]:
    index = {r["path"]: r for r in rows}
    missing = [p for p in paths if p not in index]
    if missing:
        raise RuntimeError(f"{what} lacks {len(missing)} of the {len(paths)} photos, e.g. {missing[0]}")
    return [index[p] for p in paths]


def green_by_path(cfg: dict[str, Any], root: Path, paths: list[str]) -> np.ndarray:
    rows = _by_path(read_jsonl(out_dir(cfg, root) / "green_ratio.jsonl"), paths, "green_ratio.jsonl")
    return np.array([r["green_ratio"] for r in rows], dtype=float)


def load_field(cfg: dict[str, Any], root: Path) -> Dev:
    """field_dev with the probabilities saved in the E4 record, which must cover every photo with its right label.
    A refit on an E12 model (configs/e5_e12.json) names `model.field_dev_predictions_file` instead, because an E12
    record holds only the cleaned rows; the record still gives the classes and the manifest hash."""
    require_dev(cfg["field_split"])
    info = split_info(cfg["field_split"], root)
    record = json.loads((root / RUNS_REL / f"{cfg['model']['e4_record']}.json").read_text(encoding="utf-8"))
    if record["split"] != cfg["field_split"] or record["dataset_hash"] != info.hash:
        raise RuntimeError("the E4 record is not on this field_dev manifest")
    names = record["config"]["class_names"]
    saved = cfg["model"].get("field_dev_predictions_file") or record["predictions"][cfg["model"]["name"]]
    rows = _by_path(read_jsonl(root / saved), info.ids, "the saved predictions")
    labels = np.array([r["label"] for r in rows], dtype=int)
    if [names[x] for x in labels] != info.labels:
        raise RuntimeError("the E4 predictions' labels do not match the manifest")
    green = green_by_path(cfg, root, info.ids)
    return Dev(cfg["field_split"], info.hash, info.ids, info.groups, np.array([r["probs"] for r in rows], dtype=float),
               green, labels, names)


def load_not_plant(cfg: dict[str, Any], root: Path) -> Dev:
    require_dev(cfg["not_plant_split"])
    info = split_info(cfg["not_plant_split"], root)
    rows = _by_path(read_jsonl(out_dir(cfg, root) / "not_plant_dev.probs.jsonl"), info.ids, "not_plant_dev.probs.jsonl")
    green = green_by_path(cfg, root, info.ids)
    return Dev(cfg["not_plant_split"], info.hash, info.ids, info.groups, np.array([r["probs"] for r in rows], dtype=float),
               green)


def calibrate(raw: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probabilities from saved probabilities (their log stands in for logits)."""
    return apply_temperature(np.log(np.maximum(raw, FLOOR)), temperature)


# ---- stage: probs-not-plant (GPU) ----

class Photos:
    """Photos by path, preprocessed. Module level, so DataLoader workers (spawned on Windows) can unpickle it."""

    def __init__(self, paths: list[str], root: Path):
        self.paths, self.root = paths, root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from sprout.preprocess import preprocess

        with Image.open(self.root / self.paths[i]) as img:
            return preprocess(img)


def score_images(model, paths: list[str], root: Path, batch_size: int, num_workers: int, device) -> np.ndarray:
    """Softmax outputs of the model for each photo, through sprout.preprocess."""
    import torch
    from torch.utils.data import DataLoader

    model.to(device).eval()
    out = []
    with torch.no_grad():
        for x in DataLoader(Photos(paths, root), batch_size=batch_size, num_workers=num_workers):
            out.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def stage_probs_not_plant(cfg: dict[str, Any], root: Path, log: StageLog, device=None) -> Path:
    """`device` defaults to the GPU when one is visible; tests pass the CPU."""
    import torch

    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import build_model

    require_dev(cfg["not_plant_split"])
    info = split_info(cfg["not_plant_split"], root)
    names = json.loads((root / RUNS_REL / f"{cfg['model']['e4_record']}.json").read_text(encoding="utf-8"))[
        "config"]["class_names"]
    began = time.perf_counter()
    ckpt = load_checkpoint(root / cfg["model"]["checkpoint"])
    if [k for k, _ in sorted(ckpt["class_map"].items(), key=lambda kv: kv[1])] != names:
        raise RuntimeError("the checkpoint's classes differ from the E4 record's")
    model = build_model(ckpt["config"].model, len(names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log(f"probs-not-plant: {len(info.ids)} photos on {device}, workers {cfg['num_workers']}")
    probs = score_images(model, info.ids, root, cfg["batch_size"], cfg["num_workers"], device)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    log(f"probs-not-plant: done in {time.perf_counter() - began:.1f}s, peak GPU memory {peak} bytes")
    path = out_dir(cfg, root) / "not_plant_dev.probs.jsonl"
    write_jsonl(path, [{"path": p, "probs": [float(v) for v in row]} for p, row in zip(info.ids, probs)])
    return path


# ---- stage: green (CPU) ----

def stage_green(cfg: dict[str, Any], root: Path, log: StageLog) -> Path:
    began = time.perf_counter()
    rows = []
    for split in (cfg["field_split"], cfg["not_plant_split"]):
        require_dev(split)
        for p in split_info(split, root).ids:
            with Image.open(root / p) as img:
                rows.append({"path": p, "green_ratio": green_ratio(img)})
    path = out_dir(cfg, root) / "green_ratio.jsonl"
    write_jsonl(path, rows)
    log(f"green: {len(rows)} photos in {time.perf_counter() - began:.1f}s")
    return path


# ---- stage: fit (CPU) ----

def choose_gates(field_probs: np.ndarray, field_green: np.ndarray, np_probs: np.ndarray, np_green: np.ndarray,
                 budget: float, grid: int) -> dict[str, dict[str, Any]]:
    """Each gate's settings and its rates on the fitting rows (calibrated probabilities in). The rule is in
    configs/e5.json."""
    conf, ent = field_probs.max(axis=1), normalized_entropy(field_probs)

    def rates(s: GateSettings) -> tuple[float, float]:
        return (float(1 - s.accepts_many(field_green, field_probs).mean()),
                float(1 - s.accepts_many(np_green, np_probs).mean()))

    t_conf = max(t for t in np.unique(conf) if (conf < t).mean() <= budget)
    best: tuple | None = None
    qs = np.linspace(0, budget, grid)
    for tc in np.unique(np.quantile(conf, qs, method="lower")):
        for th in np.unique(np.quantile(ent, 1 - qs, method="higher")):
            s = GateSettings("confidence_ood", confidence_threshold=float(tc), ood_threshold=float(th))
            plant_rej, np_rej = rates(s)
            if plant_rej <= budget:
                key = (-np_rej, plant_rej, float(tc), float(th))
                if best is None or key < best[0]:
                    best = (key, s)
    chosen = {
        "green_rule": GateSettings("green_rule", green_threshold=GREEN_THRESHOLD),
        "confidence": GateSettings("confidence", confidence_threshold=float(t_conf)),
    }
    if best is not None:
        chosen["confidence_ood"] = best[1]
    out = {}
    for name, s in chosen.items():
        plant_rej, np_rej = rates(s)
        out[name] = {"settings": {k: v for k, v in asdict(s).items() if v is not None},
                     "fit_plant_rejection": plant_rej, "fit_not_plant_rejection": np_rej,
                     "eligible": plant_rej <= budget}
    return out


def pick_gate(gates: dict[str, dict[str, Any]]) -> str:
    """Highest fit non-plant rejection among the eligible; the earlier gate in KINDS wins a tie."""
    eligible = [k for k in KINDS if k in gates and gates[k]["eligible"]]
    return min(eligible, key=lambda k: (-gates[k]["fit_not_plant_rejection"], KINDS.index(k)))


def fit(cfg: dict[str, Any], field: Dev, nonplant: Dev) -> dict[str, Any]:
    """Everything E5 chooses, from the fitting part of each dev split only."""
    f_fit, n_fit = ~field.scoring(cfg), ~nonplant.scoring(cfg)
    temperature = fit_temperature(np.log(np.maximum(field.probs[f_fit], FLOOR)), field.labels[f_fit])
    cal = calibrate(field.probs, temperature)
    min_coverage = cfg["abstention"]["min_coverage"]
    threshold = round(choose_threshold(cal[f_fit], field.labels[f_fit], min_coverage), 6)
    gates = choose_gates(cal[f_fit], field.green[f_fit], calibrate(nonplant.probs, temperature)[n_fit],
                         nonplant.green[n_fit], cfg["gates"]["plant_rejection_budget"], cfg["gates"]["grid"])
    picked = pick_gate(gates)
    return {
        "seed": cfg["seed"], "fit_percent": cfg["fit_percent"],
        "fit_rows": {"field_dev": int(f_fit.sum()), "not_plant_dev": int(n_fit.sum())},
        "temperature": temperature, "min_coverage": min_coverage, "abstain_threshold": threshold,
        "gates": gates, "gate": {"chosen": picked, "settings": gates[picked]["settings"]},
    }


def stage_fit(cfg: dict[str, Any], root: Path, log: StageLog) -> Path:
    began = time.perf_counter()
    chosen = fit(cfg, load_field(cfg, root), load_not_plant(cfg, root))
    if "field_dev_predictions_file" in cfg["model"]:  # a refit names the exact weights it was fitted for
        chosen["model_checkpoint"] = cfg["model"]["checkpoint"]
        chosen["model_sha256"] = hashlib.sha256((root / cfg["model"]["checkpoint"]).read_bytes()).hexdigest()
    path = out_dir(cfg, root) / "chosen.json"
    atomic_write_text(path, json.dumps(chosen, indent=1) + "\n")
    log(f"fit: T={chosen['temperature']:.4f} abstain>={chosen['abstain_threshold']} gate={chosen['gate']['chosen']} "
        f"in {time.perf_counter() - began:.1f}s")
    return path


# ---- stage: record (CPU) ----

def gate_rows(s: GateSettings, probs: np.ndarray, green: np.ndarray) -> list[dict[str, Any]]:
    """Per photo: the decision and the numbers it follows from, which check_run recomputes it from."""
    accepted = s.accepts_many(green, probs)
    rows = []
    for i, ok in enumerate(accepted):
        row: dict[str, Any] = {"accepted": bool(ok)}
        if s.kind == "green_rule":
            row |= {"green_ratio": float(green[i]), "green_threshold": s.green_threshold}
        else:
            row |= {"confidence": float(probs[i].max()), "threshold": s.confidence_threshold}
        if s.kind == "confidence_ood":
            row |= {"ood_score": float(normalized_entropy(probs[i])), "ood_threshold": s.ood_threshold}
        rows.append(row)
    return rows


def interval(name: str, groups: list[str], **arrays) -> list[float]:
    return list(check_run.recompute_interval(name, check_run.Scored(groups=groups, **arrays)))


def build_records(cfg: dict[str, Any], chosen: dict[str, Any], field: Dev, nonplant: Dev):
    """(field_dev parts, not_plant_dev parts) for write_run, on the scoring parts only."""
    temperature, t = chosen["temperature"], chosen["abstain_threshold"]
    fm, nm = field.scoring(cfg), nonplant.scoring(cfg)
    f_paths, f_groups = [p for p, k in zip(field.paths, fm) if k], [g for g, k in zip(field.groups, fm) if k]
    n_paths = [p for p, k in zip(nonplant.paths, nm) if k]
    labels, raw = field.labels[fm], field.probs[fm]
    cal, green = calibrate(raw, temperature), field.green[fm]
    names = field.class_names
    np_cal, np_green = calibrate(nonplant.probs[nm], temperature), nonplant.green[nm]

    def classifier(probs: np.ndarray, extra: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        rows = [{"path": p, "label": int(y), "probs": [float(v) for v in pr]} for p, y, pr in zip(f_paths, labels, probs)]
        return [{**a, **b} for a, b in zip(rows, extra)] if extra else rows

    scored = check_run.Scored(groups=f_groups, labels=labels, probs=cal)
    suffix = f"@{t:.6f}"
    base_metrics = standard_image_metrics(cal, labels, names)
    metrics = {
        "uncalibrated": standard_image_metrics(raw, labels, names),
        "calibrated": {
            **base_metrics,
            **{f"{k}{suffix}": check_run.recompute_metric(f"{k}{suffix}", scored)
               for k in ("coverage", "selective_accuracy", "confident_error_rate")},
            "confident_error_rate@0.0": check_run.recompute_metric("confident_error_rate@0.0", scored),
        },
    }
    predictions: dict[str, Any] = {"uncalibrated": classifier(raw), "calibrated": classifier(cal)}
    intervals = {"calibrated": {"accuracy": interval("accuracy", f_groups, labels=labels, probs=cal),
                                f"coverage{suffix}": interval(f"coverage{suffix}", f_groups, labels=labels, probs=cal)}}
    np_metrics: dict[str, Any] = {}
    np_predictions: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = [
        {"step": "fit", "temperature": temperature, "abstain_threshold": t, "min_coverage": chosen["min_coverage"],
         "fit_rows": chosen["fit_rows"]}]
    for kind in KINDS:
        s = GateSettings.from_dict(chosen["gates"][kind]["settings"])
        rows = gate_rows(s, cal, green)
        accepted = np.array([r["accepted"] for r in rows])
        sc = check_run.Scored(groups=f_groups, labels=labels, probs=cal, accepted=accepted,
                              is_plant=np.ones(len(rows), dtype=bool))
        predictions[kind] = classifier(cal, rows)
        metrics[kind] = {**base_metrics,
                         "plant_rejection_rate": check_run.recompute_metric("plant_rejection_rate", sc),
                         "end_to_end_accuracy": check_run.recompute_metric("end_to_end_accuracy", sc)}
        intervals[kind] = {"end_to_end_accuracy": interval(
            "end_to_end_accuracy", f_groups, labels=labels, probs=cal, accepted=accepted)}
        n_rows = gate_rows(s, np_cal, np_green)
        np_predictions[kind] = [{"path": p, **r} for p, r in zip(n_paths, n_rows)]
        np_metrics[kind] = {"not_plant_acceptance_rate": float(np.mean([r["accepted"] for r in n_rows]))}
        attempts.append({"step": "gate", "gate": kind, **chosen["gates"][kind]})
    return (
        {"metrics": metrics, "predictions": predictions, "intervals": intervals, "attempts": attempts},
        {"metrics": np_metrics, "predictions": np_predictions, "attempts": attempts},
    )


def stage_record(cfg: dict[str, Any], root: Path, log: StageLog, note: str = "") -> list[Path]:
    began = time.perf_counter()
    chosen = json.loads((out_dir(cfg, root) / "chosen.json").read_text(encoding="utf-8"))
    field, nonplant = load_field(cfg, root), load_not_plant(cfg, root)
    f, n = build_records(cfg, chosen, field, nonplant)
    note = " ".join([cfg["method"], *cfg["limits"], note]).strip()
    shared = {
        "seed": cfg["seed"], "fit_percent": cfg["fit_percent"], "min_coverage": chosen["min_coverage"],
        "temperature": chosen["temperature"], "abstain_threshold": chosen["abstain_threshold"],
        "chosen_gate": chosen["gate"]["chosen"], "gate_settings": {k: v["settings"] for k, v in chosen["gates"].items()},
        "method": cfg["method"], "limits": cfg["limits"], "model_record": cfg["model"]["e4_record"],
        "started_at": datetime.now(timezone.utc).isoformat(), "labels_reviewed": False,
    }
    models = {k: f"{k}: {cfg['model']['name']} from E4 record {cfg['model']['e4_record']}" for k in f["predictions"]}
    paths = [
        write_run("E5", cfg["field_split"], {**shared, "class_names": field.class_names}, models, f["metrics"],
                  f["intervals"], f["attempts"], f["predictions"], field.hash, subset=cfg["subset"], notes=note,
                  root=root),
        write_run("E5", cfg["not_plant_split"], shared, {k: models[k] for k in n["predictions"]}, n["metrics"], {},
                  n["attempts"], n["predictions"], nonplant.hash, subset=cfg["subset"], notes=note, root=root),
    ]
    log(f"record: {[p.name for p in paths]} in {time.perf_counter() - began:.1f}s")
    return paths


STAGES = ("probs-not-plant", "green", "fit", "record")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e5.json")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--note", default="", help="goes into the record's notes (record stage)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        log = StageLog(cfg, ROOT)
        if a.stage == "record":
            result: Any = stage_record(cfg, ROOT, log, a.note)
        else:
            result = {"probs-not-plant": stage_probs_not_plant, "green": stage_green, "fit": stage_fit}[a.stage](
                cfg, ROOT, log)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
