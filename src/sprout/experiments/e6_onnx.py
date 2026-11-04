"""E6: does ONNX, and ONNX quantization, keep accuracy on real photos while running fast on the CPU?

Run from the repo root with PYTHONPATH set to its src folder, one stage per command (configs/e6.json holds every
setting, including the model path: changing the model is one line there):
    .venv/Scripts/python -m sprout.experiments.e6_onnx --stage <stage>      (or tools/run_e6.py)
  export   CPU  checkpoint -> models/e6/model.onnx, .dynamic.onnx, .static.onnx; sizes and checksums -> results/e6/export.json
  probs    CPU* class probabilities of PyTorch and each ONNX file on cleaned field_dev -> results/e6/probs.<system>.jsonl
  latency  CPU  per-photo time on the CPU, one system after another. Run it ALONE, with nothing else on the machine
                (no training, no tests); results/e6/latency.json names the CPU and the thread count.
  record   CPU  scores the saved probabilities, joins the latency, writes the E6 record (refuses a dirty tree)
*The PyTorch reference runs on config `torch_device` ("cpu" or "cuda"); the device used is saved in the record.
export and probs are resumable: a rerun skips finished work, and a changed checkpoint discards the old outputs.
Only field_dev is scored; calibration photos never come from field_dev or field_test. Commit results/e6 after each stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from sprout.atomic import atomic_write_text
from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval
from sprout.eval.predict import Predictions
from sprout.eval.records import (
    ROOT,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)
from sprout.experiments.e5_confidence import calibrate, scoring_part
from sprout.experiments.e15_final_model import answered, mask_columns
from sprout.serve.export import OnnxModel, export_onnx, quantize

SPLIT = "field_dev"
REFERENCE = "pytorch"
BASE = "onnx"  # the unquantized ONNX file: built because the quantized ones come from it, never scored or timed
ONNX_FILES = {BASE: "model.onnx", "onnx_dynamic": "model.dynamic.onnx", "onnx_static": "model.static.onnx"}
CALIBRATION_SPLITS = ("lab_dev", "lab_train", "field_train")  # never field_dev (scored) or any test split
INTERVALS = ("accuracy", "macro_f1")
CAVEAT = (
    "Development numbers on cleaned field_dev only; the model was chosen on field_dev, so they are optimistic. "
    "field_test is not scored."
)


def systems(cfg: dict[str, Any]) -> tuple[str, ...]:
    """The systems compared: the PyTorch model and each quantized file named in `quantize`."""
    return (REFERENCE, *(f"onnx_{m}" for m in cfg["quantize"]))


def check_config(cfg: dict[str, Any]) -> None:
    bad = [m for m in cfg["quantize"] if m not in ("dynamic", "static")]
    if not cfg["quantize"] or bad:
        raise ValueError(f"quantize lists dynamic and/or static, not {cfg['quantize']!r}")
    if cfg["split"] != SPLIT:
        raise ValueError(f"E6 scores {SPLIT} only, not {cfg['split']!r}")
    if cfg["calibration"]["split"] not in CALIBRATION_SPLITS:
        raise ValueError(f"calibration photos come from one of {CALIBRATION_SPLITS}, not {cfg['calibration']['split']!r}")
    if cfg["crop_kept"]:
        raise ValueError("E4 kept a crop, so the segmentation model must be exported and timed too; that is not built")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    check_config(cfg)
    return cfg


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamped_with(path: Path, checkpoint_sha: str) -> bool:
    """Does this model file say it came from the checkpoint with this checksum?"""
    return OnnxModel(path).source_sha256 == checkpoint_sha


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Complete lines only: a last line cut short by a crash is dropped."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


class Paths:
    def __init__(self, cfg: dict[str, Any], root: Path):
        self.out = root / cfg["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.models = root / cfg["model_dir"]
        self.checkpoint = root / cfg["model"]["checkpoint"]
        self.export = self.out / "export.json"
        self.latency = self.out / "latency.json"
        self.log = self.out / "stages.log"

    def probs(self, system: str) -> Path:
        return self.out / f"probs.{system}.jsonl"

    def probs_meta(self, system: str) -> Path:
        return self.out / f"probs.{system}.meta.json"


def log(p: Paths, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    p.out.mkdir(parents=True, exist_ok=True)
    with open(p.log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def calibration_paths(cfg: dict[str, Any], root: Path) -> list[str]:
    """`n` photos of the configured calibration split, drawn with the seed; all of them if the split is smaller."""
    cal = cfg["calibration"]
    if cal["split"] not in CALIBRATION_SPLITS:
        raise ValueError(f"calibration photos come from one of {CALIBRATION_SPLITS}, not {cal['split']!r}")
    ids = split_info(cal["split"], root).ids
    pick = np.random.default_rng(cal["seed"]).permutation(len(ids))[: cal["n"]]
    return sorted(ids[i] for i in pick)


def scored_rows(cfg: dict[str, Any], root: Path):
    """The cleaned field_dev rows in manifest order: paths, groups, labels (class names) and the split info."""
    info = split_info(SPLIT, root)
    pos = subset_positions(cfg["subset"], SPLIT, info, root)
    return [info.ids[i] for i in pos], [info.groups[i] for i in pos], [info.labels[i] for i in pos], info


# ---------------------------------------------------------------------------------------------------------------
# export


def stage_export(cfg: dict[str, Any], root: Path) -> None:
    p = Paths(cfg, root)
    sha = sha256_file(p.checkpoint)
    state = json.loads(p.export.read_text(encoding="utf-8")) if p.export.is_file() else {}
    if state.get("checkpoint_sha256") != sha:
        state = {"checkpoint_sha256": sha, "files": {}}  # a different model: nothing built so far is reusable
    static = "onnx_static" in systems(cfg)
    calibration = {"split": cfg["calibration"]["split"], "paths": calibration_paths(cfg, root)} if static else None
    if state.get("calibration") != calibration:
        state["files"].pop("onnx_static", None)
    state["calibration"] = calibration
    base = p.models / ONNX_FILES["onnx"]

    def done(system: str) -> bool:
        info = state["files"].get(system)
        file = p.models / info["file"] if info else None
        return bool(info) and file.is_file() and sha256_file(file) == info["sha256"] and stamped_with(file, sha)

    def build(system: str, make) -> None:
        if done(system):
            log(p, f"export {system}: finished already")
            return
        t = time.time()
        out = make()
        state["files"][system] = {"file": out.name, "size_bytes": out.stat().st_size, "sha256": sha256_file(out)}
        atomic_write_text(p.export, json.dumps(state, indent=1) + "\n")
        log(p, f"export {system}: {out.stat().st_size} bytes in {time.time() - t:.1f}s")

    build(BASE, lambda: export_onnx(p.checkpoint, base))
    for system in systems(cfg)[1:]:
        mode = system.removeprefix("onnx_")
        build(system, lambda mode=mode: quantize(base, mode, [root / x for x in calibration["paths"]] if static else None))


def exported(p: Paths, cfg: dict[str, Any]) -> dict[str, Any]:
    """export.json, after checking the files on disk are the ones it describes and match the checkpoint."""
    if not p.export.is_file():
        raise RuntimeError("run the export stage first")
    state = json.loads(p.export.read_text(encoding="utf-8"))
    if state["checkpoint_sha256"] != sha256_file(p.checkpoint):
        raise RuntimeError("the checkpoint changed since the export stage; run it again")
    for system in (BASE, *systems(cfg)[1:]):
        info = state["files"].get(system)
        if not info or sha256_file(p.models / info["file"]) != info["sha256"]:
            raise RuntimeError(f"{system} is missing or changed since the export stage; run it again")
        if not stamped_with(p.models / info["file"], state["checkpoint_sha256"]):
            raise RuntimeError(f"{system} does not carry the checkpoint's checksum in its metadata; run export again")
    return state


def source_sha(state: dict[str, Any], system: str) -> str:
    return state["checkpoint_sha256"] if system == REFERENCE else state["files"][system]["sha256"]


# ---------------------------------------------------------------------------------------------------------------
# probabilities


def _torch_model(p: Paths, device: str):
    import torch

    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import build_model

    ckpt = load_checkpoint(p.checkpoint)
    model = build_model(ckpt["config"].model, len(ckpt["class_map"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(torch.device(device)).eval(), ckpt["class_map"]


def weights_only_size_bytes(checkpoint: Path) -> int:
    """Size of a file holding just the model's state dict: the fair PyTorch size next to an ONNX file, since the
    checkpoint also carries training state."""
    import tempfile

    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "weights.pt"
        torch.save(state, out)
        return out.stat().st_size


def _device_name(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return f"{device} {torch.cuda.get_device_name(torch.device(device))}"
    return "cpu"


def stage_probs(cfg: dict[str, Any], root: Path) -> None:
    p = Paths(cfg, root)
    state = exported(p, cfg)
    paths, _groups, labels, _info = scored_rows(cfg, root)
    for system in systems(cfg):
        device = _device_name(cfg["torch_device"]) if system == REFERENCE else "cpu"
        meta = {"source_sha256": source_sha(state, system), "device": device, "subset": cfg["subset"]}
        rows: list[dict[str, Any]] = []
        if p.probs(system).is_file() and p.probs_meta(system).is_file() and json.loads(
                p.probs_meta(system).read_text(encoding="utf-8")) == meta:
            rows = read_jsonl(p.probs(system))
            keep = 0
            while keep < min(len(rows), len(paths)) and rows[keep]["path"] == paths[keep]:
                keep += 1
            rows = rows[:keep]
        if len(rows) == len(paths):
            log(p, f"probs {system}: finished already")
            continue
        atomic_write_text(p.probs_meta(system), json.dumps(meta) + "\n")
        atomic_write_text(p.probs(system), "".join(json.dumps(r) + "\n" for r in rows))
        if system == REFERENCE:
            model, class_map = _torch_model(p, cfg["torch_device"])
        else:
            model = OnnxModel(p.models / state["files"][system]["file"], cfg["threads"])
            class_map = model.class_map
        t = time.time()
        for start in range(len(rows), len(paths), cfg["chunk"]):
            chunk, names = paths[start : start + cfg["chunk"]], labels[start : start + cfg["chunk"]]
            if system == REFERENCE:
                from sprout.eval.predict import predict

                frame = pd.DataFrame({"path": [str(root / x) for x in chunk], "label": names})
                probs = predict(model, frame, class_map, cfg["batch_size"]).probs
            else:
                probs = []
                for x in chunk:
                    with Image.open(root / x) as img:
                        probs.append(model.predict(img))
            with open(p.probs(system), "a", encoding="utf-8") as f:
                for path, name, pr in zip(chunk, names, probs, strict=True):
                    f.write(json.dumps({"path": path, "label": class_map[name], "probs": [float(v) for v in pr]}) + "\n")
                f.flush()
        log(p, f"probs {system}: {len(paths) - len(rows)} photos on {device} in {time.time() - t:.1f}s")


# ---------------------------------------------------------------------------------------------------------------
# latency


def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or platform.machine()


def cpu_load_percent() -> float | None:
    """The whole machine's CPU load now, to say in the record how idle it was; None where it cannot be read."""
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=30, check=True).stdout
        return float(out.strip())
    except Exception:
        return None


def stage_latency(cfg: dict[str, Any], root: Path) -> None:
    """Time one photo at a time, decoded beforehand, from resize to probabilities. Needs the machine to itself."""
    import onnxruntime
    import torch

    p = Paths(cfg, root)
    state = exported(p, cfg)
    paths = scored_rows(cfg, root)[0][: cfg["latency"]["images"]]
    images = []
    for x in paths:
        with Image.open(root / x) as img:
            images.append(img.convert("RGB"))
    machine = {
        "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "threads": cfg["threads"], "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
        "images": len(images), "warmup": cfg["latency"]["warmup"],
    }
    saved = json.loads(p.latency.read_text(encoding="utf-8")) if p.latency.is_file() else {}
    if {k: saved.get(k) for k in machine} != machine:
        saved = {**machine, "systems": {}}  # another machine or setting: earlier numbers do not carry over
    torch.set_num_threads(cfg["threads"])
    for system in systems(cfg):
        old = saved["systems"].get(system)
        if old and old["source_sha256"] == source_sha(state, system) and len(old["samples_ms"]) == len(images):
            log(p, f"latency {system}: finished already")
            continue
        if system == REFERENCE:
            from sprout.preprocess import preprocess

            model, _ = _torch_model(p, "cpu")

            def predict(img, model=model):
                with torch.no_grad():
                    return torch.softmax(model(preprocess(img)[None]), dim=1)[0].numpy()
        else:
            predict = OnnxModel(p.models / state["files"][system]["file"], cfg["threads"]).predict
        load = cpu_load_percent()  # before the warm-up, while this process is not yet loading the CPU
        for i in range(cfg["latency"]["warmup"]):
            predict(images[i % len(images)])
        samples = []
        for img in images:
            t = time.perf_counter()
            predict(img)
            samples.append((time.perf_counter() - t) * 1000)
        saved["systems"][system] = {"source_sha256": source_sha(state, system), "samples_ms": samples,
                                    "cpu_load_percent_before": load}
        atomic_write_text(p.latency, json.dumps(saved, indent=1) + "\n")
        log(p, f"latency {system}: median {float(np.median(samples)):.1f} ms over {len(samples)} photos")


# ---------------------------------------------------------------------------------------------------------------
# record


def threshold_behaviour(cfg: dict[str, Any], root: Path, checkpoint_sha: str, preds: dict[str, Predictions],
                        names: list[str], groups: list[str]) -> dict[str, Any]:
    """How the fitted temperature, abstain threshold and healthy threshold behave on each system's probabilities.
    The outputs are masked to the fitted file's `classes` and renormalized, as `Identifier` does. Counts are given on
    the refit's scoring part (the groups it did not fit on: same seed and share) and on all rows. `sick_called_healthy`
    is a sick photo answered with a healthy label."""
    path = root / cfg["thresholds"]
    fitted = json.loads(path.read_text(encoding="utf-8"))
    temperature, abstain = float(fitted["temperature"]), float(fitted["abstain_threshold"])
    healthy = max(abstain, float(fitted.get("healthy_threshold", 0.0)))
    suffix = fitted.get("healthy_suffix", "___healthy")
    classes = fitted.get("classes") or names
    part = scoring_part(groups, cfg["scoring_part"]["seed"], cfg["scoring_part"]["fit_percent"])
    flags = np.array([c.endswith(suffix) for c in classes])
    systems_out = {}
    for system, pr in preds.items():
        cal = calibrate(mask_columns(pr.probs, names, classes), temperature)
        y = np.array([classes.index(names[i]) for i in pr.label])
        top = cal.argmax(axis=1)
        ok = answered(cal, classes, abstain, healthy, suffix)
        counts = {}
        for label, rows in (("scoring_part", part), ("all_rows", np.ones(len(y), dtype=bool))):
            a = ok & rows
            counts[label] = {
                "n": int(rows.sum()), "answered": int(a.sum()), "unsure": int((rows & ~ok).sum()),
                "coverage": float(a.sum() / rows.sum()), "healthy_answers": int((a & flags[top]).sum()),
                "sick_called_healthy": int((a & ~flags[y] & flags[top]).sum()),
                "confident_wrong": int((a & (top != y)).sum()),
            }
        systems_out[system] = counts
    named = fitted.get("model_sha256")
    return {
        "file": cfg["thresholds"], "file_sha256": sha256_file(path), "temperature": temperature,
        "abstain_threshold": abstain, "healthy_threshold": healthy, "classes": len(classes), "model_sha256": named,
        "names_this_checkpoint": None if named is None else named == checkpoint_sha,
        "scoring_part": cfg["scoring_part"], "systems": systems_out,
    }


def stage_record(cfg: dict[str, Any], root: Path) -> Path:
    p = Paths(cfg, root)
    state = exported(p, cfg)
    paths, groups, labels, info = scored_rows(cfg, root)
    if not p.latency.is_file():
        raise RuntimeError("run the latency stage first (alone on the machine); the record needs its numbers")
    latency = json.loads(p.latency.read_text(encoding="utf-8"))
    class_map = OnnxModel(p.models / state["files"][BASE]["file"]).class_map
    names = sorted(class_map, key=class_map.get)
    preds, devices = {}, {}
    for system in systems(cfg):
        source = source_sha(state, system)
        meta = json.loads(p.probs_meta(system).read_text(encoding="utf-8")) if p.probs_meta(system).is_file() else {}
        rows = read_jsonl(p.probs(system)) if p.probs(system).is_file() else []
        if meta.get("source_sha256") != source or [r["path"] for r in rows] != paths:
            raise RuntimeError(f"probs for {system} are missing, partial or from another model; run the probs stage")
        if latency["systems"].get(system, {}).get("source_sha256") != source:
            raise RuntimeError(f"latency for {system} is missing or from another model; run the latency stage")
        devices[system] = meta["device"]
        preds[system] = Predictions(paths, np.array([r["label"] for r in rows], dtype=int),
                                    np.array([r["probs"] for r in rows], dtype=float))
    if [names[y] for y in preds[REFERENCE].label] != labels:
        raise RuntimeError("the saved labels do not match the manifest")
    scored = {s: Scored(groups=groups, labels=pr.label, probs=pr.probs) for s, pr in preds.items()}
    metrics = {s: standard_image_metrics(pr.probs, pr.label, names) for s, pr in preds.items()}
    for s in systems(cfg)[1:]:
        metrics[s]["changed_vs@" + REFERENCE] = float(
            (preds[s].probs.argmax(axis=1) != preds[REFERENCE].probs.argmax(axis=1)).mean())
    intervals = {s: {m: recompute_interval(m, sc) for m in INTERVALS} for s, sc in scored.items()}
    comparisons = []
    for s in systems(cfg)[1:]:
        for m in INTERVALS:
            got = recompute_comparison(scored[REFERENCE], scored[s], m)
            comparisons.append({"a": REFERENCE, "b": s, "metric": m, "diff": got["diff"], "ci": got["ci"],
                                "mcnemar_p": got["mcnemar_p"]})
    measurements = {}
    for s in systems(cfg):
        measurements[f"latency_ms_{s}"] = latency["systems"][s]["samples_ms"]
        size = weights_only_size_bytes(p.checkpoint) if s == REFERENCE else state["files"][s]["size_bytes"]
        measurements[f"size_bytes_{s}"] = [float(size)]
    measurements[f"size_bytes_{REFERENCE}_checkpoint"] = [float(p.checkpoint.stat().st_size)]
    machine = {k: v for k, v in latency.items() if k != "systems"}
    config = {
        "class_names": names,
        "model": cfg["model"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "onnx_files": {s: {k: v for k, v in state["files"][s].items() if k != "file"} for s in (BASE, *systems(cfg)[1:])},
        "segmentation_model": "not exported: E4 kept no crop (crop_kept is false in configs/e6.json), so no "
                              "segmentation step is exported or timed",
        "calibration": state["calibration"],
        "probs_device": devices,
        "latency": {**machine, "cpu_load_percent_before": {
            s: latency["systems"][s].get("cpu_load_percent_before") for s in systems(cfg)}, "measured": "one photo at a time on the CPU, decoded beforehand, from resize to "
                                           "probabilities, by the latency stage run alone"},
        "thresholds": threshold_behaviour(cfg, root, state["checkpoint_sha256"], preds, names, groups),
        "label": "development number, field_dev only",
    }
    sizes = ", ".join(f"{s} {measurements[f'size_bytes_{s}'][0] / 1e6:.1f} MB" for s in systems(cfg))
    sizes += (f" (pytorch is its weights alone; the checkpoint file with training state is "
              f"{measurements[f'size_bytes_{REFERENCE}_checkpoint'][0] / 1e6:.1f} MB)")
    loss = "; ".join(f"{c['b']} {c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
                     for c in comparisons if c["metric"] == "macro_f1")
    notes = (f"{CAVEAT} Sizes: {sizes}. Macro-F1 of pytorch minus each ONNX file: {loss}. Latency on {machine['cpu']}, "
             f"{machine['threads']} threads. Class probabilities were computed on {devices[REFERENCE]} (PyTorch) and the "
             "CPU (ONNX).")
    held = "; ".join(f"{s} {v['scoring_part']['answered']}/{v['scoring_part']['n']} answered, "
                     f"{v['scoring_part']['sick_called_healthy']} sick called healthy"
                     for s, v in config["thresholds"]["systems"].items())
    notes += f" Thresholds fitted for the PyTorch model, on the refit's scoring part: {held}."
    attempts = [{"setting": s, "result": f"macro_f1 {metrics[s]['macro_f1']:.4f}", "device": devices[s]}
                for s in systems(cfg)]
    models = {s: (f"{cfg['model']['name']} checkpoint {cfg['model']['checkpoint']}" if s == REFERENCE
                  else f"{cfg['model']['name']} {ONNX_FILES[s]}") for s in systems(cfg)}
    return write_run("E6", SPLIT, config, models, metrics, intervals, attempts, preds, info.hash, root=root,
                     notes=notes, subset=cfg["subset"], comparisons=comparisons, measurements=measurements)


STAGES = {"export": stage_export, "probs": stage_probs, "latency": stage_latency, "record": stage_record}


def run_stage(stage: str, cfg: dict[str, Any], root: Path = ROOT):
    check_config(cfg)
    return STAGES[stage](cfg, root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--config", default="configs/e6.json")
    a = ap.parse_args(argv)
    try:
        out = run_stage(a.stage, load_config(ROOT / a.config), ROOT)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if out:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
