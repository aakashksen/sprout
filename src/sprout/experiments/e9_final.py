"""E9: the final test. Each test split is scored once, with configurations frozen and committed before.

Stages (see docs/PLAN.md T8.1; every stage that touches a test split refuses unless the configs are committed, the
tree is clean and the split has not been used):
    lab-test, field-test        score every frozen system on one image test split and write its record
    wrong-disease-dev           the wrong-disease check on the final E7 served rag_dev answers (Sonnet), with its rate
    blind-sample, unblind       a seeded sample for a reader who does not see the grader's call, then the error report
    freeze                      record the check's prompt commit and sha256 in configs/e9_wrong_disease.json
    rag-retrieve, rag-generate, rag-judge-local, rag-sonnet, rag-record
                                rag_test: served retrieval, the local writer, both judges, the wrong-disease check
No stage reads a test split unless it is the one being run. Tests use fakes and tiny fixtures only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprout.data.label_review import read_log
from sprout.eval.predict import Predictions
from sprout.eval.records import (
    ROOT,
    SplitInfo,
    git,
    is_dirty,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)

CONFIG_REL = "configs/e9.json"
FROZEN_REL = "configs/e9_wrong_disease.json"
PROMPT_REL = "src/sprout/rag/prompts/wrong_disease.txt"
IMAGE_TESTS = ("lab_test", "field_test")
CALLS = ("FITS", "WRONG")


class Refused(RuntimeError):
    """A guard stopped the stage before it did anything."""


# ---------------------------------------------------------------- guards

def require_committed(root: Path, rels: list[str]) -> None:
    """Every file is tracked and identical to HEAD, so the frozen setting was committed before it was used."""
    for rel in rels:
        tracked = git(root, "ls-files", "--error-unmatch", "--", rel).returncode == 0
        same = git(root, "diff", "--quiet", "HEAD", "--", rel).returncode == 0
        if not (tracked and same and (root / rel).is_file()):
            raise Refused(f"{rel} is not committed (or differs from HEAD); freeze and commit it before the test run")


def require_clean(root: Path) -> None:
    if is_dirty(root):
        raise Refused("the working tree has uncommitted changes; commit first")


def labels_reviewed(root: Path = ROOT, tasks_file: str = "orchestration/tasks.json") -> bool:
    """True only when H2 is marked done in the task list, which is where the person's sign-off is recorded."""
    path = Path(root) / tasks_file
    tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks", []) if path.is_file() else []
    return any(t.get("id") == "H2" and t.get("status") == "done" for t in tasks)


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((Path(root) / CONFIG_REL).read_text(encoding="utf-8"))


def load_decision(cfg: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """E12's decision, read by path. Without it the chosen model is unknown, so nothing runs."""
    path = Path(root) / cfg["decision_file"]
    if not path.is_file():
        raise Refused(f"{cfg['decision_file']} does not exist: E12 has not chosen a model, so E9 cannot start")
    decision = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("chosen_model", "chosen_model_sha256", "class_names") if k not in decision]
    if missing:
        raise Refused(f"{cfg['decision_file']} lacks {missing}")
    return decision


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the wrong-disease prompt

def prompt_text(root: Path = ROOT) -> str:
    return (Path(root) / PROMPT_REL).read_text(encoding="utf-8").strip()


def wd_messages(question: str, diagnosis: str, answer: str, root: Path = ROOT) -> list[dict[str, str]]:
    """What the grader is shown: the question, the diagnosis and the answer. No passages, no faithfulness grade."""
    from sprout.experiments.e14_judge_check import clean

    body = f"Question: {clean(question)}\n\nDiagnosis: {clean(diagnosis)}\n\nAnswer: {clean(answer)}"
    return [{"role": "system", "content": prompt_text(root)}, {"role": "user", "content": body}]


def parse_call(reply: str) -> str | None:
    """FITS or WRONG for exactly that one word (case and punctuation aside); anything longer is unusable, not a guess."""
    words = " ".join(re.findall(r"[A-Za-z]+", reply)).upper()
    return words if words in CALLS else None




# ---------------------------------------------------------------- image test splits

SYSTEMS = ("baseline", "improved", "xgboost", "dinov2", "quantized")
FLOOR = 1e-12
INTERVALS = ("accuracy", "macro_f1")
COMPARED = (("improved", "baseline"), ("improved", "xgboost"), ("improved", "dinov2"), ("quantized", "improved"))


@dataclass
class Predictor:
    """One frozen system: its class order, and `run(frame)`, which returns probabilities for the frame's photos
    (columns `path`, absolute, and `label`, a class name)."""

    description: str
    class_names: list[str]
    run: Callable[[pd.DataFrame], np.ndarray]
    detail: dict[str, Any] = field(default_factory=dict)


def load_calibration(cfg: dict[str, Any], decision: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Temperature and abstention threshold from E5's chosen.json. When E12 refit E5, the file must name the model it
    was refit on, and that must be the chosen one."""
    rel = cfg["calibration"]["file"]
    path = Path(root) / rel
    if not path.is_file():
        raise Refused(f"{rel} does not exist: the calibration and threshold are not chosen")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("refit_e5") and saved.get("model_sha256") != decision["chosen_model_sha256"]:
        raise Refused(f"E12 says E5 was refit on the chosen model, but {rel} does not name that model (model_sha256)")
    return {"file": rel, "temperature": float(saved["temperature"]), "abstain_threshold": float(saved["abstain_threshold"])}


def prior_uses(root: Path, split: str) -> list[str]:
    """Every record that has ever existed on this test split, committed or not, whatever its experiment."""
    from sprout.eval.check_run import _uses

    return [stem for stem, rec in _uses(root) if rec["split"] == split]


def require_unused(root: Path, split: str, override_note: str) -> None:
    used = prior_uses(root, split)
    if used and not override_note.strip():
        raise Refused(f"{split} has already been used ({used[0]}); a later run needs --override-note saying why")


def calibrated(raw: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probabilities from saved probabilities (their log stands in for logits, as in E5)."""
    from sprout.serve.calibrate import apply_temperature

    return apply_temperature(np.log(np.maximum(raw, FLOOR)), temperature)


def device_name(cfg: dict[str, Any]) -> str:
    device = cfg.get("torch_device", "cpu")
    if device.startswith("cuda"):
        import torch

        return f"{device} {torch.cuda.get_device_name(torch.device(device))}"
    return device


def cleaned_rows(split: str, cfg: dict[str, Any], info: SplitInfo, positions: list[int], root: Path) -> dict[str, Any]:
    """How many field_test rows the label cleaning removed, and why (counts by reason and source from the log)."""
    version = cfg["exclusion_version"]
    log = read_log(Path(root) / "datasets" / "label_exclusions.csv")
    gone = set(info.ids) - {info.ids[i] for i in positions}
    hit = log[(log["version"] <= version) & log["path"].isin(gone)]
    return {"version": version, "before": len(info.ids), "after": len(positions), "removed": len(gone),
            "by_reason": {k: int(v) for k, v in hit["reason_code"].value_counts().items()},
            "by_source": {k: int(v) for k, v in hit["source"].value_counts().items()}}


def score_block(probs: dict[str, np.ndarray], labels: np.ndarray, groups: list[str], names: list[str],
                threshold: float, keep: np.ndarray) -> tuple[dict, dict, list]:
    """Metrics, intervals and paired comparisons of every system over the rows in `keep`, the way check_run recomputes."""
    from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval, recompute_metric

    scored = {n: Scored(groups=[g for g, k in zip(groups, keep, strict=True) if k], labels=labels[keep],
                        class_names=names, probs=p[keep]) for n, p in probs.items()}
    metrics = {n: standard_image_metrics(s.probs, s.labels, names) for n, s in scored.items()}
    t = f"{threshold:.6f}".rstrip("0").rstrip(".")
    for n in scored:
        if n.endswith("_calibrated"):
            for m in ("coverage", "selective_accuracy", "confident_error_rate"):
                metrics[n][f"{m}@{t}"] = recompute_metric(f"{m}@{t}", scored[n])
    metrics["quantized"]["changed_vs@improved"] = recompute_metric("changed_vs@improved", scored["quantized"], scored)
    intervals = {n: {m: recompute_interval(m, s) for m in INTERVALS} for n, s in scored.items()}
    comparisons = []
    for a, b in COMPARED:
        for m in INTERVALS:
            got = recompute_comparison(scored[a], scored[b], m)
            comparisons.append({"a": a, "b": b, "metric": m, **got})
    return metrics, intervals, comparisons


def run_images(split: str, cfg: dict[str, Any], predictors: dict[str, Predictor], root: Path = ROOT, *,
               override_note: str = "") -> list[Path]:
    """Score every frozen system once on `split` and write its record: lab_test gets a full record and a view on the
    shared classes, field_test gets one record on its cleaned rows. Refuses before any model runs unless the config
    is committed, the tree is clean, E12 has decided, and the split is unused (or an override note says why)."""
    if split not in IMAGE_TESTS:
        raise Refused(f"E9 scores lab_test or field_test here, not {split!r}")
    root = Path(root)
    require_committed(root, [CONFIG_REL])
    require_clean(root)
    decision = load_decision(cfg, root)
    calibration = load_calibration(cfg, decision, root)
    require_unused(root, split, override_note)
    absent = [n for n in SYSTEMS if n not in predictors]
    if absent:
        raise Refused(f"frozen systems missing: {absent}")
    names = decision["class_names"]
    for n in SYSTEMS:
        if predictors[n].class_names != names:
            raise Refused(f"{n} has a different class order from E12's decision")
    info = split_info(split, root)
    subset = f"clean:{cfg['exclusion_version']}" if split == "field_test" else None
    positions = subset_positions(subset, split, info, root) if subset else list(range(len(info.ids)))
    labels_by_name = [info.labels[i] for i in positions]
    unknown = sorted(set(labels_by_name) - set(names))
    if unknown:
        raise Refused(f"{split} has classes the model does not output: {unknown}")
    paths = [info.ids[i] for i in positions]
    groups = [info.groups[i] for i in positions]
    labels = np.array([names.index(x) for x in labels_by_name], dtype=int)
    frame = pd.DataFrame({"path": [str(root / p) for p in paths], "label": labels_by_name})
    started = datetime.now(timezone.utc).isoformat()
    probs: dict[str, np.ndarray] = {}
    for n in SYSTEMS:
        out = np.asarray(predictors[n].run(frame), dtype=float)
        if out.shape != (len(paths), len(names)) or not np.isfinite(out).all():
            raise Refused(f"{n} returned probabilities of shape {out.shape}; expected {(len(paths), len(names))}")
        probs[n] = out
    for n in cfg["calibrated"]:
        probs[f"{n}_calibrated"] = calibrated(probs[n], calibration["temperature"])
    order = [x for n in SYSTEMS for x in (n, f"{n}_calibrated") if x in probs]
    config = {
        "class_names": names, "split": split, "frozen_config": CONFIG_REL,
        "frozen_config_sha256": sha256_file(root / CONFIG_REL), "decision": cfg["decision_file"],
        "chosen_model": decision["chosen_model"], "chosen_model_sha256": decision["chosen_model_sha256"],
        "calibration": calibration, "labels_reviewed": labels_reviewed(root, cfg["tasks_file"]),
        "device": device_name(cfg), "started_at": started,
        "systems": {n: {"description": predictors[n].description, **predictors[n].detail} for n in SYSTEMS},
        "abstention": "a *_calibrated system answers only when its calibrated top probability is at least the "
                      "threshold; the other systems answer every photo",
        **({"cleaned_rows": cleaned_rows(split, cfg, info, positions, root)} if subset else {}),
        **({"override_note": override_note.strip()} if override_note.strip() else {}),
    }
    models = {n: predictors[n].description for n in SYSTEMS}
    models.update({f"{n}_calibrated": f"{predictors[n].description}, temperature {calibration['temperature']} from E5"
                   for n in cfg["calibrated"]})
    attempts = [{"system": n, "kind": "frozen, scored once", **predictors[n].detail} for n in SYSTEMS]
    views = [(subset, np.ones(len(paths), dtype=bool), "")]
    if split == "lab_test":
        shared = set(subset_positions("shared_classes", split, info, root))
        views.append(("shared_classes", np.array([p in shared for p in positions]),
                      " Scored on the classes lab.csv and field.csv share, with all outputs kept, so a prediction "
                      "outside that list is wrong."))
    written = []
    for name, keep, extra in views:
        metrics, intervals, comparisons = score_block(
            {n: probs[n] for n in order}, labels, groups, names, calibration["abstain_threshold"], keep)
        rows = {n: Predictions([p for p, k in zip(paths, keep, strict=True) if k], labels[keep], probs[n][keep])
                for n in order}
        notes = (f"Final test on {split}, scored once with configurations frozen before the run; every number is a "
                 f"held-out number. Options per plant are in config.options_per_plant (plant-given top-3 is easy).{extra}")
        written.append(write_run("E9", split, config, models, metrics, intervals, attempts, rows, info.hash, root=root,
                                 subset=name, comparisons=comparisons, notes=notes))
    return written
