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
import subprocess
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprout.atomic import atomic_write_text
from sprout.data.label_review import read_log
from sprout.eval.answer_checks import (
    CALLS,
    blind_intervals,
    blind_metrics,
    dev_intervals,
    dev_metrics,
    draw_blind,
    final_lists,
    parse_call,
    strata_sizes,
)
from sprout.eval.answer_checks import cleaned_rows as cleaned
from sprout.eval.check_run import (
    Scored,
    check_file,
    recompute_comparison,
    recompute_interval,
    recompute_metric,
    records_for,
)
from sprout.eval.predict import Predictions
from sprout.eval.records import (
    RAG_REL,
    ROOT,
    SplitInfo,
    git,
    is_dirty,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)
from sprout.experiments import e7_rag as e7
from sprout.experiments import e14_judge_check as e14
from sprout.experiments.e7_rag import Checkpoint
from sprout.llm.client import LLMError
from sprout.rag import serving
from sprout.rag.answer import GENERAL_CROP, AnswerConfig, diagnosis_summary
from sprout.rag.evalset import Question, load_questions
from sprout.rag.evalset import index_crops as index_crops_of
from sprout.rag.judge import rubric

CONFIG_REL = "configs/e9.json"
FROZEN_REL = "configs/e9_wrong_disease.json"
PROMPT_REL = "src/sprout/rag/prompts/wrong_disease.txt"
IMAGE_TESTS = ("lab_test", "field_test")
SIGNOFF_REL = "orchestration/label_signoff.json"
# E9 records never hold the label signature: `claims check` reads it per split from SIGNOFF_REL, so signing later needs no
# rerun of a once-only test.
LABELS_REVIEWED = False
LABEL_SIGNATURE = (f"Not held in this record: the person's sign-off on this split's labels is the row for the split in "
                   f"{SIGNOFF_REL}, which `claims check` reads. labels_reviewed is false here by design.")


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


# ---------------------------------------------------------------- image test splits

SYSTEMS = ("baseline", "improved", "alternate", "xgboost", "dinov2", "quantized")
FLOOR = 1e-12
INTERVALS = ("accuracy", "macro_f1")
COMPARED = (("improved", "baseline"), ("improved", "alternate"), ("improved", "xgboost"), ("improved", "dinov2"),
            ("quantized", "improved"))


@dataclass
class Predictor:
    """One frozen system: its class order, and `run(frame)`, which returns probabilities for the frame's photos
    (columns `path`, absolute, and `label`, a class name)."""

    description: str
    class_names: list[str]
    run: Callable[[pd.DataFrame], np.ndarray]
    detail: dict[str, Any] = field(default_factory=dict)


def candidate_roles(cfg: dict[str, Any], decision: dict[str, Any]) -> tuple[str, str]:
    """(primary, alternate) candidate keys: the primary is the model E12's decision chose, the other is reported beside it."""
    found = [k for k, v in cfg["candidates"].items() if v["checkpoint"] == decision["chosen_model"]]
    others = [k for k in cfg["candidates"] if k not in found]
    if len(found) != 1 or len(others) != 1:
        raise Refused(f"E12 chose {decision['chosen_model']!r}, which is neither of the config's candidates "
                      f"{[v['checkpoint'] for v in cfg['candidates'].values()]}")
    return found[0], others[0]


def load_calibration(rel: str, model_sha: str | None, root: Path = ROOT, *, required: bool = False) -> dict[str, Any]:
    """Temperature and abstention threshold from an E5-style chosen.json. When the file names the model it was fitted
    on (`model_sha256`), that must be the model scored; `required` demands the name (E12 says E5 was refit)."""
    path = Path(root) / rel
    if not path.is_file():
        raise Refused(f"{rel} does not exist: the calibration and threshold are not chosen")
    saved = json.loads(path.read_text(encoding="utf-8"))
    named = saved.get("model_sha256")
    if named is None and required:
        raise Refused(f"E12 says E5 was refit on the chosen model, but {rel} does not name that model (model_sha256)")
    if named is not None and named != model_sha:
        raise Refused(f"{rel} was fitted on another model than the one scored (model_sha256 differs)")
    return {"file": rel, "temperature": float(saved["temperature"]), "abstain_threshold": float(saved["abstain_threshold"])}


def calibrations(cfg: dict[str, Any], decision: dict[str, Any], root: Path = ROOT, *,
                 alternate_sha: str | None = None) -> dict[str, dict[str, Any]]:
    """Calibration per calibrated system: the primary model's own (also used for the quantized copy of it) and the
    alternate's own, so both models are reported with and without abstention."""
    primary, alternate = candidate_roles(cfg, decision)
    prim = load_calibration(cfg["candidates"][primary]["calibration"], decision["chosen_model_sha256"], root,
                            required=bool(decision.get("refit_e5")))
    alt = load_calibration(cfg["candidates"][alternate]["calibration"], alternate_sha, root)
    return {"improved": prim, "alternate": alt, "quantized": prim}


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
    log = read_log(Path(root) / "datasets" / "label_exclusions.csv")
    return cleaned(log, cfg["exclusion_version"], info.ids, [info.ids[i] for i in positions])


def score_block(probs: dict[str, np.ndarray], labels: np.ndarray, groups: list[str], names: list[str],
                thresholds: dict[str, float], keep: np.ndarray) -> tuple[dict, dict, list]:
    """Metrics, intervals and paired comparisons of every system over the rows in `keep`, the way check_run recomputes."""
    scored = {n: Scored(groups=[g for g, k in zip(groups, keep, strict=True) if k], labels=labels[keep],
                        class_names=names, probs=p[keep]) for n, p in probs.items()}
    metrics = {n: standard_image_metrics(s.probs, s.labels, names) for n, s in scored.items()}
    for n in scored:
        if n.endswith("_calibrated"):
            t = f"{thresholds[n.removesuffix('_calibrated')]:.6f}".rstrip("0").rstrip(".")
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


def run_images(split: str, cfg: dict[str, Any], predictors: dict[str, Predictor] | Callable[[], dict[str, Predictor]],
               root: Path = ROOT, *,
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
    primary, alternate = candidate_roles(cfg, decision)
    require_unused(root, split, override_note)
    if callable(predictors):  # built only after every guard has passed, so a refused run loads no model
        predictors = predictors()
    absent = [n for n in SYSTEMS if n not in predictors]
    if absent:
        raise Refused(f"frozen systems missing: {absent}")
    calibration = calibrations(cfg, decision, root, alternate_sha=predictors["alternate"].detail.get("sha256"))
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
        probs[f"{n}_calibrated"] = calibrated(probs[n], calibration[n]["temperature"])
    order = [x for n in SYSTEMS for x in (n, f"{n}_calibrated") if x in probs]
    cleaned_info = cleaned_rows(split, cfg, info, positions, root) if subset else None
    basis = (f"cleaned rows: {cleaned_info['after']} of {cleaned_info['before']} field_test photos kept "
             f"({subset}); {cleaned_info['removed']} removed by the label cleaning" if subset else
             "lab_test photos, all of them: the label cleaning covers real photos only")
    config = {
        "class_names": names, "split": split, "frozen_config": CONFIG_REL,
        "frozen_config_sha256": sha256_file(root / CONFIG_REL), "decision": cfg["decision_file"],
        "chosen_model": decision["chosen_model"], "chosen_model_sha256": decision["chosen_model_sha256"],
        "primary": {"system": "improved", "candidate": primary, "checkpoint": decision["chosen_model"],
                    "sha256": decision["chosen_model_sha256"], "why": "the model E12's decision chose"},
        "alternate": {"system": "alternate", "candidate": alternate,
                      "checkpoint": cfg["candidates"][alternate]["checkpoint"],
                      "sha256": predictors["alternate"].detail.get("sha256"),
                      "why": "the other candidate, reported beside the primary on the same rows"},
        "basis": basis, "calibration": calibration, "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
        "device": device_name(cfg), "started_at": started,
        "systems": {n: {"description": predictors[n].description, **predictors[n].detail} for n in SYSTEMS},
        "abstention": "a *_calibrated system answers only when its calibrated top probability is at least the "
                      "threshold; the other systems answer every photo",
        **({"cleaned_rows": cleaned_info} if subset else {}),
        **({"override_note": override_note.strip()} if override_note.strip() else {}),
    }
    models = {n: predictors[n].description for n in SYSTEMS}
    models["improved"] = f"primary ({primary}): {models['improved']}"
    models["alternate"] = f"alternate ({alternate}): {models['alternate']}"
    models.update({f"{n}_calibrated": f"{models[n]}, temperature {calibration[n]['temperature']}"
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
            {n: probs[n] for n in order}, labels, groups, names,
            {n: calibration[n]["abstain_threshold"] for n in cfg["calibrated"]}, keep)
        rows = {n: Predictions([p for p, k in zip(paths, keep, strict=True) if k], labels[keep], probs[n][keep])
                for n in order}
        notes = (f"Final test on {split}, scored once with configurations frozen before the run; every number is a "
                 f"held-out number on this basis: {basis}. The primary model ({primary}) is the one E12 chose; the "
                 f"alternate ({alternate}) is scored beside it on the same rows. Options per plant are in "
                 f"config.options_per_plant (plant-given top-3 is easy). {LABEL_SIGNATURE}{extra}")
        written.append(write_run("E9", split, config, models, metrics, intervals, attempts, rows, info.hash, root=root,
                                 subset=name, comparisons=comparisons, notes=notes))
    return written


# ---------------------------------------------------------------- the real models

def _torch_predictor(path: Path, label: str, batch_size: int, device: str) -> Predictor:
    import torch

    from sprout.eval.predict import predict
    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import build_model

    ckpt = load_checkpoint(path)
    class_map = ckpt["class_map"]
    model = build_model(ckpt["config"].model, len(class_map), pretrained=False)
    model.load_state_dict(ckpt["model"])

    def run(frame: pd.DataFrame) -> np.ndarray:
        model.to(torch.device(device))
        try:
            return predict(model, frame, class_map, batch_size).probs
        finally:
            model.cpu()

    names = sorted(class_map, key=class_map.get)
    return Predictor(f"{label}: {ckpt['config'].model} checkpoint {path.name}", names, run,
                     {"kind": "torch_checkpoint", "model": ckpt["config"].model, "sha256": sha256_file(path)})


def _xgboost_predictor(path: Path, features: dict[str, Any], names: list[str]) -> Predictor:
    import xgboost as xgb
    from PIL import Image

    from sprout.baselines.classic import classic_features, xgboost_probs

    booster = xgb.Booster()
    booster.load_model(str(path))

    def run(frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for p in frame["path"]:
            with Image.open(p) as img:
                rows.append(classic_features(img, **features))
        return xgboost_probs(booster, np.stack(rows))

    return Predictor("HOG and colour histograms with XGBoost", names, run,
                     {"kind": "xgboost", "features": features, "sha256": sha256_file(path)})


def _onnx_predictor(cfg: dict[str, Any], decision: dict[str, Any], root: Path) -> Predictor:
    """The quantized file E6 exported from the chosen checkpoint, checked against E6's export.json."""
    from PIL import Image

    from sprout.serve.export import OnnxModel

    spec = cfg["systems"]["quantized"]
    export = json.loads((root / spec["export_file"]).read_text(encoding="utf-8"))
    if export["checkpoint_sha256"] != decision["chosen_model_sha256"]:
        raise Refused(f"{spec['export_file']} was not exported from the chosen model; run E6 on it first")
    info = export["files"].get(spec["variant"])
    path = root / spec["model_dir"] / info["file"] if info else None
    if path is None or not path.is_file() or sha256_file(path) != info["sha256"]:
        raise Refused(f"{spec['variant']} is missing or changed since E6 exported it")
    model = OnnxModel(path)

    def run(frame: pd.DataFrame) -> np.ndarray:
        out = []
        for p in frame["path"]:
            with Image.open(p) as img:
                out.append(model.predict(img))
        return np.stack(out)

    return Predictor(f"{spec['variant']} export of the chosen model", model.class_names, run,
                     {"kind": "onnx", "variant": spec["variant"], "sha256": info["sha256"]})


def build_predictors(cfg: dict[str, Any], decision: dict[str, Any], root: Path = ROOT) -> dict[str, Predictor]:
    """The six frozen systems from the config and E12's decision. `improved` is the model E12 chose and `alternate` the
    other candidate; the chosen file must be the one the decision hashes."""
    root, size, device = Path(root), cfg["batch_size"], cfg["torch_device"]
    spec = cfg["systems"]
    primary, alternate = candidate_roles(cfg, decision)
    chosen = root / decision["chosen_model"]
    if not chosen.is_file() or sha256_file(chosen) != decision["chosen_model_sha256"]:
        raise Refused(f"{decision['chosen_model']} differs from E12's decision (sha256) or is missing")
    other = root / cfg["candidates"][alternate]["checkpoint"]
    if not other.is_file():
        raise Refused(f"{cfg['candidates'][alternate]['checkpoint']} is missing: the alternate model must be scored too")
    return {
        "baseline": _torch_predictor(root / spec["baseline"]["checkpoint"], "clean lab baseline", size, device),
        "improved": _torch_predictor(chosen, f"{primary} model chosen by E12", size, device),
        "alternate": _torch_predictor(other, f"{alternate} model, the other candidate", size, device),
        "xgboost": _xgboost_predictor(root / spec["xgboost"]["model"], spec["xgboost"]["features"],
                                      decision["class_names"]),
        "dinov2": _torch_predictor(root / spec["dinov2"]["checkpoint"], "DINOv2 small", size, device),
        "quantized": _onnx_predictor(cfg, decision, root),
    }


METRIC_LINES = (("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("ece", "calibration error (ECE)"),
                ("top_3_accuracy", "top-3"), ("top_5_accuracy", "top-5"), ("plant_given_top_1", "plant-given top-1"),
                ("plant_given_top_3", "plant-given top-3"))


def image_report(rec: dict[str, Any]) -> str:
    """The figures of an image record in words. Every line that holds a number names the rows it is on."""
    c = rec["config"]
    tag = "cleaned rows" if "cleaned_rows" in c else f"{rec['split']}" + (f" {rec['subset']}" if rec["subset"] else "")
    role = {c["primary"]["system"]: f"primary ({c['primary']['candidate']})",
            c["alternate"]["system"]: f"alternate ({c['alternate']['candidate']})"}
    lines = [f"[{tag}] basis: {c['basis']}; n = {rec['n']}"]
    if "cleaned_rows" in c:
        r = c["cleaned_rows"]
        lines.append(f"[{tag}] removed {r['removed']} of {r['before']}: by reason {r['by_reason']}, by source {r['by_source']}")
    lines.append(f"[{tag}] Options per plant: {c['options_per_plant']}; label review is signed per split in {SIGNOFF_REL}, not here.")
    for name, values in rec["metrics"].items():
        shown = ", ".join(f"{label} {values[key]:.3f}" for key, label in METRIC_LINES if values.get(key) is not None)
        base = name.removesuffix("_calibrated")
        kind = (f"{role.get(base, base)}" + (" with abstention" if name.endswith("_calibrated") else ""))
        extra = {k: v for k, v in values.items() if k.startswith(("coverage@", "selective_accuracy@", "confident_error_rate@"))}
        if extra:
            shown += "; " + ", ".join(f"{k} {v:.3f}" for k, v in extra.items() if v is not None)
        lines.append(f"[{tag}] {kind}: {shown}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Sonnet calls, two at a time

WORK_REL = Path("results/e9/work")
IN_FLIGHT = 2
CALL_TIMEOUT_S = 180.0
DEV_SPLIT = "rag_dev"
WD_DEVICE = "none: no GPU or local model is used; Sonnet runs through the claude command"
DIAGNOSIS_RULE = (
    "The diagnosis is the text the writer was given for the same question: the simulated photo diagnosis of E7 "
    "(sprout.experiments.e7_rag.simulate_identification, which names the plant's class that the question mentions, "
    "never the gold label) written by answer.diagnosis_summary. With no such class it says no photo diagnosis is "
    "available, plus the plant when the question states or names one. rag_dev and rag_test questions have no photos "
    "and use the same rule."
)


def check_budget(llm: Any, need: int, limit: int, spent: int) -> None:
    """Refuse, before any call, a run that would pass the experiment's limit or the shared Claude call cap."""
    if spent + need > limit:
        raise LLMError(f"E9 Sonnet limit of {limit} calls would be passed ({spent} made, {need} needed); no call was made")
    made, cap = getattr(llm, "calls_made", None), getattr(llm, "cap", None)
    if callable(made) and cap is not None and made() + need > cap:
        raise LLMError(f"Claude call cap {cap} would be passed ({made()} made, {need} needed); no call was made")


def run_two_at_a_time(todo: list[Any], one: Callable[[Any], dict[str, Any]], ck: Checkpoint) -> None:
    """Run `one` over `todo` with at most IN_FLIGHT calls at once, saving each row as it arrives. An error stops new
    calls, keeps what arrived, and is raised."""
    failure: LLMError | None = None
    with ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
        queue, live = iter(todo), set()
        while True:
            while failure is None and len(live) < IN_FLIGHT:
                nxt = next(queue, None)
                if nxt is None:
                    break
                live.add(pool.submit(one, nxt))
            if not live:
                break
            done, live = wait(live, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    ck.add(f.result())
                except LLMError as exc:
                    failure = failure or exc
    if failure is not None:
        raise failure


# ---------------------------------------------------------------- the wrong-disease check

def diagnosis_for(q: Question, crops: set[str], classes: list[str]) -> str:
    crop, ident = e7.simulate_identification(q, crops, classes)
    text = diagnosis_summary(ident)
    return text + (f" The plant is {crop}." if ident is None and crop else "")


def index_crops(index_dir: Path) -> set[str]:
    return set(index_crops_of(Path(index_dir) / "chunks.jsonl").values()) - {GENERAL_CROP}


def rendered_sha256(root: Path) -> str:
    """Hash of the messages built for a fixed input, so a change to the wording or the layout shows up even when the
    prompt file is unchanged."""
    return hashlib.sha256(json.dumps(wd_messages("Q", "D", "A", root), sort_keys=True).encode()).hexdigest()


def wd_key(items: list[dict[str, Any]], model: str, root: Path) -> str:
    parts = [prompt_text(root), model, [[i["id"], i["question"], i["diagnosis"], i["answer"]] for i in items]]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:10]


def wd_grade(items: list[dict[str, Any]], llm: Any, tag: str, root: Path, limit: int | None = None) -> Checkpoint:
    """One Sonnet call per item not already saved. With `limit`, refuse first if the calls would pass it or the cap."""
    ck = Checkpoint(Path(root) / WORK_REL / f"wd-{tag}-{wd_key(items, llm.model, root)}.jsonl")
    todo = [i for i in items if i["id"] not in ck.rows]
    if limit is not None:
        check_budget(llm, len(todo), limit, len(ck.rows))

    def one(item: dict[str, Any]) -> dict[str, Any]:
        reply = llm.complete(wd_messages(item["question"], item["diagnosis"], item["answer"], root),
                             max_tokens=8, timeout_s=CALL_TIMEOUT_S).text.strip()
        return {"id": item["id"], "reply": reply, "call": parse_call(reply)}

    run_two_at_a_time(todo, one, ck)
    return ck


def dev_answers(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """The final E7 served answers on rag_dev (from the judgments file the config names, checked by sha256), each with
    its question and the diagnosis the writer was given."""
    from sprout.experiments.e14_judge_check import item_of, load_source

    wd = cfg["wrong_disease"]
    judged, _ = load_source({"source": wd["dev_source"]}, root)
    questions = {q.id: q for q in e7.dev_questions(root)}
    crops, classes = index_crops(Path(root) / cfg["rag"]["index_dir"]), e7.load_classes(root)
    served = sorted((r for r in judged if r["system"] == wd["served_system"]), key=lambda r: r["id"])
    return [{"id": r["id"], "item": item_of(r["system"], r["id"]), "question": r["question"],
             "diagnosis": diagnosis_for(questions[r["id"]], crops, classes), "answer": r["answer"]} for r in served]


def prompt_fields(root: Path) -> dict[str, str]:
    return {"prompt_file": PROMPT_REL, "prompt_sha256": sha256_file(Path(root) / PROMPT_REL),
            "rendered_sha256": rendered_sha256(root)}


def stage_wrong_disease_dev(cfg: dict[str, Any], llm: Any, root: Path = ROOT, note: str = "") -> Path:
    """Run the check on the served rag_dev answers and write its record with the rate. Calls are saved as they arrive."""
    root = Path(root)
    require_clean(root)
    wd = cfg["wrong_disease"]
    items = dev_answers(cfg, root)
    started = datetime.now(timezone.utc).isoformat()
    ck = wd_grade(items, llm, "dev", root, limit=wd["dev_max_calls"])
    rows = [{**i, "call": ck.rows[i["id"]]["call"], "reply": ck.rows[i["id"]]["reply"]} for i in items]
    info = split_info(DEV_SPLIT, root)
    config = {"stage": "wrong_disease_dev", **prompt_fields(root), "source": wd["dev_source"],
              "served_system": wd["served_system"], "diagnosis_source": DIAGNOSIS_RULE, "claude_calls": len(rows),
              "device": WD_DEVICE, "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
              "started_at": started,
              "grader_prompt": "question, diagnosis and answer; the reply is read strictly as FITS or WRONG"}
    attempts = [{"stage": "wrong_disease_dev", "answers": len(rows), "claude_calls": len(rows),
                 "replies_without_a_call": sum(r["call"] is None for r in rows)}]
    return write_run("E9", DEV_SPLIT, config, {"grader": llm.model}, {"served": dev_metrics(rows)},
                     {"served": dev_intervals(rows)}, attempts, {"served": rows}, info.hash, root=root,
                     notes=("Development check of the wrong-disease grader on the final served rag_dev answers. "
                            "The prompt is frozen only after this and the blind read." + (" " + note if note else "")))


# ---------------------------------------------------------------- blind read, freeze

def latest_stage_record(root: Path, stage: str) -> tuple[Path, dict[str, Any]]:
    """The newest E9 rag_dev record of this stage, which must pass check_run."""
    for path in reversed(records_for("E9", root)):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["split"] == DEV_SPLIT and rec["config"].get("stage") == stage:
            problems = check_file(path, root)
            if problems:
                raise Refused(f"{path.name} does not pass check_run: {problems[0]}")
            return path, rec
    raise Refused(f"there is no {stage} record yet; run its stage first")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def blind_file(root: Path) -> Path:
    return Path(root) / "results" / "runs" / "E9-wrong-disease-blind.jsonl"


def stage_blind_sample(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    """Write the file a reader judges: a seeded sample of the dev answers with only the question, diagnosis and answer.
    The grader's call is not in it, and the order does not follow the call."""
    _, rec = latest_stage_record(root, "wrong_disease_dev")
    rows = _jsonl(Path(root) / rec["predictions"]["served"])
    sample = draw_blind(rows, cfg["wrong_disease"]["blind_seed"], cfg["wrong_disease"]["blind_per_call"])
    path = blind_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "".join(json.dumps({k: r[k] for k in ("item", "question", "diagnosis", "answer")}) + "\n"
                                    for r in sample))
    return path


def stage_unblind(cfg: dict[str, Any], root: Path, read_file: Path, reader: str, note: str = "") -> Path:
    """Join the reader's calls to the grader's, and record the error in both directions."""
    root = Path(root)
    require_clean(root)
    dev_path, dev = latest_stage_record(root, "wrong_disease_dev")
    fields = prompt_fields(root)
    if any(fields[k] != dev["config"][k] for k in ("prompt_sha256", "rendered_sha256")):
        raise Refused("the prompt has changed since the dev run; the blind read would measure a different prompt")
    wd = cfg["wrong_disease"]
    dev_rows = _jsonl(root / dev["predictions"]["served"])
    sample = draw_blind(dev_rows, wd["blind_seed"], wd["blind_per_call"])
    read = _jsonl(read_file)
    items = [r.get("item") for r in read]
    if len(set(items)) != len(items) or set(items) != {s["item"] for s in sample}:
        raise Refused("the reader's file must hold every item of the blind file exactly once, and no other")
    if any(r.get("reader") not in CALLS for r in read):
        raise Refused("every reader call must be FITS or WRONG")
    by_item = {r["item"]: r["reader"] for r in read}
    rows = [{"id": s["id"], "item": s["item"], "grader": s["call"], "reader": by_item[s["item"]]} for s in sample]
    strata = strata_sizes(dev_rows)
    config = {"stage": "wrong_disease_blind_read", **{k: dev["config"][k] for k in ("prompt_file", "prompt_sha256", "rendered_sha256")},
              "dev_record": dev["run_id"], "blind_seed": wd["blind_seed"], "blind_per_call": wd["blind_per_call"],
              "strata": strata, "reader": reader, "read_file": Path(read_file).name,
              "read_file_sha256": sha256_file(Path(read_file)), "claude_calls": 0, "device": WD_DEVICE,
              "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
              "started_at": datetime.now(timezone.utc).isoformat()}
    attempts = [{"stage": "wrong_disease_blind_read", "read_answers": len(rows), "reader": reader}]
    return write_run("E9", DEV_SPLIT, config, {"reader": reader, "grader": dev["models"]["grader"]},
                     {"served": blind_metrics(rows, strata)}, {"served": blind_intervals(rows)}, attempts,
                     {"served": rows}, split_info(DEV_SPLIT, root).hash, root=root,
                     notes=("A reader judged a seeded sample of the dev answers without the grader's call; the rates are "
                            "how often the reader disagrees with it, in each direction. The reader is one reader, not truth."
                            + (" " + note if note else "")))


def stage_freeze(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    """Record the prompt's commit and sha256 in a config to be committed. Needs the dev run and the blind read, both
    made with this very prompt."""
    root = Path(root)
    if (root / FROZEN_REL).exists():
        raise Refused(f"{FROZEN_REL} already frozen; a frozen prompt is not changed")
    require_committed(root, [PROMPT_REL])
    require_clean(root)
    _, dev = latest_stage_record(root, "wrong_disease_dev")
    _, blind = latest_stage_record(root, "wrong_disease_blind_read")
    if blind["config"]["dev_record"] != dev["run_id"]:
        raise Refused("the blind read belongs to another dev run")
    fields = prompt_fields(root)
    if any(fields[k] != dev["config"][k] for k in ("prompt_sha256", "rendered_sha256")):
        raise Refused("the prompt has changed since the dev run and the blind read measured it")
    commit = git(root, "log", "-1", "--format=%H", "--", PROMPT_REL).stdout.strip()
    frozen = {"experiment": "E9", **fields, "prompt_commit": commit, "grader_model": cfg["rag"]["grader"],
              "dev_record": dev["run_id"], "blind_read_record": blind["run_id"],
              "frozen_at": datetime.now(timezone.utc).isoformat(),
              "rule": "rag_test runs the wrong-disease check only with this prompt; it is not tuned after rag_test"}
    path = root / FROZEN_REL
    atomic_write_text(path, json.dumps(frozen, indent=1) + "\n")
    return path


def require_frozen(root: Path = ROOT) -> dict[str, Any]:
    """Refuse unless the frozen config and the prompt are committed, the prompt is the frozen one, and the frozen
    commit holds it."""
    root = Path(root)
    require_committed(root, [FROZEN_REL, PROMPT_REL])
    frozen = json.loads((root / FROZEN_REL).read_text(encoding="utf-8"))
    now = prompt_fields(root)
    for key in ("prompt_sha256", "rendered_sha256"):
        if now[key] != frozen[key]:
            raise Refused(f"the wrong-disease prompt ({key}) differs from the frozen one; it is not tuned after rag_test")
    shown = subprocess.run(["git", "show", f"{frozen['prompt_commit']}:{PROMPT_REL}"], cwd=root, capture_output=True)
    if shown.returncode != 0 or hashlib.sha256(shown.stdout).hexdigest() != frozen["prompt_sha256"]:
        raise Refused("the frozen commit does not hold the frozen prompt")
    return frozen


# ---------------------------------------------------------------- rag_test

RAG_TEST = "rag_test"
ROLES = {"local": "upper figure", "sonnet": "conservative figure"}
SERVED = "served"
EMBEDDER, RERANKER = "BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L6-v2"
FINAL_CAVEAT = (
    "Held-out numbers, from one run with everything frozen before it. The two judges read the same claims with the same "
    "rubric. The local judge is the upper figure and Sonnet the conservative one: the local judge errs kind and Sonnet "
    "errs literal (docs/e14-adjudication.md), so the truth lies between them and neither is the share of correct "
    "claims. The wrong-disease check is a Sonnet grader whose error was read blind on rag_dev."
)


def guard_rag_test(root: Path, override_note: str) -> dict[str, Any]:
    """Every rag_test stage starts here: configs and prompt committed, tree clean, prompt equal to the frozen one,
    split unused."""
    root = Path(root)
    require_committed(root, [CONFIG_REL, FROZEN_REL, PROMPT_REL])
    require_clean(root)
    frozen = require_frozen(root)
    require_unused(root, RAG_TEST, override_note)
    return frozen


def rag_questions(root: Path) -> list[Question]:
    return [q for q in load_questions(Path(root) / RAG_REL) if q.split == "test"]


def serve_config(sources: dict | None) -> AnswerConfig:
    """The setup E7 chose for serving (sprout.rag.serving), with nothing set here."""
    return AnswerConfig(sources=sources)


def _require_model(llm: Any, want: str, role: str) -> None:
    if llm.model != want:
        raise Refused(f"the {role} is {llm.model!r} but the config names {want!r}")


def _retrieval_checkpoint(cfg: dict[str, Any], root: Path) -> Checkpoint:
    index_dir = Path(root) / cfg["rag"]["index_dir"]
    key = e7.checkpoint_key(asdict(e7.replace_sources(serve_config(None))), serving.FLOORS, e7.index_stamp(index_dir),
                            split_info(RAG_TEST, root).hash)
    return Checkpoint(Path(root) / WORK_REL / "retrieve" / f"retrieve-{key}.jsonl")


def stage_rag_retrieve(cfg: dict[str, Any], retriever: Any, root: Path = ROOT, *, override_note: str = "") -> Checkpoint:
    """The passages each rag_test question gets under the served retrieval, saved per question."""
    guard_rag_test(root, override_note)
    ck, classes = _retrieval_checkpoint(cfg, root), e7.load_classes(root)
    for q in rag_questions(root):
        if q.id not in ck.rows:
            ck.add(e7.retrieval_for_answers(retriever, serve_config(None), q, retriever.crops, classes))
    return ck


def _generation_key(cfg: dict[str, Any], ck: Checkpoint) -> str:
    return e7.checkpoint_key(serve_config(None).prompt_version, ck.path.name, cfg["rag"]["writer"], e7.ANSWER_VERSION)


def stage_rag_generate(cfg: dict[str, Any], llm: Any, root: Path = ROOT, sources: dict | None = None, *,
                       override_note: str = "") -> list[str]:
    """The local writer's answers with the served settings. Returns the ids still without an answer (an outage)."""
    guard_rag_test(root, override_note)
    _require_model(llm, cfg["rag"]["writer"], "writer")
    questions, ck = rag_questions(root), _retrieval_checkpoint(cfg, root)
    if any(q.id not in ck.rows for q in questions):
        raise Refused("run rag-retrieve first: the retrieval is incomplete")
    crops = {r["crop"] for r in ck.rows.values() if r["crop"]} | {GENERAL_CROP}
    single = e7.single_page_sources(Path(root) / cfg["rag"]["index_dir"])
    _, missing = e7.run_generation(SERVED, questions, e7.cached_retriever(crops, ck.rows), llm, serve_config(sources),
                                   Path(root) / WORK_REL / "generate", single, root, key=_generation_key(cfg, ck))
    return missing


def _generated(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    ck = _retrieval_checkpoint(cfg, root)
    done = Checkpoint(Path(root) / WORK_REL / "generate" / f"generate-{SERVED}-{_generation_key(cfg, ck)}.jsonl")
    rows = [done.rows.get(q.id) for q in rag_questions(root)]
    if None in rows:
        raise Refused("run rag-generate first: some questions have no answer")
    return rows


def stage_rag_judge_local(cfg: dict[str, Any], llm: Any, root: Path = ROOT, *, override_note: str = "") -> Path:
    """Per-claim faithfulness by the local judge, through the code E7 used. Returns the judgments file."""
    guard_rag_test(root, override_note)
    _require_model(llm, cfg["rag"]["judge"], "local judge")
    rows = _generated(cfg, root)
    qmap = {q.id: q for q in rag_questions(root)}
    judged = e7.run_judging(SERVED, rows, qmap, llm, Path(root) / WORK_REL / "judge", key=_judge_key(cfg, root))
    return e7.write_judgments(Path(root) / WORK_REL / "local", judged)[0]


def _judge_key(cfg: dict[str, Any], root: Path) -> str:
    ck = _retrieval_checkpoint(cfg, root)
    return e7.checkpoint_key(_generation_key(cfg, ck), cfg["rag"]["judge"], rubric())


def _local_judgments(cfg: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], Path, dict[str, dict[str, Any]]]:
    rows = _generated(cfg, root)
    done = Checkpoint(Path(root) / WORK_REL / "judge" / f"judge-{SERVED}-{_judge_key(cfg, root)}.jsonl")
    if any(r["id"] not in done.rows for r in rows):
        raise Refused("run rag-judge-local first: some answers are not judged")
    judged = [done.rows[r["id"]] for r in rows if done.rows[r["id"]]["claims"]]
    full, blind = e7.write_judgments(Path(root) / WORK_REL / "local", judged)
    return judged, full, {b["item"]: b for b in _jsonl(blind)}


def _test_items(cfg: dict[str, Any], root: Path, judged: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qmap = {q.id: q for q in rag_questions(root)}
    crops, classes = index_crops(Path(root) / cfg["rag"]["index_dir"]), e7.load_classes(root)
    text = {r["id"]: r["text"] for r in rows}
    return [{"id": j["id"], "question": qmap[j["id"]].question, "diagnosis": diagnosis_for(qmap[j["id"]], crops, classes),
             "answer": text[j["id"]]} for j in judged]


def _claims_checkpoint(model: str, cfg14: dict[str, Any], root: Path) -> Checkpoint:
    return Checkpoint(Path(root) / WORK_REL / "claims" / f"claims-{e14.run_key(cfg14, model)}.jsonl")


def stage_rag_sonnet(cfg: dict[str, Any], llm: Any, root: Path = ROOT, *, override_note: str = "") -> None:
    """Sonnet grades the same claims with the same rubric (E14's code), then runs the frozen wrong-disease check once per
    answer. The cap and the stage limit are checked for both together before any call."""
    guard_rag_test(root, override_note)
    _require_model(llm, cfg["rag"]["grader"], "Sonnet grader")
    judged, full, blind = _local_judgments(cfg, root)
    limit = cfg["rag"]["max_sonnet_calls"]
    sample = [{"item": e14.item_of(SERVED, j["id"]), "id": j["id"], "system": SERVED, "stratum": SERVED} for j in judged]
    cfg14 = {"seed": 0, "sample_size": len(sample), "served_system": SERVED, "max_calls": limit,
             "source": {"judgments_sha256": sha256_file(full)}}
    items = _test_items(cfg, root, judged, _generated(cfg, root))
    claims_done = _claims_checkpoint(llm.model, cfg14, root).rows
    wd_path = Path(root) / WORK_REL / f"wd-test-{wd_key(items, llm.model, root)}.jsonl"
    wd_done = Checkpoint(wd_path).rows
    need = sum(f"{e['item']}:{k}" not in claims_done for e in sample for k in range(len(blind[e["item"]]["claims"])))
    need += sum(i["id"] not in wd_done for i in items)
    check_budget(llm, need, limit, len(claims_done) + len(wd_done))
    e14.grade(sample, blind, llm, Path(root) / WORK_REL / "claims", cfg14)
    wd_grade(items, llm, "test", root)


def final_rows(cfg: dict[str, Any], llm_model: str, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(rows in question order, generation rows) with both judges' calls on every claim and the wrong-disease call."""
    judged, full, _ = _local_judgments(cfg, root)
    gen = _generated(cfg, root)
    by_id = {j["id"]: j for j in judged}
    cfg14 = {"seed": 0, "sample_size": len(judged), "served_system": SERVED, "max_calls": cfg["rag"]["max_sonnet_calls"],
             "source": {"judgments_sha256": sha256_file(full)}}
    claims = _claims_checkpoint(llm_model, cfg14, root).rows
    items = _test_items(cfg, root, judged, gen)
    wd = Checkpoint(Path(root) / WORK_REL / f"wd-test-{wd_key(items, llm_model, root)}.jsonl").rows
    diagnosis = {i["id"]: i["diagnosis"] for i in items}
    qmap = {q.id: q for q in rag_questions(root)}
    crops, classes = index_crops(Path(root) / cfg["rag"]["index_dir"]), e7.load_classes(root)
    rows = []
    for g in gen:
        j = by_id.get(g["id"], {"claims": []})
        item = e14.item_of(SERVED, g["id"])
        got = []
        for k, c in enumerate(j["claims"]):
            r = claims.get(f"{item}:{k}")
            if r is None:
                raise ValueError("some claims are not graded by Sonnet yet; run the rag-sonnet stage")
            got.append({"claim": c["claim"], "cited": c["cited"], "local": bool(c["verdict"]), "sonnet": r["sonnet"],
                        "sonnet_reply": r["reply"]})
        check = wd.get(g["id"]) if got else None
        if got and check is None:
            raise ValueError("some answers are not wrong-disease checked yet; run the rag-sonnet stage")
        rows.append({"id": g["id"], "answered": bool(g["answered"]), "state": g["state"],
                     "diagnosis": diagnosis.get(g["id"]) or diagnosis_for(qmap[g["id"]], crops, classes), "text": g["text"],
                     "claims": got, "wrong_disease": check["call"] if check else None,
                     "wrong_disease_reply": check["reply"] if check else None,
                     "sonnet_calls": len(got) + (1 if check else 0)})
    return rows, gen


def stage_rag_record(cfg: dict[str, Any], root: Path = ROOT, *, device: str, note: str = "",
                     override_note: str = "") -> Path:
    """Write the one rag_test record from the saved stages. Refuses unless every answer, claim and check is saved."""
    frozen = guard_rag_test(root, override_note)
    root = Path(root)
    rows, gen = final_rows(cfg, cfg["rag"]["grader"], root)
    info = split_info(RAG_TEST, root)
    if [r["id"] for r in rows] != info.ids:
        raise Refused("the saved answers do not cover rag_test in order")
    answered = np.array([r["answered"] for r in rows])
    scored = Scored(groups=info.ids, answered=answered, expected=info.expected)
    metrics = {SERVED: {m: recompute_metric(m, scored) for m in e7.BEHAVIOUR_METRICS}}
    calls = sum(r["sonnet_calls"] for r in rows)
    ac = serve_config(None)
    usage = {"rows": len(gen), "model_calls": sum(g.get("llm_calls", 0) for g in gen),
             "rows_without_model_call": sum(g.get("llm_calls", 0) == 0 for g in gen),
             "model_seconds": round(sum(g.get("llm_seconds", 0.0) for g in gen), 1)}
    config = {
        "stage": "final", "served": {"prompt_version": ac.prompt_version, "mode": ac.mode, "rerank": ac.rerank, "k": ac.k,
                                     "query_mode": ac.query_mode, "floors": serving.FLOORS},
        "judge": {"model": cfg["rag"]["judge"]},
        "judges": {"local": {"model": cfg["rag"]["judge"], "role": ROLES["local"]},
                   "sonnet": {"model": cfg["rag"]["grader"], "role": ROLES["sonnet"]},
                   "same_claims_and_rubric": True},
        "rubric": rubric(), **prompt_fields(root), "frozen_wrong_disease": {"file": FROZEN_REL, **frozen},
        "diagnosis_source": DIAGNOSIS_RULE, "claude_calls": calls, "device": device, "writer_usage": usage,
        "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
        "basis": "rag_test questions as asked; the photo label cleaning does not apply to them",
        "started_at": datetime.now(timezone.utc).isoformat(), "caveat": FINAL_CAVEAT,
        **({"override_note": override_note.strip()} if override_note.strip() else {}),
    }
    models = {"writer": cfg["rag"]["writer"], "judge_upper_figure": cfg["rag"]["judge"],
              "judge_conservative_figure": cfg["rag"]["grader"], "wrong_disease_grader": cfg["rag"]["grader"],
              "embedder": EMBEDDER, "reranker": RERANKER}
    attempts = [{"stage": "rag_test", "answers": len(rows), "claims": sum(len(r["claims"]) for r in rows),
                 "claude_calls": calls}]
    return write_run("E9", RAG_TEST, config, models, metrics, {SERVED: {}}, attempts, {SERVED: rows}, info.hash,
                     root=root, judged=final_lists(rows), notes=FINAL_CAVEAT + (" " + note if note else ""))


def report(rec: dict[str, Any], root: Path = ROOT) -> str:
    """The figures of a rag_test record in words. The two judges are named as the upper and the conservative figure."""
    sidecar = json.loads((Path(root) / rec["judged"]["file"]).read_text(encoding="utf-8"))
    judges = rec["config"]["judges"]

    def share(name: str, label: str) -> str:
        v = sidecar[name]
        return f"{label}: {sum(v)} of {len(v)} claims supported = {sum(v) / len(v):.3f}"

    flagged = sidecar.get("answers_flagged_wrong_disease", [])
    tag = "[rag_test questions, no photo cleaning applies]"
    lines = [f"{tag} {rec['n']} questions, one run, configurations frozen before it. Label review is signed in "
             f"{SIGNOFF_REL}, not in this record.",
             share("claims_supported_by_local_judge",
                   f"local judge {judges['local']['model']} ({judges['local']['role']})"),
             share("claims_supported_by_sonnet", f"Sonnet ({judges['sonnet']['role']})"),
             "The two read the same claims with the same rubric; the truth lies between the figures.",
             f"wrong-disease check (Sonnet, prompt frozen before this run): {sum(flagged)} of {len(flagged)} checked "
             "answers flagged as advice for the wrong disease or plant"]
    lines += [f"{k}: {v:.3f}" if v is not None else f"{k}: undefined" for k, v in rec["metrics"][SERVED].items()]
    return "\n".join(x if x.startswith(tag) else f"{tag} {x}" for x in lines)


# ---------------------------------------------------------------- command line (wires in the real models)

STAGES = ("lab-test", "field-test", "wrong-disease-dev", "blind-sample", "unblind", "freeze", "rag-retrieve",
          "rag-generate", "rag-judge-local", "rag-sonnet", "rag-record")
EXIT_INCOMPLETE = 2


def _device_file(root: Path) -> Path:
    return Path(root) / WORK_REL / "device.json"


def _remember_device(root: Path, role: str, info: dict[str, Any]) -> None:
    path = _device_file(root)
    known = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({**known, role: info}, indent=1) + "\n")


def device_text(root: Path = ROOT) -> str:
    """Where retrieval, the writer and the local judge ran, as recorded by their stages. Refuses if a stage left none."""
    path = _device_file(root)
    known = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    missing = [r for r in ("retrieval", "writer", "judge") if r not in known]
    if missing:
        raise Refused(f"no device was recorded for {missing}; run those stages first")
    r, w, j = known["retrieval"], known["writer"], known["judge"]
    return (f"retrieval on {r['device']}; writer {w['model']} with {w['size_vram']} of {w['size']} bytes in GPU memory; "
            f"judge {j['model']} with {j['size_vram']} of {j['size']} bytes in GPU memory")


def _on_gpu(client: Any, role: str, root: Path):
    """Load the Ollama model with a tiny call, record where it sits (stopping if it is on the CPU), and unload it after."""
    import contextlib

    @contextlib.contextmanager
    def held():
        try:
            client.complete([{"role": "user", "content": "Reply with the word ok."}], max_tokens=8, timeout_s=300)
            place = client.placement()
            if place is None or place["size_vram"] <= 0:
                raise Refused(f"{client.model} is not on the GPU: {place}")
            _remember_device(root, role, {"model": client.model, **place})
            yield
        finally:
            with contextlib.suppress(LLMError):
                client.unload()

    return held()


def _real_retriever(cfg: dict[str, Any], root: Path):
    from sprout.rag.gpu_models import TorchEmbedder, TorchReranker
    from sprout.rag.retrieve import Retriever

    config = serving.retrieval_config()
    embedder = TorchEmbedder(config.embedding_model, device=cfg["rag"]["device"])
    reranker = TorchReranker(device=cfg["rag"]["device"])
    _remember_device(root, "retrieval", {**embedder.info, **reranker.info})
    return Retriever(Path(root) / cfg["rag"]["index_dir"], config=config, embedder=embedder, reranker=reranker)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from sprout.llm.client import ClaudeCodeClient, OllamaClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--override-note", default="", help="why a test split is run again (a later record needs one)")
    ap.add_argument("--note", default="")
    ap.add_argument("--read-file", help="unblind: the reader's file, one {item, reader} per line")
    ap.add_argument("--reader", default="", help="unblind: who read the blind file")
    a = ap.parse_args(argv)
    try:
        cfg, root, stage = load_config(ROOT), ROOT, a.stage
        grader = lambda: ClaudeCodeClient(model=cfg["rag"]["grader"])  # noqa: E731
        if stage in ("lab-test", "field-test"):
            split = stage.replace("-", "_")
            paths = run_images(split, cfg, lambda: build_predictors(cfg, load_decision(cfg, root), root), root,
                               override_note=a.override_note)
            print("\n".join(str(p) for p in paths))
            print(image_report(json.loads(paths[0].read_text(encoding="utf-8"))))
        elif stage == "wrong-disease-dev":
            print(stage_wrong_disease_dev(cfg, grader(), root, a.note))
        elif stage == "blind-sample":
            print(stage_blind_sample(cfg, root))
        elif stage == "unblind":
            if not (a.read_file and a.reader.strip()):
                raise Refused("unblind needs --read-file and --reader")
            print(stage_unblind(cfg, root, Path(a.read_file), a.reader.strip(), a.note))
        elif stage == "freeze":
            print(stage_freeze(cfg, root))
        elif stage == "rag-retrieve":
            guard_rag_test(root, a.override_note)
            print(f"{len(stage_rag_retrieve(cfg, _real_retriever(cfg, root), root, override_note=a.override_note).rows)} questions")
        elif stage in ("rag-generate", "rag-judge-local"):
            guard_rag_test(root, a.override_note)
            role = "writer" if stage == "rag-generate" else "judge"
            client = OllamaClient(cfg["rag"][role], keep_alive="10m")
            with _on_gpu(client, role, root):
                if role == "writer":
                    from sprout.rag.answer import load_sources

                    missing = stage_rag_generate(cfg, client, root, load_sources(root / "knowledge" / "manifest.csv"),
                                                 override_note=a.override_note)
                    print(f"{len(rag_questions(root)) - len(missing)} answers, {len(missing)} missing")
                    if missing:
                        print("Some questions got no model answer (Ollama down or timing out). Run it again to resume.")
                        return EXIT_INCOMPLETE
                else:
                    print(stage_rag_judge_local(cfg, client, root, override_note=a.override_note))
        elif stage == "rag-sonnet":
            stage_rag_sonnet(cfg, grader(), root, override_note=a.override_note)
            print("Sonnet judged the claims and ran the wrong-disease check")
        else:
            path = stage_rag_record(cfg, root, device=device_text(root), note=a.note, override_note=a.override_note)
            print(path)
            print(report(json.loads(path.read_text(encoding="utf-8")), root))
    except (Refused, LLMError, RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
