"""E9: the final test. Each test split is scored once, with configurations frozen and committed before.

Stages (see docs/PLAN.md T8.1; every stage that touches a test split refuses unless the configs are committed, the
tree is clean and the split has not been used):
    lab-test, field-test        score every frozen system on one image test split and write its record; the model is
                                the one the final-model decision (configs/e9.json decision_file) names
    wrong-disease-dev           the wrong-disease check on the final E7 served rag_dev answers (Sonnet), with its rate
    freeze                      record the check's prompt commit and sha256 in configs/e9_wrong_disease.json; the check
                                is not validated by any blind read
    rag-retrieve, rag-generate  rag_test: served retrieval, then the local writer
    rag-haiku, rag-sonnet       the Haiku judge (named by results/e14/haiku_decision.json) and the Sonnet judge grade the
                                same claims; Sonnet also runs the frozen wrong-disease check
    rag-record                  the one rag_test record
No stage reads a test split unless it is the one being run. Tests use fakes and tiny fixtures only.
"""

from __future__ import annotations

import hashlib
import json
import re
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
from sprout.eval import e9_roles
from sprout.eval.answer_checks import cleaned_rows as cleaned
from sprout.eval.answer_checks import (
    dev_intervals,
    dev_metrics,
    exif_counts,
    final_lists,
    parse_call,
    wd_counts,
)
from sprout.eval.check_run import (
    Scored,
    check_file,
    recompute_comparison,
    recompute_interval,
    recompute_metric,
    records_for,
)
from sprout.eval.claims import signoff_problems
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
from sprout.llm.client import LLMError, SafeguardRefusal
from sprout.rag import serving
from sprout.rag.answer import GENERAL_CROP, AnswerConfig, diagnosis_summary
from sprout.rag.evalset import Question, load_questions
from sprout.rag.evalset import index_crops as index_crops_of
from sprout.rag.judge import rubric, split_claims

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
    """The final-model decision, read by path. Without it the chosen model is unknown, so nothing runs."""
    path = Path(root) / cfg["decision_file"]
    if not path.is_file():
        raise Refused(f"{cfg['decision_file']} does not exist: no final model has been chosen, so E9 cannot start")
    e15 = Path(root) / cfg["e15_config"]
    if not e15.is_file():
        raise Refused(f"{cfg['e15_config']} does not exist, so the refit calibration T4.9 defines cannot be located")
    try:
        return e9_roles.normalized(json.loads(path.read_text(encoding="utf-8")), json.loads(e15.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise Refused(f"{cfg['decision_file']}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CRLF, LF = bytes([13, 10]), bytes([10])


def text_sha256(path: Path) -> str:
    """Hash of a text file as git stores it (LF endings), so it equals the hash of `git show <commit>:<path>` on a machine
    that checks files out with CRLF."""
    return hashlib.sha256(Path(path).read_bytes().replace(CRLF, LF)).hexdigest()


# ---------------------------------------------------------------- the wrong-disease prompt

def prompt_text(root: Path = ROOT) -> str:
    return (Path(root) / PROMPT_REL).read_text(encoding="utf-8").strip()


def wd_messages(question: str, diagnosis: str, answer: str, root: Path = ROOT) -> list[dict[str, str]]:
    """What the grader is shown: the question, the diagnosis and the answer. No passages, no faithfulness grade."""
    from sprout.experiments.e14_judge_check import clean

    body = f"Question: {clean(question)}\n\nDiagnosis: {clean(diagnosis)}\n\nAnswer: {clean(answer)}"
    return [{"role": "system", "content": prompt_text(root)}, {"role": "user", "content": body}]


# ---------------------------------------------------------------- image test splits

BASE_SYSTEMS = ("baseline", "improved", "xgboost", "dinov2", "quantized")
ORDER = ("baseline", "improved", "alternate", "alternate_2", "xgboost", "dinov2", "quantized")
FLOOR = 1e-12
INTERVALS = ("accuracy", "macro_f1")


@dataclass
class Predictor:
    """One frozen system: its class order, and `run(frame)`, which returns probabilities for the frame's photos
    (columns `path`, absolute, and `label`, a class name)."""

    description: str
    class_names: list[str]
    run: Callable[[pd.DataFrame], np.ndarray]
    detail: dict[str, Any] = field(default_factory=dict)


def candidate_roles(cfg: dict[str, Any], decision: dict[str, Any]) -> tuple[str, list[str]]:
    """(primary key, alternate keys): the primary is the model the decision chose, every other candidate is reported
    beside it. A chosen model that is none of the config's candidates needs the decision to name its calibration."""
    try:
        return e9_roles.roles(cfg["candidates"], decision)
    except ValueError as exc:
        raise Refused(str(exc)) from exc


def specs_of(cfg: dict[str, Any], decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidate_roles(cfg, decision)
    return e9_roles.candidate_specs(cfg["candidates"], decision)


def load_calibration(rel: str, model_sha: str | None, root: Path = ROOT, *, required: bool = False) -> dict[str, Any]:
    """Temperature and abstention threshold from an E5-style chosen.json. When the file names the model it was fitted
    on (`model_sha256`), that must be the model scored; `required` demands the name (the decision says E5 was refit)."""
    path = Path(root) / rel
    if not path.is_file():
        raise Refused(f"{rel} does not exist: the calibration and threshold are not chosen")
    saved = json.loads(path.read_text(encoding="utf-8"))
    named = saved.get("model_sha256")
    if named is None and required:
        raise Refused(f"the decision says E5 was refit on the chosen model, but {rel} does not name that model (model_sha256)")
    if named is not None and named != model_sha:
        raise Refused(f"{rel} was fitted on another model than the one scored (model_sha256 differs)")
    out = {"file": rel, "temperature": float(saved["temperature"]), "abstain_threshold": float(saved["abstain_threshold"])}
    if saved.get("healthy_threshold") is not None:  # only a refit that defined one; E4's and E12's files have none
        out["healthy_threshold"] = float(saved["healthy_threshold"])
    return out


def calibrations(cfg: dict[str, Any], decision: dict[str, Any], root: Path = ROOT, *,
                 alternate_sha: str | None = None, alternate_2_sha: str | None = None,
                 scored: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    """Calibration per calibrated system: the primary model's own (also used for the quantized copy of it) and each
    alternate's own, so every model is reported with and without abstention."""
    specs = specs_of(cfg, decision)
    primary, others = candidate_roles(cfg, decision)
    prim = load_calibration(specs[primary]["calibration"], decision["chosen_model_sha256"], root, required=True)
    out = {"improved": prim, "quantized": prim}
    for i, (key, sha) in enumerate(zip(others, (alternate_sha, alternate_2_sha), strict=False)):
        if scored is not None and e9_roles.system_of(i) not in scored:
            continue  # its saved model is not on disk and it is not scored, so its calibration is not needed
        if specs[key].get("calibration") and (Path(root) / specs[key]["calibration"]).is_file():
            out[e9_roles.system_of(i)] = load_calibration(specs[key]["calibration"], sha, root)
    return out


NO_OWN_CALIBRATION = ("the alternate has no calibration of its own (E5 was not refit on it, or the file cannot be tied to "
                      "it), and another model's calibration is not applied to it, so it is scored without abstention only")


def path_bound(checkpoint: Path, root: Path, chosen: str, e5_config: str = "configs/e5.json") -> tuple[str, str]:
    """The checkpoint's sha256 and how a calibration file with no `model_sha256` of its own (E4's, which E5 wrote before
    it recorded one) is tied to it: the checkpoint is the path configs/e5.json says E5 used, and a committed E5 record
    of the same E4 record holds the file's temperature and threshold. ValueError when that cannot be shown."""
    checkpoint = Path(checkpoint).resolve()
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fitted = json.loads((root / chosen).read_text(encoding="utf-8"))
    model = json.loads((root / e5_config).read_text(encoding="utf-8"))["model"]
    if (root / model["checkpoint"]).resolve() != checkpoint:
        raise ValueError(f"{checkpoint} is not the checkpoint E5 used ({model['checkpoint']} in {e5_config})")
    shown = [p for p in sorted((root / "results" / "runs").glob("E5-*.json"))
             if (c := json.loads(p.read_text(encoding="utf-8")).get("config", {}))
             and c.get("model_record") == model["e4_record"] and c.get("temperature") == fitted["temperature"]
             and c.get("abstain_threshold") == fitted["abstain_threshold"]]
    if not shown:
        raise ValueError(f"no E5 record of E4 record {model['e4_record']} shows the temperature and threshold in "
                         f"{chosen}, so the file cannot be tied to that model")
    rel = checkpoint.relative_to(root.resolve()).as_posix() if checkpoint.is_relative_to(root.resolve()) else str(checkpoint)
    return sha, (f"path: {rel} is the checkpoint configs/e5.json names, and {shown[0].name} shows these numbers for E4 "
                 f"record {model['e4_record']}; E5 stored no checksum when it was fitted")


def bindings(cfg: dict[str, Any], decision: dict[str, Any], calibration: dict[str, dict[str, Any]],
             shas: dict[str, str | None], quantized_source: str | None, root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Which checkpoint each calibration is bound to, shown by the file's own hash or, for E5's file that carries none,
    by `path_bound`. A calibration fitted for another checkpoint than the one it is applied to is refused. An
    alternate with no calibration of its own is dropped from `calibration` and says why. The quantized file goes the
    way serving would take it: its stamped source checkpoint must be the one the primary's calibration is bound to.
    `shas` holds the sha256 of each scored model by system name."""
    root = Path(root)
    specs = specs_of(cfg, decision)
    primary, others = candidate_roles(cfg, decision)
    out: dict[str, dict[str, Any]] = {}
    scored_roles = [("improved", primary), *[(e9_roles.system_of(i), k) for i, k in enumerate(others)]]
    for role, key in scored_roles:
        if role not in shas:
            continue
        spec, scored = specs[key], shas[role]
        if role not in calibration:
            out[role] = {"calibration_file": None, "checkpoint": spec["checkpoint"], "sha256": scored,
                         "why": NO_OWN_CALIBRATION}
            continue
        saved = json.loads((root / spec["calibration"]).read_text(encoding="utf-8"))
        try:
            if saved.get("model_sha256"):
                found, by = saved["model_sha256"], "the file names this checkpoint's sha256"
            else:
                found, by = path_bound(root / spec["checkpoint"], root, spec["calibration"])
        except (ValueError, OSError) as exc:
            if role != "improved":
                del calibration[role]
                out[role] = {"calibration_file": None, "checkpoint": spec["checkpoint"], "sha256": scored,
                             "why": f"{NO_OWN_CALIBRATION}: {exc}"}
                continue
            raise Refused(f"{spec['calibration']} cannot be tied to {spec['checkpoint']}: {exc}") from exc
        if found != scored:
            raise Refused(f"{spec['calibration']} was fitted for checkpoint {found}, which is not the {role} model "
                          f"scored ({scored})")
        out[role] = {"calibration_file": spec["calibration"], "checkpoint": spec["checkpoint"], "sha256": found, "bound_by": by}
    if quantized_source != out["improved"]["sha256"]:
        raise Refused(f"the quantized file was exported from checkpoint {quantized_source or '(none stamped)'}, but the "
                      f"calibration it is scored with is bound to {out['improved']['sha256']}")
    out["quantized"] = {**out["improved"], "bound_by": f"the quantized file is stamped as exported from checkpoint "
                                                       f"{quantized_source}, the one the calibration is bound to; "
                                                       f"{out['improved']['bound_by']}"}
    return out


def e6_thresholds(cfg: dict[str, Any], decision: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """E6's own account of how E5's temperature and threshold behave on the quantized variant scored here: read from the
    latest E6 record made from the chosen checkpoint, not typed."""
    variant = cfg["systems"]["quantized"]["variant"]
    found = []
    for path in sorted((Path(root) / "results" / "runs").glob("E6-*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        held = rec.get("config", {}).get("thresholds")
        if rec.get("experiment") == "E6" and rec["config"].get("checkpoint_sha256") == decision["chosen_model_sha256"] and held:
            found.append((rec, held))
    if not found:
        raise Refused("no E6 record of the chosen checkpoint states how E5's thresholds behave on the quantized variant; "
                      "run E6 on the chosen model first")
    rec, held = found[-1]
    if variant not in held["systems"]:
        raise Refused(f"E6 record {rec['run_id']} has no thresholds for the variant {variant}")
    return {"e6_record": rec["run_id"], "variant": variant, "e6_split": rec["split"], "file": held["file"],
            "temperature": held["temperature"], "abstain_threshold": held["abstain_threshold"],
            "names_this_checkpoint": held["names_this_checkpoint"], "on_e6_rows": held["systems"][variant]["scoring_part"]}


def prior_uses(root: Path, split: str) -> list[str]:
    """Every record that has ever existed on this test split, committed or not, whatever its experiment."""
    from sprout.eval.check_run import _uses

    return [stem for stem, rec in _uses(root) if rec["split"] == split]


def require_unused(root: Path, split: str, override_note: str) -> None:
    used = prior_uses(root, split)
    if used and not override_note.strip():
        raise Refused(f"{split} has already been used ({used[0]}); a later run needs --override-note saying why")


def statements(split: str, thresholds: dict[str, Any], exif: dict[str, int], root: Path, rows: str = "scored") -> str:
    """What the record has to say besides its numbers, from data: how E5's thresholds behaved on the quantized variant in
    E6, which side of the EXIF difference the rows fall on, and whether the labels are signed yet."""
    on = thresholds["on_e6_rows"]
    out = (f"Quantized variant {thresholds['variant']}: E6 record {thresholds['e6_record']} ({thresholds['e6_split']}) "
           f"measured the abstention threshold {thresholds['abstain_threshold']} ({thresholds['file']}) on it, on E6's "
           f"scoring part of {on['n']} photos: answered {on['answered']} (coverage {on['coverage']:.3f}), "
           f"{on['confident_wrong']} of them wrong, {on['sick_called_healthy']} sick plants called healthy. Evaluation does not apply EXIF rotation and serving does: "
           f"{exif['with_orientation_tag']} of {exif['rows']} {split} rows {rows} carry an orientation tag, "
           f"{exif['rotated_by_serving']} of them one that serving would rotate.")
    if signoff_problems(Path(root), split):
        out += (f" Provisional: the {split} labels are not signed in {SIGNOFF_REL}, so these results are provisional "
                "until the person signs.")
    return out


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


ABSTENTION_RULE = (
    "a *_calibrated system answers only when its calibrated top probability is at least the abstain threshold; the "
    "other systems answer every photo. Where the calibration file has a healthy_threshold (T4.9's refit, for the "
    "primary and its quantized copy) the metrics named coverage@T~H, selective_accuracy@T~H and "
    "confident_error_rate@T~H apply what Identifier serves, a top label ending in 'healthy' must also reach "
    "max(T, H), through sprout.serve.calibrate.answered_mask; the metrics named @T are abstention alone, so the cost "
    "of the healthy bar in unsure answers is the difference. E4's and E12's files define no healthy threshold, so "
    "those alternates use abstention alone."
)


def threshold_text(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def score_block(probs: dict[str, np.ndarray], labels: np.ndarray, groups: list[str], names: list[str],
                thresholds: dict[str, float], keep: np.ndarray,
                healthy: dict[str, float] | None = None) -> tuple[dict, dict, list]:
    """Metrics, intervals and paired comparisons of every system over the rows in `keep`, the way check_run recomputes."""
    scored = {n: Scored(groups=[g for g, k in zip(groups, keep, strict=True) if k], labels=labels[keep],
                        class_names=names, probs=p[keep]) for n, p in probs.items()}
    metrics = {n: standard_image_metrics(s.probs, s.labels, names) for n, s in scored.items()}
    for n in scored:
        if n.endswith("_calibrated"):
            base = n.removesuffix("_calibrated")
            t = threshold_text(thresholds[base])
            for m in ("coverage", "selective_accuracy", "confident_error_rate"):
                metrics[n][f"{m}@{t}"] = recompute_metric(f"{m}@{t}", scored[n])  # abstention alone
                if base in (healthy or {}):  # abstention as Identifier serves it
                    h = threshold_text(healthy[base])
                    metrics[n][f"{m}@{t}~{h}"] = recompute_metric(f"{m}@{t}~{h}", scored[n])
    metrics["quantized"]["changed_vs@improved"] = recompute_metric("changed_vs@improved", scored["quantized"], scored)
    intervals = {n: {m: recompute_interval(m, s) for m in INTERVALS} for n, s in scored.items()}
    comparisons = []
    alts = [n for n in ("alternate", "alternate_2") if n in scored]
    for a, b in (("improved", "baseline"), *[("improved", n) for n in alts], ("improved", "xgboost"),
                 ("improved", "dinov2"), ("quantized", "improved")):
        for m in INTERVALS:
            got = recompute_comparison(scored[a], scored[b], m)
            comparisons.append({"a": a, "b": b, "metric": m, **got})
    return metrics, intervals, comparisons


def run_images(split: str, cfg: dict[str, Any], predictors: dict[str, Predictor] | Callable[[], dict[str, Predictor]],
               root: Path = ROOT, *,
               override_note: str = "") -> list[Path]:
    """Score every frozen system once on `split` and write its record: lab_test gets a full record and a view on the
    shared classes, field_test gets one record on its cleaned rows. Only rows whose true class is in the decision's
    class list are scored, by every system. Refuses before any model runs unless the config is committed, the tree is
    clean, the final model is decided, and the split is unused (or an override note says why)."""
    if split not in IMAGE_TESTS:
        raise Refused(f"E9 scores lab_test or field_test here, not {split!r}")
    root = Path(root)
    require_committed(root, [CONFIG_REL])
    require_clean(root)
    decision = load_decision(cfg, root)
    require_committed(root, [cfg["decision_file"], cfg["e15_config"], decision["calibration"]])
    primary, others = candidate_roles(cfg, decision)
    require_unused(root, split, override_note)
    if callable(predictors):  # built only after every guard has passed, so a refused run loads no model
        predictors = predictors()
    absent = [n for n in BASE_SYSTEMS if n not in predictors]
    if absent:
        raise Refused(f"frozen systems missing: {absent}")
    specs = specs_of(cfg, decision)
    alternates, not_scored = {}, []
    for i, key in enumerate(others):
        system = e9_roles.system_of(i)
        if system in predictors:
            alternates[system] = key
        elif (root / specs[key]["checkpoint"]).is_file():
            raise Refused(f"frozen systems missing: {[system]}")
        else:
            not_scored.append({"candidate": key, "checkpoint": specs[key]["checkpoint"],
                               "why": "the saved model is not on disk, so it is not scored"})
    systems = [n for n in ORDER if n in predictors and (n in BASE_SYSTEMS or n in alternates)]
    shas = {"improved": decision["chosen_model_sha256"],
            **{n: predictors[n].detail.get("sha256") for n in alternates}}
    calibration = calibrations(cfg, decision, root, alternate_sha=shas.get("alternate"),
                               alternate_2_sha=shas.get("alternate_2"), scored=tuple(alternates))
    binding = bindings(cfg, decision, calibration, shas,
                       predictors["quantized"].detail.get("source_checkpoint_sha256"), root)
    cal_names = [n for n in cfg["calibrated"] if n in calibration]
    names = decision["class_names"]
    for n in systems:
        if predictors[n].class_names != names:
            raise Refused(f"{n} has a different class order from the decision's class list")
    info = split_info(split, root)
    clean = f"clean:{cfg['exclusion_version']}" if split == "field_test" else None
    base = subset_positions(clean, split, info, root) if clean else list(range(len(info.ids)))
    positions = [i for i in base if info.labels[i] in set(names)]
    outside = sorted({info.labels[i] for i in base} - set(names))
    class_part = f"classes_in:{cfg['decision_file']}"
    subset = "+".join(x for x in (clean, class_part if outside else None) if x) or None
    labels_by_name = [info.labels[i] for i in positions]
    paths = [info.ids[i] for i in positions]
    groups = [info.groups[i] for i in positions]
    labels = np.array([names.index(x) for x in labels_by_name], dtype=int)
    frame = pd.DataFrame({"path": [str(root / p) for p in paths], "label": labels_by_name})
    started = datetime.now(timezone.utc).isoformat()
    probs: dict[str, np.ndarray] = {}
    for n in systems:
        out = np.asarray(predictors[n].run(frame), dtype=float)
        if out.shape != (len(paths), len(names)) or not np.isfinite(out).all():
            raise Refused(f"{n} returned probabilities of shape {out.shape}; expected {(len(paths), len(names))}")
        probs[n] = out
    for n in cal_names:
        probs[f"{n}_calibrated"] = calibrated(probs[n], calibration[n]["temperature"])
    thresholds = e6_thresholds(cfg, decision, root)
    exif = exif_counts([Path(p) for p in frame["path"]])
    order = [x for n in systems for x in (n, f"{n}_calibrated") if x in probs]
    cleaned_info = cleaned_rows(split, cfg, info, base, root) if clean else None
    basis = (f"cleaned rows: {cleaned_info['after']} of {cleaned_info['before']} field_test photos kept "
             f"({clean}); {cleaned_info['removed']} removed by the label cleaning" if clean else
             "lab_test photos, all of them: the label cleaning covers real photos only")
    class_rule = {"classes": len(names), "rows_before_class_rule": len(base), "rows_scored": len(positions),
                  "rows_dropped": len(base) - len(positions), "labels_outside_the_list": outside,
                  "rule": "rows whose true class is outside the class list are not scored; an older model's outputs "
                          "for classes outside the list are masked and the rest renormalized before scoring"}
    if outside:
        basis += (f"; {len(base) - len(positions)} further photos of {len(outside)} classes outside the "
                  f"{len(names)}-class list left out")
    primary_spec = {"system": "improved", "candidate": primary, "checkpoint": decision["chosen_model"],
                    "sha256": decision["chosen_model_sha256"], "why": "the model the final-model decision chose"}
    alt_specs = {n: {"system": n, "candidate": k, "checkpoint": specs[k]["checkpoint"],
                     "sha256": predictors[n].detail.get("sha256"),
                     "why": "another candidate, reported beside the primary on the same rows"}
                 for n, k in alternates.items()}
    config = {
        "class_names": names, "split": split, "frozen_config": CONFIG_REL,
        "frozen_config_sha256": text_sha256(root / CONFIG_REL), "decision": cfg["decision_file"],
        "chosen_model": decision["chosen_model"], "chosen_model_sha256": decision["chosen_model_sha256"],
        "primary": primary_spec, **alt_specs,
        **({"alternates_not_scored": not_scored} if not_scored else {}),
        "basis": basis, "class_rule": class_rule, "calibration": calibration, "calibration_binding": binding,
        "quantized_thresholds": thresholds, "exif": exif, "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
        "device": device_name(cfg), "started_at": started,
        "systems": {n: {"description": predictors[n].description, **predictors[n].detail} for n in systems},
        "abstention": ABSTENTION_RULE,
        "abstention_as_served_for": [n for n in cal_names if "healthy_threshold" in calibration[n]],
        "abstention_alone_for": [n for n in cal_names if "healthy_threshold" not in calibration[n]],
        **({"cleaned_rows": cleaned_info} if clean else {}),
        **({"override_note": override_note.strip()} if override_note.strip() else {}),
    }
    models = {n: predictors[n].description for n in systems}
    models["improved"] = f"primary ({primary}): {models['improved']}"
    for n, k in alternates.items():
        models[n] = f"alternate ({k}): {models[n]}"
    models.update({f"{n}_calibrated": f"{models[n]}, temperature {calibration[n]['temperature']}"
                   for n in cal_names})
    attempts = [{"system": n, "kind": "frozen, scored once", **predictors[n].detail} for n in systems]
    views = [(subset, np.ones(len(paths), dtype=bool), "")]
    if split == "lab_test":
        view = "+".join(x for x in ("shared_classes", class_part if outside else None) if x)
        shared = set(subset_positions(view, split, info, root))
        views.append((view, np.array([p in shared for p in positions]),
                      " Scored on the classes lab.csv and field.csv share, with all outputs kept, so a prediction "
                      "outside that list is wrong."))
    written = []
    for name, keep, extra in views:
        metrics, intervals, comparisons = score_block(
            {n: probs[n] for n in order}, labels, groups, names,
            {n: calibration[n]["abstain_threshold"] for n in cal_names}, keep,
            {n: calibration[n]["healthy_threshold"] for n in cal_names if "healthy_threshold" in calibration[n]})
        rows = {n: Predictions([p for p, k in zip(paths, keep, strict=True) if k], labels[keep], probs[n][keep])
                for n in order}
        notes = (f"Final test on {split}, scored once with configurations frozen before the run; every number is a "
                 f"held-out number on this basis: {basis}. The primary model ({primary}) is the one the final-model "
                 f"decision chose; the other candidates ({', '.join(alternates.values()) or 'none scored'}) are scored "
                 f"beside it on the same rows. Options per plant are in "
                 f"config.options_per_plant (plant-given top-3 is easy). {LABEL_SIGNATURE}{extra} "
                 f"{statements(split, thresholds, exif, root)}")
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
    if model.source_sha256 != decision["chosen_model_sha256"]:
        raise Refused(f"{spec['variant']} is stamped as exported from checkpoint {model.source_sha256 or '(none)'}, not the "
                      "chosen model; export it again with E6")

    def run(frame: pd.DataFrame) -> np.ndarray:
        out = []
        for p in frame["path"]:
            with Image.open(p) as img:
                out.append(model.predict(img))
        return np.stack(out)

    return Predictor(f"{spec['variant']} export of the chosen model", model.class_names, run,
                     {"kind": "onnx", "variant": spec["variant"], "sha256": info["sha256"],
                      "source_checkpoint_sha256": model.source_sha256})


def restricted(pred: Predictor, names: list[str]) -> Predictor:
    """An older model on the final class list: keep the outputs of the listed classes, drop the others and renormalize
    (the rule fixed in T4.9 for rescoring E4 and E12). A model that cannot output a listed class is refused."""
    if pred.class_names == names:
        return pred
    missing = [n for n in names if n not in pred.class_names]
    if missing:
        raise Refused(f"{pred.description} has no output for {missing[:3]}; its classes are not a superset of the list")
    cols = [pred.class_names.index(n) for n in names]
    width = len(pred.class_names)

    def run(frame: pd.DataFrame) -> np.ndarray:
        out = np.asarray(pred.run(frame), dtype=float)
        if out.ndim != 2 or out.shape[1] != width:
            raise Refused(f"{pred.description} returned {out.shape[-1]} outputs for {width} classes")
        kept = out[:, cols]
        return kept / np.maximum(kept.sum(axis=1, keepdims=True), FLOOR)

    return Predictor(f"{pred.description}, outputs masked to the {len(names)}-class list and renormalized", names, run,
                     {**pred.detail, "masked_from_classes": width})


def build_predictors(cfg: dict[str, Any], decision: dict[str, Any], root: Path = ROOT) -> dict[str, Predictor]:
    """The frozen systems from the config and the decision. `improved` is the model the decision chose and each other
    candidate whose saved file is on disk is an alternate; the chosen file must be the one the decision hashes. Every
    older model is restricted to the decision's class list."""
    root, size, device = Path(root), cfg["batch_size"], cfg["torch_device"]
    spec = cfg["systems"]
    specs = specs_of(cfg, decision)
    primary, others = candidate_roles(cfg, decision)
    names = decision["class_names"]
    chosen = root / decision["chosen_model"]
    if not chosen.is_file() or sha256_file(chosen) != decision["chosen_model_sha256"]:
        raise Refused(f"{decision['chosen_model']} differs from the decision's sha256 or is missing")
    old = names
    if cfg.get("old_classes_file"):
        old = json.loads((root / cfg["old_classes_file"]).read_text(encoding="utf-8"))["class_names"]
    out = {
        "baseline": restricted(_torch_predictor(root / spec["baseline"]["checkpoint"], "clean lab baseline", size, device), names),
        "improved": restricted(_torch_predictor(chosen, f"{primary} model chosen by the final-model decision", size, device), names),
        "xgboost": restricted(_xgboost_predictor(root / spec["xgboost"]["model"], spec["xgboost"]["features"], old), names),
        "dinov2": restricted(_torch_predictor(root / spec["dinov2"]["checkpoint"], "DINOv2 small", size, device), names),
        "quantized": _onnx_predictor(cfg, decision, root),
    }
    for i, key in enumerate(others):
        path = root / specs[key]["checkpoint"]
        if path.is_file():
            out[e9_roles.system_of(i)] = restricted(_torch_predictor(path, f"{key} model, another candidate", size, device), names)
    return out


METRIC_LINES = (("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("ece", "calibration error (ECE)"),
                ("top_3_accuracy", "top-3"), ("top_5_accuracy", "top-5"), ("plant_given_top_1", "plant-given top-1"),
                ("plant_given_top_3", "plant-given top-3"))


def image_report(rec: dict[str, Any]) -> str:
    """The figures of an image record in words. Every line that holds a number names the rows it is on."""
    c = rec["config"]
    tag = "cleaned rows" if "cleaned_rows" in c else f"{rec['split']}" + (f" {rec['subset']}" if rec["subset"] else "")
    role = {c["primary"]["system"]: f"primary ({c['primary']['candidate']})",
            **{k: f"alternate ({c[k]['candidate']})" for k in ("alternate", "alternate_2") if k in c}}
    lines = [f"[{tag}] basis: {c['basis']}; n = {rec['n']}"]
    if "cleaned_rows" in c:
        r = c["cleaned_rows"]
        lines.append(f"[{tag}] removed {r['removed']} of {r['before']}: by reason {r['by_reason']}, by source {r['by_source']}")
    lines.append(f"[{tag}] Options per plant: {c['options_per_plant']}; label review is signed per split in {SIGNOFF_REL}, not here.")
    for name, values in rec["metrics"].items():
        shown = ", ".join(f"{label} {values[key]:.3f}" for key, label in METRIC_LINES if values.get(key) is not None)
        base = name.removesuffix("_calibrated")
        kind = (f"{role.get(base, base)}" + (" with abstention" if name.endswith("_calibrated") else ""))
        served = {k: v for k, v in values.items() if "~" in k and v is not None}
        alone = {k: v for k, v in values.items() if k.startswith(("coverage@", "selective_accuracy@", "confident_error_rate@"))
                 and "~" not in k and v is not None}
        if alone:
            shown += "; abstention alone: " + ", ".join(f"{k} {v:.3f}" for k, v in alone.items())
        if served:
            shown += "; as served (healthy bar): " + ", ".join(f"{k} {v:.3f}" for k, v in served.items())
        elif name.endswith("_calibrated"):
            shown += "; no healthy threshold in its calibration, abstention alone"
        lines.append(f"[{tag}] {kind}: {shown}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Sonnet calls, two at a time

WORK_REL = Path("results/e9/work")
IN_FLIGHT = 2
CALL_TIMEOUT_S = 180.0
DEV_SPLIT = "rag_dev"
DEV_PREDATES_FALLBACK = (
    "The saved rag_dev answers this check was developed on (E7 run E7-20261104T040010301317Z) were written before "
    "T7.4 added the search with the diagnosed disease when the question finds nothing, so they do not reflect the "
    "retrieval served now; no saved set of served rag_dev answers with the fallback exists, and none was generated "
    "for this check. rag_test is answered with the fallback."
)
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


def run_two_at_a_time(todo: list[Any], one: Callable[[Any], dict[str, Any]], ck: Checkpoint,
                      workers: int = IN_FLIGHT, stop: Callable[[], bool] = lambda: False) -> None:
    """Run `one` over `todo` with at most `workers` calls at once (two unless said), saving each row as it arrives. An
    error stops new calls, keeps what arrived, and is raised."""
    failure: LLMError | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue, live = iter(todo), set()
        while True:
            while failure is None and not stop() and len(live) < workers:
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


def main_folder(root: Path) -> Path:
    """The folder that holds the repository's .git, which is `root` itself unless `root` is a git worktree."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def claude_counter(root: Path) -> Path:
    """The Claude call counter every run counts into: the main folder's, never a worktree's own copy."""
    return main_folder(root) / "results" / "claude_calls.json"


def index_path(cfg: dict[str, Any], root: Path) -> Path:
    """The served index: this tree's `rag.index_dir` when it holds one, else the main folder's (a worktree does not carry
    the index, which is built once and is not tracked)."""
    for base in (Path(root), main_folder(root)):
        found = base / cfg["rag"]["index_dir"]
        if (found / "chunks.jsonl").is_file():
            return found
    raise Refused(f"no index at {cfg['rag']['index_dir']} in this tree or the main folder")


def index_identity(index_dir: Path) -> dict[str, Any]:
    """What identifies the index that served rag_test: its build info and the sha256 of the chunk file and the vectors,
    to be set beside the build info E7 ran on."""
    info = json.loads((Path(index_dir) / "build-info.json").read_text(encoding="utf-8"))
    vectors = Path(index_dir) / info["vectors_file"]
    return {"built": info["built"], "chunks": info["chunks"],
            "embedding_model": info["embedding_model"], "chunks_sha256": sha256_file(Path(index_dir) / "chunks.jsonl"),
            "vectors_file": info["vectors_file"], "vectors_sha256": sha256_file(vectors)}


def index_crops(index_dir: Path) -> set[str]:
    return set(index_crops_of(Path(index_dir) / "chunks.jsonl").values()) - {GENERAL_CROP}


def rendered_sha256(root: Path) -> str:
    """Hash of the messages built for a fixed input, so a change to the wording or the layout shows up even when the
    prompt file is unchanged."""
    return hashlib.sha256(json.dumps(wd_messages("Q", "D", "A", root), sort_keys=True).encode()).hexdigest()


def wd_key(items: list[dict[str, Any]], model: str, root: Path) -> str:
    parts = [prompt_text(root), model, [[i["id"], i["question"], i["diagnosis"], i["answer"]] for i in items]]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:10]


def wd_grade(items: list[dict[str, Any]], llm: Any, tag: str, root: Path, limit: int | None = None,
             workers: int = IN_FLIGHT) -> Checkpoint:
    """One Sonnet call per item not already saved. With `limit`, refuse first if the calls would pass it or the cap."""
    ck = Checkpoint(Path(root) / WORK_REL / f"wd-{tag}-{wd_key(items, llm.model, root)}.jsonl")
    todo = [i for i in items if i["id"] not in ck.rows]
    if limit is not None:
        check_budget(llm, len(todo), limit, len(ck.rows))

    def one(item: dict[str, Any]) -> dict[str, Any]:
        try:
            reply = llm.complete(wd_messages(item["question"], item["diagnosis"], item["answer"], root),
                                 max_tokens=8, timeout_s=CALL_TIMEOUT_S).text.strip()
        except SafeguardRefusal as refusal:  # this check's rows only: kept, never retried, no grade
            return {"id": item["id"], "reply": None, "call": None, "refused_by_safeguard": True,
                    "request_id": refusal.request_id}
        return {"id": item["id"], "reply": reply, "call": parse_call(reply)}

    run_two_at_a_time(todo, one, ck, workers=workers, stop=lambda: refused_rows(ck) >= WD_STOP_AFTER_REFUSALS)
    return ck


WD_STOP_AFTER_REFUSALS = 3  # in total, rows recorded by hand included; the check sends one call at a time, so none is in flight at the third


def refused_rows(ck: Checkpoint) -> int:
    return sum(bool(r.get("refused_by_safeguard")) for r in list(ck.rows.values()))


def mark_refused(cfg: dict[str, Any], root: Path, request_id: str, *, override_note: str = "") -> str:
    """Record, with no model call, the one answer the safeguard refused before its row could be saved. The check runs the
    answers in a fixed order and an error stops new calls, so that answer is the first one without a saved row. Refuses if
    a refused row exists already, if every answer has a row, or if the id is not a request id. Returns the answer's id."""
    guard_rag_test(root, override_note)
    if not re.fullmatch(r"req_[A-Za-z0-9]+", request_id):
        raise Refused(f"{request_id!r} is not a request id")
    items = _test_items(cfg, root, claim_rows(cfg, root))
    ck = Checkpoint(Path(root) / WORK_REL / f"wd-test-{wd_key(items, cfg['rag']['grader'], root)}.jsonl")
    if refused_rows(ck):
        raise Refused("a refused row is already recorded; a refusal is marked once")
    todo = [i["id"] for i in items if i["id"] not in ck.rows]
    if not todo:
        raise Refused("every answer has a saved row, so there is no refused one to record")
    ck.add({"id": todo[0], "reply": None, "call": None, "refused_by_safeguard": True, "request_id": request_id,
            "recorded_without_call": True})
    return todo[0]


def dev_answers(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """The final E7 served answers on rag_dev (from the judgments file the config names, checked by sha256), each with
    its question and the diagnosis the writer was given."""
    from sprout.experiments.e14_judge_check import item_of, load_source

    wd = cfg["wrong_disease"]
    judged, _ = load_source({"source": wd["dev_source"]}, root)
    questions = {q.id: q for q in e7.dev_questions(root)}
    crops, classes = index_crops(index_path(cfg, root)), e7.load_classes(root)
    served = sorted((r for r in judged if r["system"] == wd["served_system"]), key=lambda r: r["id"])
    return [{"id": r["id"], "item": item_of(r["system"], r["id"]), "question": r["question"],
             "diagnosis": diagnosis_for(questions[r["id"]], crops, classes), "answer": r["answer"]} for r in served]


def prompt_fields(root: Path) -> dict[str, str]:
    return {"prompt_file": PROMPT_REL, "prompt_sha256": text_sha256(Path(root) / PROMPT_REL),
            "rendered_sha256": rendered_sha256(root)}


def stage_wrong_disease_dev(cfg: dict[str, Any], llm: Any, root: Path = ROOT, note: str = "") -> Path:
    """Run the check on the served rag_dev answers and write its record with the rate. Calls are saved as they arrive."""
    root = Path(root)
    require_clean(root)
    wd = cfg["wrong_disease"]
    items = dev_answers(cfg, root)
    started = datetime.now(timezone.utc).isoformat()
    ck = wd_grade(items, llm, "dev", root, limit=wd["dev_max_calls"])
    if refused_rows(ck) or any(i["id"] not in ck.rows for i in items):
        raise Refused("the dev check is incomplete (a safeguard refusal or unsent rows); it makes no record")
    rows = [{**i, "call": ck.rows[i["id"]]["call"], "reply": ck.rows[i["id"]]["reply"]} for i in items]
    info = split_info(DEV_SPLIT, root)
    config = {"stage": "wrong_disease_dev", **prompt_fields(root), "source": wd["dev_source"],
              "served_system": wd["served_system"], "diagnosis_source": DIAGNOSIS_RULE, "claude_calls": len(rows),
              "device": WD_DEVICE, "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
              "started_at": started, "source_predates_fallback": True, "source_note": DEV_PREDATES_FALLBACK,
              "grader_prompt": "question, diagnosis and answer; the reply is read strictly as FITS or WRONG"}
    attempts = [{"stage": "wrong_disease_dev", "answers": len(rows), "claude_calls": len(rows),
                 "replies_without_a_call": sum(r["call"] is None for r in rows)}]
    return write_run("E9", DEV_SPLIT, config, {"grader": llm.model}, {"served": dev_metrics(rows)},
                     {"served": dev_intervals(rows)}, attempts, {"served": rows}, info.hash, root=root,
                     notes=("Development check of the wrong-disease grader on the final served rag_dev answers. "
                            "The prompt is frozen only after this. " + NOT_VALIDATED + (" " + note if note else "")))


# ---------------------------------------------------------------- freeze

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


NOT_VALIDATED = ("The wrong-disease check was not validated: no one read a sample of its calls on rag_dev, so how often "
                 "it flags a good answer or passes a wrong one is not known. Its rate is a count of the grader's calls.")


def stage_freeze(cfg: dict[str, Any], root: Path = ROOT) -> Path:
    """Record the prompt's commit and sha256 in a config to be committed. Needs the dev run, made with this very prompt."""
    root = Path(root)
    if (root / FROZEN_REL).exists():
        raise Refused(f"{FROZEN_REL} already frozen; a frozen prompt is not changed")
    require_committed(root, [PROMPT_REL])
    require_clean(root)
    _, dev = latest_stage_record(root, "wrong_disease_dev")
    fields = prompt_fields(root)
    if any(fields[k] != dev["config"][k] for k in ("prompt_sha256", "rendered_sha256")):
        raise Refused("the prompt has changed since the dev run that measured it")
    commit = git(root, "log", "-1", "--format=%H", "--", PROMPT_REL).stdout.strip()
    frozen = {"experiment": "E9", **fields, "prompt_commit": commit, "grader_model": cfg["rag"]["grader"],
              "dev_record": dev["run_id"], "validated": False, "validation": NOT_VALIDATED,
              "dev_source_note": DEV_PREDATES_FALLBACK,
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


# ---------------------------------------------------------------- the judges

HAIKU_DECISION_REL = "results/e14/haiku_decision.json"


def first_judge(root: Path = ROOT) -> dict[str, Any]:
    """The first judge, read from the committed decision file of the Haiku check (E14H) and never typed into a config.
    E9 has no local judge: a decision that names anything but Haiku stops the run."""
    from sprout.experiments.e14h_haiku_check import rubric_hash

    root = Path(root)
    require_committed(root, [HAIKU_DECISION_REL])
    decision = json.loads((root / HAIKU_DECISION_REL).read_text(encoding="utf-8"))
    if decision.get("first_judge") != "haiku":
        raise Refused(f"{HAIKU_DECISION_REL} says the first judge is {decision.get('first_judge')!r}; E9 has no local "
                      "judge, so it cannot run on that decision")
    if not decision.get("model") or decision.get("rubric_sha256") != rubric_hash():
        raise Refused(f"{HAIKU_DECISION_REL} names no model, or the rubric changed since the Haiku check graded with it")
    return decision


# ---------------------------------------------------------------- rag_test

RAG_TEST = "rag_test"
SERVED = "served"
EMBEDDER, RERANKER = "BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L6-v2"
FINAL_CAVEAT = (
    "Held-out numbers: the questions were answered and judged once, with everything frozen before; if the stages were "
    "interrupted, config.run_account says how. The Haiku judge (the first judge, chosen by the "
    "committed Haiku check) and the Sonnet judge (the second figure) read the same claims with the same rubric. Their "
    "figures differ, Sonnet is a reference and not truth, and neither is the share of correct claims. The wrong-disease "
    "check is a Sonnet grader that was run on rag_dev, frozen, and run once here; it was not validated by a blind read."
)


def guard_rag_test(root: Path, override_note: str) -> dict[str, Any]:
    """Every rag_test stage starts here: configs, prompt and the Haiku decision committed, tree clean, prompt equal to
    the frozen one, split unused."""
    root = Path(root)
    require_committed(root, [CONFIG_REL, FROZEN_REL, PROMPT_REL, HAIKU_DECISION_REL])
    require_clean(root)
    frozen = require_frozen(root)
    first_judge(root)
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
    index_dir = index_path(cfg, root)
    key = e7.checkpoint_key(e7.ANSWER_VERSION, asdict(e7.replace_sources(serve_config(None))), serving.FLOORS, e7.index_stamp(index_dir),
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
    single = e7.single_page_sources(index_path(cfg, root))
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


def claim_rows(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """The answers that have claims to grade, in question order, each with the passages it was written from and its
    claims (the answer's factual sentences, split by the code E7 used). Both judges are shown exactly this."""
    out = []
    for g in _generated(cfg, root):
        claims = split_claims(g["text"]) if g["state"] in ("answered", "ungrounded") else []
        if claims:
            out.append({"item": e14.item_of(SERVED, g["id"]), "id": g["id"], "answer": g["text"],
                        "passages": [{"n": i, **p} for i, p in enumerate(g["passages"], 1)],
                        "claims": [{"claim": c.text, "cited": c.cited} for c in claims]})
    return out


def _claims_key(model: str, rows: list[dict[str, Any]]) -> str:
    parts = [model, rubric(), [[r["item"], [c["claim"] for c in r["claims"]]] for r in rows]]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:10]


def _claims_checkpoint(model: str, rows: list[dict[str, Any]], root: Path) -> Checkpoint:
    return Checkpoint(Path(root) / WORK_REL / "claims" / f"claims-{model}-{_claims_key(model, rows)}.jsonl")


def _pending(rows: list[dict[str, Any]], done: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    return [(r, j) for r in rows for j in range(len(r["claims"])) if f"{r['item']}:{j}" not in done]


def grade_claims(rows: list[dict[str, Any]], llm: Any, ck: Checkpoint, workers: int) -> None:
    """One call per claim not already saved, `workers` at a time, each reply saved as it arrives (E14's code and rubric)."""
    def one(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
        got = e14.grade_claim(llm, task[0], task[1])
        return {"id": got["id"], "reply": got["reply"], "verdict": got["sonnet"]}

    run_two_at_a_time(_pending(rows, ck.rows), one, ck, workers)


def stage_rag_haiku(cfg: dict[str, Any], llm: Any, root: Path = ROOT, *, override_note: str = "") -> None:
    """The Haiku judge grades every claim. Its model is the one the committed Haiku check names."""
    guard_rag_test(root, override_note)
    _require_model(llm, first_judge(root)["model"], "Haiku judge")
    rows = claim_rows(cfg, root)
    ck = _claims_checkpoint(llm.model, rows, root)
    check_budget(llm, len(_pending(rows, ck.rows)), cfg["rag"]["max_haiku_calls"], len(ck.rows))
    grade_claims(rows, llm, ck, cfg["rag"]["judge_in_flight"])


def _test_items(cfg: dict[str, Any], root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qmap = {q.id: q for q in rag_questions(root)}
    crops, classes = index_crops(index_path(cfg, root)), e7.load_classes(root)
    return [{"id": r["id"], "question": qmap[r["id"]].question, "diagnosis": diagnosis_for(qmap[r["id"]], crops, classes),
             "answer": r["answer"]} for r in rows]


def stage_rag_sonnet(cfg: dict[str, Any], llm: Any, root: Path = ROOT, *, override_note: str = "") -> None:
    """The Sonnet judge grades the same claims with the same rubric, then runs the frozen wrong-disease check once per
    answer. The cap and the stage limit are checked for both together before any call."""
    guard_rag_test(root, override_note)
    _require_model(llm, cfg["rag"]["grader"], "Sonnet judge")
    rows = claim_rows(cfg, root)
    ck = _claims_checkpoint(llm.model, rows, root)
    items = _test_items(cfg, root, rows)
    wd_done = Checkpoint(Path(root) / WORK_REL / f"wd-test-{wd_key(items, llm.model, root)}.jsonl").rows
    need = len(_pending(rows, ck.rows)) + sum(i["id"] not in wd_done for i in items)
    check_budget(llm, need, cfg["rag"]["max_sonnet_calls"], len(ck.rows) + len(wd_done))
    grade_claims(rows, llm, ck, IN_FLIGHT)
    wd_grade(items, llm, "test", root, workers=1)  # one call at a time: nothing is in flight at the third refusal


WD_RATE_RULE = ("the rate's denominator is the graded answers with a usable call; answers the safeguard refused and "
                "answers never sent are counted as neither result")
WD_PARTIAL = ("The wrong-disease check is partial: some answers were refused by the claude command's safeguard or never "
              "sent, so its rate covers the graded answers only and is not a result for all answers with claims.")

GENERIC_FEW = 10  # wording only, chosen before rag_test was run: below it the report says "few"; no number depends on it


def is_generic_followup(q: Question, crops: set[str], classes: list[str]) -> bool:
    """A follow-up that names no plant and no disease: no plant given for the question, no plant word in it or in the
    user's earlier turns (E7's detect_crop over the index's plants), and no disease name or alias of any class in that
    text (E8's names_of). Fixed here, before rag_test is read."""
    from sprout.experiments.e8_fallback import names_of
    from sprout.rag.query import detect_crop

    text = " ".join([*(h["content"] for h in q.history if h.get("role") == "user"), q.question])
    flat = " ".join(text.lower().replace("-", " ").split())
    return (not q.crop and detect_crop(text, crops) is None
            and not any(n in flat for c in classes for n in names_of(c)))


def final_rows(cfg: dict[str, Any], haiku_model: str, sonnet_model: str, root: Path) -> list[dict[str, Any]]:
    """Rows in question order, with both judges' calls on every claim and the wrong-disease call."""
    gen, claims = _generated(cfg, root), claim_rows(cfg, root)
    by_id = {r["id"]: r for r in claims}
    haiku, sonnet = (_claims_checkpoint(m, claims, root).rows for m in (haiku_model, sonnet_model))
    items = _test_items(cfg, root, claims)
    wd = Checkpoint(Path(root) / WORK_REL / f"wd-test-{wd_key(items, sonnet_model, root)}.jsonl").rows
    qmap = {q.id: q for q in rag_questions(root)}
    crops, classes = index_crops(index_path(cfg, root)), e7.load_classes(root)
    diagnosis = {i["id"]: i["diagnosis"] for i in items}
    searches = _retrieval_checkpoint(cfg, root).rows
    rows = []
    for g in gen:
        have = by_id.get(g["id"])
        got = []
        for j, c in enumerate(have["claims"] if have else []):
            key = f"{have['item']}:{j}"
            if key not in haiku or key not in sonnet:
                raise ValueError("some claims are not graded by both judges yet; run the rag-haiku and rag-sonnet stages")
            got.append({"claim": c["claim"], "cited": c["cited"], "haiku": haiku[key]["verdict"],
                        "haiku_reply": haiku[key]["reply"], "sonnet": sonnet[key]["verdict"],
                        "sonnet_reply": sonnet[key]["reply"]})
        check = wd.get(g["id"]) if got else None
        refused = bool(check and check.get("refused_by_safeguard"))
        status = ("no_claims" if not got else "not_sent" if check is None else
                  "refused_by_safeguard" if refused else "graded")
        rows.append({"id": g["id"], "answered": bool(g["answered"]), "state": g["state"],
                     "diagnosis": diagnosis.get(g["id"]) or diagnosis_for(qmap[g["id"]], crops, classes), "text": g["text"],
                     "citations": g["citations"], "query": g["query"],
                     "fallback_fired": len(searches[g["id"]]["searches"]) > 1,
                     "generic_followup": is_generic_followup(qmap[g["id"]], crops, classes),
                     "claims": got, "wrong_disease": check["call"] if status == "graded" else None,
                     "wrong_disease_reply": check["reply"] if status == "graded" else None,
                     "wrong_disease_status": status,
                     "wrong_disease_request_id": check.get("request_id") if refused else None,
                     "haiku_calls": len(got), "sonnet_calls": len(got) + (1 if check else 0)})
    return rows


def _file_times(path: Path) -> dict[str, str | None]:
    """When a checkpoint file was created (where the system keeps that) and last written, in UTC."""
    st = Path(path).stat()
    stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()  # noqa: E731
    born = getattr(st, "st_birthtime", None)
    return {"created": stamp(born) if born else None, "last_write": stamp(st.st_mtime)}


def _commit_at(root: Path, when: str | None) -> str | None:
    """The newest first-parent commit made at or before `when`: what the tree was at, as far as file times say."""
    if when is None:
        return None
    out = git(root, "rev-list", "-1", "--first-parent", f"--before={when}", "HEAD").stdout.strip()
    return out or None


def run_account(cfg: dict[str, Any], root: Path, rows: list[dict[str, Any]], frozen: dict[str, Any],
                interruption_note: str, started_at: str) -> dict[str, Any]:
    """What the record can say about how its stages ran, from the saved checkpoints and the tree, plus the person's
    interruption note stored verbatim for what cannot be derived. Refuses when a safeguard refusal exists and there is
    no note: the commits the first part ran at and the stop itself are not in any file."""
    counts = wd_counts(rows)
    claims = claim_rows(cfg, root)
    items = _test_items(cfg, root, claims)
    haiku_model, sonnet_model = first_judge(root)["model"], cfg["rag"]["grader"]
    files = {"retrieve": _retrieval_checkpoint(cfg, root).path,
             "generate": Path(root) / WORK_REL / "generate" / f"generate-{SERVED}-{_generation_key(cfg, _retrieval_checkpoint(cfg, root))}.jsonl",
             "claims_haiku": _claims_checkpoint(haiku_model, claims, root).path,
             "claims_sonnet": _claims_checkpoint(sonnet_model, claims, root).path,
             "wrong_disease": Path(root) / WORK_REL / f"wd-test-{wd_key(items, sonnet_model, root)}.jsonl"}
    stages = {}
    for name, path in files.items():
        text = Path(path).read_bytes()
        stages[name] = {"file": Path(path).relative_to(root).as_posix(), "rows": len(text.splitlines()),
                        "sha256": hashlib.sha256(text).hexdigest(), **_file_times(path)}
    created = sorted(s["created"] for s in stages.values() if s["created"])
    written = sorted(s["last_write"] for s in stages.values())
    saved = Checkpoint(files["wrong_disease"]).rows
    interrupted = counts["refused_by_safeguard"] > 0
    if interrupted and not interruption_note.strip():
        raise Refused("a safeguard refusal stopped the wrong-disease check, so the record needs an interruption: give "
                      "--interruption-note with the commits the first part ran at and resumed at, in your words")
    return {
        "questions_answered_and_judged": {"questions": len(rows), "answers_saved": stages["generate"]["rows"],
                                          "claims_judged_by_haiku": stages["claims_haiku"]["rows"],
                                          "claims_judged_by_sonnet": stages["claims_sonnet"]["rows"],
                                          "wrong_disease_rows": stages["wrong_disease"]["rows"]},
        "stages": stages,
        "window": {"first_file_created": created[0] if created else None, "last_file_write": written[-1],
                   "record_written": started_at,
                   "note": "from checkpoint files; a stage's own start is not stored, so the window is the files' "
                           "creation and last write, and the record's own started_at is only when the record was written"},
        "interruption": {
            "occurred": interrupted, "stop_after_refusals": WD_STOP_AFTER_REFUSALS,
            "refused": [{**r, "recorded_without_call": bool(saved.get(r["id"], {}).get("recorded_without_call"))}
                        for r in counts["refused"]],
            "code_commits": {"when_wrong_disease_file_was_created": _commit_at(root, stages["wrong_disease"]["created"]),
                             "when_it_was_last_written": _commit_at(root, stages["wrong_disease"]["last_write"]),
                             "record_written_at": git(root, "rev-parse", "HEAD").stdout.strip(),
                             "basis": "newest first-parent commit made at or before the file time; approximate"},
            "note": interruption_note.strip() or None,
            "note_is": "the person's words, stored verbatim; it states what no file shows (the commits of the first part "
                       "and of the resume, and that the earlier saved rows were not changed)"},
        "frozen": {"prompt_sha256": frozen["prompt_sha256"], "rendered_sha256": frozen["rendered_sha256"],
                   "grader_model": frozen["grader_model"], "equal_to_frozen_config": prompt_fields(root)["prompt_sha256"]
                   == frozen["prompt_sha256"], "wd_key_model": sonnet_model},
        "model_calls_by_rag_record": 0,
    }


def interruption_text(account: dict[str, Any]) -> str:
    i, w = account["interruption"], account["window"]
    if not i["occurred"]:
        return ""
    ids = ", ".join(f"{r['id']} (request {r['request_id']}{', marked by a command with no resend' if r['recorded_without_call'] else ''})"
                    for r in i["refused"])
    return (f"Interruption: rag-sonnet stopped at a safeguard refusal of the wrong-disease check and was resumed later, "
            f"with the code changed between the stop and the resume. Refused rows: {ids}. The questions were answered "
            f"and judged once, with one stop; no answer or claim verdict was produced twice. The checkpoint files span "
            f"{w['first_file_created']} to {w['last_file_write']} (the record itself was written at {w['record_written']}). "
            f"Commits current at the wrong-disease file's creation and last write: "
            f"{i['code_commits']['when_wrong_disease_file_was_created']}, {i['code_commits']['when_it_was_last_written']} "
            f"(derived from file times). The frozen prompt hash, model and the saved rows' hashes are in config.run_account. "
            f"The person's note: {i['note']}")


def superseding(root: Path, old_id: str, metrics: dict[str, Any], judged: dict[str, Any]) -> dict[str, Any]:
    """config.supersedes: the committed E9 rag_test record this one replaces, and whether it carries the same numbers."""
    path = Path(root) / "results" / "runs" / f"{old_id}.json"
    if not path.is_file():
        raise Refused(f"{old_id} is not a record in results/runs")
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("experiment") != "E9" or old.get("split") != RAG_TEST:
        raise Refused(f"{old_id} is not an E9 rag_test record")
    if git(root, "cat-file", "-e", f"HEAD:results/runs/{old_id}.json").returncode != 0:
        raise Refused(f"{old_id} is not committed; the record it supersedes stays as committed history")
    old_judged = json.loads((Path(root) / old["judged"]["file"]).read_text(encoding="utf-8"))
    return {"record": old_id, "why": "adds config.run_account; made from the saved rows with no model call",
            "same_numbers": old["metrics"] == metrics and old_judged == judged}


def stage_rag_record(cfg: dict[str, Any], root: Path = ROOT, *, device: str, note: str = "",
                     override_note: str = "", interruption_note: str = "", supersedes: str = "") -> Path:
    """Write the rag_test record from the saved stages, with no model call. Refuses unless every answer, claim and check
    is saved. A second record of the split needs `override_note`; `supersedes` names the record it replaces."""
    frozen = guard_rag_test(root, override_note)
    root = Path(root)
    haiku = first_judge(root)
    rows = final_rows(cfg, haiku["model"], cfg["rag"]["grader"], root)
    gen = _generated(cfg, root)
    info = split_info(RAG_TEST, root)
    if [r["id"] for r in rows] != info.ids:
        raise Refused("the saved answers do not cover rag_test in order")
    answered = np.array([r["answered"] for r in rows])
    generic = [r for r in rows if r["generic_followup"]]
    coverage = {"questions": len(rows), "generic_followups": len(generic), "rule": GENERIC_RULE,
                "generic_answered": sum(r["answered"] for r in generic),
                "generic_with_claims": sum(bool(r["claims"]) for r in generic),
                "fallback_fired": sum(r["fallback_fired"] for r in rows),
                "fallback_fired_with_claims": sum(r["fallback_fired"] and bool(r["claims"]) for r in rows),
                "answers_with_claims": sum(bool(r["claims"]) for r in rows),
                "claims_on_generic_followups": sum(len(r["claims"]) for r in generic),
                "claims_in_all": sum(len(r["claims"]) for r in rows),
                "few_cutoff": GENERIC_FEW, "few_cutoff_note": "wording only, chosen before the run",
                "few": len(generic) < GENERIC_FEW}
    scored = Scored(groups=info.ids, answered=answered, expected=info.expected)
    metrics = {SERVED: {m: recompute_metric(m, scored) for m in e7.BEHAVIOUR_METRICS}}
    calls = {"haiku": sum(r["haiku_calls"] for r in rows), "sonnet": sum(r["sonnet_calls"] for r in rows)}
    ac = serve_config(None)
    usage = {"rows": len(gen), "model_calls": sum(g.get("llm_calls", 0) for g in gen),
             "rows_without_model_call": sum(g.get("llm_calls", 0) == 0 for g in gen),
             "model_seconds": round(sum(g.get("llm_seconds", 0.0) for g in gen), 1)}
    started_at = datetime.now(timezone.utc).isoformat()
    account = run_account(cfg, root, rows, frozen, interruption_note, started_at)
    judged = final_lists(rows)
    config = {
        "stage": "final", "served": {"prompt_version": ac.prompt_version, "mode": ac.mode, "rerank": ac.rerank, "k": ac.k,
                                     "query_mode": ac.query_mode, "floors": serving.FLOORS},
        "judge": {"model": haiku["model"]},
        "judges": {"haiku": {"model": haiku["model"], "name": "Haiku judge", "role": "first judge",
                             "chosen_by": {"file": HAIKU_DECISION_REL, "record": haiku["record_id"]}},
                   "sonnet": {"model": cfg["rag"]["grader"], "name": "Sonnet judge", "role": "second judge"},
                   "same_claims_and_rubric": True},
        "rubric": rubric(), **prompt_fields(root), "frozen_wrong_disease": {"file": FROZEN_REL, **frozen},
        "wrong_disease_validated": False, "wrong_disease_validation": NOT_VALIDATED,
        "diagnosis_source": DIAGNOSIS_RULE, "calls": calls, "claude_calls": calls["haiku"] + calls["sonnet"],
        "wrong_disease_check": {**wd_counts(rows), "stop_after_refusals": WD_STOP_AFTER_REFUSALS,
                                "rule": WD_RATE_RULE},
        "device": device, "writer_usage": usage, "index": index_identity(index_path(cfg, root)),
        "generic_followup_coverage": coverage, "dev_source_note": DEV_PREDATES_FALLBACK,
        "served_with_fallback": True,
        "labels_reviewed": LABELS_REVIEWED, "label_signature": LABEL_SIGNATURE,
        "basis": "rag_test questions as asked; the photo label cleaning does not apply to them",
        "started_at": started_at, "caveat": FINAL_CAVEAT, "run_account": account,
        **({"supersedes": superseding(root, supersedes, metrics, judged)} if supersedes else {}),
        **({"override_note": override_note.strip()} if override_note.strip() else {}),
    }
    models = {"writer": cfg["rag"]["writer"], "judge_first": haiku["model"], "judge_second": cfg["rag"]["grader"],
              "wrong_disease_grader": cfg["rag"]["grader"], "embedder": EMBEDDER, "reranker": RERANKER}
    attempts = [{"stage": "rag_test", "answers": len(rows), "claims": sum(len(r["claims"]) for r in rows),
                 "claude_calls": config["claude_calls"]}]
    return write_run("E9", RAG_TEST, config, models, metrics, {SERVED: {}}, attempts, {SERVED: rows}, info.hash,
                     root=root, judged=judged, notes=FINAL_CAVEAT + ("" if wd_counts(rows)["complete"] else " " + WD_PARTIAL)
                     + (" " + interruption_text(account) if account["interruption"]["occurred"] else "")
                     + (" " + note if note else ""))


GENERIC_RULE = ("no plant given for the question, no plant of the index and no disease name or alias of the classes "
                "in the question or the user's earlier turns")


def coverage_line(c: dict[str, Any]) -> str:
    text = (f"Generic follow-ups (no plant or disease word; rule fixed in code before the run): {c['generic_followups']} "
            f"of {c['questions']} rag_test questions, {c['generic_answered']} answered, {c['generic_with_claims']} with "
            f"claims graded. Coverage: the faithfulness figures rest on {c['claims_in_all']} claims in "
            f"{c['answers_with_claims']} answers, of which {c['claims_on_generic_followups']} claims come from generic "
            f"follow-ups. The fallback fired on {c['fallback_fired']} answers ({c['fallback_fired_with_claims']} with "
            "claims graded).")
    if c["few"]:
        text += (f" That is few (the word 'few' means fewer than {c['few_cutoff']}, a wording cutoff chosen before the "
                 "run): the faithfulness figure says little about generic follow-ups, the case the fallback exists for. "
                 "E8's dev figure for them (70 of 95 passages naming the right disease) is a floor, because the text "
                 "rule misses some aliases, and is a retrieval measure, not faithfulness.")
    return text


def report(rec: dict[str, Any], root: Path = ROOT) -> str:
    """The figures of a rag_test record in words. The two judges are named, with the model of each; neither figure is
    called accuracy, and the report says which judge's share is the higher."""
    sidecar = json.loads((Path(root) / rec["judged"]["file"]).read_text(encoding="utf-8"))
    judges = rec["config"]["judges"]

    def share(name: str, label: str) -> str:
        v = sidecar[name]
        return f"{label}: {sum(v)} of {len(v)} claims supported = {sum(v) / len(v):.3f}"

    h, s = (sum(sidecar[k]) / len(sidecar[k]) for k in ("claims_supported_by_haiku", "claims_supported_by_sonnet"))
    order = "below" if h < s else "above" if h > s else "equal to"
    wd = rec["config"]["wrong_disease_check"]
    tag = "[rag_test questions, no photo cleaning applies]"
    lines = [f"{tag} {rec['n']} questions, one run, configurations frozen before it. Label review is signed in "
             f"{SIGNOFF_REL}, not in this record.",
             share("claims_supported_by_haiku", f"{judges['haiku']['name']} {judges['haiku']['model']} ({judges['haiku']['role']})"),
             share("claims_supported_by_sonnet", f"{judges['sonnet']['name']} {judges['sonnet']['model']} ({judges['sonnet']['role']})"),
             f"The two read the same claims with the same rubric. The Haiku judge's share is {order} the Sonnet judge's; "
             "neither is the share of correct claims.",
             coverage_line(rec["config"]["generic_followup_coverage"]),
             "The saved rag_dev answers the wrong-disease prompt was developed on predate the fallback; rag_test is "
             "answered with it.",
             f"wrong-disease check (Sonnet, prompt frozen before this run): {wd['flagged_wrong_disease']} of "
             f"{wd['usable_calls']} graded answers flagged as advice for the wrong disease or plant. "
             f"Of {wd['answers_to_check']} answers with claims: {wd['graded']} graded, {wd['refused_by_safeguard']} "
             f"refused by the safeguard (request ids {[r['request_id'] for r in wd['refused']]}), {wd['not_sent']} not "
             "sent; refused and unsent answers are counted as neither result. "
             + ("The check is complete. " if wd["complete"] else WD_PARTIAL + " ") + NOT_VALIDATED]
    lines += [f"{k}: {v:.3f}" if v is not None else f"{k}: undefined" for k, v in rec["metrics"][SERVED].items()]
    return "\n".join(x if x.startswith(tag) else f"{tag} {x}" for x in lines)


# ---------------------------------------------------------------- command line (wires in the real models)

STAGES = ("lab-test", "field-test", "wrong-disease-dev", "freeze", "rag-retrieve", "rag-generate", "rag-haiku",
          "rag-sonnet", "rag-record")
EXIT_INCOMPLETE = 2


def _device_file(root: Path) -> Path:
    return Path(root) / WORK_REL / "device.json"


def _remember_device(root: Path, role: str, info: dict[str, Any]) -> None:
    path = _device_file(root)
    known = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({**known, role: info}, indent=1) + "\n")


def device_text(root: Path = ROOT) -> str:
    """Where retrieval and the writer ran, as recorded by their stages; the judges run through the claude command and use
    no local device. Refuses if a stage left none."""
    path = _device_file(root)
    known = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    missing = [r for r in ("retrieval", "writer") if r not in known]
    if missing:
        raise Refused(f"no device was recorded for {missing}; run those stages first")
    r, w = known["retrieval"], known["writer"]
    return (f"retrieval on {r['device']}; writer {w['model']} with {w['size_vram']} of {w['size']} bytes in GPU memory; "
            "both judges and the wrong-disease grader run through the claude command, with no local device")


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
    return Retriever(index_path(cfg, root), config=config, embedder=embedder, reranker=reranker)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from sprout.llm.client import ClaudeCodeClient, OllamaClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--override-note", default="", help="why a test split is run again (a later record needs one)")
    ap.add_argument("--note", default="")
    ap.add_argument("--interruption-note", default="", help="rag-record: in your words, the commits the first part ran "
                    "at and resumed at and that earlier saved rows were not changed; stored verbatim")
    ap.add_argument("--supersedes", default="", help="rag-record: the committed rag_test record this one replaces "
                    "(needs --override-note; it makes no model call)")
    ap.add_argument("--refused-request-id", default="", help="rag-sonnet: record, with no model call, the one answer the "
                    "safeguard refused before its row was saved (the first answer without a saved row)")
    a = ap.parse_args(argv)
    try:
        cfg, root, stage = load_config(ROOT), ROOT, a.stage
        counter = claude_counter(root)
        grader = lambda: ClaudeCodeClient(model=cfg["rag"]["grader"], counter_path=counter)  # noqa: E731
        if stage in ("lab-test", "field-test"):
            split = stage.replace("-", "_")
            paths = run_images(split, cfg, lambda: build_predictors(cfg, load_decision(cfg, root), root), root,
                               override_note=a.override_note)
            print("\n".join(str(p) for p in paths))
            print(image_report(json.loads(paths[0].read_text(encoding="utf-8"))))
        elif stage == "wrong-disease-dev":
            print(stage_wrong_disease_dev(cfg, grader(), root, a.note))
        elif stage == "freeze":
            print(stage_freeze(cfg, root))
        elif stage == "rag-retrieve":
            guard_rag_test(root, a.override_note)
            print(f"{len(stage_rag_retrieve(cfg, _real_retriever(cfg, root), root, override_note=a.override_note).rows)} questions")
        elif stage == "rag-generate":
            guard_rag_test(root, a.override_note)
            client = OllamaClient(cfg["rag"]["writer"], keep_alive="10m")
            with _on_gpu(client, "writer", root):
                from sprout.rag.answer import load_sources

                missing = stage_rag_generate(cfg, client, root, load_sources(root / "knowledge" / "manifest.csv"),
                                             override_note=a.override_note)
                print(f"{len(rag_questions(root)) - len(missing)} answers, {len(missing)} missing")
                if missing:
                    print("Some questions got no model answer (Ollama down or timing out). Run it again to resume.")
                    return EXIT_INCOMPLETE
        elif stage == "rag-haiku":
            guard_rag_test(root, a.override_note)
            stage_rag_haiku(cfg, ClaudeCodeClient(model=first_judge(root)["model"], counter_path=counter), root, override_note=a.override_note)
            print("the Haiku judge graded the claims")
        elif stage == "rag-sonnet":
            if a.refused_request_id:
                print(f"recorded {mark_refused(cfg, root, a.refused_request_id, override_note=a.override_note)} "
                      "as refused_by_safeguard; no call was made")
                return 0
            stage_rag_sonnet(cfg, grader(), root, override_note=a.override_note)
            print("the Sonnet judge graded the claims and ran the wrong-disease check (see rag-record for whether it is partial)")
        else:
            path = stage_rag_record(cfg, root, device=device_text(root), note=a.note, override_note=a.override_note,
                                    interruption_note=a.interruption_note, supersedes=a.supersedes)
            print(path)
            print(report(json.loads(path.read_text(encoding="utf-8")), root))
    except (Refused, LLMError, RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
