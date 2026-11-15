"""E18's field_test stage: the frozen fit applied once to the final model's saved field_test probabilities.

Also holds the old E11 question flows (the 52 saved questions, with both the original rule and the live demo's old
rule), scored on the same simulated grower so the comparison is on the same footing. See e18_questions for the rest.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval
from sprout.eval.records import split_info, standard_image_metrics, write_run
from sprout.experiments import e18_questions as e18
from sprout.serve.question_engine import (
    Bank,
    Params,
    expected_calibration_error,
    load_questions,
)

THR = 0.929437  # the photo-only bar of results/e15/refit/chosen.json, which the old flows used


# ---- the old flows ----

def old_bank(root: Path, classes: list[str]):
    rows = [q for q in e18.read_jsonl(root / e18.OLD_QUESTIONS) if any(c in classes for c in q["classes"])]
    idx = {c: i for i, c in enumerate(classes)}
    sup = [np.array([idx[c] for c in q["classes"] if c in idx]) for q in rows]
    return rows, sup


def _ent(p: np.ndarray) -> float:
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def old_live(bank_rows, sup, ans, data, i: int, k_yes=3.0, k_no=0.35, max_q=5, floor=0.05, plant_down=0.02):
    """The live demo's former rule: plant question first, then the question with the lowest expected entropy under
    P(yes)=0.8/0.15, a yes multiplies the supported classes by 3 and a no by 0.35, up to 5 questions in all."""
    p = data.probs[i].copy()
    y = int(data.labels[i])
    asked, plant_known, used = [], False, 0
    crops = np.array(data.crops)
    while float(p.max()) < THR and used < max_q:
        order = np.argsort(-p)
        cands = list(order[:3]) + [c for c in order[3:5] if p[c] >= floor]
        offered = []
        for c in cands:
            if p[c] >= floor and data.crops[c] not in offered:
                offered.append(data.crops[c])
        if not plant_known and len(offered) >= 2:
            plant_known = True
            used += 1
            if data.crops[y] in offered:
                w = np.where(crops == data.crops[y], 1.0, plant_down)
                p = p * w / (p * w).sum()
            continue
        best = None
        for qi, s in enumerate(sup):
            sc = [c for c in s if c in cands]
            if qi in asked or not sc or len(sc) == len(cands):
                continue
            share = p[sc].sum()
            p_yes = share * 0.8 + (1 - share) * 0.15
            wy, wn = np.ones_like(p), np.ones_like(p)
            wy[sc], wn[sc] = k_yes, k_no
            e = p_yes * _ent(p * wy / (p * wy).sum()) + (1 - p_yes) * _ent(p * wn / (p * wn).sum())
            if best is None or e < best[0]:
                best = (e, qi, s)
        if best is None:
            break
        _, qi, s = best
        asked.append(qi)
        used += 1
        a = ans(i, qi)
        w = np.ones_like(p)
        if a == e18.YES:
            w[s] = k_yes
        elif a == e18.NO:
            w[s] = k_no
        p = p * w / (p * w).sum()
    return p, used


def old_e11(bank_rows, sup, ans, data, i: int, k_yes=3.0, k_no=0.5, max_q=2):
    """E11's rule: only when two of the top three have 0.10 or more; the top three is rescaled by 3 on yes and 0.5 on no
    for its classes tied to the question and renormalised to the mass it had; the question that splits the top three
    closest to half; at most 2 questions; no plant question."""
    p = data.probs[i].copy()
    top = np.argsort(-p)[:3]
    if (p[top] >= 0.10).sum() < 2:
        return p, 0
    asked = []
    for _ in range(max_q):
        top = np.argsort(-p)[:3]
        mass = p[top].sum()
        best = None
        for qi, s in enumerate(sup):
            tied = [c for c in top if c in s]
            if qi in asked or not tied or len(tied) == 3:
                continue
            gap = abs(p[tied].sum() / mass - 0.5)
            if best is None or gap < best[0]:
                best = (gap, qi, tied)
        if best is None:
            break
        _, qi, tied = best
        asked.append(qi)
        a = ans(i, qi)
        if a == e18.UNSURE:
            continue
        q = p.copy()
        q[tied] *= k_yes if a == e18.YES else k_no
        q[top] *= mass / q[top].sum()
        p = q
    return p, len(asked)


def run_old(rule, bank_rows, sup, table, data, flip: float = 0.0, unsure: float = 0.0, seed: int = 0, **kw) -> dict[str, Any]:
    final = data.probs.copy()
    nq = np.zeros(len(data), dtype=int)
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random((len(data), len(bank_rows))), rng.random((len(data), len(bank_rows)))

    def ans(i: int, qi: int) -> int:
        return e18.noisy(int(table[int(data.labels[i]), qi]), u1[i, qi], u2[i, qi], flip, unsure)

    for i in range(len(data)):
        final[i], nq[i] = rule(bank_rows, sup, ans, data, i, **kw)
    return {"final": final, "symptom_questions": nq, "stops": [""] * len(data)}


# ---- scoring ----

def compare(base: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    a, b = base.argmax(1) == labels, probs.argmax(1) == labels
    return {"fixed": int((~a & b).sum()), "broken": int((a & ~b).sum())}


def confidence_calibration(final: np.ndarray, labels: np.ndarray, bank: Bank) -> dict[str, float]:
    conf, ok = final.max(1), final.argmax(1) == labels
    at93 = conf >= 0.93
    bars = np.array([bank.bar_for(f) for f in final])
    own = conf >= bars
    return {
        "ece": expected_calibration_error(conf, ok),
        "share_at_least_93": float(at93.mean()), "accuracy_at_least_93": float(ok[at93].mean()) if at93.any() else math.nan,
        "wrong_at_least_93": int((~ok & at93).sum()), "sure_at_least_93": int(at93.sum()),
        "share_own_bar": float(own.mean()), "accuracy_own_bar": float(ok[own].mean()) if own.any() else math.nan,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_stage(root: Path) -> Path:
    test, e9_rec = e18.load_test(root)
    classes = test.classes
    params = Params.load(root / e18.PARAMS_REL)
    questions = load_questions(root / "datasets/followup_v2/questions.jsonl")
    bank = Bank(questions, classes, params)
    table = e18.grower_table(root, classes, [q.id for q in questions])
    old_rows, old_sup = old_bank(root, classes)
    old_table = e18.grower_table(root, classes, [q["id"] for q in old_rows], e18.GROWER_OLD)
    labels = test.labels

    systems: dict[str, np.ndarray] = {"photo_only": test.probs}
    extra: dict[str, dict[str, Any]] = {"photo_only": {"symptom_questions": np.zeros(len(test), int), "plant_asked": np.zeros(len(test), bool)}}
    seeds_summary: dict[str, Any] = {}

    def add(name: str, out: dict[str, Any]) -> None:
        systems[name] = out["final"]
        extra[name] = out

    add("plant_only", e18.simulate(bank, test, table, use_symptoms=False))
    add("new_perfect", e18.simulate(bank, test, table))
    for name, kw in (("new_flip10", {"flip": 0.10}), ("new_flip20", {"flip": 0.20}), ("new_unsure20", {"unsure": 0.20})):
        outs = [e18.simulate(bank, test, table, seed=s, **kw) for s in e18.SEEDS]
        add(name, outs[0])
        accs = [float((o["final"].argmax(1) == labels).mean()) for o in outs]
        fb = [compare(test.probs, o["final"], labels) for o in outs]
        seeds_summary[name] = {
            "seeds": len(outs), "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1)),
            "accuracy_min": float(min(accs)), "accuracy_max": float(max(accs)),
            "fixed_mean": float(np.mean([x["fixed"] for x in fb])), "broken_mean": float(np.mean([x["broken"] for x in fb])),
            "symptom_questions_mean": float(np.mean([o["symptom_questions"].mean() for o in outs])),
            "accuracy_at_least_93_mean": float(np.nanmean([confidence_calibration(o["final"], labels, bank)["accuracy_at_least_93"] for o in outs])),
            "ece_mean": float(np.mean([confidence_calibration(o["final"], labels, bank)["ece"] for o in outs])),
        }
    add("e11_original_rule", run_old(old_e11, old_rows, old_sup, old_table, test))
    add("e11_live_rule", run_old(old_live, old_rows, old_sup, old_table, test))
    for rule_name, rule in (("e11_original_rule", old_e11), ("e11_live_rule", old_live)):
        for cond, kw in (("flip10", {"flip": 0.10}), ("flip20", {"flip": 0.20}), ("unsure20", {"unsure": 0.20})):
            outs = [run_old(rule, old_rows, old_sup, old_table, test, seed=s, **kw) for s in e18.SEEDS]
            accs = [float((o["final"].argmax(1) == labels).mean()) for o in outs]
            fb = [compare(test.probs, o["final"], labels) for o in outs]
            seeds_summary[f"{rule_name}_{cond}"] = {
                "seeds": len(outs), "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1)),
                "fixed_mean": float(np.mean([x["fixed"] for x in fb])), "broken_mean": float(np.mean([x["broken"] for x in fb])),
                "symptom_questions_mean": float(np.mean([o["symptom_questions"].mean() for o in outs]))}

    info = split_info("field_test", root)
    groups = test.groups
    scored = {n: Scored(groups=groups, labels=labels, probs=p) for n, p in systems.items()}
    metrics = {n: standard_image_metrics(p, labels, classes) for n, p in systems.items()}
    intervals = {n: {m: list(recompute_interval(m, scored[n])) for m in ("accuracy", "macro_f1")} for n in systems}
    comparisons = []
    for n in systems:
        if n != "photo_only":
            for metric in ("accuracy", "macro_f1"):
                got = recompute_comparison(scored[n], scored["photo_only"], metric)
                comparisons.append({"a": n, "b": "photo_only", "metric": metric, "diff": got["diff"], "ci": got["ci"], "mcnemar_p": got["mcnemar_p"]})
    per_system: dict[str, Any] = {}
    for n, p in systems.items():
        sym = extra[n]["symptom_questions"]
        plant = extra[n].get("plant_asked", np.zeros(len(test), bool))
        per_system[n] = {**compare(test.probs, p, labels), "mean_symptom_questions": float(sym.mean()),
                         "share_asked_any_symptom_question": float((sym > 0).mean()), "share_asked_plant": float(np.mean(plant)),
                         "calibration": confidence_calibration(p, labels, bank)}
    fit = json.loads((root / e18.FIT_REL).read_text(encoding="utf-8"))
    stops = Counter_from(extra["new_perfect"]["stops"])
    config = {
        "class_names": classes, "stage": "record", "split_use": "field_test, scored with everything fitted on field_dev",
        "override_note": ("The user asked for this field_test measurement of the E18 question engine after E9's one run: no model is "
                          "scored again; the saved field_test probabilities of E9's improved_calibrated rows are the inputs and "
                          "every fitted number is frozen in configs/e18_params.json (fitted on field_dev only) before any test row is read."),
        "inputs": {"field_test_rows": f"{e9_rec['run_id']} improved_calibrated", "field_dev_rows": f"{e18.E15} convnextv2_final",
                   "temperature_file": e18.CHOSEN.as_posix()},
        "questions": {"file": "datasets/followup_v2/questions.jsonl", "count": len(questions), "sha256": sha256(root / "datasets/followup_v2/questions.jsonl")},
        "grower": {"file": e18.GROWER.as_posix(), "sha256": sha256(root / e18.GROWER),
                   "e11_questions_file": e18.GROWER_OLD.as_posix(), "e11_questions_sha256": sha256(root / e18.GROWER_OLD),
                   "note": ("Answers written by six separate Sonnet agents, one batch of classes each, given only the question text (ids "
                            "anonymised, the two question banks mixed and shuffled), the true condition's name and guide excerpts of that "
                            "condition, never the question's support sets. The guide excerpts are the same documents the questions were "
                            "written from, so this grower is better informed than a real grower and the perfect-grower row is an upper bound.")},
        "params": params.to_json(), "params_file": e18.PARAMS_REL.as_posix(), "params_sha256": sha256(root / e18.PARAMS_REL),
        "fit": {"file": e18.FIT_REL.as_posix(), "noisy_grower_fit": fit["noisy_grower_fit"]["noise"], "dev_in_sample": fit["dev_in_sample"]},
        "systems": {
            "photo_only": "the final model's calibrated probabilities, no questions",
            "plant_only": "the plant question only, answered with the true plant when it is among the plants the photo fits",
            "new_perfect": "plant question, then the new questions answered by the simulated grower's table",
            "new_flip10": "as new_perfect with 10% of yes/no answers flipped (seed 0 saved; 20 seeds in seeds_summary)",
            "new_flip20": "as new_perfect with 20% of yes/no answers flipped",
            "new_unsure20": "as new_perfect with 20% of answers replaced by not sure",
            "e11_original_rule": "the 52 E11 questions (38 touch the 29 classes), the E11 rule (x3 yes, x0.5 no, 2 questions, no plant question), simulated grower",
            "e11_live_rule": "the same questions with the live demo's old rule (plant first, x3 yes, x0.35 no, up to 5 steps), simulated grower",
        },
        "per_system": per_system, "seeds_summary": seeds_summary, "stops_new_perfect": stops,
        "e11_real_answer_replay": {"note": ("on field_dev with E11's saved blind Sonnet-vision answers, from the 2026-11-14 audit replay: plant question only "
                                            "0.8104 -> 0.8399 (30 fixed, 0 broken); plant plus the 52 questions under the live rule 0.8389 (54 fixed, 25 broken)")},
        "label": "development fit, one-time test measurement; a simulated grower, not people", "labels_reviewed": False,
        "cleaned_rows": len(test),
    }
    f, b = per_system["new_perfect"]["fixed"], per_system["new_perfect"]["broken"]
    notes = (f"E18 question engine, field_test ({len(test)} cleaned photos), everything fitted on field_dev. Photo only "
             f"{metrics['photo_only']['accuracy']:.4f}, plant question {metrics['plant_only']['accuracy']:.4f}, plant plus new questions with a "
             f"simulated perfect grower {metrics['new_perfect']['accuracy']:.4f} ({f} fixed, {b} broken); with 10% flipped answers "
             f"{seeds_summary['new_flip10']['accuracy_mean']:.4f}, 20% flipped {seeds_summary['new_flip20']['accuracy_mean']:.4f}, 20% not sure "
             f"{seeds_summary['new_unsure20']['accuracy_mean']:.4f} (means of 20 seeds). The grower is simulated from the same guides, an upper bound.")
    models = {n: f"{n}: {desc}" for n, desc in config["systems"].items()}
    predictions = {n: [{"path": pth, "label": int(y), "probs": [float(v) for v in pr]} for pth, y, pr in zip(test.paths, labels, p, strict=True)]
                   for n, p in systems.items()}
    attempts = [{"stage": "record", "fitted_on": "field_dev", "params_file": e18.PARAMS_REL.as_posix()}]
    return write_run("E18", "field_test", config, models, metrics, intervals, attempts, predictions, info.hash, root=root,
                     notes=notes, subset=e9_rec["subset"], comparisons=comparisons)


def Counter_from(xs: list[str]) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(xs))
