"""E18: follow-up questions that finish the diagnosis, fitted on field_dev and measured once on field_test.

No GPU, no model, no Claude call here. The inputs are the final model's saved probabilities (E15 for field_dev, E9's
`improved_calibrated` rows for field_test), the question bank `datasets/followup_v2/questions.jsonl`, and the
simulated grower's answer table `datasets/followup_v2/grower_answers.jsonl` (one answer per true class and question,
written by separate Sonnet agents that were given the question text, the true disease's name and its guide passages,
and never the question's support sets).

Stages:
  fit      field_dev only. Fits the answer strengths and damping (Nelder-Mead on the log-loss of the final odds),
           then the confidence bar, the question budget and the split threshold on a small grid. Writes
           configs/e18_params.json and results/e18/fit.json.
  record   field_test, with the fitted numbers frozen. Writes the E18 run record.

The old E11 questions are scored on the same footing, with a grower table made the same way, and the old rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sprout.atomic import atomic_write_text
from sprout.eval.records import ROOT, split_info, subset_positions
from sprout.serve.question_engine import (
    NO,
    UNSURE,
    YES,
    Bank,
    Params,
    expected_calibration_error,
    load_questions,
    run,
)

E15 = "E15-20261104T220052184613Z"
E9 = "E9-20261105T003314703226Z"
CHOSEN = Path("results/e15/refit/chosen.json")
GROWER = Path("datasets/followup_v2/grower_answers.jsonl")
GROWER_OLD = Path("datasets/followup_v2/grower_answers_e11_questions.jsonl")
OLD_QUESTIONS = Path("datasets/followup/questions.jsonl")
PARAMS_REL = Path("configs/e18_params.json")
FIT_REL = Path("results/e18/fit.json")
ANSWER_CODE = {"yes": YES, "no": NO, "not sure": UNSURE}
FIT_NOISE = {"flip": 0.10, "unsure": 0.10}  # the noise the shipped fit is trained under (field_dev only)
SEEDS = tuple(range(20))
ACCURACY_TOLERANCE = 0.01


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def calibrate(raw: np.ndarray, temperature: float) -> np.ndarray:
    lg = np.log(np.maximum(raw, 1e-12)) / temperature
    e = np.exp(lg - lg.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Data:
    """A split's calibrated probabilities, labels, groups and plants."""

    def __init__(self, probs: np.ndarray, labels: np.ndarray, groups: list[str], paths: list[str], classes: list[str]):
        self.probs, self.labels, self.groups, self.paths, self.classes = probs, labels, groups, paths, classes
        self.crops = [c.split("___")[0] for c in classes]

    def __len__(self) -> int:
        return len(self.labels)


def load_dev(root: Path) -> Data:
    chosen = json.loads((root / CHOSEN).read_text(encoding="utf-8"))
    rec = json.loads((root / "results/runs" / f"{E15}.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / rec["predictions"]["convnextv2_final"])
    info = split_info("field_dev", root)
    pos = subset_positions(rec["subset"], "field_dev", info, root)
    if [r["path"] for r in rows] != [info.ids[i] for i in pos]:
        raise ValueError("E15 rows are not the record's subset in order")
    raw = np.array([r["probs"] for r in rows], dtype=float)
    return Data(calibrate(raw, chosen["temperature"]), np.array([r["label"] for r in rows]),
                [info.groups[i] for i in pos], [r["path"] for r in rows], chosen["classes"])


def load_test(root: Path) -> tuple[Data, dict[str, Any]]:
    chosen = json.loads((root / CHOSEN).read_text(encoding="utf-8"))
    rec = json.loads((root / "results/runs" / f"{E9}.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / rec["predictions"]["improved_calibrated"])
    info = split_info("field_test", root)
    pos = subset_positions(rec["subset"], "field_test", info, root)
    if [r["path"] for r in rows] != [info.ids[i] for i in pos]:
        raise ValueError("E9 rows are not the record's subset in order")
    probs = np.array([r["probs"] for r in rows], dtype=float)
    return Data(probs, np.array([r["label"] for r in rows]), [info.groups[i] for i in pos],
                [r["path"] for r in rows], chosen["classes"]), rec


def grower_table(root: Path, classes: list[str], question_ids: list[str], path: Path = GROWER) -> np.ndarray:
    """(class, question) -> YES / NO / UNSURE, from the simulated grower's saved answers."""
    ci, qi = {c: i for i, c in enumerate(classes)}, {q: i for i, q in enumerate(question_ids)}
    table = np.full((len(classes), len(question_ids)), -1, dtype=np.int8)
    for r in read_jsonl(root / path):
        if r["true_class"] in ci and r["question"] in qi:
            table[ci[r["true_class"]], qi[r["question"]]] = ANSWER_CODE[r["answer"]]
    if (table < 0).any():
        raise ValueError(f"{path}: the table has missing (class, question) answers")
    return table


def noisy(answer: int, u1: float, u2: float, flip: float, unsure: float) -> int:
    """A yes becomes a no and a no a yes with chance `flip`; then an answer becomes "not sure" with chance `unsure`."""
    if answer != UNSURE and u1 < flip:
        answer = 1 - answer
    if u2 < unsure:
        answer = UNSURE
    return answer


def simulate(bank: Bank, data: Data, table: np.ndarray, *, use_plant: bool = True, use_symptoms: bool = True,
             flip: float = 0.0, unsure: float = 0.0, seed: int = 0, rows: np.ndarray | None = None) -> dict[str, Any]:
    """Run every photo through the engine with a simulated grower. The grower knows the true plant; a question's
    answer is the table's for the true class, then noise. Returns final odds, questions asked and stop reasons."""
    n, nq = len(data), len(bank.questions)
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random((n, nq)), rng.random((n, nq))
    idx = np.arange(n) if rows is None else rows
    final = data.probs.copy()
    sym = np.zeros(n, dtype=int)
    plant = np.zeros(n, dtype=bool)
    stops: list[str] = [""] * n
    asked: list[list[tuple[str, int]]] = [[] for _ in range(n)]
    for i in idx:
        y = int(data.labels[i])
        crop = data.crops[y]
        tr = run(bank, data.probs[i], lambda options, crop=crop: crop if crop in options else None,
                 lambda qi, i=i, y=y: noisy(int(table[y, qi]), u1[i, qi], u2[i, qi], flip, unsure),
                 use_plant=use_plant, use_symptoms=use_symptoms)
        final[i], sym[i], plant[i], stops[i], asked[i] = tr.final, tr.n_symptom, tr.plant_asked, tr.stop, tr.asked
    return {"final": final, "symptom_questions": sym, "plant_asked": plant, "stops": stops, "asked": asked}


# ---- fitting on field_dev ----

def softmax3(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def params_from_vector(v: np.ndarray, base: Params) -> Params:
    from dataclasses import replace

    gamma = 1.0 / (1.0 + np.exp(-v[4]))
    return replace(base, yes_row=tuple(softmax3(np.array([v[0], v[1], 0.0]))), no_row=tuple(softmax3(np.array([v[2], v[3], 0.0]))),
                   gamma=float(gamma))


def vector_from_params(p: Params) -> np.ndarray:
    y, n = np.array(p.yes_row), np.array(p.no_row)
    g = min(max(p.gamma, 1e-3), 1 - 1e-3)
    return np.array([np.log(y[0] / y[2]), np.log(y[1] / y[2]), np.log(n[0] / n[2]), np.log(n[1] / n[2]), np.log(g / (1 - g))])


def dev_loss(bank: Bank, data: Data, table: np.ndarray, flip: float, unsure: float, seeds: tuple[int, ...]) -> float:
    """Mean negative log-likelihood of the true class in the final odds, over the photos that are not already sure."""
    unsure_rows = np.array([i for i in range(len(data)) if not bank.sure(data.probs[i])])
    total = 0.0
    for s in seeds:
        out = simulate(bank, data, table, flip=flip, unsure=unsure, seed=s, rows=unsure_rows)
        f = out["final"][unsure_rows, data.labels[unsure_rows]]
        total += float(-np.log(np.maximum(f, 1e-9)).mean())
    return total / len(seeds)


def fit_strengths(bank: Bank, data: Data, table: np.ndarray, flip: float, unsure: float, seeds=(0, 1), start: Params | None = None,
                  maxiter: int = 160) -> tuple[Params, dict[str, Any]]:
    from scipy.optimize import minimize

    base = start or bank.params
    history: list[float] = []

    def f(v: np.ndarray) -> float:
        bank.set_params(params_from_vector(v, base))
        loss = dev_loss(bank, data, table, flip, unsure, seeds)
        history.append(loss)
        return loss

    v0 = vector_from_params(base)
    res = minimize(f, v0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-4, "initial_simplex": None})
    best = params_from_vector(res.x, base)
    bank.set_params(best)
    return best, {"start_loss": history[0], "final_loss": float(res.fun), "evaluations": len(history), "converged": bool(res.success)}


def summarise(bank: Bank, data: Data, out: dict[str, Any]) -> dict[str, float]:
    final, y = out["final"], data.labels
    pred = final.argmax(axis=1)
    conf = final.max(axis=1)
    ok = pred == y
    bars = np.array([bank.bar_for(f) for f in final])
    sure = conf >= bars
    return {"accuracy": float(ok.mean()), "mean_symptom_questions": float(out["symptom_questions"].mean()),
            "sure_share": float(sure.mean()), "sure_precision": float(ok[sure].mean()) if sure.any() else float("nan"),
            "ece": expected_calibration_error(conf, ok)}


def choose_settings(bank: Bank, data: Data, table: np.ndarray, flip: float, unsure: float, seeds=(0, 1)) -> tuple[Params, list[dict]]:
    """The bar, the question budget and the split threshold, on a small grid, on field_dev. Rule: the sure photos must be
    at least 95% right on field_dev; among those settings, the one that asks the fewest questions on average while
    staying within ACCURACY_TOLERANCE of the best accuracy."""
    from dataclasses import replace

    base = bank.params
    grid = []
    for bar in (0.90, 0.93, 0.95, 0.97):
        for maxq in (3, 5, 8):
            for gain in (0.01, 0.03):
                p = replace(base, bar=bar, healthy_bar=min(0.995, bar + 0.035), max_questions=maxq, min_gain=gain)
                bank.set_params(p)
                rs = [summarise(bank, data, simulate(bank, data, table, flip=flip, unsure=unsure, seed=s)) for s in seeds]
                grid.append({"bar": bar, "max_questions": maxq, "min_gain": gain,
                             **{k: float(np.mean([r[k] for r in rs])) for k in rs[0]}})
    ok = [g for g in grid if g["sure_precision"] >= 0.95] or grid
    top = max(g["accuracy"] for g in ok)
    best = min((g for g in ok if g["accuracy"] >= top - ACCURACY_TOLERANCE), key=lambda g: g["mean_symptom_questions"])
    chosen = replace(base, bar=best["bar"], healthy_bar=min(0.995, best["bar"] + 0.035), max_questions=best["max_questions"],
                     min_gain=best["min_gain"])
    bank.set_params(chosen)
    return chosen, grid


def fit_stage(root: Path) -> Path:

    dev = load_dev(root)
    questions = load_questions(root / "datasets/followup_v2/questions.jsonl")
    bank = Bank(questions, dev.classes, Params())
    table = grower_table(root, dev.classes, [q.id for q in questions])
    log: dict[str, Any] = {"rows": len(dev)}
    perfect, rep_p = fit_strengths(bank, dev, table, 0.0, 0.0, start=Params(bar=0.93, healthy_bar=0.965, max_questions=5, min_gain=0.02))
    log["perfect_grower_fit"] = {"params": perfect.to_json(), **rep_p}
    noisy_fit, rep_n = fit_strengths(bank, dev, table, **FIT_NOISE, start=perfect)
    log["noisy_grower_fit"] = {"noise": FIT_NOISE, "params": noisy_fit.to_json(), **rep_n}
    bank.set_params(noisy_fit)
    chosen, grid = choose_settings(bank, dev, table, **FIT_NOISE)
    log["grid"] = grid
    log["chosen"] = chosen.to_json()
    # dev numbers under each grower, fitted-on rows (in sample)
    dev_rows = {}
    for name, (f, u) in {"photo_only_bar": (None, None), "perfect": (0.0, 0.0), "flip10": (0.10, 0.0), "flip20": (0.20, 0.0), "unsure20": (0.0, 0.20)}.items():
        if f is None:
            dev_rows[name] = {"accuracy": float((dev.probs.argmax(1) == dev.labels).mean())}
        else:
            rs = [summarise(bank, dev, simulate(bank, dev, table, flip=f, unsure=u, seed=s)) for s in SEEDS[:5]]
            dev_rows[name] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    pl = simulate(bank, dev, table, use_symptoms=False)
    dev_rows["plant_only"] = summarise(bank, dev, pl)
    log["dev_in_sample"] = dev_rows
    log["note"] = ("Everything here is fitted and scored on field_dev (in sample); the grower's answers are the saved table. "
                   "The shipped strengths are the ones fitted under 10% flipped and 10% not-sure answers.")
    atomic_write_text(root / PARAMS_REL, json.dumps({"experiment": "E18", "fitted_on": "field_dev", "e15_record": E15,
                                                      "params": chosen.to_json(), "perfect_grower_fit": perfect.to_json()}, indent=1) + "\n")
    (root / FIT_REL).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(root / FIT_REL, json.dumps(log, indent=1) + "\n")
    return root / PARAMS_REL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fit", "record"))
    a = ap.parse_args(argv)
    try:
        if a.stage == "fit":
            print(fit_stage(ROOT))
        else:
            from sprout.experiments.e18_record import record_stage

            print(record_stage(ROOT))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
