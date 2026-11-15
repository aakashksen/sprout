"""The follow-up question engine (E18): a Bayesian-style update over all classes, with strengths fitted on field_dev.

Nothing here touches torch, an image or a language model. The odds start as the photo's calibrated probabilities and
are changed only by answers, by this fixed code:

* The plant question comes first whenever the photo fits more than one plant (it is almost free accuracy).
* A symptom question names the classes a "yes" supports and the classes a "no" supports (`yes_supports`,
  `no_supports`); every other class is a bystander for that question. Each class is therefore in one of three roles
  for a question: Y (a "yes" supports it), N (a "no" supports it) or O (neither).
* An answer multiplies each class by the strength for (its role, the answer). The strengths are the three-answer
  likelihoods P(yes | role), P(no | role), P(not sure | role) in `Params`; a bystander uses the mean of the Y and N
  rows. The k-th symptom question's update is raised to gamma**k, because the questions are not independent.
* The next question is the one with the largest expected information gain about the current top three (the
  distribution renormalised over them). It stops when the confidence bar is reached, when `max_questions` have been
  asked, or when no question splits the top three by at least `min_gain`.
* A healthy class needs a higher bar than a disease class, as in the photo-only rule.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "datasets" / "followup_v2" / "questions.jsonl"
PARAMS = ROOT / "configs" / "e18_params.json"
ANSWERS = ("yes", "no", "not sure")
YES, NO, UNSURE = 0, 1, 2
TOP_K = 3
PLANT_FLOOR = 0.05  # a plant is offered in the plant question when its classes hold at least this much probability


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    kind: str  # leaf_top, leaf_underside, stem, fruit, spread, timing or weather
    crop: str
    yes_supports: tuple[str, ...]
    no_supports: tuple[str, ...]
    passages: tuple[dict, ...]  # {"document", "page", "passage"}: the guide's own words

    @property
    def source(self) -> str:
        return self.passages[0]["document"] if self.passages else ""


@dataclass(frozen=True)
class Params:
    """The fitted numbers. `yes_row` is (P(yes), P(no), P(not sure)) for a class a "yes" supports; `no_row` the same
    for a class a "no" supports."""

    yes_row: tuple[float, float, float] = (0.55, 0.30, 0.15)
    no_row: tuple[float, float, float] = (0.12, 0.70, 0.18)
    gamma: float = 0.85
    plant_down: float = 0.02  # an answered plant question keeps the other plants' classes at this share of their weight
    bar: float = 0.93  # confidence that ends the questions
    healthy_bar: float = 0.965  # the stricter bar when the top class is a healthy class
    max_questions: int = 5
    min_gain: float = 0.02  # nats; below this no question "splits" the top three

    @staticmethod
    def load(path: Path = PARAMS) -> "Params":
        d = json.loads(path.read_text(encoding="utf-8"))["params"]
        return Params(tuple(d["yes_row"]), tuple(d["no_row"]), float(d["gamma"]), float(d["plant_down"]),
                      float(d["bar"]), float(d["healthy_bar"]), int(d["max_questions"]), float(d["min_gain"]))

    def to_json(self) -> dict:
        return asdict(self)


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [Question(r["id"], r["text"], r["kind"], r["crop"], tuple(r["yes_supports"]), tuple(r["no_supports"]),
                     tuple(r["passages"])) for r in rows]


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def is_healthy(cls: str) -> bool:
    return cls.endswith("healthy")


class Bank:
    """The questions as arrays over a fixed class list, so one call scores every question at once."""

    def __init__(self, questions: Sequence[Question], classes: Sequence[str], params: Params):
        self.questions = list(questions)
        self.classes = list(classes)
        self.params = params
        n, k = len(self.questions), len(self.classes)
        index = {c: i for i, c in enumerate(self.classes)}
        role = np.full((n, k), 2, dtype=np.int8)  # 0 Y, 1 N, 2 O
        for i, q in enumerate(self.questions):
            for c in q.yes_supports:
                if c in index:
                    role[i, index[c]] = 0
            for c in q.no_supports:
                if c in index:
                    role[i, index[c]] = 1
        self.role = role
        self.crops = [crop_of(c) for c in self.classes]
        self.healthy = np.array([is_healthy(c) for c in self.classes])
        self.set_params(params)

    def set_params(self, params: Params) -> None:
        self.params = params
        y, n = np.array(params.yes_row, float), np.array(params.no_row, float)
        self.like = np.stack([y, n, (y + n) / 2])  # (role, answer), strengths; rows are the likelihoods of each answer

    # ---- the plant question ----
    def plants(self, p: np.ndarray) -> list[str]:
        """Plants worth offering, likeliest first; fewer than two means the plant question is not needed."""
        mass: dict[str, float] = {}
        for crop, v in zip(self.crops, p, strict=True):
            mass[crop] = mass.get(crop, 0.0) + float(v)
        return [c for c, m in sorted(mass.items(), key=lambda t: -t[1]) if m >= PLANT_FLOOR]

    def answer_plant(self, p: np.ndarray, crop: str | None) -> np.ndarray:
        if crop is None:
            return p
        w = np.array([1.0 if c == crop else self.params.plant_down for c in self.crops])
        q = p * w
        return q / q.sum()

    # ---- symptom questions ----
    def update(self, p: np.ndarray, qi: int, answer: int, step: int) -> np.ndarray:
        """The odds after answer `answer` (YES, NO or UNSURE) to question `qi`, which is the `step`-th symptom question."""
        lik = self.like[self.role[qi], answer] ** (self.params.gamma ** step)
        q = p * lik
        return q / q.sum()

    def gains(self, p: np.ndarray, asked: Sequence[int] = ()) -> np.ndarray:
        """Expected drop in entropy of the top three for every question (-1 for the ones already asked)."""
        top = np.argsort(-p, kind="stable")[:TOP_K]
        q3 = p[top] / p[top].sum()
        lik = self.like[self.role[:, top]]  # (questions, 3 classes, 3 answers)
        joint = q3[None, :, None] * lik  # P(class, answer)
        p_ans = joint.sum(axis=1)  # (questions, answers)
        post = joint / np.maximum(p_ans[:, None, :], 1e-12)
        ent = -(post * np.log(np.maximum(post, 1e-12))).sum(axis=1)
        before = -(q3 * np.log(np.maximum(q3, 1e-12))).sum()
        g = before - (p_ans * ent).sum(axis=1)
        for i in asked:
            g[i] = -1.0
        return g

    def pick(self, p: np.ndarray, asked: Sequence[int] = ()) -> int | None:
        g = self.gains(p, asked)
        i = int(np.argmax(g))
        return i if g[i] >= self.params.min_gain else None

    # ---- confidence ----
    def bar_for(self, p: np.ndarray) -> float:
        return self.params.healthy_bar if self.healthy[int(np.argmax(p))] else self.params.bar

    def sure(self, p: np.ndarray) -> bool:
        return float(p.max()) >= self.bar_for(p)


@dataclass
class Trace:
    start: np.ndarray
    final: np.ndarray
    asked: list[tuple[str, int]] = field(default_factory=list)  # (question id or "plant", answer or -1)
    plant_asked: bool = False
    stop: str = ""  # sure, max_questions, no_split, sure_from_photo

    @property
    def n_symptom(self) -> int:
        return sum(1 for q, _ in self.asked if q != "plant")


def run(bank: Bank, p0: np.ndarray, plant_answer: Callable[[list[str]], str | None],
        answer: Callable[[int], int] | None, use_plant: bool = True, use_symptoms: bool = True) -> Trace:
    """Simulate (or drive) one check. `plant_answer(options)` returns the plant or None for "not listed";
    `answer(question index)` returns YES, NO or UNSURE."""
    p = p0.copy()
    tr = Trace(p0.copy(), p)
    if bank.sure(p):
        tr.stop = "sure_from_photo"
        return tr
    if use_plant:
        options = bank.plants(p)
        if len(options) >= 2:
            crop = plant_answer(options)
            tr.plant_asked = True
            tr.asked.append(("plant", -1))
            if crop in options:
                p = bank.answer_plant(p, crop)
    asked: list[int] = []
    while use_symptoms:
        if bank.sure(p):
            tr.stop = "sure"
            break
        if len(asked) >= bank.params.max_questions:
            tr.stop = "max_questions"
            break
        qi = bank.pick(p, asked)
        if qi is None:
            tr.stop = "no_split"
            break
        a = answer(qi)
        p = bank.update(p, qi, a, len(asked))
        asked.append(qi)
        tr.asked.append((bank.questions[qi].id, a))
    else:
        tr.stop = "sure" if bank.sure(p) else "no_symptom_questions"
    tr.final = p
    return tr


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    n, total = len(conf), 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum()) if p.size else math.nan
