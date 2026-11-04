"""Follow-up questions for a photo the model is unsure about.

When two or more of the model's top-3 classes are plausible, ask what the person can see, using only symptom
text taken from the documents, and re-rank the top-3 with the answer. Nothing here touches torch or an image.

A question lists the classes whose document passage describes the symptom. "yes" raises those classes inside
the top-3, "no" lowers them, "unknown" changes nothing. The total probability of the top-3 stays the same, so
the probabilities outside it are untouched. Healthy classes have no symptom text, so no question is about one;
a "no" still helps them, because it lowers the diseases the symptom belongs to.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "datasets" / "followup" / "questions.jsonl"
CONFIG = ROOT / "configs" / "e11.json"
ANSWERS = ("yes", "no", "unknown")
TOP_K = 3

Top3 = list[tuple[str, float]]


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    classes: tuple[str, ...]  # classes whose document describes this symptom
    document: str
    page: int
    passage: str  # the document's own words, kept so a verifier can check the question against it


@dataclass(frozen=True)
class Rule:
    yes_factor: float  # a class tied to the symptom has its probability multiplied by this on "yes"
    no_factor: float  # ... and by this on "no"
    max_questions: int


def normalize(text: str) -> str:
    """The form in which a passage is compared with a page: straight quotes, one space, lower case."""
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [Question(r["id"], r["text"], tuple(r["classes"]), r["document"], int(r["page"]), r["passage"]) for r in rows]


def load_rule(path: Path = CONFIG) -> Rule:
    r = json.loads(path.read_text(encoding="utf-8"))["rule"]
    return Rule(float(r["yes_factor"]), float(r["no_factor"]), int(r["max_questions"]))


@cache
def _default_bank() -> tuple[Question, ...]:
    return tuple(load_questions())


@cache
def _default_rule() -> Rule:
    return load_rule()


def top3(probs: Sequence[float], names: Sequence[str]) -> Top3:
    """The three largest probabilities, largest first; a tie goes to the lower class index."""
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:TOP_K]
    return [(names[i], float(probs[i])) for i in order]


def rerank(probs: Sequence[float], names: Sequence[str], new_top3: Top3) -> list[float]:
    """The full probability vector with the top-3 entries replaced by their updated values."""
    out = [float(p) for p in probs]
    index = {n: i for i, n in enumerate(names)}
    for name, p in new_top3:
        out[index[name]] = p
    return out


def is_unsure(top: Top3, min_probability: float, min_plausible: int = 2) -> bool:
    """True when at least `min_plausible` of the top-3 classes have a probability of at least `min_probability`."""
    return sum(p >= min_probability for _, p in top) >= min_plausible


def uncovered(top: Top3, bank: Sequence[Question] | None = None) -> list[str]:
    """Top-3 classes that have no question and are not healthy. Healthy classes have no symptom text, which is
    not a gap in the documents."""
    bank = _default_bank() if bank is None else bank
    covered = {c for item in bank for c in item.classes}
    return [c for c, _ in top if c not in covered and not c.endswith("healthy")]


def askable(top: Top3, bank: Sequence[Question] | None = None) -> list[Question]:
    """Every question that could be asked about this top-3: tied to at least one of its classes and not to all
    of them, since a question every candidate shares tells them apart not at all."""
    bank = _default_bank() if bank is None else bank
    names = {c for c, _ in top}
    return [item for item in bank if names & set(item.classes) and not names <= set(item.classes)]


def next_question(top: Top3, *, asked: Sequence[str], bank: Sequence[Question] | None = None) -> Question | None:
    """The unasked askable question that best splits the top-3 probability in half, or None.

    The one whose classes hold the share of top-3 probability closest to one half is chosen; a tie goes to the
    lower question id. Classes without a question are never the subject of one."""
    total = sum(p for _, p in top)
    if total <= 0:
        return None
    best: tuple[float, str] | None = None
    chosen: Question | None = None
    for item in askable(top, bank):
        if item.id in asked:
            continue
        share = sum(p for c, p in top if c in item.classes) / total
        key = (abs(share - 0.5), item.id)
        if best is None or key < best:
            best, chosen = key, item
    return chosen


def update(top: Top3, question: Question, answer: str, *, rule: Rule | None = None) -> Top3:
    """The top-3 after an answer, largest first, with the same total probability. "unknown" returns it unchanged."""
    if answer not in ANSWERS:
        raise ValueError(f"answer must be one of {ANSWERS}, not {answer!r}")
    if answer == "unknown":
        return list(top)
    rule = _default_rule() if rule is None else rule
    factor = rule.yes_factor if answer == "yes" else rule.no_factor
    weights = [p * (factor if c in question.classes else 1.0) for c, p in top]
    total, scale = sum(p for _, p in top), sum(weights)
    if scale <= 0:
        return list(top)
    new = [(c, w * total / scale) for (c, _), w in zip(top, weights, strict=True)]
    return sorted(new, key=lambda cp: -cp[1])  # stable: equal values keep their order
