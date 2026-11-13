"""Temperature scaling and abstention: when the model's answer is shown, and when it says it is unsure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ALTERNATIVES = 3
MIN_ALTERNATIVE = 0.10


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Softmax of the logits divided by the temperature. The argmax never changes."""
    return np.exp(_log_softmax(np.asarray(logits, dtype=float) / temperature))


def fit_temperature(logits, labels) -> float:
    """The temperature that minimises negative log-likelihood on these rows (golden-section search on its log,
    between 0.05 and 20). Log-probabilities work as logits, because softmax ignores a per-row constant."""
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    rows = np.arange(len(labels))

    def nll(log_t: float) -> float:
        return float(-_log_softmax(logits / np.exp(log_t))[rows, labels].mean())

    lo, hi = np.log(0.05), np.log(20.0)
    phi = (np.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = nll(b)
    return float(np.exp((lo + hi) / 2))


def choose_threshold(probs, labels, min_coverage: float = 0.5) -> float:
    """The confidence threshold with the lowest confident error rate among those that still answer at least
    `min_coverage` of the rows; a tie goes to the lower threshold. Candidates are the rows' own confidences.
    Call it on dev rows only."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    wrong = probs.argmax(axis=1) != labels
    best: tuple[float, float] | None = None
    for t in np.unique(conf):
        answered = conf >= t
        if answered.mean() < min_coverage:
            continue
        key = (float(wrong[answered].mean()), float(t))
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError("no threshold keeps the requested coverage")
    return best[1]


@dataclass(frozen=True)
class Decision:
    status: str  # "answered" or "unsure"
    confidence: float
    top_k: tuple[tuple[int, float], ...]  # (class index, probability): at most 3, each above 10%


def healthy_bar(threshold: float, healthy_threshold: float | None) -> float:
    """The bar a top class ending in "healthy" must clear: the stricter of the two thresholds."""
    return max(float(threshold), float(healthy_threshold or 0.0))


def healthy_unsure(label: str, confidence: float, bar: float) -> bool:
    """True when a healthy top label is below the healthy bar (the photo is then called unsure)."""
    return label.endswith("healthy") and confidence < bar


def answered_mask(probs, class_names: list[str], threshold: float, healthy_threshold: float | None = None) -> np.ndarray:
    """Per row, whether the served rule answers: top probability at the threshold, and a healthy top label also at the
    healthy bar. `Identifier` applies the same two functions to one photo."""
    probs = np.asarray(probs, dtype=float)
    bar = healthy_bar(threshold, healthy_threshold)
    top, conf = probs.argmax(axis=1), probs.max(axis=1)
    return np.array([c >= threshold and not healthy_unsure(class_names[i], c, bar) for i, c in zip(top, conf, strict=True)],
                    dtype=bool)


def decide(probs, threshold: float) -> Decision:
    """Answer when the top probability reaches the threshold, otherwise say unsure. The most likely classes
    come along either way, so an unsure reply can still name them."""
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")[:MAX_ALTERNATIVES]
    top_k = tuple((int(i), float(probs[i])) for i in order if probs[i] > MIN_ALTERNATIVE)
    confidence = float(probs.max())
    return Decision("answered" if confidence >= threshold else "unsure", confidence, top_k)
