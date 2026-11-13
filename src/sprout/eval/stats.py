import math

import numpy as np


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mcnemar_exact(a_correct, b_correct) -> float:
    """Two-sided exact McNemar p-value from paired correct/incorrect outcomes."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _group_sums(values: np.ndarray, groups) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return np.bincount(inverse, weights=values), np.bincount(inverse).astype(float)


def _resampled_means(sums: np.ndarray, counts: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Pooled mean after drawing whole groups with replacement."""
    rng = np.random.default_rng(seed)
    g = len(sums)
    out = np.empty(n)
    step = 1000
    for start in range(0, n, step):
        m = min(step, n - start)
        draw = rng.integers(0, g, size=(m, g))
        out[start : start + m] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return out


def bootstrap_ci(values, groups, n: int = 10000, seed: int = 13) -> tuple[float, float]:
    """95% interval of the mean, resampling whole groups."""
    sums, counts = _group_sums(np.asarray(values, dtype=float), groups)
    lo, hi = np.percentile(_resampled_means(sums, counts, n, seed), [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    a, b, groups, n: int = 10000, seed: int = 13
) -> tuple[float, float, float]:
    """Mean(a) - mean(b) with a 95% interval, resampling whole groups of paired items."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sums, counts = _group_sums(diffs, groups)
    lo, hi = np.percentile(_resampled_means(sums, counts, n, seed), [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(lo), float(hi)


def _group_confusions(labels, pred, groups, num_classes: int) -> np.ndarray:
    """One flattened confusion matrix per group: shape (groups, num_classes**2)."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    cell = np.asarray(labels) * num_classes + np.asarray(pred)
    flat = np.zeros((inverse.max() + 1, num_classes**2))
    np.add.at(flat, (inverse, cell), 1.0)
    return flat


def _macro_f1_of(cms: np.ndarray, num_classes: int) -> np.ndarray:
    """Macro-F1 for each confusion matrix (rows of `cms`), over classes with support."""
    cm = cms.reshape(-1, num_classes, num_classes)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    support, predicted = cm.sum(axis=2), cm.sum(axis=1)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return (f1 * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1)


def _macro_f1_draws(flats: list[np.ndarray], num_classes: int, n: int, seed: int) -> list[np.ndarray]:
    """Macro-F1 of each system on the same resamples of whole groups."""
    rng = np.random.default_rng(seed)
    g = flats[0].shape[0]
    out = [np.empty(n) for _ in flats]
    step = 250
    for start in range(0, n, step):
        m = min(step, n - start)
        draw = rng.integers(0, g, size=(m, g)) + (np.arange(m) * g)[:, None]
        weights = np.bincount(draw.ravel(), minlength=m * g).reshape(m, g).astype(float)
        for o, flat in zip(out, flats, strict=True):
            o[start : start + m] = _macro_f1_of(weights @ flat, num_classes)
    return out


def bootstrap_macro_f1(labels, pred, groups, num_classes: int, n: int = 10000, seed: int = 13) -> np.ndarray:
    """Macro-F1 recomputed on each of n resamples that draw whole groups with replacement."""
    flat = _group_confusions(labels, pred, groups, num_classes)
    return _macro_f1_draws([flat], num_classes, n, seed)[0]


def macro_f1_ci(labels, pred, groups, num_classes: int, n: int = 10000, seed: int = 13) -> tuple[float, float]:
    lo, hi = np.percentile(bootstrap_macro_f1(labels, pred, groups, num_classes, n, seed), [2.5, 97.5])
    return float(lo), float(hi)


def paired_macro_f1_diff(
    labels, pred_a, pred_b, groups, num_classes: int, n: int = 10000, seed: int = 13
) -> tuple[float, float, float]:
    """Macro-F1(a) - macro-F1(b) with a 95% interval; both systems use the same resampled groups."""
    fa = _group_confusions(labels, pred_a, groups, num_classes)
    fb = _group_confusions(labels, pred_b, groups, num_classes)
    point = _macro_f1_of(fa.sum(axis=0, keepdims=True), num_classes)[0] - _macro_f1_of(
        fb.sum(axis=0, keepdims=True), num_classes
    )[0]
    da, db = _macro_f1_draws([fa, fb], num_classes, n, seed)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return float(point), float(lo), float(hi)
