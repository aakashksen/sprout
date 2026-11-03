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
