"""Agreement between two graders of the same claims, from saved rows. E14 writes these numbers and check_run
recomputes them with the same code, so a changed row cannot keep its old number.

A row is one graded answer: `local` holds the local judge's grade per claim, `sonnet` the second grader's (None when
its reply was not one of the two allowed verdicts; such a claim is left out of every number here).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Counts are laid out [both supported, local only, second grader only, both not supported].
BOTH, LOCAL_ONLY, OTHER_ONLY, NEITHER = range(4)
ROW_KEYS = ("id", "item", "system", "claims", "local", "sonnet", "sonnet_reply", "calls")


def kappa(c: np.ndarray) -> np.ndarray | float:
    """Cohen's kappa from 2x2 counts (last axis of length 4); nan when chance agreement is already 1."""
    c = np.asarray(c, dtype=float)
    total = c.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = (c[..., BOTH] + c[..., NEITHER]) / total
        local_yes = (c[..., BOTH] + c[..., LOCAL_ONLY]) / total
        other_yes = (c[..., BOTH] + c[..., OTHER_ONLY]) / total
        chance = local_yes * other_yes + (1 - local_yes) * (1 - other_yes)
        out = np.where(chance == 1.0, np.nan, (observed - chance) / (1 - chance))
    return float(out) if out.ndim == 0 else out


def _share(num: np.ndarray, c: np.ndarray) -> np.ndarray | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(num, dtype=float) / np.asarray(c, dtype=float).sum(axis=-1)
    return float(out) if out.ndim == 0 else out


STATS = {
    "agreement": lambda c: _share(c[..., BOTH] + c[..., NEITHER], c),
    "kappa": kappa,
    "local_supported": lambda c: _share(c[..., BOTH] + c[..., LOCAL_ONLY], c),
    "sonnet_supported": lambda c: _share(c[..., BOTH] + c[..., OTHER_ONLY], c),
}
SERVED = "served_"


def _counts(local: list[bool], other: list[bool]) -> np.ndarray:
    pairs = list(zip(local, other, strict=True))
    return np.array([sum(a and b for a, b in pairs), sum(a and not b for a, b in pairs),
                     sum(b and not a for a, b in pairs), sum(not a and not b for a, b in pairs)], dtype=float)


def direction(local: list[bool], other: list[bool]) -> dict[str, int]:
    """Each claim the two graders split on is counted once: the local judge said supported and the other did not
    (the local judge is the kinder one), or the other way round."""
    c = _counts(local, other)
    return {"local_kinder": int(c[LOCAL_ONLY]), "local_stricter": int(c[OTHER_ONLY])}


def graded_claims(rows: list[dict[str, Any]]) -> tuple[list[bool], list[bool], list[str]]:
    """(local, second grader, question id) for every claim the second grader gave a verdict on."""
    local, other, groups = [], [], []
    for r in rows:
        for a, b in zip(r["local"], r["sonnet"], strict=True):
            if b is not None:
                local.append(bool(a))
                other.append(bool(b))
                groups.append(r["id"])
    return local, other, groups


def _served(rows: list[dict[str, Any]], served: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["system"] == served]


def _num(x: float) -> float | None:
    return None if x != x else float(x)


def metrics_from_rows(rows: list[dict[str, Any]], served: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for prefix, part in (("", rows), (SERVED, _served(rows, served))):
        local, other, _ = graded_claims(part)
        c = _counts(local, other)
        for name, fn in STATS.items():
            out[prefix + name] = _num(fn(c)) if c.sum() else None
        out[prefix + "claims_compared"] = float(len(local))
        out[prefix + "claims_not_graded"] = float(sum(b is None for r in part for b in r["sonnet"]))
        for k, v in direction(local, other).items():
            out[prefix + k] = float(v)
    served_rows = _served(rows, served)
    flagged = lambda key: sum(any(g is False for g in r[key]) for r in served_rows)  # noqa: E731
    out[SERVED + "answers"] = float(len(served_rows))
    out[SERVED + "answers_with_unsupported_local"] = float(flagged("local"))
    out[SERVED + "answers_with_unsupported_sonnet"] = float(flagged("sonnet"))
    return out


def _draws(group_counts: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Pooled counts after drawing whole groups with replacement: shape (n, 4)."""
    rng = np.random.default_rng(seed)
    g = len(group_counts)
    out = np.empty((n, 4))
    step = 1000
    for start in range(0, n, step):
        m = min(step, n - start)
        out[start : start + m] = group_counts[rng.integers(0, g, size=(m, g))].sum(axis=1)
    return out


def intervals_from_rows(rows: list[dict[str, Any]], served: str, n: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% bootstrap intervals, resampling whole questions (an answer to the same question under several prompts moves
    together). A statistic that is undefined in some resamples is taken over the resamples where it is defined."""
    out = {}
    for prefix, part in (("", rows), (SERVED, _served(rows, served))):
        local, other, groups = graded_claims(part)
        if not local:
            continue
        ids = sorted(set(groups))
        index = {g: i for i, g in enumerate(ids)}
        per_group = np.zeros((len(ids), 4))
        for g, a, b in zip(groups, local, other, strict=True):
            per_group[index[g], (BOTH if b else LOCAL_ONLY) if a else (OTHER_ONLY if b else NEITHER)] += 1
        pooled = _draws(per_group, n, seed)
        for name, fn in STATS.items():
            values = np.asarray(fn(pooled))
            if np.isfinite(values).any():
                lo, hi = np.nanpercentile(values, [2.5, 97.5])
                out[prefix + name] = (float(lo), float(hi))
    return out


def row_problems(rows: list[dict[str, Any]], dev_ids: set[str]) -> list[str]:
    """Why saved rows cannot be trusted as graded answers, if they cannot."""
    out = []
    if not rows:
        return ["no rows"]
    seen_item, seen_pair = set(), set()
    for i, r in enumerate(rows):
        tag = f"row {i}: "
        if set(r) != set(ROW_KEYS):
            out.append(tag + f"fields must be exactly {', '.join(ROW_KEYS)}")
            continue
        n = len(r["claims"]) if isinstance(r["claims"], list) else -1
        lists_ok = n > 0 and all(isinstance(r[k], list) and len(r[k]) == n for k in ("local", "sonnet", "sonnet_reply"))
        if not lists_ok:
            out.append(tag + "claims, local, sonnet and sonnet_reply must be lists of one length with at least one claim")
            continue
        if not all(isinstance(x, bool) for x in r["local"]) or not all(x is None or isinstance(x, bool) for x in r["sonnet"]):
            out.append(tag + "grades must be true or false (the second grader's may be null)")
        if r["calls"] != n:
            out.append(tag + "calls must be one per claim")
        if r["id"] not in dev_ids:
            out.append(tag + f"{r['id']} is not a rag_dev question")
        if r["item"] in seen_item or (r["id"], r["system"]) in seen_pair:
            out.append(tag + "an answer appears twice")
        seen_item.add(r["item"])
        seen_pair.add((r["id"], r["system"]))
    return out
