"""The numbers of E9's wrong-disease check, recomputed from saved rows. E9 writes them and check_run recomputes them
with these functions, so a changed row cannot keep its old number.

Dev rows (one per checked answer): `id, item, question, diagnosis, answer, call, reply`; `call` is FITS, WRONG or None
when the reply was not exactly one of the two words. Blind-read rows: `id, item, grader, reader`.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

CALLS = ("FITS", "WRONG")
DEV_ROW_KEYS = ("id", "item", "question", "diagnosis", "answer", "call", "reply")
BLIND_ROW_KEYS = ("id", "item", "grader", "reader")


def parse_call(reply: str) -> str | None:
    """FITS or WRONG for exactly that one word (case and punctuation aside); anything longer is unusable, not a guess."""
    words = " ".join(re.findall(r"[A-Za-z]+", reply)).upper()
    return words if words in CALLS else None


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for k of n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def dev_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    usable = [r for r in rows if r["call"] in CALLS]
    flagged = sum(r["call"] == "WRONG" for r in usable)
    return {"answers_checked": float(len(rows)), "usable_calls": float(len(usable)),
            "unusable_replies": float(len(rows) - len(usable)), "flagged_wrong_disease": float(flagged),
            "flagged_rate": flagged / len(usable) if usable else None}


def dev_intervals(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    usable = [r for r in rows if r["call"] in CALLS]
    return {"flagged_rate": wilson(sum(r["call"] == "WRONG" for r in usable), len(usable))} if usable else {}


def strata_sizes(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many usable dev answers the grader called WRONG and FITS: the populations the blind sample is drawn from."""
    return {"grader_wrong": sum(r["call"] == "WRONG" for r in rows), "grader_fits": sum(r["call"] == "FITS" for r in rows)}


def draw_blind(rows: list[dict[str, Any]], seed: int, per_call: int) -> list[dict[str, Any]]:
    """A seeded sample with up to `per_call` answers from each of the grader's two calls, in a shuffled order, so the
    reader can judge both directions of error and the order tells nothing about the grader's call."""
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for call in CALLS:
        pool = sorted((r for r in rows if r["call"] == call), key=lambda r: r["id"])
        chosen += rng.sample(pool, min(per_call, len(pool)))
    rng.shuffle(chosen)
    return chosen


def blind_metrics(rows: list[dict[str, Any]], strata: dict[str, int]) -> dict[str, float | None]:
    """Where an independent reader disagrees with the grader, in each direction. `estimated_disagreement_rate` weights
    the two directions by how many dev answers the grader put in each call."""
    wrong = [r for r in rows if r["grader"] == "WRONG"]
    fits = [r for r in rows if r["grader"] == "FITS"]
    false_alarm = sum(r["reader"] == "FITS" for r in wrong)
    missed = sum(r["reader"] == "WRONG" for r in fits)
    fa = false_alarm / len(wrong) if wrong else None
    miss = missed / len(fits) if fits else None
    total = strata["grader_wrong"] + strata["grader_fits"]
    est = (strata["grader_wrong"] * fa + strata["grader_fits"] * miss) / total if fa is not None and miss is not None and total else None
    return {"read_answers": float(len(rows)), "read_where_grader_said_wrong": float(len(wrong)),
            "read_where_grader_said_fits": float(len(fits)), "reader_says_fits_after_wrong_call": float(false_alarm),
            "reader_says_wrong_after_fits_call": float(missed), "false_alarm_rate": fa, "missed_rate": miss,
            "agreement_in_sample": sum(r["grader"] == r["reader"] for r in rows) / len(rows) if rows else None,
            "estimated_disagreement_rate": est}


def blind_intervals(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    wrong = [r for r in rows if r["grader"] == "WRONG"]
    fits = [r for r in rows if r["grader"] == "FITS"]
    out: dict[str, list[float]] = {}
    if wrong:
        out["false_alarm_rate"] = wilson(sum(r["reader"] == "FITS" for r in wrong), len(wrong))
    if fits:
        out["missed_rate"] = wilson(sum(r["reader"] == "WRONG" for r in fits), len(fits))
    return out


FINAL_ROW_KEYS = ("id", "answered", "state", "diagnosis", "text", "claims", "wrong_disease", "wrong_disease_reply",
                  "sonnet_calls")
CLAIM_KEYS = ("claim", "cited", "local", "sonnet", "sonnet_reply")


def final_lists(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    """The per-claim and per-answer samples behind the figures of rag_test: the local judge's and Sonnet's call on the
    same claims, and the wrong-disease check's flag per checked answer. An empty sample is left out."""
    claims = [c for r in rows for c in r["claims"]]
    out = {"claims_supported_by_local_judge": [bool(c["local"]) for c in claims],
           "claims_supported_by_sonnet": [bool(c["sonnet"]) for c in claims if c["sonnet"] is not None],
           "answers_flagged_wrong_disease": [r["wrong_disease"] == "WRONG" for r in rows if r["wrong_disease"] in CALLS]}
    return {k: v for k, v in out.items() if v}


def exif_counts(paths: list[Path]) -> dict[str, int]:
    """How many of these photos carry an EXIF orientation tag, and how many of those serving would rotate (tag 2 to 8;
    evaluation reads the pixels as stored). Reads file headers only, never a model's output."""
    from PIL import Image

    tagged = rotated = 0
    for p in paths:
        with Image.open(p) as img:
            tag = img.getexif().get(0x0112)
        tagged += tag is not None
        rotated += tag in range(2, 9)
    return {"rows": len(paths), "with_orientation_tag": tagged, "rotated_by_serving": rotated}


def cleaned_rows(log: Any, version: int, before_ids: list[str], kept_ids: list[str]) -> dict[str, Any]:
    """How many test rows the label cleaning removed and why, from the exclusion log (a DataFrame with `version`,
    `path`, `reason_code` and `source`). E9 writes it and check_run recomputes it."""
    gone = set(before_ids) - set(kept_ids)
    hit = log[(log["version"] <= version) & log["path"].isin(gone)]
    return {"version": version, "before": len(before_ids), "after": len(kept_ids), "removed": len(gone),
            "by_reason": {k: int(v) for k, v in hit["reason_code"].value_counts().items()},
            "by_source": {k: int(v) for k, v in hit["source"].value_counts().items()}}
