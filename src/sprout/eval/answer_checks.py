"""The numbers of E9's wrong-disease check, recomputed from saved rows. E9 writes them and check_run recomputes them
with these functions, so a changed row cannot keep its old number.

Dev rows (one per checked answer): `id, item, question, diagnosis, answer, call, reply`; `call` is FITS, WRONG or None
when the reply was not exactly one of the two words.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

CALLS = ("FITS", "WRONG")
DEV_ROW_KEYS = ("id", "item", "question", "diagnosis", "answer", "call", "reply")


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


FINAL_ROW_KEYS = ("id", "answered", "state", "diagnosis", "text", "claims", "wrong_disease", "wrong_disease_reply",
                  "haiku_calls", "sonnet_calls", "citations", "query", "fallback_fired", "generic_followup",
                  "wrong_disease_status", "wrong_disease_request_id")
CLAIM_KEYS = ("claim", "cited", "haiku", "haiku_reply", "sonnet", "sonnet_reply")


WD_STATUSES = ("no_claims", "graded", "refused_by_safeguard", "not_sent")


def wd_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The wrong-disease check's own account from the rows: how many answers it was to check, how many it graded, how
    many the safeguard refused (with their request ids), how many were never sent. The rate's denominator is the
    graded rows with a usable call; refused and unsent rows are counted as neither result. Complete means every
    answer with claims was graded."""
    checked = [r for r in rows if r["wrong_disease_status"] != "no_claims"]
    graded = [r for r in checked if r["wrong_disease_status"] == "graded"]
    refused = [r for r in checked if r["wrong_disease_status"] == "refused_by_safeguard"]
    unsent = [r for r in checked if r["wrong_disease_status"] == "not_sent"]
    usable = [r for r in graded if r["wrong_disease"] in CALLS]
    return {"answers_to_check": len(checked), "graded": len(graded), "usable_calls": len(usable),
            "unusable_replies": len(graded) - len(usable),
            "flagged_wrong_disease": sum(r["wrong_disease"] == "WRONG" for r in usable),
            "refused_by_safeguard": len(refused), "not_sent": len(unsent),
            "refused": [{"id": r["id"], "request_id": r["wrong_disease_request_id"]} for r in refused],
            "complete": not refused and not unsent}


def final_lists(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    """The per-claim and per-answer samples behind the figures of rag_test: the Haiku judge's and the Sonnet judge's call
    on the same claims (a reply that is not a verdict counts as not supported, for both), and the wrong-disease check's
    flag per checked answer. An empty sample is left out."""
    claims = [c for r in rows for c in r["claims"]]
    out = {"claims_supported_by_haiku": [c["haiku"] is True for c in claims],
           "claims_supported_by_sonnet": [c["sonnet"] is True for c in claims],
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
