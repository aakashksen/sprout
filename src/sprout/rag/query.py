"""Search text built from the question, the conversation and the photo diagnosis (kept apart from instructions)."""

from __future__ import annotations

import re

from sprout.serve.types import Identification

QUERY_MODES = ("question", "history", "diseases", "full")
MAX_QUERY_CHARS = 400
HISTORY_TURNS_IN_QUERY = 2
MIN_CANDIDATE_CONFIDENCE = 0.05


def crop_of_label(label: str, known_crops: set[str]) -> str | None:
    """The crop's own documents for a classifier label such as Pepper,_bell___healthy, or None."""
    word = re.match(r"[a-z]+", label.split("___", 1)[0].lower())
    crop = word.group(0) if word else None
    return crop if crop in known_crops else None


def _forms(crop: str) -> str:
    plural = crop[:-1] + "ies" if crop.endswith("y") else crop + "(?:s|es)?"
    return rf"\b(?:{re.escape(crop)}(?:s|es)?|{plural})\b"


def detect_crop(text: str, crops: set[str]) -> str | None:
    """The plant named first in the text, among crops that have documents."""
    best = None
    for crop in crops:
        m = re.search(_forms(crop), text.lower())
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), crop)
    return best[1] if best else None


def disease_of_label(label: str) -> str:
    """'Tomato___Early_blight' -> 'early blight'; healthy and unlabeled classes give ''."""
    _, _, disease = label.partition("___")
    disease = disease.replace("_", " ").strip().lower()
    return "" if disease in ("", "healthy") else disease


def top_diseases(ident: Identification | None, n: int = 3) -> list[str]:
    if ident is None or ident.top is None:
        return []
    found: list[str] = []
    for c in [ident.top, *ident.alternatives][:n]:
        d = disease_of_label(c.label)
        if d and c.confidence >= MIN_CANDIDATE_CONFIDENCE and d not in found:
            found.append(d)
    return found


def build_query(question: str, ident: Identification | None, history: list[dict], mode: str = "question") -> str:
    """`question` alone, or with the user's last turns and/or the top-3 diseases. Only the user's words and
    disease names go in; assistant turns and instructions never do."""
    if mode not in QUERY_MODES:
        raise ValueError(f"unknown query mode {mode!r}")
    parts = [question.strip()]
    if mode in ("history", "full"):
        user_turns = [h["content"].strip() for h in history if h.get("role") == "user"]
        parts = user_turns[-HISTORY_TURNS_IN_QUERY:] + parts
    if mode in ("diseases", "full"):
        parts += top_diseases(ident)
    return " ".join(p for p in parts if p)[:MAX_QUERY_CHARS]
