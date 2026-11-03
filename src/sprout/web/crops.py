"""Map a classifier label such as Pepper,_bell___healthy to a crop used by the knowledge index."""

from __future__ import annotations

import re


def crop_for_label(label: str, known_crops: set[str]) -> str | None:
    """The crop's own documents, or None when the plant has none. Never another plant's crop."""
    plant = label.split("___", 1)[0]
    word = re.match(r"[a-z]+", plant.lower())
    crop = word.group(0) if word else None
    return crop if crop in known_crops else None
