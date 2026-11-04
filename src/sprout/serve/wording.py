"""Words the app uses about a healthy result, in one place. A photo can look healthy and still hide disease (early
or inside the plant), and no confidence figure changes that, so every place that presents a healthy label says so."""

from __future__ import annotations

HEALTHY_CAUTION = (
    "A healthy result does not rule out disease, at any confidence: early or hidden problems can look healthy "
    "in a photo."
)


def is_healthy(label: str | None) -> bool:
    return bool(label) and "healthy" in label.lower()
