"""Input checks: is this photo of a plant at all? The official PlantVillage release has no background class,
so the model alone cannot say "no plant"; E5 compares the checks below on field_dev and not_plant_dev."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = ROOT / "results" / "e5" / "chosen.json"
# The old rule, from legacy/PlantProject/application.py: OpenCV HSV (H 0-179) inside [35, 40, 40]..[85, 255, 255]
# counts as green, and a photo with under 5% green pixels is rejected.
GREEN_LOW, GREEN_HIGH = (35, 40, 40), (85, 255, 255)
GREEN_THRESHOLD = 0.05
KINDS = ("green_rule", "confidence", "confidence_ood")


def green_ratio(image: Image.Image) -> float:
    """Share of pixels the old rule calls green, with OpenCV's 8-bit RGB to HSV conversion."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    span = v - rgb.min(axis=-1)
    s = np.where(v > 0, 255.0 * span / np.where(v > 0, v, 1), 0.0)
    safe = np.where(span > 0, span, 1)
    h = np.where(v == r, 60 * (g - b) / safe, np.where(v == g, 120 + 60 * (b - r) / safe, 240 + 60 * (r - g) / safe))
    h = np.where(span > 0, h % 360, 0.0) / 2
    h, s = np.round(h), np.round(s)
    mask = (h >= GREEN_LOW[0]) & (h <= GREEN_HIGH[0]) & (s >= GREEN_LOW[1]) & (v >= GREEN_LOW[2])
    return float(mask.mean())


def normalized_entropy(probs) -> float:
    """Entropy of the probabilities divided by log(number of classes): 0 for one sure class, 1 for a flat spread.
    It is the out-of-distribution score, because it needs nothing but the probabilities."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(np.asarray(probs))))


@dataclass(frozen=True)
class GateSettings:
    kind: str
    green_threshold: float | None = None  # green_rule: accept at or above this share of green pixels
    confidence_threshold: float | None = None  # confidence, confidence_ood: accept at or above this top probability
    ood_threshold: float | None = None  # confidence_ood: and at or below this normalized entropy

    def accepts(self, *, green: float, probs) -> bool:
        if self.kind == "green_rule":
            return green >= self.green_threshold
        ok = float(np.max(probs)) >= self.confidence_threshold
        if self.kind == "confidence_ood":
            ok = ok and normalized_entropy(probs) <= self.ood_threshold
        return bool(ok)

    @classmethod
    def from_dict(cls, d: dict) -> GateSettings:
        if d.get("kind") not in KINDS:
            raise ValueError(f"unknown gate kind {d.get('kind')!r}")
        return cls(**d)


def load_settings(path: Path | None = None) -> GateSettings:
    """The gate E5 chose on its fitting part of dev data (results/e5/chosen.json)."""
    chosen = json.loads(Path(path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return GateSettings.from_dict(chosen["gate"]["settings"])


def is_plant_image(image: Image.Image, probs) -> bool:
    """True when the chosen gate lets the photo through. `probs` are the calibrated class probabilities."""
    return load_settings().accepts(green=green_ratio(image), probs=probs)
