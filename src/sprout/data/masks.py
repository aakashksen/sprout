"""Leaf masks: the official segmented PlantVillage image for lab photos, a colour threshold for any photo.

All masks are boolean arrays of shape (size, size), True on the leaf.
"""
from __future__ import annotations

import collections
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

SEGMENTED_DIR = "lab_segmented"
SEGMENTED_SUFFIX = "_final_masked"
BLACK_BELOW = 12  # the official segmented images have a black background; JPEG noise stays under this
HUE_RANGE = (12, 125)  # PIL hue is 0-255: about 17 to 176 degrees, yellow-brown through green to cyan
MIN_SATURATION = 60
MIN_VALUE = 50


@lru_cache(maxsize=None)
def _segmented_index(folder: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_stem: dict[str, Path] = {}
    by_tail: dict[str, list[Path]] = collections.defaultdict(list)
    for f in folder.iterdir():
        stem = f.stem.removesuffix(SEGMENTED_SUFFIX)
        by_stem[stem] = f
        by_tail[stem.split("___", 1)[-1]].append(f)
    return by_stem, dict(by_tail)


def segmented_path(lab_path: str | Path) -> Path | None:
    """The official segmented copy of a lab image, matched by file name in the same class folder.
    A few originals have no uuid prefix while their segmented copy has one, so the part after
    '___' is used when it names exactly one file. None when there is no match."""
    p = Path(lab_path)
    folder = p.parents[1].parent / SEGMENTED_DIR / p.parent.name
    if not folder.is_dir():
        return None
    by_stem, by_tail = _segmented_index(folder)
    if p.stem in by_stem:
        return by_stem[p.stem]
    found = by_tail.get(p.stem.split("___", 1)[-1], [])
    return found[0] if len(found) == 1 else None


def segmented_mask(lab_path: str | Path, size: int) -> np.ndarray:
    path = segmented_path(lab_path)
    if path is None:
        raise FileNotFoundError(f"no segmented copy of {lab_path}")
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB").resize((size, size), Image.NEAREST))
    return rgb.max(axis=2) >= BLACK_BELOW


def hsv_mask(image: Image.Image, size: int) -> np.ndarray:
    """Pixels that are saturated, not dark, and leaf-coloured. Soil, wood and skin can pass it."""
    hsv = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR).convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1]) & (s >= MIN_SATURATION) & (v >= MIN_VALUE)


def energy_outside(cam: np.ndarray, mask: np.ndarray) -> float | None:
    """Share of the Grad-CAM values (summed) that fall off the leaf; None when the map is all zero."""
    total = float(cam.sum())
    return float(cam[~mask].sum() / total) if total > 0 else None


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union; 1.0 when both masks are empty (nothing to find, nothing predicted)."""
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
