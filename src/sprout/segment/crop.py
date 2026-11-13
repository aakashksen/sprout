"""Cut a photo down to the predicted disease area."""
from __future__ import annotations

import numpy as np
from PIL import Image

TINY_FRACTION = 0.005  # a predicted area under this share of the photo is treated as noise


def crop_to_disease(image: Image.Image, mask: np.ndarray, margin: float = 0.1) -> Image.Image:
    """The box around the predicted disease pixels, grown by `margin` (a fraction of the box's width and height)
    on every side and kept inside the photo. Returns the full image when the area is tiny or empty."""
    w, h = image.size
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=bool)
    if mask.sum() < TINY_FRACTION * h * w:
        return image
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    dx, dy = round(margin * (x1 - x0)), round(margin * (y1 - y0))
    return image.crop((max(x0 - dx, 0), max(y0 - dy, 0), min(x1 + dx, w), min(y1 + dy, h)))
