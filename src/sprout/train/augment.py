"""Training-only image changes added for E4: stronger colour, blur, and new backgrounds behind a leaf."""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

EXTRA = ("strong_color", "blur", "background")
BACKGROUND_CHANCE = 0.7
BLUR_CHANCE = 0.5
MAX_BLUR_RADIUS = 2.5


def strong_color(image: Image.Image, rng: random.Random) -> Image.Image:
    """Wider brightness, contrast, saturation and sharpness swings, and a hue shift."""
    for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color, ImageEnhance.Sharpness):
        image = enhance(image).enhance(rng.uniform(0.4, 1.6))
    shift = rng.randint(-12, 12)
    h, s, v = image.convert("HSV").split()
    h = h.point(lambda x: (x + shift) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def blur(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() >= BLUR_CHANCE:
        return image
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, MAX_BLUR_RADIUS)))


def random_background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """A plain colour, a two-colour gradient, or blurred colour noise, in soil, sky, grass and cloth tones."""
    w, h = size
    nprng = np.random.default_rng(rng.getrandbits(32))

    def colour() -> np.ndarray:
        return nprng.integers(20, 236, size=3).astype(float)

    kind = rng.choice(("solid", "gradient", "noise"))
    if kind == "solid":
        arr = np.broadcast_to(colour(), (h, w, 3))
    elif kind == "gradient":
        t = np.linspace(0, 1, h if rng.random() < 0.5 else w)
        a, b = colour(), colour()
        ramp = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
        arr = ramp[:, None, :].repeat(w, 1) if len(t) == h else ramp[None, :, :].repeat(h, 0)
    else:
        small = nprng.integers(0, 256, size=(6, 6, 3)).astype(np.uint8)
        return Image.fromarray(small).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(4))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def paste_background(image: Image.Image, leaf: np.ndarray, rng: random.Random) -> Image.Image:
    """The leaf pixels of `image` (True in `leaf`, shape H x W) kept as they are, everything else replaced."""
    image = image.convert("RGB")
    if leaf.shape != (image.height, image.width):
        leaf = np.asarray(Image.fromarray(leaf.astype(np.uint8)).resize(image.size, Image.NEAREST), dtype=bool)
    bg = np.asarray(random_background(image.size, rng))
    out = np.where(leaf[..., None], np.asarray(image), bg)
    return Image.fromarray(out.astype(np.uint8))


def augment_extra(image: Image.Image, transforms: tuple[str, ...]) -> Image.Image:
    """The E4 colour and blur changes, drawing from Python's `random` like the older transforms."""
    if "strong_color" in transforms:
        image = strong_color(image, random)
    if "blur" in transforms:
        image = blur(image, random)
    return image
