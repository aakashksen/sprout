"""Upload checks: the bytes must decode as a JPEG or PNG of sensible size."""

from __future__ import annotations

import io
import secrets
from pathlib import Path

from PIL import Image, UnidentifiedImageError

FORMATS = {"JPEG": "jpg", "PNG": "png"}
MAX_SIDE = 10_000
MAX_PIXELS = 50_000_000
MIN_SIDE = 16


class BadImage(ValueError):
    pass


def check_image(data: bytes) -> str:
    """Return the file extension for a valid image, else raise BadImage."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            if fmt not in FORMATS:
                raise BadImage("only JPEG and PNG images are accepted")
            if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
                raise BadImage("image dimensions are out of range")
            im.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise BadImage("file is not a readable image") from e
    return FORMATS[fmt]


def save_private(upload_dir: Path, user_id: int, data: bytes, ext: str) -> str:
    """Write under uploads/<user_id>/ with a random name; return the path relative to upload_dir."""
    folder = upload_dir / str(user_id)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{secrets.token_hex(16)}.{ext}"
    (folder / name).write_bytes(data)
    return f"{user_id}/{name}"
