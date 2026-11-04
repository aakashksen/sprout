"""The real identifier: photo bytes in, an `Identification` out. No torch.

Order of checks: bytes and header size, EXIF orientation, model, temperature, then the input gate E5 chose
(`not_plant`), then the abstention threshold E5 chose (`unsure`). Paths are arguments; the calibration file is the
`results/e5/chosen.json` shape E5 writes (`temperature`, `abstain_threshold`, `gate.settings`), plus `model_sha256`:
the sha256 of the checkpoint the numbers were fitted for. The model file carries the sha256 of the checkpoint it was
exported from, and the two must be equal, or nothing is loaded. There is no way around the check.
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from sprout.serve.calibrate import apply_temperature, decide
from sprout.serve.export import OnnxModel
from sprout.serve.gate import green_ratio, load_settings
from sprout.serve.types import Candidate, Identification

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 60_000_000  # a 48 MP phone photo fits; checked from the header, before any pixel is decoded
FLOOR = 1e-12  # the same floor E5 puts under probabilities before taking logs

PHOTO_TIPS = (
    "Take a close photo of one affected leaf in good, even light (daylight, no flash glare or shadow), "
    "in focus and filling most of the frame, against a plain background."
)
NOT_PLANT = "This doesn't look like a plant leaf. Photograph one leaf of the plant, close up."
INVALID = "That file could not be read as a photo. Upload a JPEG or PNG picture."
TOO_LARGE = "That photo is too large. Upload a smaller picture (under 20 MB and 60 megapixels)."


class Identifier:
    def __init__(self, model_path: Path, calibration_path: Path):
        self.model = OnnxModel(Path(model_path))
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        _require_same_checkpoint(self.model.source_sha256, calibration.get("model_sha256"), model_path, calibration_path)
        self.temperature = float(calibration["temperature"])
        self.threshold = float(calibration["abstain_threshold"])
        self.gate = load_settings(Path(calibration_path))

    def identify(self, image_bytes: bytes) -> Identification:
        try:
            image = self._decode(image_bytes)
        except (_TooLarge, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return Identification("invalid", guidance=TOO_LARGE)
        except Exception:
            return Identification("invalid", guidance=INVALID)
        raw = self.model.predict(image)
        probs = apply_temperature(np.log(np.maximum(raw, FLOOR)), self.temperature)
        if not self.gate.accepts(green=green_ratio(image), probs=probs):
            return Identification("not_plant", guidance=NOT_PLANT)
        decision = decide(probs, self.threshold)
        names = self.model.class_names
        found = [Candidate(names[i], p) for i, p in decision.top_k]
        if decision.status == "answered" and found:
            return Identification("answered", found[0], found[1:])
        return Identification("unsure", found[0] if found else None, found[1:], PHOTO_TIPS)

    @staticmethod
    def _decode(data: bytes) -> Image.Image:
        if len(data) > MAX_BYTES:
            raise _TooLarge
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            if image.width * image.height > MAX_PIXELS:
                raise _TooLarge
            image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def _require_same_checkpoint(model_sha: str | None, calibration_sha: str | None, model_path, calibration_path) -> None:
    for sha, what, path in ((model_sha, "model file", model_path), (calibration_sha, "calibration", calibration_path)):
        if not sha:
            raise ValueError(f"the {what} {path} carries no checkpoint checksum, so it cannot be checked against the "
                             f"{'calibration' if what == 'model file' else 'model file'}")
    if model_sha != calibration_sha:
        raise ValueError(f"the calibration {calibration_path} was fitted for another model: it names checkpoint "
                         f"{calibration_sha}, but {model_path} was exported from checkpoint {model_sha}")


class _TooLarge(Exception):
    pass
