"""ONNX export, quantization and the torch-free runtime (E6, T5.2).

`OnnxModel` and `quantize` import no torch, so the serving image does not need it. Only `export_module` and
`export_onnx` do, and they import it inside the function. The class map and the preprocessing settings are
written into the file's metadata, so a model file says which class each output column is. So is the sha256 of the
checkpoint the file came from (`SOURCE_KEY`), which the identifier compares with the one the calibration was fitted for.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from sprout.preprocess_np import IMAGE_SIZE, MEAN, STD, preprocess_array

INPUT, OUTPUT = "image", "logits"
OPSET = 17
MODES = ("dynamic", "static")
SOURCE_KEY = "source_checkpoint_sha256"


def _set_metadata(path: Path, props: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in props.items():
        model.metadata_props.add(key=key, value=value)
    onnx.save(model, str(path))


def _metadata(path: Path) -> dict[str, str]:
    import onnx

    return {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}


def export_module(model, class_map: dict[str, int], path: Path, model_name: str = "", source_sha256: str = "") -> Path:
    """Write a torch module that takes (batch, 3, 224, 224) and returns logits; the batch size stays open.
    `source_sha256` is stamped into the metadata when given."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(path), input_names=[INPUT], output_names=[OUTPUT], opset_version=OPSET,
            dynamic_axes={INPUT: {0: "batch"}, OUTPUT: {0: "batch"}}, dynamo=False,
        )
    _set_metadata(path, {
        "class_map": json.dumps(class_map),
        "preprocess": json.dumps({"size": IMAGE_SIZE, "mean": MEAN, "std": STD}),
        "model_name": model_name,
        **({SOURCE_KEY: source_sha256} if source_sha256 else {}),
    })
    return path


def export_onnx(checkpoint: Path, path: Path) -> Path:
    """Export a training checkpoint (any model `build_model` knows) with its class map."""
    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import build_model

    ckpt = load_checkpoint(Path(checkpoint))
    name = ckpt["config"].model
    model = build_model(name, len(ckpt["class_map"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return export_module(model, ckpt["class_map"], path, name, hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())


class _Calibration:
    """Feeds the calibration photos to the quantizer one at a time."""

    def __init__(self, images: Iterable[Image.Image | str | Path]):
        self._arrays = iter(self._load(images))

    @staticmethod
    def _load(images):
        for item in images:
            if isinstance(item, Image.Image):
                yield {INPUT: preprocess_array(item)[None]}
            else:
                with Image.open(item) as img:
                    yield {INPUT: preprocess_array(img)[None]}

    def get_next(self):
        return next(self._arrays, None)


def quantize(path: Path, mode: str, calibration_images=None) -> Path:
    """Write `<name>.dynamic.onnx` or `<name>.static.onnx` beside the model and return its path.
    Static mode needs calibration images (PIL images or file paths)."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    path = Path(path)
    out = path.with_name(f"{path.stem}.{mode}.onnx")
    if mode == "dynamic":
        quantize_dynamic(str(path), str(out), weight_type=QuantType.QInt8)
    else:
        images = list(calibration_images or [])
        if not images:
            raise ValueError("static quantization needs calibration images")
        reader = type("Reader", (_Calibration, CalibrationDataReader), {})(images)
        quantize_static(
            str(path), str(out), reader, quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        )
    _set_metadata(out, _metadata(path))  # the quantizer drops metadata; put it back, source checksum included
    return out


class OnnxModel:
    """A model file ready to answer: `predict(image)` returns the class probabilities in class-map order."""

    def __init__(self, path: Path, threads: int = 0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 lets the runtime choose
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if "class_map" not in meta:
            raise ValueError(f"{path} has no class map in its metadata")
        self.class_map: dict[str, int] = json.loads(meta["class_map"])
        self.source_sha256: str | None = meta.get(SOURCE_KEY) or None
        self.class_names = [n for n, _ in sorted(self.class_map.items(), key=lambda kv: kv[1])]

    def predict(self, image: Image.Image) -> np.ndarray:
        logits = self.session.run([OUTPUT], {INPUT: preprocess_array(image)[None]})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        return e / e.sum()
