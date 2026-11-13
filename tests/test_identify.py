import io
import json
import struct
import zlib

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from sprout.experiments.e5_confidence import FLOOR as E5_FLOOR
from sprout.preprocess import preprocess
from sprout.serve import identify as identify_module
from sprout.serve.export import export_module
from sprout.serve.identify import Identifier
from sprout.serve.types import Identification

CLASS_MAP = {"Apple___healthy": 0, "Apple___scab": 1, "Tomato___blight": 2}
ACCEPT_ALL = {"kind": "confidence", "confidence_threshold": 0.0}
CHECKPOINT_SHA = "c" * 64


def _model_path(tmp_path, source_sha256=CHECKPOINT_SHA, class_map=CLASS_MAP, bias=None):
    torch.manual_seed(1)
    net = nn.Sequential(
        nn.AdaptiveAvgPool2d(32), nn.Flatten(), nn.Linear(3 * 32 * 32, len(class_map)), nn.Identity()
    ).eval()
    with torch.no_grad():
        net[2].weight.mul_(40)  # spread the logits so photos get different answers
        if bias:
            net[2].bias.copy_(torch.tensor(bias))
    return export_module(net, class_map, tmp_path / "m.onnx", source_sha256=source_sha256)


def _calibration(tmp_path, abstain=0.0, gate=ACCEPT_ALL, temperature=1.0, model_sha256=CHECKPOINT_SHA,
                 classes=tuple(CLASS_MAP), healthy=None):
    path = tmp_path / "chosen.json"
    body = {"temperature": temperature, "abstain_threshold": abstain,
            "gate": {"chosen": gate["kind"], "settings": gate}}
    if classes is not None:
        body["classes"] = list(classes)
    if healthy is not None:
        body["healthy_threshold"] = healthy
    if model_sha256 is not None:
        body["model_sha256"] = model_sha256
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _identifier(tmp_path, **kw):
    return Identifier(_model_path(tmp_path), _calibration(tmp_path, **kw))


def _photo(w=120, h=80):
    x = np.linspace(0, 255, w)[None, :, None] * np.ones((h, 1, 3))
    x[: h // 2, :, 1] = 40  # top half is different from the bottom, so a rotation changes the picture
    x[:, : w // 3, 0] = 220
    return Image.fromarray(x.astype(np.uint8))


def _png(image, exif_orientation=None):
    buf = io.BytesIO()
    exif = None
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
    image.save(buf, "PNG", exif=exif)
    return buf.getvalue()


def test_exif_rotation_gives_same_result_as_upright(tmp_path):
    ident = _identifier(tmp_path)
    upright = _photo()
    # orientation 6 means "turn 90 degrees clockwise to display", so the stored pixels are the picture turned
    # 90 degrees counter-clockwise
    stored = upright.rotate(90, expand=True)
    plain, tagged = ident.identify(_png(upright)), ident.identify(_png(stored, exif_orientation=6))
    assert plain.status == "answered"
    assert tagged == plain
    assert ident.identify(_png(stored)) != plain  # without the tag the photo really does look different


def test_preprocess_matches_evaluation_exactly(tmp_path):
    ident = _identifier(tmp_path)
    data = _png(_photo(97, 61))
    seen = []
    run = ident.model.session.run
    ident.model.session.run = lambda names, feed: (seen.append(feed["image"]), run(names, feed))[1]
    ident.identify(data)
    expected = preprocess(Image.open(io.BytesIO(data))).numpy()[None]
    assert seen[0].shape == expected.shape and seen[0].dtype == np.float32
    assert np.allclose(seen[0], expected, rtol=0, atol=1e-6)
    assert E5_FLOOR == identify_module.FLOOR


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"0" * 40, _png(_photo())[:60]])
def test_invalid_bytes_returns_invalid(tmp_path, data):
    result = _identifier(tmp_path).identify(data)
    assert isinstance(result, Identification)
    assert result.status == "invalid" and result.top is None and result.guidance


def _png_claiming(width, height):
    """A small valid PNG whose header says it is width x height, with the checksum fixed."""
    data = bytearray(_png(Image.new("RGB", (4, 4))))
    ihdr = struct.pack(">II", width, height) + bytes(data[24:29])
    data[16:29] = ihdr
    data[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return bytes(data)


def test_oversize_image_is_rejected_before_decode_fully(tmp_path, monkeypatch):
    ident = _identifier(tmp_path)
    huge, decoded = _png_claiming(60000, 60000), []
    over_cap = _png_claiming(9000, 9000)  # under the PIL bomb limit, over ours
    monkeypatch.setattr(Image.Image, "load", lambda self: decoded.append(1))
    result = ident.identify(huge)
    assert result.status == "invalid" and "large" in result.guidance
    assert ident.identify(over_cap).status == "invalid"
    assert not decoded
    big_file = b"\0" * (identify_module.MAX_BYTES + 1)
    assert ident.identify(big_file).status == "invalid"


def test_unsure_includes_specific_photo_guidance(tmp_path):
    result = _identifier(tmp_path, abstain=1.01).identify(_png(_photo()))
    assert result.status == "unsure"
    assert result.top is not None
    for advice in ("close", "light", "leaf", "background"):
        assert advice in result.guidance


def test_flat_distribution_is_unsure_without_candidates(tmp_path):
    twenty = {f"c{i}": i for i in range(20)}
    ident = Identifier(_model_path(tmp_path, class_map=twenty), _calibration(tmp_path, abstain=0.5, classes=list(twenty)))
    ident.model.predict = lambda image: np.full(20, 0.05)
    result = ident.identify(_png(_photo()))
    assert result.status == "unsure" and result.top is None and result.alternatives == []
    assert Identification.from_json(result.to_json()) == result


def test_gate_rejection_is_not_plant_and_answer_carries_alternatives(tmp_path):
    reject = {"kind": "confidence", "confidence_threshold": 1.01}
    assert _identifier(tmp_path, gate=reject).identify(_png(_photo())).status == "not_plant"
    result = _identifier(tmp_path).identify(_png(_photo()))
    assert result.status == "answered" and result.top.label in CLASS_MAP
    assert all(a.confidence <= result.top.confidence for a in result.alternatives)


def test_green_rule_gate_reads_the_photo(tmp_path):
    green_rule = {"kind": "green_rule", "green_threshold": 0.05}
    ident = _identifier(tmp_path, gate=green_rule)
    assert ident.identify(_png(Image.new("RGB", (64, 64), (200, 30, 30)))).status == "not_plant"
    assert ident.identify(_png(Image.new("RGB", (64, 64), (30, 200, 30)))).status == "answered"


def test_a_calibration_fitted_for_another_model_is_refused_naming_both_checksums(tmp_path):
    with pytest.raises(ValueError, match=r"d{64}") as err:
        Identifier(_model_path(tmp_path), _calibration(tmp_path, model_sha256="d" * 64))
    assert CHECKPOINT_SHA in str(err.value) and "d" * 64 in str(err.value)


@pytest.mark.parametrize("model_sha,calibration_sha,missing", [
    (None, CHECKPOINT_SHA, "model file"), (CHECKPOINT_SHA, None, "calibration"), (None, None, "model file")])
def test_a_missing_checksum_on_either_side_is_refused(tmp_path, model_sha, calibration_sha, missing):
    model = _model_path(tmp_path, source_sha256=model_sha or "")
    with pytest.raises(ValueError, match="checksum") as err:
        Identifier(model, _calibration(tmp_path, model_sha256=calibration_sha))
    assert missing in str(err.value)


@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_a_quantized_file_is_bound_to_the_same_checkpoint(tmp_path, mode):
    from sprout.serve.export import quantize

    quantized = quantize(_model_path(tmp_path), mode, [_photo()])
    assert Identifier(quantized, _calibration(tmp_path)).identify(_png(_photo())).status == "answered"
    with pytest.raises(ValueError, match="another model"):
        Identifier(quantized, _calibration(tmp_path, model_sha256="e" * 64))


WIDE_MAP = {**CLASS_MAP, "Grape___esca": 3}  # a model that still has an output for a class that was dropped


def test_a_class_outside_the_list_is_never_named(tmp_path):
    model = _model_path(tmp_path, class_map=WIDE_MAP, bias=[0, 0, 0, 30])  # the dropped class wins every row
    ident = Identifier(model, _calibration(tmp_path))
    result = ident.identify(_png(_photo()))
    named = {result.top.label, *(a.label for a in result.alternatives)} if result.top else set()
    assert named <= set(CLASS_MAP) and "Grape___esca" not in named
    assert result.status == "answered"  # its probability went to the listed classes, renormalized


def test_dropped_outputs_are_masked_then_renormalized(tmp_path):
    ident = Identifier(_model_path(tmp_path, class_map=WIDE_MAP), _calibration(tmp_path))
    ident.model.predict = lambda image: np.array([0.2, 0.2, 0.1, 0.5])
    result = ident.identify(_png(_photo()))
    assert result.top.label == "Apple___healthy" and result.top.confidence == pytest.approx(0.4)
    assert [a.label for a in result.alternatives] == ["Apple___scab", "Tomato___blight"]
    assert result.alternatives[1].confidence == pytest.approx(0.2) and result.alternatives[0].confidence == pytest.approx(0.4)


def test_a_model_missing_a_listed_class_is_refused(tmp_path):
    cal = _calibration(tmp_path, classes=[*CLASS_MAP, "Rose___rust"])
    with pytest.raises(ValueError, match="Rose___rust"):
        Identifier(_model_path(tmp_path), cal)


def test_a_calibration_without_a_class_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="classes"):
        Identifier(_model_path(tmp_path), _calibration(tmp_path, classes=None))


def test_healthy_threshold_is_stricter(tmp_path):
    ident = Identifier(_model_path(tmp_path), _calibration(tmp_path, abstain=0.5, healthy=0.9))
    ident.model.predict = lambda image: np.array([0.8, 0.1, 0.1])
    assert ident.identify(_png(_photo())).status == "unsure"  # a healthy call at 0.8 is below its own bar
    ident.model.predict = lambda image: np.array([0.1, 0.8, 0.1])
    assert ident.identify(_png(_photo())).status == "answered"  # the same confidence for a disease answers
    ident.model.predict = lambda image: np.array([0.95, 0.03, 0.02])
    assert ident.identify(_png(_photo())).status == "answered"
    # a healthy threshold below the abstention threshold never loosens it
    loose = Identifier(_model_path(tmp_path), _calibration(tmp_path, abstain=0.85, healthy=0.5))
    loose.model.predict = lambda image: np.array([0.8, 0.1, 0.1])
    assert loose.identify(_png(_photo())).status == "unsure"


def test_no_probability_left_after_masking_is_unsure(tmp_path):
    ident = Identifier(_model_path(tmp_path, class_map=WIDE_MAP), _calibration(tmp_path))
    ident.model.predict = lambda image: np.array([0.0, 0.0, 0.0, 1.0])
    result = ident.identify(_png(_photo()))
    assert result.status == "unsure" and result.top is None and result.guidance
