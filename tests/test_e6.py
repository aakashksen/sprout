import numpy as np
import pytest
import torch
from PIL import Image

from sprout.preprocess import preprocess
from sprout.preprocess_np import preprocess_array


def _photo(w=97, h=61, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_numpy_preprocess_is_identical_to_the_torch_one(mode):
    image = _photo().convert(mode)
    got = preprocess_array(image)
    assert got.dtype == np.float32 and got.shape == (3, 224, 224)
    assert torch.equal(preprocess(image), torch.from_numpy(got))
    assert torch.equal(preprocess(image, 64), torch.from_numpy(preprocess_array(image, 64)))


# ---------------------------------------------------------------------------------------------------------------
# export, run and quantize

from torch import nn  # noqa: E402

from sprout.serve.export import OnnxModel, export_module, export_onnx, quantize  # noqa: E402
from sprout.train.config import TrainConfig  # noqa: E402
from sprout.train.loop import build_model  # noqa: E402

CLASS_MAP = {"Apple___healthy": 0, "Apple___scab": 1, "Tomato___blight": 2}


def _checkpoint(tmp_path, model_name="tiny_cnn"):
    torch.manual_seed(0)
    model = build_model(model_name, len(CLASS_MAP), pretrained=False)
    config = TrainConfig(manifest="m.csv", out_dir="o", model=model_name)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "class_map": CLASS_MAP, "config": config.to_dict()}, path)
    return path, model.eval()


def _mlp():
    torch.manual_seed(1)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(32), nn.Flatten(), nn.Linear(3 * 32 * 32, 256), nn.ReLU(), nn.Linear(256, len(CLASS_MAP))
    ).eval()


def test_onnx_matches_torch_within_tolerance(tmp_path):
    ckpt, model = _checkpoint(tmp_path)
    onnx_path = export_onnx(ckpt, tmp_path / "m.onnx")
    served = OnnxModel(onnx_path)
    for seed in range(4):
        image = _photo(seed=seed)
        with torch.no_grad():
            want = torch.softmax(model(preprocess(image)[None]), dim=1)[0].numpy()
        got = served.predict(image)
        assert got.shape == (len(CLASS_MAP),) and got.sum() == pytest.approx(1.0, abs=1e-5)
        np.testing.assert_allclose(got, want, atol=1e-5)


def test_quantized_runs_and_is_smaller(tmp_path):
    onnx_path = export_module(_mlp(), CLASS_MAP, tmp_path / "mlp.onnx")
    calibration = [_photo(seed=s) for s in range(6)]
    base = onnx_path.stat().st_size
    for mode in ("dynamic", "static"):
        out = quantize(onnx_path, mode, calibration)
        assert out != onnx_path and out.is_file()
        assert out.stat().st_size < base * 0.6, (mode, out.stat().st_size, base)
        probs = OnnxModel(out).predict(_photo(seed=9))
        assert probs.shape == (len(CLASS_MAP),) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_static_quantization_needs_calibration_images(tmp_path):
    onnx_path = export_module(_mlp(), CLASS_MAP, tmp_path / "mlp.onnx")
    with pytest.raises(ValueError, match="calibration"):
        quantize(onnx_path, "static")
    with pytest.raises(ValueError, match="mode"):
        quantize(onnx_path, "int4")


def test_class_map_travels_with_the_model(tmp_path):
    ckpt, _ = _checkpoint(tmp_path)
    onnx_path = export_onnx(ckpt, tmp_path / "m.onnx")
    assert OnnxModel(onnx_path).class_map == CLASS_MAP
    assert OnnxModel(onnx_path).class_names == list(CLASS_MAP)
    for mode in ("dynamic", "static"):
        assert OnnxModel(quantize(onnx_path, mode, [_photo()])).class_map == CLASS_MAP


def test_running_and_quantizing_a_model_loads_no_torch(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    onnx_path = export_module(_mlp(), CLASS_MAP, tmp_path / "mlp.onnx")
    code = (
        "import sys\nfrom PIL import Image\n"
        "from sprout.serve.export import OnnxModel, quantize\n"
        f"p = quantize({str(onnx_path)!r}, 'dynamic')\n"
        "m = OnnxModel(p); m.predict(Image.new('RGB', (40, 30), (10, 200, 10)))\n"
        "sys.exit('torch loaded' if 'torch' in sys.modules else 0)\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(src)})
    assert done.returncode == 0, done.stderr[-500:]
