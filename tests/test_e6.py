from pathlib import Path

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


def test_every_variant_carries_the_checkpoint_checksum(tmp_path):
    import hashlib

    ckpt, _ = _checkpoint(tmp_path)
    want = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    base = export_onnx(ckpt, tmp_path / "m.onnx")
    assert OnnxModel(base).source_sha256 == want
    for mode in ("dynamic", "static"):
        assert OnnxModel(quantize(base, mode, [_photo()])).source_sha256 == want
    assert OnnxModel(export_module(_mlp(), CLASS_MAP, tmp_path / "bare.onnx")).source_sha256 is None


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


# ---------------------------------------------------------------------------------------------------------------
# the experiment, on a tiny world: tiny_cnn, a handful of generated photos, a throwaway git repo

import json  # noqa: E402
import subprocess  # noqa: E402

from sprout.eval.check_run import check  # noqa: E402
from sprout.experiments import e6_onnx as e6  # noqa: E402

NAMES = ["A", "B"]
FIELD_ROWS = [(f"f/dev{i}.jpg", NAMES[i % 2], f"g{i}", "field_dev") for i in range(8)]
FIELD_ROWS += [("f/test0.jpg", "A", "gt0", "field_test"), ("f/train0.jpg", "B", "gtr0", "field_train"),
               ("f/train1.jpg", "A", "gtr1", "field_train")]
LAB_ROWS = [(f"l/dev{i}.jpg", NAMES[i % 2], f"lg{i}", "lab_dev") for i in range(3)]


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _commit(root):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "stage")


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "w"
    for rel, rows, extra in (("field", FIELD_ROWS, ",s,plantdoc"), ("lab", LAB_ROWS, ",s")):
        lines = ["path,label,group,split,sha1" + (",source" if extra.endswith("plantdoc") else "")]
        lines += [f"{p},{label},{g},{s}{extra}" for p, label, g, s in rows]
        (root / "datasets/manifests").mkdir(parents=True, exist_ok=True)
        (root / f"datasets/manifests/{rel}.csv").write_text("\n".join(lines) + "\n")
        for k, (p, *_rest) in enumerate(rows):
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            _photo(60 + k, 40 + 3 * k, seed=k).save(root / p)
    (root / "datasets/label_exclusions.csv").write_text(
        "version,image_id,path,source,split,label,reason_code,note,began_as\n"
        "1,s,f/dev1.jpg,plantdoc,field_dev,B,wrong_plant,x,remove\n")
    (root / "orchestration").mkdir()
    (root / "orchestration/tasks.json").write_text("{}")
    (root / "models/e4").mkdir(parents=True)
    torch.manual_seed(0)
    model = build_model("tiny_cnn", 2, pretrained=False)
    config = TrainConfig(manifest="m.csv", out_dir="o", model="tiny_cnn")
    torch.save({"model": model.state_dict(), "class_map": {"A": 0, "B": 1}, "config": config.to_dict()},
               root / "models/e4/best.pt")
    (root / "results/e5").mkdir(parents=True)
    (root / "results/e5/chosen.json").write_text(json.dumps({"temperature": 1.5, "abstain_threshold": 0.6}))
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _commit(root)
    return root


@pytest.fixture
def cfg():
    return {
        "experiment": "E6", "split": "field_dev", "subset": "clean:1", "out_dir": "results/e6",
        "model_dir": "models/e6", "model": {"name": "tiny", "checkpoint": "models/e4/best.pt"},
        "crop_kept": False, "torch_device": "cpu", "threads": 1, "batch_size": 3, "chunk": 2,
        "quantize": ["dynamic"], "scoring_part": {"seed": 0, "fit_percent": 50},
        "calibration": {"split": "lab_dev", "n": 3, "seed": 0}, "thresholds": "results/e5/chosen.json",
        "latency": {"images": 3, "warmup": 1},
    }


def _run_stages(cfg, root, stages=("export", "probs", "latency")):
    for stage in stages:
        e6.run_stage(stage, cfg, root)
        _commit(root)


def _rows(root, system):
    return [json.loads(x) for x in (root / f"results/e6/probs.{system}.jsonl").read_text().splitlines()]


def test_stages_end_to_end_and_the_record_passes_check_run(world, cfg):
    _run_stages(cfg, world)
    record = e6.run_stage("record", cfg, world)
    rec = json.loads(record.read_text())
    assert rec["experiment"] == "E6" and rec["split"] == "field_dev" and rec["subset"] == "clean:1"
    assert set(rec["predictions"]) == {"pytorch", "onnx_dynamic"}  # the unquantized ONNX file is not scored
    assert rec["n"] == 7  # eight field_dev photos, one excluded; field_test and field_train never read
    assert {r["path"] for r in _rows(world, "pytorch")} == {f"f/dev{i}.jpg" for i in range(8)} - {"f/dev1.jpg"}
    assert 0.0 <= rec["metrics"]["onnx_dynamic"]["changed_vs@pytorch"] <= 1.0
    assert "changed_vs@pytorch" not in rec["metrics"]["pytorch"]
    c = rec["config"]
    assert set(c["probs_device"].values()) == {"cpu"} and c["latency"]["threads"] == 1 and c["latency"]["cpu"]
    assert set(c["latency"]["cpu_load_percent_before"]) == {"pytorch", "onnx_dynamic"}
    assert c["segmentation_model"].startswith("not exported") and "crop" in c["segmentation_model"]
    assert c["calibration"] is None  # dynamic quantization uses no calibration photos
    assert {"latency_ms_onnx_dynamic_p50", "latency_ms_onnx_dynamic_p95"} <= set(rec["measurements"]["values"])
    assert {f"size_bytes_{s}_p50" for s in rec["predictions"]} <= set(rec["measurements"]["values"])
    assert check("E6", root=world, min_n=7, split="field_dev") == []


def test_every_exported_file_carries_the_checkpoint_checksum(world, cfg):
    _run_stages(cfg, world, ("export",))
    state = json.loads((world / "results/e6/export.json").read_text())
    for system, info in state["files"].items():
        assert OnnxModel(world / "models/e6" / info["file"]).source_sha256 == state["checkpoint_sha256"], system


def test_record_states_how_the_fitted_thresholds_behave_on_every_system(world, cfg):
    from sprout.experiments.e5_confidence import calibrate, scoring_part

    _run_stages(cfg, world)
    rows = _rows(world, "pytorch")
    top = calibrate(np.array([r["probs"] for r in rows]), 1.5).max(axis=1)
    abstain = float(np.sort(top)[3])  # answers some photos and abstains on others
    (world / "results/e5/chosen.json").write_text(json.dumps(
        {"temperature": 1.5, "abstain_threshold": abstain, "healthy_threshold": 1.0, "healthy_suffix": "A"}))
    _commit(world)
    rec = json.loads(e6.run_stage("record", cfg, world).read_text())
    t = rec["config"]["thresholds"]
    assert t["file"] == "results/e5/chosen.json" and t["temperature"] == 1.5 and t["abstain_threshold"] == abstain
    assert t["model_sha256"] is None and t["names_this_checkpoint"] is None  # this file names no model
    assert set(t["systems"]) == {"pytorch", "onnx_dynamic"}
    cal = calibrate(np.array([r["probs"] for r in rows]), 1.5)
    y = np.array([r["label"] for r in rows])
    # class "A" (column 0) counts as healthy here, and its bar is 1.0: it is never answered
    ok = (cal.max(axis=1) >= abstain) & (cal.argmax(axis=1) != 0)
    got = t["systems"]["pytorch"]["all_rows"]
    assert 0 < ok.sum() < len(rows)
    assert got["n"] == 7 and got["answered"] == ok.sum() and got["unsure"] == len(rows) - ok.sum()
    assert got["coverage"] == pytest.approx(ok.mean())
    assert got["confident_wrong"] == (ok & (cal.argmax(axis=1) != y)).sum()
    assert got["healthy_answers"] == 0 and got["sick_called_healthy"] == 0
    names = [n for n, _ in sorted({"A": 0, "B": 1}.items(), key=lambda kv: kv[1])]
    groups = [g for _p, _l, g, s_ in FIELD_ROWS if s_ == "field_dev" and _p != "f/dev1.jpg"]
    part = scoring_part(groups, 0, 50)
    assert t["systems"]["pytorch"]["scoring_part"]["n"] == part.sum() and names
    assert "Thresholds fitted for the PyTorch model" in rec["notes"]


def test_fitted_classes_may_be_listed_in_another_order_than_the_model_s(world, cfg):
    _run_stages(cfg, world)
    counts = []
    for classes in (["A", "B"], ["B", "A"]):
        (world / "results/e5/chosen.json").write_text(json.dumps(
            {"temperature": 1.0, "abstain_threshold": 0.5, "classes": classes}))
        _commit(world)
        rec = json.loads(e6.run_stage("record", cfg, world).read_text())
        counts.append(rec["config"]["thresholds"]["systems"]["pytorch"]["all_rows"])
        _commit(world)
    assert counts[0] == counts[1] and counts[0]["n"] == 7


def test_record_says_whether_the_thresholds_name_this_checkpoint(world, cfg):
    import hashlib

    sha = hashlib.sha256((world / "models/e4/best.pt").read_bytes()).hexdigest()
    for named, want in ((sha, True), ("0" * 64, False)):
        (world / "results/e5/chosen.json").write_text(
            json.dumps({"temperature": 1.5, "abstain_threshold": 0.6, "model_sha256": named}))
        _commit(world)
        _run_stages(cfg, world)
        rec = json.loads(e6.run_stage("record", cfg, world).read_text())
        assert rec["config"]["thresholds"]["names_this_checkpoint"] is want
        _commit(world)


def test_record_refuses_without_latency(world, cfg):
    _run_stages(cfg, world, ("export", "probs"))
    with pytest.raises(RuntimeError, match="latency"):
        e6.run_stage("record", cfg, world)


def test_probs_stage_resumes_and_does_not_redo_finished_rows(world, cfg, monkeypatch):
    _run_stages(cfg, world, ("export", "probs"))
    path = world / "results/e6/probs.onnx_dynamic.jsonl"
    full = path.read_text()
    path.write_text("".join(full.splitlines(keepends=True)[:3]) + '{"path": "f/dev')  # crash mid-line
    calls = []
    real = OnnxModel.predict
    monkeypatch.setattr(OnnxModel, "predict", lambda self, img: calls.append(1) or real(self, img))
    e6.run_stage("probs", cfg, world)
    assert len(calls) == 4 and path.read_text() == full


def test_new_checkpoint_discards_old_outputs(world, cfg):
    _run_stages(cfg, world, ("export", "probs"))
    before = _rows(world, "onnx_dynamic")
    torch.manual_seed(5)
    model = build_model("tiny_cnn", 2, pretrained=False)
    ckpt = TrainConfig(manifest="m.csv", out_dir="o", model="tiny_cnn")
    torch.save({"model": model.state_dict(), "class_map": {"A": 0, "B": 1}, "config": ckpt.to_dict()},
               world / "models/e4/best.pt")
    _run_stages(cfg, world, ("export", "probs"))
    after = _rows(world, "onnx_dynamic")
    assert [r["path"] for r in after] == [r["path"] for r in before] and after != before


def test_calibration_never_comes_from_field_dev_or_test(world, cfg, tmp_path):
    for split in ("field_dev", "field_test", "lab_test"):
        with pytest.raises(ValueError, match="calibration"):
            e6.check_config({**cfg, "calibration": {"split": split, "n": 3, "seed": 0}})
    paths = e6.calibration_paths({**cfg, "calibration": {"split": "field_train", "n": 5, "seed": 0}}, world)
    assert sorted(paths) == ["f/train0.jpg", "f/train1.jpg"]  # fewer rows than asked: all of them


def test_crop_kept_is_read_from_config(cfg):
    with pytest.raises(ValueError, match="segmentation"):
        e6.check_config({**cfg, "crop_kept": True})
    with pytest.raises(ValueError, match="field_dev"):
        e6.check_config({**cfg, "split": "field_test"})


def test_real_config_is_valid_and_names_the_model_path():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "e6.json").read_text())
    e6.check_config(cfg)
    assert cfg["model"]["checkpoint"].endswith(".pt")


def test_static_mode_is_built_scored_and_timed_when_the_config_asks_for_it(world, cfg):
    both = {**cfg, "quantize": ["dynamic", "static"], "calibration": {"split": "field_train", "n": 3, "seed": 0}}
    _run_stages(both, world)
    rec = json.loads(e6.run_stage("record", both, world).read_text())
    assert set(rec["predictions"]) == {"pytorch", "onnx_dynamic", "onnx_static"}
    assert rec["config"]["calibration"]["split"] == "field_train" and len(rec["config"]["calibration"]["paths"]) == 2
    with pytest.raises(ValueError, match="quantize"):
        e6.check_config({**cfg, "quantize": ["half"]})


def test_pytorch_size_is_the_weights_alone_and_the_checkpoint_keeps_its_own_name(world, cfg, tmp_path):
    path = world / "models/e4/best.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["optimizer"] = {"state": torch.zeros(20000)}  # training state a serving file does not need
    torch.save(ckpt, path)
    _commit(world)
    weights_file = tmp_path / "weights.pt"
    torch.save(ckpt["model"], weights_file)
    _run_stages(cfg, world)
    values = json.loads(e6.run_stage("record", cfg, world).read_text())["measurements"]["values"]
    assert values["size_bytes_pytorch_p50"] == weights_file.stat().st_size
    assert values["size_bytes_pytorch_checkpoint_p50"] == path.stat().st_size
    assert values["size_bytes_pytorch_p50"] < values["size_bytes_pytorch_checkpoint_p50"] - 80000
