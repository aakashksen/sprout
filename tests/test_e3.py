import json
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import sprout.data.splits as splits
from sprout.baselines.classic import classic_features, fit_xgboost, xgboost_probs
from sprout.baselines.frozen import (
    cached_features,
    encode_batches,
    freeze,
    head_probs,
    train_head,
)
from sprout.eval.check_run import check
from sprout.experiments import e3_baselines as e3
from sprout.train.config import TrainConfig
from sprout.train.loop import build_model

XGB = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0,
       "tree_method": "hist", "device": "cpu", "n_jobs": 1}


def leaf_like(color, seed, size=96) -> Image.Image:
    """A coloured square with a seeded texture of dark flecks on a black ground."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), np.uint8)
    arr[16:-16, 16:-16] = color
    arr[rng.integers(16, size - 16, 30), rng.integers(16, size - 16, 30)] = 0
    return Image.fromarray(arr)


def test_features_are_stable():
    img = leaf_like((200, 30, 30), 1)
    a, b = classic_features(img), classic_features(img.copy())
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert np.isfinite(a).all()
    # size and mode of the input do not change the length of the vector
    assert classic_features(img.resize((300, 170))).shape == a.shape
    assert classic_features(img.convert("L")).shape == a.shape
    assert classic_features(Image.new("RGB", (40, 40), (9, 9, 9))).shape == a.shape
    # 7x7 blocks of 2x2 cells with 9 bins, then 3 channels of 32 bins
    assert a.shape == (7 * 7 * 4 * 9 + 3 * 32,)
    assert np.isclose(a[-96:].sum(), 3.0)
    assert not np.array_equal(a, classic_features(leaf_like((30, 30, 200), 1)))


def test_xgboost_learns_a_separable_fixture():
    colors = {0: (200, 30, 30), 1: (30, 200, 30), 2: (30, 30, 200)}
    def build(seeds):
        ys = [k for k in colors for _ in seeds]
        xs = np.stack([classic_features(leaf_like(colors[k], s)) for k in colors for s in seeds])
        return xs, np.array(ys)
    x, y = build(range(8))
    dev_x, dev_y = build(range(100, 104))
    test_x, test_y = build(range(200, 206))
    booster, history, seconds = fit_xgboost(x, y, dev_x, dev_y, 3, XGB, seed=0)
    probs = xgboost_probs(booster, test_x)
    assert probs.shape == (18, 3) and np.allclose(probs.sum(axis=1), 1)
    assert (probs.argmax(axis=1) == test_y).all()
    assert len(history["train_loss"]) == len(history["val_loss"]) == 20 and seconds > 0
    assert history["train_loss"][-1] < history["train_loss"][0]


def tiny_encoder() -> nn.Module:
    torch.manual_seed(0)
    return freeze(nn.Sequential(nn.Conv2d(3, 6, 5, stride=4), nn.BatchNorm2d(6), nn.ReLU(),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten()))


def test_frozen_backbone_has_no_grad(tmp_path):
    enc = tiny_encoder()
    assert not enc.training and not any(p.requires_grad for p in enc.parameters())
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    x = torch.randn(10, 3, 32, 32)
    (feats,) = list(encode_batches(enc, [x], torch.device("cpu")))
    assert isinstance(feats, np.ndarray) and feats.shape == (10, 6)
    assert not enc(x).requires_grad  # frozen parameters leave nothing to differentiate
    # training a head on its features leaves the backbone untouched, batch-norm statistics included
    y = (feats[:, 0] > np.median(feats[:, 0])).astype(np.int64)
    settings = {"lr": 0.1, "weight_decay": 0.0, "epochs": 2, "batch_size": 4}
    fit = train_head(feats, y, feats, y, 2, settings, tmp_path / "head", seed=0)
    assert fit.model.weight.shape == (2, 6) and len(fit.history) == 2
    assert head_probs(fit.model, feats).shape == (10, 2)
    for k, v in enc.state_dict().items():
        assert torch.equal(before[k], v)
    assert not any(p.requires_grad for p in enc.parameters())


def test_head_stops_on_a_non_finite_loss_and_keeps_only_best_and_latest(tmp_path):
    x = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    y = np.array([0, 1] * 6)
    settings = {"lr": 0.05, "weight_decay": 0.0, "epochs": 3, "batch_size": 4}
    train_head(x, y, x, y, 2, settings, tmp_path / "ok", seed=0)
    assert sorted(p.name for p in (tmp_path / "ok").iterdir()) == ["best.pt", "latest.pt"]
    bad = x.copy()
    bad[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="not finite"):
        train_head(bad, y, x, y, 2, settings, tmp_path / "bad", seed=0)


def test_a_head_checkpoint_write_that_fails_keeps_the_previous_one(tmp_path, monkeypatch):
    import sprout.baselines.frozen as frozen

    x = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    y = np.array([0, 1] * 6)
    settings = {"lr": 0.05, "weight_decay": 0.0, "epochs": 2, "batch_size": 4}
    real, calls = torch.save, []

    def dies_on_epoch_two(obj, f):
        calls.append(1)
        if len(calls) <= 2:  # epoch 1: latest.pt, then best.pt
            return real(obj, f)
        handle = open(f, "wb") if hasattr(f, "__fspath__") else f
        handle.write(b"half a checkpoint")
        handle.flush()
        raise OSError("disk full")

    monkeypatch.setattr(frozen.torch, "save", dies_on_epoch_two)
    with pytest.raises(OSError):
        train_head(x, y, x, y, 2, settings, tmp_path, seed=0)
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["epoch"] == 1


def test_head_resumes_and_refuses_changed_settings(tmp_path):
    x = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    y = np.array([0, 1] * 6)
    settings = {"lr": 0.05, "weight_decay": 0.0, "epochs": 2, "batch_size": 4}
    first = train_head(x, y, x, y, 2, settings, tmp_path, seed=0)
    again = train_head(x, y, x, y, 2, settings, tmp_path, seed=0)
    assert again.history == first.history  # finished work is read back, not redone
    with pytest.raises(ValueError, match="different settings"):
        train_head(x, y, x, y, 2, {**settings, "lr": 0.5}, tmp_path, seed=0)


def test_feature_cache_reuses_finished_shards(tmp_path):
    rows = pd.DataFrame({"path": [f"p{i}" for i in range(5)]})
    calls = []

    def make(piece):
        calls.append(list(piece["path"]))
        yield np.arange(len(piece), dtype=np.float32).reshape(-1, 1) + len(calls)

    one = cached_features(tmp_path, "t", "k", rows, make, shard_rows=2)
    assert one.x.shape == (5, 1) and len(calls) == 3 and len(one.seconds) == 3
    two = cached_features(tmp_path, "t", "k", rows, make, shard_rows=2)
    assert len(calls) == 3 and np.array_equal(one.x, two.x) and two.seconds == one.seconds
    cached_features(tmp_path, "t", "other settings", rows, make, shard_rows=2)
    assert len(calls) == 6  # a different key never reads old files


# ---- runner on a tiny fixture ------------------------------------------------------------------

A, B, C = "Apple___healthy", "Cherry___healthy", "Corn___only_in_field"
E1_ID = "E1-20260101T000000000000Z"
BEST = 0.75


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def save_image(path, color, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    leaf_like(color, seed, 64).save(path)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    mdir = root / "datasets" / "manifests"
    mdir.mkdir(parents=True)
    lab, field = [], []
    for i in range(24):
        label = [A, B][i % 2]
        color = (200, 30, 30) if i % 2 == 0 else (30, 200, 30)
        split = "lab_train" if i < 8 else "lab_dev" if i < 16 else "lab_test"
        p = root / "datasets" / "raw" / "lab" / label / f"{i}.png"
        if split != "lab_test":  # images that must not be read do not exist
            save_image(p, color, i)
        lab.append((str(p), label, f"g{i}", split, f"sha{i}"))
    for i in range(12):
        label = [A, B, C][i % 3]
        split = "field_train" if i < 3 else "field_dev" if i < 9 else "field_test"
        p = root / "datasets" / "raw" / "field" / f"f{i}.png"
        if split == "field_dev":
            save_image(p, (200, 30, 30) if label == A else (30, 200, 30), 50 + i)
        field.append((str(p), label, f"h{i}", split, f"fsha{i}", "plantdoc"))
    cols = ["path", "label", "group", "split", "sha1"]
    pd.DataFrame(lab, columns=cols).to_csv(mdir / "lab.csv", index=False)
    pd.DataFrame(field, columns=[*cols, "source"]).to_csv(mdir / "field.csv", index=False)
    torch.manual_seed(0)
    model = build_model("tiny_cnn", 2, pretrained=False)
    ckpt = root / "models" / "e1" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    train_cfg = TrainConfig(manifest="lab", out_dir="x", model="tiny_cnn", epochs=1)
    torch.save({"model": model.state_dict(), "class_map": {A: 0, B: 1}, "config": train_cfg.to_dict(),
                "best_metric": BEST, "epoch": 1}, ckpt)
    runs = root / "results" / "runs"
    runs.mkdir(parents=True)
    history = [{"epoch": 1, "seconds": 12.5}, {"epoch": 2, "seconds": 11.5}]
    (runs / f"{E1_ID}.json").write_text(json.dumps({"attempts": [{"best_dev_accuracy": BEST, "history": history}]}))
    (root / ".gitignore").write_text("models/\n")
    sh(root, "init", "-q")
    sh(root, "config", "user.email", "t@t")
    sh(root, "config", "user.name", "t")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    monkeypatch.setattr(e3, "load_convnext_encoder", lambda name: tiny_encoder())
    monkeypatch.setattr(e3, "load_dino_encoder", tiny_encoder)
    return root


@pytest.fixture
def cfg():
    head = {"lr": 0.05, "weight_decay": 0.0, "epochs": 3, "batch_size": 4}
    return {
        "experiment": "E3", "checkpoint": "models/e1/best.pt", "e1_run_id": E1_ID,
        "splits": ["field_dev", "lab_dev"], "cache_dir": "models/e3", "min_free_disk_gb": 0,
        "batch_size": 4, "num_workers": 0, "shard_rows": 3, "seed": 0,
        "classic": {"features": {"size": 64, "orientations": 9, "pixels_per_cell": 16, "cells_per_block": 2,
                                 "bins": 16}, "xgboost": XGB},
        "frozen_convnext": {"model": "convnext_tiny", "head": head},
        "dinov2_small": {"head": head},
    }


def records(paths):
    return {r["split"]: r for r in (json.loads(p.read_text()) for p in paths)}


def test_record_includes_all_four_systems(repo, cfg):
    recs = records(e3.run(cfg, repo))
    assert set(recs) == {"field_dev", "lab_dev"}
    for split, rec in recs.items():
        assert set(rec["models"]) == set(rec["predictions"]) == set(rec["metrics"]) == set(e3.SYSTEMS)
        assert rec["subset"] == "shared_classes" and rec["n"] == (4 if split == "field_dev" else 8)
        pairs = {(c["a"], c["b"], c["metric"]) for c in rec["comparisons"]}
        assert len(pairs) == 12 and ("e1", "xgboost", "macro_f1") in pairs
        values = rec["measurements"]["values"]
        for s in e3.SYSTEMS:
            assert values[f"train_seconds_{s}_p50"] > 0 and values[f"model_size_bytes_{s}_p50"] > 0
        assert values["train_seconds_e1_p50"] == 24.0
        assert rec["config"]["xgboost_device"] == "cpu"
        assert rec["config"]["dinov2"]["license"] == "Apache License 2.0" and rec["config"]["dinov2"]["gated"] is False
        assert [a["system"] for a in rec["attempts"]] == list(e3.SYSTEMS)
        assert "e1" not in rec["models"]["xgboost"]
    assert check("E3", root=repo, split="field_dev") == [] and check("E3", root=repo, split="lab_dev") == []
    log = (repo / "models" / "e3" / "stages.log").read_text()
    for name in ("classic_features", "classic_fit", "frozen_convnext_head", "dinov2_small_features", "e1_predict"):
        assert f"stage {name} done seconds=" in log


def test_only_train_and_dev_splits_are_read(repo, cfg):
    for split in ("lab_test", "field_test", "field_train"):
        with pytest.raises(ValueError, match="reads only"):
            e3.load_split(split)
        cfg["splits"] = [split]
        with pytest.raises(ValueError, match="scores only"):
            e3.run(cfg, repo)
    cfg["splits"] = ["field_dev", "lab_dev"]
    e3.run(cfg, repo)  # lab_test, field_test and field_train images do not exist, so a read would fail


def test_rerun_reuses_cache_and_trained_models(repo, cfg):
    e3.run(cfg, repo)
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "records")
    e3.run(cfg, repo)
    log = (repo / "models" / "e3" / "stages.log").read_text()
    assert "reused from cache" in log and "reused the finished XGBoost model" in log
    head_dir = repo / "models" / "e3" / "dinov2_small-head"
    assert sorted(p.name for p in head_dir.iterdir()) == ["best.pt", "latest.pt"]


def test_runner_refuses_dirty_tree_missing_checkpoint_and_low_disk(repo, cfg):
    (repo / "extra.txt").write_text("x")
    with pytest.raises(RuntimeError, match="uncommitted"):
        e3.run(cfg, repo)
    (repo / "extra.txt").unlink()
    with pytest.raises(RuntimeError, match="free"):
        e3.run({**cfg, "min_free_disk_gb": 10**9}, repo)
    (repo / "models" / "e1" / "best.pt").unlink()
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        e3.run(cfg, repo)


def test_runner_refuses_a_checkpoint_from_another_run(repo, cfg):
    runs = repo / "results" / "runs"
    (runs / f"{E1_ID}.json").write_text(json.dumps({"attempts": [{"best_dev_accuracy": 0.1, "history": []}]}))
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "other")
    with pytest.raises(RuntimeError, match="not produced by that run"):
        e3.run(cfg, repo)


@pytest.mark.real
def test_real_dino_encoder_is_frozen_and_gives_384_features():
    from sprout.baselines.dino import load_dino_encoder

    enc = load_dino_encoder()
    assert not any(p.requires_grad for p in enc.parameters())
    with torch.no_grad():
        assert enc(torch.randn(2, 3, 224, 224)).shape == (2, 384)
