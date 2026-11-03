import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from sprout.baselines.classic import classic_features, fit_xgboost, xgboost_probs
from sprout.baselines.frozen import (
    cached_features,
    encode_batches,
    freeze,
    head_probs,
    train_head,
)

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
