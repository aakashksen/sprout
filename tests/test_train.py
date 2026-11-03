import dataclasses
import random

import pandas as pd
import pytest
import torch
from PIL import Image

import sprout.data.splits as splits
from sprout.data.splits import dataset_hash
from sprout.preprocess import preprocess
from sprout.train.checkpoint import load_checkpoint, save_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import (
    build_datasets,
    build_loaders,
    make_class_map,
    train_transform,
    val_transform,
)
from sprout.train.loop import build_model, train


def _noise_image(seed: int) -> Image.Image:
    g = torch.Generator().manual_seed(seed)
    data = (torch.rand(3, 64, 64, generator=g) * 255).to(torch.uint8)
    return Image.fromarray(data.permute(1, 2, 0).numpy())


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    rows = []
    for i in range(12):
        label = "red" if i % 2 == 0 else "blue"
        color = (200, 30, 30) if label == "red" else (30, 30, 200)
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 64), color).save(p)
        split = "lab_train" if i < 8 else "lab_dev"
        rows.append((str(p), label, f"g{i}", split, f"sha{i}"))
    df = pd.DataFrame(rows, columns=["path", "label", "group", "split", "sha1"])
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    df.to_csv(mdir / "tiny.csv", index=False)
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    return df


def _config(tmp_path, **kw) -> TrainConfig:
    base = dict(
        manifest="tiny",
        out_dir=str(tmp_path / "run"),
        model="tiny_cnn",
        seed=1,
        lr=0.05,
        batch_size=4,
        epochs=2,
    )
    base.update(kw)
    return TrainConfig(**base)


def test_train_transform_differs_from_val_transform():
    img = _noise_image(0)
    tfs = TrainConfig(manifest="m", out_dir="o").transforms
    random.seed(0)
    outs = [train_transform(img, tfs) for _ in range(4)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    assert torch.equal(val_transform(img), val_transform(img))
    assert torch.equal(val_transform(img), preprocess(img))


def test_val_dataset_has_no_random_augmentation(manifest, tmp_path):
    cfg = _config(tmp_path)
    train_ds, val_ds = build_datasets(manifest, cfg, make_class_map(manifest))
    assert len(train_ds) == 8 and len(val_ds) == 4
    assert val_ds.transform is val_transform
    a, _ = val_ds[0]
    b, _ = val_ds[0]
    assert torch.equal(a, b)


def test_resume_requires_explicit_path(manifest, tmp_path):
    cfg = _config(tmp_path, epochs=1)
    train(cfg)
    with pytest.raises(FileExistsError):
        train(cfg)  # checkpoints exist, but resume was not asked for
    with pytest.raises(FileNotFoundError):
        train(_config(tmp_path, out_dir=str(tmp_path / "other")), resume=tmp_path / "nope.pt")
    out = train(dataclasses.replace(cfg, epochs=2), resume=tmp_path / "run" / "latest.pt")
    assert out.epochs_done == 2 and len(out.history) == 1


def test_checkpoint_round_trip_includes_class_map_and_optimizer(tmp_path):
    cfg = TrainConfig(manifest="m", out_dir="o", model="tiny_cnn", manifest_hash="abc")
    model = build_model("tiny_cnn", 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.zeros(1, 3, 224, 224)).sum().backward()
    opt.step()
    p = tmp_path / "c.pt"
    save_checkpoint(
        p,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        epoch=3,
        best_metric=0.5,
        class_map={"a": 0, "b": 1},
        config=cfg,
    )
    ck = load_checkpoint(p)
    assert ck["class_map"] == {"a": 0, "b": 1}
    assert ck["epoch"] == 3 and ck["best_metric"] == 0.5
    assert ck["config"] == cfg
    assert ck["optimizer"]["state"]  # AdamW moments saved
    assert "scheduler" in ck and "scaler" in ck
    m2 = build_model("tiny_cnn", 2)
    m2.load_state_dict(ck["model"])
    for a, b in zip(model.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_best_and_latest_are_separate(manifest, tmp_path):
    out = train(_config(tmp_path))
    assert out.best_path != out.latest_path
    assert out.best_path.is_file() and out.latest_path.is_file()
    assert load_checkpoint(out.latest_path)["epoch"] == 2
    assert load_checkpoint(out.best_path)["best_metric"] == out.best_metric
    assert load_checkpoint(out.latest_path)["config"].manifest_hash == dataset_hash(manifest)


def test_two_step_run_on_tiny_model_reduces_loss_on_cpu(manifest, tmp_path):
    out = train(_config(tmp_path, transforms=()))
    assert out.history[1]["train_loss"] < out.history[0]["train_loss"]


def test_same_seed_same_split_and_first_batch(manifest, tmp_path):
    cfg = _config(tmp_path)
    cm = make_class_map(manifest)
    a_tr, a_val = build_loaders(manifest, cfg, cm)
    xa, ya = next(iter(a_tr))
    b_tr, b_val = build_loaders(manifest, cfg, cm)
    xb, yb = next(iter(b_tr))
    assert a_tr.dataset.paths == b_tr.dataset.paths
    assert a_val.dataset.paths == b_val.dataset.paths
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
