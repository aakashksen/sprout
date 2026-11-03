import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import sprout.data.splits as splits
from sprout.train.augment import paste_background
from sprout.train.config import TrainConfig
from sprout.train.dataset import build_datasets, make_class_map
from sprout.train.joint import (
    JointDataset,
    MaskStore,
    cap_per_class,
    load_joint_manifest,
    select_rows,
)

CLASSES = ["Apple___healthy", "Cherry_(including_sour)___healthy"]
COLUMNS = ["path", "label", "group", "split", "sha1"]


def _save(path, color, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """lab.csv (12 train, 2 dev, 2 test per class) and field.csv (two sources). Test images do not exist."""
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    lab, field = [], []
    for c, label in enumerate(CLASSES):
        color = (200, 30, 30) if c == 0 else (30, 30, 200)
        for split, n in (("lab_train", 12), ("lab_dev", 2), ("lab_test", 2)):
            for i in range(n):
                p = tmp_path / "lab" / label / f"{split}{i}.png"
                if split != "lab_test":
                    _save(p, color)
                lab.append((str(p), label, f"l{c}{split}{i}", split, f"s{c}{split}{i}"))
        for source in ("plantdoc", "plantseg"):
            for split, n in (("field_train", 3), ("field_dev", 2), ("field_test", 2)):
                for i in range(n):
                    p = tmp_path / source / label / f"{split}{i}.png"
                    if split != "field_test":
                        _save(p, color)
                    field.append((str(p), label, f"f{c}{source}{split}{i}", split, f"f{c}{source}{split}{i}", source))
    pd.DataFrame(lab, columns=COLUMNS).to_csv(mdir / "lab.csv", index=False)
    pd.DataFrame(field, columns=COLUMNS + ["source"]).to_csv(mdir / "field.csv", index=False)
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    return tmp_path


def _cfg(tmp_path, **kw) -> TrainConfig:
    base = dict(manifest="lab", out_dir=str(tmp_path / "run"), model="tiny_cnn", field_manifest="field",
                train_splits=("lab_train", "field_train"), select_split="field_dev", batch_size=4, epochs=1)
    base.update(kw)
    return TrainConfig(**base)


def test_lab_cap_limits_every_class(world):
    cfg = _cfg(world, lab_cap=5)
    train, _ = select_rows(load_joint_manifest(cfg), cfg)
    lab = train[train["source"] == "lab"]
    assert lab.groupby("label").size().to_dict() == {c: 5 for c in CLASSES}
    assert (train["source"] != "lab").sum() == 2 * 2 * 3  # every real training photo stays
    again, _ = select_rows(load_joint_manifest(cfg), cfg)
    assert list(again["path"]) == list(train["path"])  # seeded, so repeatable
    assert len(cap_per_class(lab, 0, 0)) == len(lab)  # 0 means no cap
    small = lab[lab["label"] == CLASSES[0]].head(3)
    assert len(cap_per_class(small, 5, 0)) == 3  # a class under the cap keeps all its photos


def test_field_test_is_never_read(world):
    manifest = load_joint_manifest(_cfg(world))
    assert not manifest["split"].str.endswith("_test").any()
    for bad in (dict(select_split="field_test"), dict(train_splits=("lab_train", "field_test")),
                dict(train_splits=("lab_test",))):
        with pytest.raises(ValueError, match="test split"):
            load_joint_manifest(_cfg(world, **bad))
        with pytest.raises(ValueError, match="test split"):
            build_datasets(manifest, _cfg(world, **bad), make_class_map(manifest))


def test_field_dev_is_never_trained_on(world):
    cfg = _cfg(world)
    manifest = load_joint_manifest(cfg)
    train, val = build_datasets(manifest, cfg, make_class_map(manifest))
    dev = set(manifest[manifest["split"] == "field_dev"]["path"])
    assert dev and dev.isdisjoint(train.paths) and set(val.paths) == dev
    assert not any("lab_dev" in p or "lab_test" in p for p in train.paths + val.paths)
    for bad in (dict(train_splits=("lab_train", "field_dev")), dict(select_split="lab_dev")):
        with pytest.raises(ValueError):
            build_datasets(manifest, _cfg(world, **bad), make_class_map(manifest))


def test_held_out_source_is_absent_from_training_and_selection(world):
    for held, other in (("plantdoc", "plantseg"), ("plantseg", "plantdoc")):
        cfg = _cfg(world, held_out_source=held)
        manifest = load_joint_manifest(cfg)
        train, sel = select_rows(manifest, cfg)
        assert held not in set(train["source"]) | set(sel["source"])
        assert set(sel["source"]) == {other} and other in set(train["source"]) and "lab" in set(train["source"])
        ds_train, ds_val = build_datasets(manifest, cfg, make_class_map(manifest))
        assert not any(f"/{held}/" in p.replace("\\", "/") for p in ds_train.paths + ds_val.paths)


def test_background_paste_keeps_leaf_pixels_and_changes_background():
    rng = random.Random(3)
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8))
    leaf = np.zeros((40, 40), bool)
    leaf[10:30, 10:30] = True
    out = np.asarray(paste_background(image, leaf, rng))
    src = np.asarray(image)
    assert np.array_equal(out[leaf], src[leaf])
    assert not np.array_equal(out[~leaf], src[~leaf])
    assert np.mean(np.any(out[~leaf] != src[~leaf], axis=1)) > 0.9
    other = np.asarray(paste_background(image, leaf, random.Random(4)))
    assert not np.array_equal(other[~leaf], out[~leaf])  # backgrounds vary


def test_crop_arm_uses_full_image_on_tiny_mask(world):
    cfg = _cfg(world)
    manifest = load_joint_manifest(cfg)
    rows = manifest[manifest["split"] == "field_dev"].head(2)
    wide = np.zeros((320, 320), bool)
    wide[100:140, 200:260] = True
    tiny = np.zeros((320, 320), bool)
    tiny[5, 5] = True
    paths = list(rows["path"])
    for p in paths:  # make the photos non-square so a crop is visible
        Image.new("RGB", (200, 100), (10, 200, 10)).save(p)
    store = MaskStore({paths[0]: wide, paths[1]: tiny})
    ds = JointDataset(rows, make_class_map(manifest), train=False, masks=store)
    assert ds.image(0).size != (200, 100) and ds.image(0).size[0] < 200
    assert ds.image(1).size == (200, 100)  # a tiny predicted area leaves the whole photo
    with pytest.raises(KeyError):
        JointDataset(rows, make_class_map(manifest), train=False, masks=MaskStore({})).image(0)


class FakeEncoder(torch.nn.Module):
    """Same shape as DINOv2's interface: `blocks`, `norm`, class-token output."""

    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(2), torch.nn.Flatten(), torch.nn.Linear(12, 8))
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(4)])
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


def test_dino_classifier_freezes_all_but_the_head_and_the_chosen_blocks():
    from sprout.baselines.dino import DinoClassifier

    head_only = DinoClassifier(FakeEncoder(), 5, 0, feature_dim=8)
    assert head_only.backbone_parameters() == []
    assert {n for n, p in head_only.named_parameters() if p.requires_grad} == {"head.weight", "head.bias"}
    tuned = DinoClassifier(FakeEncoder(), 5, 2, feature_dim=8)
    names = {n for n, p in tuned.named_parameters() if p.requires_grad}
    assert {"encoder.blocks.2.weight", "encoder.blocks.3.weight", "encoder.norm.weight", "head.weight"} == {
        n for n in names if n.endswith(".weight")
    }
    assert not any(".blocks.0." in n or ".blocks.1." in n or "stem" in n for n in names)
    tuned.train()
    assert not tuned.encoder.training  # the frozen encoder never leaves evaluation mode
    state = tuned.state_dict()  # one state holds encoder and head, so it can be exported whole
    assert any(k.startswith("encoder.") for k in state) and "head.weight" in state
    assert tuned(torch.zeros(2, 3, 8, 8)).shape == (2, 5)


def test_optimizer_gives_unfrozen_blocks_their_own_rate():
    from sprout.baselines.dino import DinoClassifier
    from sprout.train.loop import _optimizer

    model = DinoClassifier(FakeEncoder(), 5, 1, feature_dim=8)
    opt = _optimizer(model, TrainConfig(manifest="m", out_dir="o", lr=1e-3, backbone_lr=1e-5))
    assert sorted(g["lr"] for g in opt.param_groups) == [1e-5, 1e-3]
    frozen = _optimizer(DinoClassifier(FakeEncoder(), 5, 0, feature_dim=8), TrainConfig(manifest="m", out_dir="o"))
    assert len(frozen.param_groups) == 1 and len(frozen.param_groups[0]["params"]) == 2


def test_mask_store_round_trips(tmp_path):
    m = np.zeros((320, 320), bool)
    m[10:50, 20:90] = True
    MaskStore({"a/b.jpg": m, "c.jpg": ~m}).save(tmp_path / "m.npz")
    back = MaskStore.load(tmp_path / "m.npz")
    assert np.array_equal(back.get("a/b.jpg"), m) and np.array_equal(back.get("c.jpg"), ~m)


def test_joint_run_trains_on_real_and_lab_selects_on_field_dev(world):
    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import train

    cfg = _cfg(world, lab_cap=4, epochs=2, seed=1, lr=0.05)
    out = train(cfg)
    assert out.epochs_done == 2 and len(out.history) == 2
    ckpt = load_checkpoint(out.best_path)
    assert ckpt["config"].select_split == "field_dev" and ckpt["config"].lab_cap == 4
    assert sorted(p.name for p in (world / "run").iterdir()) == ["best.pt", "latest.pt"]
