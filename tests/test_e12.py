import pandas as pd
import pytest
from PIL import Image

import sprout.data.splits as splits
from sprout.train.config import TrainConfig
from sprout.train.dataset import build_datasets, make_class_map
from sprout.train.joint import load_joint_manifest, select_rows

CLASSES = ["Apple___healthy", "Cherry_(including_sour)___healthy"]
COLUMNS = ["path", "label", "group", "split", "sha1"]
SPLITS3 = ("lab_train", "field_train", "extra_train")


def _save(path, color, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """lab, field (two sources) and extra manifests under tmp_path/repo. field_test images do not exist."""
    mdir = tmp_path / "repo" / "datasets" / "manifests"
    mdir.mkdir(parents=True)
    lab, field, extra = [], [], []
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
        for source in ("plantwild", "plantpathology2021"):
            for i in range(4):
                p = tmp_path / source / label / f"x{i}.png"
                _save(p, color)
                extra.append((str(p), label, f"x{c}{source}{i}", "extra_train", f"x{c}{source}{i}", source))
    pd.DataFrame(lab, columns=COLUMNS).to_csv(mdir / "lab.csv", index=False)
    pd.DataFrame(field, columns=COLUMNS + ["source"]).to_csv(mdir / "field.csv", index=False)
    pd.DataFrame(extra, columns=COLUMNS + ["source"]).to_csv(mdir / "extra_train.csv", index=False)
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    return tmp_path


def _cfg(tmp_path, **kw) -> TrainConfig:
    base = dict(manifest="lab", out_dir=str(tmp_path / "run"), model="tiny_cnn", field_manifest="field",
                extra_manifest="extra_train", train_splits=SPLITS3, select_split="field_dev", batch_size=4, epochs=1)
    base.update(kw)
    return TrainConfig(**base)


def test_extra_rows_join_training_only(world):
    cfg = _cfg(world, lab_cap=5)
    manifest = load_joint_manifest(cfg)
    train, sel = select_rows(manifest, cfg)
    extras = train[train["source"].isin(["plantwild", "plantpathology2021"])]
    assert len(extras) == 16 and set(extras["split"]) == {"extra_train"}  # none capped: the lab cap is for lab only
    assert set(sel["split"]) == {"field_dev"} and not sel["source"].isin(["plantwild", "plantpathology2021"]).any()
    ds_train, ds_val = build_datasets(manifest, cfg, make_class_map(manifest))
    assert set(extras["path"]) <= set(ds_train.paths) and set(extras["path"]).isdisjoint(ds_val.paths)
    # without the extra split in train_splits, nothing from the extra manifest trains
    plain, _ = select_rows(load_joint_manifest(_cfg(world, train_splits=("lab_train", "field_train"))),
                           _cfg(world, train_splits=("lab_train", "field_train")))
    assert not plain["split"].eq("extra_train").any()


def test_extra_split_needs_its_manifest(world):
    with pytest.raises(ValueError, match="extra_manifest"):
        load_joint_manifest(_cfg(world, extra_manifest=""))


def test_held_out_source_is_absent_from_training_and_selection(world):
    # a second fixture case: two extras copy photos of the held-out source (one by sha1, one by group)
    extra = pd.read_csv(world / "repo" / "datasets" / "manifests" / "extra_train.csv")
    field = pd.read_csv(world / "repo" / "datasets" / "manifests" / "field.csv")
    seg = field[(field["source"] == "plantseg") & (field["split"] == "field_train")].iloc[0]
    doc = field[(field["source"] == "plantdoc") & (field["split"] == "field_train")].iloc[0]
    extra.loc[0, "sha1"] = seg["sha1"]
    extra.loc[1, "group"] = seg["group"]
    extra.loc[2, "sha1"] = doc["sha1"]
    extra.to_csv(world / "repo" / "datasets" / "manifests" / "extra_train.csv", index=False)
    copies = {"plantseg": set(extra.loc[[0, 1], "path"]), "plantdoc": {extra.loc[2, "path"]}}
    for held, other in (("plantdoc", "plantseg"), ("plantseg", "plantdoc")):
        cfg = _cfg(world, held_out_source=held)
        manifest = load_joint_manifest(cfg)
        train, sel = select_rows(manifest, cfg)
        assert held not in set(train["source"]) | set(sel["source"])
        assert set(sel["source"]) == {other} and other in set(train["source"]) and "lab" in set(train["source"])
        assert copies[held].isdisjoint(train["path"]) and copies[other] <= set(train["path"])
        assert (train["split"] == "extra_train").sum() == 16 - len(copies[held])
        ds_train, ds_val = build_datasets(manifest, cfg, make_class_map(manifest))
        assert not any(f"/{held}/" in p.replace("\\", "/") for p in ds_train.paths + ds_val.paths)


def test_held_out_source_is_absent_on_the_real_manifests():
    """Real columns: lab.csv, field.csv and extra_train.csv as committed. Reads manifests only, no image."""
    field = pd.read_csv("datasets/manifests/field.csv")
    extra = pd.read_csv("datasets/manifests/extra_train.csv")
    assert len(extra) == 3484 and set(extra["split"]) == {"extra_train"}
    for held, other in (("plantdoc", "plantseg"), ("plantseg", "plantdoc")):
        cfg = TrainConfig(manifest="lab", out_dir="unused", field_manifest="field", extra_manifest="extra_train",
                          train_splits=SPLITS3, select_split="field_dev", lab_cap=60, held_out_source=held)
        train, sel = select_rows(load_joint_manifest(cfg), cfg)
        assert held not in set(train["source"]) | set(sel["source"])
        assert set(sel["source"]) == {other} and set(sel["split"]) == {"field_dev"}
        assert not train["path"].str.contains(f"/{held}/").any() and not sel["path"].str.contains(f"/{held}/").any()
        assert (train["split"] == "extra_train").sum() == 3484  # no extra shares a sha1 or group with the held-out source
        mine = field[field["source"] == held]  # every split of the source, test included, compared on columns only
        ex = train[train["split"] == "extra_train"]  # lab and field group numbers restart per build, so only extras are compared
        assert set(ex["sha1"]).isdisjoint(mine["sha1"]) and set(ex["group"]).isdisjoint(mine["group"])
        assert set(train["sha1"]).isdisjoint(mine["sha1"])
        assert set(sel["path"]) == set(field[(field["source"] == other) & (field["split"] == "field_dev")]["path"])
