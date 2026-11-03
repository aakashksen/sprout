from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sprout.data.extra_sources import (
    CAP_PER_CLASS,
    EXTRA_SPLIT,
    PLANTWILD_MAP,
    build_extra,
    extra_inventory,
    make_extra_class_map,
    safe_extract,
)
from sprout.data.inventory import file_sha1
from sprout.data.splits import MANIFEST_DIR, dataset_hash, load_manifest

FIELD_HASH = "0330ee7ee93fd99d0c61a9e9fe899b4c08eac7ff6a94deabc4ac3232206f437e"


def _photo(path: Path, seed: int, size=128, flip=False) -> str:
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((6, 6, 3)) * 255).astype("uint8"))
    img = img.resize((size, size), Image.BICUBIC)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path.as_posix()


def _rows(paths, label, split=None):
    df = pd.DataFrame(
        {"path": paths, "label": label, "sha1": [file_sha1(Path(p)) for p in paths]}
    )
    df["group"] = [f"g{label}{i}" for i in range(len(df))]
    if split is not None:
        df["split"] = split
    return df


def _world(tmp_path, n_new=4):
    lab = _rows([_photo(tmp_path / "lab" / "l0.png", 1)], "A")
    f_test = _photo(tmp_path / "field" / "t.png", 2)
    f_dev = _photo(tmp_path / "field" / "d.png", 3)
    f_train = _photo(tmp_path / "field" / "r.png", 4)
    field = pd.concat(
        [
            _rows([f_test], "A", "field_test"),
            _rows([f_dev], "A", "field_dev"),
            _rows([f_train], "A", "field_train"),
        ]
    )
    field["source"] = "plantdoc"
    ex = tmp_path / "extra"
    paths = {
        "x/test_exact": _photo(ex / "x" / "test_exact.png", 2),
        "x/dev_resized": _photo(ex / "x" / "dev_resized.png", 3, size=64),
        "x/train_flipped": _photo(ex / "x" / "train_flipped.png", 4, flip=True),
        "x/lab_copy": _photo(ex / "x" / "lab_copy.png", 1),
    }
    new = [_photo(ex / "x" / f"new{i}.png", 100 + i) for i in range(n_new)]
    other = [_photo(ex / "y" / "new.png", 200)]
    unmapped = [_photo(ex / "z" / "new.png", 300)]
    inv = pd.concat(
        [
            _rows(list(paths.values()) + new, "x"),
            _rows(other, "y"),
            _rows(unmapped, "z"),
        ],
        ignore_index=True,
    )
    inv["source"] = "plantwild"
    cmap = pd.DataFrame(
        {
            "source": "plantwild",
            "source_class": ["x", "y", "z"],
            "lab_class": ["A", "B", ""],
            "reason": ["", "", "no lab class for this plant or disease"],
        }
    )
    return inv, cmap, lab, field, paths, new, other, unmapped


def test_no_extra_image_duplicates_a_field_dev_or_field_test_photo(tmp_path):
    inv, cmap, lab, field, paths, new, *_ = _world(tmp_path)
    out, _, removed = build_extra(inv, cmap, lab, field, cap=100)
    kept = set(out["path"])
    for copy in paths.values():
        assert copy not in kept
    assert set(new) <= kept
    assert removed["exact copy of field_test"] == 1
    assert removed["near copy of field_dev"] == 1
    assert removed["near copy of field_train"] == 1
    assert removed["exact copy of lab"] == 1
    field_sha = set(field["sha1"])
    assert not field_sha & set(out["sha1"])


def test_extra_sources_only_add_training_rows(tmp_path):
    inv, cmap, lab, field, *_ = _world(tmp_path)
    out, _, _ = build_extra(inv, cmap, lab, field, cap=100)
    assert set(out["split"]) == {EXTRA_SPLIT}
    assert list(out.columns) == ["path", "label", "group", "split", "sha1", "source"]
    real = MANIFEST_DIR / "extra_train.csv"
    if real.exists():
        assert set(load_manifest("extra_train")["split"]) == {EXTRA_SPLIT}
        assert not set(load_manifest("extra_train")["sha1"]) & set(
            load_manifest("field")["sha1"]
        )
        assert not set(load_manifest("extra_train")["sha1"]) & set(
            load_manifest("lab")["sha1"]
        )


def test_field_manifest_is_unchanged(tmp_path):
    before = (MANIFEST_DIR / "field.csv").read_bytes()
    inv, cmap, lab, field, *_ = _world(tmp_path)
    build_extra(inv, cmap, lab, field, cap=100)
    assert (MANIFEST_DIR / "field.csv").read_bytes() == before
    assert dataset_hash(load_manifest("field")) == FIELD_HASH


def test_cap_limits_every_class(tmp_path):
    inv, cmap, lab, field, *_ = _world(tmp_path, n_new=6)
    out, _, removed = build_extra(inv, cmap, lab, field, cap=2)
    assert out.groupby("label").size().max() == 2
    assert removed["over the per-class cap"] == 4
    real = MANIFEST_DIR / "extra_train.csv"
    if real.exists():
        assert (
            load_manifest("extra_train").groupby("label").size().max() <= CAP_PER_CLASS
        )


def test_unmapped_classes_excluded(tmp_path):
    inv, cmap, lab, field, _, _, _, unmapped = _world(tmp_path)
    out, cmap_out, _ = build_extra(inv, cmap, lab, field, cap=100)
    assert unmapped[0] not in set(out["path"])
    z = cmap_out[cmap_out["source_class"] == "z"].iloc[0]
    assert z["status"] == "excluded" and z["reason"] and z["n_kept"] == 0
    names = make_extra_class_map({"plantwild": ["apple scab", "rice blast"]})
    rice = names[names["source_class"] == "rice blast"].iloc[0]
    assert rice["lab_class"] == "" and rice["reason"]
    assert PLANTWILD_MAP["apple scab"] == "Apple___Apple_scab"


def test_inventory_lists_class_folders(tmp_path):
    _photo(tmp_path / "r" / "c1" / "a.png", 1)
    _photo(tmp_path / "r" / "c2" / "b.png", 2)
    inv = extra_inventory({"s": tmp_path / "r"})
    assert sorted(inv["label"]) == ["c1", "c2"] and set(inv["source"]) == {"s"}


def test_extract_rejects_paths_that_escape(tmp_path):
    import zipfile

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("ok/a.txt", "x")
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_hash_close_lookalike_of_a_field_photo_is_dropped(tmp_path):
    inv, cmap, lab, field, paths, new, *_ = _world(tmp_path)
    noisy = tmp_path / "extra" / "x" / "noisy.png"
    arr = np.asarray(Image.open(tmp_path / "field" / "d.png").convert("RGB")).astype(
        int
    )
    arr = np.clip(arr + np.random.default_rng(0).integers(-12, 12, arr.shape), 0, 255)
    Image.fromarray(arr.astype("uint8")).save(noisy)
    extra_row = _rows([noisy.as_posix()], "x")
    extra_row["source"] = "plantwild"
    inv = pd.concat([inv, extra_row], ignore_index=True)
    out, _, _ = build_extra(inv, cmap, lab, field, cap=100)
    assert noisy.as_posix() not in set(out["path"])
