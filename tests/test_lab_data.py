import numpy as np
import pandas as pd
from PIL import Image

from sprout.data.grouping import group_near_duplicates, source_id
from sprout.data.splits import dataset_hash, make_splits


def _pattern(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = (rng.random((8, 8, 3)) * 255).astype("uint8")
    return Image.fromarray(small).resize((64, 64), Image.BICUBIC)


def _save(img: Image.Image, path):
    img.save(path)
    return str(path)


def test_duplicate_images_share_a_group(tmp_path):
    base = _pattern(1)
    paths = [
        _save(base, tmp_path / "a___X 1.png"),
        _save(base.resize((48, 48)), tmp_path / "b___X 2.png"),
        _save(base.transpose(Image.FLIP_LEFT_RIGHT), tmp_path / "c___X 3.png"),
        _save(base.rotate(90, expand=True), tmp_path / "d___X 4.png"),
        _save(_pattern(2), tmp_path / "e___X 5.png"),
    ]
    groups = group_near_duplicates(paths)
    assert len({groups[p] for p in paths[:4]}) == 1
    assert groups[paths[4]] != groups[paths[0]]


def test_same_source_id_shares_a_group(tmp_path):
    paths = [
        _save(_pattern(3), tmp_path / "0007___JR_HL 9885.JPG"),
        _save(_pattern(4), tmp_path / "0009___JR_HL 9885 copy.JPG"),
        _save(_pattern(5), tmp_path / "0008___JR_HL 9886.JPG"),
    ]
    groups = group_near_duplicates(paths)
    assert groups[paths[0]] == groups[paths[1]]
    assert groups[paths[2]] != groups[paths[0]]


def test_source_id_strips_the_suffixes_the_data_contains():
    base = source_id("c/0007___JR_HL 9885.JPG")
    assert source_id("c/0009___JR_HL 9885 copy.JPG") == base
    assert source_id("c/0009___JR_HL 9885 copy 2.jpg") == base
    assert source_id("c/0009___JR_HL 9885 copy 3.jpg") == base
    for odd in (" copy 5.jpg", " copy3.jpg", " copy  3.jpg", " copy .jpg"):
        assert source_id("c/0009___JR_HL 9885" + odd) == base
    assert source_id("c/0007___JR_HL 9885_final_masked.jpg") == base
    assert source_id("c/x___JR_HL 9885copy.jpg") == base
    assert source_id("c/x___Photocopy 12.jpg") == "c/Photocopy 12"
    assert source_id("c/x___Xcopy.jpg") == "c/Xcopy"
    assert source_id("c/RS_Rust 1563.JPG") == source_id("c/0007___RS_Rust 1563_final_masked.jpg")
    assert source_id("d/0007___JR_HL 9885.JPG") != base


def _df() -> pd.DataFrame:
    rows = []
    for label in ["a", "b", "c"]:
        for g in range(30):
            for i in range(1 + g % 3):
                rows.append((f"{label}/{g}_{i}.png", label, f"{label}{g}", "x"))
    return pd.DataFrame(rows, columns=["path", "label", "group", "sha1"])


def test_no_group_in_two_splits():
    out = make_splits(_df())
    assert (out.groupby("group")["split"].nunique() == 1).all()


def test_split_is_deterministic():
    assert make_splits(_df(), seed=13).equals(make_splits(_df(), seed=13))


def test_every_class_in_every_split():
    out = make_splits(_df())
    assert (out.groupby("label")["split"].nunique() == 3).all()
    tiny = pd.DataFrame(
        [(f"t/{i}.png", "t", f"t{i}", "x") for i in range(3)],
        columns=["path", "label", "group", "sha1"],
    )
    assert set(make_splits(tiny)["split"]) == {"lab_train", "lab_dev", "lab_test"}


def test_hash_changes_when_a_label_changes():
    m = make_splits(_df())
    changed = m.copy()
    changed.loc[0, "label"] = "b"
    assert dataset_hash(m) != dataset_hash(changed)
    assert dataset_hash(m) == dataset_hash(m.copy())


def test_split_names_are_lab_train_dev_test():
    assert set(make_splits(_df())["split"]) == {"lab_train", "lab_dev", "lab_test"}


def test_raw_data_is_git_ignored_but_pointers_are_not():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def ignored(p: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "-q", p], cwd=root, capture_output=True
        ).returncode == 0

    for p in [
        "datasets/raw/lab/x.jpg",
        "datasets/raw/coco/x.jpg",
        "datasets/raw/plantdoc/x.jpg",
        "datasets/raw/plantseg/x.jpg",
        "datasets/raw/foo.zip",
    ]:
        assert ignored(p), p
    assert not ignored("datasets/raw/lab.dvc")
    assert not ignored("datasets/raw/.gitignore")
