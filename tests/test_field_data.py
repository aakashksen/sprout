import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from sprout.data.field import (
    FIELD_SPLITS,
    PLANTDOC_MAP,
    assemble_field,
    hash_pairs,
    link_same_scene,
    make_class_map,
    not_plant_images,
)
from sprout.data.inventory import file_sha1
from sprout.data.splits import MANIFEST_DIR, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def _img(path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    small = (rng.random((8, 8, 3)) * 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(small).resize((64, 64), Image.BICUBIC).save(path)
    return path.as_posix()


def _smooth(seed: int, box=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((6, 6, 3)) * 255).astype("uint8"))
    img = img.resize((128, 128), Image.BICUBIC)
    return img.crop(box).resize((128, 128), Image.BICUBIC) if box else img


def _inventory(paths_by_class: dict[str, list[str]], source: str) -> pd.DataFrame:
    rows = [
        (p, c, file_sha1(Path(p)), source)
        for c, ps in paths_by_class.items()
        for p in ps
    ]
    return pd.DataFrame(rows, columns=["path", "label", "sha1", "source"])


def _fixture(tmp_path):
    seed = iter(range(1000))
    lab_paths = [
        _img(tmp_path / "lab" / "A" / f"l{i}.png", next(seed)) for i in range(2)
    ]
    lab = pd.DataFrame(
        {"path": lab_paths, "sha1": [file_sha1(Path(p)) for p in lab_paths]}
    )
    exact = tmp_path / "f" / "a" / "copy_exact.png"
    exact.parent.mkdir(parents=True)
    exact.write_bytes(Path(lab_paths[0]).read_bytes())
    near = tmp_path / "f" / "a" / "copy_near.png"
    Image.open(lab_paths[1]).resize((48, 48)).save(near)
    a_images = [_img(tmp_path / "f" / "a" / f"x{i}.png", next(seed)) for i in range(12)]
    dup = tmp_path / "f" / "a" / "x0_dup.png"
    dup.write_bytes(Path(a_images[0]).read_bytes())
    # same scene at a different crop: pHash far, so only the field pass links them
    same = [tmp_path / "f" / "a" / "p1.png", tmp_path / "f" / "a" / "p1_crop.png"]
    _smooth(2).save(same[0])
    _smooth(2, (0, 0, 124, 124)).save(same[1])
    clash = [tmp_path / "f" / "a" / "q.png", tmp_path / "f" / "b" / "q_crop.png"]
    _smooth(4).save(clash[0])
    clash[1].parent.mkdir(parents=True, exist_ok=True)
    _smooth(4, (4, 4, 124, 124)).save(clash[1])
    fresh = {
        "a": [
            str(exact),
            str(near),
            str(dup),
            str(same[0]),
            str(same[1]),
            str(clash[0]),
        ]
        + a_images,
        "b": [str(clash[1])]
        + [_img(tmp_path / "f" / "b" / f"y{i}.png", next(seed)) for i in range(12)],
        "weird": [
            _img(tmp_path / "f" / "weird" / f"z{i}.png", next(seed)) for i in range(5)
        ],
        "tiny": [
            _img(tmp_path / "f" / "tiny" / f"t{i}.png", next(seed)) for i in range(2)
        ],
    }
    inv = _inventory(
        {k: [Path(p).as_posix() for p in v] for k, v in fresh.items()}, "s"
    )
    cmap = pd.DataFrame(
        {
            "source": ["s"] * 4,
            "source_class": ["a", "b", "weird", "tiny"],
            "lab_class": ["A", "B", "", "T"],
            "reason": ["", "", "no lab counterpart", ""],
        }
    )
    return inv, cmap, lab


def test_not_plant_excludes_potted_plant_images(tmp_path):
    ann = {
        "categories": [{"id": 1, "name": "person"}, {"id": 64, "name": "potted plant"}],
        "images": [{"id": i, "file_name": f"{i:012d}.jpg"} for i in (1, 2, 3)],
        "annotations": [
            {"image_id": 1, "category_id": 1},
            {"image_id": 2, "category_id": 1},
            {"image_id": 2, "category_id": 64},
        ],
    }
    f = tmp_path / "instances.json"
    f.write_text(json.dumps(ann))
    names = [p.name for p in not_plant_images(f, tmp_path)]
    assert names == ["000000000001.jpg", "000000000003.jpg"]


def test_no_coco_or_plantdoc_image_is_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    images = {".jpg", ".jpeg", ".png"}
    bad = [
        t
        for t in tracked
        if Path(t).suffix.lower() in images
        and (
            t.startswith("datasets/raw/")
            or "coco" in t.lower()
            or "plantdoc" in t.lower()
        )
    ]
    assert bad == []


def test_unmapped_classes_excluded(tmp_path):
    inv, cmap, lab = _fixture(tmp_path)
    field, out, _ = assemble_field(inv, cmap, lab)
    row = out[out["source_class"] == "weird"].iloc[0]
    assert row["status"] == "excluded" and row["reason"] == "no lab counterpart"
    assert set(field["label"]) == {"A", "B"}
    assert not field["path"].str.contains("/weird/|/tiny/").any()
    real = pd.read_csv(MANIFEST_DIR.parent / "class_map.csv", keep_default_na=False)
    excluded = real[real["status"] == "excluded"]
    assert len(excluded) > 0 and (excluded["lab_class"] == "").all()
    assert (excluded["reason"] != "").all()


def test_no_field_image_duplicates_a_lab_image(tmp_path):
    inv, cmap, lab = _fixture(tmp_path)
    field, _, _ = assemble_field(inv, cmap, lab)
    assert not field["path"].str.contains("copy_").any()
    assert field["sha1"].is_unique
    assert set(field["sha1"]).isdisjoint(set(lab["sha1"]))
    real = load_manifest("field")
    assert set(real["sha1"]).isdisjoint(set(load_manifest("lab")["sha1"]))


def test_no_group_in_two_field_splits(tmp_path):
    inv, cmap, lab = _fixture(tmp_path)
    field, _, _ = assemble_field(inv, cmap, lab)
    assert field.groupby("group")["split"].nunique().max() == 1
    assert set(field["split"]) <= set(FIELD_SPLITS)
    real = load_manifest("field")
    assert real.groupby("group")["split"].nunique().max() == 1
    assert real.groupby("sha1")["split"].nunique().max() == 1
    assert real.groupby("label")["split"].nunique().eq(len(FIELD_SPLITS)).all()
    by_name = field.set_index(field["path"].str.rsplit("/", n=1).str[-1])
    assert by_name.loc["p1.png", "group"] == by_name.loc["p1_crop.png", "group"]


def test_field_test_has_at_least_200_images():
    m = load_manifest("field")
    assert (m["split"] == "field_test").sum() >= 200


def test_every_shared_class_is_in_field_test(tmp_path):
    inv, cmap, lab = _fixture(tmp_path)
    field, out, _ = assemble_field(inv, cmap, lab)
    shared = set(out.loc[out["status"] == "shared", "lab_class"])
    assert shared == {"A", "B"}
    assert shared <= set(field[field["split"] == "field_test"]["label"])
    tiny = out[out["source_class"] == "tiny"].iloc[0]
    assert tiny["status"] == "dropped" and tiny["reason"]

    real = load_manifest("field")
    real_map = pd.read_csv(MANIFEST_DIR.parent / "class_map.csv", keep_default_na=False)
    real_shared = set(real_map.loc[real_map["status"] == "shared", "lab_class"])
    assert real_shared <= set(real[real["split"] == "field_test"]["label"])


def test_shared_class_list_is_recorded():
    real_map = pd.read_csv(MANIFEST_DIR.parent / "class_map.csv", keep_default_na=False)
    lab_classes = set(load_manifest("lab")["label"])
    assert set(real_map["status"]) <= {"shared", "dropped", "excluded"}
    shared = set(real_map.loc[real_map["status"] == "shared", "lab_class"])
    assert shared and shared <= lab_classes
    assert shared == set(load_manifest("field")["label"])
    dropped = real_map[real_map["status"] == "dropped"]
    assert (dropped["reason"] != "").all()
    assert set(PLANTDOC_MAP) <= set(
        real_map.loc[real_map["source"] == "plantdoc", "source_class"]
    )


def test_conflicting_near_duplicates_are_removed_and_listed(tmp_path):
    inv, cmap, lab = _fixture(tmp_path)
    field, _, conflicts = assemble_field(inv, cmap, lab)
    names = set(field["path"].str.rsplit("/", n=1).str[-1])
    assert not names & {"q.png", "q_crop.png"}
    assert sorted(conflicts["path"].str.rsplit("/", n=1).str[-1]) == [
        "q.png",
        "q_crop.png",
    ]
    assert (conflicts["reason"] != "").all()

    real = load_manifest("field")
    listed = pd.read_csv(MANIFEST_DIR / "field_conflicts.csv", keep_default_na=False)
    assert len(listed) > 0 and (listed["reason"] != "").all()
    assert set(listed["path"]).isdisjoint(set(real["path"]))


def test_class_map_reasons_match_status():
    sources = pd.DataFrame(
        {"Plant": ["Apple", "Rice"], "Disease": ["apple scab", "rice blast"]}
    )
    out = make_class_map(["Apple Scab Leaf", "Mystery"], sources)
    mapped = out[out["lab_class"] != ""]
    assert (mapped["reason"] == "").all()
    assert (out[out["lab_class"] == ""]["reason"] != "").all()
    real = pd.read_csv(MANIFEST_DIR.parent / "class_map.csv", keep_default_na=False)
    assert (real[real["status"] == "shared"]["reason"] == "").all()


def _shape(kind: str, path: Path) -> str:
    img = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(img)
    if kind == "ellipse":
        draw.ellipse((14, 44, 114, 84), fill=(40, 140, 50))
    else:
        draw.polygon([(20, 110), (64, 14), (108, 110), (64, 80)], fill=(30, 100, 40))
    img.save(path)
    return path.as_posix()


def test_loose_hash_candidates_cannot_merge_unrelated_images(tmp_path):
    paths = [
        _img(tmp_path / "a.png", 0),
        _img(tmp_path / "b.png", 1),
        _shape("ellipse", tmp_path / "e.png"),
        _shape("kite", tmp_path / "k.png"),
    ]
    _smooth(10).save(tmp_path / "s.png")
    _smooth(10, (4, 4, 124, 124)).save(tmp_path / "s_crop.png")
    _smooth(10, (0, 0, 120, 120)).save(tmp_path / "s_crop2.png")
    paths += [(tmp_path / n).as_posix() for n in ("s.png", "s_crop.png", "s_crop2.png")]
    # hash_distance 64 proposes every pair, so only the verification can keep them apart
    out = link_same_scene(
        paths, {p: f"g{i}" for i, p in enumerate(paths)}, hash_distance=64
    )
    assert len({out[p] for p in paths[:4]}) == 4
    assert len({out[p] for p in paths[4:]}) == 1
    assert out[paths[4]] not in {out[p] for p in paths[:4]}


def test_no_unlisted_cross_split_near_duplicate_in_real_manifest():
    real = load_manifest("field")
    if not Path(real["path"].iloc[0]).exists():
        pytest.skip("raw images are not on this machine")
    split = dict(zip(real["path"], real["split"]))
    found = hash_pairs(list(real["path"]), 8)
    found = found[
        [split[a] != split[b] for a, b in zip(found["path_a"], found["path_b"])]
    ]
    listed = pd.read_csv(MANIFEST_DIR / "field_residual_pairs.csv")
    assert set(zip(found["path_a"], found["path_b"])) <= set(
        zip(listed["path_a"], listed["path_b"])
    )


def test_residual_pairs_are_recorded_with_scores():
    real = load_manifest("field")
    split = dict(zip(real["path"], real["split"]))
    listed = pd.read_csv(MANIFEST_DIR / "field_residual_pairs.csv")
    assert len(listed) > 0 and listed["scene_score"].between(0, 1).all()
    assert all(split[a] != split[b] for a, b in zip(listed["path_a"], listed["path_b"]))
    assert (listed["scene_score"] < 0.88).all()
