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


def _watermarked_crop(src: Path, dest: Path) -> str:
    from PIL import ImageDraw

    with Image.open(src) as img:
        w, h = img.size
        box = (int(w * 0.02), int(h * 0.02), int(w * 0.98), int(h * 0.98))
        out = img.convert("RGB").crop(box).resize((w, h), Image.BICUBIC)
    ImageDraw.Draw(out).rectangle((w - 40, h - 14, w - 6, h - 6), fill=(255, 255, 255))
    out.save(dest.with_suffix(".jpg"), quality=80)
    return dest.with_suffix(".jpg").as_posix()


def test_correlation_check_catches_a_watermarked_crop_under_another_label(tmp_path):
    from sprout.data.extra_sources import drop_copies_of_eval_photos

    inv, cmap, lab, field, _, new, other, _ = _world(tmp_path)
    test_photo = field[field["split"] == "field_test"]["path"].iloc[0]
    copy = _watermarked_crop(Path(test_photo), tmp_path / "extra" / "y" / "wm")
    unrelated = _photo(tmp_path / "extra" / "y" / "far.png", 999)
    extra = _rows([copy, unrelated], "y")
    extra["source"] = "plantwild"
    inv = pd.concat([inv, extra], ignore_index=True)
    gone = drop_copies_of_eval_photos(inv, cmap, field, "cpu")
    assert gone[copy] == "field_test"
    assert unrelated not in gone
    # not a field_train photo: only dev and test are read
    train_photo = field[field["split"] == "field_train"]["path"].iloc[0]
    assert train_photo not in set(gone.values())


def test_correlation_check_matches_a_plain_numpy_reference_and_the_gpu(tmp_path):
    from sprout.data.copy_check import gray_vector, max_correlation, variants

    paths = [_photo(tmp_path / f"p{i}.png", 50 + i) for i in range(5)]
    refs = paths[:3] + [_watermarked_crop(Path(paths[3]), tmp_path / "w")]
    scores, best = max_correlation(paths, refs, "cpu")
    for i, p in enumerate(paths):
        want = max(
            float((variants(gray_vector(r)) @ gray_vector(p)).max()) for r in refs
        )
        assert scores[i] == pytest.approx(want, abs=1e-5)
    assert best[3] == 3 and scores[0] > 0.999
    import torch

    if torch.cuda.is_available():
        s2, b2 = max_correlation(paths, refs, "cuda")
        assert list(b2) == list(best)
        assert np.allclose(s2, scores, atol=1e-4)


def test_extra_groups_have_their_own_namespace(tmp_path):
    inv, cmap, lab, field, *_ = _world(tmp_path)
    out, _, _ = build_extra(inv, cmap, lab, field, cap=100)
    assert out["group"].str.startswith("x").all()
    assert not set(out["group"]) & (set(field["group"]) | set(lab["group"]))
    real = MANIFEST_DIR / "extra_train.csv"
    if real.exists():
        groups = set(load_manifest("extra_train")["group"])
        assert not groups & set(load_manifest("field")["group"])
        assert not groups & set(load_manifest("lab")["group"])


_P = "datasets/raw/plantwild/plantwild/images/"
# The 13 leaks found by the two failed verifications (6, then 7), plus 3 more seen while checking.
VERIFIER_PAIRS = [
    _P + "strawberry leaf/google_0526.jpg",
    _P + "soybean leaf/google_0562.jpg",
    _P + "blueberry leaf/169.jpg",
    _P + "raspberry leaf/61.jpg",
    _P + "strawberry leaf/google_0201.jpg",
    _P + "raspberry leaf/google_0075.jpg",
    _P + "tomato mosaic virus/google_0165.jpg",
    _P + "tomato early blight/google_0068.jpg",
    _P + "potato late blight/google_0313 - Copy (3) - Copy.jpg",
    _P + "corn gray leaf spot/google_0175.jpg",
    _P + "apple rust/google_0184.jpg",
    _P + "bell pepper leaf spot/google_0071.jpg",
    _P + "potato late blight/google_0097.jpg",
    _P + "apple leaf/1.jpg",
    _P + "apple scab/57 - Copy.jpg",
    _P + "bell pepper leaf spot/google_0015.jpg",
]


def _zoomed_tokens(seed: int):
    """Patch tokens of a photo and of a 2x zoom on its centre (each B cell copies the A cell it shows)."""
    import torch

    from sprout.data.crop_copies import GRID

    rng = np.random.default_rng(seed)
    a = torch.nn.functional.normalize(
        torch.from_numpy(rng.normal(size=(GRID * GRID, 64))).float(), dim=1
    )
    b = torch.stack(
        [
            a[(y // 2 + GRID // 4) * GRID + x // 2 + GRID // 4]
            for y in range(GRID)
            for x in range(GRID)
        ]
    )
    b = torch.nn.functional.normalize(b + 0.05 * torch.randn_like(b), dim=1)
    return a, b


def test_patch_matching_finds_a_zoomed_copy_and_not_an_unrelated_photo():
    from sprout.data.crop_copies import INLIERS, mutual_matches, ransac_inliers

    a, zoom = _zoomed_tokens(1)
    other, _ = _zoomed_tokens(2)
    ia, ib, _ = mutual_matches(a, zoom)
    assert ransac_inliers(ia, ib) >= INLIERS
    ia, ib, _ = mutual_matches(other, zoom)
    assert ransac_inliers(ia, ib) < INLIERS


_D = "datasets/raw/plantdoc/"
KNOWN_COPY_REFS = {
    VERIFIER_PAIRS[0]: _D
    + "Strawberry leaf/stock-photo-strawberry-leaf-isolated-on-white-background-572055775.jpg",
    VERIFIER_PAIRS[6]: _D
    + "Tomato leaf mosaic virus/ill-tomato-of-tomato-mosaic-virus-DBE17N.jpg",
    VERIFIER_PAIRS[7]: _D + "Tomato Early blight leaf/Tomato-blight-large.jpg",
    VERIFIER_PAIRS[8]: _D + "Potato leaf late blight/fig1a_500.jpg",
    VERIFIER_PAIRS[9]: _D + "Corn rust leaf/corn-diseases-fig-1-so-rust.jpg",
    VERIFIER_PAIRS[11]: _D + "Bell_pepper leaf spot/pepper_V5a.jpg",
    VERIFIER_PAIRS[12]: _D + "Potato leaf late blight/fig1a_500.jpg",
}


@pytest.mark.real
def test_dino_crop_check_flags_known_cropped_and_captioned_copies():
    from sprout.baselines.dino import load_dino_encoder
    from sprout.data.crop_copies import crop_copies

    field = load_manifest("field").set_index("path")
    refs = sorted(set(KNOWN_COPY_REFS.values()))
    if not Path(refs[0]).exists():
        pytest.skip("raw images are not on this machine")
    splits = [field.loc[r, "split"] for r in refs]
    encoder = load_dino_encoder().eval()
    found = crop_copies(list(KNOWN_COPY_REFS), refs, splits, encoder, "cpu")
    assert set(found) == set(KNOWN_COPY_REFS)
    lab = load_manifest("lab")["path"].iloc[0]
    assert not crop_copies([lab], refs, splits, encoder, "cpu")


@pytest.mark.real
def test_dino_crop_check_flags_nothing_in_the_real_manifest():
    import torch

    from sprout.baselines.dino import load_dino_encoder
    from sprout.data.crop_copies import crop_copies

    extra = load_manifest("extra_train")
    paths = list(extra["path"])
    if not Path(paths[0]).exists() or not torch.cuda.is_available():
        pytest.skip("needs the raw images and a GPU")
    field = load_manifest("field")
    held = field[field["split"].isin(["field_dev", "field_test"])]
    encoder = load_dino_encoder().to("cuda").eval()
    flagged = crop_copies(
        paths, list(held["path"]), list(held["split"]), encoder, "cuda"
    )
    assert not flagged, f"{len(flagged)} extras look like crops of dev or test photos"


def _pp_parquet(tmp_path: Path) -> Path:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        ("a.jpg", ["scab"]),
        ("b", ["healthy"]),
        ("c.jpg", ["rust"]),
        ("d.jpg", ["scab", "frog_eye_leaf_spot"]),
        ("e.jpg", ["frog_eye_leaf_spot"]),
        ("f.jpg", ["complex"]),
        ("g.jpg", ["powdery_mildew"]),
        ("h.jpg", ["rust", "complex"]),
        ("i", ["healthy"]),
    ]

    def jpeg(i: int) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), 20 * i, dtype="uint8")).save(buf, "JPEG")
        return buf.getvalue()

    table = pa.table(
        {
            "image": [{"bytes": jpeg(i), "path": n} for i, (n, _) in enumerate(rows)],
            "labels": [list(range(len(c))) for _, c in rows],
            "label_names": [c for _, c in rows],
            "image_id": [n for n, _ in rows],
        }
    )
    out = tmp_path / "pp" / "data"
    out.mkdir(parents=True)
    pq.write_table(table, out / "train-00000-of-00001.parquet")
    return out


def test_only_single_label_rows_that_map_are_used(tmp_path):
    from sprout.data.extra_sources import (
        PP_SOURCE,
        extract_plant_pathology,
        plant_pathology_counts,
    )

    files = sorted(_pp_parquet(tmp_path).glob("*.parquet"))
    dest = tmp_path / "pp_img"
    counts = extract_plant_pathology(files, dest)
    on_disk = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.*"))
    assert on_disk == ["healthy/b.jpg", "healthy/i.jpg", "rust/c.jpg", "scab/a.jpg"]
    assert counts == plant_pathology_counts(files)
    assert counts["healthy"] == 2 and counts["frog_eye_leaf_spot"] == 1
    assert counts["frog_eye_leaf_spot+scab"] == 1 and counts["complex+rust"] == 1
    assert sum(counts.values()) == 9
    cmap = make_extra_class_map({PP_SOURCE: sorted(counts)})
    mapped = dict(zip(cmap["source_class"], cmap["lab_class"]))
    assert {k: v for k, v in mapped.items() if v} == {
        "scab": "Apple___Apple_scab",
        "healthy": "Apple___healthy",
        "rust": "Apple___Cedar_apple_rust",
    }
    reasons = dict(zip(cmap["source_class"], cmap["reason"]))
    assert "multi-label" in reasons["complex+rust"]
    assert reasons["powdery_mildew"] and reasons["complex"]
    with pytest.raises(FileExistsError):
        extract_plant_pathology(files, dest)


def test_a_new_source_fills_only_the_room_the_cap_leaves(tmp_path):
    from sprout.data.extra_sources import PP_SOURCE

    inv, cmap, lab, field, *_ = _world(tmp_path, n_new=3)
    old, _, _ = build_extra(inv, cmap, lab, field, cap=4)
    ex = tmp_path / "extra"
    new = [_photo(ex / "pp" / f"p{i}.png", 700 + i) for i in range(6)]
    # a second copy of an established photo: the established one must stay
    dup = _photo(ex / "pp" / "dup.png", 100, size=96)
    add = _rows(new + [dup], "scab")
    add["source"] = PP_SOURCE
    cmap2 = pd.concat(
        [
            cmap,
            pd.DataFrame(
                {
                    "source": PP_SOURCE,
                    "source_class": ["scab"],
                    "lab_class": ["A"],
                    "reason": [""],
                }
            ),
        ]
    )
    both, _, removed = build_extra(
        pd.concat([inv, add], ignore_index=True), cmap2, lab, field, cap=4
    )
    est = both[both["source"] == "plantwild"]
    assert set(est["path"]) == set(old["path"])  # nothing established moved
    assert len(both[both["label"] == "A"]) == 4
    assert len(both[both["source"] == PP_SOURCE]) == 4 - len(old[old["label"] == "A"])
    assert dup not in set(both["path"])
    # the new rows kept are the first by sha1 among those that survived the checks
    got = both[both["source"] == PP_SOURCE]["sha1"].tolist()
    assert got == sorted(got)
    assert removed["over the per-class cap"] == 6 - len(got)


def test_plant_pathology_rows_in_the_real_manifest_are_single_label_and_mapped():
    from sprout.data.extra_sources import CLASS_MAP_EXTRA_PATH, PP_MAP, PP_SOURCE

    extra = load_manifest("extra_train")
    pp = extra[extra["source"] == PP_SOURCE]
    assert len(pp) > 0
    for r in pp.itertuples():
        cls = Path(r.path).parent.name
        assert Path(r.path).parent.parent.name == "plantpathology2021"
        assert PP_MAP[cls] == r.label
    cmap = pd.read_csv(CLASS_MAP_EXTRA_PATH).fillna("")
    cmap = cmap[cmap["source"] == PP_SOURCE]
    assert cmap["n_images"].sum() == 18632  # every row of the mirror is accounted for
    assert set(cmap[cmap["lab_class"] != ""]["source_class"]) == set(PP_MAP)
    assert (cmap[cmap["lab_class"] == ""]["n_kept"] == 0).all()
    assert (cmap[cmap["lab_class"] == ""]["reason"] != "").all()
    assert cmap["n_kept"].sum() == len(pp)
