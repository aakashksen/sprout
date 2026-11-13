import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from sprout.data.splits import MANIFEST_DIR, dataset_hash
from sprout.eval.check_run import check
from sprout.experiments import e10_segment as e10
from sprout.segment.crop import crop_to_disease
from sprout.segment.data import (
    build_seg_manifest,
    exclude_held_out,
    extract_masks,
    mask_iou,
    rle_decode,
    rle_encode,
)
from sprout.segment.model import build_model, predict_mask

IMAGES = Path("datasets/raw/plantseg/plantseg/images")
MASKS = Path("datasets/raw/plantseg_masks")


def noise_photo(seed: int, size=(96, 80)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 200, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.BICUBIC)


def blob_mask(shape, box) -> np.ndarray:
    m = np.zeros(shape, bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_no_field_dev_or_field_test_photo_in_seg_manifest():
    seg = pd.read_csv(MANIFEST_DIR / "seg.csv")
    field = pd.read_csv(MANIFEST_DIR / "field.csv")
    held = field[field["split"].isin(["field_dev", "field_test"])]
    held_names = {Path(p).name for p, s in zip(held["path"], held["source"], strict=False) if s == "plantseg"}
    assert not held_names & {Path(p).name for p in seg["path"]}
    assert not set(held["sha1"]) & set(seg["sha1"])
    assert set(seg["split"]) == {"seg_train", "seg_dev"}
    assert (seg["split"] == "seg_dev").sum() >= 50
    # a group never spans both splits
    assert seg.groupby("group")["split"].nunique().max() == 1
    # held-out PlantSeg photos exist, so the name check above is not vacuous
    assert len(held_names) > 100


def test_near_duplicate_of_held_out_photo_is_excluded(tmp_path):
    held = noise_photo(1)
    photos = {
        "same_name.jpg": noise_photo(2),
        "flipped_copy.jpg": held.transpose(Image.FLIP_LEFT_RIGHT).resize((120, 100)),  # not in field.csv
        "other_a.jpg": noise_photo(3),
        "other_b.jpg": noise_photo(4),
        "other_c.jpg": noise_photo(5),
    }
    (tmp_path / IMAGES / "train").mkdir(parents=True)
    for name, img in photos.items():
        img.save(tmp_path / IMAGES / "train" / name, quality=95)
    (tmp_path / "doc").mkdir()
    held.save(tmp_path / "doc" / "held.jpg", quality=95)
    photos["same_name.jpg"].save(tmp_path / "doc" / "same_name.jpg", quality=95)
    field = pd.DataFrame({
        "path": ["doc/held.jpg", "doc/same_name.jpg"], "label": "X", "group": ["g1", "g2"],
        "split": ["field_test", "field_dev"], "sha1": ["h1", "h2"], "source": ["plantdoc", "plantseg"],
    })
    meta = pd.DataFrame({"Name": list(photos), "Disease": "d", "Label file": [n[:-4] + ".png" for n in photos],
                         "Split": "Training"})

    groups, drop = exclude_held_out([str(tmp_path / IMAGES / "train" / n) for n in photos],
                                    [str(tmp_path / "doc" / "held.jpg")])
    assert {Path(p).name for p in drop} == {"flipped_copy.jpg"}

    seg = build_seg_manifest(meta, field, root=tmp_path, seed=1)
    kept = {Path(p).name for p in seg["path"]}
    assert kept == {"other_a.jpg", "other_b.jpg", "other_c.jpg"}
    assert set(seg["split"]) <= {"seg_train", "seg_dev"}
    assert list(seg["mask"]) == [(MASKS / "train" / (Path(p).stem + ".png")).as_posix() for p in seg["path"]]


def test_crop_includes_margin():
    image = noise_photo(1, size=(100, 100))
    mask = blob_mask((100, 100), (40, 60, 30, 70))  # 20 rows by 40 columns
    tight = crop_to_disease(image, mask, 0.0)
    assert tight.size == (40, 20)
    padded = crop_to_disease(image, mask, 0.25)
    assert padded.size == (60, 30)  # 10 columns and 5 rows added on each side
    assert np.array_equal(np.asarray(padded)[5:25, 10:50], np.asarray(tight))
    # the margin stops at the photo's edge
    edge = crop_to_disease(image, blob_mask((100, 100), (0, 20, 80, 100)), 0.5)
    assert edge.size == (30, 30) and edge.size[0] <= 100
    # a mask at another resolution is stretched to the photo
    assert crop_to_disease(image, blob_mask((50, 50), (20, 30, 15, 35)), 0.0).size == (40, 20)


def test_tiny_mask_falls_back_to_full_image():
    image = noise_photo(1, size=(100, 100))
    tiny = blob_mask((100, 100), (50, 52, 50, 52))  # 4 of 10,000 pixels
    assert crop_to_disease(image, tiny, 0.1) is image
    assert crop_to_disease(image, np.zeros((100, 100), bool), 0.1) is image


def test_rle_roundtrip_and_iou():
    for m in (np.zeros((3, 4), bool), np.ones((3, 4), bool), blob_mask((5, 7), (1, 3, 2, 6)),
              np.array([[True, False], [False, True]])):
        assert np.array_equal(rle_decode(rle_encode(m)), m)
    a, b = blob_mask((10, 10), (0, 5, 0, 10)), blob_mask((10, 10), (0, 10, 0, 5))
    assert mask_iou(a, b) == pytest.approx(25 / 75)
    assert mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0


def test_extract_masks_takes_only_masks_and_rejects_escapes(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("plantseg/annotations/train/a.png", b"\x89PNG")
        z.writestr("plantseg/images/train/a.jpg", b"jpg")
        z.writestr("plantseg/Metadata.csv", b"x")
    assert extract_masks(good, tmp_path / "out") == 1
    assert [p.name for p in (tmp_path / "out").rglob("*") if p.is_file()] == ["a.png"]
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("plantseg/annotations/../../evil.png", b"x")
    with pytest.raises(ValueError, match="unsafe"):
        extract_masks(bad, tmp_path / "out2")
    assert not (tmp_path / "out2").exists() and not (tmp_path / "evil.png").exists()


def test_failed_extraction_leaves_no_partial_folder_behind(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("plantseg/annotations/train/a.png", b"x")
        z.writestr("plantseg/annotations/../../evil.png", b"x")
    with pytest.raises(ValueError, match="unsafe"):
        extract_masks(bad, tmp_path / "out")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.zip"]


# ---- training and records on a tiny fixture, on the CPU ----

def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def make_root(tmp_path, n_train=6, n_dev=4) -> tuple[Path, pd.DataFrame]:
    rows = []
    for i in range(n_train + n_dev):
        split = "seg_train" if i < n_train else "seg_dev"
        img_rel, mask_rel = IMAGES / "train" / f"p{i}.jpg", MASKS / "train" / f"p{i}.png"
        for rel in (img_rel, mask_rel):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        noise_photo(i, size=(64, 48)).save(tmp_path / img_rel)
        mask = blob_mask((48, 64), (10, 30, 12, 40))
        Image.fromarray(mask.astype(np.uint8) * 3).save(tmp_path / mask_rel)
        rows.append((img_rel.as_posix(), "d", f"g{i}", split, f"s{i}", mask_rel.as_posix()))
    df = pd.DataFrame(rows, columns=["path", "label", "group", "split", "sha1", "mask"])
    (tmp_path / "datasets/manifests").mkdir(parents=True)
    df.to_csv(tmp_path / "datasets/manifests/seg.csv", index=False)
    (tmp_path / "orchestration").mkdir()
    (tmp_path / "orchestration/tasks.json").write_text("{}")
    (tmp_path / ".gitignore").write_text("models/\n*.tmp\n")
    sh(tmp_path, "init", "-q")
    sh(tmp_path, "config", "user.email", "t@t")
    sh(tmp_path, "config", "user.name", "t")
    sh(tmp_path, "add", "-A")
    sh(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path, df


CFG = {"experiment": "E10", "manifest": "seg", "out_dir": "models/e10", "min_free_disk_gb": 0,
       "model": "lraspp_mobilenet_v3_large", "pretrained": False, "size": 64, "batch_size": 4, "num_workers": 0,
       "epochs": 2, "lr": 0.001, "weight_decay": 0.01, "seed": 0}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root, df = make_root(tmp_path_factory.mktemp("e10"))
    out = e10.train_seg(CFG, df, root, device=torch.device("cpu"))
    return root, df, out


def test_training_keeps_best_and_latest_and_logs_each_epoch(trained):
    root, _, out = trained
    assert sorted(p.name for p in (root / "models/e10").iterdir() if p.suffix == ".pt") == ["best.pt", "latest.pt"]
    assert [h["epoch"] for h in out["history"]] == [1, 2]
    assert all(np.isfinite(h["train_loss"]) and h["seconds"] > 0 for h in out["history"])
    assert len((root / "models/e10/epochs.log").read_text().splitlines()) == 2


def test_resume_continues_and_refuses_changed_settings(trained):
    root, df, _ = trained
    with pytest.raises(FileExistsError):
        e10.train_seg(CFG, df, root, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="settings differ"):
        e10.train_seg({**CFG, "lr": 0.5}, df, root, resume=True, device=torch.device("cpu"))
    more = e10.train_seg({**CFG, "epochs": 2}, df, root, resume=True, device=torch.device("cpu"))
    assert [h["epoch"] for h in more["history"]] == [1, 2]  # already finished: nothing re-run
    changed = df.copy()
    changed.loc[0, "sha1"] = "other"
    with pytest.raises(ValueError, match="different manifest"):
        e10.train_seg(CFG, changed, root, resume=True, device=torch.device("cpu"))


def test_non_finite_loss_stops_training(tmp_path):
    root, df = make_root(tmp_path)
    with pytest.raises(RuntimeError, match="not finite"):
        e10.train_seg({**CFG, "lr": 1e30}, df, root, device=torch.device("cpu"))


def test_num_workers_above_eight_is_refused(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({**CFG, "num_workers": 9}))
    with pytest.raises(ValueError, match="at most 8"):
        e10.load_config(p)
    assert e10.load_config(Path("configs/e10.json"))["num_workers"] <= 8


def test_predict_mask_matches_photo_size():
    model = build_model(pretrained=False)
    image = noise_photo(1, size=(70, 50))
    mask = predict_mask(model, image, size=64)
    assert mask.shape == (50, 70) and mask.dtype == bool


def test_saved_iou_is_recomputed(trained):
    root, df, _ = trained
    path = e10.write_records(CFG, df, root, device=torch.device("cpu"))
    assert check("E10", root=root, split="seg_dev", min_n=4) == []
    rec = json.loads(path.read_text())
    assert rec["split"] == "seg_dev" and rec["n"] == 4 and set(rec["predictions"]) == {"e10", "whole_image"}
    assert rec["dataset_hash"] == dataset_hash(df)

    # change one row's saved IoU: the record no longer recomputes
    pred_path = root / rec["predictions"]["e10"]
    original = pred_path.read_text()
    rows = [json.loads(x) for x in original.splitlines()]
    rows[0]["iou"] = min(rows[0]["iou"] + 0.2, 1.0) if rows[0]["iou"] < 0.8 else rows[0]["iou"] - 0.2
    pred_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    problems = check("E10", root=root, split="seg_dev")
    assert any("iou does not recompute" in p for p in problems), problems

    # a changed mask changes the IoU, so a row whose iou was left alone fails too
    rows[0]["iou"] = json.loads(original.splitlines()[0])["iou"]
    rows[0]["pred"] = rle_encode(~rle_decode(rows[0]["pred"]))
    pred_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert any("iou does not recompute" in p for p in check("E10", root=root, split="seg_dev"))
    pred_path.write_text(original)
    assert check("E10", root=root, split="seg_dev") == []


def test_non_finite_loss_in_record_fails_check_run(trained):
    root, _, _ = trained
    path = sorted((root / "results/runs").glob("E10-*.json"))[-1]
    rec = json.loads(path.read_text())
    rec["attempts"][0]["history"][0]["train_loss"] = float("nan")
    path.write_text(json.dumps(rec))
    assert any("non-finite loss" in p for p in check("E10", root=root, split="seg_dev"))
