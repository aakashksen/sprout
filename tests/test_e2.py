import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import sprout.data.splits as splits
from sprout.data.masks import (
    energy_outside,
    hsv_mask,
    iou,
    segmented_mask,
    segmented_path,
)
from sprout.eval.check_run import check
from sprout.eval.predict import Predictions
from sprout.experiments import e2_review as review
from sprout.experiments.e2_field import load_rows, report, run, shared_classes
from sprout.explain.gradcam import grad_cam, overlay, target_layer
from sprout.train.config import TrainConfig
from sprout.train.loop import build_model

A, B, C = "Apple___healthy", "Cherry___healthy", "Corn___only_in_field"
E1_ID = "E1-20260101T000000000000Z"
BEST = 0.75


def signal_model() -> nn.Sequential:
    """Reads only the red channel and says class 0 where it is bright."""
    conv = nn.Conv2d(3, 1, 1, bias=False)
    fc = nn.Linear(1, 2)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1))
        fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        fc.bias.zero_()
    return nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)


def test_gradcam_output_matches_image_size():
    torch.manual_seed(0)
    model = build_model("tiny_cnn", 4, pretrained=False)
    x = torch.randn(3, 3, 224, 224)
    cam, logits = grad_cam(model, x, target_layer(model, "tiny_cnn"))
    assert cam.shape == (3, 224, 224) and logits.shape == (3, 4)
    assert cam.min() >= 0 and cam.max() <= 1
    photo = Image.new("RGB", (300, 180), (10, 120, 10))
    assert overlay(photo, cam[0]).size == (300, 180)
    assert all(p.requires_grad for p in model.parameters())


def test_gradcam_highlights_the_only_signal():
    model = signal_model()
    x = torch.zeros(1, 3, 32, 32)
    x[0, 0, 8:16, 8:16] = 1.0
    cam, _ = grad_cam(model, x, model[1])
    inside = np.zeros((32, 32), dtype=bool)
    inside[8:16, 8:16] = True
    assert cam[0][inside].min() > 0.99
    assert cam[0][~inside].max() < 0.1
    assert energy_outside(cam[0], inside) < 0.05
    assert energy_outside(cam[0], ~inside) > 0.95


# ---- runner on a tiny fixture ------------------------------------------------------------------


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    img.paste(Image.new("RGB", (32, 32), color), (16, 16))
    img.save(path)


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
        p = root / "datasets" / "raw" / "lab" / label / f"{i}___x{i}.png"
        if split != "lab_test":
            save_image(p, color)
            save_image(root / "datasets" / "raw" / "lab_segmented" / label / f"{i}___x{i}_final_masked.png", color)
        lab.append((str(p), label, f"g{i}", split, f"sha{i}"))
    for i in range(12):
        label = [A, B, C][i % 3]
        split = "field_train" if i < 3 else "field_dev" if i < 9 else "field_test"
        p = root / "datasets" / "raw" / "field" / f"f{i}.png"
        if split != "field_test":  # the test images do not exist: reading one would fail the run
            save_image(p, (200, 30, 30) if label == A else (30, 200, 30))
        field.append((str(p), label, f"h{i}", split, f"fsha{i}", "plantdoc"))
    cols = ["path", "label", "group", "split", "sha1"]
    pd.DataFrame(lab, columns=cols).to_csv(mdir / "lab.csv", index=False)
    pd.DataFrame(field, columns=[*cols, "source"]).to_csv(mdir / "field.csv", index=False)
    torch.manual_seed(0)
    model = build_model("tiny_cnn", 2, pretrained=False)
    ckpt = root / "models" / "e1" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    train_cfg = TrainConfig(manifest="lab", out_dir="x", model="tiny_cnn", epochs=1)
    torch.save(
        {"model": model.state_dict(), "class_map": {A: 0, B: 1}, "config": train_cfg.to_dict(),
         "best_metric": BEST, "epoch": 1},
        ckpt,
    )
    runs = root / "results" / "runs"
    runs.mkdir(parents=True)
    (runs / f"{E1_ID}.json").write_text(json.dumps({"attempts": [{"best_dev_accuracy": BEST}]}))
    (root / ".gitignore").write_text("models/\nresults/e2-gradcam/field/\n")
    sh(root, "init", "-q")
    sh(root, "config", "user.email", "t@t")
    sh(root, "config", "user.name", "t")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    return root


@pytest.fixture
def cfg():
    return {
        "experiment": "E2", "checkpoint": "models/e1/best.pt", "e1_run_id": E1_ID,
        "splits": ["field_dev", "lab_dev"], "batch_size": 4, "num_workers": 0, "seed": 0,
        "out_dir": "results/e2-gradcam", "overlays_per_cell": 2,
        "mistakes_n": 3, "mistakes_sheet": "results/e2-mistakes.csv",
        "mistakes_report": "results/error-analysis-e2.md",
        "mask_sample_n": 2, "mask_sheet": "results/e2-mask-check.csv", "mask_min_good": 0.8,
        "field_masks_reliable": False,
    }


def records(paths):
    recs = [json.loads(p.read_text()) for p in paths]
    return {r["split"]: r for r in recs}


def test_only_shared_classes_are_scored(repo, cfg):
    assert shared_classes() == {A, B}
    assert set(load_rows("field_dev", shared_classes())["label"]) == {A, B}
    field = records(run(cfg, repo))["field_dev"]
    assert field["n"] == 4 and field["subset"] == "shared_classes"  # 6 dev photos, 2 of class C left out
    rows = [json.loads(x) for x in (repo / field["predictions"]["e1"]).read_text().splitlines()]
    assert len(rows) == 4 and all(len(r["probs"]) == 2 for r in rows)
    assert not any(r["path"].endswith(("f2.png", "f5.png")) for r in rows)
    assert check("E2", root=repo) == []


def test_other_splits_are_refused_by_the_loader(repo):
    for split in ("field_test", "field_train", "lab_test", "lab_train"):
        with pytest.raises(ValueError, match="reads only"):
            load_rows(split, {A, B})


def test_runner_refuses_missing_checkpoint_and_dirty_tree(repo, cfg):
    (repo / "models" / "e1" / "best.pt").unlink()
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        run(cfg, repo)
    (repo / "extra.txt").write_text("x")
    with pytest.raises(RuntimeError, match="uncommitted"):
        run(cfg, repo)


def test_runner_refuses_a_checkpoint_from_another_run(repo, cfg):
    cfg["e1_run_id"] = "E1-20260202T000000000000Z"
    with pytest.raises(RuntimeError, match="E1 record"):
        run(cfg, repo)
    other = repo / "results" / "runs" / "E1-20260202T000000000000Z.json"
    other.write_text(json.dumps({"attempts": [{"best_dev_accuracy": 0.1}]}))
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "other")
    with pytest.raises(RuntimeError, match="not produced by that run"):
        run(cfg, repo)


def test_record_names_checkpoint_and_labels_the_masked_number(repo, cfg):
    recs = records(run(cfg, repo))
    want = hashlib.sha256((repo / "models" / "e1" / "best.pt").read_bytes()).hexdigest()
    for rec in recs.values():
        assert rec["config"]["checkpoint_sha256"] == want and rec["config"]["e1_run_id"] == E1_ID
        assert "secondary" in rec["models"]["e1_masked"]
        assert rec["measurements"]["values"]["batch_seconds_p50"] > 0
    assert check("E2", root=repo, split="field_dev") == [] and check("E2", root=repo, split="lab_dev") == []


def test_real_photo_energy_is_withheld_until_masks_are_judged_reliable(repo, cfg):
    recs = records(run(cfg, repo))
    assert "energy_outside_leaf_p50" not in recs["field_dev"]["measurements"]["values"]
    assert "not reported" in recs["field_dev"]["notes"]
    lab = recs["lab_dev"]["measurements"]["values"]
    assert 0 <= lab["energy_outside_leaf_p50"] <= 1 and 0 <= lab["hsv_vs_segmented_iou_p50"] <= 1
    assert (repo / "results" / "e2-gradcam" / "field" / "energy.csv").is_file()


def test_reliable_field_masks_report_energy(repo, cfg):
    cfg["field_masks_reliable"] = True
    recs = records(run(cfg, repo))
    assert "energy_outside_leaf_p50" in recs["field_dev"]["measurements"]["values"]
    assert check("E2", root=repo, split="field_dev") == []


def test_overlays_and_sheets_are_written_and_sheets_are_kept(repo, cfg):
    run(cfg, repo)
    out = repo / "results" / "e2-gradcam"
    assert list((out / "lab").glob("*/*.jpg")) and list((out / "field").glob("*/*.jpg"))
    sheet = pd.read_csv(repo / "results" / "e2-mistakes.csv", dtype=str, keep_default_na=False)
    assert list(sheet.columns) == review.MISTAKE_COLUMNS and set(sheet["category"]) <= {""}
    assert len(pd.read_csv(repo / "results" / "e2-mask-check.csv")) == 2
    (repo / "results" / "e2-mistakes.csv").write_text("kept")
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "results")
    run(cfg, repo)
    assert (repo / "results" / "e2-mistakes.csv").read_text() == "kept"


# ---- masks and the by-hand sheets ---------------------------------------------------------------


def test_segmented_mask_matches_by_name_and_falls_back_to_the_source_id(tmp_path):
    lab = tmp_path / "raw" / "lab" / "K"
    seg = tmp_path / "raw" / "lab_segmented" / "K"
    lab.mkdir(parents=True)
    seg.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(lab / "u1___Rust 1.JPG")
    Image.new("RGB", (8, 8)).save(lab / "Rust 2.JPG")
    seg_img = Image.new("RGB", (8, 8))
    seg_img.paste((0, 200, 0), (0, 0, 4, 8))
    seg_img.save(seg / "u1___Rust 1_final_masked.png")
    seg_img.save(seg / "u2___Rust 2_final_masked.png")
    assert segmented_path(lab / "u1___Rust 1.JPG").name == "u1___Rust 1_final_masked.png"
    assert segmented_path(lab / "Rust 2.JPG").name == "u2___Rust 2_final_masked.png"
    assert segmented_path(lab / "Rust 3.JPG") is None
    mask = segmented_mask(lab / "Rust 2.JPG", 8)
    assert mask.dtype == bool and mask[:, :4].all() and not mask[:, 4:].any()
    assert iou(mask, mask) == 1.0


def test_hsv_mask_keeps_green_and_drops_grey_dark_and_red():
    img = Image.new("RGB", (4, 1))
    for x, color in enumerate([(30, 160, 30), (128, 128, 128), (5, 20, 5), (180, 40, 40)]):
        img.putpixel((x, 0), color)
    assert hsv_mask(img, 4)[0].tolist() == [True, False, False, False]


def test_mistake_selection_is_reproducible_and_counts_only_what_was_viewed():
    probs = np.array([[0.9, 0.1]] * 6 + [[0.1, 0.9]] * 6)
    labels = np.array([0, 1] * 6)
    preds = Predictions([f"p{i:02d}" for i in range(12)], labels, probs)
    one = review.select_mistakes(preds, ["a", "b"], 4, seed=1)
    two = review.select_mistakes(preds, ["a", "b"], 4, seed=1)
    assert one.equals(two) and len(one) == 4 and (one["label"] != one["predicted"]).all()
    assert len(review.select_mistakes(preds, ["a", "b"], 99, seed=1)) == 6
    one.loc[0, "category"] = "blur"
    assert review.mistake_counts(one) == {**{c: 0 for c in review.CATEGORIES}, "blur": 1, "not viewed": 3}
    one.loc[1, "category"] = "smudge"
    with pytest.raises(ValueError, match="unknown categories"):
        review.mistake_counts(one)


def test_mask_verdict_needs_every_sampled_mask_judged():
    sheet = review.select_mask_sample([f"p{i}" for i in range(5)], 5, seed=0)
    assert review.mask_verdict(sheet, 0.8)["reliable"] is False
    sheet["good"] = ["yes", "yes", "yes", "yes", "no"]
    assert review.mask_verdict(sheet, 0.8) == {
        "sampled": 5, "judged": 5, "good": 4, "share_good": 0.8, "min_good": 0.8, "reliable": True}
    sheet["good"] = ["yes", "no", "no", "yes", "no"]
    assert review.mask_verdict(sheet, 0.8)["reliable"] is False


def test_report_says_when_masks_are_not_reliable(repo, cfg):
    run(cfg, repo)
    text = report(cfg, repo).read_text()
    assert "NOT shown to be reliable" in text and "not viewed" in text and "![" not in text


# ---- the real backbone, untrained, on CPU -------------------------------------------------------


def test_gradcam_on_convnext_tiny_last_stage():
    torch.manual_seed(0)
    model = build_model("convnext_tiny", 38, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    cam, logits = grad_cam(model, x, target_layer(model, "convnext_tiny"))
    assert cam.shape == (2, 224, 224) and logits.shape == (2, 38)
    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert all(np.ptp(c) > 0 for c in cam)
    assert all(p.requires_grad for p in model.parameters())


def test_runner_smoke_with_convnext_tiny(repo, cfg):
    torch.manual_seed(0)
    names = [A, B] + [f"Other___{i}" for i in range(36)]
    model = build_model("convnext_tiny", 38, pretrained=False)
    train_cfg = TrainConfig(manifest="lab", out_dir="x", model="convnext_tiny", epochs=1)
    torch.save(
        {"model": model.state_dict(), "class_map": {n: i for i, n in enumerate(names)},
         "config": train_cfg.to_dict(), "best_metric": BEST, "epoch": 1},
        repo / "models" / "e1" / "best.pt",
    )
    recs = records(run(cfg, repo))
    for rec in recs.values():
        assert rec["config"]["model"] == "convnext_tiny" and len(rec["config"]["class_names"]) == 38
        assert len(rec["config"]["outputs_without_shared_class"]) == 36
    assert recs["lab_dev"]["measurements"]["values"]["energy_outside_leaf_p50"] >= 0
    assert check("E2", root=repo) == []
