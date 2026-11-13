import json
import subprocess

import pandas as pd
import pytest
import torch
from PIL import Image

import sprout.data.splits as splits
from sprout.data.splits import load_manifest
from sprout.eval.check_run import check
from sprout.experiments.e1_lab import (
    BACKGROUND,
    LEGACY_LABEL,
    legacy_class_names,
    map_legacy,
    run,
)
from sprout.train.loop import build_model

REPO = splits.MANIFEST_DIR.parents[1]
APPLICATION = REPO / "legacy" / "PlantProject" / "application.py"
CLASSES = ["Apple___healthy", "Cherry_(including_sour)___healthy"]
LEGACY_NAMES = legacy_class_names(APPLICATION)


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    mdir = root / "datasets" / "manifests"
    mdir.mkdir(parents=True)
    rows = []
    for i in range(18):
        label = CLASSES[i % 2]
        color = (200, 30, 30) if i % 2 == 0 else (30, 30, 200)
        split = "lab_train" if i < 8 else "lab_dev" if i < 14 else "lab_test"
        p = tmp_path / "img" / f"{i}.png"
        p.parent.mkdir(exist_ok=True)
        if split != "lab_test":  # the test images do not exist: reading one would fail the run
            Image.new("RGB", (64, 64), color).save(p)
        rows.append((str(p), label, f"g{i}", split, f"sha{i}"))
    pd.DataFrame(rows, columns=["path", "label", "group", "split", "sha1"]).to_csv(mdir / "lab.csv", index=False)
    (root / ".gitignore").write_text("models/\nmlruns/\n")
    sh(root, "init", "-q")
    sh(root, "config", "user.email", "t@t")
    sh(root, "config", "user.name", "t")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    return root


@pytest.fixture
def cfg(tmp_path):
    torch.manual_seed(0)
    old = tmp_path / "legacy.pth"
    torch.save(build_model("tiny_cnn", len(LEGACY_NAMES), pretrained=False).state_dict(), old)
    return {
        "experiment": "E1",
        "manifest": "lab",
        "split": "lab_dev",
        "out_dir": "models/e1",
        "train": {"model": "tiny_cnn", "seed": 1, "lr": 0.05, "batch_size": 4, "epochs": 2, "transforms": ["hflip"]},
        "legacy": {
            "checkpoint": str(old),
            "application": str(APPLICATION),
            "model": "tiny_cnn",
            "sensible_accuracy": 0.5,
        },
    }


def test_runner_on_tiny_fixture_writes_a_valid_record(repo, cfg):
    path = run(cfg, repo)
    rec = json.loads(path.read_text())
    assert rec["experiment"] == "E1" and rec["split"] == "lab_dev" and rec["n"] == 6
    assert rec["dirty"] is False
    assert set(rec["predictions"]) == {"e1", "legacy"}
    assert rec["config"]["class_names"] == sorted(CLASSES)
    assert rec["attempts"][0]["setting"]["lr"] == 0.05
    assert (repo / "mlruns").is_dir()
    assert check("E1", root=repo, split="lab_dev") == []


def test_legacy_result_is_labeled(repo, cfg):
    rec = json.loads(run(cfg, repo).read_text())
    assert rec["models"]["legacy"] == LEGACY_LABEL
    assert rec["config"]["legacy"]["label"] == LEGACY_LABEL
    assert LEGACY_LABEL in rec["notes"]
    assert BACKGROUND in rec["config"]["legacy"]["unmapped_dropped"]
    assert len(rec["config"]["legacy"]["mapping"]) == 2
    assert "dropped" in rec["config"]["legacy"]["background_handling"]


def test_runner_refuses_a_dirty_tree(repo, cfg):
    (repo / "datasets" / "extra.txt").write_text("x")
    with pytest.raises(RuntimeError, match="uncommitted"):
        run(cfg, repo)
    assert not (repo / "models").exists()


def test_legacy_classes_map_onto_38_shared_classes():
    official = sorted(load_manifest("lab")["label"].unique())
    assert len(LEGACY_NAMES) == 39 and len(official) == 38
    mapping, dropped = map_legacy(LEGACY_NAMES, official)
    assert dropped == [BACKGROUND]
    assert len(mapping) == 38 and sorted(mapping.values()) == official
    assert mapping["Cherry___healthy"] == "Cherry_(including_sour)___healthy"
    assert mapping["Corn___Common_rust"] == "Corn_(maize)___Common_rust_"


def test_note_is_appended_to_the_record_notes(repo, cfg):
    rec = json.loads(run(cfg, repo, note="Restore of the lost checkpoint; same config and seed.").read_text())
    assert LEGACY_LABEL in rec["notes"]
    assert rec["notes"].endswith("Restore of the lost checkpoint; same config and seed.")
