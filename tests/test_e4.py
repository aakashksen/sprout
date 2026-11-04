import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import sprout.data.splits as splits
from sprout.eval.check_run import check
from sprout.experiments import e4_improve as e4
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
    mdir = tmp_path / "repo" / "datasets" / "manifests"
    mdir.mkdir(parents=True)
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


def test_mask_store_saves_to_the_exact_path_it_loads_from(tmp_path):
    m = np.zeros((320, 320), bool)
    m[5:9, 5:9] = True
    MaskStore({"a.jpg": m}).save(tmp_path / "masks")  # np.savez would have written masks.npz
    assert np.array_equal(MaskStore.load(tmp_path / "masks").get("a.jpg"), m)
    assert [p.name for p in tmp_path.iterdir()] == ["masks"]


def test_joint_run_trains_on_real_and_lab_selects_on_field_dev(world):
    from sprout.train.checkpoint import load_checkpoint
    from sprout.train.loop import train

    cfg = _cfg(world, lab_cap=4, epochs=2, seed=1, lr=0.05)
    out = train(cfg)
    assert out.epochs_done == 2 and len(out.history) == 2
    ckpt = load_checkpoint(out.best_path)
    assert ckpt["config"].select_split == "field_dev" and ckpt["config"].lab_cap == 4
    assert sorted(p.name for p in (world / "run").iterdir()) == ["best.pt", "latest.pt"]


# ---------------------------------------------------------------------------------------------------------------
# the runner

BASELINE_ID = "E3-20261103T000000000000Z"
BASELINES = ("e1", "dinov2_small", "frozen_convnext", "xgboost")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "e4.json"


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(world, monkeypatch):
    """A git repo holding the manifests, a stand-in for the saved E3 field_dev rows, and a tiny E1 checkpoint."""
    from sprout.train.loop import train

    root = world / "repo"
    (root / ".gitignore").write_text("models/\n")
    field = pd.read_csv(root / "datasets" / "manifests" / "field.csv")
    dev = field[field["split"] == "field_dev"]
    rng = np.random.default_rng(0)
    runs = root / "results" / "runs"
    runs.mkdir(parents=True)
    predictions = {}
    for system in BASELINES:
        p = runs / f"{BASELINE_ID}.{system}.predictions.jsonl"
        rows = []
        for path, label in zip(dev["path"], dev["label"], strict=False):
            probs = rng.dirichlet(np.ones(2))
            rows.append({"path": path, "label": CLASSES.index(label), "probs": list(map(float, probs))})
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        predictions[system] = p.relative_to(root).as_posix()
    (runs / f"{BASELINE_ID}.json").write_text(
        json.dumps({"predictions": predictions, "config": {"class_names": CLASSES}}))
    train(TrainConfig(manifest="lab", out_dir=str(root / "models" / "e1"), model="tiny_cnn", epochs=1, batch_size=4))
    sh(root, "init", "-q")
    sh(root, "config", "user.email", "t@t")
    sh(root, "config", "user.name", "t")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    # the stand-in lab photos have no official segmented copy: call the centre the leaf for the Grad-CAM energy
    import sprout.experiments.e2_field as e2

    monkeypatch.setattr(e2, "segmented_mask", lambda path, size: np.pad(np.ones((size // 2, size // 2), bool), size // 4))
    return root


@pytest.fixture
def e4cfg():
    cfg = e4.load_config(CONFIG)
    cfg.update(num_workers=0, min_free_disk_gb=0, lab_cap=5, cam_sample=4)
    cfg["baselines"] = {**cfg["baselines"], "record": BASELINE_ID}
    for arm in cfg["arms"]:
        if arm["kind"] == "train":
            arm.update(model="tiny_cnn", lr=0.05, epochs=2, batch_size=4)
            arm.pop("unfreeze_blocks", None)
            arm.pop("backbone_lr", None)
    return cfg


def _record(path):
    return json.loads(Path(path).read_text())


def test_config_fixes_the_arms_and_the_rules_before_any_run():
    cfg = e4.load_config(CONFIG)
    assert [a["name"] for a in cfg["arms"] if a["kind"] != "heldout"] == [
        "convnext_joint", "convnext_strong_aug", "convnext_background", "dinov2_head", "dinov2_finetune",
        "best_with_crop"]
    assert [a["held_out_source"] for a in cfg["arms"] if a["kind"] == "heldout"] == ["plantdoc", "plantseg"]
    assert cfg["lab_cap"] == 60 and cfg["lab_cap_reason"] and cfg["num_workers"] <= 4
    assert cfg["dropped"][0]["name"] == "convnext_aug_background"
    assert "macro-F1" in cfg["rules"]["best_arm"] and "optimistic" in cfg["rules"]["best_arm"]


def test_attempts_lists_each_run(repo, e4cfg, monkeypatch):
    # the stand-in photos are plain red and blue, so the colour threshold finds no leaf and arm 3 must be refused
    for name in ("convnext_joint", "convnext_strong_aug", "convnext_background", "dinov2_head"):
        path = e4.run_arm(e4cfg, name, repo)
        assert (path is None) == (name == "convnext_background")
    notrun = json.loads((repo / "models" / "e4" / "not_run" / "convnext_background.json").read_text())
    assert "believable leaf mask" in notrun["reason"]

    first = _record(sorted((repo / "results" / "runs").glob("E4-*[0-9]Z.json"))[0])
    assert first["config"]["arm"] == "convnext_joint" and first["subset"] == "shared_classes"
    assert first["config"]["trained_on_plantdoc"] is True
    assert set(first["predictions"]) == {*BASELINES, "convnext_joint"}
    assert {e["arm"]: e["status"] for e in first["attempts"]}["convnext_joint"] == "run"
    assert {e["arm"]: e["status"] for e in first["attempts"]}["convnext_background"] == "pending"

    # arm 5 follows the stated rule, read from the two records
    results = e4.arm_results(e4cfg, repo)
    expect_run = results["dinov2_head"]["macro_f1"] > results["convnext_joint"]["macro_f1"]
    path = e4.run_arm(e4cfg, "dinov2_finetune", repo)
    assert (path is not None) == expect_run
    if not expect_run:
        assert "not above" in json.loads((repo / "models/e4/not_run/dinov2_finetune.json").read_text())["reason"]

    for name in ("heldout_plantdoc", "heldout_plantseg"):
        rec = _record(e4.run_arm(e4cfg, name, repo))
        held = name.removeprefix("heldout_")
        assert rec["subset"] == f"source:{held}" and rec["n"] == 4
        assert rec["config"]["trained_on_plantdoc"] is (held != "plantdoc")
        assert rec["config"]["train"]["held_out_source"] == held

    monkeypatch.setattr(e4, "load_segmenter", lambda path, device: object())
    box = np.zeros((64, 64), bool)
    box[10:50, 5:40] = True
    monkeypatch.setattr(e4, "predict_mask", lambda model, image: box)
    (repo / "models" / "e10").mkdir(parents=True)
    (repo / "models" / "e10" / "best.pt").write_bytes(b"x")
    final = _record(e4.run_arm(e4cfg, "best_with_crop", repo))
    assert final["config"]["crop"]["e10_record"] and final["config"]["train"]["crop_masks"]
    assert final["config"]["base_arm"] in ("convnext_joint", "convnext_strong_aug", "dinov2_head", "dinov2_finetune")
    by_arm = {e["arm"]: e for e in final["attempts"]}
    assert set(by_arm) == {a["name"] for a in e4cfg["arms"]} | {"convnext_aug_background"}
    assert by_arm["convnext_aug_background"]["status"] == "dropped"
    assert by_arm["convnext_background"]["status"] == "not_run" and by_arm["convnext_background"]["reason"]
    assert by_arm["best_with_crop"]["status"] == "run" and by_arm["best_with_crop"]["setting"]["crop"] is True
    assert by_arm["heldout_plantseg"]["status"] == "run" and by_arm["dinov2_head"]["status"] == "run"
    assert by_arm["dinov2_finetune"]["status"] == ("run" if expect_run else "not_run")
    assert not [e for e in final["attempts"] if e["status"] == "pending"]
    assert check("E4", root=repo, split="field_dev") == []


def test_standard_metrics_and_measurements_are_recorded(repo, e4cfg):
    rec = _record(e4.run_arm(e4cfg, "convnext_joint", repo))
    for system, m in rec["metrics"].items():
        assert {"accuracy", "macro_f1", "ece", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy",
                "plant_given_top_1", "plant_given_top_3"} <= set(m), system
    values = rec["measurements"]["values"]
    assert "epoch_seconds_convnext_joint_p50" in values
    assert "energy_outside_leaf_convnext_joint_p50" in values and "energy_outside_leaf_e1_p50" in values
    assert {(c["a"], c["b"], c["metric"]) for c in rec["comparisons"]} == {
        ("convnext_joint", b, m) for b in BASELINES for m in ("accuracy", "macro_f1")}
    assert "optimistic" in rec["notes"] and "development number" in rec["notes"] and "PlantDoc" in rec["notes"]
    assert (repo / "models/e4/convnext_joint/epochs.log").read_text().count("epoch ") == 2
    assert sorted(p.name for p in (repo / "models/e4/convnext_joint").glob("*.pt")) == ["best.pt", "latest.pt"]


def test_interrupted_run_writes_no_record_and_resume_lists_both_attempts(repo, e4cfg, monkeypatch):
    import sprout.train.loop as loop

    real, calls = loop._evaluate, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt  # stands for a killed process
        return real(*a, **k)

    monkeypatch.setattr(loop, "_evaluate", flaky)
    with pytest.raises(KeyboardInterrupt):
        e4.run_arm(e4cfg, "convnext_joint", repo)
    assert not list((repo / "results" / "runs").glob("E4-*"))  # a partial run is no result
    with pytest.raises(FileExistsError):  # without --resume the checkpoints are protected
        e4.run_arm(e4cfg, "convnext_joint", repo)
    rec = _record(e4.run_arm(e4cfg, "convnext_joint", repo, resume=True))
    entry = {e["arm"]: e for e in rec["attempts"]}["convnext_joint"]
    assert [a["status"] for a in entry["attempts"]] == ["interrupted", "complete"]
    assert entry["attempts"][0]["epochs_in_latest_checkpoint"] == 1
    assert entry["attempts"][1]["resumed_from_latest"] is True
    assert len(entry["history"]) == 1 and entry["epochs_done"] == 2
    assert check("E4", root=repo, split="field_dev") == []


def test_arms_that_read_earlier_results_refuse_to_guess(repo, e4cfg):
    with pytest.raises(e4.Blocked):
        e4.run_arm(e4cfg, "dinov2_finetune", repo)
    with pytest.raises(e4.Blocked):
        e4.run_arm(e4cfg, "best_with_crop", repo)
    with pytest.raises(e4.Blocked):
        e4.run_arm(e4cfg, "convnext_strong_aug", repo)  # its reference arm has no record
    assert not (repo / "models" / "e4" / "not_run").exists()
    # refused before any training: no checkpoints, no attempt entry
    assert not (repo / "models" / "e4" / "convnext_strong_aug" / "latest.pt").exists()
    assert not (repo / "models" / "e4" / "convnext_strong_aug" / "attempts.json").exists()


def test_best_arm_is_the_highest_macro_f1_and_ties_go_to_the_lower_number(e4cfg):
    res = {"convnext_joint": {"macro_f1": 0.3}, "convnext_strong_aug": {"macro_f1": 0.4},
           "dinov2_head": {"macro_f1": 0.4}, "heldout_plantdoc": {"macro_f1": 0.9}}
    assert e4.pick_best(e4cfg, res) == "convnext_strong_aug"
    ok, why = e4.arm5_rule({"dinov2_head": {"macro_f1": 0.4}, "convnext_joint": {"macro_f1": 0.4}})
    assert ok is False and "not above" in why


def test_worker_limit_and_disk_guard(tmp_path, e4cfg):
    (tmp_path / "c.json").write_text(json.dumps({**e4cfg, "num_workers": 9}))
    with pytest.raises(ValueError, match="at most 8"):
        e4.load_config(tmp_path / "c.json")
    with pytest.raises(RuntimeError, match="free"):
        e4.check_disk(tmp_path, 10**9)
