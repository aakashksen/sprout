import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sprout.eval import metrics as M
from sprout.eval.check_run import check
from sprout.eval.records import scoring_part, split_info
from sprout.experiments import e5_confidence as e5
from sprout.serve import gate as gate_mod
from sprout.serve.calibrate import (
    apply_temperature,
    choose_threshold,
    decide,
    fit_temperature,
)
from sprout.serve.gate import GateSettings, green_ratio, is_plant_image


def overconfident(n=600, k=5, seed=0):
    """Logits that are right 70% of the time but written with a scale of 6, so they claim far more."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < 0.7, labels, (labels + rng.integers(1, k, n)) % k)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), pred] += 4
    return logits * 3, labels


def test_temperature_reduces_ece_on_overconfident_fixture():
    logits, labels = overconfident()
    t = fit_temperature(logits, labels)
    before = M.ece(apply_temperature(logits, 1.0), labels)
    after = M.ece(apply_temperature(logits, t), labels)
    assert t > 1.5
    assert after < before / 2
    assert (apply_temperature(logits, t).argmax(1) == logits.argmax(1)).all()


def test_unsure_below_threshold():
    probs = np.array([0.5, 0.3, 0.2])
    assert decide(probs, 0.6).status == "unsure"
    assert decide(probs, 0.5).status == "answered"
    assert decide(probs, 0.5).confidence == 0.5


def test_top_k_is_limited():
    probs = np.array([0.30, 0.25, 0.2, 0.15, 0.1, 0.0])
    d = decide(probs, 0.9)
    assert [i for i, _ in d.top_k] == [0, 1, 2]
    low = decide(np.array([0.85, 0.11, 0.09, 0.05]), 0.5)
    assert [i for i, _ in low.top_k] == [0, 1]  # 0.10 and below are left out
    assert all(p > 0.10 for _, p in low.top_k)


def test_choose_threshold_rule():
    # rows below 0.75 confidence are wrong, so the best threshold that keeps half the rows is the median
    conf = np.linspace(0.5, 1.0, 100)
    probs = np.stack([conf, 1 - conf], axis=1)
    labels = np.where(conf > 0.75, 0, 1)
    t = choose_threshold(probs, labels, min_coverage=0.5)
    assert 0.75 < t <= conf[50]
    assert M.selective_metrics(probs, labels, t)["coverage"] >= 0.5
    assert choose_threshold(probs, labels, min_coverage=1.0) == conf.min()


def image(color, size=32):
    return Image.new("RGB", (size, size), color)


def test_green_rule_follows_the_old_definition():
    assert green_ratio(image((40, 160, 50))) == 1.0
    assert green_ratio(image((190, 160, 40))) == 0.0  # yellow
    assert green_ratio(image((120, 80, 40))) == 0.0  # brown
    half = image((120, 80, 40))
    half.paste(image((40, 160, 50), 16), (0, 0))
    assert green_ratio(half) == 0.25


# ---- the experiment, end to end on a synthetic world (no GPU, no real photos) ----

CLASSES = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Early_blight"]
GREEN, YELLOW, BROWN, BLUE, GRAY = (40, 160, 50), (190, 160, 40), (120, 80, 40), (30, 60, 200), (128, 128, 128)
NF = 120  # field_dev photos
NN = 120  # not_plant_dev photos
E4_ID = "E4-20260101T000000000000Z"


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def probs_row(rng, k, top, conf):
    p = rng.dirichlet(np.ones(k)) * (1 - conf)
    p[top] += conf
    return p / p.sum()


def save_photo(root, rel, color):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(root / rel)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5world")
    rng = np.random.default_rng(1)
    field, nonplant, preds, np_probs = [], [], [], []
    for i in range(NF):
        label = i % 3
        color = GREEN if i % 4 == 0 else (YELLOW, BROWN)[i % 2]  # three quarters are yellow or brown leaves
        rel = f"datasets/raw/f/{i}.png"
        save_photo(root, rel, color)
        field.append((rel, CLASSES[label], f"g{i // 2}", "field_dev", f"s{i}", "plantdoc"))
        guess = label if rng.random() < 0.7 else (label + 1) % 3
        row = probs_row(rng, 3, guess, rng.uniform(0.7, 0.99))
        preds.append({"path": rel, "label": label, "probs": [float(v) for v in row]})
    for i in range(NN):
        color = GREEN if i % 5 == 0 else (BLUE, GRAY)[i % 2]  # a fifth are green walls, lawns and the like
        rel = f"datasets/raw/n/{i}.png"
        save_photo(root, rel, color)
        nonplant.append((rel, "not_plant", f"n{i}", "not_plant_dev", f"z{i}"))
        np_probs.append({"path": rel, "probs": [float(v) for v in rng.dirichlet(np.ones(3) * 3)]})
    for i in range(10):  # test rows whose photos do not exist: reading one would fail
        field.append((f"datasets/raw/f/test{i}.png", CLASSES[i % 3], f"t{i}", "field_test", f"t{i}", "plantdoc"))
        nonplant.append((f"datasets/raw/n/test{i}.png", "not_plant", f"nt{i}", "not_plant_test", f"nt{i}"))
    fm = root / "datasets/manifests"
    fm.mkdir(parents=True)
    (fm / "field.csv").write_text("path,label,group,split,sha1,source\n" + "\n".join(",".join(r) for r in field) + "\n")
    (fm / "not_plant.csv").write_text("path,label,group,split,sha1\n" + "\n".join(",".join(r) for r in nonplant) + "\n")
    runs = root / "results/runs"
    runs.mkdir(parents=True)
    (runs / f"{E4_ID}.convnext_joint.predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds))
    (runs / f"{E4_ID}.json").write_text(json.dumps({
        "split": "field_dev", "dataset_hash": split_info("field_dev", root).hash, "config": {"class_names": CLASSES},
        "predictions": {"convnext_joint": f"results/runs/{E4_ID}.convnext_joint.predictions.jsonl"}}))
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/e5.json").read_text())
    cfg["model"]["e4_record"] = E4_ID
    cfg["num_workers"] = 0
    (root / "configs").mkdir()
    (root / "configs/e5.json").write_text(json.dumps(cfg))
    (root / "results/e5").mkdir()
    (root / "results/e5/not_plant_dev.probs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in np_probs))
    (root / "orchestration").mkdir()
    (root / "orchestration/tasks.json").write_text("{}")
    sh(root, "init", "-q")
    sh(root, "config", "user.email", "t@t")
    sh(root, "config", "user.name", "t")
    cfg = e5.load_config(root / "configs/e5.json")
    log = e5.StageLog(cfg, root)
    e5.stage_green(cfg, root, log)
    e5.stage_fit(cfg, root, log)
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "inputs")
    paths = e5.stage_record(cfg, root, log)
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "records")
    return root, cfg, paths


def test_scoring_part_keeps_groups_whole_and_is_repeatable():
    groups = [f"g{i // 3}" for i in range(90)]
    a = scoring_part(groups, 0, 50)
    assert (a == scoring_part(groups, 0, 50)).all()
    assert not (a == scoring_part(groups, 1, 50)).all()
    for g in set(groups):
        assert len({a[i] for i, x in enumerate(groups) if x == g}) == 1
    assert a.sum() == 45


def test_threshold_is_chosen_without_test_data(world):
    root, cfg, _ = world
    chosen = json.loads((root / "results/e5/chosen.json").read_text())
    field = e5.load_field(cfg, root)
    fit = ~field.scoring(cfg)
    cal = e5.calibrate(field.probs, chosen["temperature"])
    assert chosen["abstain_threshold"] == round(choose_threshold(cal[fit], field.labels[fit], 0.5), 6)
    assert chosen["fit_rows"]["field_dev"] == int(fit.sum()) < NF
    seen = {json.loads(x)["path"] for x in (root / "results/e5/green_ratio.jsonl").read_text().splitlines()}
    assert len(seen) == NF + NN and not any("test" in p for p in seen)
    for split in ("field_test", "not_plant_test", "field_train"):
        with pytest.raises(ValueError):
            e5.require_dev(split)
    (root / "configs/bad.json").write_text(json.dumps({**cfg, "field_split": "field_test"}))
    with pytest.raises(ValueError):
        e5.load_config(root / "configs/bad.json")
    (root / "configs/bad.json").unlink()


def test_yellow_brown_leaf_fixture_is_not_rejected_by_chosen_gate(world, monkeypatch):
    root, cfg, _ = world
    chosen = json.loads((root / "results/e5/chosen.json").read_text())
    assert chosen["gates"]["green_rule"]["fit_plant_rejection"] > 0.5  # most of the fixture leaves are not green
    assert not chosen["gates"]["green_rule"]["eligible"]
    assert chosen["gate"]["chosen"] != "green_rule"
    monkeypatch.setattr(gate_mod, "SETTINGS_PATH", root / "results/e5/chosen.json")
    sure = np.array([0.93, 0.05, 0.02])
    for color in (YELLOW, BROWN):
        assert is_plant_image(image(color), sure)
    # the old rule alone would have thrown the same photo out
    assert not GateSettings.from_dict(chosen["gates"]["green_rule"]["settings"]).accepts(
        green=green_ratio(image(YELLOW)), probs=sure)


def test_green_non_plant_fixture_is_measured_not_silently_passed(world, monkeypatch):
    root, cfg, paths = world
    rec = json.loads(paths[1].read_text())
    assert set(rec["metrics"]) == {"green_rule", "confidence", "confidence_ood"}
    # a fifth of the fixture's non-plants are green, and the green rule lets every one of them in
    assert rec["metrics"]["green_rule"]["not_plant_acceptance_rate"] >= 0.15
    rows = [json.loads(x) for x in (root / rec["predictions"]["green_rule"]).read_text().splitlines()]
    green_rows = [r for r in rows if r["green_ratio"] == 1.0]
    assert green_rows and all(r["accepted"] for r in green_rows)
    assert rec["n"] == len(rows) and rec["subset"] == "group_part:0:50"
    # the chosen gate rejects a green photo the model is unsure about, which the green rule cannot
    monkeypatch.setattr(gate_mod, "SETTINGS_PATH", root / "results/e5/chosen.json")
    assert not is_plant_image(image(GREEN), np.array([0.4, 0.35, 0.25]))


def test_records_pass_check_run_and_state_their_method(world):
    root, cfg, paths = world
    assert check("E5", root=root) == []
    rec = json.loads(paths[0].read_text())
    assert rec["split"] == "field_dev" and rec["n"] < NF
    assert "seen-source" in rec["notes"] and "selected on field_dev" in rec["notes"]
    m = rec["metrics"]
    assert m["calibrated"]["ece"] < m["uncalibrated"]["ece"]
    t = rec["config"]["abstain_threshold"]
    assert f"coverage@{t:.6f}" in m["calibrated"] and "confident_error_rate@0.0" in m["calibrated"]
    assert "plant_rejection_rate" in m["confidence"] and "end_to_end_accuracy" in m["confidence"]


def tamper(world, split_index, system, edit):
    root, _, paths = world
    rec = json.loads(paths[split_index].read_text())
    p = root / rec["predictions"][system]
    original = p.read_text()
    rows = [json.loads(x) for x in original.splitlines()]
    edit(rows)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        return check("E5", root=root)
    finally:
        p.write_text(original)


def test_check_run_rejects_flags_that_do_not_follow_from_confidences(world):
    def flip(rows):
        rows[0]["accepted"] = not rows[0]["accepted"]

    for split_index, system in ((0, "confidence"), (0, "confidence_ood"), (0, "green_rule"), (1, "confidence_ood")):
        problems = tamper(world, split_index, system, flip)
        assert any("accepted does not follow from the saved scores and thresholds" in p for p in problems), system

    def lower_threshold(rows):  # claim a lower bar than the one the gate used
        for r in rows:
            r["threshold"] = 0.0

    assert any("accepted does not follow" in p for p in tamper(world, 1, "confidence", lower_threshold))

    def drop_pair(rows):
        for r in rows:
            del r["threshold"]

    assert any("numeric" in p for p in tamper(world, 1, "confidence", drop_pair))
    assert check("E5", root=world[0]) == []


def test_probs_stage_scores_not_plant_photos_from_a_checkpoint(world, tmp_path):
    import torch

    from sprout.preprocess import preprocess
    from sprout.train.checkpoint import save_checkpoint
    from sprout.train.config import TrainConfig
    from sprout.train.loop import build_model

    root, cfg, _ = world
    torch.manual_seed(0)
    model = build_model("tiny_cnn", 3)
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(tmp_path / "best.pt", model=model, optimizer=opt, scheduler=torch.optim.lr_scheduler.StepLR(opt, 1),
                    scaler=torch.amp.GradScaler("cuda", enabled=False), epoch=1, best_metric=0.0,
                    class_map={c: i for i, c in enumerate(CLASSES)}, config=TrainConfig(manifest="x", out_dir="y", model="tiny_cnn"))
    local = {**cfg, "out_dir": str(tmp_path / "out"), "model": {**cfg["model"], "checkpoint": str(tmp_path / "best.pt")}}
    path = e5.stage_probs_not_plant(local, root, e5.StageLog(local, root), device=torch.device("cpu"))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == NN and all(abs(sum(r["probs"]) - 1) < 1e-5 for r in rows)
    with Image.open(root / rows[0]["path"]) as img:
        expected = torch.softmax(model.eval()(preprocess(img)[None]), dim=1)[0].detach().numpy()
    assert np.allclose(rows[0]["probs"], expected, atol=1e-5)
    assert "probs-not-plant: done" in (tmp_path / "out" / "stages.log").read_text()


def test_datasets_used_with_workers_can_be_pickled(world):
    import pickle

    root, cfg, _ = world
    photos = e5.Photos(["datasets/raw/n/0.png"], root)  # worker processes on Windows receive it pickled
    assert pickle.loads(pickle.dumps(photos))[0].shape == (3, 224, 224)
    assert len(pickle.loads(pickle.dumps(photos))) == 1


def test_stage_log_does_not_dirty_the_tree(world):
    root, cfg, _ = world
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text().splitlines()
    assert f"{cfg['out_dir']}/stages.log" in ignore

