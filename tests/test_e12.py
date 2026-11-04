from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import sprout.data.splits as splits
from sprout.train.config import TrainConfig
from sprout.train.dataset import build_datasets, make_class_map
from sprout.train.joint import load_joint_manifest, select_rows

CLASSES = ["Apple___healthy", "Cherry_(including_sour)___healthy"]
COLUMNS = ["path", "label", "group", "split", "sha1"]
REAL = Path(__file__).resolve().parents[1] / "datasets" / "manifests"
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


def _real_manifests_hold_out_cleanly(monkeypatch):
    """Real columns: lab.csv, field.csv and extra_train.csv as committed. Reads manifests only, no image."""
    monkeypatch.setattr(splits, "MANIFEST_DIR", REAL)
    field = pd.read_csv(REAL / "field.csv")
    extra = pd.read_csv(REAL / "extra_train.csv")
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


def test_held_out_source_is_absent_from_training_and_selection(world, monkeypatch):
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
    _real_manifests_hold_out_cleanly(monkeypatch)


# ---------------------------------------------------------------------------------------------------------------
# the runner

import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

from sprout.data.splits import dataset_hash, load_manifest  # noqa: E402
from sprout.eval.check_run import check  # noqa: E402
from sprout.eval.records import split_info  # noqa: E402
from sprout.experiments import e4_improve as e4  # noqa: E402
from sprout.experiments import e12_more_sources as e12  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "e12.json"
E4_CONFIG = ROOT / "configs" / "e4.json"
RUNS = ("convnext_joint_extra", "heldout_plantdoc_extra", "heldout_plantseg_extra")
LOG_HEAD = "version,image_id,path,source,split,label,reason_code,note,began_as\n"


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _record(path):
    return json.loads(Path(path).read_text())


def _lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


@pytest.fixture
def repo(world):
    """A git repo with the manifests, one cleaning exclusion on field_dev, and stand-ins for E4's saved rows."""
    root = world / "repo"
    (root / ".gitignore").write_text("models/\n")
    field = pd.read_csv(root / "datasets" / "manifests" / "field.csv")
    dev = field[field["split"] == "field_dev"]
    gone = dev[(dev["source"] == "plantdoc") & (dev["label"] == CLASSES[0])].iloc[0]
    (root / "datasets" / "label_exclusions.csv").write_text(
        LOG_HEAD + f"1,{gone['sha1']},{gone['path']},plantdoc,field_dev,{gone['label']},wrong_plant,x,remove\n")
    rng = np.random.default_rng(0)
    runs = root / "results" / "runs"
    runs.mkdir(parents=True)
    cfg = json.loads(CONFIG.read_text())
    info = split_info("field_dev", root)
    for i, name in enumerate(RUNS):
        spec = next(r for r in cfg["runs"] if r["name"] == name)
        part = dev if spec["subset"] == "shared_classes" else dev[dev["source"] == spec["held_out_source"]]
        rid = f"E4-20261103T0000000000{i}0Z"
        predictions = {}
        for system in spec["compare"]:
            p = runs / f"{rid}.{system}.predictions.jsonl"
            p.write_text("".join(json.dumps({"path": path, "label": CLASSES.index(label),
                                             "probs": list(map(float, rng.dirichlet(np.ones(2))))}) + "\n"
                                 for path, label in zip(part["path"], part["label"], strict=True)))
            predictions[system] = p.relative_to(root).as_posix()
        (runs / f"{rid}.json").write_text(json.dumps({
            "run_id": rid, "split": "field_dev", "dataset_hash": info.hash, "predictions": predictions,
            "config": {"class_names": CLASSES}}))
    sh(root, "init", "-q")
    sh(root, "config", "user.email", "t@t")
    sh(root, "config", "user.name", "t")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def cfg(repo):
    c = e12.load_config(CONFIG)
    c["settings"].update(model="tiny_cnn", lr=0.05, epochs=2, batch_size=4, lab_cap=5)
    c.update(min_free_disk_gb=0)
    c["extra"].update(rows=16, dataset_hash=dataset_hash(load_manifest("extra_train")))
    c["e4"]["records"] = {n: f"E4-20261103T0000000000{i}0Z" for i, n in enumerate(RUNS)}
    return c


def test_config_is_fixed_before_any_run(monkeypatch):
    c = e12.load_config(CONFIG)
    assert c["extra"]["rows"] == 3484 and c["split"] == "field_dev" and c["clean_version"] == 1
    assert c["num_workers"]["default"] == 2 and c["num_workers"]["e4"] == 4
    assert c["decision"]["fixed_before_any_e12_run"] is True and c["decision"]["heldout_margin"] == 0.02
    assert [r["held_out_source"] for r in c["runs"]] == ["", "plantdoc", "plantseg"]
    monkeypatch.setattr(splits, "MANIFEST_DIR", REAL)
    e12.check_extra_frozen(c)  # the committed extra manifest is the frozen one


def test_extra_manifest_must_be_the_frozen_one(repo, cfg):
    e12.check_extra_frozen(cfg)
    cfg["extra"]["dataset_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen"):
        e12.run(cfg, RUNS[0], repo, num_workers=0)
    assert not (repo / "models" / "e12" / RUNS[0] / "latest.pt").exists()


def test_settings_match_e4_best_apart_from_the_extra_rows():
    c, c4 = e12.load_config(CONFIG), e4.load_config(E4_CONFIG)
    results = e4.arm_results(c4, ROOT)
    assert e4.pick_best(c4, results) == c["e4"]["arm"] == "convnext_joint"  # E4's own rule, on E4's own records
    neutral = {"out_dir", "num_workers", "extra_manifest", "train_splits", "manifest_hash"}
    for spec in c["runs"]:
        e4_arm = spec["name"].removesuffix("_extra") if spec["held_out_source"] else c["e4"]["arm"]
        mine = e12.train_config(c, spec, ROOT, 2)
        base = e4.train_config(c4, e4.resolved_spec(c4, e4_arm, results), ROOT)
        a, b = mine.to_dict(), base.to_dict()
        assert {k: v for k, v in a.items() if k not in neutral} == {k: v for k, v in b.items() if k not in neutral}
        assert a["train_splits"] == (*b["train_splits"], "extra_train") and a["extra_manifest"] == "extra_train"
        assert b["extra_manifest"] == "" and mine.lab_cap == 60 and mine.select_split == "field_dev"
        # and the settings the E4 record says it trained with
        rec = json.loads((ROOT / "results" / "runs" / f"{c['e4']['records'][spec['name']]}.json").read_text())
        saved = {k: v for k, v in rec["config"]["train"].items() if k not in neutral | {"extra_manifest"}}
        mine_json = json.loads(json.dumps(a))
        assert {k: v for k, v in mine_json.items() if k not in neutral | {"extra_manifest"}} == saved
        assert rec["config"]["train"]["num_workers"] == c["num_workers"]["e4"] and mine.num_workers == 2


def test_workers_come_from_flag_then_environment_then_the_default():
    c = e12.load_config(CONFIG)
    assert e12.resolve_workers(c, None, {}) == 2
    assert e12.resolve_workers(c, None, {"SPROUT_NUM_WORKERS": "3"}) == 3
    assert e12.resolve_workers(c, 1, {"SPROUT_NUM_WORKERS": "3"}) == 1
    for bad in (5, -1):
        with pytest.raises(ValueError, match="between 0 and 4"):
            e12.resolve_workers(c, bad, {})


def _write(repo, data):
    path = repo.parent / "c.json"
    path.write_text(json.dumps(data))
    return path


def test_field_test_is_never_read(repo, cfg, monkeypatch):
    with pytest.raises(ValueError, match="field_dev only"):
        e12.load_config(_write(repo, {**json.loads(CONFIG.read_text()), "split": "field_test"}))
    worse = json.loads(CONFIG.read_text())
    worse["settings"]["train_splits"].append("field_test")
    with pytest.raises(ValueError, match="never reads a test split"):
        e12.load_config(_write(repo, worse))
    opened = []
    real_open = Image.open

    def spy(fp, *a, **k):
        opened.append(str(fp))
        return real_open(fp, *a, **k)

    monkeypatch.setattr(Image, "open", spy)
    rec = _record(e12.run(cfg, RUNS[0], repo, num_workers=0))  # field_test and lab_test photos do not exist
    assert opened and not [p for p in opened if Path(p).name.startswith(("field_test", "lab_test"))]
    assert rec["config"]["field_test_read"] is False and rec["split"] == "field_dev"
    assert not any(Path(r["path"]).name.startswith("field_test") for r in _lines(repo / rec["predictions"][RUNS[0]]))
    assert set(pd.read_csv(REAL / "extra_train.csv")["split"]) == {"extra_train"}  # no test row in the real extras


def test_seen_run_scores_the_clean_rows_beside_e4_and_check_run_recomputes_it(repo, cfg):
    rec = _record(e12.run(cfg, RUNS[0], repo, num_workers=0))
    assert rec["subset"] == "shared_classes+clean:1" and rec["n"] == 7  # 8 field_dev photos, 1 excluded
    assert set(rec["predictions"]) == {"e1", "convnext_joint", RUNS[0]}
    for system, m in rec["metrics"].items():  # standard_image_metrics
        assert {"accuracy", "macro_f1", "ece", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy",
                "plant_given_top_1", "plant_given_top_3"} <= set(m), system
    assert {(c["a"], c["b"], c["metric"]) for c in rec["comparisons"]} == {
        (RUNS[0], b, m) for b in ("e1", "convnext_joint") for m in ("accuracy", "macro_f1")}
    device = rec["config"]["device"]
    assert device["type"] in ("cpu", "cuda") and device["name"]
    assert device["num_workers"] == 0 and device["e4_num_workers"] == 4
    assert rec["config"]["extra_rows_in_training"] == 16 and rec["config"]["train"]["extra_manifest"] == "extra_train"
    assert "development number" in rec["config"]["label"] and "optimistic" in rec["notes"]
    assert f"epoch_seconds_{RUNS[0]}_p50" in rec["measurements"]["values"]
    gone = pd.read_csv(repo / "datasets" / "label_exclusions.csv")["path"][0]
    rows = _lines(repo / rec["predictions"]["convnext_joint"])
    stand_in = [r for r in _lines(repo / "results/runs/E4-20261103T000000000000Z.convnext_joint.predictions.jsonl")
                if r["path"] != gone]
    assert len(rows) == 7 and rows == stand_in  # E4's numbers are its own saved rows, only filtered
    # every predicted field_dev photo is kept beside the checkpoint for a later refit of E5
    assert len(_lines(repo / "models/e12" / RUNS[0] / "field_dev.predictions.jsonl")) == 8
    assert (repo / "models/e12" / RUNS[0] / "epochs.log").read_text().count("epoch ") == 2
    assert check("E12", root=repo, min_n=1, split="field_dev") == []

    # tamper: a changed probability, then a changed metric, must each be caught
    path = repo / rec["predictions"][RUNS[0]]
    good = path.read_text()
    lines = good.splitlines()
    first = json.loads(lines[0])
    first["probs"] = first["probs"][::-1]
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    assert any("recompute" in p for p in check("E12", root=repo, min_n=1, split="field_dev"))
    path.write_text(good)
    assert check("E12", root=repo, min_n=1, split="field_dev") == []
    rec_path = next((repo / "results/runs").glob("E12-*[0-9]Z.json"))
    tampered = json.loads(rec_path.read_text())
    tampered["metrics"][RUNS[0]]["macro_f1"] += 0.1
    rec_path.write_text(json.dumps(tampered))
    assert any("macro_f1 does not recompute" in p for p in check("E12", root=repo, min_n=1, split="field_dev"))


def test_interrupted_run_writes_no_record_and_resume_lists_both_attempts(repo, cfg, monkeypatch):
    import sprout.train.loop as loop

    real, calls = loop._evaluate, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt  # stands for a killed process
        return real(*a, **k)

    monkeypatch.setattr(loop, "_evaluate", flaky)
    with pytest.raises(KeyboardInterrupt):
        e12.run(cfg, RUNS[0], repo, num_workers=0)
    assert not list((repo / "results" / "runs").glob("E12-*"))  # a partial run is no result
    with pytest.raises(FileExistsError):
        e12.run(cfg, RUNS[0], repo, num_workers=0)
    with pytest.raises(ValueError, match="same count"):  # other augmentation draws would follow
        e12.run(cfg, RUNS[0], repo, resume=True, num_workers=1)
    rec = _record(e12.run(cfg, RUNS[0], repo, resume=True, num_workers=0))
    entry = rec["attempts"][0]
    assert [a["status"] for a in entry["attempts"]] == ["interrupted", "complete"]
    assert entry["attempts"][0]["epochs_in_latest_checkpoint"] == 1
    assert entry["attempts"][1]["resumed_from_latest"] is True
    assert len(entry["history"]) == 1 and entry["epochs_done"] == 2
    assert check("E12", root=repo, min_n=1, split="field_dev") == []


def test_held_out_runs_are_scored_on_their_source_beside_e4s_held_out_rows(repo, cfg):
    for name, held in ((RUNS[1], "plantdoc"), (RUNS[2], "plantseg")):
        rec = _record(e12.run(cfg, name, repo, num_workers=0))
        assert rec["subset"] == f"source:{held}+clean:1"
        assert rec["n"] == (3 if held == "plantdoc" else 4)  # one PlantDoc photo was excluded
        assert set(rec["predictions"]) == {"e1", "convnext_joint", name.removesuffix("_extra"), name}
        assert rec["config"]["train"]["held_out_source"] == held
        assert rec["config"]["trained_on_plantdoc"] is (held != "plantdoc")
        assert rec["attempts"][0]["train_rows"].get(held, 0) == 0
        assert check("E12", root=repo, min_n=1, split="field_dev") == []


def _rec(diffs):
    return {"run_id": "r", "comparisons": [
        {"a": a, "b": b, "metric": "macro_f1", "diff": d, "ci": ci, "mcnemar_p": 1.0} for (a, b), (d, ci) in diffs.items()]}


def test_decision_rule_needs_a_clear_seen_gain_and_no_held_out_loss_past_the_margin():
    c = e12.load_config(CONFIG)

    def seen(lo):
        return _rec({("convnext_joint_extra", "convnext_joint"): (0.03, [lo, 0.06])})

    def held(dd, dg):
        return {"heldout_plantdoc_extra": _rec({("heldout_plantdoc_extra", "heldout_plantdoc"): (dd, [-0.1, 0.1])}),
                "heldout_plantseg_extra": _rec({("heldout_plantseg_extra", "heldout_plantseg"): (dg, [-0.1, 0.1])})}

    assert e12.apply_rule(c, seen(0.01), held(0.05, -0.02))["clearly_helps"] is True  # exactly the margin is allowed
    assert e12.apply_rule(c, seen(0.0), held(0.05, 0.05))["clearly_helps"] is False  # interval touches zero
    assert e12.apply_rule(c, seen(-0.01), held(0.05, 0.05))["clearly_helps"] is False
    out = e12.apply_rule(c, seen(0.01), held(0.05, -0.021))
    assert out["clearly_helps"] is False and out["heldout"]["heldout_plantseg_extra"]["ok"] is False


def test_decide_refuses_without_all_three_records_then_writes_the_decision(repo, cfg):
    (repo / "models/e4/convnext_joint").mkdir(parents=True)
    (repo / "models/e4/convnext_joint/best.pt").write_bytes(b"e4")
    e12.run(cfg, RUNS[0], repo, num_workers=0)
    with pytest.raises(RuntimeError, match="no record yet"):
        e12.decide(cfg, repo)
    for name in RUNS[1:]:
        e12.run(cfg, name, repo, num_workers=0)
    d = json.loads(e12.decide(cfg, repo).read_text())
    assert set(d["records"]) == set(RUNS) and d["class_names"] == CLASSES
    assert d["chosen_model"] == ("models/e12/convnext_joint_extra/best.pt" if d["clearly_helps"]
                                 else "models/e4/convnext_joint/best.pt")
    assert d["refit_e5"] is d["clearly_helps"] and len(d["chosen_model_sha256"]) == 64
    assert {r["model"] for r in d["side_by_side"]} == {"E4 seen-source", "E12 seen-source", "E4 held-out", "E12 held-out"}
    assert {r["scored_on"] for r in d["side_by_side"] if r["model"] == "E12 held-out"} == {"plantdoc", "plantseg"}
    focus = d["per_class_focus"]
    assert [r["class"] for r in focus] == ["Apple___healthy"]
    assert focus[0]["extra_rows"] == 8 and focus[0]["extra_by_source"]["plantwild"] == 4
