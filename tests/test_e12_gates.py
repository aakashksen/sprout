# ruff: noqa: F811  (fixtures imported from test_e12 are used as arguments)
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_e12 import CLASSES, RUNS, repo, sh, world  # noqa: F401  (fixtures)
from test_near_extras import ColourEncoder

from sprout.data.splits import dataset_hash, load_manifest
from sprout.eval.check_run import Scored, check
from sprout.experiments import e4_improve as e4
from sprout.experiments import e5_confidence as e5
from sprout.experiments import e12_gates as g
from sprout.experiments import e12_more_sources as e12

_part = g._part

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "configs" / "e12_gates.json"
BASE_RUNS = ("convnext_joint_extra", "heldout_plantdoc_extra", "heldout_plantseg_extra")
E4_SEED1 = ("convnext_joint_seed1", "heldout_plantdoc_seed1", "heldout_plantseg_seed1")
E12_SEED1 = ("convnext_joint_extra_seed1", "heldout_plantdoc_extra_seed1", "heldout_plantseg_extra_seed1")
TRIM = "heldout_plantdoc_extra_trim"


# ---------------------------------------------------------------------------------------------------------------
# the config is fixed and its runs are E4's and E12's settings again

def test_gate_config_fixes_the_thresholds_seed_and_workers_before_any_run():
    c = e12.load_config(GATES)
    gt = c["gates"]
    assert gt["fixed_before_any_gate_run"] is True and gt["min_n"] == 200
    assert gt["g1"]["min_gain"] == gt["g1"]["reference_gain"] / 2 == 0.1445 and gt["g1"]["trim"]["threshold"] == 0.75
    assert gt["g1"]["trim"]["verifier"] == {"extra_rows": 3484, "against_evaluated": 434, "against_training": 716}
    assert gt["g2"]["seed"] == 1 == e4.load_config(ROOT / "configs" / "e4.json")["seed"] + 1
    assert gt["g2"]["heldout_ci_lower_above"] == 0 and gt["g2"]["seen_point_at_least"] == 0
    assert gt["g3"]["max_more_confident_errors_than_e4"] == 0
    assert [r["name"] for r in c["runs"]] == [TRIM, *E4_SEED1, *E12_SEED1]
    assert c["num_workers"]["default"] == c["num_workers"]["min"] == c["num_workers"]["max"] == 4
    assert e12.resolve_workers(c, None, {}) == 4 and e12.resolve_workers(c, 4, {}) == 4
    for bad in (0, 2, 3):
        with pytest.raises(ValueError, match="between 4 and 4"):
            e12.resolve_workers(c, bad, {})
    # every pair names real runs, with the E4 side first and on the same seed
    names = [r["name"] for r in c["runs"]]
    for p in gt["g2"]["pairs"]:
        e12_spec, e4_spec = e12.run_spec(c, p["e12"]), e12.run_spec(c, p["e4"])
        assert e12_spec["paired_with"] == p["e4"] and names.index(p["e4"]) < names.index(p["e12"])
        assert e12_spec["seed"] == e4_spec["seed"] == 1 and e12_spec["subset"] == e4_spec["subset"]


def test_gate_runs_repeat_the_settings_of_the_runs_they_come_from():
    c, c4 = e12.load_config(GATES), e4.load_config(ROOT / "configs" / "e4.json")
    first = e12.load_config(ROOT / "configs" / "e12.json")
    results = e4.arm_results(c4, ROOT)
    neutral = {"out_dir", "seed", "num_workers"}

    def same(a, b, skip):
        a, b = a.to_dict(), b.to_dict()
        assert {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}

    # E4 arms again: exactly E4's TrainConfig apart from where they are saved and the seed
    for name, arm in zip(E4_SEED1, ("convnext_joint", "heldout_plantdoc", "heldout_plantseg"), strict=True):
        mine = e12.train_config(c, e12.run_spec(c, name), ROOT, 4)
        base = e4.train_config(c4, e4.resolved_spec(c4, arm, results), ROOT)
        same(mine, base, neutral)
        assert mine.seed == 1 and base.seed == 0 and mine.extra_manifest == "" and "extra_train" not in mine.train_splits
        assert mine.out_dir.replace("\\", "/").endswith(f"models/e4_seed1/{name}")
    # E12 again: exactly the first E12 run apart from the seed and where it is saved
    for name, old in zip(E12_SEED1, BASE_RUNS, strict=True):
        mine = e12.train_config(c, e12.run_spec(c, name), ROOT, 4)
        same(mine, e12.train_config(first, e12.run_spec(first, old), ROOT, 4), neutral)
        assert mine.seed == 1 and mine.extra_manifest == "extra_train"
        assert mine.out_dir.replace("\\", "/").endswith(f"models/e12/seed1/{name}")
    # the trimmed run: the first PlantDoc held-out run with the trimmed extras, same seed
    mine = e12.train_config(c, e12.run_spec(c, TRIM), ROOT, 4)
    same(mine, e12.train_config(first, e12.run_spec(first, "heldout_plantdoc_extra"), ROOT, 4), neutral | {"extra_manifest"})
    assert mine.seed == 0 and mine.extra_manifest == "extra_train_trim_plantdoc"
    # no gate run saves over a model that already exists
    dirs = {str(e12.run_dir(c, r, ROOT)) for r in c["runs"]}
    old_dirs = {str(e12.run_dir(first, r, ROOT)) for r in first["runs"]} | {str(ROOT / "models/e4" / a["name"]) for a in c4["arms"]}
    assert len(dirs) == 7 and not dirs & old_dirs


def test_refit_config_changes_only_the_model_and_where_it_writes():
    old, new = (json.loads((ROOT / "configs" / f).read_text()) for f in ("e5.json", "e5_e12.json"))
    skip = {"out_dir", "model", "refit"}
    assert {k: v for k, v in old.items() if k not in skip} == {k: v for k, v in new.items() if k not in skip}
    assert new["out_dir"] == "results/e12/e5_refit" != old["out_dir"]
    assert new["model"]["field_dev_predictions_file"] == "models/e12/convnext_joint_extra/field_dev.predictions.jsonl"
    assert new["model"]["checkpoint"] == "models/e12/convnext_joint_extra/best.pt"
    assert e12.load_config(GATES)["gates"]["g3"]["e12_chosen"] == "results/e12/e5_refit/chosen.json"


# ---------------------------------------------------------------------------------------------------------------
# the rules, each failing on a fixture that violates it

G1 = {"min_gain": 0.1445, "reference_gain": 0.289, "ci_lower_above": 0.0}
SICK_CLEAR = {"sick_only_macro_f1": {"ci_touches_zero": False}}
SICK_ZERO = {"sick_only_macro_f1": {"ci_touches_zero": True}}


def test_g1_needs_half_the_gain_and_an_interval_above_zero():
    assert g.evaluate_g1(G1, {"diff": 0.20, "ci": [0.10, 0.30]}, SICK_CLEAR)["pass"] is True
    assert g.evaluate_g1(G1, {"diff": 0.1445, "ci": [0.01, 0.30]}, SICK_CLEAR)["pass"] is True  # exactly half passes
    low = g.evaluate_g1(G1, {"diff": 0.144, "ci": [0.05, 0.30]}, SICK_CLEAR)
    assert low["pass"] is False and low["checks"] == {"gain_at_least_min": False, "interval_above_zero": True}
    touching = g.evaluate_g1(G1, {"diff": 0.20, "ci": [0.0, 0.30]}, SICK_CLEAR)
    assert touching["pass"] is False and touching["checks"]["interval_above_zero"] is False
    # a sick-only interval that touches zero does not fail the gate, but the statement changes
    soft = g.evaluate_g1(G1, {"diff": 0.20, "ci": [0.10, 0.30]}, SICK_ZERO)
    assert soft["pass"] is True and soft["sick_only_interval_touches_zero"] is True
    assert "not described as better disease detection" in soft["statement"]
    assert "not described" not in g.evaluate_g1(G1, {"diff": 0.20, "ci": [0.10, 0.30]}, SICK_CLEAR)["statement"]


G2 = {"heldout_ci_lower_above": 0.0, "seen_point_at_least": 0.0}


def _pairs(seen, doc, seg):
    return [{"kind": k, "comparison": {"diff": d, "ci": ci}}
            for k, (d, ci) in (("seen", seen), ("heldout_plantdoc", doc), ("heldout_plantseg", seg))]


def test_g2_needs_both_heldout_intervals_clear_and_a_seen_point_not_below_zero():
    ok = _pairs((0.0, [-0.03, 0.03]), (0.2, [0.1, 0.3]), (0.2, [0.1, 0.3]))
    assert g.evaluate_g2(G2, ok)["pass"] is True  # the seen interval may span zero; only its point is gated
    assert g.evaluate_g2(G2, _pairs((-0.001, [-0.03, 0.03]), (0.2, [0.1, 0.3]), (0.2, [0.1, 0.3])))["pass"] is False
    out = g.evaluate_g2(G2, _pairs((0.02, [0.0, 0.04]), (0.2, [0.0, 0.3]), (0.2, [0.1, 0.3])))
    assert out["pass"] is False and [p["pass"] for p in out["pairs"]] == [True, False, True]
    assert g.evaluate_g2(G2, _pairs((0.02, [0.0, 0.04]), (0.2, [0.1, 0.3]), (0.2, [-0.1, 0.3])))["pass"] is False


G3 = {"max_more_confident_errors_than_e4": 0, "low_count_below": 5}


def test_g3_fails_only_when_e12_has_more_confident_errors_than_e4():
    assert g.evaluate_g3(G3, 20, 12)["pass"] is True
    tie = g.evaluate_g3(G3, 2, 2)
    assert tie["pass"] is True and tie["tie"] is True and tie["e4_count_too_low_to_mean_much"] is True
    assert g.evaluate_g3(G3, 20, 20)["e4_count_too_low_to_mean_much"] is False
    worse = g.evaluate_g3(G3, 3, 4)
    assert worse["pass"] is False and worse["tie"] is False


def test_confident_sick_called_healthy_uses_the_calibrated_probability_and_any_healthy_class():
    healthy = np.array([True, True, False, False])  # classes 0,1 healthy; 2,3 sick
    labels = np.array([2, 3, 2, 0, 2])
    probs = np.array([
        [0.01, 0.01, 0.97, 0.01],  # sick, called sick
        [0.90, 0.02, 0.03, 0.05],  # sick, called healthy class 0, confident
        [0.02, 0.60, 0.20, 0.18],  # sick, called healthy class 1, not confident
        [0.02, 0.01, 0.95, 0.02],  # healthy photo called sick: not counted here
        [0.05, 0.93, 0.01, 0.01],  # sick, called a different healthy class, confident
    ])
    assert g.confident_sick_called_healthy(probs, labels, healthy, 1.0, 0.85) == 2
    assert g.confident_sick_called_healthy(probs, labels, healthy, 1.0, 0.95) == 0
    r = g.crossover_rates(probs, labels, healthy)
    assert (r["sick_rows"], r["sick_called_healthy"], r["healthy_rows"], r["healthy_called_sick"]) == (4, 3, 1, 1)
    assert r["sick_called_healthy_rate"] == 0.75 and r["healthy_called_sick_rate"] == 1.0


def test_sick_only_and_healthy_split_is_computed_on_the_right_rows():
    healthy = np.array([True, False, False])
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    groups = [f"g{i}" for i in range(30)]
    onehot = lambda p: np.eye(3)[np.array(p)] * 0.9 + 0.05  # noqa: E731
    good = Scored(groups=groups, labels=labels, probs=onehot(labels))
    bad_pred = np.where(labels == 0, 1, labels)  # every healthy photo called sick; sick perfect
    bad = Scored(groups=groups, labels=labels, probs=onehot(bad_pred))
    out = g.sick_healthy_split(good, bad, healthy)
    assert (out["n_sick"], out["n_healthy"]) == (20, 10)
    assert out["sick_only_macro_f1"]["new"] == out["sick_only_macro_f1"]["old"] == 1.0
    assert out["sick_only_macro_f1"]["diff"] == 0 and out["sick_only_macro_f1"]["ci_touches_zero"] is True
    assert (out["healthy_accuracy"]["new"], out["healthy_accuracy"]["old"], out["healthy_accuracy"]["diff"]) == (1.0, 0.0, 1.0)
    assert out["healthy_accuracy"]["ci"][0] > 0 and out["healthy_accuracy"]["ci_touches_zero"] is False
    only_sick = ~np.isin(labels, [0])  # a source with no healthy photo, like PlantSeg
    one = g.sick_healthy_split(_part(good, only_sick), _part(bad, only_sick), healthy)
    assert one["healthy_accuracy"] is None and one["n_healthy"] == 0 and one["sick_only_macro_f1"]["new"] == 1.0


# ---------------------------------------------------------------------------------------------------------------
# the whole path on tiny fixtures: list tool, seven runs, check_run, gates

def _tool():
    spec = importlib.util.spec_from_file_location("extra_trim_tool", ROOT / "tools" / "extra_trim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_unlike_extras(root):
    """Two red extras become green, so no PlantDoc photo is near them and the list keeps them."""
    extra = pd.read_csv(root / "datasets/manifests/extra_train.csv")
    for path in extra[extra["label"] == CLASSES[0]]["path"].iloc[:2]:
        Image.new("RGB", (64, 64), (10, 220, 10)).save(path)


def _commit(root, message):
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", message)


@pytest.fixture
def gcfg(repo):
    c = e12.load_config(GATES)
    c["settings"].update(model="tiny_cnn", lr=0.05, epochs=2, batch_size=4, lab_cap=5)
    c.update(min_free_disk_gb=0)
    c["extra"].update(rows=16, dataset_hash=dataset_hash(load_manifest("extra_train")))
    c["num_workers"].update(default=0, min=0)
    for r in c["runs"]:
        if r["name"] == TRIM:
            r["e4_record"] = "E4-20261103T000000000010Z"  # the fixture's stand-in for E4's PlantDoc held-out record
    c["gates"]["min_n"] = 1
    c["gates"]["healthy_suffix"] = "Apple___healthy"  # the fixture's class 0 is healthy, class 1 stands for a sick one
    return c


@pytest.fixture
def base_cfg(repo):
    c = e12.load_config(ROOT / "configs" / "e12.json")
    c["settings"].update(model="tiny_cnn", lr=0.05, epochs=2, batch_size=4, lab_cap=5)
    c.update(min_free_disk_gb=0)
    c["extra"].update(rows=16, dataset_hash=dataset_hash(load_manifest("extra_train")))
    c["e4"]["records"] = {n: f"E4-20261103T0000000000{i}0Z" for i, n in enumerate(RUNS)}
    return c


def _record(path):
    return json.loads(Path(path).read_text())


def test_trim_list_is_checked_before_a_trimmed_run_and_the_run_trains_on_the_trimmed_extras(repo, gcfg):
    tool = _tool()
    _two_unlike_extras(repo)
    spec = e12.run_spec(gcfg, TRIM)
    with pytest.raises(RuntimeError, match="does not exist"):  # no list committed yet
        e12.run(gcfg, TRIM, repo, num_workers=0)
    result = tool.make(gcfg, repo, ColourEncoder(), "cpu", 1)
    assert result["reproduces_verifier"] is False  # the fixture does not hold the verifier's 434 and 716
    observed = {k: result[k] for k in gcfg["gates"]["g1"]["trim"]["verifier"]}
    gcfg["gates"]["g1"]["trim"]["verifier"] = observed
    result = tool.make(gcfg, repo, ColourEncoder(), "cpu", 1)
    assert result["reproduces_verifier"] is True and result["extra_rows"] == 16
    assert result["union"] == 14 and result["trimmed_rows"] == 2
    tool.write_outputs(result, gcfg, repo)
    _commit(repo, "list")
    kept = load_manifest(spec["extra"]["manifest"])
    assert len(kept) == result["trimmed_rows"] and dataset_hash(kept) == result["trimmed_dataset_hash"]
    assert set(kept["path"]).isdisjoint(e["path"] for e in result["remove"])
    rec = _record(e12.run(gcfg, TRIM, repo, num_workers=0))
    assert rec["config"]["extra_manifest"] == "extra_train_trim_plantdoc" and rec["n"] == 3
    assert rec["config"]["extra_rows_in_training"] == result["trimmed_rows"] and rec["config"]["trained_on_plantdoc"] is False
    assert set(rec["predictions"]) == {"e1", "convnext_joint", "heldout_plantdoc", TRIM}
    assert (repo / "models/e12" / TRIM / "best.pt").is_file()
    # a manifest that is not the extras minus the list is refused
    kept.iloc[1:].to_csv(repo / "datasets/manifests/extra_train_trim_plantdoc.csv", index=False)
    with pytest.raises(RuntimeError, match="not the extras minus"):
        e12.check_run_extras(gcfg, spec, repo)


def test_second_seed_runs_pair_by_seed_on_the_same_photos_and_the_gates_stage_writes_its_file(repo, gcfg, base_cfg, monkeypatch):
    opened = []
    real_open = Image.open

    def spy(fp, *a, **k):
        opened.append(str(fp))
        return real_open(fp, *a, **k)

    monkeypatch.setattr(Image, "open", spy)
    tool = _tool()
    _two_unlike_extras(repo)
    result = tool.make(gcfg, repo, ColourEncoder(), "cpu", 1)
    gcfg["gates"]["g1"]["trim"]["verifier"] = {k: result[k] for k in gcfg["gates"]["g1"]["trim"]["verifier"]}
    tool.write_outputs(tool.make(gcfg, repo, ColourEncoder(), "cpu", 1), gcfg, repo)
    _commit(repo, "list")
    for name in BASE_RUNS:  # the first records, which the gates read and must not change
        e12.run(base_cfg, name, repo, num_workers=0)
    first_models = {n: (repo / "models/e12" / n / "best.pt").read_bytes() for n in BASE_RUNS}
    with pytest.raises(RuntimeError, match="run it before"):  # E12 needs its E4 pair's record first
        e12.run(gcfg, E12_SEED1[0], repo, num_workers=0)
    for name in (TRIM, *E4_SEED1, *E12_SEED1):
        e12.run(gcfg, name, repo, num_workers=0)
    with pytest.raises(ValueError, match="between 0 and 4"):
        e12.run(gcfg, E4_SEED1[0], repo, num_workers=9)

    runs = repo / "results" / "runs"
    recs = {r["config"]["run"]: r for r in map(_record, runs.glob("E12-*[0-9]Z.json"))}
    assert set(recs) == {*BASE_RUNS, TRIM, *E4_SEED1, *E12_SEED1}
    for e4_name, e12_name in zip(E4_SEED1, E12_SEED1, strict=True):
        a, b = recs[e4_name], recs[e12_name]
        assert a["config"]["train"]["seed"] == b["config"]["train"]["seed"] == 1 and a["comparisons"] == []
        assert a["config"]["trained_on_plantdoc"] is (e4_name != E4_SEED1[1]) and a["config"]["extra_rows_in_training"] == 0
        assert b["config"]["paired_record"] == a["run_id"] and set(b["predictions"]) == {e12_name, e4_name}
        assert {(c["a"], c["b"]) for c in b["comparisons"]} == {(e12_name, e4_name)}
        rows = [[json.loads(x)["path"] for x in (repo / b["predictions"][s]).read_text().splitlines()] for s in b["predictions"]]
        assert rows[0] == rows[1]  # identical row IDs
        assert (repo / "models/e12/seed1" / e12_name / "best.pt").is_file() and (repo / "models/e4_seed1" / e4_name / "best.pt").is_file()
        assert (repo / "models/e4_seed1" / e4_name / "attempts.json").is_file()
    assert {n: (repo / "models/e12" / n / "best.pt").read_bytes() for n in BASE_RUNS} == first_models  # nothing overwritten
    assert not [p for p in opened if "field_test" in p or "lab_test" in p]
    assert check("E12", root=repo, min_n=1, split="field_dev") == []

    # E5's refit files and E4's, written by hand here (their stages are tested in tests/test_e5.py)
    sha = __import__("hashlib").sha256((repo / "models/e12/convnext_joint_extra/best.pt").read_bytes()).hexdigest()
    (repo / "results/e5").mkdir(parents=True, exist_ok=True)
    (repo / "results/e12/e5_refit").mkdir(parents=True)
    (repo / "results/e5/chosen.json").write_text(json.dumps({"temperature": 1.0, "abstain_threshold": 0.5}))
    (repo / "results/e12/e5_refit/chosen.json").write_text(
        json.dumps({"temperature": 1.0, "abstain_threshold": 0.5, "model_sha256": sha}))
    decision = repo / "results/e12/decision.json"
    decision.write_text('{"keep": "me"}')
    out = g.gates(gcfg, repo)
    assert out == repo / "results/e12/gates.json" and decision.read_text() == '{"keep": "me"}'
    got = _record(out)
    assert set(got) >= {"g1", "g2", "g3", "all_pass", "records", "git_commit"} and got["field_test_read"] is False
    assert got["g1"]["n"] == 3 and got["g1"]["record"] == recs[TRIM]["run_id"]
    assert set(got["g1"]["split"]) >= {"sick_only_macro_f1", "healthy_accuracy", "n_sick", "n_healthy"}
    comp = {c["metric"]: c for c in recs[TRIM]["comparisons"] if c["b"] == "heldout_plantdoc"}["macro_f1"]
    assert got["g1"]["gain"] == comp["diff"] and got["g1"]["ci"] == comp["ci"]
    assert got["g1"]["pass"] == (comp["diff"] >= 0.1445 and comp["ci"][0] > 0)
    assert [p["kind"] for p in got["g2"]["pairs"]] == ["seen", "heldout_plantdoc", "heldout_plantseg"]
    for p in got["g2"]["pairs"]:
        c = next(c for c in recs[p["e12"]]["comparisons"] if c["metric"] == "macro_f1")
        assert p["gain"] == c["diff"] and p["records"] == [recs[p["e12"]]["run_id"], recs[p["e4"]]["run_id"]]
    assert got["g2"]["gpu_minutes"]["total"] >= 0 and got["g2"]["seed"] == 1
    assert got["g3"]["rows"] == 7 and set(got["g3"]["unthresholded"]) == {"seen", "plantdoc", "plantseg"}
    assert got["all_pass"] == (got["g1"]["pass"] and got["g2"]["pass"] and got["g3"]["pass"])
    assert not [p for p in opened if "field_test" in p or "lab_test" in p]

    # G3 alone gives the same answer, in its own file, with nothing that could read as all three gates passing
    alone = _record(g.g3_only(gcfg, repo))
    assert alone["g3"] == got["g3"] and alone["g1_g2"] == "not run"
    assert "all_pass" not in alone and "g1" not in alone and "g2" not in alone

    # the stage refuses wrong inputs: weights the refit was not fitted for, a pair on different seeds, a tampered record
    chosen = repo / "results/e12/e5_refit/chosen.json"
    good = chosen.read_text()
    chosen.write_text(json.dumps({"temperature": 1.0, "abstain_threshold": 0.5, "model_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="different weights"):
        g.gates(gcfg, repo)
    chosen.write_text(good)
    pair_path = next(p for p in runs.glob("E12-*[0-9]Z.json") if _record(p)["config"]["run"] == E4_SEED1[1])
    saved = pair_path.read_text()
    skewed = json.loads(saved)
    skewed["config"]["train"]["seed"] = 0
    pair_path.write_text(json.dumps(skewed))
    with pytest.raises(RuntimeError, match="seed 1"):
        g.gates(gcfg, repo)
    pair_path.write_text(saved)
    seed1 = next(p for p in runs.glob("E12-*[0-9]Z.json") if _record(p)["config"]["run"] == E12_SEED1[0])
    saved = seed1.read_text()
    tampered = json.loads(saved)
    tampered["comparisons"][0]["diff"] += 0.1
    seed1.write_text(json.dumps(tampered))
    assert any("does not recompute" in p for p in check("E12", root=repo, min_n=1, split="field_dev"))
    with pytest.raises(RuntimeError, match="does not pass check_run"):
        g.gates(gcfg, repo)
    seed1.write_text(saved)
    gcfg["gates"]["min_n"] = 200  # the real floor: these tiny fixtures are below it
    with pytest.raises(RuntimeError, match="min-n 200"):
        g.gates(gcfg, repo)


def test_interrupted_second_seed_run_resumes_from_its_own_folder(repo, gcfg, monkeypatch):
    import sprout.train.loop as loop

    real, calls = loop._evaluate, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(loop, "_evaluate", flaky)
    name = E4_SEED1[0]
    with pytest.raises(KeyboardInterrupt):
        e12.run(gcfg, name, repo, num_workers=0)
    assert not list((repo / "results" / "runs").glob("E12-*"))
    with pytest.raises(FileExistsError):
        e12.run(gcfg, name, repo, num_workers=0)
    rec = _record(e12.run(gcfg, name, repo, resume=True, num_workers=0))
    entry = rec["attempts"][0]
    assert [a["status"] for a in entry["attempts"]] == ["interrupted", "complete"] and entry["epochs_done"] == 2
    assert (repo / "models/e4_seed1" / name / "attempts.json").is_file() and not (repo / "models/e12" / name).exists()


def test_refit_reads_the_saved_field_dev_rows_and_names_its_weights(tmp_path):
    """E5's load_field takes model.field_dev_predictions_file; fit adds model_sha256 (hash of the checkpoint)."""
    paths = [f"p{i}.jpg" for i in range(4)]
    labels = ["a", "a", "b", "b"]
    cfg = {"field_split": "field_dev", "model": {"name": "m", "e4_record": "R", "checkpoint": "m.pt",
                                                  "field_dev_predictions_file": "rows.jsonl"}}
    (tmp_path / "results/runs").mkdir(parents=True)
    (tmp_path / "datasets/manifests").mkdir(parents=True)
    pd.DataFrame({"path": paths, "label": labels, "group": paths, "split": "field_dev", "sha1": paths, "source": "s"}).to_csv(
        tmp_path / "datasets/manifests/field.csv", index=False)
    info = e5.split_info("field_dev", tmp_path)
    (tmp_path / "results/runs/R.json").write_text(json.dumps(
        {"split": "field_dev", "dataset_hash": info.hash, "config": {"class_names": ["a", "b"]}, "predictions": {}}))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps({"path": p, "label": int(y == "b"), "probs": [0.7, 0.3]}) + "\n"
                                                 for p, y in zip(paths, labels)))
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "green_ratio.jsonl").write_text("".join(json.dumps({"path": p, "green_ratio": 0.5}) + "\n" for p in paths))
    cfg["out_dir"] = "."
    dev = e5.load_field(cfg, tmp_path)
    assert len(dev.paths) == 4 and dev.class_names == ["a", "b"] and dev.probs.shape == (4, 2)


def test_fit_for_a_refit_records_the_checkpoint_hash(tmp_path, monkeypatch):
    (tmp_path / "m.pt").write_bytes(b"weights")
    cfg = {"out_dir": "out", "model": {"checkpoint": "m.pt", "field_dev_predictions_file": "rows.jsonl"}}
    monkeypatch.setattr(e5, "load_field", lambda c, r: None)
    monkeypatch.setattr(e5, "load_not_plant", lambda c, r: None)
    monkeypatch.setattr(e5, "fit", lambda c, f, n: {"temperature": 1.0, "abstain_threshold": 0.5, "gate": {"chosen": "x"}})
    log = e5.StageLog(cfg, tmp_path)
    saved = json.loads(e5.stage_fit(cfg, tmp_path, log).read_text())
    assert saved["model_sha256"] == __import__("hashlib").sha256(b"weights").hexdigest() and saved["model_checkpoint"] == "m.pt"
    del cfg["model"]["field_dev_predictions_file"]  # E4's own fit has no such key and records no hash
    assert "model_sha256" not in json.loads(e5.stage_fit(cfg, tmp_path, log).read_text())
