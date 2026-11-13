import json
import subprocess
from pathlib import Path

import pytest

from sprout.eval.records import split_info, subset_positions

FIELD = ["path,label,group,split,sha1,source"] + [
    f"f/{i}.jpg,{'AB'[i % 2] if i < 10 else 'C'},h{i},field_dev,s{i},{'plantdoc' if i % 3 else 'plantseg'}"
    for i in range(12)
]
LAB = ["path,label,group,split,sha1"] + [f"l/{i}.jpg,{'AB'[i % 2]},g{i},lab_dev,l{i}" for i in range(4)]
LOG = (
    "version,image_id,path,source,split,label,reason_code,note\n"
    "1,s1,f/1.jpg,plantdoc,field_dev,B,wrong_plant,x\n"
    "1,s4,f/4.jpg,plantdoc,field_dev,A,not_plant_photo,y\n"
    "2,s6,f/6.jpg,plantseg,field_dev,A,label_contradicted,z\n"
)


@pytest.fixture
def root(tmp_path):
    for rel, text in {"datasets/manifests/field.csv": FIELD, "datasets/manifests/lab.csv": LAB}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(text) + "\n")
    (tmp_path / "datasets/label_exclusions.csv").write_text(LOG)
    return tmp_path


def _paths(root, subset):
    info = split_info("field_dev", root)
    return [info.ids[i] for i in subset_positions(subset, "field_dev", info, root)]


def test_clean_subset_drops_exclusions_up_to_its_version(root):
    everything = [f"f/{i}.jpg" for i in range(12)]
    assert _paths(root, "clean:0") == everything
    assert _paths(root, "clean:1") == [p for p in everything if p not in ("f/1.jpg", "f/4.jpg")]
    assert "f/6.jpg" not in _paths(root, "clean:2")
    with pytest.raises(ValueError, match="version"):
        _paths(root, "clean:3")


def test_clean_subset_combines_with_the_old_subsets(root):
    shared = _paths(root, "shared_classes")  # labels A and B, not C
    assert len(shared) == 10
    assert _paths(root, "shared_classes+clean:1") == [p for p in shared if p not in ("f/1.jpg", "f/4.jpg")]
    both = _paths(root, "source:plantdoc+clean:1")
    assert both == [p for p in _paths(root, "source:plantdoc") if p not in ("f/1.jpg", "f/4.jpg")]


# ---------------------------------------------------------------------------------------------------------------
# the experiment, on fake earlier records

import numpy as np  # noqa: E402

from sprout.eval.check_run import check  # noqa: E402
from sprout.eval.predict import Predictions  # noqa: E402
from sprout.eval.records import write_run  # noqa: E402
from sprout.experiments import e13_clean_labels as e13  # noqa: E402

NAMES = ["A", "B"]
HEAVY_FIELD = ["path,label,group,split,sha1,source"] + [
    f"f/{i}.jpg,{'AB'[i % 2]},h{i},{'field_test' if i >= 30 else 'field_dev'},s{i},{'plantdoc' if i % 3 else 'plantseg'}"
    for i in range(36)
]
HEAVY_LOG = (
    "version,image_id,path,source,split,label,reason_code,note,began_as\n"
    "1,s1,f/1.jpg,plantdoc,field_dev,B,wrong_plant,x,remove\n"
    "1,s4,f/4.jpg,plantdoc,field_dev,A,not_plant_photo,y,remove\n"
    "1,s31,f/31.jpg,plantdoc,field_test,B,wrong_plant,z,remove\n"
)


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _probs(labels, wrong):
    p = np.full((len(labels), 2), 0.2)
    for i, y in enumerate(labels):
        p[i, 1 - y if i in wrong else y] = 0.8
    return p


@pytest.fixture
def world(tmp_path):
    files = {"datasets/manifests/field.csv": HEAVY_FIELD, "datasets/manifests/lab.csv": LAB}
    for rel, lines in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(lines) + "\n")
    (tmp_path / "datasets/label_exclusions.csv").write_text(HEAVY_LOG)
    (tmp_path / "orchestration").mkdir()
    (tmp_path / "orchestration/tasks.json").write_text("{}")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _old_record(root, systems, wrongs):
    info = split_info("field_dev", root)
    pos = subset_positions("shared_classes", "field_dev", info, root)
    paths = [info.ids[i] for i in pos]
    labels = np.array([NAMES.index(info.labels[i]) for i in pos])
    preds = {s: Predictions(paths, labels, _probs(labels, w)) for s, w in zip(systems, wrongs)}
    metrics = {s: e13.standard_image_metrics(p.probs, labels, NAMES) for s, p in preds.items()}
    path = write_run("E4", "field_dev", {"class_names": NAMES}, {s: f"{s}-model" for s in systems}, metrics, {},
                     [{"setting": "x"}], preds, info.hash, root=root, subset="shared_classes")
    return path.stem


def _cfg(root, rid):
    return {"experiment": "E13", "split": "field_dev", "exclusion_version": 1, "reference": "joint",
            "views": [{"name": "shared", "subset": "shared_classes", "systems": {"joint": rid, "weak": rid}}]}


def test_clean_scores_match_filtered_old_rows_and_pass_check_run(world):
    rid = _old_record(world, ["joint", "weak"], [{0}, {0, 2, 3, 5, 7}])
    out = e13.run_view(_cfg(world, rid), _cfg(world, rid)["views"][0], world)
    rec = json.loads(out.read_text())
    assert rec["subset"] == "shared_classes+clean:1" and rec["split"] == "field_dev"
    assert rec["config"]["rows"] == {"before": 30, "after": 28, "removed": 2}
    assert rec["config"]["removed_by_reason"] == {"wrong_plant": 1, "not_plant_photo": 1}
    old = rec["config"]["old_metrics_same_systems_all_rows"]["joint"]["accuracy"]
    assert old == pytest.approx(29 / 30) and rec["n"] == 28
    assert rec["config"]["device"].startswith("none")
    assert check("E13", root=world, min_n=20, split="field_dev") == []
    assert {c["b"] for c in rec["comparisons"]} == {"weak"}


def test_field_test_is_never_scored(world):
    rid = _old_record(world, ["joint", "weak"], [set(), {1}])
    out = e13.run_view(_cfg(world, rid), _cfg(world, rid)["views"][0], world)
    rows = [json.loads(x) for x in (world / json.loads(out.read_text())["predictions"]["joint"]).read_text().splitlines()]
    assert rows and all(r["path"] in {f"f/{i}.jpg" for i in range(30)} for r in rows)  # no field_test photo
    with pytest.raises(ValueError, match="field_dev only"):
        (world / "c.json").write_text(json.dumps({**_cfg(world, rid), "split": "field_test"}))
        e13.load_config(world / "c.json")
    src = Path(e13.__file__).read_text(encoding="utf-8")
    assert "PIL" not in src and "torch" not in src and "Image.open" not in src  # no image, no model


def test_system_without_saved_predictions_is_refused(world):
    rid = _old_record(world, ["joint", "weak"], [set(), {1}])
    cfg = _cfg(world, rid)
    cfg["views"][0]["systems"]["ghost"] = rid
    with pytest.raises(ValueError, match="does not run models"):
        e13.run_view(cfg, cfg["views"][0], world)


def test_unknown_exclusion_version_is_refused(world):
    rid = _old_record(world, ["joint", "weak"], [set(), {1}])
    cfg = {**_cfg(world, rid), "exclusion_version": 2}
    with pytest.raises(ValueError, match="version"):
        e13.run_view(cfg, cfg["views"][0], world)


# ---------------------------------------------------------------------------------------------------------------
# earlier records


def test_old_records_still_pass_check_run():
    root = Path(__file__).resolve().parents[1]
    runs = root / "results" / "runs"
    found = [e for e in ("E1", "E2", "E3", "E4", "E10") if list(runs.glob(f"{e}-*.json"))]
    assert found, "no earlier record to check"
    for exp in found:
        assert check(exp, root=root, min_n=200) == [], exp


def test_real_config_names_saved_systems():
    root = Path(__file__).resolve().parents[1]
    cfg = e13.load_config(root / "configs" / "e13.json")
    for view in cfg["views"]:
        for system, rid in view["systems"].items():
            rec = json.loads((root / "results" / "runs" / f"{rid}.json").read_text())
            assert system in rec["predictions"] and rec["subset"] == view["subset"]
