import json
import subprocess

import numpy as np
import pytest

from sprout.eval import metrics as M
from sprout.eval.check_run import check
from sprout.eval.predict import Predictions
from sprout.eval.records import (
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)

NAMES = ["P___a", "P___b", "P___c", "Q___a", "Q___b", "Q___c", "Q___d"]


def test_top_k_on_a_fixed_case():
    probs = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
    labels = [2, 3, 0]  # true class ranks 2nd, 4th, 1st
    assert M.top_k_accuracy(probs, labels, 1) == pytest.approx(1 / 3)
    assert M.top_k_accuracy(probs, labels, 3) == pytest.approx(2 / 3)
    assert M.top_k_accuracy(probs, labels, 5) == 1.0
    # a tie goes to the lower index, so k=1 agrees with argmax
    assert M.top_k_accuracy([[0.5, 0.5]], [1], 1) == 0.0
    assert M.top_k_accuracy([[0.5, 0.5]], [0], 1) == 1.0


def test_plant_given_restricts_to_the_true_plant():
    names = ["P___a", "P___b", "Q___a", "Q___b", "Q___c", "Q___d"]
    probs = [[0.1, 0.2, 0.5, 0.1, 0.05, 0.05],   # true P___a; Q___a leads overall
             [0.05, 0.05, 0.3, 0.1, 0.2, 0.3]]   # true Q___c; Q___a/Q___d lead
    labels = [0, 4]
    assert M.top_k_accuracy(probs, labels, 1) == 0.0
    assert M.plant_given_top_k(probs, labels, names, 1) == 0.0   # P___b beats P___a; Q___a, Q___d beat Q___c
    assert M.plant_given_top_k(probs, labels, names, 2) == 0.5   # only P has two options
    assert M.options_per_plant(names) == {"P": 2, "Q": 4}
    assert M.plant_given_top_k(probs, [1, 4], names, 1) == 0.5   # P___b is the best P class
    # undefined, not wrong, when the true label has no plant
    assert np.isnan(M.plant_given_top_k([[0.5, 0.5]], [1], ["P___a", "not_plant"], 1))


def test_plant_with_three_classes_scores_one_at_top_3():
    probs = np.full((3, 7), 0.01)
    probs[np.arange(3), [3, 4, 5]] = 0.9  # always confidently a Q class, never the true P class
    labels = [0, 1, 2]
    assert M.top_k_accuracy(probs, labels, 1) == 0.0
    assert M.plant_given_top_k(probs, labels, NAMES, 1) == pytest.approx(1 / 3)  # ties: lowest index wins
    assert M.options_per_plant(NAMES)["P"] == 3
    assert M.plant_given_top_k(probs, labels, NAMES, 3) == 1.0


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def make_root(tmp_path, field_rows):
    lab = ["path,label,group,split,sha1"] + [
        f"img/{i}.jpg,{NAMES[i % 7]},g{i},lab_dev,s{i}" for i in range(14)
    ]
    field = ["path,label,group,split,sha1,source"] + field_rows
    for rel, lines in {"datasets/manifests/lab.csv": lab, "datasets/manifests/field.csv": field,
                       "orchestration/tasks.json": ["{}"]}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(lines) + "\n")
    sh(tmp_path, "init", "-q")
    sh(tmp_path, "config", "user.email", "t@t")
    sh(tmp_path, "config", "user.name", "t")
    sh(tmp_path, "add", "-A")
    sh(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def predictions(info, positions):
    rng = np.random.default_rng(0)
    labels = np.array([NAMES.index(info.labels[i]) for i in positions])
    return Predictions([info.ids[i] for i in positions], labels, rng.dirichlet(np.ones(7), len(positions)))


def record(root, split, subset=None):
    info = split_info(split, root)
    positions = subset_positions(subset, split, info, root) if subset else list(range(len(info.ids)))
    p = predictions(info, positions)
    return write_run(
        "E1", split, {"class_names": NAMES}, {"classifier": "resnet-test"},
        {"a": standard_image_metrics(p.probs, p.label, NAMES)}, {}, [{"setting": "default", "result": "ok"}],
        {"a": p}, info.hash, root=root, subset=subset,
    )


def edit(path, fn):
    rec = json.loads(path.read_text())
    fn(rec)
    path.write_text(json.dumps(rec))


def test_saved_top_k_is_recomputed_and_tampering_fails(tmp_path):
    root = make_root(tmp_path, ["f/0.jpg,P___a,h0,field_dev,t0,plantdoc"])
    path = record(root, "lab_dev")
    rec = json.loads(path.read_text())
    assert {"top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "plant_given_top_1", "plant_given_top_3"} <= set(
        rec["metrics"]["a"])
    assert rec["config"]["options_per_plant"] == {"P": 3, "Q": 4}
    assert check("E1", root=root) == []
    original = path.read_text()
    for key in ("top_3_accuracy", "plant_given_top_1", "top_5_accuracy"):
        path.write_text(original)
        edit(path, lambda r, key=key: r["metrics"]["a"].update({key: 0.123}))
        problems = check("E1", root=root)
        assert any(f"metric {key} does not recompute" in p for p in problems), problems
    path.write_text(original)
    edit(path, lambda r: r["config"].pop("options_per_plant"))
    assert any("options_per_plant" in p for p in check("E1", root=root))
    path.write_text(original)
    edit(path, lambda r: r["config"].update(options_per_plant={"P": 2, "Q": 5}))
    assert any("options_per_plant" in p for p in check("E1", root=root))


def test_source_subset_rebuilds_from_the_manifest(tmp_path):
    root = make_root(tmp_path, [
        "f/0.jpg,P___a,h0,field_dev,t0,plantdoc",
        "f/1.jpg,P___b,h1,field_dev,t1,plantseg",
        "f/2.jpg,Q___a,h2,field_dev,t2,plantseg",
        "f/3.jpg,Q___a,h3,field_test,t3,plantseg",
    ])
    info = split_info("field_dev", root)
    assert subset_positions("source:plantseg", "field_dev", info, root) == [1, 2]
    assert subset_positions("source:plantdoc", "field_dev", info, root) == [0]
    with pytest.raises(ValueError):
        subset_positions("source:plantseg", "lab_dev", split_info("lab_dev", root), root)
    record(root, "field_dev", "source:plantseg")
    assert check("E1", root=root) == []
    # the record's n and ids no longer fit once the manifest says a row came from another source
    field = root / "datasets/manifests/field.csv"
    field.write_text(field.read_text().replace("h2,field_dev,t2,plantseg", "h2,field_dev,t2,plantdoc"))
    sh(root, "commit", "-qam", "edit manifest")
    assert check("E1", root=root) != []
