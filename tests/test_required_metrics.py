import subprocess

import numpy as np
import pytest

from sprout.eval.check_run import check
from sprout.eval.predict import Predictions
from sprout.eval.records import split_info, standard_image_metrics, write_run

NAMES = ["P___a", "P___b"]


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def root(tmp_path):
    lab = ["path,label,group,split,sha1"] + [f"i/{i}.jpg,{NAMES[i % 2]},g{i},lab_dev,s{i}" for i in range(6)]
    gate = ["path,label,group,split,sha1"] + [f"n/{i}.jpg,not_plant,n{i},not_plant_dev,z{i}" for i in range(3)]
    for rel, text in {"datasets/manifests/lab.csv": lab, "datasets/manifests/not_plant.csv": gate,
                      "orchestration/tasks.json": ["{}"]}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(text) + "\n")
    sh(tmp_path, "init", "-q")
    sh(tmp_path, "config", "user.email", "t@t")
    sh(tmp_path, "config", "user.name", "t")
    sh(tmp_path, "add", "-A")
    sh(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def classifier_run(root, experiment, complete):
    info = split_info("lab_dev", root)
    labels = np.array([NAMES.index(x) for x in info.labels])
    probs = np.where(np.eye(2)[labels] > 0, 0.8, 0.2)
    metrics = standard_image_metrics(probs, labels, NAMES) if complete else {"accuracy": 1.0}
    return write_run(experiment, "lab_dev", {"class_names": NAMES}, {"classifier": "resnet-test"},
                     {"a": metrics}, {}, [{"setting": "default", "result": "ok"}],
                     {"a": Predictions(list(info.ids), labels, probs)}, info.hash, root=root)


def test_new_experiment_must_report_top_k_and_plant_given(root):
    classifier_run(root, "E4", complete=False)
    problems = check("E4", root=root)
    for name in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "macro_f1", "ece",
                 "plant_given_top_1", "plant_given_top_3"):
        assert any("missing required metric" in p and name in p for p in problems), (name, problems)


def test_complete_new_record_passes(root):
    classifier_run(root, "E4", complete=True)
    assert check("E4", root=root) == []


@pytest.mark.parametrize("experiment", ["E1", "E2", "E3"])
def test_named_older_experiments_are_exempt(root, experiment):
    classifier_run(root, experiment, complete=False)
    assert check(experiment, root=root) == []


def test_gate_record_on_not_plant_split_is_exempt(root):
    info = split_info("not_plant_dev", root)
    rows = [{"path": p, "accepted": False} for p in info.ids]
    write_run("E5", "not_plant_dev", {}, {"gate": "gate-test"}, {"g": {"not_plant_acceptance_rate": 0.0}}, {},
              [{"setting": "default", "result": "ok"}], {"g": rows}, info.hash, root=root)
    assert check("E5", root=root) == []
