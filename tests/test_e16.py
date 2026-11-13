import json
import subprocess

import numpy as np
import pytest

from sprout.eval.check_run import check
from sprout.eval.records import split_info, subset_positions, write_run
from sprout.experiments import e16_remeasure as e16

NAMES = ["A", "B"]
DEV = 16
FIELD = ["path,label,group,split,sha1,source"] + [
    f"f/{i}.jpg,{NAMES[i % 2]},h{i},field_dev,s{i},{'plantdoc' if i % 2 else 'plantseg'}" for i in range(DEV)
] + [f"t/{i}.jpg,{NAMES[i % 2]},t{i},field_train,u{i},plantdoc" for i in range(4)]
LOG = ("version,image_id,path,source,split,label,reason_code,note,began_as\n"
       "1,u0,t/0.jpg,plantdoc,field_train,A,wrong_plant,x,remove\n")
GREEN = [0.0, 0.5, 0.5, 0.5] * (DEV // 4)


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _commit(root, message):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


@pytest.fixture
def world(tmp_path):
    def put(rel, text):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)

    put("datasets/manifests/field.csv", "\n".join(FIELD) + "\n")
    put("datasets/manifests/lab.csv", "path,label,group,split,sha1\nl/0.jpg,A,g0,lab_dev,l0\n")
    put("datasets/label_exclusions.csv", LOG)
    put("orchestration/tasks.json", "{}")
    put("results/e5/green_ratio.jsonl", "".join(json.dumps({"path": f"f/{i}.jpg", "green_ratio": g}) + "\n"
                                                 for i, g in enumerate(GREEN)))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _commit(tmp_path, "init")
    return tmp_path


def _e15(root, wrong=frozenset({1, 2})):
    info = split_info("field_dev", root)
    labels = np.array([NAMES.index(x) for x in info.labels])
    probs = np.full((DEV, 2), 0.1)
    for i, y in enumerate(labels):
        probs[i, 1 - y if i in wrong else y] = 0.9
    rows = [{"path": p, "label": int(y), "probs": [float(v) for v in pr]} for p, y, pr in zip(info.ids, labels, probs)]
    metrics = {e16.SYSTEM: e16.standard_image_metrics(probs, labels, NAMES)}
    path = write_run("E15", "field_dev", {"class_names": NAMES}, {e16.SYSTEM: "x"}, metrics, {}, [{"s": 1}],
                     {e16.SYSTEM: rows}, info.hash, root=root, subset="clean:1")
    return path.stem


def _refit(root, e15_id, t=1.0):
    gate = {"settings": {"kind": "confidence_ood", "confidence_threshold": 0.5, "ood_threshold": 0.9},
            "fit_plant_rejection": 0.0, "fit_not_plant_rejection": 1.0, "eligible": True}
    chosen = {"seed": 0, "fit_percent": 50, "temperature": t, "classes": NAMES, "e15_record": e15_id,
              "gates": {"green_rule": {**gate, "settings": {"kind": "green_rule", "green_threshold": 0.05}},
                        "confidence": {**gate, "settings": {"kind": "confidence", "confidence_threshold": 0.5}},
                        "confidence_ood": gate},
              "gate": {"chosen": "confidence_ood", "settings": gate["settings"]}}
    p = root / "results/e15/refit/chosen.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(chosen))
    _commit(root, "refit")


def test_rejection_counts_a_mask():
    acc = np.array([True, False, False, True])
    assert e16.rejection(acc, np.ones(4, dtype=bool)) == {"n": 4, "refused": 2, "rate": 0.5}
    assert e16.rejection(acc, np.array([True, True, False, False])) == {"n": 2, "refused": 1, "rate": 0.5}
    assert e16.rejection(acc, np.zeros(4, dtype=bool))["rate"] is None


def test_gate_record_counts_refusals_from_saved_probabilities_and_passes_check_run(world):
    rid = _e15(world)
    _refit(world, rid)
    out = e16.gate_record(world, rid)
    rec = json.loads(out.read_text())
    assert rec["experiment"] == "E16" and rec["n"] == DEV and rec["labels_reviewed"] is False
    # wrong rows still have a 0.9 top probability, so the confidence part accepts every row; entropy of (0.9, 0.1)
    # is 0.469 of the maximum, under 0.9. Nothing is refused by the chosen gate, and the green rule refuses the 4 at 0.0.
    assert rec["config"]["rows"]["confidence_ood"]["all_rows"] == {"n": DEV, "refused": 0, "rate": 0.0}
    assert rec["config"]["rows"]["green_rule"]["all_rows"]["refused"] == DEV // 4
    assert rec["metrics"]["green_rule"]["plant_rejection_rate"] == pytest.approx(0.25)
    parts = rec["config"]["rows"]["green_rule"]
    assert parts["fitted_on"]["n"] + parts["not_fitted_on"]["n"] == DEV
    assert rec["config"]["no_tuning"].startswith("T4.9's settings")
    assert check("E16", root=world, min_n=DEV, split="field_dev") == []


def test_gate_is_refused_when_the_refit_is_for_another_record(world):
    rid = _e15(world)
    _refit(world, "E15-other")
    with pytest.raises(ValueError, match="was not fitted"):
        e16.gate_record(world, rid)


def test_a_temperature_changes_who_is_refused(world):
    rid = _e15(world)
    _refit(world, rid, t=50.0)  # nearly flat probabilities: entropy over the limit and confidence under 0.5
    rec = json.loads(e16.gate_record(world, rid).read_text())
    assert rec["config"]["rows"]["confidence_ood"]["all_rows"]["refused"] == DEV


def test_best_neighbours_takes_the_higher_of_photo_and_mirror_and_names_the_photo():
    dev = np.array([[1.0, 0.0], [0.0, 1.0]])
    mirror = np.array([[0.0, 1.0], [0.0, 1.0]])
    train = np.array([[1.0, 0.0], [0.0, 1.0]])
    cos, idx = e16.best_neighbours(dev, mirror, train)
    assert cos.tolist() == [1.0, 1.0] and idx.tolist() == [0, 1]
    cos, _ = e16.best_neighbours(np.array([[0.6, 0.8]]), np.array([[0.8, 0.6]]), np.array([[1.0, 0.0]]))
    assert cos[0] == pytest.approx(0.8)


def test_neighbours_file_lists_the_photos_at_the_threshold_and_the_closest():
    paths = ["a", "b", "c", "d"]
    cos = np.array([0.99, 0.90, 0.89, 0.2])
    body = e16.neighbours_file(paths, cos, ["t1", "t2", "t3", "t4"], {"e15_record": "x"})
    assert body["paths"] == ["a", "b"] and body["at_or_above_threshold"] == 2
    assert body["at_or_above_strict_threshold"] == 1
    assert [c["path"] for c in body["closest"]][:3] == ["a", "b", "c"]


def test_listed_and_unlisted_subsets_split_the_rows_and_read_the_file_at_the_commit(world):
    listing = world / "results/t82/lookalikes.json"
    listing.parent.mkdir(parents=True)
    listing.write_text(json.dumps({"paths": ["f/1.jpg", "f/5.jpg"]}))
    _commit(world, "list")
    info = split_info("field_dev", world)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=world, capture_output=True, text=True).stdout.strip()
    name = "results/t82/lookalikes.json"
    listed = subset_positions(f"listed:{name}", "field_dev", info, world, sha)
    unlisted = subset_positions(f"unlisted:{name}", "field_dev", info, world, sha)
    assert [info.ids[i] for i in listed] == ["f/1.jpg", "f/5.jpg"]
    assert sorted(listed + unlisted) == list(range(DEV)) and not set(listed) & set(unlisted)
    listing.write_text(json.dumps({"paths": []}))  # a later edit does not change what the record means
    assert subset_positions(f"listed:{name}", "field_dev", info, world, sha) == listed
    assert subset_positions(f"clean:1+unlisted:{name}", "field_dev", info, world) == list(range(DEV))


def test_lookalike_records_pass_check_run_and_use_the_saved_rows(world):
    rid = _e15(world, wrong={1, 2, 3})
    body = {"e15_record": rid, "threshold": 0.9, "strict_threshold": 0.95, "train_rows": 4, "at_or_above_threshold": 3,
            "at_or_above_strict_threshold": 1, "paths": ["f/1.jpg", "f/2.jpg", "f/4.jpg"],
            "closest": [{"path": "f/1.jpg", "cosine": 0.97, "nearest_train": "t/1.jpg"},
                        {"path": "f/2.jpg", "cosine": 0.92, "nearest_train": "t/0.jpg"},
                        {"path": "f/4.jpg", "cosine": 0.91, "nearest_train": "t/0.jpg"}]}
    p = world / e16.NEIGHBOURS_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(body))
    _commit(world, "neighbours")
    # the fake E15 subset is clean:1, as the fixture's record; the real one adds final_classes
    out = [json.loads(x.read_text()) for x in e16.lookalike_records(world, rid)]
    listed, unlisted, doc, seg = out
    assert (listed["n"], unlisted["n"]) == (3, DEV - 3) and listed["experiment"] == "E17"
    assert listed["metrics"][e16.SYSTEM]["accuracy"] == pytest.approx(1 / 3)  # f/1 and f/2 wrong, f/4 right
    assert doc["n"] + seg["n"] == DEV
    assert listed["config"]["accuracy_at_or_above_strict_threshold"] == 0.0
    assert check("E17", root=world, min_n=3, split="field_dev") == []


def test_lookalike_file_for_another_record_is_refused(world):
    rid = _e15(world)
    p = world / e16.NEIGHBOURS_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"e15_record": "other", "paths": [], "closest": []}))
    _commit(world, "n")
    with pytest.raises(ValueError, match="another E15 record"):
        e16.lookalike_records(world, rid)


def test_no_test_split_is_read():
    src = e16.__file__
    text = open(src, encoding="utf-8").read()
    assert "field_test" not in text and "field_dev" in text
