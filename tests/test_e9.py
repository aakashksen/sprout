"""E9, the final test, with fakes and tiny fixtures only: no test photo, question or prediction is read, no GPU, no
Ollama, no Claude call. The guards, the records check_run recomputes, and the answer-quality stages."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprout.eval.check_run import check_file
from sprout.experiments import e9_final as e9

REAL_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Apple___rust", "Apple___scab", "Tomato___blight", "Tomato___healthy"]
LAB = ["path,label,group,split,sha1"] + [
    f"l/{i}.jpg,{CLASSES[i % 4]},g{i},{'lab_test' if i >= 8 else 'lab_dev'},l{i}" for i in range(24)
]
FIELD = ["path,label,group,split,sha1,source"] + [
    f"f/{i}.jpg,{CLASSES[i % 3]},h{i},{'field_test' if i >= 6 else 'field_dev'},s{i},{'plantdoc' if i % 2 else 'plantseg'}"
    for i in range(30)
]
LOG = (
    "version,image_id,path,source,split,label,reason_code,note,began_as\n"
    "1,s7,f/7.jpg,plantdoc,field_test,Apple___scab,wrong_plant,x,remove\n"
    "1,s10,f/10.jpg,plantseg,field_test,Apple___rust,not_plant_photo,y,remove\n"
    "1,s11,f/11.jpg,plantdoc,field_test,Apple___scab,wrong_plant,z,remove\n"
    "1,s2,f/2.jpg,plantseg,field_dev,Tomato___blight,wrong_plant,w,remove\n"
)
TASKS = {"tasks": [{"id": "H1", "status": "done"}, {"id": "H2", "status": "pending"}]}
SHA = "a" * 64
CONFIG = {
    "experiment": "E9", "exclusion_version": 1, "decision_file": "results/e12/decision.json",
    "calibration": {"file": "results/e5/chosen.json"}, "tasks_file": "orchestration/tasks.json",
    "batch_size": 4, "torch_device": "cpu",
    "systems": {
        "baseline": {"kind": "torch_checkpoint", "checkpoint": "models/e1/best.pt"},
        "xgboost": {"kind": "xgboost", "model": "models/e3/xgboost/model.ubj", "features": {"size": 32}},
        "dinov2": {"kind": "torch_checkpoint", "checkpoint": "models/e4/dinov2_head/best.pt"},
        "quantized": {"kind": "onnx", "export_file": "results/e6/export.json", "variant": "onnx_dynamic",
                      "model_dir": "models/e6"},
    },
    "calibrated": ["improved", "quantized"],
    "rag": {"index_dir": "knowledge/index", "device": "cuda", "writer": "writer-model", "judge": "judge-model",
            "grader": "sonnet", "max_sonnet_calls": 400, "judge_in_flight": 2},
    "wrong_disease": {"dev_source": {}, "served_system": "served", "blind_seed": 5, "blind_per_call": 3,
                      "dev_max_calls": 100},
}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True)


def commit(root, message="x"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


def write(root, rel, text):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text if isinstance(text, str) else json.dumps(text), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    write(r, ".gitignore", "results/e9/work/\nmlruns/\n")
    write(r, "datasets/manifests/lab.csv", "\n".join(LAB) + "\n")
    write(r, "datasets/manifests/field.csv", "\n".join(FIELD) + "\n")
    write(r, "datasets/label_exclusions.csv", LOG)
    write(r, "orchestration/tasks.json", TASKS)
    write(r, "configs/e9.json", CONFIG)
    write(r, "results/e12/decision.json", {"chosen_model": "models/e4/convnext_joint/best.pt",
                                           "chosen_model_sha256": SHA, "class_names": CLASSES, "refit_e5": False})
    write(r, "results/e5/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6})
    prompt = e9.PROMPT_REL
    (r / prompt).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_ROOT / prompt, r / prompt)
    commit(r, "base")
    return r


def set_h2(root, status):
    write(root, "orchestration/tasks.json", {"tasks": [{"id": "H1", "status": "done"}, {"id": "H2", "status": status}]})


# ----------------------------------------------------------------------------------------------- guards


def test_labels_reviewed_is_false_before_h2_and_true_after(repo):
    assert e9.labels_reviewed(repo) is False  # H2 pending
    set_h2(repo, "done")
    assert e9.labels_reviewed(repo) is True
    write(repo, "orchestration/tasks.json", {"tasks": [{"id": "H1", "status": "done"}]})
    assert e9.labels_reviewed(repo) is False  # no H2 at all is not a signature
    write(repo, "orchestration/tasks.json", {"tasks": [{"id": "H2", "status": "blocked"}]})
    assert e9.labels_reviewed(repo) is False


def test_refuses_when_config_not_committed(repo):
    e9.require_committed(repo, ["configs/e9.json"])  # committed and unchanged: fine
    write(repo, "configs/e9.json", {**CONFIG, "batch_size": 8})
    with pytest.raises(e9.Refused, match="not committed"):
        e9.require_committed(repo, ["configs/e9.json"])
    git(repo, "checkout", "--", "configs/e9.json")
    write(repo, "configs/new.json", {})
    with pytest.raises(e9.Refused, match="not committed"):
        e9.require_committed(repo, ["configs/new.json"])  # never added
    write(repo, "configs/e9.json", {**CONFIG, "batch_size": 8})
    called = []
    with pytest.raises(e9.Refused, match="not committed"):  # the stage itself refuses before any model runs
        e9.run_images("lab_test", e9.load_config(repo), {"baseline": called.append}, repo)
    assert called == []


def test_missing_decision_file_refuses(repo):
    (repo / "results/e12/decision.json").unlink()
    commit(repo, "no decision")
    with pytest.raises(e9.Refused, match="decision.json"):
        e9.load_decision(e9.load_config(repo), repo)


def test_only_test_splits_are_run_here(repo):
    with pytest.raises(e9.Refused, match="lab_test or field_test"):
        e9.run_images("lab_dev", e9.load_config(repo), {}, repo)


# ----------------------------------------------------------------------------------------------- wrong-disease prompt


def test_wrong_disease_prompt_shows_question_and_diagnosis():
    msgs = e9.wd_messages("How do I treat this?", "The photo suggests Tomato - Early blight (90% sure).",
                          "Remove the lower leaves [1].", REAL_ROOT)
    shown = json.dumps(msgs)
    assert "How do I treat this?" in shown and "Tomato - Early blight" in shown and "Remove the lower leaves" in shown
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "disease" in msgs[0]["content"] and "plant" in msgs[0]["content"]
    for hidden in ("supported", "verdict", "local", "gemma"):  # nothing from the faithfulness judges
        assert hidden not in shown.lower()


def test_the_wrong_disease_reading_is_strict():
    assert e9.parse_call("FITS") == "FITS" and e9.parse_call("wrong.") == "WRONG"
    assert e9.parse_call("WRONG because it is a different plant") is None
    assert e9.parse_call("") is None and e9.parse_call("FITS WRONG") is None


# ----------------------------------------------------------------------------------------------- image splits

SYSTEMS = ("baseline", "improved", "xgboost", "dinov2", "quantized")


def tiny(seed, hit=0.7, names=CLASSES, seen=None):
    def run(frame):
        if seen is not None:
            seen.append(list(frame["path"]))
        out = np.full((len(frame), len(names)), 0.05)
        for i, label in enumerate(frame["label"]):
            true = names.index(label)
            out[i, true if ((i * 7 + seed) % 10) < hit * 10 else (true + 1) % len(names)] = 0.85
        return out / out.sum(axis=1, keepdims=True)

    return e9.Predictor(f"tiny test model {seed}", list(names), run, {"seed": seed})


def predictors(seen=None):
    return {n: tiny(i, 0.5 + 0.1 * i, seen=seen) for i, n in enumerate(SYSTEMS)}


def read_rows(root, rec, system):
    return [json.loads(x) for x in (root / rec["predictions"][system]).read_text().splitlines()]


def test_lab_test_gets_one_record_and_a_shared_class_view(repo):
    paths = e9.run_images("lab_test", e9.load_config(repo), predictors(), repo)
    full, view = (json.loads(p.read_text()) for p in paths)
    assert full["split"] == "lab_test" and full["subset"] is None and full["experiment"] == "E9"
    assert view["subset"] == "shared_classes" and view["split"] == "lab_test"
    assert set(full["predictions"]) == {"baseline", "improved", "improved_calibrated", "xgboost", "dinov2",
                                        "quantized", "quantized_calibrated"}
    shared = {"Apple___rust", "Apple___scab", "Tomato___blight"}  # the classes both lab.csv and field.csv hold
    assert full["n"] == 16 and view["n"] == 12  # lab_test rows i=8..23; Tomato___healthy is lab only
    assert {CLASSES[r["label"]] for r in read_rows(repo, view, "baseline")} == shared
    m = full["metrics"]["improved"]
    for k in ("accuracy", "macro_f1", "ece", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy",
              "plant_given_top_1", "plant_given_top_3"):
        assert k in m
    assert full["config"]["options_per_plant"] == {"Apple": 2, "Tomato": 2}  # stated with the plant-given numbers
    # with and without abstention: the calibrated systems also report coverage and the error among answered photos
    c = full["metrics"]["improved_calibrated"]
    assert {"coverage@0.6", "selective_accuracy@0.6", "confident_error_rate@0.6"} <= set(c)
    assert not any(k.startswith("coverage@") for k in m)
    assert full["metrics"]["quantized"]["changed_vs@improved"] is not None
    assert full["config"]["calibration"] == {"file": "results/e5/chosen.json", "temperature": 1.5, "abstain_threshold": 0.6}
    for p in paths:
        assert check_file(p, repo, min_n=1) == []  # the view of the full record is not a second use


def test_field_test_scores_only_cleaned_rows_and_reports_why_others_went(repo):
    seen = []
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(seen), repo)
    rec = json.loads(path.read_text())
    assert rec["subset"] == "clean:1" and rec["n"] == 21
    scored = {Path(p).relative_to(repo).as_posix() for p in seen[0]}
    assert len(scored) == 21 and not scored & {"f/7.jpg", "f/10.jpg", "f/11.jpg"}  # removed rows never reach a model
    assert rec["config"]["cleaned_rows"] == {"before": 24, "after": 21, "removed": 3,
                                              "by_reason": {"wrong_plant": 2, "not_plant_photo": 1},
                                              "by_source": {"plantdoc": 2, "plantseg": 1}, "version": 1}
    assert check_file(path, repo, min_n=21) == []


def test_each_image_split_is_scored_once_per_system_and_labels_reviewed_comes_from_h2(repo):
    set_h2(repo, "done")
    commit(repo, "h2")
    seen = []
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(seen), repo)
    assert len(seen) == 5  # one pass over the rows per frozen system
    assert json.loads(path.read_text())["labels_reviewed"] is True
    assert json.loads(path.read_text())["config"]["labels_reviewed"] is True


def test_refuses_second_test_run_without_override_note(repo):
    e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    seen = []
    with pytest.raises(e9.Refused, match="already been used"):
        e9.run_images("field_test", e9.load_config(repo), predictors(seen), repo)
    with pytest.raises(e9.Refused, match="already been used"):
        e9.run_images("field_test", e9.load_config(repo), predictors(seen), repo, override_note="  ")
    assert seen == []  # refused before any model ran
    # the other split is still unused
    e9.run_images("lab_test", e9.load_config(repo), predictors(), repo)
    (again,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo, override_note="rerun: reason")
    assert json.loads(again.read_text())["config"]["override_note"] == "rerun: reason"
    assert check_file(again, repo, min_n=1) == []  # check_run accepts a later record that carries a note


def test_a_system_with_other_classes_than_the_decision_is_refused(repo):
    bad = predictors()
    bad["xgboost"] = tiny(2, names=CLASSES[::-1])
    with pytest.raises(e9.Refused, match="class order"):
        e9.run_images("lab_test", e9.load_config(repo), bad, repo)
    with pytest.raises(e9.Refused, match="missing"):
        e9.run_images("lab_test", e9.load_config(repo), {"baseline": tiny(0)}, repo)


def test_calibration_must_belong_to_the_chosen_model(repo):
    cfg = e9.load_config(repo)
    decision = e9.load_decision(cfg, repo)
    assert e9.load_calibration(cfg, decision, repo)["temperature"] == 1.5
    with pytest.raises(e9.Refused, match="refit"):
        e9.load_calibration(cfg, {**decision, "refit_e5": True}, repo)  # chosen.json names no model
    write(repo, "results/e5/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6, "model_sha256": SHA})
    assert e9.load_calibration(cfg, {**decision, "refit_e5": True}, repo)["abstain_threshold"] == 0.6


def test_image_record_is_recomputed_and_tampering_is_caught(repo):
    paths = e9.run_images("lab_test", e9.load_config(repo), predictors(), repo)
    rec = json.loads(paths[0].read_text())
    pred = repo / rec["predictions"]["improved"]
    original = pred.read_text()
    lines = original.splitlines()
    row = json.loads(lines[0])
    row["probs"] = row["probs"][::-1]
    pred.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    assert any("does not recompute" in p for p in check_file(paths[0], repo, min_n=1))
    pred.write_text(original)
    assert check_file(paths[0], repo, min_n=1) == []
    cal = repo / rec["predictions"]["improved_calibrated"]
    text = cal.read_text()
    cal.write_text(text.replace('"label": 1', '"label": 3', 1))
    assert check_file(paths[0], repo, min_n=1) != []
    cal.write_text(text)
    saved = json.loads(paths[0].read_text())
    saved["metrics"]["improved"]["accuracy"] += 0.01
    paths[0].write_text(json.dumps(saved))
    assert any("accuracy does not recompute" in p for p in check_file(paths[0], repo, min_n=1))
    saved["metrics"]["improved"]["accuracy"] -= 0.01
    saved["n"] = 3
    paths[0].write_text(json.dumps(saved))
    assert check_file(paths[0], repo, min_n=1) != []
    # a field record that quietly scores the removed rows too is caught: its n and ids no longer fit the subset
    (fpath,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    f = json.loads(fpath.read_text())
    f["subset"] = None
    fpath.write_text(json.dumps(f))
    assert check_file(fpath, repo, min_n=1) != []
