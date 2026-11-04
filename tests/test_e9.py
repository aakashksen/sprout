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
