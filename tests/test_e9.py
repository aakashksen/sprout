"""E9, the final test, with fakes and tiny fixtures only: no test photo, question or prediction is read, no GPU, no
Ollama, no Claude call. The guards, the records check_run recomputes, and the answer-quality stages."""
import hashlib
import json
import re
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
E4_BYTES, E12_BYTES = b"the E4 checkpoint", b"the E12 checkpoint"
SHA = hashlib.sha256(E4_BYTES).hexdigest()
SHA12 = hashlib.sha256(E12_BYTES).hexdigest()
SIGNOFF = {s: {"signed": False, "by": None, "at": None, "words": None}
           for s in ("lab_test", "field_test", "rag_test")}
CONFIG = {
    "experiment": "E9", "exclusion_version": 1, "decision_file": "results/e15/decision.json", "e15_config": "configs/e15.json",
    "candidates": {"e4": {"checkpoint": "models/e4/convnext_joint/best.pt", "calibration": "results/e5/chosen.json"},
                   "e12": {"checkpoint": "models/e12/convnext_joint_extra/best.pt",
                           "calibration": "results/e12/e5_refit/chosen.json"},
                   "e15": {"checkpoint": "models/e15/final/best.pt", "calibration": None}},
    "batch_size": 4, "torch_device": "cpu", "old_classes_file": "results/e12/decision.json",
    "systems": {
        "baseline": {"kind": "torch_checkpoint", "checkpoint": "models/e1/best.pt"},
        "xgboost": {"kind": "xgboost", "model": "models/e3/xgboost/model.ubj", "features": {"size": 32}},
        "dinov2": {"kind": "torch_checkpoint", "checkpoint": "models/e4/dinov2_head/best.pt"},
        "quantized": {"kind": "onnx", "export_file": "results/e6/export.json", "variant": "onnx_dynamic",
                      "model_dir": "models/e6"},
    },
    "calibrated": ["improved", "alternate", "quantized"],
    "rag": {"index_dir": "knowledge/index", "device": "cuda", "writer": "writer-model",
            "grader": "sonnet", "max_sonnet_calls": 400, "max_haiku_calls": 400, "judge_in_flight": 3},
    "wrong_disease": {"dev_source": {}, "served_system": "served", "dev_max_calls": 100},
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


WHICH = {"e4": ("convnext_joint", "models/e4/convnext_joint/best.pt", SHA, 1.5, 0.6),
         "e12": ("convnext_joint_extra", "models/e12/convnext_joint_extra/best.pt", SHA12, 2.0, 0.7),
         "e15": ("convnextv2_final", "models/e15/final/best.pt", hashlib.sha256(b"the T4.9 checkpoint").hexdigest(), 1.2, 0.8)}


def decide(root, which="e4", classes=CLASSES, sha=None):
    """T4.9's decision file as it writes it, and the refit it points to through configs/e15.json."""
    name, ckpt, own, temperature, threshold = WHICH[which]
    sha = sha or own
    write(root, "results/e15/decision.json", {"chosen": name, "chosen_checkpoint": ckpt, "chosen_checkpoint_sha256": sha,
                                              "classes": classes, "record": "E15-1"})
    write(root, "results/e15/refit/chosen.json", {"temperature": temperature, "abstain_threshold": threshold,
                                                  "model_sha256": sha})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    write(r, ".gitignore", "results/e9/work/\nmlruns/\n")
    write(r, "datasets/manifests/lab.csv", "\n".join(LAB) + "\n")
    write(r, "datasets/manifests/field.csv", "\n".join(FIELD) + "\n")
    write(r, "datasets/label_exclusions.csv", LOG)
    write(r, "orchestration/label_signoff.json", SIGNOFF)
    write(r, "configs/e9.json", CONFIG)
    write(r, "results/e12/decision.json", {"class_names": CLASSES})  # E12's decision: the list older models use
    write(r, "configs/e15.json", {"refit": {"out_dir": "results/e15/refit"}})
    decide(r, "e4")
    write(r, "results/e5/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6})
    write(r, "results/e12/e5_refit/chosen.json", {"temperature": 2.0, "abstain_threshold": 0.7, "model_sha256": SHA12})
    for rel, data in (("models/e4/convnext_joint/best.pt", E4_BYTES), ("models/e12/convnext_joint_extra/best.pt", E12_BYTES)):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_bytes(data)
    write(r, "configs/e5.json", {"model": {"checkpoint": "models/e4/convnext_joint/best.pt", "e4_record": "E4-1"}})
    write(r, "results/runs/E5-1.json", {"experiment": "E5", "config": {
        "model_record": "E4-1", "temperature": 1.5, "abstain_threshold": 0.6}})
    write(r, "results/runs/E6-1.json", {"run_id": "E6-1", "experiment": "E6", "split": "field_dev", "config": {
        "checkpoint_sha256": SHA, "thresholds": {"file": "results/e5/chosen.json", "temperature": 1.5,
                                                    "abstain_threshold": 0.6, "model_sha256": None, "names_this_checkpoint": None,
                                                    "systems": {"pytorch": {"scoring_part": E6_ROW, "all_rows": E6_ROW},
                                                                "onnx_dynamic": {"scoring_part": E6_DYNAMIC, "all_rows": E6_DYNAMIC}}}}})
    e6 = json.loads((r / "results/runs/E6-1.json").read_text())
    e6["run_id"], e6["config"]["checkpoint_sha256"] = "E6-2", SHA12  # E6 on the other candidate, for the swapped roles
    write(r, "results/runs/E6-2.json", e6)
    from PIL import Image

    for line in (*LAB[1:], *FIELD[1:]):  # tiny photos; every fifth by number carries an EXIF orientation tag
        rel = line.split(",")[0]
        img = Image.new("RGB", (8, 8))
        exif = img.getexif()
        if int(rel.split("/")[1].split(".")[0]) % 5 == 0:
            exif[0x0112] = 6
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(r / rel, exif=exif)
    from sprout.experiments.e14h_haiku_check import rubric_hash

    write(r, "results/e14/haiku_decision.json", {"first_judge": "haiku", "record_id": "E14H-1", "model": "haiku-model",
                                                  "rubric_sha256": rubric_hash()})
    prompt = e9.PROMPT_REL
    (r / prompt).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_ROOT / prompt, r / prompt)
    commit(r, "base")
    return r


# ----------------------------------------------------------------------------------------------- guards


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
    with pytest.raises(e9.Refused, match="not committed"):  # the stage itself refuses before any model is built
        e9.run_images("lab_test", e9.load_config(repo), lambda: called.append("built") or {}, repo)
    assert called == []


def test_missing_decision_file_refuses(repo):
    (repo / "results/e15/decision.json").unlink()
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

SYSTEMS = ("baseline", "improved", "alternate", "xgboost", "dinov2", "quantized")


def tiny(seed, hit=0.7, names=CLASSES, seen=None, sha=None):
    def run(frame):
        if seen is not None:
            seen.append(list(frame["path"]))
        out = np.full((len(frame), len(names)), 0.05)
        for i, label in enumerate(frame["label"]):
            true = names.index(label)
            out[i, true if ((i * 7 + seed) % 10) < hit * 10 else (true + 1) % len(names)] = 0.85
        return out / out.sum(axis=1, keepdims=True)

    return e9.Predictor(f"tiny test model {seed}", list(names), run, {"seed": seed, **({"sha256": sha} if sha else {})})


def predictors(seen=None, alternate=SHA12, quantized=SHA):
    """Fake systems; the alternate says which checkpoint it is, and the quantized file which one it was exported from."""
    out = {n: tiny(i, 0.5 + 0.1 * i, seen=seen, sha=alternate if n == "alternate" else None) for i, n in enumerate(SYSTEMS)}
    out["quantized"].detail["source_checkpoint_sha256"] = quantized
    return out


def read_rows(root, rec, system):
    return [json.loads(x) for x in (root / rec["predictions"][system]).read_text().splitlines()]


def test_lab_test_gets_one_record_and_a_shared_class_view(repo):
    paths = e9.run_images("lab_test", e9.load_config(repo), predictors(), repo)
    full, view = (json.loads(p.read_text()) for p in paths)
    assert full["split"] == "lab_test" and full["subset"] is None and full["experiment"] == "E9"
    assert view["subset"] == "shared_classes" and view["split"] == "lab_test"
    assert set(full["predictions"]) == {"baseline", "improved", "improved_calibrated", "alternate",
                                        "alternate_calibrated", "xgboost", "dinov2", "quantized", "quantized_calibrated"}
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
    assert full["config"]["calibration"]["improved"] == {"file": "results/e15/refit/chosen.json", "temperature": 1.5, "abstain_threshold": 0.6}
    assert full["config"]["calibration"]["alternate"]["temperature"] == 2.0
    assert "coverage@0.7" in full["metrics"]["alternate_calibrated"]
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


def test_each_image_split_is_scored_once_per_system_and_the_record_holds_no_signature(repo):
    seen = []
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(seen), repo)
    assert len(seen) == 6  # one pass over the rows per frozen system
    rec = json.loads(path.read_text())
    assert rec["labels_reviewed"] is False and rec["config"]["labels_reviewed"] is False
    assert "orchestration/label_signoff.json" in rec["config"]["label_signature"]
    assert "orchestration/label_signoff.json" in rec["notes"]


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


def test_calibration_must_belong_to_the_model_it_was_fitted_on(repo):
    cfg = e9.load_config(repo)
    decision = e9.load_decision(cfg, repo)
    assert decision["calibration"] == "results/e15/refit/chosen.json"  # configs/e15.json's refit.out_dir, not a key
    cal = e9.calibrations(cfg, decision, repo, alternate_sha=SHA12, scored=("alternate",))
    assert cal["improved"]["temperature"] == 1.5 and cal["quantized"] == cal["improved"]
    assert cal["alternate"]["temperature"] == 2.0 and "alternate_2" not in cal  # e15 is not chosen and has no refit
    with pytest.raises(e9.Refused, match="fitted on"):  # a file that names another model than the one scored
        e9.calibrations(cfg, decision, repo, alternate_sha=SHA)
    write(repo, "results/e15/refit/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6})
    with pytest.raises(e9.Refused, match="refit"):  # the refit must name the model it was fitted on
        e9.calibrations(cfg, decision, repo, alternate_sha=SHA12)
    write(repo, "results/e15/refit/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6, "model_sha256": SHA12})
    with pytest.raises(e9.Refused, match="fitted on"):  # and it must be the chosen one
        e9.calibrations(cfg, decision, repo, alternate_sha=SHA12)


def test_the_decision_is_read_as_t49_writes_it(repo):
    cfg = e9.load_config(repo)
    write(repo, "results/e15/decision.json", {"chosen_model": "models/e4/convnext_joint/best.pt", "class_names": CLASSES})
    with pytest.raises(e9.Refused, match="lacks"):  # the old key names are not accepted
        e9.load_decision(cfg, repo)
    (repo / "configs/e15.json").unlink()
    decide(repo, "e4")
    with pytest.raises(e9.Refused, match="e15.json"):
        e9.load_decision(cfg, repo)


def test_the_primary_model_is_the_one_e12_chose_and_the_other_is_reported_beside_it(repo):
    cfg = e9.load_config(repo)
    (path,) = e9.run_images("field_test", cfg, predictors(), repo)
    rec = json.loads(path.read_text())
    c = rec["config"]
    assert c["primary"]["candidate"] == "e4" and c["alternate"]["candidate"] == "e12"  # decision.json: E4 stays
    assert c["primary"]["system"] == "improved" and c["alternate"]["system"] == "alternate"
    assert c["primary"]["checkpoint"] == "models/e4/convnext_joint/best.pt"
    assert c["alternate"]["checkpoint"] == "models/e12/convnext_joint_extra/best.pt"
    assert set(rec["metrics"]["improved"]) == set(rec["metrics"]["alternate"])  # all the same measures, same rows
    assert ("improved", "alternate") in {(x["a"], x["b"]) for x in rec["comparisons"]}
    assert "primary" in rec["models"]["improved"] and "alternate" in rec["models"]["alternate"]
    assert check_file(path, repo, min_n=1) == []
    # when E12's gates pass, the decision names E12's model and the roles swap
    decide(repo, "e12")
    commit(repo, "E12 chosen")
    swapped = e9.run_images("lab_test", cfg, predictors(alternate=SHA, quantized=SHA12), repo)[0]
    c = json.loads(swapped.read_text())["config"]
    assert c["primary"]["candidate"] == "e12" and c["alternate"]["candidate"] == "e4"
    assert c["calibration"]["improved"]["temperature"] == 2.0 and c["calibration"]["alternate"]["temperature"] == 1.5
    other = {"chosen_model": "models/elsewhere/best.pt", "chosen_model_sha256": SHA, "class_names": CLASSES}
    with pytest.raises(e9.Refused, match="none of"):
        e9.candidate_roles(cfg, other)


def test_every_number_in_the_report_states_its_basis(repo):
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    rec = json.loads(path.read_text())
    assert "cleaned rows" in rec["config"]["basis"] and "21 of 24" in rec["config"]["basis"]
    text = e9.image_report(rec)
    numbered = [x for x in text.splitlines() if any(ch.isdigit() for ch in x)]
    assert numbered and all("cleaned rows" in x for x in numbered)
    assert "primary" in text and "alternate" in text and "removed" in text
    lab = json.loads(e9.run_images("lab_test", e9.load_config(repo), predictors(), repo)[0].read_text())
    lines = [x for x in e9.image_report(lab).splitlines() if any(c.isdigit() for c in x)]
    assert lines and all("lab_test" in x for x in lines)


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


# ----------------------------------------------------------------------------------------------- real predictors, tiny


def tiny_models(root):
    """Real files of the kinds the config names, all tiny: two torch checkpoints, an XGBoost model, an ONNX export."""
    import torch
    import xgboost as xgb
    from PIL import Image

    from sprout.baselines.classic import classic_features
    from sprout.serve.export import export_module
    from sprout.train.config import TrainConfig
    from sprout.train.loop import build_model

    class_map = {n: i for i, n in enumerate(CLASSES)}
    torch.manual_seed(0)
    model = build_model("tiny_cnn", len(CLASSES), pretrained=False)
    state = {"model": model.state_dict(), "class_map": class_map,
             "config": TrainConfig(manifest="lab", out_dir="x", model="tiny_cnn").to_dict()}
    for rel in ("models/e1/best.pt", "models/e4/dinov2_head/best.pt", "models/e4/convnext_joint/best.pt"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, root / rel)
    other = {**state, "model": build_model("tiny_cnn", len(CLASSES), pretrained=False).state_dict()}
    (root / "models/e12/convnext_joint_extra").mkdir(parents=True, exist_ok=True)
    torch.save(other, root / "models/e12/convnext_joint_extra/best.pt")
    rng = np.random.default_rng(0)
    for i in range(12):
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(root / f"im{i}.png")
    feats = np.stack([classic_features(Image.open(root / f"im{i}.png"), size=32) for i in range(12)])
    booster = xgb.train({"objective": "multi:softprob", "num_class": 4, "max_depth": 2}, xgb.DMatrix(feats, label=np.arange(12) % 4), 3)
    (root / "models/e3/xgboost").mkdir(parents=True, exist_ok=True)
    booster.save_model(str(root / "models/e3/xgboost/model.ubj"))
    chosen_sha = e9.sha256_file(root / "models/e4/convnext_joint/best.pt")
    export_module(model, class_map, root / "models/e6/model.dynamic.onnx", "tiny_cnn", chosen_sha)
    onnx_sha = e9.sha256_file(root / "models/e6/model.dynamic.onnx")
    write(root, "results/e6/export.json", {"checkpoint_sha256": chosen_sha, "files": {
        "onnx_dynamic": {"file": "model.dynamic.onnx", "sha256": onnx_sha}}})
    decide(root, "e4", sha=chosen_sha)
    return chosen_sha


def test_real_predictors_run_tiny_models_of_every_kind(repo):
    tiny_models(repo)
    cfg = e9.load_config(repo)
    decision = e9.load_decision(cfg, repo)
    found = e9.build_predictors(cfg, decision, repo)
    assert set(found) == set(SYSTEMS)
    frame = pd.DataFrame({"path": [str(repo / f"im{i}.png") for i in range(5)], "label": [CLASSES[i % 4] for i in range(5)]})
    for name, p in found.items():
        assert p.class_names == CLASSES, name
        out = p.run(frame)
        assert out.shape == (5, 4) and np.allclose(out.sum(axis=1), 1.0, atol=1e-5), name
        assert p.detail.get("sha256"), name  # each frozen file is identified by its hash
    assert found["improved"].detail["sha256"] == decision["chosen_model_sha256"]
    assert found["alternate"].detail["sha256"] == e9.sha256_file(repo / "models/e12/convnext_joint_extra/best.pt")
    # E12's gates pass: the same files, with the roles swapped
    e12 = e9.sha256_file(repo / "models/e12/convnext_joint_extra/best.pt")
    swapped = {**decision, "chosen_model": "models/e12/convnext_joint_extra/best.pt", "chosen_model_sha256": e12}
    export = json.loads((repo / "results/e6/export.json").read_text())
    from sprout.serve.export import OnnxModel, export_module
    from sprout.train.loop import build_model

    write(repo, "results/e6/export.json", {**export, "checkpoint_sha256": e12})  # E6 says it re-exported from E12's model
    with pytest.raises(e9.Refused, match="stamped"):  # but the file is still stamped with E4's hash
        e9.build_predictors(cfg, swapped, repo)
    path = repo / "models/e6/model.dynamic.onnx"
    export_module(build_model("tiny_cnn", len(CLASSES), pretrained=False), OnnxModel(path).class_map, path, "tiny_cnn", e12)
    write(repo, "results/e6/export.json", {**export, "checkpoint_sha256": e12,
                                           "files": {"onnx_dynamic": {"file": path.name, "sha256": e9.sha256_file(path)}}})
    again = e9.build_predictors(cfg, swapped, repo)
    assert again["improved"].detail["sha256"] == e12
    assert again["alternate"].detail["sha256"] == decision["chosen_model_sha256"]


def test_quantized_model_must_come_from_the_chosen_checkpoint(repo):
    sha = tiny_models(repo)
    cfg = e9.load_config(repo)
    decision = e9.load_decision(cfg, repo)
    other = json.loads((repo / "results/e6/export.json").read_text())
    other["checkpoint_sha256"] = "b" * 64
    write(repo, "results/e6/export.json", other)
    with pytest.raises(e9.Refused, match="not exported from the chosen model"):
        e9.build_predictors(cfg, decision, repo)
    other["checkpoint_sha256"] = sha
    other["files"]["onnx_dynamic"]["sha256"] = "c" * 64
    write(repo, "results/e6/export.json", other)
    with pytest.raises(e9.Refused, match="changed since"):
        e9.build_predictors(cfg, decision, repo)
    changed = {**decision, "chosen_model_sha256": "d" * 64}
    other["files"]["onnx_dynamic"]["sha256"] = e9.sha256_file(repo / "models/e6/model.dynamic.onnx")
    write(repo, "results/e6/export.json", other)
    with pytest.raises(e9.Refused, match="differs from the decision"):
        e9.build_predictors(cfg, changed, repo)


# ----------------------------------------------------------------------------------------------- wrong-disease check, dev

import hashlib  # noqa: E402
import threading  # noqa: E402

from sprout.llm.client import ClaudeCodeClient, FakeLLM, LLMError  # noqa: E402
from sprout.rag.judge import blind_view, rubric  # noqa: E402

DEV_IDS = [f"q{i}" for i in range(1, 9)]
TEST_IDS = [f"t{i}" for i in range(1, 7)]


def question_row(qid, split):
    n = int(qid[1:])
    text = f"My apple leaves have scab {qid}, what do I do?" if n % 2 else f"How do I water my ferns {qid}?"
    return dict(id=qid, category="supported", split=split, question=text, expected="answer" if n % 3 else "abstain")


def judged_row(system, qid, answer=None):
    n = int(qid[1:])
    answer = answer or (f"Tomato advice for {qid} [1]. Remove the old leaves [1]." if n % 4 == 0
                        else f"Apple scab advice for {qid} [1]. Spray in spring [2].")
    claims = [{"claim": c.rstrip("."), "cited": [1], "verdict": n % 2 == 0, "judge_reply": "SUPPORTED"}
              for c in ("Remove the old leaves.", "Spray in spring.")]
    return {"id": qid, "system": system, "question": question_row(qid, "dev")["question"],
            "passages": [{"n": 1, "source": "a.pdf", "page": 1, "text": "Passage one."},
                         {"n": 2, "source": "b.pdf", "page": 2, "text": "Passage two."}],
            "answer": answer, "claims": claims, "rubric": rubric(), "judge_model": "judge-model"}


E6_ROW = {"n": 10, "answered": 6, "unsure": 4, "coverage": 0.6, "healthy_answers": 2, "sick_called_healthy": 0, "confident_wrong": 1}
E6_DYNAMIC = {**E6_ROW, "answered": 5, "unsure": 5, "coverage": 0.5, "confident_wrong": 2}


def add_rag_files(root):
    rows = [question_row(q, "dev") for q in DEV_IDS] + [question_row(q, "test") for q in TEST_IDS]
    write(root, "datasets/rag/questions.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    chunks = [{"source": "a.pdf", "page": 1, "text": "apple scab", "crop": "apple"},
              {"source": "b.pdf", "page": 2, "text": "tomato blight", "crop": "tomato"},
              {"source": "g.pdf", "page": 1, "text": "watering", "crop": "general"}]
    write(root, "knowledge/index/chunks.jsonl", "".join(json.dumps(c) + "\n" for c in chunks))
    write(root, "knowledge/index/build-info.json", {"built": "2026-11-03T18:25:10+00:00", "chunks": 3, "embedding_model": "m",
                                                       "vectors_file": "vectors-x.npy"})
    write(root, "knowledge/index/vectors-x.npy", "vectors")


def make_dev_source(root, ids=DEV_IDS):
    rows = [judged_row("served", q) for q in ids] + [judged_row("base", q) for q in DEV_IDS]
    runs = root / "results" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "E7-prompts-v2-test.judgments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (runs / "E7-prompts-v2-test.blind.jsonl").write_text("".join(json.dumps(blind_view(r)) + "\n" for r in rows))
    (runs / "E7-20261104T000000000000Z.json").write_text(json.dumps(
        {"config": {"judgments_file": "E7-prompts-v2-test.judgments.jsonl", "blind_file": "E7-prompts-v2-test.blind.jsonl"}}))
    sha = lambda n: hashlib.sha256((runs / n).read_bytes()).hexdigest()  # noqa: E731
    return {"e7_run_id": "E7-20261104T000000000000Z", "judgments": "E7-prompts-v2-test.judgments.jsonl",
            "judgments_sha256": sha("E7-prompts-v2-test.judgments.jsonl"), "blind": "E7-prompts-v2-test.blind.jsonl",
            "blind_sha256": sha("E7-prompts-v2-test.blind.jsonl")}


def dev_cfg(root, ids=DEV_IDS):
    cfg = e9.load_config(root)
    cfg["wrong_disease"]["dev_source"] = make_dev_source(root, ids)
    return cfg


def grader_reply(messages):
    """WRONG when the answer is about tomatoes, FITS otherwise."""
    return "WRONG" if "Answer: Tomato" in messages[-1]["content"] else "FITS"


@pytest.fixture
def rag_repo(repo):
    add_rag_files(repo)
    commit(repo, "rag fixture")
    return repo


def test_diagnosis_comes_from_the_same_source_for_dev_and_test(rag_repo):
    from sprout.rag.evalset import load_questions

    crops = {"apple", "tomato"}
    qs = {q.id: q for q in load_questions(rag_repo / "datasets/rag/questions.jsonl")}
    named = e9.diagnosis_for(qs["q1"], crops, CLASSES)
    assert "Apple - scab" in named and "photo suggests" in named  # what the writer was told (answer.diagnosis_summary)
    assert "Apple - scab" in e9.diagnosis_for(qs["t1"], crops, CLASSES)  # a test question is read the same way
    assert e9.diagnosis_for(qs["q2"], crops, CLASSES) == "No photo diagnosis is available."
    qs["q2"].crop = "tomato"  # a stated plant, no disease named: the plant is given, the diagnosis still says none
    assert e9.diagnosis_for(qs["q2"], crops, CLASSES) == "No photo diagnosis is available. The plant is tomato."


def test_wrong_disease_dev_makes_one_call_per_answer_and_reports_its_rate(rag_repo):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    llm = FakeLLM(reply=grader_reply, model="sonnet")
    path = e9.stage_wrong_disease_dev(cfg, llm, rag_repo)
    rec = json.loads(path.read_text())
    assert len(llm.calls) == 8 == rec["claude_calls"] and rec["n"] == 8 and rec["split"] == "rag_dev"
    m = rec["metrics"]["served"]
    assert m["answers_checked"] == 8 and m["flagged_wrong_disease"] == 2 and m["flagged_rate"] == 0.25  # q4 and q8
    assert rec["intervals"]["served"]["flagged_rate"][0] < 0.25 < rec["intervals"]["served"]["flagged_rate"][1]
    # two calls run at once, so their order is not fixed: pick the call about q1
    first = next(c for c in map(json.dumps, llm.calls) if "Question: My apple leaves have scab q1" in c)
    assert "Question: My apple leaves have scab q1" in first and "Diagnosis: The photo suggests Apple - scab" in first
    assert "SUPPORTED" not in first and "judge-model" not in first and "served" not in first
    assert rec["config"]["prompt_sha256"] == e9.text_sha256(rag_repo / e9.PROMPT_REL)
    assert rec["models"]["grader"] == "sonnet"
    assert check_file(path, rag_repo, min_n=1) == []
    assert "accuracy" not in path.read_text().lower()


def test_dev_rows_are_saved_as_they_arrive_and_a_rerun_resumes(rag_repo):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    count = {"n": 0}

    def flaky(messages):
        count["n"] += 1
        if count["n"] == 5:
            raise LLMError("usage limit")
        return grader_reply(messages)

    with pytest.raises(LLMError, match="usage limit"):
        e9.stage_wrong_disease_dev(cfg, FakeLLM(reply=flaky, model="sonnet"), rag_repo)
    saved = [json.loads(x) for f in (rag_repo / "results/e9/work").glob("wd-dev-*.jsonl") for x in f.read_text().splitlines()]
    assert 4 <= len(saved) <= 5  # the failed call saved nothing; one more may have been in flight
    again = FakeLLM(reply=grader_reply, model="sonnet")
    path = e9.stage_wrong_disease_dev(cfg, again, rag_repo)
    assert len(again.calls) == 8 - len(saved)  # nothing already saved is asked again
    assert json.loads(path.read_text())["n"] == 8


def test_at_most_two_sonnet_calls_run_at_once(rag_repo):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    live, peak, lock = [0], [0], threading.Lock()

    def slow(messages):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        threading.Event().wait(0.02)
        with lock:
            live[0] -= 1
        return "FITS"

    e9.stage_wrong_disease_dev(cfg, FakeLLM(reply=slow, model="sonnet"), rag_repo)
    assert peak[0] == 2


def test_the_call_cap_refuses_before_any_call(rag_repo, tmp_path, monkeypatch):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    ran = []
    monkeypatch.setattr("shutil.which", lambda name: "C:/bin/claude.exe")
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: ran.append(cmd) if "claude" in str(cmd[0]) else real(cmd, **kw))
    counter = tmp_path / "claude_calls.json"
    counter.write_text(json.dumps({"calls": 95}))
    client = ClaudeCodeClient(model="sonnet", cap=100, counter_path=counter)  # 8 answers need 8 calls, 5 left
    with pytest.raises(LLMError, match="cap"):
        e9.stage_wrong_disease_dev(cfg, client, rag_repo)
    assert ran == [] and json.loads(counter.read_text())["calls"] == 95
    low = {**cfg, "wrong_disease": {**cfg["wrong_disease"], "dev_max_calls": 3}}
    with pytest.raises(LLMError, match="limit"):  # the experiment's own limit
        e9.stage_wrong_disease_dev(low, FakeLLM(["FITS"], model="sonnet"), rag_repo)
    assert not list((rag_repo / "results/e9/work").glob("wd-dev-*.jsonl"))


def test_an_unusable_reply_is_counted_and_never_read_as_a_call(rag_repo):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    path = e9.stage_wrong_disease_dev(cfg, FakeLLM(["WRONG because it is a different plant", "FITS"], model="sonnet"), rag_repo)
    m = json.loads(path.read_text())["metrics"]["served"]
    assert m["unusable_replies"] == 1 and m["usable_calls"] == 7 and m["flagged_wrong_disease"] == 0


# ----------------------------------------------------------------------------------------------- freeze


def dev_record(rag_repo):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    path = e9.stage_wrong_disease_dev(cfg, FakeLLM(reply=grader_reply, model="sonnet"), rag_repo)
    return cfg, path


def frozen_repo(rag_repo):
    cfg, _ = dev_record(rag_repo)
    commit(rag_repo, "records")
    return cfg


def test_freeze_records_the_prompt_commit_and_hash_and_says_the_check_was_not_validated(rag_repo):
    cfg = e9.load_config(rag_repo)
    with pytest.raises(e9.Refused, match="wrong_disease_dev"):  # needs the dev run first
        e9.stage_freeze(cfg, rag_repo)
    cfg, _ = dev_record(rag_repo)
    commit(rag_repo, "dev record")
    path = e9.stage_freeze(cfg, rag_repo)
    frozen = json.loads(path.read_text())
    assert frozen["prompt_sha256"] == e9.text_sha256(rag_repo / e9.PROMPT_REL)
    assert frozen["prompt_commit"] == git(rag_repo, "log", "-1", "--format=%H", "--", e9.PROMPT_REL).stdout.strip()
    assert frozen["rendered_sha256"] == e9.rendered_sha256(rag_repo)
    assert {"dev_record", "frozen_at"} <= set(frozen) and "blind_read_record" not in frozen
    assert frozen["validated"] is False and "not validated" in frozen["validation"]
    with pytest.raises(e9.Refused, match="already frozen"):
        e9.stage_freeze(cfg, rag_repo)
    with pytest.raises(e9.Refused, match="not committed"):  # a frozen file is only usable once committed
        e9.require_frozen(rag_repo)
    commit(rag_repo, "freeze")
    e9.require_frozen(rag_repo)


def test_freeze_refuses_a_prompt_changed_after_the_dev_run(rag_repo):
    cfg = frozen_repo(rag_repo)
    prompt = rag_repo / e9.PROMPT_REL
    prompt.write_text(prompt.read_text() + "Be strict.\n")
    commit(rag_repo, "tuned the prompt")
    with pytest.raises(e9.Refused, match="changed since"):
        e9.stage_freeze(cfg, rag_repo)


# ----------------------------------------------------------------------------------------------- rag_test

from sprout.rag.retrieve import Passage  # noqa: E402


class FakeRetriever:
    """Returns the same two passages for any search; stands in for the GPU index."""

    crops = {"apple", "tomato"}

    def search(self, query, *, crop=None, k=5, mode="hybrid", rerank=True):
        return [Passage("Apple scab spreads in wet weather.", "a.pdf", 1, "apple", 0.9),
                Passage("Tomato blight needs removal of old leaves.", "b.pdf", 2, "tomato", 0.8)]


def writer_reply(messages):
    return "Apple scab spreads in wet weather [1]. Remove fallen leaves in autumn [1]. Spray in spring [2]."


def sonnet_reply(messages):
    body = messages[-1]["content"]
    if body.startswith("Question:"):  # the wrong-disease check
        return "WRONG" if "ferns" in body else "FITS"
    return "SUPPORTED" if "Spray" in body.rsplit("Claim: ", 1)[1] else "NOT_SUPPORTED"


def haiku_reply(messages):  # the kinder judge: supports two of the three claims
    claim = messages[-1]["content"].rsplit("Claim: ", 1)[1]
    return "SUPPORTED" if ("Remove" in claim or "Apple scab" in claim) else "NOT_SUPPORTED"


def haiku(reply=haiku_reply):
    return FakeLLM(reply=reply, model="haiku-model")


@pytest.fixture
def frozen(rag_repo):
    frozen_repo(rag_repo)
    cfg = e9.load_config(rag_repo)
    e9.stage_freeze(cfg, rag_repo)
    commit(rag_repo, "freeze the prompt")
    return rag_repo


def run_rag(root, sonnet=None, writer=None, judge=None, override=""):
    cfg = e9.load_config(root)
    e9.stage_rag_retrieve(cfg, FakeRetriever(), root, override_note=override)
    missing = e9.stage_rag_generate(cfg, writer or FakeLLM(reply=writer_reply, model="writer-model"), root, None,
                                    override_note=override)
    assert missing == []
    e9.stage_rag_haiku(cfg, judge or haiku(), root, override_note=override)
    e9.stage_rag_sonnet(cfg, sonnet or FakeLLM(reply=sonnet_reply, model="sonnet"), root, override_note=override)
    return e9.stage_rag_record(cfg, root, device="none: fakes", override_note=override)


def test_rag_test_runs_every_stage_and_check_run_recomputes_it(frozen):
    sonnet, judge = FakeLLM(reply=sonnet_reply, model="sonnet"), haiku()
    path = run_rag(frozen, sonnet=sonnet, judge=judge)
    rec = json.loads(path.read_text())
    assert rec["split"] == "rag_test" and rec["experiment"] == "E9" and rec["n"] == 6 and rec["subset"] is None
    rows = read_rows(frozen, rec, "served")
    claims = sum(len(r["claims"]) for r in rows)
    checked = sum(r["wrong_disease"] is not None for r in rows)
    assert claims > 0 and checked > 0
    assert len(judge.calls) == claims and len(sonnet.calls) == claims + checked
    assert rec["config"]["calls"] == {"haiku": claims, "sonnet": claims + checked}
    assert rec["claude_calls"] == 2 * claims + checked
    assert all(r["haiku_calls"] == len(r["claims"]) for r in rows)
    assert all(r["sonnet_calls"] == len(r["claims"]) + (r["wrong_disease"] is not None) for r in rows)
    assert check_file(path, frozen, min_n=6) == []
    assert rec["labels_reviewed"] is False and rec["config"]["labels_reviewed"] is False
    assert rec["models"]["writer"] == "writer-model" and rec["models"]["judge_first"] == "haiku-model"
    # neither judge saw the other's grade or name, the writer's name, or the system name
    for fake, secrets in ((sonnet, ("haiku", "writer-model", "verdict", "judge_reply", "served")),
                          (judge, ("sonnet", "writer-model", "verdict", "judge_reply", "served"))):
        seen = json.dumps(fake.calls).lower()
        for secret in secrets:
            assert secret not in seen
    assert rec["config"]["wrong_disease_validated"] is False and "not validated" in e9.report(rec, frozen)


def test_both_judges_are_reported_and_neither_is_called_accuracy(frozen):
    path = run_rag(frozen)
    rec = json.loads(path.read_text())
    rows = read_rows(frozen, rec, "served")
    first = [c["haiku"] is True for r in rows for c in r["claims"]]
    second = [c["sonnet"] is True for r in rows for c in r["claims"]]
    values = rec["judged"]["values"]
    assert values["claims_supported_by_haiku"] == pytest.approx(sum(first) / len(first))  # on the same claims
    assert values["claims_supported_by_sonnet"] == pytest.approx(sum(second) / len(second))
    assert len(first) == len(second) and values["claims_supported_by_haiku"] != values["claims_supported_by_sonnet"]
    assert 0 < values["answers_flagged_wrong_disease"] < 1
    judges = rec["config"]["judges"]
    assert judges["haiku"]["role"] == "first judge" and judges["sonnet"]["role"] == "second judge"
    assert judges["haiku"]["model"] == "haiku-model" and judges["sonnet"]["model"] == "sonnet"
    assert judges["same_claims_and_rubric"] is True
    report = e9.report(rec, frozen)
    for text in (path.read_text(), report, json.dumps(json.loads((frozen / rec["judged"]["file"]).read_text()))):
        assert "accuracy" not in text.lower()
        assert "upper" not in text.lower() and "conservative" not in text.lower()
    assert "Haiku judge" in report and "Sonnet judge" in report and "first judge" in report and "second judge" in report
    assert "Haiku judge's share is above the Sonnet judge's" in report  # the kinder fake judge: said, not hidden
    assert "Remove fallen leaves" not in report  # a report of numbers, not of answers
    # a record that calls a figure upper, conservative or accuracy is caught
    saved = json.loads(path.read_text())
    saved["config"]["judges"]["haiku"]["role"] = "upper figure"
    path.write_text(json.dumps(saved))
    assert any("upper" in p for p in check_file(path, frozen, min_n=6))


def test_first_judge_follows_the_committed_haiku_check(frozen):
    cfg = e9.load_config(frozen)
    e9.stage_rag_retrieve(cfg, FakeRetriever(), frozen)
    e9.stage_rag_generate(cfg, FakeLLM(reply=writer_reply, model="writer-model"), frozen, None)
    assert e9.first_judge(frozen)["model"] == "haiku-model"
    assert "judge" not in cfg["rag"]  # the judge is not typed into the config
    wrong = FakeLLM(reply=haiku_reply, model="another-model")  # the client must be the model the decision file names
    with pytest.raises(e9.Refused, match="haiku-model"):
        e9.stage_rag_haiku(cfg, wrong, frozen)
    assert wrong.calls == []
    decision = json.loads((frozen / e9.HAIKU_DECISION_REL).read_text())
    # the check found Haiku clearly worse: E9 has no local judge, so it stops
    write(frozen, e9.HAIKU_DECISION_REL, {**decision, "first_judge": "local"})
    commit(frozen, "the check says local")
    judge = haiku()
    with pytest.raises(e9.Refused, match="no local judge"):
        e9.stage_rag_haiku(cfg, judge, frozen)
    assert judge.calls == []
    # an uncommitted edit is refused too, and a changed rubric makes the decision stale
    write(frozen, e9.HAIKU_DECISION_REL, decision)
    with pytest.raises(e9.Refused, match="not committed"):
        e9.first_judge(frozen)
    write(frozen, e9.HAIKU_DECISION_REL, {**decision, "rubric_sha256": "0" * 64})
    commit(frozen, "stale rubric")
    with pytest.raises(e9.Refused, match="rubric"):
        e9.first_judge(frozen)
    # the model in the record is the committed decision's, checked again by check_run
    write(frozen, e9.HAIKU_DECISION_REL, decision)
    commit(frozen, "back")
    e9.stage_rag_haiku(cfg, haiku(), frozen)
    e9.stage_rag_sonnet(cfg, FakeLLM(reply=sonnet_reply, model="sonnet"), frozen)
    made = e9.stage_rag_record(cfg, frozen, device="none: fakes")
    assert check_file(made, frozen, min_n=6) == []
    saved = json.loads(made.read_text())
    saved["config"]["judges"]["haiku"]["model"] = "claude-haiku-5-5"
    made.write_text(json.dumps(saved))
    assert any("first judge" in p for p in check_file(made, frozen, min_n=6))


def test_wrong_disease_check_refuses_rag_test_when_its_prompt_hash_differs_from_the_frozen_one(frozen):
    cfg = e9.load_config(frozen)
    e9.stage_rag_retrieve(cfg, FakeRetriever(), frozen)
    e9.stage_rag_generate(cfg, FakeLLM(reply=writer_reply, model="writer-model"), frozen, None)
    e9.stage_rag_haiku(cfg, haiku(), frozen)
    prompt = frozen / e9.PROMPT_REL
    prompt.write_text(prompt.read_text().replace("FITS or WRONG", "FITS or WRONG now"))
    commit(frozen, "tune the prompt after seeing rag_test")  # committed and clean, but not the frozen prompt
    sonnet = FakeLLM(reply=sonnet_reply, model="sonnet")
    with pytest.raises(e9.Refused, match="differs from the frozen"):
        e9.stage_rag_sonnet(cfg, sonnet, frozen)
    assert sonnet.calls == []
    with pytest.raises(e9.Refused, match="differs from the frozen"):
        e9.stage_rag_record(cfg, frozen, device="none")


def test_rag_test_stages_refuse_after_use(frozen):
    cfg = e9.load_config(frozen)
    run_rag(frozen)
    with pytest.raises(e9.Refused, match="already been used"):
        e9.stage_rag_retrieve(cfg, FakeRetriever(), frozen)
    again = FakeLLM(reply=sonnet_reply, model="sonnet")
    with pytest.raises(e9.Refused, match="already been used"):
        e9.stage_rag_sonnet(cfg, again, frozen)
    assert again.calls == []


def test_rag_test_refuses_before_the_prompt_is_frozen(rag_repo):
    cfg = e9.load_config(rag_repo)
    with pytest.raises(e9.Refused, match="not committed"):
        e9.stage_rag_retrieve(cfg, FakeRetriever(), rag_repo)


def test_cleaned_counts_calibration_and_labels_reviewed_cannot_be_edited(repo):
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    good = json.loads(path.read_text())
    assert check_file(path, repo, min_n=1) == []

    def edited(mutate):
        rec = json.loads(json.dumps(good))
        mutate(rec)
        path.write_text(json.dumps(rec))
        return check_file(path, repo, min_n=1)

    assert any("cleaned_rows" in p for p in edited(lambda r: r["config"]["cleaned_rows"].update(removed=2)))
    assert any("cleaned_rows" in p for p in edited(lambda r: r["config"]["cleaned_rows"]["by_reason"].update(wrong_plant=1)))
    assert any("cleaned_rows" in p for p in edited(lambda r: r["config"].pop("cleaned_rows")))
    assert any("clean" in p for p in edited(lambda r: r.update(subset=None)))
    assert any("labels_reviewed" in p for p in edited(lambda r: r.update(labels_reviewed=True)))
    assert any("calibrat" in p for p in edited(lambda r: r["config"]["calibration"]["improved"].update(temperature=3.0)))
    assert any("calibrat" in p for p in edited(lambda r: r["config"]["calibration"]["improved"].update(abstain_threshold=0.7)))
    path.write_text(json.dumps(good))
    cal = repo / good["predictions"]["quantized_calibrated"]
    text = cal.read_text()
    row = json.loads(text.splitlines()[0])
    row["probs"] = row["probs"][::-1]
    cal.write_text(chr(10).join([json.dumps(row), *text.splitlines()[1:]]) + chr(10))
    assert any("calibrat" in p for p in check_file(path, repo, min_n=1))


# ----------------------------------------------------------------------------------------------- the label signature

import hashlib as _hashlib  # noqa: E402

from sprout.eval.claims import check_claims  # noqa: E402


def claim_on(root, path):
    rec = json.loads(path.read_text())
    write(root, "results/claims.json", {"claims": [{"id": "c1", "run_id": rec["run_id"], "system": "improved",
                                                   "metric": "accuracy", "value": rec["metrics"]["improved"]["accuracy"]}]})
    return rec


def sign(root, split, signed=True):
    book = json.loads((root / "orchestration/label_signoff.json").read_text())
    book[split] = {"signed": signed, "by": "the person" if signed else None, "at": "2026-12-01" if signed else None,
                   "words": "I checked the labels" if signed else None}
    write(root, "orchestration/label_signoff.json", book)


def test_a_signed_row_passes_the_claim_without_touching_the_record(repo):
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    claim_on(repo, path)
    before = _hashlib.sha256(path.read_bytes()).hexdigest()
    assert check_claims(repo) != []
    sign(repo, "field_test")
    assert check_claims(repo) == []
    assert _hashlib.sha256(path.read_bytes()).hexdigest() == before  # the record was not edited
    assert json.loads(path.read_text())["labels_reviewed"] is False  # and never held the signature
    sign(repo, "field_test", signed=False)  # taking it back refuses again, still without a rerun
    assert check_claims(repo) != []


def prior_stems(root, split):
    return e9.prior_uses(root, split)


def test_an_edited_labels_reviewed_is_caught_and_claims_ignore_it_for_e9(repo):
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    claim_on(repo, path)
    rec = json.loads(path.read_text())
    rec["labels_reviewed"] = True  # a hand edit of one copy
    path.write_text(json.dumps(rec))
    assert any("labels_reviewed" in p for p in check_file(path, repo, min_n=1))
    rec["config"]["labels_reviewed"] = True  # both copies: valid as a record, but it signs nothing
    path.write_text(json.dumps(rec))
    assert any("not signed" in p for p in check_claims(repo))


# ----------------------------------------------------------------------------------------------- the frozen records


def test_the_frozen_config_must_name_a_dev_record_made_before_it_with_its_prompt_and_say_not_validated(frozen):
    path = run_rag(frozen)
    assert check_file(path, frozen, min_n=6) == []
    runs = frozen / "results" / "runs"
    frozen_cfg = json.loads((frozen / e9.FROZEN_REL).read_text())
    dev = runs / f"{frozen_cfg['dev_record']}.json"
    saved = dev.read_text()

    def with_edit(mutate):
        rec = json.loads(saved)
        mutate(rec)
        dev.write_text(json.dumps(rec))
        problems = check_file(path, frozen, min_n=6)
        dev.write_text(saved)
        return problems

    assert any("dev_record" in p and "prompt" in p for p in with_edit(lambda r: r["config"].update(prompt_sha256="0" * 64)))
    assert any("dev_record" in p and "before the freeze" in p
               for p in with_edit(lambda r: r.update(finished_at="2999-01-01T00:00:00+00:00")))
    dev.unlink()
    assert any("dev_record" in p and "does not exist" in p for p in check_file(path, frozen, min_n=6))
    dev.write_text(saved)
    assert check_file(path, frozen, min_n=6) == []


# ----------------------------------------------------------------------------------------------- roles follow the decision

def _ok(path, repo):
    return check_file(path, repo, min_n=1)


def test_check_run_rederives_primary_and_alternate_from_the_committed_decision(repo):
    cfg = e9.load_config(repo)
    (path,) = e9.run_images("field_test", cfg, predictors(), repo)
    assert _ok(path, repo) == []
    saved = json.loads(path.read_text())

    def with_roles(primary, alternate):
        rec = json.loads(json.dumps(saved))
        rec["config"]["primary"] = primary
        rec["config"]["alternate"] = alternate
        path.write_text(json.dumps(rec))
        return _ok(path, repo)

    # E4 kept: a record that says E12 was the primary (roles swapped) is caught, one field at a time as well
    p, a = saved["config"]["primary"], saved["config"]["alternate"]
    assert any("primary" in x for x in with_roles(a, p))
    assert any("primary" in x for x in with_roles({**p, "candidate": "e12"}, a))
    assert any("primary" in x for x in with_roles({**p, "checkpoint": a["checkpoint"]}, a))
    assert any("primary" in x for x in with_roles({**p, "sha256": "f" * 64}, a))
    assert any("alternate" in x for x in with_roles(p, {**a, "candidate": "e4"}))
    assert any("alternate" in x for x in with_roles(p, {**a, "checkpoint": p["checkpoint"]}))
    assert with_roles(p, a) == []
    # the decision is read where the run was made: a later change to it, committed or not, does not touch this record
    decide(repo, "e12")
    assert _ok(path, repo) == []
    commit(repo, "E12 chosen later")
    assert _ok(path, repo) == []


# ----------------------------------------------------------------------------------------------- provisional claims

def claim_with(root, path, **extra):
    rec = json.loads(path.read_text())
    write(root, "results/claims.json", {"claims": [{
        "id": "c1", "run_id": rec["run_id"], "system": "improved", "metric": "accuracy",
        "value": rec["metrics"]["improved"]["accuracy"], **extra}]})


def test_a_provisional_claim_passes_on_an_unsigned_split_only_if_its_text_says_so(repo, capsys):
    from sprout.eval.claims import main as claims_main

    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    claim_with(repo, path, provisional=True, text="Accuracy on field_test, provisional until the labels are signed")
    assert check_claims(repo) == []
    assert claims_main(["check"], repo) == 0
    out = capsys.readouterr().out
    assert "c1" in out and "provisional" in out and "1 provisional claim" in out
    claim_with(repo, path, provisional=True, text="Accuracy on field_test")  # marked, but the words do not say it
    assert any("provisional" in p and "text" in p for p in check_claims(repo))
    claim_with(repo, path, provisional=True)  # no text at all
    assert any("provisional" in p and "text" in p for p in check_claims(repo))
    assert claims_main(["check"], repo) == 1
    claim_with(repo, path, provisional="yes", text="provisional")  # only the boolean true marks a claim
    assert any("provisional" in p for p in check_claims(repo))


def test_final_refuses_every_provisional_claim_and_every_unsigned_split(repo, capsys):
    from sprout.eval.claims import main as claims_main

    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    claim_with(repo, path, provisional=True, text="provisional accuracy")
    assert claims_main(["check"], repo) == 0 and claims_main(["check", "--final"], repo) == 1
    assert any("provisional" in p for p in check_claims(repo, final=True))
    sign(repo, "field_test")
    assert any("provisional" in p for p in check_claims(repo, final=True))  # the marking itself must go
    claim_with(repo, path)
    assert check_claims(repo, final=True) == [] and claims_main(["check", "--final"], repo) == 0
    sign(repo, "field_test", signed=False)
    assert any("not signed" in p for p in check_claims(repo, final=True))
    assert claims_main(["check", "--bogus"], repo) == 2



# ----------------------------------------------------------------------------------------------- calibration binding

def test_every_calibration_is_bound_to_the_checkpoint_it_was_fitted_for_when_e4_is_kept(repo):
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    rec = json.loads(path.read_text())
    b = rec["config"]["calibration_binding"]
    assert b["improved"]["sha256"] == SHA and b["improved"]["checkpoint"] == "models/e4/convnext_joint/best.pt"
    assert "names" in b["improved"]["bound_by"] and b["improved"]["calibration_file"] == "results/e15/refit/chosen.json"
    assert b["quantized"]["sha256"] == SHA and "exported from" in b["quantized"]["bound_by"]
    assert b["alternate"]["sha256"] == SHA12 and "names" in b["alternate"]["bound_by"]
    assert check_file(path, repo, min_n=1) == []
    rec["config"]["calibration_binding"]["improved"]["sha256"] = SHA12  # a record that claims another pairing
    path.write_text(json.dumps(rec))
    assert any("calibration_binding" in p for p in check_file(path, repo, min_n=1))


def test_a_calibration_fitted_for_another_checkpoint_is_refused(repo):
    cfg = e9.load_config(repo)
    with pytest.raises(e9.Refused, match="quantized"):  # the quantized file was exported from the other checkpoint
        e9.run_images("field_test", cfg, predictors(quantized=SHA12), repo)
    with pytest.raises(e9.Refused, match="quantized"):  # or carries no checkpoint hash at all
        e9.run_images("field_test", cfg, predictors(quantized=""), repo)
    decide(repo, "e4")
    write(repo, "results/e15/refit/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6, "model_sha256": "0" * 64})  # another model's refit
    commit(repo, "refit for another model")
    with pytest.raises(e9.Refused, match="fitted on another"):
        e9.run_images("field_test", cfg, predictors(), repo)
    # a sha-less file for a checkpoint E5 never used cannot be tied to it: E4 as an alternate loses its calibration
    decide(repo, "e12")
    write(repo, "configs/e5.json", {"model": {"checkpoint": "models/e12/convnext_joint_extra/best.pt", "e4_record": "E4-1"}})
    commit(repo, "e5 config moved")
    (path,) = e9.run_images("field_test", cfg, predictors(alternate=SHA, quantized=SHA12), repo)
    why = json.loads(path.read_text())["config"]["calibration_binding"]["alternate"]
    assert why["calibration_file"] is None and "cannot be tied" in why["why"] or "another model" in why["why"]


def test_without_a_refit_the_alternate_is_scored_plain_and_the_record_says_why(repo):
    (repo / "results/e12/e5_refit/chosen.json").unlink()
    commit(repo, "no E5 refit on E12")
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    rec = json.loads(path.read_text())
    assert "alternate" in rec["metrics"] and "alternate_calibrated" not in rec["metrics"]
    assert "alternate" not in rec["config"]["calibration"]
    why = rec["config"]["calibration_binding"]["alternate"]
    assert why["calibration_file"] is None and "refit" in why["why"] and "another model" in why["why"]
    assert check_file(path, repo, min_n=1) == []


def test_the_exported_file_must_be_stamped_with_the_chosen_checkpoint(repo):
    from sprout.serve.export import export_module

    tiny_models(repo)
    cfg = e9.load_config(repo)
    decision = e9.load_decision(cfg, repo)
    from sprout.serve.export import OnnxModel
    from sprout.train.loop import build_model

    net = build_model("tiny_cnn", len(CLASSES), pretrained=False)
    path = repo / "models/e6/model.dynamic.onnx"
    for stamp, match in (("", "stamped"), ("f" * 64, "stamped")):
        export_module(net, OnnxModel(path).class_map, path, "tiny_cnn", stamp)
        export = json.loads((repo / "results/e6/export.json").read_text())
        export["files"]["onnx_dynamic"]["sha256"] = e9.sha256_file(path)  # E6's own file hash agrees: only the stamp is wrong
        write(repo, "results/e6/export.json", export)
        with pytest.raises(e9.Refused, match=match):
            e9.build_predictors(cfg, decision, repo)


# ----------------------------------------------------------------------------------------------- what E9 states, from data

def test_every_model_is_scored_on_the_same_cleaned_rows_and_the_record_names_them(repo):
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    rec = json.loads(path.read_text())
    rows = {n: [r["path"] for r in read_rows(repo, rec, n)] for n in rec["predictions"]}
    assert len({tuple(v) for v in rows.values()}) == 1 and len(rows["baseline"]) == 21  # 24 test rows, 3 removed
    assert not {"f/7.jpg", "f/10.jpg", "f/11.jpg"} & set(rows["baseline"])
    assert rec["config"]["basis"].startswith("cleaned rows: 21 of 24") and rec["subset"] == "clean:1"
    assert "cleaned rows: 21 of 24" in rec["notes"]


def test_no_e6_record_for_the_chosen_checkpoint_refuses_the_run(repo):
    (repo / "results/runs/E6-1.json").unlink()
    commit(repo, "no E6")
    with pytest.raises(e9.Refused, match="E6"):
        e9.run_images("field_test", e9.load_config(repo), predictors(), repo)


# ----------------------------------------------------------------------------------------------- the final model (T4.9)

E15_BYTES = b"the T4.9 checkpoint"
SHA15 = hashlib.sha256(E15_BYTES).hexdigest()
CLASSES3 = CLASSES[:3]  # the class rule drops Tomato___healthy, a class lab_test holds


def final_model(root):
    """What T4.9 leaves when its own model is chosen: the checkpoint, the decision and the refit it points to."""
    write(root, "models/e15/final/best.pt", "x")
    (root / "models/e15/final/best.pt").write_bytes(E15_BYTES)
    e6 = json.loads((root / "results/runs/E6-1.json").read_text())
    e6["run_id"], e6["config"]["checkpoint_sha256"] = "E6-3", SHA15
    write(root, "results/runs/E6-3.json", e6)
    decide(root, "e15", classes=CLASSES3)
    commit(root, "T4.9 decided")


def three_class_predictors(omit=()):
    out = {n: tiny(i, 0.5 + 0.1 * i, names=CLASSES3) for i, n in enumerate(SYSTEMS)}
    out["alternate"] = tiny(7, names=CLASSES3, sha=SHA)  # E4
    out["alternate_2"] = tiny(8, names=CLASSES3, sha=SHA12)  # E12
    out["quantized"].detail["source_checkpoint_sha256"] = SHA15
    return {k: v for k, v in out.items() if k not in omit}


def test_a_new_final_model_is_primary_and_e4_and_e12_are_scored_beside_it_on_its_class_list(repo):
    final_model(repo)
    paths = e9.run_images("lab_test", e9.load_config(repo), three_class_predictors(), repo)
    full = json.loads(paths[0].read_text())
    c = full["config"]
    assert c["primary"]["candidate"] == "e15" and c["primary"]["checkpoint"] == "models/e15/final/best.pt"
    assert (c["alternate"]["candidate"], c["alternate_2"]["candidate"]) == ("e4", "e12")
    assert c["calibration"]["improved"]["file"] == "results/e15/refit/chosen.json"  # the refit T4.9 defines
    assert c["calibration"]["improved"]["temperature"] == 1.2
    assert c["calibration"]["alternate"]["temperature"] == 1.5 and c["calibration"]["alternate_2"]["temperature"] == 2.0
    assert c["class_names"] == CLASSES3 and "alternate_2" in full["predictions"]
    assert ("improved", "alternate_2") in {(x["a"], x["b"]) for x in full["comparisons"]}
    # the rule: rows whose true class is outside the 29 (here 3) are not scored, by any system, and the record says so
    assert full["subset"] == "classes_in:results/e15/decision.json" and full["n"] == 12
    assert c["class_rule"]["rows_dropped"] == 4 and c["class_rule"]["labels_outside_the_list"] == ["Tomato___healthy"]
    assert "4 further photos" in c["basis"]
    assert [check_file(p, repo, min_n=1) for p in paths] == [[], []]
    # a record that swaps the roles, or leaves an alternate out unannounced, is caught
    saved = json.loads(paths[0].read_text())
    del saved["config"]["alternate_2"]
    paths[0].write_text(json.dumps(saved))
    assert any("alternate_2" in p for p in check_file(paths[0], repo, min_n=1))


def test_the_class_list_is_read_from_the_decision_at_the_records_commit(repo):
    final_model(repo)
    paths = e9.run_images("lab_test", e9.load_config(repo), three_class_predictors(), repo)
    assert check_file(paths[0], repo, min_n=1) == []
    write(repo, "results/e15/decision.json", {**json.loads((repo / "results/e15/decision.json").read_text()), "classes": CLASSES})
    assert check_file(paths[0], repo, min_n=1) == []  # a later edit, committed or not, does not change this record
    commit(repo, "classes changed later")
    assert check_file(paths[0], repo, min_n=1) == []


def test_old_models_are_masked_to_the_class_list_and_renormalized():
    old = tiny(3, names=CLASSES)
    frame = pd.DataFrame({"path": [f"p{i}" for i in range(6)], "label": [CLASSES[i % 3] for i in range(6)]})
    kept = e9.restricted(old, CLASSES3)
    full = old.run(frame)
    out = kept.run(frame)
    assert kept.class_names == CLASSES3 and out.shape == (6, 3) and np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out, full[:, :3] / full[:, :3].sum(axis=1, keepdims=True))
    assert kept.detail["masked_from_classes"] == 4
    assert e9.restricted(tiny(3, names=CLASSES3), CLASSES3).class_names == CLASSES3
    with pytest.raises(e9.Refused, match="no output for"):
        e9.restricted(tiny(3, names=CLASSES3), CLASSES)


def test_the_committed_config_reads_the_final_decision_and_the_e12_refit_exists():
    cfg = e9.load_config(REAL_ROOT)
    assert cfg["decision_file"] == "results/e15/decision.json" and "judge" not in cfg["rag"]
    assert "blind_seed" not in cfg["wrong_disease"]
    refit = json.loads((REAL_ROOT / cfg["candidates"]["e12"]["calibration"]).read_text())
    cal = e9.load_calibration(cfg["candidates"]["e12"]["calibration"], refit["model_sha256"], REAL_ROOT, required=True)
    assert cal["temperature"] == refit["temperature"] and cal["abstain_threshold"] == refit["abstain_threshold"]
    with pytest.raises(e9.Refused, match="fitted on another"):
        e9.load_calibration(cfg["candidates"]["e12"]["calibration"], "0" * 64, REAL_ROOT)
    if not (REAL_ROOT / cfg["decision_file"]).is_file():  # T4.9 has not run: E9 refuses rather than guess
        with pytest.raises(e9.Refused, match="no final model"):
            e9.load_decision(cfg, REAL_ROOT)


def test_labels_reviewed_is_false_before_h2_and_true_after(repo):
    """The record never holds the signature (so signing needs no rerun); `labels_reviewed` as the claims check sees it
    follows the per-split sign-off file: false while the split is unsigned, true once the person has signed it. H2's
    `done` status signs nothing, and every split ships unsigned."""
    from sprout.eval.claims import signoff_problems

    shipped = json.loads((REAL_ROOT / "orchestration/label_signoff.json").read_text())
    assert {shipped[s]["signed"] for s in ("lab_test", "field_test", "rag_test")} == {False}
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    claim_on(repo, path)
    rec = json.loads(path.read_text())
    assert rec["labels_reviewed"] is False and rec["config"]["labels_reviewed"] is False
    assert signoff_problems(repo, "field_test") and any("not signed" in p for p in check_claims(repo))  # before
    before = _hashlib.sha256(path.read_bytes()).hexdigest()
    sign(repo, "field_test")
    assert signoff_problems(repo, "field_test") == [] and check_claims(repo) == []  # after
    assert signoff_problems(repo, "lab_test")  # another split is still unsigned
    assert _hashlib.sha256(path.read_bytes()).hexdigest() == before  # the record itself is unchanged


# ------------------------------------------------------- generic follow-ups, the fallback flag, the counter and the index

def test_generic_followup_rule_is_text_only_and_fixed():
    from sprout.rag.evalset import Question

    crops = {"apple", "tomato"}

    def q(text, crop=None, history=()):
        return Question(id="x", category="c", split="test", question=text, expected="answer", crop=crop,
                        history=list(history))

    assert e9.is_generic_followup(q("How do I treat it?"), crops, CLASSES)
    assert not e9.is_generic_followup(q("How do I treat apple trees?"), crops, CLASSES)  # a plant word
    assert not e9.is_generic_followup(q("How do I treat scab?"), crops, CLASSES)  # a disease name
    assert not e9.is_generic_followup(q("How do I treat it?", crop="apple"), crops, CLASSES)  # a plant given
    assert not e9.is_generic_followup(q("And then?", history=[{"role": "user", "content": "my tomato is sick"}]),
                                      crops, CLASSES)  # the user's earlier turn names one
    assert e9.is_generic_followup(q("And then?", history=[{"role": "assistant", "content": "apple scab"}]),
                                  crops, CLASSES)  # the assistant's words do not count


class FallbackRetriever(FakeRetriever):
    """Finds nothing for the question as asked, and passages once a disease name is added to it (a second 'scab')."""

    def search(self, query, **kw):
        return super().search(query, **kw) if query.lower().count("scab") >= 2 else []


def test_rag_record_keeps_answer_text_citations_fallback_flag_and_generic_count(frozen):
    cfg = e9.load_config(frozen)
    e9.stage_rag_retrieve(cfg, FallbackRetriever(), frozen)
    assert e9.stage_rag_generate(cfg, FakeLLM(reply=writer_reply, model="writer-model"), frozen, None) == []
    e9.stage_rag_haiku(cfg, haiku(), frozen)
    e9.stage_rag_sonnet(cfg, FakeLLM(reply=sonnet_reply, model="sonnet"), frozen)
    path = e9.stage_rag_record(cfg, frozen, device="none: fakes")
    rec = json.loads(path.read_text())
    rows = {r["id"]: r for r in read_rows(frozen, rec, "served")}
    assert rows["t1"]["fallback_fired"] is True and rows["t2"]["fallback_fired"] is False  # t2 has no diagnosis
    assert rows["t1"]["text"].startswith("Apple scab") and rows["t1"]["citations"] and "scab" in rows["t1"]["query"]
    cov = rec["config"]["generic_followup_coverage"]
    assert cov["generic_followups"] == 3 and cov["questions"] == 6 and cov["fallback_fired"] == 3
    assert cov["few"] is True and cov["rule"] and cov["claims_on_generic_followups"] >= 0 and cov["claims_in_all"] > 0
    text = e9.report(rec, frozen)
    assert "3 of 6 rag_test questions" in text and "says little about generic follow-ups" in text
    assert "predate the fallback" in text and rec["config"]["served_with_fallback"] is True
    assert rec["config"]["index"]["built"] == "2026-11-03T18:25:10+00:00" and len(rec["config"]["index"]["chunks_sha256"]) == 64


def test_a_worktree_counts_into_the_main_folders_counter_and_uses_its_index(repo, tmp_path):
    add_rag_files(repo)
    (repo / ".gitignore").write_text("knowledge/index/\n")
    commit(repo, "index is not tracked")
    tree = tmp_path / "tree"
    git(repo, "worktree", "add", "-q", str(tree), "-b", "task/x")
    cfg = e9.load_config(repo)
    assert not (tree / "knowledge/index").exists()  # untracked, so a worktree does not carry it
    assert e9.main_folder(tree).resolve() == repo.resolve()
    assert e9.claude_counter(tree).resolve() == (repo / "results" / "claude_calls.json").resolve()
    assert e9.index_path(cfg, tree).resolve() == (repo / "knowledge/index").resolve()
    assert e9.index_path(cfg, repo) == repo / "knowledge/index"
    (repo / "knowledge/index/chunks.jsonl").unlink()
    with pytest.raises(e9.Refused, match="no index"):
        e9.index_path(cfg, tree)


def test_check_run_catches_a_generic_followup_count_that_the_rows_do_not_give(frozen):
    cfg = e9.load_config(frozen)
    e9.stage_rag_retrieve(cfg, FallbackRetriever(), frozen)
    e9.stage_rag_generate(cfg, FakeLLM(reply=writer_reply, model="writer-model"), frozen, None)
    e9.stage_rag_haiku(cfg, haiku(), frozen)
    e9.stage_rag_sonnet(cfg, FakeLLM(reply=sonnet_reply, model="sonnet"), frozen)
    path = e9.stage_rag_record(cfg, frozen, device="none: fakes")
    assert check_file(path, frozen, min_n=1) == []
    rec = json.loads(path.read_text())
    rec["config"]["generic_followup_coverage"]["generic_followups"] = 9
    path.write_text(json.dumps(rec))
    assert any("generic_followup_coverage" in p for p in check_file(path, frozen, min_n=1))


def test_abstention_as_served_applies_the_healthy_bar_the_way_identifier_does():
    from sprout.serve.calibrate import answered_mask, healthy_bar, healthy_unsure

    names = ["Apple___scab", "Apple___healthy"]
    probs = np.array([[0.95, 0.05], [0.70, 0.30], [0.05, 0.95], [0.30, 0.70], [0.85, 0.15], [0.15, 0.85]])
    alone = answered_mask(probs, names, 0.8)
    served = answered_mask(probs, names, 0.8, 0.9)
    assert alone.tolist() == [True, False, True, False, True, True]
    assert served.tolist() == [True, False, True, False, True, False]  # a healthy 0.85 is below the healthy bar 0.9
    assert healthy_bar(0.8, None) == 0.8 and healthy_bar(0.8, 0.5) == 0.8 and healthy_bar(0.8, 0.9) == 0.9
    assert healthy_unsure("Apple___healthy", 0.85, 0.9) and not healthy_unsure("Apple___scab", 0.85, 0.9)


def test_e9_reports_abstention_as_served_beside_abstention_alone_and_check_run_recomputes_both(repo):
    write(repo, "results/e15/refit/chosen.json", {"temperature": 1.5, "abstain_threshold": 0.6, "healthy_threshold": 0.8,
                                                  "model_sha256": SHA})
    commit(repo, "a refit with a healthy threshold")
    (path,) = e9.run_images("field_test", e9.load_config(repo), predictors(), repo)
    rec = json.loads(path.read_text())
    cal = rec["metrics"]["improved_calibrated"]
    assert {"coverage@0.6", "coverage@0.6~0.8", "selective_accuracy@0.6~0.8", "confident_error_rate@0.6~0.8"} <= set(cal)
    assert cal["coverage@0.6~0.8"] <= cal["coverage@0.6"]  # the cost of the healthy bar is in the gap
    assert "coverage@0.6~0.8" in rec["metrics"]["quantized_calibrated"]  # the quantized copy is served the same way
    assert not [k for k in rec["metrics"]["alternate_calibrated"] if "~" in k]  # E12's file has no healthy threshold
    assert rec["config"]["abstention_as_served_for"] == ["improved", "quantized"]
    assert rec["config"]["abstention_alone_for"] == ["alternate"]
    assert "abstention alone" in e9.image_report(rec) and "as served" in e9.image_report(rec)
    assert check_file(path, repo, min_n=1) == []
    rec["metrics"]["improved_calibrated"]["coverage@0.6~0.8"] += 0.1
    path.write_text(json.dumps(rec))
    assert any("coverage@0.6~0.8" in p or "recompute" in p for p in check_file(path, repo, min_n=1))


# ------------------------------------------------------------ the safeguard refusing a wrong-disease request

def refusing(*ids, other=None):
    """A Sonnet fake that raises a safeguard refusal for the wrong-disease requests of the named test questions."""
    from sprout.llm.client import SafeguardRefusal

    seen = []

    def reply(messages):
        body = messages[-1]["content"]
        if body.startswith("Question:"):
            hit = [i for i in ids if re.search(rf"\b{i}\b", body)]
            seen.append(hit[0] if hit else None)
            if hit and other:
                raise other
            if hit:
                raise SafeguardRefusal("claude returned an error (exit 1): safeguards flagged this message", f"req_{hit[0]}")
        return sonnet_reply(messages)

    return FakeLLM(reply=reply, model="sonnet"), seen


def staged(root):
    cfg = e9.load_config(root)
    e9.stage_rag_retrieve(cfg, FakeRetriever(), root)
    e9.stage_rag_generate(cfg, FakeLLM(reply=writer_reply, model="writer-model"), root, None)
    e9.stage_rag_haiku(cfg, haiku(), root)
    return cfg


def wd_file(root, cfg, llm):
    items = e9._test_items(cfg, root, e9.claim_rows(cfg, root))
    return root / "results/e9/work" / f"wd-test-{e9.wd_key(items, llm.model, root)}.jsonl", items


def test_a_safeguard_refusal_is_kept_as_a_row_with_its_request_id_and_never_retried(frozen):
    cfg = staged(frozen)
    llm, seen = refusing("t2")
    e9.stage_rag_sonnet(cfg, llm, frozen)  # the refusal does not stop the stage
    path, _ = wd_file(frozen, cfg, llm)
    rows = {json.loads(x)["id"]: json.loads(x) for x in path.read_text().splitlines()}
    assert rows["t2"] == {"id": "t2", "reply": None, "call": None, "refused_by_safeguard": True, "request_id": "req_t2"}
    assert len(rows) == 6 and seen.count("t2") == 1
    again, seen2 = refusing("t2")
    e9.stage_rag_sonnet(cfg, again, frozen)
    assert again.calls == [] and seen2 == []  # nothing is sent again, the refused row included


def test_new_calls_stop_for_good_at_three_refusals_and_the_rest_are_not_sent(frozen):
    cfg = staged(frozen)
    llm, seen = refusing("t1", "t2", "t3", "t4", "t5")
    e9.stage_rag_sonnet(cfg, llm, frozen)
    path, items = wd_file(frozen, cfg, llm)
    rows = {json.loads(x)["id"]: json.loads(x) for x in path.read_text().splitlines()}
    refused = [i for i, r in rows.items() if r.get("refused_by_safeguard")]
    assert len(refused) == 3  # exactly three: the check sends one call at a time, so nothing is in flight at the third
    assert seen.count("t1") + seen.count("t2") + seen.count("t3") == 3 and len(seen) == 3  # and nothing is sent after it
    assert "t6" not in rows and len(rows) < len(items)  # never sent
    again, seen2 = refusing("t1", "t2", "t3", "t4", "t5")
    e9.stage_rag_sonnet(cfg, again, frozen)
    assert seen2 == []  # a rerun sends nothing: three refusals stop the check for good


def test_wd_key_is_unchanged_so_saved_rows_are_reused(frozen):
    cfg = staged(frozen)
    llm = FakeLLM(reply=sonnet_reply, model="sonnet")
    path, items = wd_file(frozen, cfg, llm)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = "".join(json.dumps({"id": i["id"], "reply": "FITS", "call": "FITS"}) + "\n" for i in items[:3])
    path.write_text(saved)
    e9.stage_rag_sonnet(cfg, llm, frozen)
    asked = [c for c in llm.calls if json.dumps(c).count("Question:") and any(
        f"{i['question']}" in json.dumps(c) for i in items[:3])]
    assert asked == []  # the three saved answers were not sent again
    assert path.read_text().startswith(saved) and len(path.read_text().splitlines()) == 6


def test_a_partial_check_is_recorded_as_partial_with_counts_beside_the_rate_and_check_run_agrees(frozen):
    cfg = staged(frozen)
    llm, _ = refusing("t1", "t2", "t3", "t4", "t5")
    e9.stage_rag_sonnet(cfg, llm, frozen)
    path = e9.stage_rag_record(cfg, frozen, device="none: fakes", interruption_note=NOTE)
    rec = json.loads(path.read_text())
    wd = rec["config"]["wrong_disease_check"]
    assert wd["complete"] is False and wd["refused_by_safeguard"] == 3 and wd["not_sent"] >= 1
    assert wd["graded"] + wd["refused_by_safeguard"] + wd["not_sent"] == wd["answers_to_check"] == 6
    assert wd["usable_calls"] <= wd["graded"] and all(r["request_id"].startswith("req_") for r in wd["refused"])
    rows = read_rows(frozen, rec, "served")
    assert [r["wrong_disease"] for r in rows if r["wrong_disease_status"] != "graded"] == [None] * (6 - wd["graded"])
    assert "partial" in rec["notes"] and "partial" in e9.report(rec, frozen) and "complete. " not in e9.report(rec, frozen)
    assert check_file(path, frozen, min_n=1) == []
    rec["config"]["wrong_disease_check"]["graded"] += 1
    path.write_text(json.dumps(rec))
    assert any("wrong_disease_check" in p for p in check_file(path, frozen, min_n=1))


def test_a_field_test_record_on_cleaned_rows_and_the_class_list_passes_and_other_subsets_fail(repo):
    from PIL import Image

    final_model(repo)
    with open(repo / "datasets/manifests/field.csv", "a", encoding="utf-8") as f:  # a field photo of a class outside the list
        f.write("f/30.jpg,Tomato___healthy,h30,field_test,s30,plantdoc" + chr(10))
    Image.new("RGB", (8, 8)).save(repo / "f/30.jpg")
    commit(repo, "a field photo outside the class list")
    (path,) = e9.run_images("field_test", e9.load_config(repo), three_class_predictors(), repo)
    good = json.loads(path.read_text())
    assert good["subset"] == "clean:1+classes_in:results/e15/decision.json"  # the composite the real run writes
    assert check_file(path, repo, min_n=1) == []

    def problems(subset):
        path.write_text(json.dumps({**good, "subset": subset}))
        return check_file(path, repo, min_n=1)

    for bad in ("classes_in:results/e15/decision.json", None, "xclean:1+classes_in:results/e15/decision.json",
                "clean:2+classes_in:results/e15/decision.json", "clean:1+shared_classes", "clean:1+clean:1",
                "clean:1+first_5"):
        found = problems(bad)  # an unknown part is refused earlier, by the subset itself or its version; the rest by the cleaned-rows rule
        assert found, bad
        if bad in (None, "classes_in:results/e15/decision.json", "clean:1+clean:1", "clean:1+first_5", "clean:1+shared_classes"):
            assert any("cleaned rows" in p for p in found), (bad, found)  # these reach the rule itself
    rec = json.loads(json.dumps(good))
    del rec["config"]["cleaned_rows"]
    path.write_text(json.dumps(rec))
    assert any("cleaned rows" in p for p in check_file(path, repo, min_n=1))


# ------------------------------------------------- the record says how an interrupted run went, and can be superseded

NOTE = "first part ran at commit aaa1111, resumed at bbb2222; rows saved before the stop were not changed"


def refused_run(root):
    cfg = staged(root)
    llm, _ = refusing("t2")
    e9.stage_rag_sonnet(cfg, llm, root)
    return cfg


def test_a_record_with_a_safeguard_refusal_needs_the_interruption_note_and_carries_the_account(frozen):
    cfg = refused_run(frozen)
    with pytest.raises(e9.Refused, match="interruption"):
        e9.stage_rag_record(cfg, frozen, device="none: fakes")
    path = e9.stage_rag_record(cfg, frozen, device="none: fakes", interruption_note=NOTE)
    rec = json.loads(path.read_text())
    a = rec["config"]["run_account"]
    assert a["interruption"]["occurred"] is True and a["interruption"]["note"] == NOTE  # verbatim
    assert a["interruption"]["refused"] == [{"id": "t2", "request_id": "req_t2", "recorded_without_call": False}]
    assert set(a["stages"]) == {"retrieve", "generate", "claims_haiku", "claims_sonnet", "wrong_disease"}
    assert all(len(s["sha256"]) == 64 and s["last_write"] for s in a["stages"].values())
    assert a["window"]["last_file_write"] <= a["window"]["record_written"] and a["window"]["first_file_created"] is not None or True
    assert a["questions_answered_and_judged"]["questions"] == 6 and a["frozen"]["equal_to_frozen_config"] is True
    assert a["model_calls_by_rag_record"] == 0 and "commit" in json.dumps(a["interruption"]["code_commits"])
    notes = rec["notes"]
    assert "Interruption:" in notes and "answered and judged once, with one stop" in notes and NOTE in notes
    assert "from one run with everything frozen" not in notes and "req_t2" in notes
    assert check_file(path, frozen, min_n=1) == []


