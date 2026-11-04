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
    rng = np.random.default_rng(0)
    for i in range(12):
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(root / f"im{i}.png")
    feats = np.stack([classic_features(Image.open(root / f"im{i}.png"), size=32) for i in range(12)])
    booster = xgb.train({"objective": "multi:softprob", "num_class": 4, "max_depth": 2}, xgb.DMatrix(feats, label=np.arange(12) % 4), 3)
    (root / "models/e3/xgboost").mkdir(parents=True)
    booster.save_model(str(root / "models/e3/xgboost/model.ubj"))
    export_module(model, class_map, root / "models/e6/model.dynamic.onnx", "tiny_cnn")
    onnx_sha = e9.sha256_file(root / "models/e6/model.dynamic.onnx")
    chosen_sha = e9.sha256_file(root / "models/e4/convnext_joint/best.pt")
    write(root, "results/e6/export.json", {"checkpoint_sha256": chosen_sha, "files": {
        "onnx_dynamic": {"file": "model.dynamic.onnx", "sha256": onnx_sha}}})
    write(root, "results/e12/decision.json", {"chosen_model": "models/e4/convnext_joint/best.pt",
                                              "chosen_model_sha256": chosen_sha, "class_names": CLASSES, "refit_e5": False})
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
    with pytest.raises(e9.Refused, match="differs from E12's decision"):
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


def add_rag_files(root):
    rows = [question_row(q, "dev") for q in DEV_IDS] + [question_row(q, "test") for q in TEST_IDS]
    write(root, "datasets/rag/questions.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    chunks = [{"source": "a.pdf", "page": 1, "text": "apple scab", "crop": "apple"},
              {"source": "b.pdf", "page": 2, "text": "tomato blight", "crop": "tomato"},
              {"source": "g.pdf", "page": 1, "text": "watering", "crop": "general"}]
    write(root, "knowledge/index/chunks.jsonl", "".join(json.dumps(c) + "\n" for c in chunks))


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
    first = json.dumps(llm.calls[0])
    assert "Question: My apple leaves have scab q1" in first and "Diagnosis: The photo suggests Apple - scab" in first
    assert "SUPPORTED" not in first and "judge-model" not in first and "served" not in first
    assert rec["config"]["prompt_sha256"] == e9.sha256_file(rag_repo / e9.PROMPT_REL)
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


# ----------------------------------------------------------------------------------------------- blind read, freeze


def dev_record(rag_repo):
    cfg = dev_cfg(rag_repo)
    commit(rag_repo, "dev source")
    path = e9.stage_wrong_disease_dev(cfg, FakeLLM(reply=grader_reply, model="sonnet"), rag_repo)
    return cfg, path


def reader_file(root, cfg, wrong_after_fits=1, fits_after_wrong=1, name="E9-wrong-disease-blind-read.jsonl"):
    """What a reader returns for the blind file: agrees with the grader except for the stated number of answers."""
    rec = json.loads(e9.latest_stage_record(root, "wrong_disease_dev")[0].read_text())
    calls = {r["item"]: r["call"] for r in map(json.loads, (root / rec["predictions"]["served"]).read_text().splitlines())}
    blind = [json.loads(x) for x in e9.blind_file(root).read_text().splitlines()]
    out, flip = [], {"FITS": wrong_after_fits, "WRONG": fits_after_wrong}
    for b in blind:
        call = calls[b["item"]]
        if flip[call]:
            flip[call] -= 1
            call = "WRONG" if call == "FITS" else "FITS"
        out.append({"item": b["item"], "reader": call})
    path = root / "results" / "runs" / name
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    return path


def test_blind_file_hides_the_graders_call_and_is_the_same_every_time(rag_repo):
    cfg, _ = dev_record(rag_repo)
    path = e9.stage_blind_sample(cfg, rag_repo)
    text = path.read_text()
    rows = [json.loads(x) for x in text.splitlines()]
    assert len(rows) == 5  # 3 of the 6 answers called FITS and both called WRONG
    assert all(set(r) == {"item", "question", "diagnosis", "answer"} for r in rows)
    for secret in ("FITS", "WRONG", "reply", "call", "grader", "sonnet", "served"):
        assert secret not in text
    assert sum("Tomato advice" in r["answer"] for r in rows) == 2  # both of the grader's WRONG calls are in it
    path.unlink()
    assert e9.stage_blind_sample(cfg, rag_repo).read_text() == text  # the seed fixes the draw and its order
    assert len({r["item"] for r in rows}) == 5


def test_unblind_reports_error_in_both_directions(rag_repo):
    cfg, _ = dev_record(rag_repo)
    e9.stage_blind_sample(cfg, rag_repo)
    read = reader_file(rag_repo, cfg, wrong_after_fits=1, fits_after_wrong=1)
    path = e9.stage_unblind(cfg, rag_repo, read, "second reader")
    rec = json.loads(path.read_text())
    m = rec["metrics"]["served"]
    assert m["false_alarm_rate"] == 0.5 and m["missed_rate"] == pytest.approx(1 / 3)  # WRONG calls the reader passes; FITS calls it fails
    assert m["estimated_disagreement_rate"] == pytest.approx((2 * 0.5 + 6 * (1 / 3)) / 8)  # weighted by the dev calls
    assert m["reader_says_fits_after_wrong_call"] == 1 and m["reader_says_wrong_after_fits_call"] == 1
    assert set(rec["intervals"]["served"]) == {"false_alarm_rate", "missed_rate"}
    assert rec["config"]["strata"] == {"grader_wrong": 2, "grader_fits": 6} and rec["config"]["reader"] == "second reader"
    assert rec["claude_calls"] == 0 and rec["models"]["reader"] == "second reader"
    assert check_file(path, rag_repo, min_n=1) == []


def test_unblind_refuses_a_reader_file_that_does_not_match_the_sample(rag_repo):
    cfg, _ = dev_record(rag_repo)
    e9.stage_blind_sample(cfg, rag_repo)
    read = reader_file(rag_repo, cfg)
    lines = read.read_text().splitlines()
    read.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(e9.Refused, match="every item"):
        e9.stage_unblind(cfg, rag_repo, read, "r")
    row = json.loads(lines[0])
    read.write_text("\n".join([json.dumps({**row, "reader": "MAYBE"}), *lines[1:]]) + "\n")
    with pytest.raises(e9.Refused, match="FITS or WRONG"):
        e9.stage_unblind(cfg, rag_repo, read, "r")


def test_blind_read_record_tampering_is_caught(rag_repo):
    cfg, _ = dev_record(rag_repo)
    e9.stage_blind_sample(cfg, rag_repo)
    path = e9.stage_unblind(cfg, rag_repo, reader_file(rag_repo, cfg), "r")
    rec = json.loads(path.read_text())
    pred = rag_repo / rec["predictions"]["served"]
    original = pred.read_text()
    lines = original.splitlines()
    for field, value, expect in (("reader", None, "do not recompute"), ("grader", None, "seeded sample")):
        row = json.loads(lines[0])
        row[field] = "WRONG" if row[field] == "FITS" else "FITS"
        pred.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
        assert any(expect in p for p in check_file(path, rag_repo, min_n=1)), field
    pred.write_text(original)
    assert check_file(path, rag_repo, min_n=1) == []
    saved = json.loads(path.read_text())
    saved["config"]["strata"] = {"grader_wrong": 5, "grader_fits": 3}
    path.write_text(json.dumps(saved))
    assert any("strata" in p for p in check_file(path, rag_repo, min_n=1))


def test_dev_record_tampering_is_caught(rag_repo):
    _, path = dev_record(rag_repo)
    rec = json.loads(path.read_text())
    pred = rag_repo / rec["predictions"]["served"]
    original = pred.read_text()
    lines = original.splitlines()
    row = json.loads(lines[0])
    pred.write_text("\n".join([json.dumps({**row, "call": "WRONG"}), *lines[1:]]) + "\n")
    assert any("call is not what its reply says" in p for p in check_file(path, rag_repo, min_n=1))
    pred.write_text("\n".join([json.dumps({**row, "reply": "WRONG", "call": "WRONG"}), *lines[1:]]) + "\n")
    assert any("do not recompute" in p for p in check_file(path, rag_repo, min_n=1))
    pred.write_text("\n".join([json.dumps({**row, "answer": "Different words."}), *lines[1:]]) + "\n")
    assert any("differ from the source judgments" in p for p in check_file(path, rag_repo, min_n=1))
    pred.write_text(original)
    saved = json.loads(path.read_text())
    saved["config"]["prompt_sha256"] = "0" * 64
    path.write_text(json.dumps(saved))
    assert any("prompt_sha256" in p for p in check_file(path, rag_repo, min_n=1))


def frozen_repo(rag_repo):
    cfg, _ = dev_record(rag_repo)
    e9.stage_blind_sample(cfg, rag_repo)
    e9.stage_unblind(cfg, rag_repo, reader_file(rag_repo, cfg), "r")
    commit(rag_repo, "records")
    return cfg


def test_freeze_records_the_prompt_commit_and_hash_and_needs_the_blind_read(rag_repo):
    cfg, _ = dev_record(rag_repo)
    commit(rag_repo, "dev record")
    with pytest.raises(e9.Refused, match="wrong_disease_blind_read"):
        e9.stage_freeze(cfg, rag_repo)
    e9.stage_blind_sample(cfg, rag_repo)
    e9.stage_unblind(cfg, rag_repo, reader_file(rag_repo, cfg), "r")
    commit(rag_repo, "records")
    path = e9.stage_freeze(cfg, rag_repo)
    frozen = json.loads(path.read_text())
    assert frozen["prompt_sha256"] == e9.sha256_file(rag_repo / e9.PROMPT_REL)
    assert frozen["prompt_commit"] == git(rag_repo, "log", "-1", "--format=%H", "--", e9.PROMPT_REL).stdout.strip()
    assert frozen["rendered_sha256"] == e9.rendered_sha256(rag_repo)
    assert {"dev_record", "blind_read_record", "frozen_at"} <= set(frozen)
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
    with pytest.raises(e9.Refused, match="changed since"):
        e9.stage_unblind(cfg, rag_repo, reader_file(rag_repo, cfg, name="again.jsonl"), "r")
