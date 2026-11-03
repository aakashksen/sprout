import json
import subprocess

import numpy as np
import pytest

from sprout.eval import metrics as M
from sprout.eval.check_run import (
    Scored,
    check,
    recompute_comparison,
    recompute_interval,
    recompute_metric,
)
from sprout.eval.claims import check_claims
from sprout.eval.predict import Predictions
from sprout.eval.records import split_info, write_run
from sprout.serve.types import FakeIdentifier, Identification

NAMES = ["A", "B"]
LAB = ["path,label,group,split,sha1"] + [
    f"img/{i}.jpg,{'AB'[i % 2]},g{i // 2},{s},s{i}" for i, s in enumerate(
        ["lab_train"] * 4 + ["lab_dev"] * 6 + ["lab_test"] * 6)
]
FIELD = ["path,label,group,split,sha1,source"] + [
    f"f/{i}.jpg,A,h{i},field_test,t{i},plantdoc" for i in range(4)
] + ["f/9.jpg,C,h9,field_test,t9,plantdoc"]
RAG = [json.dumps({"id": f"q{i}", "split": "dev" if i < 4 else "test", "question": "?"}) for i in range(6)]
PERF = [f"p/{i}.jpg" for i in range(3)]


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def root(tmp_path):
    for rel, lines in {
        "datasets/manifests/lab.csv": LAB,
        "datasets/manifests/field.csv": FIELD,
        "datasets/rag/questions.jsonl": RAG,
        "datasets/perf/images.txt": PERF,
        "orchestration/tasks.json": ["{}"],
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(lines) + "\n")
    sh(tmp_path, "init", "-q")
    sh(tmp_path, "config", "user.email", "t@t")
    sh(tmp_path, "config", "user.name", "t")
    sh(tmp_path, "add", "-A")
    sh(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def preds(root, split="lab_dev", flip=()):
    info = split_info(split, root)
    labels = np.array([NAMES.index(x) for x in info.labels])
    probs = np.full((len(labels), 2), 0.2)
    for i, y in enumerate(labels):
        probs[i, y if i not in flip else 1 - y] = 0.8
    return Predictions(list(info.ids), labels, probs), info


def scored(p, info):
    return Scored(groups=info.groups, labels=p.label, probs=p.probs)


def default_args(root, split):
    p, info = preds(root, split, flip=(0, 3))
    q, _ = preds(root, split, flip=(0,))
    sp, sq = scored(p, info), scored(q, info)
    names = ("accuracy", "macro_f1", "ece")
    return dict(
        config={"class_names": NAMES, "labels_reviewed": True},
        models={"classifier": "resnet-test"},
        metrics={"a": {m: recompute_metric(m, sp) for m in names}, "b": {"accuracy": recompute_metric("accuracy", sq)}},
        intervals={"a": {"accuracy": list(recompute_interval("accuracy", sp))}},
        attempts=[{"setting": "default", "result": "ok"}],
        predictions={"a": p, "b": q},
        dataset_hash=info.hash,
        comparisons=[{"a": "a", "b": "b", "metric": "accuracy", **recompute_comparison(sp, sq, "accuracy")}],
    )


def make(root, split="lab_test", experiment="E1", **kw):
    args = {} if "predictions" in kw else default_args(root, split)
    args.setdefault("attempts", [{"setting": "default", "result": "ok"}])
    args.setdefault("models", {"classifier": "resnet-test"})
    args.update(kw)
    return write_run(experiment, split, root=root, **args)


def edit(path, fn):
    rec = json.loads(path.read_text())
    fn(rec)
    path.write_text(json.dumps(rec))


def failing(root, text, experiment="E1", **kw):
    problems = check(experiment, root=root, **kw)
    assert any(text in p for p in problems), problems


def test_valid_record_passes(root):
    make(root)
    assert check("E1", root=root) == []


def test_write_run_refuses_dirty_tree(root):
    (root / "datasets/rag/questions.jsonl").write_text("changed\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        make(root)
    make(root, allow_dirty=True)


def test_dirty_check_ignores_orchestration(root):
    (root / "orchestration/tasks.json").write_text('{"x": 1}')
    make(root)


def test_dirty_record_fails(root):
    edit(make(root), lambda r: r.update(dirty=True))
    failing(root, "dirty tree")


def test_real_runs_cannot_use_the_fake(root):
    fake = FakeIdentifier([Identification("answered")])
    make(root, models={"identifier": type(fake).__name__})
    failing(root, "fake or stub")


def test_stub_model_name_fails(root):
    make(root, models={"writer": "stub-llm"})
    failing(root, "fake or stub")


def test_dataset_hash_must_match_manifest(root):
    edit(make(root), lambda r: r.update(dataset_hash="0" * 64))
    failing(root, "dataset_hash")


def test_changed_manifest_fails_hash(root):
    make(root)
    (root / "datasets/manifests/lab.csv").write_text("\n".join(LAB).replace("s1", "zz") + "\n")
    failing(root, "dataset_hash")


def test_commit_must_be_ancestor_of_head(root):
    edit(make(root), lambda r: r.update(git_commit="1" * 40))
    failing(root, "ancestor")


def test_n_must_equal_split_size(root):
    edit(make(root), lambda r: r.update(n=5))
    failing(root, "differs from the split's 6")


def test_n_below_min_n(root):
    make(root)
    failing(root, "below --min-n", min_n=100)


def test_attempts_must_not_be_empty(root):
    make(root, attempts=[])
    failing(root, "attempts is empty")


def test_metric_must_recompute(root):
    path = make(root)
    edit(path, lambda r: r["metrics"]["a"].update(accuracy=0.99))
    failing(root, "metric accuracy does not recompute")


def test_unknown_metric_fails(root):
    path = make(root)
    edit(path, lambda r: r["metrics"]["a"].update(vibes=1.0))
    failing(root, "metric vibes cannot be recomputed")


def test_interval_must_recompute(root):
    edit(make(root), lambda r: r["intervals"]["a"].update(accuracy=[0.1, 0.2]))
    failing(root, "interval accuracy does not recompute")


def test_comparison_must_recompute(root):
    edit(make(root), lambda r: r["comparisons"][0].update(mcnemar_p=0.001))
    failing(root, "does not recompute")


def test_saved_labels_must_match_manifest(root):
    path = make(root)
    pred = next(root.glob("results/runs/E1-*.a.predictions.jsonl"))
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    rows[0]["label"] = 1 - rows[0]["label"]
    pred.write_text("".join(json.dumps(r) + "\n" for r in rows))
    failing(root, "labels do not match")
    assert path.exists()


def test_predictions_must_cover_split(root):
    make(root)
    pred = next(root.glob("results/runs/E1-*.a.predictions.jsonl"))
    lines = pred.read_text().splitlines()
    pred.write_text("\n".join(lines[:-1] + lines[:1]) + "\n")
    failing(root, "exactly once")


def test_measurement_must_match_saved_file(root):
    path = make(root, split="perf", metrics={"a": {}, "b": {}},
                intervals={}, comparisons=[], predictions={"a": [{"path": p} for p in PERF],
                                                          "b": [{"path": p} for p in PERF]},
                config={}, measurements={"latency_ms": [10.0, 20.0, 30.0]},
                dataset_hash=split_info("perf", root).hash)
    assert check("E1", root=root, split="perf") == []
    edit(path, lambda r: r["measurements"]["values"].update(latency_ms_p50=1.0))
    failing(root, "does not match its saved file")


def test_judged_must_match_saved_file_and_name_judge(root):
    path = make(root, judged={"faithfulness": [True, False, True, True]},
                config={"class_names": NAMES, "judge": {"model": "judge-x"}}, models={"writer": "w"})
    assert check("E1", root=root) == []
    edit(path, lambda r: r["judged"]["values"].update(faithfulness=1.0))
    failing(root, "judged value faithfulness does not match")
    edit(path, lambda r: r["config"].update(judge={"model": "w"}))
    failing(root, "different model from the writer")


def test_second_test_record_needs_override_note(root):
    make(root)
    assert check("E1", root=root) == []
    second = make(root, experiment="E9")
    failing(root, "more than one lab_test record", experiment="E9")
    edit(second, lambda r: r["config"].update(override_note="redo after a bug"))
    assert check("E9", root=root) == []


def test_dev_split_may_repeat(root):
    make(root, split="lab_dev")
    make(root, split="lab_dev")
    assert check("E1", root=root) == []


def test_subset_rebuilds_ids(root):
    p, info = preds(root, "lab_dev")
    first = Predictions(p.path[:2], p.label[:2], p.probs[:2])
    s = Scored(groups=info.groups[:2], labels=first.label, probs=first.probs)
    make(root, split="lab_dev", subset="first_2", predictions={"a": first}, comparisons=[],
         config={"class_names": NAMES}, dataset_hash=info.hash,
         metrics={"a": {"accuracy": recompute_metric("accuracy", s)}}, intervals={})
    assert check("E1", root=root) == []


def test_shared_classes_subset_on_field(root):
    info = split_info("field_test", root)
    keep = [i for i, label in enumerate(info.labels) if label in NAMES]
    labels = np.array([NAMES.index(info.labels[i]) for i in keep])
    probs = np.tile([0.9, 0.1], (len(keep), 1))
    pred = Predictions([info.ids[i] for i in keep], labels, probs)
    s = Scored(groups=[info.groups[i] for i in keep], labels=labels, probs=probs)
    make(root, split="field_test", subset="shared_classes", predictions={"a": pred}, comparisons=[],
         config={"class_names": NAMES}, dataset_hash=split_info("field_test", root).hash,
         metrics={"a": {"accuracy": recompute_metric("accuracy", s)}}, intervals={})
    assert check("E1", root=root) == []


def test_rag_split_reads_questions(root):
    info = split_info("rag_dev", root)
    assert len(info.ids) == 4
    rows = [{"id": i, "rank": r} for i, r in zip(info.ids, [1, 3, None, 2])]
    s = Scored(groups=info.groups, ranks=[r["rank"] for r in rows])
    make(root, split="rag_dev", predictions={"a": rows}, comparisons=[], config={},
         metrics={"a": {"recall@3": recompute_metric("recall@3", s), "mrr": recompute_metric("mrr", s)}},
         intervals={"a": {"recall@3": list(recompute_interval("recall@3", s))}},
         dataset_hash=info.hash)
    assert check("E1", root=root) == []
    assert recompute_metric("mrr", s) == pytest.approx((1 + 1 / 3 + 0 + 0.5) / 4)


def test_perf_uses_fixed_image_list(root):
    assert len(split_info("perf", root).ids) == 3


def test_claim_without_matching_run_fails(root):
    (root / "results").mkdir()
    (root / "results/claims.json").write_text(json.dumps(
        {"claims": [{"id": "c1", "run_id": "E1-20250101T000000000000Z", "metric": "accuracy", "value": 1.0}]}))
    assert any("no run record" in p for p in check_claims(root))


def claim(root, run_id, value):
    (root / "results").mkdir(exist_ok=True)
    (root / "results/claims.json").write_text(json.dumps({"claims": [
        {"id": "c1", "run_id": run_id, "system": "a", "metric": "accuracy", "value": value}]}))


def test_claim_value_must_equal_metric_in_run(root):
    path = make(root, split="lab_dev")
    saved = json.loads(path.read_text())["metrics"]["a"]["accuracy"]
    claim(root, path.stem, saved)
    assert check_claims(root) == []
    claim(root, path.stem, saved + 0.01)
    assert any("value differs" in p for p in check_claims(root))


def test_claim_on_unreviewed_run_fails(root):
    path = make(root, split="lab_dev", config={"class_names": NAMES})
    saved = json.loads(path.read_text())["metrics"]["a"]["accuracy"]
    claim(root, path.stem, saved)
    assert any("labels_reviewed" in p for p in check_claims(root))


def macro_make(root, split="lab_dev"):
    p, info = preds(root, split, flip=(0, 3))
    q, _ = preds(root, split, flip=(1,))
    sp, sq = scored(p, info), scored(q, info)
    return make(
        root, split=split,
        metrics={"a": {"macro_f1": recompute_metric("macro_f1", sp)}, "b": {"macro_f1": recompute_metric("macro_f1", sq)}},
        intervals={"a": {"macro_f1": list(recompute_interval("macro_f1", sp))}},
        comparisons=[{"a": "a", "b": "b", "metric": "macro_f1", **recompute_comparison(sp, sq, "macro_f1")}],
    ), sp, sq


def test_macro_f1_interval_and_comparison_pass(root):
    path, sp, sq = macro_make(root)
    assert check("E1", root=root) == []
    rec = json.loads(path.read_text())
    assert rec["bootstrap"] == {"n": 10000, "seed": 13}
    lo, hi = rec["intervals"]["a"]["macro_f1"]
    assert lo <= recompute_metric("macro_f1", sp) <= hi


def test_macro_f1_bootstrap_resamples_whole_groups(root):
    from sprout.eval.stats import bootstrap_macro_f1

    labels = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 0])
    one = bootstrap_macro_f1(labels, pred, ["g"] * 4, 2, 500, 13)
    each = bootstrap_macro_f1(labels, pred, ["a", "b", "c", "d"], 2, 500, 13)
    assert one.std() < 1e-12 and each.std() > 0
    assert one[0] == pytest.approx(M.macro_f1(labels, pred))


def test_macro_f1_interval_must_recompute(root):
    path, _, _ = macro_make(root)
    edit(path, lambda r: r["intervals"]["a"].update(macro_f1=[0.0, 1.0]))
    failing(root, "interval macro_f1 does not recompute")


def test_macro_f1_comparison_must_recompute(root):
    path, _, _ = macro_make(root)
    edit(path, lambda r: r["comparisons"][0].update(ci=[-1.0, 1.0]))
    failing(root, "macro_f1 for a vs b does not recompute")


def test_bootstrap_n_must_be_ten_thousand(root):
    path, _, _ = macro_make(root)
    edit(path, lambda r: r["bootstrap"].update(n=100))
    failing(root, "10,000 resamples")


def test_systems_scored_on_different_images_fail(root):
    make(root)
    pred = next(root.glob("results/runs/E1-*.b.predictions.jsonl"))
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    rows[0]["path"] = "img/other.jpg"
    pred.write_text("".join(json.dumps(r) + "\n" for r in rows))
    failing(root, "system b: prediction ids do not match")


def test_undefined_metric_saved_as_none_is_fine(root):
    p, info = preds(root, "lab_dev")
    s = scored(p, info)
    assert recompute_metric("selective_accuracy@1.01", s) is None
    path = make(root, split="lab_dev")
    edit(path, lambda r: r["metrics"]["a"].update({"selective_accuracy@1.01": None}))
    assert check("E1", root=root) == []
    edit(path, lambda r: r["metrics"]["a"].update({"selective_accuracy@1.01": 0.5}))
    failing(root, "selective_accuracy@1.01 does not recompute")


def test_check_needs_manifests_and_predictions_not_images(root):
    make(root, split="lab_dev")
    assert not (root / "datasets/raw").exists() and not (root / "img").exists()
    assert check("E1", root=root) == []
