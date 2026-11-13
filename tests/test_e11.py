"""E11: follow-up questions on the model's top-3, answered by a vision agent that never sees the label."""
import csv
import json
import re
from pathlib import Path

import pytest

from sprout.serve import followup as F

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "knowledge" / "raw"
needs_raw = pytest.mark.skipif(not RAW.exists(), reason="knowledge/raw is git-ignored and absent here")
BANK = F.load_questions()
COVERAGE = list(csv.DictReader((ROOT / "docs" / "document-coverage.csv").read_text(encoding="utf-8").splitlines()))
COVERED = {r["label"] for r in COVERAGE if r["symptoms_described"] == "yes"}


def q(id_, classes, text="Is it spotted?"):
    return F.Question(id_, text, tuple(classes), "doc.pdf", 1, "passage")


# ---------------------------------------------------------------------------------------------------------------
# the questions


def test_questions_cite_a_document_and_page():
    assert BANK
    manifest = {r["file"] for r in csv.DictReader((ROOT / "knowledge" / "manifest.csv").read_text("utf-8").splitlines())}
    for item in BANK:
        assert item.document in manifest, item.id
        assert isinstance(item.page, int) and item.page >= 1, item.id
        assert item.passage.strip(), item.id
        if item.document.endswith((".html", ".md")):
            assert item.page == 1, f"{item.id}: an HTML or markdown document is a single page 1"
    assert len({item.id for item in BANK}) == len(BANK)


def test_uncovered_class_gets_no_question():
    not_covered = {r["label"] for r in COVERAGE} - COVERED
    assert all(c not in not_covered for item in BANK for c in item.classes)
    assert {c for item in BANK for c in item.classes} <= COVERED
    # a class with no question is never the subject of one, and is reported rather than guessed about
    bank = [q("a1", ["A"]), q("b1", ["B"])]
    top3 = [("A", 0.4), ("X", 0.3), ("B", 0.2)]
    asked = F.next_question(top3, asked=(), bank=bank)
    assert asked is not None and "X" not in asked.classes
    assert F.uncovered(top3, bank) == ["X"]


def test_healthy_classes_have_no_question_and_are_not_reported_uncovered():
    assert not any(c.endswith("healthy") for item in BANK for c in item.classes)
    assert F.uncovered([("Tomato___healthy", 0.5), ("Tomato___Late_blight", 0.3)], BANK) == []


def test_every_covered_disease_class_has_a_question():
    assert COVERED <= {c for item in BANK for c in item.classes}


def test_question_text_does_not_name_a_candidate_class():
    generic = {"black", "leaf", "spot", "spotted", "gray", "common", "northern", "early", "late", "target", "blight", "rot",
               "orange", "yellow", "curl", "bell"}  # colours and plain symptom words say nothing about the class
    for item in BANK:
        text = item.text.lower()
        for label in COVERED | {r["label"] for r in COVERAGE}:
            crop, _, rest = label.partition("___")
            words = {w for w in re.split(r"[^a-z]+", f"{crop} {rest}".lower()) if len(w) >= 4} - generic
            words.discard("including")
            hit = [w for w in words if w in text]
            assert not hit, f"{item.id} names {hit} from {label}"
        assert "blight" not in text and "rot " not in text and "scab" not in text, item.id


def test_every_cited_document_has_a_known_source():
    manifest = {r["file"]: r for r in csv.DictReader((ROOT / "knowledge" / "manifest.csv").read_text("utf-8").splitlines())}
    for item in BANK:
        row = manifest[item.document]
        assert row["source_url"] and not row["source_url"].lower().startswith("unknown"), item.id
        assert row["publisher"] and not row["publisher"].lower().startswith("unknown"), item.id


def test_questions_are_yes_no_questions():
    assert all(item.text.endswith("?") for item in BANK)


# ---------------------------------------------------------------------------------------------------------------
# the flow


TOP3 = [("A", 0.5), ("B", 0.3), ("C", 0.1)]


def test_unknown_answer_leaves_ranking_unchanged():
    assert F.update(TOP3, q("a1", ["A"]), "unknown") == TOP3
    assert F.update(TOP3, q("b1", ["B", "C"]), "unknown") == TOP3


def test_yes_lifts_the_classes_the_document_ties_to_the_symptom_and_keeps_the_mass():
    rule = F.Rule(yes_factor=4.0, no_factor=0.5, max_questions=2)
    out = F.update(TOP3, q("b1", ["B"]), "yes", rule=rule)
    assert [c for c, _ in out] == ["B", "A", "C"]
    assert sum(p for _, p in out) == pytest.approx(0.9)
    assert out[0][1] == pytest.approx(0.9 * 1.2 / (1.2 + 0.5 + 0.1))


def test_no_lowers_them():
    rule = F.Rule(yes_factor=4.0, no_factor=0.25, max_questions=2)
    out = F.update(TOP3, q("a1", ["A"]), "no", rule=rule)
    assert [c for c, _ in out] == ["B", "A", "C"]
    assert sum(p for _, p in out) == pytest.approx(0.9)


def test_a_question_about_no_class_in_the_top3_changes_nothing():
    assert F.update(TOP3, q("z1", ["Z"]), "yes") == TOP3


def test_bad_answer_is_refused():
    with pytest.raises(ValueError, match="answer"):
        F.update(TOP3, q("a1", ["A"]), "maybe")


def test_next_question_splits_the_top3_and_never_repeats():
    bank = [q("a1", ["A"]), q("a2", ["A"]), q("b1", ["B"]), q("c1", ["A", "B", "C"])]
    first = F.next_question(TOP3, asked=(), bank=bank)
    assert first is not None and first.id == "a1"  # a1 and a2 split the mass the same way; the lower id wins
    second = F.next_question(TOP3, asked=("a1",), bank=bank)
    assert second is not None and second.id == "a2"
    third = F.next_question(TOP3, asked=("a1", "a2", "b1"), bank=bank)
    assert third is None  # c1 covers every class in the top 3, so it cannot tell them apart
    assert F.next_question(TOP3, asked=(), bank=[]) is None


def test_next_question_is_deterministic():
    bank = [q("b1", ["B"]), q("a1", ["A"])]
    assert F.next_question(TOP3, asked=(), bank=bank) == F.next_question(TOP3, asked=(), bank=list(reversed(bank)))


def test_is_unsure_needs_two_plausible_classes_in_the_top3():
    assert F.is_unsure(TOP3, 0.10) and not F.is_unsure([("A", 0.9), ("B", 0.05), ("C", 0.02)], 0.10)
    assert F.is_unsure([("A", 0.5), ("B", 0.10), ("C", 0.0)], 0.10)  # exactly the threshold counts


def test_top3_takes_the_three_largest_in_order():
    out = F.top3([0.1, 0.4, 0.05, 0.3, 0.15], ["a", "b", "c", "d", "e"])
    assert out == [("b", 0.4), ("d", 0.3), ("e", 0.15)]


def test_rerank_puts_the_new_values_back_into_the_full_vector():
    names = ["a", "b", "c", "d", "e"]
    probs = [0.1, 0.4, 0.05, 0.3, 0.15]
    new = F.rerank(probs, names, [("d", 0.4), ("b", 0.3), ("e", 0.15)])
    assert new == pytest.approx([0.1, 0.3, 0.05, 0.4, 0.15])
    assert sum(new) == pytest.approx(sum(probs))


def test_real_config_sets_the_rule_before_any_run():
    cfg = json.loads((ROOT / "configs" / "e11.json").read_text(encoding="utf-8"))
    assert F.load_rule(ROOT / "configs" / "e11.json") == F.Rule(
        cfg["rule"]["yes_factor"], cfg["rule"]["no_factor"], cfg["rule"]["max_questions"])
    assert cfg["split"] == "field_dev" and cfg["subset"] == "clean:1"
    assert cfg["unsure"]["min_probability"] == 0.10


# ---------------------------------------------------------------------------------------------------------------
# plan, sheets and answers, on a tiny world

import subprocess  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sprout.experiments import e11_followup as E  # noqa: E402

NAMES = ["Tomato___Late_blight", "Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight",
         "Potato___Early_blight", "Corn___Rust"]
FAKE_BANK = [
    F.Question(f"{tag}-{n}", f"Is feature {tag}{n} visible?", (cls,), "d.pdf", 1, "p")
    for tag, cls in (("tl", NAMES[0]), ("te", NAMES[1]), ("pl", NAMES[3]), ("pe", NAMES[4]), ("cr", NAMES[5]))
    for n in (1, 2)
]
SHEET_CFG = {"seed": 13, "photos_per_sheet": 4, "columns": 2, "cell_px": 64, "questions_per_photo": 8}
UNSURE = {"min_probability": 0.10, "min_plausible": 2}


def _probs(top):
    p = np.zeros(len(NAMES))
    for i, v in top.items():
        p[i] = v
    return p / p.sum()


@pytest.fixture
def world(tmp_path):
    """Ten field_dev photos whose file names and folders carry their class, as the real ones do."""
    rows, probs = [], []
    for i in range(10):
        label = NAMES[i % 3]
        rel = f"datasets/raw/plantdoc/{label}/{label}_secret{i}.jpg"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80 + i, 60), (10 * i, 120, 40)).save(tmp_path / rel)
        rows.append((rel, label))
        probs.append(_probs({0: 0.5, 1: 0.3, 3: 0.2}) if i % 2 == 0 else _probs({0: 0.95, 1: 0.03, 2: 0.02}))
    return tmp_path, rows, probs


def _rows(world):
    _, rows, probs = world
    return {"m": [(p, pr) for (p, _), pr in zip(rows, probs, strict=True)]}


def test_plan_gives_every_photo_the_same_number_of_questions_and_hides_the_top3(world):
    plan = E.build_plan(_rows(world), NAMES, FAKE_BANK, UNSURE, SHEET_CFG)
    assert [r["path"] for r in plan] == sorted(r["path"] for r in plan)
    assert len(plan) == 5  # the five photos that are unsure
    for r in plan:
        assert len(r["needed"]) + len(r["decoys"]) == SHEET_CFG["questions_per_photo"]
        assert not set(r["needed"]) & set(r["decoys"])
        assert set(r["needed"]) == {"tl-1", "tl-2", "te-1", "te-2", "pl-1", "pl-2"}
    assert E.build_plan(_rows(world), NAMES, FAKE_BANK, UNSURE, SHEET_CFG) == plan  # deterministic


def test_decoys_prefer_questions_from_the_crops_in_the_top3(world):
    plan = E.build_plan(_rows(world), NAMES, FAKE_BANK, UNSURE, SHEET_CFG)
    # the top 3 is two tomato classes and a potato class; the only other potato class supplies the decoys
    assert all(r["decoys"] == ["pe-1", "pe-2"] for r in plan)
    plan = E.build_plan(_rows(world), NAMES, FAKE_BANK, UNSURE, {**SHEET_CFG, "questions_per_photo": 10})
    assert all({"pe-1", "pe-2"} <= set(r["decoys"]) and len(r["decoys"]) == 4 for r in plan)


def test_a_photo_with_no_askable_question_is_left_out(world):
    root, rows, probs = world
    shared = [F.Question("x", "Is it x?", (NAMES[0], NAMES[1], NAMES[3]), "d.pdf", 1, "p")]
    assert E.build_plan({"m": [(rows[0][0], probs[0])]}, NAMES, shared, UNSURE, SHEET_CFG) == []


def test_plan_refuses_more_questions_than_the_sheet_allows(world):
    with pytest.raises(ValueError, match="questions_per_photo"):
        E.build_plan(_rows(world), NAMES, FAKE_BANK, UNSURE, {**SHEET_CFG, "questions_per_photo": 3})


def _sheets(world, seed=13):
    root = world[0]
    cfg = {**SHEET_CFG, "seed": seed}
    plan = E.build_plan(_rows(world), NAMES, FAKE_BANK, UNSURE, cfg)
    out, key = root / f"sheets{seed}", root / f"key{seed}" / "key.json"
    counts = E.build_sheets(plan, FAKE_BANK, out, key, root, cfg)
    return plan, out, key, counts


def test_sheets_show_the_photo_and_questions_and_nothing_that_names_the_label_or_prediction(world):
    root, rows, probs = world
    plan, out, key, counts = _sheets(world)
    assert counts == {"photos": 5, "sheets": 2}
    secrets = {n.lower() for n in NAMES} | {"secret", "plantdoc", "datasets", "field_dev", "0.95", "0.03"}
    secrets |= {p.lower() for p, _ in rows}
    files = sorted(out.iterdir())
    assert {f.suffix for f in files} == {".jpg", ".md", ".json"}
    for f in files:
        blob = f.read_bytes().lower()
        for s in secrets:
            assert s.encode() not in blob, f"{f.name} contains {s!r}"
    text = (out / "s001.md").read_text(encoding="utf-8")
    assert "Photo 1" in text and "Is feature" in text
    assert not any(r["path"] in text for r in plan) and "tl-1" not in text  # no path, no question id
    assert key.is_file() and out not in key.parents  # the key is kept away from what reviewers get


def test_sheet_order_is_shuffled_by_a_fixed_seed(world):
    _, _, key1, _ = _sheets(world, seed=13)
    first = json.loads(key1.read_text(encoding="utf-8"))
    key1.unlink()
    again = _sheets(world, seed=13)
    assert json.loads(again[2].read_text(encoding="utf-8")) == first
    order = [first[s][c]["path"] for s in sorted(first) for c in sorted(first[s])]
    assert order != sorted(order) and len(set(order)) == len(order)
    other = _sheets(world, seed=14)
    assert json.loads(other[2].read_text(encoding="utf-8")) != first


def test_blank_templates_list_every_cell_and_question(world):
    plan, out, key, _ = _sheets(world)
    tpl = json.loads((out / "s001.template.json").read_text(encoding="utf-8"))
    assert set(tpl) == {"1", "2", "3", "4"}
    assert all(len(v) == SHEET_CFG["questions_per_photo"] and set(v.values()) == {None} for v in tpl.values())


def _fill(out, answer_for):
    for tpl in sorted(out.glob("*.template.json")):
        data = json.loads(tpl.read_text(encoding="utf-8"))
        data = {c: {q: answer_for(tpl.name, c, q) for q in qs} for c, qs in data.items()}
        (out / tpl.name.replace(".template", ".answers")).write_text(json.dumps(data), encoding="utf-8")


def test_collect_joins_answers_to_photos_and_counts_unknown(world):
    root = world[0]
    plan, out, key, _ = _sheets(world)
    _fill(out, lambda f, c, q: "unknown" if q == "1" else "yes")
    dest = root / "answers.jsonl"
    meta = E.collect_answers(out, key, dest, "a test agent")
    got = [json.loads(x) for x in dest.read_text(encoding="utf-8").splitlines()]
    assert len(got) == 5 * SHEET_CFG["questions_per_photo"] == meta["rows"]
    assert {tuple(sorted(r)) for r in got} == {("answer", "path", "question")}
    assert meta["unknown"] == 5 and meta["answerer"] == "a test agent" and meta["photos"] == 5
    assert got == sorted(got, key=lambda r: (r["path"], r["question"]))
    first = [r for r in plan if r["path"] == got[0]["path"]][0]
    assert {r["question"] for r in got if r["path"] == first["path"]} == set(first["needed"]) | set(first["decoys"])


def test_collect_refuses_missing_or_invalid_answers(world):
    root = world[0]
    plan, out, key, _ = _sheets(world)
    with pytest.raises(ValueError, match="no answers"):
        E.collect_answers(out, key, root / "a.jsonl", "x")
    _fill(out, lambda f, c, q: "maybe")
    with pytest.raises(ValueError, match="yes, no or unknown"):
        E.collect_answers(out, key, root / "a.jsonl", "x")
    _fill(out, lambda f, c, q: None if (f, c, q) == ("s001.template.json", "2", "3") else "no")
    with pytest.raises(ValueError, match="unanswered"):
        E.collect_answers(out, key, root / "a.jsonl", "x")


def test_sheet_code_never_reads_a_label():
    import inspect

    for fn in (E.build_plan, E.build_sheets, E.collect_answers):
        assert "label" not in inspect.getsource(fn).lower(), fn.__name__


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "datasets/followup").mkdir(parents=True)
    (tmp_path / "results/runs").mkdir(parents=True)
    (tmp_path / "README").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _commit(root, rel, text, msg):
    (root / rel).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def test_answers_were_saved_blind_to_the_label(tmp_path):
    root = _repo(tmp_path)
    ans = "datasets/followup/answers.jsonl"
    rec = "results/runs/E11-20261104T000000000000Z.json"
    assert "no answers" in E.commit_order_problems(root)[0]  # nothing saved yet
    (root / ans).write_text("{}\n")
    assert "not committed" in E.commit_order_problems(root)[0]
    _commit(root, ans, "{}\n", "answers")
    assert E.commit_order_problems(root) == []  # saved, and no record yet
    (root / rec).write_text("{}")
    assert E.commit_order_problems(root) == []  # an uncommitted record is not yet in history
    _commit(root, rec, "{}", "first record")
    assert E.commit_order_problems(root) == []  # answers first, record after
    _commit(root, ans, '{"a": 1}\n', "answers changed after the record")
    assert any("after the first E11 record" in p for p in E.commit_order_problems(root))


def test_answers_committed_in_the_same_commit_as_the_first_record_fail(tmp_path):
    root = _repo(tmp_path)
    (root / "datasets/followup/answers.jsonl").write_text("{}\n")
    (root / "results/runs/E11-20261104T000000000000Z.json").write_text("{}")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "both at once")
    assert any("after the first E11 record" in p for p in E.commit_order_problems(root))


# ---------------------------------------------------------------------------------------------------------------
# the flow and the records, on fake earlier records

from sprout.eval.check_run import check  # noqa: E402
from sprout.eval.predict import Predictions  # noqa: E402
from sprout.eval.records import split_info, subset_positions, write_run  # noqa: E402

N_PHOTOS = 36


def _field_rows():
    rows = ["path,label,group,split,sha1,source"]
    for i in range(N_PHOTOS + 3):
        split = "field_test" if i >= N_PHOTOS else "field_dev"
        rows.append(f"f/{i}.jpg,{NAMES[i % 6]},h{i},{split},s{i},{'plantdoc' if i % 3 else 'plantseg'}")
    return rows


def _model_probs(label, kind):
    """The model is unsure: the true class is second for 'seen' rows and first for others; a rival is nearby."""
    rival = (label + 1) % 6 if (label + 1) % 6 != 2 else 3
    p = np.full(6, 0.02)
    if kind == "hard":
        p[rival], p[label] = 0.45, 0.35
    elif kind == "easy":
        p[label], p[rival] = 0.45, 0.35
    else:
        p[label] = 0.9
    return p / p.sum()


def _kind(i, model):
    if i % 4 == 3:
        return "sure"
    return "hard" if (i + (model == "heldout")) % 2 == 0 else "easy"


@pytest.fixture
def lab(tmp_path):
    (tmp_path / "datasets/manifests").mkdir(parents=True)
    (tmp_path / "datasets/manifests/field.csv").write_text("\n".join(_field_rows()) + "\n")
    (tmp_path / "datasets/manifests/lab.csv").write_text(
        "\n".join(["path,label,group,split,sha1"] + [f"l/{i}.jpg,{n},g{i},lab_dev,l{i}" for i, n in enumerate(NAMES)]) + "\n")
    (tmp_path / "datasets/label_exclusions.csv").write_text(
        "version,image_id,path,source,split,label,reason_code,note,began_as\n"
        "1,s1,f/1.jpg,plantdoc,field_dev,Tomato___Early_blight,wrong_plant,x,remove\n")
    (tmp_path / "orchestration").mkdir()
    (tmp_path / "orchestration/tasks.json").write_text("{}")
    (tmp_path / "datasets/followup").mkdir()
    (tmp_path / ".gitignore").write_text("datasets/followup/sheets/\ndatasets/followup/key/\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _e4(root, system_kinds, subset):
    info = split_info("field_dev", root)
    pos = subset_positions(subset, "field_dev", info, root)
    paths = [info.ids[i] for i in pos]
    labels = np.array([NAMES.index(info.labels[i]) for i in pos])
    preds = {}
    for system, model in system_kinds.items():
        probs = np.array([_model_probs(y, _kind(int(p.split("/")[1].split(".")[0]), model)) for p, y in zip(paths, labels)])
        preds[system] = Predictions(paths, labels, probs)
    metrics = {s: E.standard_image_metrics(p.probs, labels, NAMES) for s, p in preds.items()}
    out = write_run("E4", "field_dev", {"class_names": NAMES}, {s: s for s in preds}, metrics, {}, [{"setting": "x"}],
                    preds, info.hash, root=root, subset=subset)
    return out.stem


def _config(rids):
    cfg = json.loads((ROOT / "configs" / "e11.json").read_text(encoding="utf-8"))
    cfg["sheets"] = {**cfg["sheets"], **SHEET_CFG, "photos_per_sheet": 6, "columns": 3}
    cfg["models"] = {
        "seen": {"record": rids[0], "system": "convnext_joint"},
        "heldout_plantdoc": {"record": rids[1], "system": "heldout_plantdoc", "source": "plantdoc"},
        "heldout_plantseg": {"record": rids[2], "system": "heldout_plantseg", "source": "plantseg"},
    }
    return cfg


def _answered_world(lab, oracle=True, flip=False):
    """Fake E4 records, a plan, and answers a test oracle gives (it knows the class; the real answerer does not)."""
    rids = [
        _e4(lab, {"convnext_joint": "joint"}, "shared_classes"),
        _e4(lab, {"convnext_joint": "joint", "heldout_plantdoc": "heldout"}, "source:plantdoc"),
        _e4(lab, {"convnext_joint": "joint", "heldout_plantseg": "heldout"}, "source:plantseg"),
    ]
    cfg = _config(rids)
    plan = E.make_plan(cfg, lab, FAKE_BANK, NAMES)
    truth = {r.split(",")[0]: r.split(",")[1] for r in _field_rows()[1:]}
    answers = []
    for entry in plan["plan"]:
        for q in entry["needed"] + entry["decoys"]:
            cls = next(b.classes[0] for b in FAKE_BANK if b.id == q)
            yes = (cls == truth[entry["path"]]) != flip
            answers.append({"path": entry["path"], "question": q, "answer": "yes" if yes else "no"})
    answers.sort(key=lambda r: (r["path"], r["question"]))
    (lab / "datasets/followup/answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in answers))
    (lab / "datasets/followup/answers.meta.json").write_text(json.dumps({
        "answerer": "a test oracle", "kind": E.KIND, "sheets": 1, "photos": len(plan["plan"]), "rows": len(answers),
        "yes": 0, "no": 0, "unknown": 0}))
    _git(lab, "add", "-A")
    _git(lab, "commit", "-q", "-m", "answers")
    return cfg, plan


def test_plan_only_covers_unsure_cleaned_field_dev_photos(lab):
    cfg, plan = _answered_world(lab)
    paths = [e["path"] for e in plan["plan"]]
    assert paths and "f/1.jpg" not in paths  # excluded by the label cleaning
    assert all(int(p.split("/")[1].split(".")[0]) < N_PHOTOS for p in paths)  # no field_test photo
    assert plan["summary"]["seen"]["unsure"] > 0
    assert set(plan["summary"]) == {"seen", "heldout_plantdoc", "heldout_plantseg"}


def test_unsure_subset_gain_is_not_mistaken_for_a_whole_split_gain(lab):
    cfg, _ = _answered_world(lab)
    paths = E.run_all(cfg, lab, bank=FAKE_BANK)
    assert len(paths) == 3
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert rec["experiment"] == "E11" and rec["subset"] == "clean:1" and rec["split"] == "field_dev"
    assert rec["config"]["device"].startswith("none")
    flow = rec["config"]["flow"]["joint"]
    assert flow["asked_subset"]["accuracy_before"] < flow["asked_subset"]["accuracy_after"]
    assert flow["whole_split"]["accuracy_after"] - flow["whole_split"]["accuracy_before"] < (
        flow["asked_subset"]["accuracy_after"] - flow["asked_subset"]["accuracy_before"])
    assert flow["better"] > 0 and flow["worse"] == 0 and flow["n_rows"] == rec["n"]
    # the whole-split numbers in the record are the ones check_run recomputes
    assert rec["metrics"]["joint_after"]["accuracy"] == pytest.approx(flow["whole_split"]["accuracy_after"])
    assert check("E11", root=lab, min_n=10, split="field_dev") == []


def test_wrong_answers_make_it_worse_and_that_is_counted(lab):
    cfg, _ = _answered_world(lab, flip=True)
    rec = json.loads(E.run_all(cfg, lab, bank=FAKE_BANK)[0].read_text(encoding="utf-8"))
    flow = rec["config"]["flow"]["joint"]
    assert flow["worse"] > 0 and flow["better"] == 0
    assert flow["asked_subset"]["accuracy_after"] < flow["asked_subset"]["accuracy_before"]
    lo, hi = flow["asked_subset"]["accuracy_diff_ci"]
    assert hi < 0


def test_flow_summary_matches_the_saved_rows(lab):
    cfg, _ = _answered_world(lab)
    rec = json.loads(E.run_all(cfg, lab, bank=FAKE_BANK)[0].read_text(encoding="utf-8"))
    rows = {s: [json.loads(x) for x in (lab / rec["predictions"][s]).read_text().splitlines()]
            for s in ("joint_before", "joint_after")}
    before = {r["path"]: r for r in rows["joint_before"]}
    asked = [r for r in rows["joint_after"] if r["asked"]]
    flow = rec["config"]["flow"]["joint"]
    assert flow["questions_asked"] == sum(len(r["asked"]) for r in asked)
    assert flow["asked_subset"]["n"] == len(asked)
    assert max(len(r["asked"]) for r in asked) <= 2
    better = sum(np.argmax(r["probs"]) == r["label"] and np.argmax(before[r["path"]]["probs"]) != r["label"] for r in asked)
    assert flow["better"] == better
    assert all(r["probs"] == before[r["path"]]["probs"] for r in rows["joint_after"] if not r["asked"])
    for r in asked:  # the top-3 mass does not move
        assert sum(r["probs"]) == pytest.approx(sum(before[r["path"]]["probs"]))
    assert flow["answers"] == {"yes": flow["answers"]["yes"], "no": flow["answers"]["no"], "unknown": 0}
    assert flow["unknown_rate"] == 0


def test_held_out_models_sit_beside_the_seen_one_with_the_same_answers(lab):
    cfg, _ = _answered_world(lab)
    seen, pdoc, pseg = (json.loads(p.read_text(encoding="utf-8")) for p in E.run_all(cfg, lab, bank=FAKE_BANK))
    assert pdoc["subset"] == "source:plantdoc+clean:1" and pseg["subset"] == "source:plantseg+clean:1"
    assert set(pdoc["config"]["flow"]) == {"joint", "heldout"} == set(pseg["config"]["flow"])
    assert {c["a"] for c in pdoc["comparisons"]} == {"joint_after", "heldout_after"}
    assert pdoc["config"]["seen_source_number"]["accuracy_after"] == pytest.approx(
        seen["config"]["flow"]["joint"]["whole_split"]["accuracy_after"])
    assert check("E11", root=lab, min_n=10, split="field_dev") == []


def test_mismatched_saved_rows_are_refused(lab):
    cfg, _ = _answered_world(lab)
    rec = json.loads((lab / "results/runs" / f"{cfg['models']['heldout_plantdoc']['record']}.json").read_text())
    path = lab / rec["predictions"]["convnext_joint"]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        r["probs"][0] += 0.01
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="do not match"):
        E.run_all(cfg, lab, bank=FAKE_BANK)


def test_record_refuses_answers_that_are_not_committed_first(lab):
    cfg, _ = _answered_world(lab)
    (lab / "datasets/followup/answers.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="answers"):
        E.run_all(cfg, lab, bank=FAKE_BANK)


def test_missing_answer_for_a_planned_question_is_refused(lab):
    cfg, _ = _answered_world(lab)
    path = lab / "datasets/followup/answers.jsonl"
    path.write_text("")
    _git(lab, "add", "-A")
    _git(lab, "commit", "-q", "-m", "answers emptied")
    with pytest.raises(ValueError, match="no saved answer"):
        E.run_all(cfg, lab, bank=FAKE_BANK)


def test_unknown_answers_are_counted_and_change_nothing(lab):
    cfg, _ = _answered_world(lab)
    path = lab / "datasets/followup/answers.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    path.write_text("".join(json.dumps({**r, "answer": "unknown"}) + "\n" for r in rows))
    _git(lab, "add", "-A")
    _git(lab, "commit", "-q", "-m", "all unknown")
    rec = json.loads(E.run_all(cfg, lab, bank=FAKE_BANK)[0].read_text(encoding="utf-8"))
    flow = rec["config"]["flow"]["joint"]
    assert flow["unknown_rate"] == 1 and flow["better"] == 0 and flow["worse"] == 0
    assert rec["metrics"]["joint_after"] == rec["metrics"]["joint_before"]


def test_field_test_is_never_read(lab):
    cfg, plan = _answered_world(lab)
    for rec_path in E.run_all(cfg, lab, bank=FAKE_BANK):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        assert rec["split"] == "field_dev"
        for rel in rec["predictions"].values():
            for line in (lab / rel).read_text().splitlines():
                assert int(json.loads(line)["path"].split("/")[1].split(".")[0]) < N_PHOTOS
    (lab / "c.json").write_text(json.dumps({**cfg, "split": "field_test"}))
    with pytest.raises(ValueError, match="field_dev only"):
        E.load_config(lab / "c.json")


def test_ece_difference_interval_is_paired_and_zero_when_nothing_changed():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=60)
    labels = rng.integers(0, 4, size=60)
    groups = [f"g{i // 2}" for i in range(60)]
    diff, lo, hi = E.ece_diff(probs, probs, labels, groups, n=200, seed=1)
    assert diff == lo == hi == 0
    sharper = probs ** 3 / (probs ** 3).sum(axis=1, keepdims=True)
    diff, lo, hi = E.ece_diff(probs, sharper, labels, groups, n=200, seed=1)
    assert diff == pytest.approx(E.M.ece(sharper, labels) - E.M.ece(probs, labels)) and lo <= diff <= hi


def test_yes_rates_split_decoys_wrong_candidates_and_the_true_class():
    import importlib.util

    spec = importlib.util.spec_from_file_location("e11_answer_check", ROOT / "tools" / "e11_answer_check.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    plan = [{"path": "p", "needed": ["tl-1", "te-1"], "decoys": ["cr-1"]}]
    answers = [{"path": "p", "question": q, "answer": a} for q, a in (("tl-1", "yes"), ("te-1", "no"), ("cr-1", "unknown"))]
    got = tool.yes_rates(answers, plan, {"p": NAMES[0]}, FAKE_BANK)
    assert got["candidate_true_class"]["yes_rate"] == 1 and got["candidate_wrong_class"]["yes_rate"] == 0
    assert got["decoy"]["unknown_rate"] == 1 and got["decoy"]["n"] == 1
