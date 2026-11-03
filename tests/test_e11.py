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


@needs_raw
def test_every_passage_is_in_the_cited_page():
    from sprout.rag.ingest import _extract

    pages: dict[str, dict[int, str]] = {}
    for item in BANK:
        if item.document not in pages:
            pages[item.document] = {n: F.normalize(t) for n, t in _extract(RAW / item.document)}
        assert item.page in pages[item.document], f"{item.id}: {item.document} has no page {item.page}"
        assert F.normalize(item.passage) in pages[item.document][item.page], f"{item.id}: passage not on the page"


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


def test_real_answers_precede_every_real_record():
    records = list((ROOT / "results" / "runs").glob("E11-*.json"))
    assert not records or E.commit_order_problems(ROOT) == []
    answers = ROOT / "datasets" / "followup" / "answers.jsonl"
    if answers.exists():
        for line in answers.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"path", "question", "answer"}
