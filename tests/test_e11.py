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
