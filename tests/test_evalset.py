import re
from collections import Counter
from pathlib import Path

import pytest

from sprout.rag.evalset import (
    CATEGORIES,
    index_crops,
    index_pages,
    load_questions,
    normalize,
    write_review_sample,
)
from sprout.rag.evalset import (
    index_rows as _index_rows,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "datasets" / "rag" / "questions.jsonl"
SAMPLE = ROOT / "datasets" / "review" / "rag-sample.md"
CHUNKS = ROOT / "knowledge" / "index" / "chunks.jsonl"

needs_index = pytest.mark.skipif(not CHUNKS.exists(), reason="knowledge/index is git-ignored and absent here")


def test_category_counts():
    qs = load_questions(QUESTIONS)
    counts = Counter(q.category for q in qs)
    assert set(counts) == set(CATEGORIES)
    assert all(counts[c] >= 20 for c in CATEGORIES)
    assert len(qs) >= 150
    assert len({q.id for q in qs}) == len(qs)
    for c in CATEGORIES:
        assert {q.split for q in qs if q.category == c} == {"dev", "test"}


def test_no_document_in_both_splits_for_supported_questions():
    qs = load_questions(QUESTIONS)
    by_doc: dict[str, set[str]] = {}
    for q in qs:
        if q.gold_source:
            by_doc.setdefault(q.gold_source, set()).add(q.split)
    assert by_doc
    assert all(len(s) == 1 for s in by_doc.values()), {d: s for d, s in by_doc.items() if len(s) > 1}
    assert all(q.gold_source for q in qs if q.category in ("supported", "multi_turn"))


@needs_index
def test_gold_pages_exist_and_contain_the_answer_terms():
    pages = index_pages(CHUNKS)
    gold = [q for q in load_questions(QUESTIONS) if q.gold_source]
    assert gold
    for q in gold:
        key = (q.gold_source, q.gold_page)
        assert key in pages, f"{q.id}: no page {key}"
        assert q.answer_terms, q.id
        for term in q.answer_terms:
            assert normalize(term) in pages[key], f"{q.id}: {term!r} not on {key}"


@needs_index
def test_not_in_documents_terms_are_absent_and_wrong_crop_traps_are_wrong():
    corpus = " ".join(index_pages(CHUNKS).values())
    crops = index_crops(CHUNKS)
    for q in load_questions(QUESTIONS):
        if q.category == "not_in_documents":
            assert q.absent_terms, q.id
            for t in q.absent_terms:
                assert not re.search(rf"\b{re.escape(normalize(t))}", corpus), f"{q.id}: {t!r} is in the corpus"
        if q.category == "wrong_crop":
            assert crops[q.distractor_source] != q.crop, q.id


@needs_index
def test_wrong_crop_trap_terms_absent_for_stated_crop_and_present_in_distractor():
    rows = list(_index_rows(CHUNKS))
    for q in load_questions(QUESTIONS):
        if q.category != "wrong_crop":
            continue
        assert q.trap_terms, f"{q.id}: no trap_terms"
        own = " ".join(normalize(r["text"]) for r in rows if r["crop"] == q.crop)
        theirs = " ".join(normalize(r["text"]) for r in rows if r["source"] == q.distractor_source)
        for t in q.trap_terms:
            pat = rf"\b{re.escape(normalize(t))}"
            assert not re.search(pat, own), f"{q.id}: {t!r} is documented for {q.crop}"
            assert re.search(pat, theirs), f"{q.id}: {t!r} not in {q.distractor_source}"


def test_review_sample_has_thirty_entries(tmp_path):
    qs = load_questions(QUESTIONS)
    out = tmp_path / "s.md"
    write_review_sample(qs, out, n=30)
    assert len(re.findall(r"^### ", out.read_text(encoding="utf8"), re.M)) == 30
    committed = SAMPLE.read_text(encoding="utf8")
    assert len(re.findall(r"^### ", committed, re.M)) == 30
    assert {q.id for q in qs} >= set(re.findall(r"^### (\S+)", committed, re.M))
