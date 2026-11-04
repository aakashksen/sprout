"""Fixed RAG evaluation set: loading, page lookup, and the human review sample."""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("supported", "not_in_documents", "wrong_crop", "ambiguous", "multi_turn")


@dataclass
class Question:
    id: str
    category: str
    split: str
    question: str
    expected: str
    crop: str | None = None
    history: list[dict] = field(default_factory=list)
    gold_source: str | None = None
    gold_page: int | None = None
    answer_terms: list[str] = field(default_factory=list)
    distractor_source: str | None = None
    absent_terms: list[str] = field(default_factory=list)
    trap_terms: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def load_questions(path: Path) -> list[Question]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\xad", "")
    text = re.sub("[‐-―]", "-", text)
    return re.sub(r"\s+", " ", text).lower()


def _rows(chunks_path: Path):
    with open(chunks_path, encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def index_rows(chunks_path: Path) -> list[dict]:
    return list(_rows(chunks_path))


def index_pages(chunks_path: Path) -> dict[tuple[str, int], str]:
    pages: dict[tuple[str, int], list[str]] = {}
    for r in _rows(chunks_path):
        pages.setdefault((r["source"], r["page"]), []).append(r["text"])
    return {k: normalize(" ".join(v)) for k, v in pages.items()}


def index_crops(chunks_path: Path) -> dict[str, str]:
    return {r["source"]: r["crop"] for r in _rows(chunks_path)}


def write_review_sample(questions: list[Question], path: Path, n: int = 30, chunks_path: Path | None = None) -> None:
    """Evenly spaced picks per category, so the sample is the same every run."""
    per = n // len(CATEGORIES)
    picked: list[Question] = []
    for c in CATEGORIES:
        group = sorted((q for q in questions if q.category == c), key=lambda q: q.id)
        picked += [group[i * len(group) // per] for i in range(per)]
    pages = index_pages(chunks_path) if chunks_path and Path(chunks_path).exists() else {}
    out = ["# RAG evaluation set: review sample (#53)", "",
           "Check each gold label against the source page. Mark any entry that is wrong or unclear.", ""]
    for q in picked:
        out += [f"### {q.id} ({q.category}, {q.split})", ""]
        for turn in q.history:
            out.append(f"- {turn['role']}: {turn['content']}")
        out += [f"- question: {q.question}", f"- expected: {q.expected}"]
        if q.crop:
            out.append(f"- stated crop: {q.crop}")
        if q.gold_source:
            out += [f"- gold: {q.gold_source}, page {q.gold_page}", f"- answer terms: {', '.join(q.answer_terms)}"]
            text = pages.get((q.gold_source, q.gold_page), "")
            at = text.find(normalize(q.answer_terms[0])) if q.answer_terms else -1
            if at >= 0:
                out.append(f"- passage: ...{text[max(0, at - 150):at + 250]}...")
        if q.distractor_source:
            out.append(f"- the disease is documented for another crop in: {q.distractor_source}")
        if q.absent_terms:
            out.append(f"- not in the documents: {', '.join(q.absent_terms)}")
        if q.missing:
            out.append(f"- missing detail: {', '.join(q.missing)}")
        out.append("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf8")


def answer_candidates(chunks_path: Path, plant_terms: list[str], topic: str) -> list[tuple[str, int]]:
    """Pages holding a chunk that names the plant (or a family term) and the problem together."""
    plant = re.compile(r"\b(?:%s)" % "|".join(re.escape(t) for t in plant_terms), re.I)
    problem = re.compile(topic, re.I)
    found = {
        (r["source"], r["page"]) for r in _rows(chunks_path) if plant.search(r["text"]) and problem.search(r["text"])
    }
    return sorted(found)
