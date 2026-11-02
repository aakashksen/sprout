import json
from pathlib import Path

import numpy as np

from sprout.rag.retrieve import RetrievalConfig, Retriever

CHUNKS = [
    ("Tomato early blight causes concentric rings on leaves.", "tomato.pdf", 3, "tomato", ["early blight"]),
    ("Tomato leaf mold appears as yellow patches.", "tomato2.pdf", 1, "tomato", ["leaf mold"]),
    ("Potato late blight rots tubers quickly.", "potato.pdf", 7, "potato", ["late blight"]),
    ("Always wash hands and clean pruning tools between plants.", "general.pdf", 2, "general", []),
]

TABLE = {
    "early blight": [1, 0, 0, 0],
    "leaf mold": [0, 1, 0, 0],
    "late blight": [0, 0, 1, 0],
    "wash hands": [0, 0, 0, 1],
}


class FakeEmbedder:
    """Vector per text from a keyword table, so scores are fully controlled."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def _vec(self, text):
        for key, v in self.table.items():
            if key in text.lower():
                return np.array(v, dtype=float)
        return np.array([0.0, 0.0, 0.0, 1.0])

    def embed_passages(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make(tmp_path: Path, embedder=None, **cfg) -> Retriever:
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for text, source, page, crop, diseases in CHUNKS:
            row = dict(text=text, source=source, page=page, crop=crop, diseases=diseases, title="t")
            f.write(json.dumps(row) + "\n")
    return Retriever(tmp_path, config=RetrievalConfig(**cfg), embedder=embedder or FakeEmbedder(TABLE))


def test_crop_filter_excludes_other_crops(tmp_path):
    r = make(tmp_path, min_score=0.0)
    res = r.search("blight", crop="tomato", mode="dense", rerank=False)
    assert res and {p.crop for p in res} == {"tomato"}


def test_general_docs_only_as_fallback(tmp_path):
    r = make(tmp_path, min_score=0.5)
    hit = r.search("early blight", crop="tomato", mode="dense", rerank=False)
    assert [p.crop for p in hit] == ["tomato"]
    fallback = r.search("wash hands", crop="tomato", mode="dense", rerank=False)
    assert [p.crop for p in fallback] == ["general"]


def test_low_score_passages_are_dropped_individually(tmp_path):
    table = {"early blight": [1, 0, 0, 0], "leaf mold": [0.6, 0.8, 0, 0], "late blight": [0, 0, 1, 0]}
    r = make(tmp_path, embedder=FakeEmbedder(table), min_score=0.5)
    res = r.search("early blight", crop="tomato", mode="dense", rerank=False)
    assert [p.source for p in res] == ["tomato.pdf", "tomato2.pdf"]  # 1.0 and 0.6 stay
    r2 = make(tmp_path, embedder=FakeEmbedder(table), min_score=0.7)
    res2 = r2.search("early blight", crop="tomato", mode="dense", rerank=False)
    assert [p.source for p in res2] == ["tomato.pdf"]  # 0.6 dropped, 1.0 kept


def test_bm25_finds_exact_disease_name(tmp_path):
    r = make(tmp_path, min_score=0.5)
    res = r.search("late blight", mode="bm25", rerank=False)
    assert res[0].source == "potato.pdf"


def test_passage_carries_source_and_page(tmp_path):
    r = make(tmp_path, min_score=0.0)
    p = r.search("leaf mold", mode="hybrid", rerank=False)[0]
    assert (p.source, p.page, p.crop) == ("tomato2.pdf", 1, "tomato")
    assert p.text and p.score > 0


def test_empty_result_is_empty_not_error(tmp_path):
    r = make(tmp_path, min_score=0.99)
    assert r.search("zzz unknown", mode="hybrid", rerank=False) == []
    assert Retriever(tmp_path / "missing").search("anything") == []
    assert r.search("   ") == []
