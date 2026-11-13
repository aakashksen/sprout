"""Retriever options measured in E7: floor per mode, crops from titles, context header, CUDA-or-fail."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sprout.rag.gpu_models import resolve_device
from sprout.rag.retrieve import RetrievalConfig, Retriever, chunk_search_text


class Vec:
    """Two-dimensional vectors from keywords: tomato on x, pepper on y."""

    def _v(self, text):
        t = text.lower()
        return np.array([1.0 if "tomato" in t else 0.0, 1.0 if "pepper" in t else 0.0]) + 0.01

    def embed_passages(self, texts):
        return np.array([self._v(t) for t in texts])

    def embed_query(self, text):
        return self._v(text)


def make(tmp_path, rows, **cfg):
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"diseases": [], "title": "t", **r}) + "\n")
    return Retriever(tmp_path, config=RetrievalConfig(**cfg), embedder=Vec())


MIXED = {"text": "Bacterial spot makes dark lesions.", "source": "mixed.md", "page": 1, "crop": "tomato",
         "title": "Bacterial spot of tomato and pepper"}
PEPPER = {"text": "Pepper plants need warm soil.", "source": "pep.pdf", "page": 1, "crop": "pepper"}


def test_floor_is_chosen_per_mode_and_rerank(tmp_path):
    r = make(tmp_path, [MIXED, PEPPER], min_score=0.0, floors={"dense": 0.9, "hybrid+rerank": 0.1})
    assert r.config.floor_for("dense", False) == 0.9
    assert r.config.floor_for("hybrid", True) == 0.1
    assert r.config.floor_for("bm25", False) == 0.0  # a mode with no floor of its own uses min_score
    assert r.search("pepper", mode="dense", rerank=False) != []  # the pepper chunk scores about 1
    assert r.search("tomato", crop="pepper", mode="dense", rerank=False) == []  # about 0.01, under 0.9


def test_crops_from_title_lets_a_mixed_page_answer_for_both_crops(tmp_path):
    off = make(tmp_path, [MIXED, PEPPER], min_score=0.0)
    on = make(tmp_path, [MIXED, PEPPER], min_score=0.0, crops_from_title=True)
    assert "mixed.md" not in {p.source for p in off.search("bacterial spot", crop="pepper", rerank=False)}
    assert "mixed.md" in {p.source for p in on.search("bacterial spot", crop="pepper", rerank=False)}
    assert {p.source for p in on.search("bacterial spot", crop="tomato", rerank=False)} == {"mixed.md"}


def test_crops_from_title_never_adds_a_crop_the_title_lacks(tmp_path):
    apple = {"text": "Tomato ringspot virus is a pest.", "source": "apple.pdf", "page": 1, "crop": "apple",
             "title": "Apple pest guide"}
    r = make(tmp_path, [MIXED, apple], min_score=0.0, crops_from_title=True)
    assert {p.source for p in r.search("virus", crop="tomato", rerank=False)} == {"mixed.md"}


def test_header_is_searched_but_not_shown(tmp_path):
    row = {"text": "Dark lesions on leaves.", "source": "a.pdf", "page": 1, "crop": "pepper",
           "header": "Document: Anthracnose handbook"}
    r = make(tmp_path, [row], min_score=0.0)
    found = r.search("anthracnose", mode="bm25", rerank=False)
    assert [p.text for p in found] == ["Dark lesions on leaves."]
    assert chunk_search_text(row).startswith("Document: Anthracnose handbook")
    assert chunk_search_text({"text": "x"}) == "x"


def test_cuda_is_required_not_replaced_by_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    assert resolve_device("cpu") == "cpu"
