"""Hybrid retrieval over the chunk index: dense + BM25, optional rerank, per-passage score filter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

GENERAL_CROP = "general"
_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    metric: str = "cosine"  # vectors are L2-normalised, so the dot product is the cosine
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    min_score: float = 0.3  # applied to the final score of each passage on its own
    rerank_pool: int = 3  # rerank k * rerank_pool candidates
    dense_weight: float = 0.5  # hybrid = dense_weight * dense + (1 - dense_weight) * bm25
    floors: dict[str, float] = field(default_factory=dict)  # per "mode" or "mode+rerank"; absent means min_score
    crops_from_title: bool = False  # a page whose title names several crops answers for each of them

    def floor_for(self, mode: str, rerank: bool) -> float:
        return self.floors.get(f"{mode}+rerank" if rerank else mode, self.min_score)


@dataclass
class Passage:
    text: str
    source: str
    page: int
    crop: str
    score: float


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts)))

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(next(iter(self._model.query_embed(text))))


class FastReranker:
    def __init__(self, model: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        # cross-encoder logits squashed to 0..1 so one threshold scale fits every mode
        return [1 / (1 + math.exp(-s)) for s in self._model.rerank(query, texts)]


def vectors_cache_path(index_dir: Path, model: str, texts: list[str]) -> Path:
    """Where the embedded chunks of an index are saved, keyed by model and chunk text."""
    key = hashlib.sha1(chr(0).join([model, *texts]).encode()).hexdigest()[:16]
    return Path(index_dir) / f"vectors-{key}.npy"


def chunk_search_text(row: dict) -> str:
    """What is embedded, matched and reranked for a chunk: its context header (when the index has one) and text."""
    header = row.get("header")
    return header + "\n" + row["text"] if header else row["text"]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n == 0, 1, n)


class Retriever:
    def __init__(
        self,
        index_dir: Path,
        *,
        config: RetrievalConfig | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._embedder = embedder
        self._reranker = reranker
        self._index_dir = Path(index_dir)
        path = self._index_dir / "chunks.jsonl"
        self._chunks: list[dict] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self._chunks = [json.loads(line) for line in f if line.strip()]
        self._texts = [chunk_search_text(c) for c in self._chunks]
        self._toks = [_tokens(t) for t in self._texts]
        self._crops_of = self._chunk_crops()
        self._df = Counter(t for toks in self._toks for t in set(toks))
        self._avg_len = (sum(map(len, self._toks)) / len(self._toks)) if self._toks else 0.0
        self._vectors: np.ndarray | None = None
        self._lock = threading.RLock()

    def _chunk_crops(self) -> list[set[str]]:
        """Crops each chunk answers for: its own, plus (when enabled) other crops its title or diseases name."""
        own = [{c["crop"]} for c in self._chunks]
        if not self.config.crops_from_title:
            return own
        names = {c["crop"] for c in self._chunks} - {GENERAL_CROP}
        found = []
        for crops, c in zip(own, self._chunks, strict=True):
            label = " ".join([c.get("title", ""), *c.get("diseases", [])]).lower()
            found.append(crops | {n for n in names if re.search(r'\b' + re.escape(n) + r'(?:s|es)?\b', label)})
        return found

    def _get_embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = FastEmbedder(self.config.embedding_model)
            return self._embedder

    def _get_reranker(self) -> Reranker:
        with self._lock:
            if self._reranker is None:
                self._reranker = FastReranker(self.config.reranker_model)
            return self._reranker

    def _load_vectors(self) -> np.ndarray:
        """Embed every chunk once. With the default embedder the result is saved in the index
        folder, keyed by model and chunk text, so a new Retriever starts without re-embedding.
        An unreadable or wrong-sized cache file is ignored and rewritten."""
        texts = self._texts
        cache = None
        if self._embedder is None or getattr(self._embedder, "cache_ok", False):
            cache = vectors_cache_path(self._index_dir, self.config.embedding_model, texts)
            try:
                saved = np.load(cache)
                if saved.ndim == 2 and saved.shape[0] == len(texts):
                    return saved
            except (OSError, ValueError, EOFError):
                pass
        vectors = _unit(np.asarray(self._get_embedder().embed_passages(texts)))
        if cache is not None:
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                with tmp.open("wb") as f:
                    np.save(f, vectors)
                os.replace(tmp, cache)  # readers never see a half-written file
            except OSError:
                tmp.unlink(missing_ok=True)  # a read-only index folder only costs the re-embedding
        return vectors

    @property
    def crops(self) -> set[str]:
        """Plants that have documents of their own (general guidance is not a plant)."""
        return {c["crop"] for c in self._chunks} - {GENERAL_CROP}

    @property
    def warmed(self) -> bool:
        return self._vectors is not None

    def warm(self, *, rerank: bool = False) -> None:
        """Do the slow first-use work now: embed (or load) the index, and optionally load the reranker."""
        if self._chunks:
            self._ensure_vectors()
        if rerank:
            self._get_reranker()

    def _ensure_vectors(self) -> np.ndarray:
        with self._lock:  # one embed at a time; later callers reuse its result
            if self._vectors is None:
                self._vectors = self._load_vectors()
            return self._vectors

    def _dense(self, query: str, idx: list[int]) -> dict[int, float]:
        vectors = self._vectors if self._vectors is not None else self._ensure_vectors()
        q = _unit(np.asarray(self._get_embedder().embed_query(query)))
        sims = vectors[idx] @ q
        return {i: float(max(s, 0.0)) for i, s in zip(idx, sims, strict=True)}

    def _bm25(self, query: str, idx: list[int]) -> dict[int, float]:
        n, k1, b = len(self._chunks), 1.5, 0.75
        q = set(_tokens(query))
        raw: dict[int, float] = {}
        for i in idx:
            tf = Counter(self._toks[i])
            s = 0.0
            for t in q & tf.keys():
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                norm = tf[t] + k1 * (1 - b + b * len(self._toks[i]) / self._avg_len)
                s += idf * tf[t] * (k1 + 1) / norm
            raw[i] = s
        top = max(raw.values(), default=0.0)
        return {i: (s / top if top > 0 else 0.0) for i, s in raw.items()}

    def _rank(self, query: str, idx: list[int], k: int, mode: str, rerank: bool) -> list[Passage]:
        if not idx:
            return []
        w = self.config.dense_weight
        if mode == "dense":
            scores = self._dense(query, idx)
        elif mode == "bm25":
            scores = self._bm25(query, idx)
        else:
            d, bm = self._dense(query, idx), self._bm25(query, idx)
            scores = {i: w * d[i] + (1 - w) * bm[i] for i in idx}
        order = sorted(idx, key=lambda i: scores[i], reverse=True)
        if rerank:
            order = order[: k * self.config.rerank_pool]
            texts = [self._texts[i] for i in order]
            scores = dict(zip(order, self._get_reranker().score(query, texts), strict=True))
            order.sort(key=lambda i: scores[i], reverse=True)
        floor = self.config.floor_for(mode, rerank)
        kept = [i for i in order if scores[i] >= floor][:k]
        return [
            Passage(
                text=self._chunks[i]["text"],
                source=self._chunks[i]["source"],
                page=self._chunks[i]["page"],
                crop=self._chunks[i]["crop"],
                score=scores[i],
            )
            for i in kept
        ]

    def search(
        self,
        query: str,
        *,
        crop: str | None = None,
        disease: str | None = None,
        k: int = 8,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> list[Passage]:
        """`query` is only the retrieval text; conversation instructions must not be mixed into it.

        With a crop, only that crop's documents are searched; general guidance is used
        only when none of them survives the score filter.
        """
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        if not self._chunks or not query.strip():
            return []
        crop = crop.strip().lower() if crop else None
        disease = disease.strip().lower() if disease else None

        def has_disease(c: dict) -> bool:
            return disease is None or disease in (d.lower() for d in c["diseases"])

        if crop is None:
            idx = [i for i, c in enumerate(self._chunks) if has_disease(c)]
            return self._rank(query, idx, k, mode, rerank)
        idx = [i for i, c in enumerate(self._chunks) if crop in self._crops_of[i] and has_disease(c)]
        found = self._rank(query, idx, k, mode, rerank)
        if found:
            return found
        general = [i for i, c in enumerate(self._chunks) if c["crop"] == GENERAL_CROP]
        return self._rank(query, general, k, mode, rerank)
