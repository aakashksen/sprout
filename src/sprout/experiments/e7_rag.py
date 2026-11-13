"""E7: which retrieval setup and prompt give faithful, cited answers, on rag_dev only.

The stage logic lives here so tests can run it with fakes; tools/run_e7.py wires in the GPU, Ollama and the
command line. Every stage checkpoints per question, so a crash resumes; a partial run is never a result.
rag_test is never read here.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sprout.atomic import atomic_write_text
from sprout.eval.check_run import (
    Scored,
    recompute_comparison,
    recompute_interval,
    recompute_metric,
)
from sprout.eval.records import RAG_REL, ROOT, split_info, subset_positions, write_run
from sprout.eval.stats import mcnemar_exact, paired_bootstrap_diff
from sprout.rag.answer import (
    Answer,
    AnswerConfig,
    answer,
    bounded_history,
    retrieve,
)
from sprout.rag.evalset import Question, index_rows, load_questions, normalize
from sprout.rag.judge import blind_view, judge_answer
from sprout.rag.query import build_query, crop_of_label, detect_crop, disease_of_label
from sprout.rag.retrieve import Passage
from sprout.serve.types import Candidate, Identification

SPLIT = "rag_dev"
RANK_K = 10  # gold is looked for in the top 10 passages
RETRIEVAL_METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")
DECIDING_METRICS = ("recall@5", "mrr")
METHOD_VERSION = 2  # bumped when a selection rule changes; checkpoints and work folders of older versions are not reused
SERVED_K = 5
# Bumped when what answer() does with a reply changes (notes, declines, plant reading). Generated rows are keyed by it,
# so rows made by older code are never reused or mixed into a record.
ANSWER_VERSION = 3  # 3: answer() searches again with the diagnosed disease when the question finds nothing
FLOOR_MIN_ANSWERS = 0.80
MIN_GRADED = 200  # graded answers one full run should yield (T6.6 re-checks 200 of them)
BEHAVIOUR_METRICS = (
    "answers_when_answerable", "declines_when_unanswerable", "declines_when_crop_mismatch", "declines_when_ambiguous",
)
GOLD_RULE = (
    "A passage is the gold hit when its source and page equal the question's gold source and page. A document the "
    "index holds as a single page (every HTML page and Markdown file is 'page 1') would match any of its chunks, so "
    "there the passage must also contain one of the question's answer terms."
)
CROP_RULE = (
    "The plant comes from the question text (or the question's stated crop), never from the gold label. A top "
    "disease is simulated from the class names of that plant that the question names; a plant with no such class "
    "has no photo diagnosis."
)


@dataclass(frozen=True)
class RetrievalSpec:
    index: str = "index"  # folder under knowledge/
    mode: str = "hybrid"
    rerank: bool = False
    query_mode: str = "question"
    crops_from_title: bool = False

    @classmethod
    def from_json(cls, d: dict) -> RetrievalSpec:
        return cls(**d)


# ---------------------------------------------------------------- checkpoints

class Checkpoint:
    """One JSON row per line, appended and fsynced. A torn last line (a crash mid-write) is dropped on load."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        good: list[str] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    break
                self.rows[row["id"]] = row
                good.append(line)
            if len(good) != len(self.path.read_text(encoding="utf-8").splitlines()):
                atomic_write_text(self.path, "".join(x + "\n" for x in good))

    def add(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows[row["id"]] = row


def index_stamp(index_dir: Path) -> str:
    """Identifies the index build, so a rebuilt index never reuses old checkpoints."""
    info = Path(index_dir) / "build-info.json"
    if info.is_file():
        return str(json.loads(info.read_text(encoding="utf-8")).get("built"))
    chunks = Path(index_dir) / "chunks.jsonl"
    return f"{chunks.stat().st_size}-{int(chunks.stat().st_mtime)}" if chunks.is_file() else "none"


def checkpoint_key(*parts: Any) -> str:
    """Short hash of everything a checkpointed row depends on (method version, settings, index, questions)."""
    return hashlib.sha1(json.dumps([METHOD_VERSION, *parts], sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- questions, plant, diagnosis

def dev_questions(root: Path = ROOT) -> list[Question]:
    """rag_dev questions in the file's order, the order the record checks use."""
    return [q for q in load_questions(root / RAG_REL) if q.split == SPLIT.removeprefix("rag_")]


def load_classes(root: Path = ROOT) -> list[str]:
    import csv

    with (root / "datasets" / "manifests" / "lab.csv").open(newline="", encoding="utf-8") as f:
        return sorted({r["label"] for r in csv.DictReader(f)})


def simulate_identification(q: Question, crops: set[str], classes: list[str]) -> tuple[str | None, Identification | None]:
    """(plant, simulated photo diagnosis) for a question. The diagnosis names the plant's class that the question
    mentions, as a correct identifier would; its alternatives are the plant's other classes. None when there is none."""
    text = " ".join([*(h["content"] for h in q.history if h.get("role") == "user"), q.question])
    crop = (q.crop or "").lower() or detect_crop(text, crops)
    if crop is None:
        return None, None
    own = [c for c in classes if crop_of_label(c, {crop}) == crop]
    named = [c for c in own if disease_of_label(c) and disease_of_label(c) in text.lower().replace("-", " ")]
    if not named:
        return crop, None
    others = [c for c in own if c != named[0] and disease_of_label(c)]
    alts = [Candidate(c, p) for c, p in zip(others, (0.06, 0.04), strict=False)]
    return crop, Identification("answered", Candidate(named[0], 0.9), alts)


# ---------------------------------------------------------------- gold matching and ranks

def single_page_sources(index_dir: Path) -> set[str]:
    pages: dict[str, int] = {}
    for r in index_rows(Path(index_dir) / "chunks.jsonl"):
        pages[r["source"]] = max(pages.get(r["source"], 0), r["page"])
    return {s for s, n in pages.items() if n <= 1}


def is_gold(p, q: Question, single: set[str]) -> bool:
    if q.gold_source is None or p.source != q.gold_source or p.page != q.gold_page:
        return False
    if p.source not in single:
        return True
    text = normalize(p.text)
    return any(normalize(t) in text for t in q.answer_terms)


def gold_rank(passages: list, q: Question, single: set[str]) -> int | None:
    for i, p in enumerate(passages, 1):
        if is_gold(p, q, single):
            return i
    return None


# ---------------------------------------------------------------- retrieval stages

def term_rank(passages: list, q: Question) -> int | None:
    """First passage, from any document, that holds one of the question's answer terms (a lenient cross-check of the
    strict gold-page rank: the index grew from 28 to 237 documents, so other documents may answer legitimately)."""
    terms = [normalize(t) for t in q.answer_terms]
    for i, p in enumerate(passages, 1):
        if any(t in normalize(p.text) for t in terms):
            return i
    return None


def retrieval_row(
    retriever, q: Question, spec: RetrievalSpec, crops: set[str], classes: list[str], single: set[str],
    diagnose: bool = False,
) -> dict:
    """Rank of the gold hit among the top RANK_K passages (None when absent or the question has no gold) and of
    the first passage holding an answer term, the best passage's score, and what was searched. The floor is off
    here, so ranking quality is measured on its own. With `diagnose`, the same search without the plant filter."""
    crop, ident = simulate_identification(q, crops, classes)
    query = build_query(q.question, ident, q.history, spec.query_mode)
    found = retriever.search(query, crop=crop, k=RANK_K, mode=spec.mode, rerank=spec.rerank)
    row = {
        "id": q.id, "rank": gold_rank(found, q, single) if q.gold_source else None,
        "term_rank": term_rank(found, q) if q.answer_terms else None,
        "top_score": found[0].score if found else 0.0, "crop": crop, "query": query, "diagnosed": ident is not None,
    }
    if diagnose:
        wide = retriever.search(query, crop=None, k=RANK_K, mode=spec.mode, rerank=spec.rerank)
        row["nocrop_top_score"] = wide[0].score if wide else 0.0
        row["nocrop_rank"] = gold_rank(wide, q, single) if q.gold_source else None
    return row


def run_retrieval(
    name: str, retriever, spec: RetrievalSpec, questions: list[Question], work: Path, root: Path = ROOT,
    diagnose: bool = False,
) -> list[dict]:
    """All dev questions for one system, resumable. `retriever` must be built for `spec`. The checkpoint file name
    carries a hash of the method version, settings, index build and questions, so a change starts fresh."""
    key = checkpoint_key(asdict(spec), diagnose, index_stamp(retriever._index_dir), split_info(SPLIT, root).hash)
    ck = Checkpoint(work / f"{name}-{key}.jsonl")
    crops, classes = retriever.crops, load_classes(root)
    single = single_page_sources(retriever._index_dir)
    for q in questions:
        if q.id not in ck.rows:
            ck.add(retrieval_row(retriever, q, spec, crops, classes, single, diagnose))
    return [ck.rows[q.id] for q in questions]


# Retrieval rule, version 2. Revised after the first pass: there, dense was kept over hybrid_rerank although
# hybrid_rerank had recall@1 0.600 vs 0.417, recall@5 0.833 vs 0.717 and MRR 0.712 vs 0.543, because the recall@5
# interval [0.0, 0.233] touched zero at n=60. A change is now kept when the paired interval of either recall@5 or
# MRR excludes zero (lower bound above zero) and the other metric's point estimate is not worse.
def better(comps: list[dict]) -> bool:
    by = {c["metric"]: c for c in comps}
    if set(by) != set(DECIDING_METRICS):
        raise ValueError(f"need comparisons on {DECIDING_METRICS}")
    for m, other in zip(DECIDING_METRICS, reversed(DECIDING_METRICS), strict=True):
        if by[m]["ci"][0] > 0 and by[other]["diff"] >= 0:
            return True
    return False


def gain(comps: list[dict]) -> float:
    """Ranking of kept candidates: the sum of their recall@5 and MRR gains."""
    return float(sum(c["diff"] for c in comps))


def _ranked(rows: list[dict], root: Path, field_name: str = "rank") -> tuple[list[str], list[str], dict[str, int | None]]:
    info = split_info(SPLIT, root)
    pos = subset_positions("answerable", SPLIT, info, root)
    return [info.ids[i] for i in pos], [info.groups[i] for i in pos], {r["id"]: r[field_name] for r in rows}


def pair_comparisons(rows_a: list[dict], rows_b: list[dict], root: Path = ROOT, names: tuple[str, str] = ("a", "b")) -> list[dict]:
    """a minus b on recall@5 and on MRR over the answerable questions, with paired intervals (see better())."""
    ids, groups, ma = _ranked(rows_a, root)
    _, _, mb = _ranked(rows_b, root)
    a, b = (Scored(groups=groups, ranks=[m[i] for i in ids]) for m in (ma, mb))
    return [{"a": names[0], "b": names[1], "metric": m, **recompute_comparison(a, b, m)} for m in DECIDING_METRICS]


def device_fields(info: dict) -> dict:
    """The device as a plain string (what MLflow's `device` parameter shows) and the details beside it."""
    return {"device": info["device"], "device_info": info}


def write_retrieval_record(
    stage: str, rows: dict[str, list[dict]], specs: dict[str, RetrievalSpec], base: str, device: dict,
    root: Path = ROOT, allow_dirty: bool = False, measurements: dict[str, list[float]] | None = None,
    extra_config: dict | None = None,
) -> tuple[Path, list[dict]]:
    """One E7 record on the answerable questions of rag_dev. Every system also appears with `_terms` appended, ranked
    by the first passage holding an answer term (a cross-check, not used to decide). Comparisons with the base
    are on recall@5 and MRR."""
    info = split_info(SPLIT, root)
    pos = subset_positions("answerable", SPLIT, info, root)
    ids, groups = [info.ids[i] for i in pos], [info.groups[i] for i in pos]
    scored: dict[str, Scored] = {}
    out_rows: dict[str, list[dict]] = {}
    for name, system_rows in rows.items():
        by = {r["id"]: r for r in system_rows}
        for tag, field_name in ((name, "rank"), (f"{name}_terms", "term_rank")):
            out_rows[tag] = [{"id": i, "rank": by[i][field_name]} for i in ids]
            scored[tag] = Scored(groups=groups, ranks=[by[i][field_name] for i in ids])
    metrics = {n: {m: recompute_metric(m, s) for m in RETRIEVAL_METRICS} for n, s in scored.items()}
    intervals = {n: {m: recompute_interval(m, s) for m in DECIDING_METRICS} for n, s in scored.items()}
    comparisons = []
    for n in rows:
        if n != base:
            for m in DECIDING_METRICS:
                comparisons.append({"a": n, "b": base, "metric": m, **recompute_comparison(scored[n], scored[base], m)})
    config = {
        "stage": stage, "method_version": METHOD_VERSION, "base": base, "rank_k": RANK_K, "gold_matching": GOLD_RULE,
        "crop_source": CROP_RULE, **device_fields(device), "labels_reviewed": False,
        "decision_rule": "keep when the paired interval of recall@5 or of MRR excludes zero and the other point "
                         "estimate is not worse (version 2, revised after the first pass)",
        **(extra_config or {}),
    }
    attempts = [{"system": n, "setting": asdict(specs[n])} for n in rows]
    models = {"embedder": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"}
    path = write_run("E7", SPLIT, config, models, metrics, intervals, attempts, out_rows, info.hash, root=root,
                     subset="answerable", comparisons=comparisons, allow_dirty=allow_dirty, measurements=measurements,
                     notes="Retrieval only, on the answerable questions; the score floor is off.")
    return path, comparisons


class TimedReranker:
    """Wraps a reranker and records the milliseconds of each call."""

    def __init__(self, inner) -> None:
        self.inner, self.ms = inner, []

    def score(self, query: str, texts: list[str]) -> list[float]:
        t = time.perf_counter()
        out = self.inner.score(query, texts)
        self.ms.append((time.perf_counter() - t) * 1000)
        return out


def time_search(retriever, timed: TimedReranker, spec: RetrievalSpec, questions: list[Question], root: Path = ROOT,
                warmup: int = 3) -> tuple[list[float], list[float]]:
    """(whole-search ms, reranker-call ms) per question, as served (k=SERVED_K, rerank on). The first `warmup`
    questions are searched once untimed, so model loading and kernel start-up are not counted."""
    crops, classes = retriever.crops, load_classes(root)
    search_ms: list[float] = []

    def one(q: Question) -> None:
        crop, ident = simulate_identification(q, crops, classes)
        query = build_query(q.question, ident, q.history, spec.query_mode)
        retriever.search(query, crop=crop, k=SERVED_K, mode=spec.mode, rerank=True)

    for q in questions[:warmup]:
        one(q)
    timed.ms.clear()
    for q in questions:
        t = time.perf_counter()
        one(q)
        search_ms.append((time.perf_counter() - t) * 1000)
    return search_ms, list(timed.ms)


# A rule added to the base prompt is kept when the thing it targets improves and answering answerable questions does
# not drop by more than this. There are too few graded items per version for an interval, so the rule is about
# direction only. Each addition is measured alone against the base; the kept ones are then combined and re-checked.
MAX_ANSWER_DROP = 0.05
ADDITIONS = (("faithful", "faithfulness"), ("abstain", "declines_when_unanswerable"), ("ask", "declines_when_ambiguous"))


def choose_prompt(values: dict[str, dict[str, float | None]]) -> tuple[str, list[dict], bool]:
    """(prompt version, decisions, whether the kept additions still need a combined run).
    `values[name]` holds the behaviour metrics and `faithfulness` for `base`, each addition, and `combined`."""
    base = values["base"]
    decisions, kept = [], []
    for add, target in ADDITIONS:
        if add not in values:
            continue
        gain = (values[add].get(target) or 0.0) - (base.get(target) or 0.0)
        drop = (base.get("answers_when_answerable") or 0.0) - (values[add].get("answers_when_answerable") or 0.0)
        ok = gain > 0 and drop <= MAX_ANSWER_DROP
        decisions.append({"addition": add, "target": target, "gain": gain, "answer_drop": drop, "kept": ok})
        if ok:
            kept.append((add, gain))
    names = [a for a, _ in kept]
    if len(names) <= 1:
        return "+".join(["base", *names]), decisions, False
    if "combined" not in values:
        return "+".join(["base", *names]), decisions, True
    comb = values["combined"]
    targets = dict(ADDITIONS)
    drop = (base.get("answers_when_answerable") or 0.0) - (comb.get("answers_when_answerable") or 0.0)
    holds = drop <= MAX_ANSWER_DROP and all((comb.get(targets[a]) or 0.0) > (base.get(targets[a]) or 0.0) for a in names)
    gains = {a: (comb.get(targets[a]) or 0.0) - (base.get(targets[a]) or 0.0) for a in names}
    decisions.append({"addition": "combined", "kept": holds, "answer_drop": drop, "target_gains": gains})
    best = max(kept, key=lambda x: x[1])[0]
    return ("+".join(["base", *names]) if holds else f"base+{best}"), decisions, False


# Floor rule, version 2. Revised after the first pass: the old rule balanced "keeps answerable" against "drops
# unanswerable" equally, which picked a dense floor of 0.80 that declined every unanswerable question but answered only
# 65% of the answerable ones. Now: among floors that answer at least 80% of answerable questions, take the highest
# declines_when_unanswerable (ties: more answers, then the lower floor); if none reaches 80%, take the floor with the
# highest mean of the two and say so.
def choose_floor(answerable_scores: list[float], unanswerable_scores: list[float]) -> dict:
    values = sorted(set(answerable_scores) | set(unanswerable_scores))
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)]
    candidates += [values[-1] + 1e-6] if values else []

    def rates(f: float) -> tuple[float, float]:
        ans = float(np.mean([s >= f for s in answerable_scores])) if answerable_scores else 0.0
        dec = float(np.mean([s < f for s in unanswerable_scores])) if unanswerable_scores else 0.0
        return ans, dec

    ok = [f for f in candidates if rates(f)[0] >= FLOOR_MIN_ANSWERS]
    if ok:
        best, rule = max(ok, key=lambda f: (rates(f)[1], rates(f)[0], -f)), f"highest declines among floors answering at least {FLOOR_MIN_ANSWERS:.0%}"
    else:
        best, rule = max(candidates, key=lambda f: (sum(rates(f)) / 2, -f)), f"no floor reached {FLOOR_MIN_ANSWERS:.0%} answers; highest mean of the two rates"
    ans, dec = rates(best)
    steps = np.unique(np.quantile(values, np.linspace(0, 1, 11))) if values else []
    return {"floor": round(float(best), 6), "answers_when_answerable": ans, "declines_when_unanswerable": dec,
            "rule_applied": rule, "curve": [{"floor": round(float(f), 4), "answers": rates(f)[0], "declines": rates(f)[1]} for f in steps]}


def floor_groups(questions: list[Question], rows: list[dict]) -> tuple[list[float], list[float]]:
    top = {r["id"]: r["top_score"] for r in rows}
    return ([top[q.id] for q in questions if q.expected == "answer"],
            [top[q.id] for q in questions if q.expected == "abstain"])


def floor_diagnosis(questions: list[Question], rows: list[dict], floor: float) -> dict:
    """Why answerable questions fall under the floor: which kinds, whether retrieval found the gold page anyway,
    whether the plant filter or a missing simulated diagnosis could be the cause."""
    by = {r["id"]: r for r in rows}
    under = [q for q in questions if q.expected == "answer" and by[q.id]["top_score"] < floor]
    out: dict[str, Any] = {"floor": floor, "answerable_under_floor": len(under), "of": sum(q.expected == "answer" for q in questions)}
    out["by_category"] = {c: sum(q.category == c for q in under) for c in sorted({q.category for q in under})}
    out["gold_in_top10_anyway"] = sum(by[q.id]["rank"] is not None for q in under)
    out["gold_missing_from_top10"] = sum(by[q.id]["rank"] is None for q in under)
    out["no_simulated_diagnosis"] = sum(not by[q.id]["diagnosed"] for q in under)
    out["plant_filter_applied"] = sum(by[q.id]["crop"] is not None for q in under)
    if all("nocrop_top_score" in by[q.id] for q in under):
        out["would_pass_floor_without_plant_filter"] = sum(by[q.id]["nocrop_top_score"] >= floor for q in under)
    return out


def write_floor_record(
    questions: list[Question], by_mode: dict[str, list[dict]], floors: dict[str, dict], default_floor: float,
    device: dict, root: Path = ROOT, allow_dirty: bool = False, diagnosis: dict | None = None, extra_config: dict | None = None,
) -> Path:
    """Answer-or-decline from the best passage's score alone: at the old 0.3 floor and at the chosen one, per mode."""
    info = split_info(SPLIT, root)
    metrics, out_rows = {}, {}
    for mode, rows in by_mode.items():
        top = {r["id"]: r["top_score"] for r in rows}
        key = mode.replace("+", "_")
        for tag, floor in ((f"floor_chosen_{key}", floors[mode]["floor"]), (f"floor_default_{key}", default_floor)):
            out_rows[tag] = [{"id": i, "answered": bool(top[i] >= floor)} for i in info.ids]
            s = Scored(groups=info.ids, answered=np.array([r["answered"] for r in out_rows[tag]]), expected=info.expected)
            metrics[tag] = {m: recompute_metric(m, s) for m in BEHAVIOUR_METRICS}
    config = {"stage": "floors", "method_version": METHOD_VERSION, "floors": floors, "default_floor": default_floor,
              **device_fields(device), "labels_reviewed": False, "floor_diagnosis": diagnosis or {},
              "decline_rule": "a question is declined when its best passage scores under the floor",
              "floor_rule": "among floors answering at least 80% of answerable questions, the highest declines_when_unanswerable; "
                            "else the highest mean of the two (version 2, revised after the first pass)",
              **(extra_config or {})}
    return write_run("E7", SPLIT, config, {"embedder": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"},
                     metrics, {n: {} for n in out_rows}, [{"mode": m, **{k: v for k, v in floors[m].items() if k != "curve"}} for m in floors],
                     out_rows, info.hash, root=root, allow_dirty=allow_dirty,
                     notes="Floors were picked on rag_dev itself, so these rates are optimistic.")


# ---------------------------------------------------------------- answers

class CachedRetriever:
    """Serves passages found earlier for the same query and plant, so the writer stage needs no torch."""

    def __init__(self, crops: set[str], found: dict[tuple[str, str | None], list[Passage]]) -> None:
        self.crops, self._found = crops, found

    def search(self, query: str, *, crop: str | None = None, **_: Any) -> list[Passage]:
        return self._found[(query, crop)]


class RecordingRetriever:
    """Passes searches through and keeps each one (query, plant, passages), so every search answer() makes can be replayed."""

    def __init__(self, inner) -> None:
        self.inner, self.crops, self.searches = inner, inner.crops, []

    def search(self, query: str, *, crop: str | None = None, **kw: Any) -> list[Passage]:
        found = self.inner.search(query, crop=crop, **kw)
        self.searches.append({"query": query, "crop": crop, "passages": passages_to_json(found)})
        return found


def passages_to_json(passages: list) -> list[dict]:
    return [asdict(p) for p in passages]


def retrieval_for_answers(retriever, cfg: AnswerConfig, q: Question, crops: set[str], classes: list[str]) -> dict:
    """What answer() will search for, and what it finds, so a later stage can replay it without torch."""
    crop, ident = simulate_identification(q, crops, classes)
    recorder = RecordingRetriever(retriever)
    found, query, searched = retrieve(q.question, identification=ident, history=bounded_history(q.history, cfg),
                                      retriever=recorder, cfg=cfg, crop=crop)  # the same code answer() runs
    return {"id": q.id, "query": query, "crop": searched, "passages": passages_to_json(found), "searches": recorder.searches}


def answer_retrieval_file(
    work: Path, tag: str, cfg: AnswerConfig, floors: dict, min_score: float, index_dir: Path, root: Path = ROOT,
) -> Path:
    """Where the passages for `tag` (with floors, or without) are cached; the name carries a hash of everything
    they depend on, so a changed method, index or question set never reuses old rows."""
    key = checkpoint_key(ANSWER_VERSION, asdict(replace_sources(cfg)), floors, min_score, index_stamp(index_dir), split_info(SPLIT, root).hash)
    return Path(work) / f"{tag}-{key}.jsonl"


def run_answer_retrieval(
    retriever, cfg: AnswerConfig, questions: list[Question], work: Path, root: Path = ROOT, tag: str = "answers",
) -> dict:
    """Passages each question gets under `cfg` and the retriever's floors. `tag` names the cache (with floors or without)."""
    ck = Checkpoint(answer_retrieval_file(work, tag, cfg, retriever.config.floors, retriever.config.min_score,
                                          retriever._index_dir, root))
    classes = load_classes(root)
    for q in questions:
        if q.id not in ck.rows:
            ck.add(retrieval_for_answers(retriever, cfg, q, retriever.crops, classes))
    return ck.rows


def replace_sources(cfg: AnswerConfig) -> AnswerConfig:
    from dataclasses import replace

    return replace(cfg, sources=None)


def cached_retriever(crops: set[str], rows: dict[str, dict]) -> CachedRetriever:
    return CachedRetriever(crops, {(s["query"], s["crop"]): [Passage(**p) for p in s["passages"]]
                                   for r in rows.values() for s in r["searches"]})


class TimedLLM:
    """Counts the model calls and the seconds each took, so a record can show the writer really ran."""

    def __init__(self, llm) -> None:
        self.llm, self.model, self.seconds = llm, llm.model, []

    def complete(self, messages: list[dict], *, max_tokens: int, timeout_s: float):
        t = time.perf_counter()
        try:
            return self.llm.complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)
        finally:
            self.seconds.append(time.perf_counter() - t)


def generation_row(q: Question, ans: Answer, system: str, single: set[str], calls: int = 0, seconds: float = 0.0) -> dict:
    cited_gold = None
    if q.gold_source:
        cited_gold = any(c.source == q.gold_source and c.page == q.gold_page for c in ans.citations)
    return {
        "id": q.id, "system": system, "answered": bool(ans.grounded), "state": ans.state, "asks_for": ans.asks_for,
        "cited_gold": cited_gold, "gold_rank": gold_rank(ans.passages, q, single) if q.gold_source else None,
        "query": ans.query, "text": ans.text, "passages": passages_to_json(ans.passages),
        "citations": [asdict(c) for c in ans.citations], "llm_calls": calls, "llm_seconds": round(seconds, 3),
    }


FAILED_STATES = ("unavailable", "retrieval_failed")


def run_generation(
    system: str, questions: list[Question], retriever, llm, cfg: AnswerConfig, work: Path, single: set[str],
    root: Path = ROOT, key: str = "",
) -> tuple[list[dict], list[str]]:
    """(rows in question order for finished questions, ids still missing). An outage is never recorded as an answer."""
    ck = Checkpoint(work / f"generate-{system}{'-' + key if key else ''}.jsonl")
    classes = load_classes(root)
    timed = TimedLLM(llm)
    missing = []
    for q in questions:
        if q.id in ck.rows:
            continue
        crop, ident = simulate_identification(q, retriever.crops, classes)
        before = len(timed.seconds)
        ans = answer(q.question, identification=ident, history=q.history, retriever=retriever, llm=timed, config=cfg, crop=crop)
        if ans.state in FAILED_STATES:
            missing.append(q.id)
            continue
        used = timed.seconds[before:]
        ck.add(generation_row(q, ans, system, single, len(used), sum(used)))
    return [ck.rows[q.id] for q in questions if q.id in ck.rows], missing


# ---------------------------------------------------------------- judging

def row_to_answer(row: dict) -> Answer:
    return Answer(row["text"], [], row["answered"], row["asks_for"], row["state"],
                  [Passage(**p) for p in row["passages"]], row["query"])


def run_judging(
    system: str, rows: list[dict], questions: dict[str, Question], judge_llm, work: Path, key: str = "",
) -> list[dict]:
    """Per-claim judgments for the answers of one system, resumable per answer. Answers with no claims are skipped."""
    ck = Checkpoint(work / f"judge-{system}{'-' + key if key else ''}.jsonl")
    for row in rows:
        if row["id"] in ck.rows:
            continue
        judged = judge_answer(judge_llm, row["id"], system, questions[row["id"]].question, row_to_answer(row))
        ck.add(judged or {"id": row["id"], "claims": []})
    return [ck.rows[r["id"]] for r in rows if ck.rows[r["id"]]["claims"]]


def write_judgments(stem: Path, judged: list[dict]) -> tuple[Path, Path]:
    """The full file (with verdicts) and the blind file (without) that another grader is given. The names carry a hash
    of the content, so a rerun with other content makes new files and never changes one that is already committed;
    the same content gives the same names and is not rewritten."""
    full_text = "".join(json.dumps(r) + "\n" for r in judged)
    tag = hashlib.sha1(full_text.encode()).hexdigest()[:8]
    full, blind = Path(f"{stem}-{tag}.judgments.jsonl"), Path(f"{stem}-{tag}.blind.jsonl")
    if not full.is_file():
        atomic_write_text(full, full_text)
    if not blind.is_file():
        atomic_write_text(blind, "".join(json.dumps(blind_view(r)) + "\n" for r in judged))
    return full, blind


def write_generation_record(
    stage: str, split_subset: str | None, rows: dict[str, list[dict]], judged: dict[str, list[dict]],
    models: dict[str, str], config: dict, root: Path = ROOT, allow_dirty: bool = False,
) -> Path:
    info = split_info(SPLIT, root)
    pos = subset_positions(split_subset, SPLIT, info, root) if split_subset else list(range(len(info.ids)))
    ids = [info.ids[i] for i in pos]
    expected = [info.expected[i] for i in pos]
    metrics, out_rows = {}, {}
    for name, system_rows in rows.items():
        by = {r["id"]: r for r in system_rows}
        out_rows[name] = [{"id": i, "answered": by[i]["answered"], "state": by[i]["state"]} for i in ids]
        s = Scored(groups=ids, answered=np.array([r["answered"] for r in out_rows[name]]), expected=expected)
        metrics[name] = {m: recompute_metric(m, s) for m in BEHAVIOUR_METRICS}
    samples = {f"faithfulness_{n}": [bool(c["verdict"]) for r in judged.get(n, []) for c in r["claims"]] for n in rows}
    samples = {k: v for k, v in samples.items() if v}
    usage = {n: {"rows": len(r), "model_calls": sum(x.get("llm_calls", 0) for x in r),
                 "rows_without_model_call": sum(x.get("llm_calls", 0) == 0 for x in r),
                 "model_seconds": round(sum(x.get("llm_seconds", 0.0) for x in r), 1)} for n, r in rows.items()}
    config = {**config, "stage": stage, "method_version": METHOD_VERSION, "labels_reviewed": False,
              "gold_matching": GOLD_RULE, "crop_source": CROP_RULE, "writer_usage": usage}
    return write_run("E7", SPLIT, config, models, metrics, {n: {} for n in rows},
                     [{"system": n, "answers_graded": len(judged.get(n, []))} for n in rows], out_rows, info.hash,
                     root=root, subset=split_subset, judged=samples or None, allow_dirty=allow_dirty,
                     notes="Faithfulness is the share of claims the judge model found supported by the cited passages.")


SUPERSEDED_REL = Path("results/e7/superseded.json")


def superseded_records(root: Path = ROOT) -> dict[str, dict]:
    """Records that a later one replaces (`superseded`: id -> {by, reason}), and records whose answers were produced by
    code that has since changed and need a rerun (`needs_rerun`: id -> reason). Old records are never edited; reports
    read this file to know which to leave out."""
    path = root / SUPERSEDED_REL
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"superseded": {}, "needs_rerun": {}}


@dataclass
class Chosen:
    """results/e7/v2/chosen.json: the settings each stage settled on, and every keep-or-drop decision."""

    retrieval: dict = field(default_factory=lambda: asdict(RetrievalSpec()))
    floors: dict = field(default_factory=dict)
    prompt: str = "base"
    combined: str = ""  # the prompt the combined run uses, fixed when the single additions are decided
    decisions: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Chosen:
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(asdict(self), indent=1) + "\n")


# ---------------------------------------------------------------- intervals for the prompt stage

_GOLD_FOR = {"answers_when_answerable": "answer", "declines_when_unanswerable": "abstain",
             "declines_when_crop_mismatch": "crop_mismatch", "declines_when_ambiguous": "clarify"}


def behaviour_gain(metric: str, questions: list[Question], rows_other: list[dict], rows_base: list[dict],
                   n: int = 10000, seed: int = 13) -> dict:
    """other minus base for a behaviour rate, paired over the questions it is defined on, with McNemar's exact p."""
    gold = _GOLD_FOR[metric]
    ids = [q.id for q in questions if q.expected == gold]
    ao, ab = ({r["id"]: r["answered"] for r in rows} for rows in (rows_other, rows_base))
    flip = (lambda x: x) if metric == "answers_when_answerable" else (lambda x: not x)
    a = np.array([flip(ao[i]) for i in ids], dtype=float)
    b = np.array([flip(ab[i]) for i in ids], dtype=float)
    diff, lo, hi = paired_bootstrap_diff(a, b, ids, n, seed)
    return {"metric": metric, "n": len(ids), "diff": diff, "ci": [lo, hi], "mcnemar_p": mcnemar_exact(a > 0.5, b > 0.5)}


def faithfulness_gain(judged_other: list[dict], judged_base: list[dict], n: int = 10000, seed: int = 13) -> dict:
    """other minus base share of claims judged supported. Whole answers are resampled, each system on its own, since
    the two systems' answers are different texts (not paired)."""
    def counts(judged):
        return np.array([[sum(bool(c["verdict"]) for c in r["claims"]), len(r["claims"])] for r in judged], dtype=float)

    a, b = counts(judged_other), counts(judged_base)
    rng = np.random.default_rng(seed)
    ia, ib = rng.integers(0, len(a), (n, len(a))), rng.integers(0, len(b), (n, len(b)))
    boot = a[ia, 0].sum(1) / a[ia, 1].sum(1) - b[ib, 0].sum(1) / b[ib, 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"metric": "faithfulness", "answers": [len(a), len(b)], "diff": float(a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()),
            "ci": [float(lo), float(hi)]}


def prompt_intervals(questions: list[Question], rows: dict[str, list[dict]], graded: dict[str, list[dict]]) -> dict[str, dict]:
    """For each addition: its gain over the base on its target, and on answers_when_answerable, with intervals.
    An interval that includes zero means the difference is not distinguishable from noise."""
    out = {}
    for add, target in ADDITIONS:
        if add not in rows:
            continue
        gain_target = (faithfulness_gain(graded[add], graded["base"]) if target == "faithfulness"
                       else behaviour_gain(target, questions, rows[add], rows["base"]))
        out[add] = {"target": gain_target, "answers": behaviour_gain("answers_when_answerable", questions, rows[add], rows["base"]),
                    "distinguishable": bool(gain_target["ci"][0] > 0)}
    return out


def category_breakdown(rows: list[dict], questions: list[Question]) -> dict[str, dict]:
    """recall@5 and MRR per question category among the answerable ones, so a gain that only helps follow-ups
    (or hurts them) is visible instead of being averaged away. Reported next to the pooled numbers, not used to decide."""
    by = {r["id"]: r["rank"] for r in rows}
    out = {}
    for cat in sorted({q.category for q in questions if q.expected == "answer"}):
        ranks = [by[q.id] for q in questions if q.category == cat and q.expected == "answer"]
        out[cat] = {"n": len(ranks), "recall@5": float(np.mean([r is not None and r <= 5 for r in ranks])),
                    "mrr": float(np.mean([0.0 if r is None else 1.0 / r for r in ranks]))}
    return out
