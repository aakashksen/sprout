"""E7: which retrieval setup and prompt give faithful, cited answers, on rag_dev only.

The stage logic lives here so tests can run it with fakes; tools/run_e7.py wires in the GPU, Ollama and the
command line. Every stage checkpoints per question, so a crash resumes; a partial run is never a result.
rag_test is never read here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sprout.atomic import atomic_write_text
from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval, recompute_metric
from sprout.eval.records import RAG_REL, ROOT, split_info, subset_positions, write_run
from sprout.rag.answer import GENERAL_CROP, Answer, AnswerConfig, answer, bounded_history
from sprout.rag.evalset import Question, index_rows, load_questions, normalize
from sprout.rag.judge import blind_view, judge_answer
from sprout.rag.query import build_query, crop_of_label, disease_of_label
from sprout.rag.retrieve import Passage
from sprout.serve.types import Candidate, Identification

SPLIT = "rag_dev"
RANK_K = 10  # gold is looked for in the top 10 passages
RETRIEVAL_METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")
DECIDING_METRIC = "recall@5"
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


# ---------------------------------------------------------------- questions, plant, diagnosis

def dev_questions(root: Path = ROOT) -> list[Question]:
    """rag_dev questions in the file's order, the order the record checks use."""
    return [q for q in load_questions(root / RAG_REL) if q.split == SPLIT.removeprefix("rag_")]


def load_classes(root: Path = ROOT) -> list[str]:
    import csv

    with (root / "datasets" / "manifests" / "lab.csv").open(newline="", encoding="utf-8") as f:
        return sorted({r["label"] for r in csv.DictReader(f)})


def _forms(crop: str) -> str:
    plural = crop[:-1] + "ies" if crop.endswith("y") else crop + "(?:s|es)?"
    return rf"\b(?:{re.escape(crop)}(?:s|es)?|{plural})\b"


def detect_crop(text: str, crops: set[str]) -> str | None:
    """The plant named first in the text, among crops that have documents."""
    best = None
    for crop in crops:
        m = re.search(_forms(crop), text.lower())
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), crop)
    return best[1] if best else None


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

def retrieval_row(retriever, q: Question, spec: RetrievalSpec, crops: set[str], classes: list[str], single: set[str]) -> dict:
    """Rank of the gold hit among the top RANK_K passages (None when absent or the question has no gold), and the
    score of the best passage. The floor is off here, so ranking quality is measured on its own."""
    crop, ident = simulate_identification(q, crops, classes)
    query = build_query(q.question, ident, q.history, spec.query_mode)
    found = retriever.search(query, crop=crop, k=RANK_K, mode=spec.mode, rerank=spec.rerank)
    return {"id": q.id, "rank": gold_rank(found, q, single) if q.gold_source else None,
            "top_score": found[0].score if found else 0.0}


def run_retrieval(name: str, retriever, spec: RetrievalSpec, questions: list[Question], work: Path, root: Path = ROOT) -> list[dict]:
    """All dev questions for one system, resumable. `retriever` must be built for `spec`."""
    ck = Checkpoint(work / f"{name}.jsonl")
    crops, classes = retriever.crops, load_classes(root)
    single = single_page_sources(retriever._index_dir)
    for q in questions:
        if q.id not in ck.rows:
            ck.add(retrieval_row(retriever, q, spec, crops, classes, single))
    return [ck.rows[q.id] for q in questions]


def better(comp: dict) -> bool:
    """A candidate is kept only when the paired interval of its recall@5 gain over the base excludes zero."""
    return bool(comp["ci"][0] > 0)


def write_retrieval_record(
    stage: str, rows: dict[str, list[dict]], specs: dict[str, RetrievalSpec], base: str, device: dict,
    root: Path = ROOT, allow_dirty: bool = False,
) -> tuple[Path, list[dict]]:
    """One E7 record on the answerable questions of rag_dev, and the paired comparisons of every system with base."""
    info = split_info(SPLIT, root)
    pos = subset_positions("answerable", SPLIT, info, root)
    ids, groups = [info.ids[i] for i in pos], [info.groups[i] for i in pos]
    scored: dict[str, Scored] = {}
    out_rows: dict[str, list[dict]] = {}
    for name, system_rows in rows.items():
        by = {r["id"]: r for r in system_rows}
        out_rows[name] = [{"id": i, "rank": by[i]["rank"]} for i in ids]
        scored[name] = Scored(groups=groups, ranks=[by[i]["rank"] for i in ids])
    metrics = {n: {m: recompute_metric(m, s) for m in RETRIEVAL_METRICS} for n, s in scored.items()}
    intervals = {n: {m: recompute_interval(m, s) for m in ("recall@5", "mrr")} for n, s in scored.items()}
    comparisons = []
    for n in scored:
        if n != base:
            got = recompute_comparison(scored[n], scored[base], DECIDING_METRIC)
            comparisons.append({"a": n, "b": base, "metric": DECIDING_METRIC, **got})
    config = {
        "stage": stage, "base": base, "rank_k": RANK_K, "gold_matching": GOLD_RULE, "crop_source": CROP_RULE,
        "device": device, "labels_reviewed": False,
    }
    attempts = [{"system": n, "setting": asdict(specs[n])} for n in rows]
    models = {"embedder": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"}
    path = write_run("E7", SPLIT, config, models, metrics, intervals, attempts, out_rows, info.hash, root=root,
                     subset="answerable", comparisons=comparisons, allow_dirty=allow_dirty,
                     notes="Retrieval only, on the answerable questions; the score floor is off.")
    return path, comparisons


def pair_comparison(rows_a: list[dict], rows_b: list[dict], root: Path = ROOT) -> dict:
    """recall@5 of a minus b on the answerable questions, with the paired interval (see better())."""
    info = split_info(SPLIT, root)
    pos = subset_positions("answerable", SPLIT, info, root)
    ids, groups = [info.ids[i] for i in pos], [info.groups[i] for i in pos]
    sa, sb = ({r["id"]: r["rank"] for r in rows} for rows in (rows_a, rows_b))
    a, b = (Scored(groups=groups, ranks=[m[i] for i in ids]) for m in (sa, sb))
    return recompute_comparison(a, b, DECIDING_METRIC)


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
    decisions.append({"addition": "combined", "kept": holds, "answer_drop": drop})
    best = max(kept, key=lambda x: x[1])[0]
    return ("+".join(["base", *names]) if holds else f"base+{best}"), decisions, False


def choose_floor(kept_scores: list[float], dropped_scores: list[float]) -> dict:
    """The floor that best keeps the answerable questions' best passage and drops the unanswerable ones' (balanced
    over the two groups); halfway between the two neighbouring scores. Picked on rag_dev, so slightly optimistic."""
    values = sorted(set(kept_scores) | set(dropped_scores))
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)] + ([values[-1] + 1e-6] if values else [])

    def balanced(f: float) -> float:
        keep = np.mean([s >= f for s in kept_scores]) if kept_scores else 0.0
        drop = np.mean([s < f for s in dropped_scores]) if dropped_scores else 0.0
        return float((keep + drop) / 2)

    best = max(candidates, key=lambda f: (balanced(f), -f))
    return {"floor": round(float(best), 6), "balanced": balanced(best),
            "answerable_kept": float(np.mean([s >= best for s in kept_scores])) if kept_scores else None,
            "unanswerable_dropped": float(np.mean([s < best for s in dropped_scores])) if dropped_scores else None}


def floor_groups(questions: list[Question], rows: list[dict]) -> tuple[list[float], list[float]]:
    top = {r["id"]: r["top_score"] for r in rows}
    kept = [top[q.id] for q in questions if q.expected == "answer"]
    dropped = [top[q.id] for q in questions if q.expected in ("abstain", "crop_mismatch")]
    return kept, dropped


def write_floor_record(
    questions: list[Question], by_mode: dict[str, list[dict]], floors: dict[str, dict], default_floor: float,
    device: dict, root: Path = ROOT, allow_dirty: bool = False,
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
    config = {"stage": "floors", "floors": floors, "default_floor": default_floor, "device": device,
              "labels_reviewed": False, "decline_rule": "a question is declined when its best passage scores under the floor"}
    return write_run("E7", SPLIT, config, {"embedder": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"},
                     metrics, {n: {} for n in out_rows}, [{"mode": m, **floors[m]} for m in floors], out_rows, info.hash,
                     root=root, allow_dirty=allow_dirty,
                     notes="Floors were picked on rag_dev itself, so these rates are optimistic.")


# ---------------------------------------------------------------- answers

class CachedRetriever:
    """Serves passages found earlier for the same query and plant, so the writer stage needs no torch."""

    def __init__(self, crops: set[str], found: dict[tuple[str, str | None], list[Passage]]) -> None:
        self.crops, self._found = crops, found

    def search(self, query: str, *, crop: str | None = None, **_: Any) -> list[Passage]:
        return self._found[(query, crop)]


def passages_to_json(passages: list) -> list[dict]:
    return [asdict(p) for p in passages]


def retrieval_for_answers(retriever, cfg: AnswerConfig, q: Question, crops: set[str], classes: list[str]) -> dict:
    """What answer() will search for, and what it finds, so a later stage can replay it without torch."""
    crop, ident = simulate_identification(q, crops, classes)
    searched = None if crop is None else (crop if crop in crops else GENERAL_CROP)
    query = build_query(q.question, ident, bounded_history(q.history, cfg), cfg.query_mode)
    found = retriever.search(query, crop=searched, k=cfg.k, mode=cfg.mode, rerank=cfg.rerank)
    return {"id": q.id, "query": query, "crop": searched, "passages": passages_to_json(found)}


def run_answer_retrieval(retriever, cfg: AnswerConfig, questions: list[Question], work: Path, root: Path = ROOT) -> dict:
    ck = Checkpoint(work / "answer-retrieval.jsonl")
    classes = load_classes(root)
    for q in questions:
        if q.id not in ck.rows:
            ck.add(retrieval_for_answers(retriever, cfg, q, retriever.crops, classes))
    return ck.rows


def cached_retriever(crops: set[str], rows: dict[str, dict]) -> CachedRetriever:
    return CachedRetriever(crops, {(r["query"], r["crop"]): [Passage(**p) for p in r["passages"]] for r in rows.values()})


def generation_row(q: Question, ans: Answer, system: str, single: set[str]) -> dict:
    cited_gold = None
    if q.gold_source:
        cited_gold = any(c.source == q.gold_source and c.page == q.gold_page for c in ans.citations)
    return {
        "id": q.id, "system": system, "answered": bool(ans.grounded), "state": ans.state, "asks_for": ans.asks_for,
        "cited_gold": cited_gold, "gold_rank": gold_rank(ans.passages, q, single) if q.gold_source else None,
        "query": ans.query, "text": ans.text, "passages": passages_to_json(ans.passages),
        "citations": [asdict(c) for c in ans.citations],
    }


FAILED_STATES = ("unavailable", "retrieval_failed")


def run_generation(
    system: str, questions: list[Question], retriever, llm, cfg: AnswerConfig, work: Path, single: set[str],
    root: Path = ROOT,
) -> tuple[list[dict], list[str]]:
    """(rows in question order for finished questions, ids still missing). An outage is never recorded as an answer."""
    ck = Checkpoint(work / f"generate-{system}.jsonl")
    classes = load_classes(root)
    missing = []
    for q in questions:
        if q.id in ck.rows:
            continue
        crop, ident = simulate_identification(q, retriever.crops, classes)
        ans = answer(q.question, identification=ident, history=q.history, retriever=retriever, llm=llm, config=cfg, crop=crop)
        if ans.state in FAILED_STATES:
            missing.append(q.id)
            continue
        ck.add(generation_row(q, ans, system, single))
    return [ck.rows[q.id] for q in questions if q.id in ck.rows], missing


# ---------------------------------------------------------------- judging

def row_to_answer(row: dict) -> Answer:
    return Answer(row["text"], [], row["answered"], row["asks_for"], row["state"],
                  [Passage(**p) for p in row["passages"]], row["query"])


def run_judging(system: str, rows: list[dict], questions: dict[str, Question], judge_llm, work: Path) -> list[dict]:
    """Per-claim judgments for the answers of one system, resumable per answer. Answers with no claims are skipped."""
    ck = Checkpoint(work / f"judge-{system}.jsonl")
    for row in rows:
        if row["id"] in ck.rows:
            continue
        judged = judge_answer(judge_llm, row["id"], system, questions[row["id"]].question, row_to_answer(row))
        ck.add(judged or {"id": row["id"], "claims": []})
    return [ck.rows[r["id"]] for r in rows if ck.rows[r["id"]]["claims"]]


def write_judgments(stem: Path, judged: list[dict]) -> tuple[Path, Path]:
    """The full file (with verdicts) and the blind file (without) that another grader is given."""
    full, blind = Path(f"{stem}.judgments.jsonl"), Path(f"{stem}.blind.jsonl")
    atomic_write_text(full, "".join(json.dumps(r) + "\n" for r in judged))
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
    config = {**config, "stage": stage, "labels_reviewed": False, "gold_matching": GOLD_RULE, "crop_source": CROP_RULE}
    return write_run("E7", SPLIT, config, models, metrics, {n: {} for n in rows},
                     [{"system": n, "answers_graded": len(judged.get(n, []))} for n in rows], out_rows, info.hash,
                     root=root, subset=split_subset, judged=samples or None, allow_dirty=allow_dirty,
                     notes="Faithfulness is the share of claims the judge model found supported by the cited passages.")


@dataclass
class Chosen:
    """results/e7/chosen.json: the settings each stage settled on, and every keep-or-drop decision."""

    retrieval: dict = field(default_factory=lambda: asdict(RetrievalSpec()))
    floors: dict = field(default_factory=dict)
    prompt: str = "v1"
    decisions: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Chosen:
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else cls()

    def save(self, path: Path) -> None:
        atomic_write_text(path, json.dumps(asdict(self), indent=1) + "\n")
