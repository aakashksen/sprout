"""Experiment E7 on rag_dev, method version 2, one resumable stage at a time. Run from the worktree:

    set PYTHONPATH=src
    .venv\\Scripts\\python tools/run_e7.py <stage> [--variants ...]

Version 2 starts fresh in results/e7/work/v2 and results/e7/v2; the first pass's records, chosen.json and drafts
stay untouched as evidence. Commit between stages (a record needs a clean tree).

Stages, in order:
    indexes          build knowledge/index-structured, index-header, index-structured-header (GPU embedding)
    baseline         dense vs hybrid vs hybrid + rerank on the main index    -> record, results/e7/v2/chosen.json
    latency          rerank cost: CUDA torch, CPU torch and CPU ONNX (serving's path), for T7.4; run with the GPU idle
    chunking         structure-aware chunks with whole tables, against the base chosen above
    header           context header on each chunk, against the base
    crops            shared-crop pages, against the base
    queries          question + conversation / top-3 diseases / both, against the base
    floors           score floor per mode on the final retrieval setup, with a diagnosis of answerable questions under it
    retrieve-answers cache each dev question's passages, with the floors and with none (GPU)
    generate         write answers with the local writer (Ollama); variants base faithful abstain ask combined served
    judge            judge every claim with the local judge, a different model (Ollama)
    prompts-record   one record for the prompt variants, intervals, judgments files, prompt choice
    claude           the chosen prompt on Claude (Sonnet) over the first 100 dev questions, judged locally, one record
    status           what is done

A stage that uses the GPU takes `taskctl gpu-acquire T6.4` from the main folder just before and releases it right
after, even on failure. A stage that dies leaves checkpoints in results/e7/work/v2; run it again. rag_test is never read.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from build_index_gpu import build as build_index_gpu  # noqa: E402
from build_index_gpu import main_folder, taskctl  # noqa: E402

from sprout.eval.records import is_dirty  # noqa: E402
from sprout.experiments import e7_rag as e7  # noqa: E402
from sprout.llm.client import ClaudeCodeClient, LLMError, OllamaClient  # noqa: E402
from sprout.rag.answer import AnswerConfig, load_sources  # noqa: E402
from sprout.rag.judge import rubric  # noqa: E402
from sprout.rag.retrieve import RetrievalConfig, Retriever  # noqa: E402

TASK = "T6.4"
WRITER = "qwen3.5:4b-q4_K_M"
JUDGE = "gemma4:e4b-it-qat"
WORK = ROOT / "results" / "e7" / "work" / "v2"
CHOSEN = ROOT / "results" / "e7" / "v2" / "chosen.json"
DRAFTS = ROOT / "results" / "e7" / "v2" / "decisions"
INDEXES = {  # name -> (chunker, header)
    "index-structured": ("structured", False),
    "index-header": ("plain", True),
    "index-structured-header": ("structured", True),
}
BASELINES = {
    "dense": e7.RetrievalSpec(mode="dense", rerank=False),
    "hybrid": e7.RetrievalSpec(mode="hybrid", rerank=False),
    "hybrid_rerank": e7.RetrievalSpec(mode="hybrid", rerank=True),
}
FLOOR_MODES = {"dense": ("dense", False), "hybrid": ("hybrid", False), "hybrid+rerank": ("hybrid", True)}
PROMPT_VARIANTS = ("base", "faithful", "abstain", "ask")
ALL_VARIANTS = (*PROMPT_VARIANTS, "combined", "served")
EXIT_GPU_BUSY = 75
EXIT_INCOMPLETE = 2
EXIT_TOO_FEW_GRADED = 3


def knowledge() -> Path:
    return main_folder() / "knowledge"


@contextlib.contextmanager
def gpu():
    """Hold the GPU share for one stage; released even when the stage fails."""
    taskctl("gpu-acquire", TASK)
    try:
        yield
    finally:
        taskctl("gpu-release", TASK)


class Models:
    """The embedder and reranker, on CUDA through torch, built on first use."""

    def __init__(self) -> None:
        self._embedder = self._reranker = None
        self.config = RetrievalConfig()

    @property
    def embedder(self):
        if self._embedder is None:
            from sprout.rag.gpu_models import TorchEmbedder

            self._embedder = TorchEmbedder(self.config.embedding_model, device="cuda")
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sprout.rag.gpu_models import TorchReranker

            self._reranker = TorchReranker(device="cuda")
        return self._reranker

    def info(self) -> dict:
        out: dict = {}
        for m in (self._embedder, self._reranker):
            if m is not None:
                out.update(m.info)
        if not out:
            raise RuntimeError("no model was loaded, so no device is recorded")
        if out["device"] != "cuda":
            raise RuntimeError(f"expected CUDA, got {out['device']}")
        import torch

        out["peak_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        return out


def make_retriever(spec: e7.RetrievalSpec, models: Models, floors: dict | None = None, reranker=None) -> Retriever:
    cfg = RetrievalConfig(min_score=0.0, crops_from_title=spec.crops_from_title, floors=floors or {})
    return Retriever(knowledge() / spec.index, config=cfg, embedder=models.embedder,
                     reranker=reranker or (models.reranker if spec.rerank else None))


def require_clean() -> None:
    if is_dirty(ROOT):
        sys.exit("The working tree has uncommitted changes, and a record needs a clean tree. Commit first, e.g.\n"
                 "  git add -A results docs && git commit -m \"E7: stage results\"")


def finish(stage: str, run_id: str, draft: str) -> None:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / f"{stage}.md").write_text(draft, encoding="utf-8")
    print(f"\nrecord: results/runs/{run_id}.json\ndraft DECISIONS entry (check it, then add to docs/DECISIONS.md): "
          f"results/e7/v2/decisions/{stage}.md\n\n{draft}\n"
          f"\nNext: git add -A results docs && git commit -m \"E7 {stage}: record and decision\"")


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def ci_text(c: dict) -> str:
    return f"{c['diff']:+.3f} (95% CI {c['ci'][0]:+.3f} to {c['ci'][1]:+.3f})"


# ------------------------------------------------------------------ retrieval stages

def cmd_indexes(_args) -> None:
    for name, (chunker, header) in INDEXES.items():
        index = knowledge() / name
        if (index / "build-info.json").is_file():
            print(f"{name}: already built")
            continue
        info = build_index_gpu(index, task=TASK, manifest=ROOT / "knowledge" / "manifest.csv",
                               raw=ROOT / "knowledge" / "raw", chunker=chunker, header=header)
        print(f"{name}: {info['chunks']} chunks, {info['failures']} failed documents, device {info['device']}")


def retrieval_draft(stage, record, comps, winner, specs, device, questions, rows, sequential=None) -> str:
    kept_text = "base kept (no change met the rule)" if winner == "base" else f"{winner} kept"
    lines = [f"### T6.4 - Retrieval, {stage}: {kept_text} (method v2, {date.today()})",
             f"- **Chose:** {winner} = {asdict(specs[winner])}",
             "- **Considered:** " + "; ".join(f"{n} = {asdict(s)}" for n, s in specs.items() if n != winner),
             "- **Rule:** keep a change when the paired interval of recall@5 or of MRR excludes zero and the other point estimate "
             "is not worse. Revised after the first pass, whose numbers showed the old recall@5-only rule dropping a clear gain.",
             f"- **Evidence:** record `{record['run_id']}`, n={record['n']} answerable rag_dev questions. "
             "recall@1 / @3 / @5 / @10 / MRR per system (the `_terms` rows count any document holding an answer term):"]
    for n, m in record["metrics"].items():
        lines.append(f"  - {n}: " + " / ".join(fmt(m[k]) for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")))
    for c in [*comps, *(sequential or [])]:
        lines.append(f"  - {c['metric']} {c['a']} minus {c['b']}: {ci_text(c)}, McNemar p={c['mcnemar_p']:.3f}")
    for n, r in rows.items():
        lines.append(f"  - by category, {n}: " + "; ".join(
            f"{cat} n={v['n']} recall@5 {fmt(v['recall@5'])} MRR {fmt(v['mrr'])}" for cat, v in e7.category_breakdown(r, questions).items()))
    lines += [f"- **Device:** {device}",
              "- **Would change if:** a larger or reviewed question set shows a different gain."]
    return "\n".join(lines)


def retrieval_stage(stage: str, candidates: dict[str, e7.RetrievalSpec], extra: dict | None = None) -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    questions = e7.dev_questions(ROOT)
    base = e7.RetrievalSpec.from_json(chosen.retrieval)
    specs = {"base": base, **candidates}
    rows: dict[str, list[dict]] = {}
    with gpu():
        models = Models()
        for name, spec in specs.items():
            rows[name] = e7.run_retrieval(f"{stage}-{name}", make_retriever(spec, models), spec, questions, WORK / stage, ROOT)
        device = models.info()
    path, comps = e7.write_retrieval_record(stage, rows, specs, "base", device, ROOT, extra_config=extra)
    record = json.loads(path.read_text(encoding="utf-8"))
    kept = [n for n in candidates if e7.better([c for c in comps if c["a"] == n])]
    winner = max(kept, key=lambda n: e7.gain([c for c in comps if c["a"] == n])) if kept else "base"
    chosen.retrieval = asdict(specs[winner])
    chosen.decisions.append({"stage": stage, "kept": winner, "record": path.stem, "comparisons": comps, "device": device})
    chosen.save(CHOSEN)
    finish(stage, path.stem, retrieval_draft(stage, record, comps, winner, specs, device, questions, rows))


def cmd_baseline(_args) -> None:
    require_clean()
    questions = e7.dev_questions(ROOT)
    rows: dict[str, list[dict]] = {}
    with gpu():
        models = Models()
        for name, spec in BASELINES.items():
            rows[name] = e7.run_retrieval(f"baseline-{name}", make_retriever(spec, models), spec, questions, WORK / "baseline", ROOT)
        device = models.info()
    path, comps = e7.write_retrieval_record("baseline", rows, BASELINES, "dense", device, ROOT)
    best, sequential = "dense", []
    for cand in ("hybrid", "hybrid_rerank"):  # simpler first; a step up must beat the best so far
        pairs = e7.pair_comparisons(rows[cand], rows[best], ROOT, (cand, best))
        if best != "dense":
            sequential += pairs
        if e7.better(pairs):
            best = cand
    chosen = e7.Chosen.load(CHOSEN)
    chosen.retrieval = asdict(BASELINES[best])
    chosen.decisions.append({"stage": "baseline", "kept": best, "record": path.stem, "comparisons": comps, "device": device})
    chosen.save(CHOSEN)
    record = json.loads(path.read_text(encoding="utf-8"))
    finish("baseline", path.stem, retrieval_draft("baseline", record, comps, best, BASELINES, device, questions, rows, sequential))


def cmd_chunking(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    retrieval_stage("chunking", {"structured": replace(base, index="index-structured")})


def cmd_header(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    index = "index-structured-header" if base.index == "index-structured" else "index-header"
    retrieval_stage("header", {"header": replace(base, index=index)})


def cmd_crops(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    probe = Retriever(knowledge() / base.index, config=RetrievalConfig(crops_from_title=True))
    widened = {probe._chunks[i]["source"] for i, c in enumerate(probe._crops_of) if len(c) > 1}
    print(f"{len(widened)} documents answer for a second crop when their title names one")
    retrieval_stage("crops", {"crops_from_title": replace(base, crops_from_title=True)},
                    extra={"documents_with_extra_crops": len(widened)})


def cmd_queries(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    retrieval_stage("queries", {m: replace(base, query_mode=m) for m in ("history", "diseases", "full")})


# ------------------------------------------------------------------ latency of reranking, for T7.4

def gpu_memory_in_use_mib() -> int | None:
    done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(done.stdout.split()[0]) if done.returncode == 0 and done.stdout.split() else None


def cmd_latency(_args) -> None:
    """Time the reranker on the CUDA torch path, the CPU torch path and the CPU ONNX path that serving uses. All three
    rerank the same candidates for the same questions, so their ranks also show whether the paths agree."""
    require_clean()
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    spec = replace(base, rerank=True)
    questions = e7.dev_questions(ROOT)
    answerable = [q for q in questions if q.expected == "answer"]
    rows, samples, specs = {}, {}, {}
    with gpu():
        busy = gpu_memory_in_use_mib()
        models = Models()
        from sprout.rag.gpu_models import TorchReranker
        from sprout.rag.retrieve import FastReranker

        paths = {"cuda": lambda: models.reranker, "cpu_torch": lambda: TorchReranker(device="cpu"),
                 "cpu_onnx": lambda: FastReranker(RetrievalConfig().reranker_model)}
        for tag, build in paths.items():
            timed = e7.TimedReranker(build())
            retriever = make_retriever(spec, models, reranker=timed)
            name = f"rerank_{tag}"
            specs[name] = spec
            rows[name] = e7.run_retrieval(name, retriever, spec, questions, WORK / "latency", ROOT)
            search_ms, rerank_ms = e7.time_search(retriever, timed, spec, answerable, ROOT)
            samples[f"search_ms_{tag}"], samples[f"rerank_ms_{tag}"] = search_ms, rerank_ms
        device = models.info()
    machine = {"cpu": platform.processor(), "cpu_count": __import__("os").cpu_count(), "python": platform.python_version(),
               "gpu_memory_in_use_mib_when_stage_began": busy, "candidates_reranked_per_search": e7.SERVED_K * RetrievalConfig().rerank_pool,
               "searches_timed": len(answerable), "warmup_searches": 3,
               "note": "Embedding the query runs on CUDA in every path, so search_ms differs between paths by the reranker only."}
    path, comps = e7.write_retrieval_record("latency", rows, specs, "rerank_cuda", device, ROOT, measurements=samples,
                                            extra_config={"machine": machine})
    record = json.loads(path.read_text(encoding="utf-8"))
    vals = record["measurements"]["values"]
    lines = [f"### T6.4 - Cost of reranking, for T7.4 ({date.today()})",
             "- **Chose:** nothing; this is a measurement. Milliseconds per search with 15 candidates reranked (k=5, pool 3):"]
    for tag in paths:
        lines.append(f"  - {tag}: reranker call p50 {vals[f'rerank_ms_{tag}_p50']:.1f} / p95 {vals[f'rerank_ms_{tag}_p95']:.1f} ms; "
                     f"whole search p50 {vals[f'search_ms_{tag}_p50']:.1f} / p95 {vals[f'search_ms_{tag}_p95']:.1f} ms")
    lines += ["- **Do the paths agree?** recall@5 per path: " + ", ".join(f"{n} {fmt(m['recall@5'])}" for n, m in record["metrics"].items() if not n.endswith('_terms')),
              f"- **Machine:** {machine}", f"- **Device:** {device}", f"- **Evidence:** record `{path.stem}`.",
              "- **Would change if:** another machine or a busier GPU gives other numbers; this is one laptop."]
    finish("latency", path.stem, "\n".join(lines))


# ------------------------------------------------------------------ floors

def cmd_floors(_args) -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    base = e7.RetrievalSpec.from_json(chosen.retrieval)
    questions = e7.dev_questions(ROOT)
    by_mode: dict[str, list[dict]] = {}
    with gpu():
        models = Models()
        for key, (mode, rerank) in FLOOR_MODES.items():
            spec = replace(base, mode=mode, rerank=rerank)
            by_mode[key] = e7.run_retrieval(f"floors-{key.replace('+', '_')}", make_retriever(spec, models), spec,
                                            questions, WORK / "floors", ROOT, diagnose=True)
        device = models.info()
    floors = {k: e7.choose_floor(*e7.floor_groups(questions, rows)) for k, rows in by_mode.items()}
    diagnosis = {k: e7.floor_diagnosis(questions, by_mode[k], floors[k]["floor"]) for k in floors}
    default = RetrievalConfig().min_score
    path = e7.write_floor_record(questions, by_mode, floors, default, device, ROOT, diagnosis=diagnosis,
                                 extra_config={"retrieval": asdict(base)})
    chosen.floors = {k: v["floor"] for k, v in floors.items()}
    chosen.decisions.append({"stage": "floors", "record": path.stem, "floors": {k: {a: b for a, b in v.items() if a != "curve"} for k, v in floors.items()},
                             "device": device})
    chosen.save(CHOSEN)
    record = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"### T6.4 - Score floor per mode, picked on rag_dev (method v2, {date.today()})",
             f"- **Chose:** floors {chosen.floors} on retrieval setup {asdict(base)}, replacing the single 0.3.",
             "- **Rule:** among floors that answer at least 80% of answerable questions, the one with the highest "
             "declines_when_unanswerable; if none reaches 80%, the highest mean of the two (said below). Revised after the "
             "first pass, whose rule picked a dense floor that answered only 65% of answerable questions.",
             f"- **Considered:** the old single floor of {default}, and the first pass's balanced rule.",
             f"- **Evidence:** record `{path.stem}`; the floor is picked on rag_dev itself, so these rates are optimistic. "
             "answers / declines (abstain) / declines (wrong crop) / declines (ambiguous) per system:"]
    for n, m in record["metrics"].items():
        lines.append(f"  - {n}: " + " / ".join(fmt(m[k]) for k in ("answers_when_answerable", "declines_when_unanswerable",
                                                                   "declines_when_crop_mismatch", "declines_when_ambiguous")))
    for k, f in floors.items():
        lines.append(f"  - {k}: floor {f['floor']}, {f['rule_applied']}")
    lines.append("- **Why answerable questions fall under the floor:**")
    for k, d in diagnosis.items():
        lines.append(f"  - {k}: {d}")
    lines += [f"- **Device:** {device}", "- **Would change if:** floors picked on rag_dev do not hold on new questions."]
    finish("floors", path.stem, "\n".join(lines))


# ------------------------------------------------------------------ answers

def answer_config(chosen: e7.Chosen, version: str) -> AnswerConfig:
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    return AnswerConfig(prompt_version=version, mode=spec.mode, rerank=spec.rerank, query_mode=spec.query_mode,
                        sources=load_sources(ROOT / "knowledge" / "manifest.csv"))


def cache_tag(variant: str) -> str:
    """Prompt variants run with no score floor, so every question reaches the writer and the prompt effect is not
    hidden by retrieval; `served` (and Claude) run with the chosen floors, as the app would."""
    return "answers-floor" if variant == "served" else "answers-nofloor"


def cache_file(chosen: e7.Chosen, tag: str) -> Path:
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    floors = chosen.floors if tag == "answers-floor" else {}
    return e7.answer_retrieval_file(WORK / "answers", tag, answer_config(chosen, "base"), floors, 0.0,
                                    knowledge() / spec.index, ROOT)


def cmd_retrieve_answers(_args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    if not chosen.floors:
        sys.exit("Run the floors stage first: the served answers need the score floor.")
    questions = e7.dev_questions(ROOT)
    with gpu():
        models = Models()
        for tag, floors in (("answers-floor", chosen.floors), ("answers-nofloor", {})):
            retriever = make_retriever(spec, models, floors=floors)
            rows = e7.run_answer_retrieval(retriever, answer_config(chosen, "base"), questions, WORK / "answers", ROOT, tag=tag)
            print(f"{tag}: {len(rows)} questions, {sum(not r['passages'] for r in rows.values())} with no passage")
        print(f"device {models.info()}")


def variant_version(chosen: e7.Chosen, variant: str) -> str:
    if variant == "combined":
        return chosen.combined
    if variant == "served":
        return chosen.prompt
    return "base" if variant == "base" else f"base+{variant}"


def gen_key(chosen: e7.Chosen, variant: str) -> str:
    return e7.checkpoint_key(variant_version(chosen, variant), cache_file(chosen, cache_tag(variant)).name, WRITER, e7.ANSWER_VERSION)


def judge_key(chosen: e7.Chosen, variant: str) -> str:
    return e7.checkpoint_key(gen_key(chosen, variant), JUDGE, rubric())


def answer_retriever(chosen: e7.Chosen, tag: str):
    path = cache_file(chosen, tag)
    rows = e7.Checkpoint(path).rows
    questions = e7.dev_questions(ROOT)
    if len(rows) < len(questions):
        sys.exit(f"Run retrieve-answers first ({path.name} is missing or incomplete).")
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    crops = {r["crop"] for r in rows.values() if r["crop"]} | {"general"}
    return e7.cached_retriever(crops, rows), e7.single_page_sources(knowledge() / spec.index)


@contextlib.contextmanager
def ollama_stage(client: OllamaClient, role: str):
    """Hold the GPU for one Ollama stage, load the model with a tiny call, record where it sits (and stop if it is
    on the CPU), and unload it afterwards."""
    with gpu():
        try:
            client.complete([{"role": "user", "content": "Reply with the word ok."}], max_tokens=8, timeout_s=300)
            place = client.placement()
            if place is None or place["size_vram"] <= 0:
                raise RuntimeError(f"{client.model} is not on the GPU: {place}")
            (WORK / "device").mkdir(parents=True, exist_ok=True)
            (WORK / "device" / f"{role}.json").write_text(json.dumps({"model": client.model, **place}), encoding="utf-8")
            yield
        finally:
            with contextlib.suppress(LLMError):
                client.unload()


def ollama_device(role: str) -> tuple[str, dict]:
    path = WORK / "device" / f"{role}.json"
    if not path.is_file():
        sys.exit(f"No GPU placement was recorded for the {role}; run its stage (generate or judge) first.")
    p = json.loads(path.read_text(encoding="utf-8"))
    share = p["size_vram"] / p["size"] if p["size"] else 0.0
    return ("cuda" if share >= 0.999 else f"cuda+cpu ({share:.0%} of the model in GPU memory)"), p


def cmd_generate(args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    questions = e7.dev_questions(ROOT)
    writer = OllamaClient(WRITER, keep_alive="10m")
    incomplete = 0
    with ollama_stage(writer, "writer"):
        for variant in args.variants:
            retriever, single = answer_retriever(chosen, cache_tag(variant))
            cfg = answer_config(chosen, variant_version(chosen, variant))
            rows, missing = e7.run_generation(variant, questions, retriever, writer, cfg, WORK / "generate", single, ROOT,
                                              key=gen_key(chosen, variant))
            calls = sum(r["llm_calls"] for r in rows)
            secs = sum(r["llm_seconds"] for r in rows)
            print(f"{variant}: {len(rows)}/{len(questions)} rows, {calls} writer calls, {secs:.0f} s of writer time, "
                  f"{sum(r['llm_calls'] == 0 for r in rows)} rows skipped the writer (no passage), {len(missing)} missing")
            incomplete += len(missing)
    if incomplete:
        print("Some questions got no model answer (Ollama down or timing out). Run the same command again to resume.")
        sys.exit(EXIT_INCOMPLETE)


def generated(chosen: e7.Chosen, variant: str) -> list[dict]:
    ck = e7.Checkpoint(WORK / "generate" / f"generate-{variant}-{gen_key(chosen, variant)}.jsonl")
    qs = e7.dev_questions(ROOT)
    if any(q.id not in ck.rows for q in qs):
        sys.exit(f"{variant}: generation is incomplete; run generate first.")
    return [ck.rows[q.id] for q in qs]


def cmd_judge(args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    questions = {q.id: q for q in e7.dev_questions(ROOT)}
    judge = OllamaClient(JUDGE, keep_alive="10m")
    try:
        with ollama_stage(judge, "judge"):
            for variant in args.variants:
                graded = e7.run_judging(variant, generated(chosen, variant), questions, judge, WORK / "judge",
                                        key=judge_key(chosen, variant))
                print(f"{variant}: {len(graded)} answers graded, {sum(len(r['claims']) for r in graded)} claims")
    except LLMError as exc:
        print(f"The judge failed: {exc}. Run the same command again to resume.")
        sys.exit(EXIT_INCOMPLETE)


def judged(chosen: e7.Chosen, variant: str, rows: list[dict]) -> list[dict]:
    ck = e7.Checkpoint(WORK / "judge" / f"judge-{variant}-{judge_key(chosen, variant)}.jsonl")
    if any(r["id"] not in ck.rows for r in rows):
        sys.exit(f"{variant}: judging is incomplete; run judge first.")
    return [ck.rows[r["id"]] for r in rows if ck.rows[r["id"]]["claims"]]


def present(chosen: e7.Chosen, variant: str) -> bool:
    if variant == "combined" and not chosen.combined:
        return False
    return (WORK / "generate" / f"generate-{variant}-{gen_key(chosen, variant)}.jsonl").is_file()


def values(record: dict, name: str) -> dict:
    return {**record["metrics"][name], "faithfulness": (record["judged"].get("values") or {}).get(f"faithfulness_{name}")}


def device_config() -> dict:
    w, wp = ollama_device("writer")
    j, jp = ollama_device("judge")
    return {"device": f"writer {w}; judge {j}", "device_info": {"writer": wp, "judge": jp}}


def cmd_prompts_record(_args) -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    questions = e7.dev_questions(ROOT)
    variants = [v for v in ALL_VARIANTS if present(chosen, v)]
    if "base" not in variants:
        sys.exit("Nothing generated yet for the base prompt.")
    rows = {v: generated(chosen, v) for v in variants}
    graded = {v: judged(chosen, v, rows[v]) for v in variants}
    config = {"writer": WRITER, "judge": {"model": JUDGE}, "retrieval": chosen.retrieval, "floors": chosen.floors,
              "prompts": {v: variant_version(chosen, v) for v in variants},
              "caches": {v: cache_tag(v) for v in variants},
              "answer_config": {k: v for k, v in asdict(answer_config(chosen, "base")).items() if k != "sources"},
              **device_config()}
    full, blind = e7.write_judgments(ROOT / "results" / "runs" / "E7-prompts-v2", [r for v in variants for r in graded[v]])
    config["judgments_file"], config["blind_file"] = full.name, blind.name
    path = e7.write_generation_record("prompts", None, rows, graded, {"writer": WRITER, "judge": JUDGE}, config, ROOT)
    record = json.loads(path.read_text(encoding="utf-8"))
    version, decisions, needs_combined = e7.choose_prompt({v: values(record, v) for v in variants if v != "served"})
    intervals = e7.prompt_intervals(questions, rows, graded)
    if needs_combined:
        chosen.combined = version
    else:
        chosen.prompt = version  # `served` is keyed by this, so a served run made with another prompt is no longer found
    chosen.decisions.append({"stage": "prompts", "record": path.stem, "prompt": version, "decisions": decisions, "intervals": intervals})
    chosen.save(CHOSEN)
    total = sum(len(g) for g in graded.values())
    lines = [f"### T6.4 - Prompt versions, one rule at a time, on {WRITER} (method v2, {date.today()})",
             f"- **Chose:** `{version}`" + (" (the kept additions still need a `combined` run: generate, judge, then prompts-record again)" if needs_combined else ""),
             "- **Considered, direction rule (kept when the target rises and answers_when_answerable falls by at most 0.05), with intervals:**"]
    for d in decisions:
        iv = intervals.get(d["addition"])
        gain_text = (f"target gains over base {', '.join(f'{a} {g:+.3f}' for a, g in d['target_gains'].items())}"
                     if "target_gains" in d else f"target gain {d['gain']:+.3f}")
        text = f"  - {d['addition']}: {'kept' if d['kept'] else 'dropped'}; {gain_text}, answers change {-d['answer_drop']:+.3f}"
        if iv:
            text += (f"; target {ci_text(iv['target'])}, {'distinguishable from noise' if iv['distinguishable'] else 'NOT distinguishable from noise'}"
                     f"; answers_when_answerable {ci_text(iv['answers'])}, McNemar p={iv['answers']['mcnemar_p']:.3f}")
        lines.append(text)
    lines += [f"- **Evidence:** record `{path.stem}` (n={record['n']}); judge {JUDGE}; {total} graded answers. Per variant "
              "faithfulness / answers_when_answerable / declines_when_unanswerable / declines_when_ambiguous:"]
    for v in variants:
        usage = record["config"]["writer_usage"][v]
        lines.append(f"  - {v}: " + " / ".join(fmt(values(record, v)[k]) for k in ("faithfulness", "answers_when_answerable",
                                                                                 "declines_when_unanswerable", "declines_when_ambiguous"))
                     + f"; writer calls {usage['model_calls']}, rows without a call {usage['rows_without_model_call']}, "
                       f"{usage['model_seconds']} s; graded answers {len(graded[v])}")
    lines += [f"- **Device:** {record['config']['device']}",
              "- **Would change if:** a second grader (T6.6) disagrees with the local judge on the claims in the blind file."]
    finish("prompts", path.stem, "\n".join(lines))
    print(f"graded answers across variants: {total} (T6.6 needs at least {e7.MIN_GRADED})")
    if total < e7.MIN_GRADED:
        print(f"FEWER than {e7.MIN_GRADED} graded answers. Add variants (generate and judge `combined` and `served`) and run again.")
        sys.exit(EXIT_TOO_FEW_GRADED)


def cmd_claude(args) -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    if not present(chosen, "served"):
        sys.exit("Run generate and judge for the `served` variant first: it is the local answer Claude is compared with.")
    questions = e7.dev_questions(ROOT)[:100]
    retriever, single = answer_retriever(chosen, "answers-floor")
    claude = ClaudeCodeClient(model="sonnet", counter_path=main_folder() / "results" / "claude_calls.json")
    key = e7.checkpoint_key("claude", variant_version(chosen, "served"), cache_file(chosen, "answers-floor").name, e7.ANSWER_VERSION)
    rows, missing = e7.run_generation("claude_sonnet", questions, retriever, claude, answer_config(chosen, chosen.prompt),
                                      WORK / "generate", single, ROOT, key=key)
    if missing:
        print(f"{len(missing)} questions got no Claude answer. Run again to resume.")
        sys.exit(EXIT_INCOMPLETE)
    qa = {q.id: q for q in questions}
    local = [r for r in generated(chosen, "served") if r["id"] in qa]
    jkey = e7.checkpoint_key(key, JUDGE, rubric())
    done = e7.Checkpoint(WORK / "judge" / f"judge-claude_sonnet-{jkey}.jsonl").rows
    try:
        if all(r["id"] in done for r in rows):  # everything is judged already: no model is loaded
            claude_graded = [done[r["id"]] for r in rows if done[r["id"]]["claims"]]
        else:
            judge = OllamaClient(JUDGE, keep_alive="10m")
            with ollama_stage(judge, "judge"):
                claude_graded = e7.run_judging("claude_sonnet", rows, qa, judge, WORK / "judge", key=jkey)
    except LLMError as exc:
        print(f"The judge failed: {exc}. Run again to resume.")
        sys.exit(EXIT_INCOMPLETE)
    local_graded = [r for r in judged(chosen, "served", generated(chosen, "served")) if r["id"] in qa]
    name = f"local_{chosen.prompt.replace('+', '_')}"
    full, blind = e7.write_judgments(ROOT / "results" / "runs" / "E7-claude-v2", [*local_graded, *claude_graded])
    # counted from the saved rows, so a stage that was stopped and resumed still reports every call it made
    claude_calls = sum(r["llm_calls"] for r in rows)
    config = {"writer": WRITER, "claude_writer": "sonnet", "judge": {"model": JUDGE}, "prompt": chosen.prompt,
              "retrieval": chosen.retrieval, "floors": chosen.floors, "claude_calls": claude_calls,
              "judgments_file": full.name, "blind_file": blind.name, **device_config()}
    if args.supersedes:
        config["supersedes"] = args.supersedes
    path = e7.write_generation_record("claude", "first_100", {name: local, "claude_sonnet": rows},
                                      {name: local_graded, "claude_sonnet": claude_graded},
                                      {"writer": WRITER, "claude_writer": "sonnet", "judge": JUDGE}, config, ROOT)
    record = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"### T6.4 - Claude (Sonnet) against the local writer on 100 dev questions (method v2, {date.today()})",
             "- **Chose:** reported as a comparison only; the local writer stays the shipped default.",
             f"- **Evidence:** record `{path.stem}`, judged by {JUDGE} (never Claude), {config['claude_calls']} Claude calls, "
             f"prompt `{chosen.prompt}` with the chosen floors for both writers. " + "; ".join(
                 f"{n}: faithfulness {fmt(values(record, n)['faithfulness'])}, answers {fmt(values(record, n)['answers_when_answerable'])}, "
                 f"declines {fmt(values(record, n)['declines_when_unanswerable'])}" for n in record["metrics"]),
             f"- **Device:** {record['config']['device']} (Claude runs remotely)",
             "- **Would change if:** the person changes the model or the cap."]
    finish("claude", path.stem, "\n".join(lines))


def cmd_status(_args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    print("chosen:", json.dumps({"retrieval": chosen.retrieval, "floors": chosen.floors, "prompt": chosen.prompt}))
    print("decisions so far:", [d["stage"] for d in chosen.decisions])
    marks = e7.superseded_records(ROOT)
    print(f"superseded records: {len(marks['superseded'])}; records needing a rerun: {sorted(marks['needs_rerun'])}")
    for folder in sorted(WORK.glob("*")) if WORK.is_dir() else []:
        for f in sorted(folder.glob("*.jsonl")):
            print(f"  {folder.name}/{f.name}: {sum(1 for _ in f.open(encoding='utf-8'))} rows")


COMMANDS = {
    "indexes": cmd_indexes, "baseline": cmd_baseline, "latency": cmd_latency, "chunking": cmd_chunking,
    "header": cmd_header, "crops": cmd_crops, "queries": cmd_queries, "floors": cmd_floors,
    "retrieve-answers": cmd_retrieve_answers, "generate": cmd_generate, "judge": cmd_judge,
    "prompts-record": cmd_prompts_record, "claude": cmd_claude, "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(COMMANDS))
    ap.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS), choices=ALL_VARIANTS,
                    help="generate and judge: prompt variants")
    ap.add_argument("--supersedes", help="claude: id of an earlier record this one corrects (kept untouched, named in config)")
    args = ap.parse_args(argv)
    try:
        COMMANDS[args.stage](args)
    except RuntimeError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return EXIT_GPU_BUSY if "taskctl" in str(exc) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
