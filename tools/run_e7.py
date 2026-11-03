"""Experiment E7 on rag_dev, one resumable stage at a time. Run from the worktree, with the shared venv:

    set PYTHONPATH=src
    <local path> tools/run_e7.py <stage> [options]

Stages, in order (commit between stages; a record needs a clean tree):
    indexes          build knowledge/index-structured, index-header, index-structured-header (GPU embedding)
    baseline         dense vs hybrid vs hybrid + rerank on the main index      -> E7 record, results/e7/chosen.json
    chunking         structure-aware chunks with whole tables vs the base
    header           context header on each chunk vs the base
    crops            shared-crop pages (a title naming two crops) vs the base
    queries          question + conversation / top-3 diseases / both vs the base
    floors           score floor per mode (dense, hybrid, hybrid+rerank) for the final retrieval setup
    retrieve-answers cache the passages each dev question gets under the final setup (GPU)
    generate         write answers with the local writer for each prompt variant (Ollama)
    judge            judge every claim with the local judge, a different model (Ollama)
    prompts-record   one E7 record for the prompt variants; picks the prompt; judgments files
    claude           best prompt on Claude (Sonnet) over the first 100 dev questions, judged locally, one record
    status           what is done

GPU: a stage that uses the GPU takes `taskctl gpu-acquire T6.4` from the main folder just before it and releases
it right after, even on failure. A stage that dies leaves its checkpoints in results/e7/work/; run it again.
rag_test is never read.
"""

from __future__ import annotations

import argparse
import contextlib
import json
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
from sprout.rag.retrieve import RetrievalConfig, Retriever  # noqa: E402

TASK = "T6.4"
WRITER = "qwen3.5:4b-q4_K_M"
JUDGE = "gemma4:e4b-it-qat"
WORK = ROOT / "results" / "e7" / "work"
CHOSEN = ROOT / "results" / "e7" / "chosen.json"
DRAFTS = ROOT / "results" / "e7" / "decisions"
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
PROMPT_VARIANTS = ("base", "faithful", "abstain", "ask")
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


def make_retriever(spec: e7.RetrievalSpec, models: Models, floors: dict | None = None) -> Retriever:
    cfg = RetrievalConfig(min_score=0.0, crops_from_title=spec.crops_from_title, floors=floors or {})
    return Retriever(knowledge() / spec.index, config=cfg, embedder=models.embedder,
                     reranker=models.reranker if spec.rerank else None)


def require_clean() -> None:
    if is_dirty(ROOT):
        sys.exit("The working tree has uncommitted changes, and a record needs a clean tree. Commit first, e.g.\n"
                 "  git add -A results docs && git commit -m \"E7: stage results\"")


def finish(stage: str, run_id: str, draft: str) -> None:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / f"{stage}.md").write_text(draft, encoding="utf-8")
    print(f"\nrecord: results/runs/{run_id}.json\ndraft DECISIONS entry (check it, then add to docs/DECISIONS.md): "
          f"results/e7/decisions/{stage}.md\n\n{draft}\n"
          f"\nNext: git add -A results docs && git commit -m \"E7 {stage}: record and decision\" "
          "(the results/runs files and chosen.json are tracked once committed)")


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


# ------------------------------------------------------------------ stages

def cmd_indexes(_args) -> None:
    for name, (chunker, header) in INDEXES.items():
        index = knowledge() / name
        if (index / "build-info.json").is_file():
            print(f"{name}: already built")
            continue
        info = build_index_gpu(index, task=TASK, manifest=ROOT / "knowledge" / "manifest.csv",
                               raw=ROOT / "knowledge" / "raw", chunker=chunker, header=header)
        print(f"{name}: {info['chunks']} chunks, {info['failures']} failed documents, device {info['device']}")


def retrieval_stage(stage: str, candidates: dict[str, e7.RetrievalSpec], base_name: str = "base") -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    questions = e7.dev_questions(ROOT)
    base = e7.RetrievalSpec.from_json(chosen.retrieval)
    specs = {base_name: base, **candidates}
    rows: dict[str, list[dict]] = {}
    with gpu():
        models = Models()
        for name, spec in specs.items():
            retriever = make_retriever(spec, models)
            rows[name] = e7.run_retrieval(f"{stage}-{name}", retriever, spec, questions, WORK / stage, ROOT)
        device = models.info()
    path, comps = e7.write_retrieval_record(stage, rows, specs, base_name, device, ROOT)
    record = json.loads(path.read_text(encoding="utf-8"))
    kept = [c for c in comps if e7.better(c)]
    winner = max(kept, key=lambda c: c["diff"])["a"] if kept else base_name
    chosen.retrieval = asdict(specs[winner])
    chosen.decisions.append({"stage": stage, "kept": winner, "record": path.stem,
                             "comparisons": comps, "device": device})
    chosen.save(CHOSEN)
    finish(stage, path.stem, retrieval_draft(stage, record, comps, winner, specs, device))


def retrieval_draft(stage, record, comps, winner, specs, device) -> str:
    lines = [f"### T6.4 - Retrieval, {stage}: {'kept ' + winner if winner != 'base' else 'dropped, base kept'} ({date.today()})",
             f"- **Chose:** {winner} = {asdict(specs[winner])}",
             "- **Considered:** " + "; ".join(f"{n} = {asdict(s)}" for n, s in specs.items() if n != winner),
             "- **Evidence:** record `" + record["run_id"] + "`, n=" + str(record["n"]) + " answerable rag_dev questions, "
             "gold hit per `config.gold_matching`. Per system recall@1 / @3 / @5 / @10 / MRR: "]
    for n, m in record["metrics"].items():
        lines[-1] += f"{n}: " + " / ".join(fmt(m[k]) for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")) + "; "
    for c in comps:
        lines.append(f"  - recall@5 {c['a']} minus {c['b']}: {c['diff']:+.3f}, 95% CI [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], "
                     f"McNemar p={c['mcnemar_p']:.3f}" + (" (kept: interval excludes zero)" if e7.better(c) else " (no measurable difference)"))
    lines.append(f"- **Device:** {device}")
    lines.append("- **Would change if:** a larger or reviewed question set shows a different recall@5 gain.")
    return "\n".join(lines)


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
    best = "dense"
    for cand in ("hybrid", "hybrid_rerank"):  # simpler first; a step up must beat the best so far
        if e7.better(e7.pair_comparison(rows[cand], rows[best], ROOT)):
            best = cand
    chosen = e7.Chosen.load(CHOSEN)
    chosen.retrieval = asdict(BASELINES[best])
    chosen.decisions.append({"stage": "baseline", "kept": best, "record": path.stem, "comparisons": comps, "device": device})
    chosen.save(CHOSEN)
    record = json.loads(path.read_text(encoding="utf-8"))
    finish("baseline", path.stem, retrieval_draft("baseline", record, comps, best, BASELINES, device))


def cmd_chunking(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    retrieval_stage("chunking", {"structured": replace(base, index="index-structured")})


def cmd_header(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    index = "index-structured-header" if base.index == "index-structured" else "index-header"
    retrieval_stage("header", {"header": replace(base, index=index)})


def cmd_crops(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    retrieval_stage("crops", {"crops_from_title": replace(base, crops_from_title=True)})


def cmd_queries(_args) -> None:
    base = e7.RetrievalSpec.from_json(e7.Chosen.load(CHOSEN).retrieval)
    retrieval_stage("queries", {m: replace(base, query_mode=m) for m in ("history", "diseases", "full")})


FLOOR_MODES = {
    "dense": ("dense", False), "hybrid": ("hybrid", False), "hybrid+rerank": ("hybrid", True),
}


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
                                            questions, WORK / "floors", ROOT)
        device = models.info()
    floors = {k: e7.choose_floor(*e7.floor_groups(questions, rows)) for k, rows in by_mode.items()}
    default = RetrievalConfig().min_score
    path = e7.write_floor_record(questions, by_mode, floors, default, device, ROOT)
    chosen.floors = {k: v["floor"] for k, v in floors.items()}
    chosen.decisions.append({"stage": "floors", "record": path.stem, "floors": floors, "device": device})
    chosen.save(CHOSEN)
    record = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"### T6.4 - Score floor per mode, picked on rag_dev ({date.today()})",
             f"- **Chose:** floors {chosen.floors} (retrieval setup {asdict(base)}), replacing the single 0.3.",
             "- **Considered:** the single floor of 0.3 (the T6.5 verifier found it filters almost nothing without reranking).",
             f"- **Evidence:** record `{path.stem}`. Per system answers_when_answerable / declines_when_unanswerable / "
             "declines_when_crop_mismatch: " + "; ".join(
                 f"{n}: " + " / ".join(fmt(m[k]) for k in ("answers_when_answerable", "declines_when_unanswerable",
                                                            "declines_when_crop_mismatch")) for n, m in record["metrics"].items()),
             "- **Would change if:** floors picked on rag_dev do not hold on new questions (they are optimistic here).",
             f"- **Device:** {device}"]
    finish("floors", path.stem, "\n".join(lines))


def answer_config(chosen: e7.Chosen, version: str) -> AnswerConfig:
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    return AnswerConfig(prompt_version=version, mode=spec.mode, rerank=spec.rerank, query_mode=spec.query_mode,
                        sources=load_sources(ROOT / "knowledge" / "manifest.csv"))


def cmd_retrieve_answers(_args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    if not chosen.floors:
        sys.exit("Run the floors stage first: the answers need the score floor.")
    questions = e7.dev_questions(ROOT)
    with gpu():
        models = Models()
        retriever = make_retriever(spec, models, floors=chosen.floors)
        rows = e7.run_answer_retrieval(retriever, answer_config(chosen, "base"), questions, WORK / "answers", ROOT)
        print(f"{len(rows)} questions retrieved; device {models.info()}")


def variant_version(chosen: e7.Chosen, variant: str) -> str:
    return chosen.prompt if variant == "combined" else "base" if variant == "base" else f"base+{variant}"


def answer_retriever(chosen: e7.Chosen):
    rows = e7.Checkpoint(WORK / "answers" / "answer-retrieval.jsonl").rows
    questions = e7.dev_questions(ROOT)
    if len(rows) < len(questions):
        sys.exit("Run retrieve-answers first.")
    spec = e7.RetrievalSpec.from_json(chosen.retrieval)
    crops = {r["crop"] for r in rows.values() if r["crop"]} - {"general"}
    return e7.cached_retriever(crops | {"general"}, rows), e7.single_page_sources(knowledge() / spec.index)


def run_with_ollama(client: OllamaClient):
    """Hold the GPU for one Ollama stage, and unload the model afterwards."""
    @contextlib.contextmanager
    def ctx():
        with gpu():
            try:
                yield
            finally:
                with contextlib.suppress(LLMError):
                    client.unload()
    return ctx()


def cmd_generate(args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    questions = e7.dev_questions(ROOT)
    retriever, single = answer_retriever(chosen)
    writer = OllamaClient(WRITER, keep_alive="10m")
    incomplete = 0
    with run_with_ollama(writer):
        for variant in args.variants:
            cfg = answer_config(chosen, variant_version(chosen, variant))
            rows, missing = e7.run_generation(variant, questions, retriever, writer, cfg, WORK / "generate", single, ROOT)
            print(f"{variant}: {len(rows)}/{len(questions)} answered by the writer, {len(missing)} missing")
            incomplete += len(missing)
    if incomplete:
        print("Some questions got no model answer (Ollama down or timing out). Run the same command again to resume.")
        sys.exit(EXIT_INCOMPLETE)


def generated(variant: str) -> list[dict]:
    qs = e7.dev_questions(ROOT)
    ck = e7.Checkpoint(WORK / "generate" / f"generate-{variant}.jsonl")
    if any(q.id not in ck.rows for q in qs):
        sys.exit(f"{variant}: generation is incomplete; run generate again first.")
    return [ck.rows[q.id] for q in qs]


def cmd_judge(args) -> None:
    questions = {q.id: q for q in e7.dev_questions(ROOT)}
    judge = OllamaClient(JUDGE, keep_alive="10m")
    try:
        with run_with_ollama(judge):
            for variant in args.variants:
                rows = generated(variant)
                graded = e7.run_judging(variant, rows, questions, judge, WORK / "judge")
                print(f"{variant}: {len(graded)} answers graded, {sum(len(r['claims']) for r in graded)} claims")
    except LLMError as exc:
        print(f"The judge failed: {exc}. Run the same command again to resume.")
        sys.exit(EXIT_INCOMPLETE)


def judged(variant: str, rows: list[dict]) -> list[dict]:
    ck = e7.Checkpoint(WORK / "judge" / f"judge-{variant}.jsonl")
    if any(r["id"] not in ck.rows for r in rows):
        sys.exit(f"{variant}: judging is incomplete; run judge again first.")
    return [ck.rows[r["id"]] for r in rows if ck.rows[r["id"]]["claims"]]


def values(record: dict, name: str) -> dict:
    return {**record["metrics"][name], "faithfulness": (record["judged"].get("values") or {}).get(f"faithfulness_{name}")}


def cmd_prompts_record(args) -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    variants = [v for v in (*PROMPT_VARIANTS, "combined") if (WORK / "generate" / f"generate-{v}.jsonl").is_file()]
    rows = {v: generated(v) for v in variants}
    graded = {v: judged(v, rows[v]) for v in variants}
    config = {"writer": WRITER, "judge": {"model": JUDGE}, "retrieval": chosen.retrieval, "floors": chosen.floors,
              "prompts": {v: variant_version(chosen, v) for v in variants},
              "answer_config": {k: v for k, v in asdict(answer_config(chosen, "base")).items() if k != "sources"}}
    stem = ROOT / "results" / "runs" / f"E7-prompts-{date.today():%Y%m%d}"
    all_judged = [r for v in variants for r in graded[v]]
    e7.write_judgments(stem, all_judged)
    config["judgments_file"] = f"{stem.name}.judgments.jsonl"
    config["blind_file"] = f"{stem.name}.blind.jsonl"
    path = e7.write_generation_record("prompts", None, rows, graded, {"writer": WRITER, "judge": JUDGE}, config, ROOT)
    record = json.loads(path.read_text(encoding="utf-8"))
    version, decisions, needs_combined = e7.choose_prompt({v: values(record, v) for v in variants})
    chosen.prompt = version
    chosen.decisions.append({"stage": "prompts", "record": path.stem, "prompt": version, "decisions": decisions})
    chosen.save(CHOSEN)
    total = sum(len(g) for g in graded.values())
    lines = [f"### T6.4 - Prompt versions, one rule at a time, on {WRITER} ({date.today()})",
             f"- **Chose:** `{version}`" + (" (the kept additions still need a `combined` run: generate, judge, then prompts-record again)" if needs_combined else ""),
             "- **Considered:** " + "; ".join(f"{d['addition']} ({'kept' if d['kept'] else 'dropped'}: target {d.get('gain', 0):+.3f}, "
                                              f"answers_when_answerable change {-d['answer_drop']:+.3f})" for d in decisions),
             f"- **Evidence:** record `{path.stem}` (n={record['n']}); judge {JUDGE}; {total} graded answers. Per variant "
             "faithfulness / answers_when_answerable / declines_when_unanswerable / declines_when_ambiguous: " + "; ".join(
                 f"{v}: " + " / ".join(fmt(values(record, v)[k]) for k in ("faithfulness", "answers_when_answerable",
                                                                          "declines_when_unanswerable", "declines_when_ambiguous"))
                 for v in variants),
             "- **Would change if:** a second grader (T6.6) disagrees with the local judge on the claims in the blind file."]
    finish("prompts", path.stem, "\n".join(lines))
    print(f"graded answers across variants: {total} (T6.6 needs at least {e7.MIN_GRADED})")
    if total < e7.MIN_GRADED:
        print(f"FEWER than {e7.MIN_GRADED} graded answers.")
        sys.exit(EXIT_TOO_FEW_GRADED)


def cmd_claude(args) -> None:
    require_clean()
    chosen = e7.Chosen.load(CHOSEN)
    questions = e7.dev_questions(ROOT)[:100]
    retriever, single = answer_retriever(chosen)
    counter = main_folder() / "results" / "claude_calls.json"
    claude = ClaudeCodeClient(model="sonnet", counter_path=counter)
    before = claude.calls_made()
    cfg = answer_config(chosen, chosen.prompt)
    rows, missing = e7.run_generation("claude_sonnet", questions, retriever, claude, cfg, WORK / "generate", single, ROOT)
    if missing:
        print(f"{len(missing)} questions got no Claude answer. Run again to resume.")
        sys.exit(EXIT_INCOMPLETE)
    qa = {q.id: q for q in questions}
    local_name = f"local_{chosen.prompt.replace('+', '_')}"
    local = [r for r in generated_for(chosen.prompt, chosen) if r["id"] in qa]
    judge = OllamaClient(JUDGE, keep_alive="10m")
    try:
        with run_with_ollama(judge):
            claude_graded = e7.run_judging("claude_sonnet", rows, qa, judge, WORK / "judge")
            local_graded = e7.run_judging(local_name, local, qa, judge, WORK / "judge")
    except LLMError as exc:
        print(f"The judge failed: {exc}. Run again to resume.")
        sys.exit(EXIT_INCOMPLETE)
    stem = ROOT / "results" / "runs" / f"E7-claude-{date.today():%Y%m%d}"
    e7.write_judgments(stem, [*local_graded, *claude_graded])
    config = {"writer": WRITER, "claude_writer": "sonnet", "judge": {"model": JUDGE}, "prompt": chosen.prompt,
              "retrieval": chosen.retrieval, "floors": chosen.floors, "claude_calls": claude.calls_made() - before,
              "judgments_file": f"{stem.name}.judgments.jsonl", "blind_file": f"{stem.name}.blind.jsonl"}
    path = e7.write_generation_record("claude", "first_100", {local_name: local, "claude_sonnet": rows},
                                      {local_name: local_graded, "claude_sonnet": claude_graded},
                                      {"writer": WRITER, "claude_writer": "sonnet", "judge": JUDGE}, config, ROOT)
    record = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"### T6.4 - Claude (Sonnet) against the local writer on 100 dev questions ({date.today()})",
             "- **Chose:** reported as a comparison only; the local writer stays the shipped default.",
             f"- **Evidence:** record `{path.stem}`, judged by {JUDGE} (never Claude), {config['claude_calls']} Claude calls. "
             + "; ".join(f"{n}: faithfulness {fmt(values(record, n)['faithfulness'])}, answers {fmt(values(record, n)['answers_when_answerable'])}, "
                         f"declines {fmt(values(record, n)['declines_when_unanswerable'])}" for n in record["metrics"]),
             "- **Would change if:** the person changes the model or the cap."]
    finish("claude", path.stem, "\n".join(lines))


def generated_for(version: str, chosen: e7.Chosen) -> list[dict]:
    """Rows of the variant whose prompt is `version` (base, an addition, or the combined run)."""
    for variant in (*PROMPT_VARIANTS, "combined"):
        if variant_version(chosen, variant) == version and (WORK / "generate" / f"generate-{variant}.jsonl").is_file():
            return generated(variant)
    sys.exit(f"No local generation exists for prompt {version}; run generate for it first.")


def cmd_status(_args) -> None:
    chosen = e7.Chosen.load(CHOSEN)
    print("chosen:", json.dumps({"retrieval": chosen.retrieval, "floors": chosen.floors, "prompt": chosen.prompt}))
    print("decisions so far:", [d["stage"] for d in chosen.decisions])
    for folder in sorted(WORK.glob("*")) if WORK.is_dir() else []:
        for f in sorted(folder.glob("*.jsonl")):
            print(f"  {folder.name}/{f.name}: {sum(1 for _ in f.open(encoding='utf-8'))} rows")


COMMANDS = {
    "indexes": cmd_indexes, "baseline": cmd_baseline, "chunking": cmd_chunking, "header": cmd_header,
    "crops": cmd_crops, "queries": cmd_queries, "floors": cmd_floors, "retrieve-answers": cmd_retrieve_answers,
    "generate": cmd_generate, "judge": cmd_judge, "prompts-record": cmd_prompts_record, "claude": cmd_claude,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(COMMANDS))
    ap.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS),
                    help="generate and judge: prompt variants (base faithful abstain ask combined)")
    args = ap.parse_args(argv)
    try:
        COMMANDS[args.stage](args)
    except RuntimeError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return EXIT_GPU_BUSY if "taskctl" in str(exc) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
