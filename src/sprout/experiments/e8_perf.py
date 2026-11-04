"""E8: how fast and how large the real photo-to-answer path is, in the serving container on the CPU.

Two halves, because the serving image has no pandas:
  - `measure` and `cold` run INSIDE the container with the serving dependencies only. They run the real `Identifier`
    and the real `answer()` (a real index, the real reranker, the writer through Ollama on the host) and save raw
    per-request numbers to a JSON file.
  - `record` runs on the host. It turns that file into a run record with `write_run`, so `check_run E8` can check it.
`floor-check` and `multi-citations` are host-side checks that call no model.
Every request is timed by wrapping the real objects, never by replacing them. Nothing here uses a fake; tests do.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sprout.atomic import atomic_write_text

SPLIT = "perf"
LEVELS = (1, 5, 20)
REQUEST_STAGES = ("preprocess_ms", "inference_ms", "retrieval_ms", "generation_ms", "end_to_end_ms")
COLD_STAGES = ("identifier_load_ms", "retriever_warm_ms", "first_request_ms", "total_ms")
FAILED_STATES = ("unavailable", "retrieval_failed")  # an outage is never recorded as a measurement
RSS_INTERVAL_S = 0.05
QUESTIONS = (
    "What is wrong with my plant and how do I treat it?",
    "How can I stop this from spreading to my other plants?",
    "What causes this and how do I prevent it next season?",
    "Is it safe to eat the fruit from a plant with this?",
)
NEAR_FLOOR = 0.05  # a reranker score within this of the floor counts as near it
SERVED_K, E7_K = 5, 10  # the k the app serves with, and the k E7 picked the floors at


def now_ms() -> float:
    return time.perf_counter() * 1000


# ---------------------------------------------------------------- timing the real objects

class StageTimers:
    """Milliseconds per stage for the request the calling thread is running."""

    def __init__(self) -> None:
        self._local = threading.local()

    def begin(self) -> None:
        self._local.stages = {}

    def add(self, name: str, ms: float) -> None:
        stages = getattr(self._local, "stages", None)
        if stages is not None:
            stages[name] = stages.get(name, 0.0) + ms

    def end(self) -> dict[str, float]:
        stages, self._local.stages = self._local.stages, None
        return stages


def _timed(timers: StageTimers, name: str, fn):
    def run(*args, **kwargs):
        start = now_ms()
        try:
            return fn(*args, **kwargs)
        finally:
            timers.add(name, now_ms() - start)

    return run


class _TimedSession:
    """The ONNX Runtime session, with `run` timed as inference."""

    def __init__(self, session, timers: StageTimers) -> None:
        self._session = session
        self.run = _timed(timers, "inference_ms", session.run)

    def __getattr__(self, name):
        return getattr(self._session, name)


class TimedRetriever:
    def __init__(self, retriever, timers: StageTimers) -> None:
        self._retriever = retriever
        self.search = _timed(timers, "retrieval_ms", retriever.search)

    @property
    def crops(self):
        return self._retriever.crops


class TimedLLM:
    def __init__(self, llm, timers: StageTimers) -> None:
        self.model = llm.model
        self.complete = _timed(timers, "generation_ms", llm.complete)


def crop_for(ident, crops: set[str]) -> str | None:
    """The plant the photo's top label names (a plant with no documents of its own maps to general guidance),
    or None when the photo gave no diagnosis, so `answer()` reads the plant from the words instead."""
    from sprout.rag.answer import GENERAL_CROP
    from sprout.rag.query import crop_of_label

    if ident.top is None:
        return None
    return crop_of_label(ident.top.label, crops) or GENERAL_CROP


def serve_request(identifier, retriever, llm, cfg, image_bytes: bytes, question: str):
    """The photo-to-answer path of one request: identify, then answer with the plant the photo named."""
    from sprout.rag.answer import answer

    ident = identifier.identify(image_bytes)
    ans = answer(question, identification=ident, history=[], retriever=retriever, llm=llm, config=cfg,
                 crop=crop_for(ident, retriever.crops))
    return ident, ans


class Stack:
    """The real identifier, retriever and writer, wrapped so every request reports its stage times.
    Use as a context manager: the identifier is instrumented inside it and restored after."""

    def __init__(self, identifier, retriever, llm, cfg) -> None:
        self.timers = StageTimers()
        self.identifier, self.cfg = identifier, cfg
        self.retriever = TimedRetriever(retriever, self.timers)
        self.llm = TimedLLM(llm, self.timers)
        self._undo: list = []

    def __enter__(self) -> Stack:
        from sprout.serve import export

        model = self.identifier.model
        before = (export.preprocess_array, model.session, getattr(self.identifier, "_decode"))
        export.preprocess_array = _timed(self.timers, "preprocess_ms", before[0])
        model.session = _TimedSession(before[1], self.timers)
        self.identifier._decode = _timed(self.timers, "preprocess_ms", before[2])
        self._undo = [before]
        return self

    def __exit__(self, *exc) -> None:
        from sprout.serve import export

        export.preprocess_array, self.identifier.model.session, _ = self._undo.pop()
        del self.identifier.__dict__["_decode"]  # the class's own method is back

    def request(self, path: str, data: bytes, question: str) -> dict:
        self.timers.begin()
        start = now_ms()
        ident, ans = serve_request(self.identifier, self.retriever, self.llm, self.cfg, data, question)
        total = now_ms() - start
        stages = self.timers.end()
        row = {"path": path, "identification": ident.status, "answer_state": ans.state}
        for name in REQUEST_STAGES[:-1]:
            row[name] = round(stages[name], 3) if name in stages else None  # None: the stage did not run
        row["end_to_end_ms"] = round(total, 3)
        return row


# ---------------------------------------------------------------- memory and machine

def read_rss_mb() -> float | None:
    """Resident memory of this process, from /proc; None where there is no /proc (not Linux)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


class RssSampler:
    def __init__(self, read, interval: float) -> None:
        self._read, self._interval = read, interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._read())
            self._stop.wait(self._interval)

    def __enter__(self) -> RssSampler:
        self.samples.append(self._read())
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(self._read())


def _first_line(path: str, key: str | None = None) -> str | None:
    try:
        for line in Path(path).read_text().splitlines():
            if key is None or line.startswith(key):
                return line.split(":", 1)[1].strip() if key else line.strip()
    except OSError:
        return None
    return None


def machine_info() -> dict:
    """What the measuring process can see: inside Docker Desktop this is its Linux VM, not the host."""
    info = {
        "platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count(),
        "cpu_model": _first_line("/proc/cpuinfo", "model name"),
        "memory_total": _first_line("/proc/meminfo", "MemTotal"),
        "cgroup_cpu_max": _first_line("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": _first_line("/sys/fs/cgroup/memory.max"),
    }
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
        info["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------- measuring

def run_level(stack: Stack, level: int, work: list[tuple[str, str]], images_root: Path,
              read_rss=read_rss_mb, interval: float = RSS_INTERVAL_S) -> dict:
    """Every image once, with `level` requests in flight at a time. Rows come back in `work` order."""

    def one(item: tuple[str, str]) -> dict:
        path, question = item
        return stack.request(path, (images_root / path).read_bytes(), question)

    start = time.perf_counter()
    with RssSampler(read_rss, interval) as sampler, ThreadPoolExecutor(level) as pool:
        rows = list(pool.map(one, work))
    wall = time.perf_counter() - start
    failed = [r["path"] for r in rows if r["answer_state"] in FAILED_STATES]
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(rows)} requests at concurrency {level} failed "
                           f"(the writer or the search was unavailable), e.g. {failed[0]}; this is not a result")
    return {"level": level, "wall_s": round(wall, 3), "rows": rows,
            "rss_mb": [round(x, 2) for x in sampler.samples], "peak_rss_mb": round(max(sampler.samples), 2)}


def measure(stack: Stack, work: list[tuple[str, str]], images_root: Path, *, levels=LEVELS, warmup: int = 1,
            read_rss=read_rss_mb, interval: float = RSS_INTERVAL_S) -> dict[str, dict]:
    if read_rss() is None:
        raise RuntimeError("memory cannot be read here (there is no /proc); run the measurement in the Linux container")
    for path, question in work[:warmup]:  # not recorded: the first request of the process is the cold start
        stack.request(path, (images_root / path).read_bytes(), question)
    return {str(level): run_level(stack, level, work, images_root, read_rss, interval) for level in levels}


def cold_start(make_identifier, make_retriever, llm, cfg, first: tuple[str, str], images_root: Path,
               read_rss=read_rss_mb) -> tuple[Stack, dict]:
    """Build the identifier and the retriever (loading the index and the reranker) and answer one request, timing each.
    Starts when called, so the Python start-up before it is not included."""
    start = now_ms()
    identifier = make_identifier()
    loaded = now_ms()
    retriever = make_retriever()
    retriever.warm(rerank=True)
    warmed = now_ms()
    stack = Stack(identifier, retriever, llm, cfg)
    with stack:
        row = stack.request(first[0], (images_root / first[0]).read_bytes(), first[1])
    done = now_ms()
    if row["answer_state"] in FAILED_STATES:
        raise RuntimeError(f"the first request failed ({row['answer_state']}); this is not a result")
    cold = {"identifier_load_ms": loaded - start, "retriever_warm_ms": warmed - loaded,
            "first_request_ms": done - warmed, "total_ms": done - start, "rss_mb": read_rss(), "first_request": row}
    return stack, {k: round(v, 3) if isinstance(v, float) else v for k, v in cold.items()}


def work_list(paths: list[str]) -> list[tuple[str, str]]:
    return [(p, QUESTIONS[i % len(QUESTIONS)]) for i, p in enumerate(paths)]


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_list(path: Path) -> list[str]:
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def run_measurement(args: argparse.Namespace, *, levels=LEVELS) -> dict:
    """Everything `measure` and `cold` do, with the real components. `levels` is empty for a cold-only run."""
    from sprout.llm.client import OllamaClient
    from sprout.rag import serving
    from sprout.rag.answer import AnswerConfig, load_sources
    from sprout.rag.retrieve import Retriever
    from sprout.serve.identify import Identifier

    started = time.time()
    root = Path(args.images_root)
    paths = read_list(args.images)
    work = work_list(paths)
    llm = OllamaClient(args.writer, base_url=args.ollama_url, keep_alive="30m")
    sources = load_sources(Path(args.manifest)) if args.manifest else None
    cfg = AnswerConfig(sources=sources)
    stack, cold = cold_start(
        lambda: Identifier(Path(args.model), Path(args.calibration)),
        lambda: Retriever(Path(args.index), config=serving.retrieval_config()),
        llm, cfg, work[0], root)
    out = {
        "format": 1, "started_at": started, "n_images": len(paths), "warmup": 1, "questions": list(QUESTIONS),
        "model_sha256": sha256_file(Path(args.model)), "calibration_sha256": sha256_file(Path(args.calibration)),
        "model_file": Path(args.model).name, "writer": args.writer, "ollama_url": args.ollama_url,
        "machine": machine_info(), "cold": cold, "levels": {},
        "answer_config": {"prompt_version": cfg.prompt_version, "mode": cfg.mode, "rerank": cfg.rerank, "k": cfg.k,
                          "query_mode": cfg.query_mode, "max_tokens": cfg.max_tokens, "floors": serving.FLOORS},
    }
    if levels:
        with stack:
            out["levels"] = measure(stack, work, root, levels=levels)
    out["ollama_placement"] = llm.placement()
    return out


# ---------------------------------------------------------------- the record

def _stage_samples(rows: list[dict], stage: str) -> list[float]:
    return [r[stage] for r in rows if r[stage] is not None]


def build_measurements(data: dict, cold_files: list[dict]) -> tuple[dict[str, list[float]], dict]:
    """Raw samples under the names `check_run` recomputes percentiles from, and how many requests ran each stage."""
    samples: dict[str, list[float]] = {}
    ran: dict[str, dict[str, int]] = {}
    for name in COLD_STAGES:
        samples[f"cold_start_{name}"] = [c[name] for c in [data["cold"], *(f["cold"] for f in cold_files)]]
    samples["cold_start_rss_mb"] = [c["rss_mb"] for c in [data["cold"], *(f["cold"] for f in cold_files)]]
    for key, block in data["levels"].items():
        rows = block["rows"]
        ran[f"c{key}"] = {}
        for stage in REQUEST_STAGES:
            got = _stage_samples(rows, stage)
            if not got:
                raise ValueError(f"no request at concurrency {key} ran the {stage} stage, so it has no number to record")
            samples[f"c{key}_{stage}"] = got
            ran[f"c{key}"][stage] = len(got)
        samples[f"c{key}_rss_mb"] = block["rss_mb"]
        samples[f"c{key}_peak_rss_mb"] = [block["peak_rss_mb"]]
        samples[f"c{key}_requests_per_s"] = [round(len(rows) / block["wall_s"], 4)]
    return samples, ran


def build_record(data: dict, *, machine: str, image: str, variant: str, root=None, cold_files: list[dict] | None = None,
                 allow_dirty: bool = False) -> Path:
    """Write the E8 run record from a measurement file. `machine` and `image` are stated by the person who ran it."""
    from sprout.eval.check_run import FAKE
    from sprout.eval.records import ROOT, split_info, write_run

    root = ROOT if root is None else root
    if not machine.strip() or not image.strip() or not variant.strip():
        raise ValueError("machine, image and variant must each say something: the record has to state what was measured")
    names = [data["writer"], data["model_file"]]
    if any(FAKE.search(n) for n in names):
        raise ValueError(f"{names} looks like a fake or stub; E8 measures the real path")
    if sorted(data["levels"], key=int) != [str(x) for x in LEVELS]:
        raise ValueError(f"a record needs every concurrency level {LEVELS}, the file has {sorted(data['levels'])}")
    info = split_info(SPLIT, root)
    samples, ran = build_measurements(data, cold_files or [])
    systems = {}
    for key, block in data["levels"].items():
        if [r["path"] for r in block["rows"]] != info.ids:
            raise ValueError(f"the requests at concurrency {key} are not the perf list's images, once each, in order")
        systems[f"c{key}"] = [{"path": r["path"], "identification": r["identification"], "answer_state": r["answer_state"]}
                              for r in block["rows"]]
    config = {
        "machine_stated": machine.strip(), "image": image.strip(), "variant": variant.strip(),
        "measured_inside": data["machine"], "levels": [int(k) for k in data["levels"]], "requests_per_level": data["n_images"],
        "requests_that_ran_each_stage": ran, "warmup_requests": data["warmup"], "questions": data["questions"],
        "model_sha256": data["model_sha256"], "calibration_sha256": data["calibration_sha256"],
        "answer": data["answer_config"], "ollama_url": data["ollama_url"], "ollama_placement": data["ollama_placement"],
        "cold_start_samples": 1 + len(cold_files or []),
        "device": "container on CPU (onnxruntime CPUExecutionProvider, no torch); the writer runs in Ollama on the host",
        "concurrency": "N requests in flight at once as threads in one process, as one gunicorn worker serves them; "
                       "HTTP and the web server's own queue are not included",
        "memory": "resident memory of the measuring process, sampled every 50 ms",
        "started_at": datetime.fromtimestamp(data["started_at"], timezone.utc).isoformat(),
        "labels_reviewed": False, "claude_calls": 0,
    }
    models = {"classifier": f"{data['model_file']} sha256 {data['model_sha256'][:12]}", "writer": data["writer"],
              "embedder": "BAAI/bge-small-en-v1.5 (fastembed ONNX)", "reranker": "Xenova/ms-marco-MiniLM-L-6-v2 (fastembed ONNX)"}
    return write_run(
        "E8", SPLIT, config, models, {s: {} for s in systems}, {},
        [{"setting": variant.strip(), "result": "measured", "levels": config["levels"]}], systems, info.hash,
        root=root, allow_dirty=allow_dirty, measurements=samples,
        notes="Latency and memory of the real Identifier and the real answer in the container on the CPU. "
              "Requests are the perf list's photos with a fixed question each. Nothing here is accuracy.")


# ---------------------------------------------------------------- the image list

def write_image_list(root: Path, n: int, split: str = "field_dev") -> Path:
    """datasets/perf/images.txt: `n` photos spread evenly over the sorted paths of a real-photo dev split.
    Only paths are listed; labels are never read for timing."""
    import csv

    import numpy as np

    if n < 100:
        raise ValueError("the perf list needs at least 100 photos for check_run --min-n 100")
    with (root / "datasets" / "manifests" / "field.csv").open(newline="", encoding="utf-8") as f:
        paths = sorted(r["path"] for r in csv.DictReader(f) if r["split"] == split)
    if len(paths) < n:
        raise ValueError(f"{split} has {len(paths)} photos, fewer than the {n} asked for")
    chosen = [paths[i] for i in np.linspace(0, len(paths) - 1, n).round().astype(int)]
    out = root / "datasets" / "perf" / "images.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, "\n".join(chosen) + "\n")
    return out


# ---------------------------------------------------------------- the floor check

class RecordingReranker:
    """Passes scores through and remembers every (query, texts, scores) it was asked for."""

    def __init__(self, inner) -> None:
        self.inner, self.calls = inner, []

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = list(self.inner.score(query, texts))
        self.calls.append({"query": query, "texts": list(texts), "scores": scores})
        return scores


def _declined(call_scores: list[list[float]], floor: float) -> bool:
    """search() keeps a pool's passages at or above the floor, and falls back to the next pool only when none is."""
    return not any(s and max(s) >= floor for s in call_scores)


def floor_check(retriever, recorder: RecordingReranker, torch_reranker, questions, classes: list[str], *,
                floor: float, near: float = NEAR_FLOOR) -> dict:
    """Score the very same candidate pools with the serving reranker (through `retriever`) and with `torch_reranker`,
    at the served k and at E7's k, and report what that does to the retrieval-level decline rates."""
    import numpy as np

    from sprout.eval.check_run import Scored, recompute_metric
    from sprout.experiments.e7_rag import BEHAVIOUR_METRICS, simulate_identification
    from sprout.rag import serving
    from sprout.rag.answer import GENERAL_CROP, AnswerConfig, bounded_history
    from sprout.rag.query import build_query

    cfg, crops = AnswerConfig(), retriever.crops
    torch_cache: dict[tuple[str, str], float] = {}

    def torch_scores(call: dict) -> list[float]:
        missing = [t for t in call["texts"] if (call["query"], t) not in torch_cache]
        if missing:
            for t, s in zip(missing, torch_reranker.score(call["query"], missing), strict=True):
                torch_cache[(call["query"], t)] = s
        return [torch_cache[(call["query"], t)] for t in call["texts"]]

    declined: dict[str, list[bool]] = {f"{r}_k{k}": [] for k in (SERVED_K, E7_K) for r in ("onnx", "torch")}
    pairs: dict[int, list[tuple[float, float]]] = {SERVED_K: [], E7_K: []}
    for q in questions:
        crop, ident = simulate_identification(q, crops, classes)
        searched = None if crop is None else (crop if crop in crops else GENERAL_CROP)
        query = build_query(q.question, ident, bounded_history(q.history, cfg), cfg.query_mode)
        for k in (SERVED_K, E7_K):
            before = len(recorder.calls)
            found = retriever.search(query, crop=searched, k=k, mode=serving.MODE, rerank=True)
            calls = recorder.calls[before:]
            onnx = [c["scores"] for c in calls]
            if _declined(onnx, floor) != (not found):
                raise RuntimeError(f"{q.id}: the floor rule does not reproduce the retriever's own result; "
                                   f"the retriever's floor is not {floor}")
            declined[f"onnx_k{k}"].append(_declined(onnx, floor))
            torch = [torch_scores(c) for c in calls]
            declined[f"torch_k{k}"].append(_declined(torch, floor))
            pairs[k] += [(o, t) for os_, ts in zip(onnx, torch, strict=True) for o, t in zip(os_, ts, strict=True)]
    ids = [q.id for q in questions]
    expected = [q.expected for q in questions]
    rates = {name: {m: recompute_metric(m, Scored(groups=ids, answered=~np.array(flags), expected=expected))
                    for m in BEHAVIOUR_METRICS} for name, flags in declined.items()}

    def differences(k: int) -> dict:
        arr = np.array(pairs[k], dtype=float)
        gap = np.abs(arr[:, 0] - arr[:, 1])
        close = (np.abs(arr - floor) <= near).any(axis=1)
        return {"passages_scored": len(arr), "mean_abs_diff": float(gap.mean()), "max_abs_diff": float(gap.max()),
                "near_floor_passages": int(close.sum()),
                "max_abs_diff_near_floor": float(gap[close].max()) if close.any() else None,
                "passages_that_cross_the_floor": int(((arr[:, 0] >= floor) != (arr[:, 1] >= floor)).sum())}

    def flips(a: str, b: str) -> list[str]:
        return [i for i, x, y in zip(ids, declined[a], declined[b], strict=True) if x != y]

    return {
        "split": "rag_dev", "questions": len(ids), "floor": floor, "near_floor_window": near,
        "pool": {f"k{SERVED_K}": SERVED_K * retriever.config.rerank_pool, f"k{E7_K}": E7_K * retriever.config.rerank_pool},
        "decline_means": "the question's best reranked passage scores under the floor (retrieval only; the writer's "
                         "own NO_SUPPORT declines are not in this)",
        "rates": rates, "score_differences": {f"k{k}": differences(k) for k in (SERVED_K, E7_K)},
        "questions_whose_decline_changes": {
            "reranker_only_at_served_k (onnx_k5 vs torch_k5)": flips("onnx_k5", "torch_k5"),
            "pool_only_with_onnx (onnx_k5 vs onnx_k10)": flips("onnx_k5", "onnx_k10"),
            "served_vs_how_E7_chose_the_floor (onnx_k5 vs torch_k10)": flips("onnx_k5", "torch_k10"),
        },
    }


# ---------------------------------------------------------------- [1, 2] citations in saved answers

_LIST_MARK = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]")
_SINGLE_MARK = re.compile(r"\[(\d+)\]")


def count_list_citations(rows: list[dict]) -> dict:
    """On saved answer rows (E7 generation files), how often a reply cites several passages in one bracket. The saved
    text keeps such brackets as written. `lost_sources` counts rows where a listed number is a shown passage that no
    single `[n]` cites; `flips_to_answered` counts those that were ungrounded only because of it."""
    with_list, lost, flips = [], [], []
    for r in rows:
        listed = {int(n) for m in _LIST_MARK.findall(r["text"]) for n in re.findall(r"\d+", m)}
        if not listed:
            continue
        with_list.append(r["id"])
        shown = {n for n in listed if 1 <= n <= len(r["passages"])}
        single = {int(n) for n in _SINGLE_MARK.findall(r["text"]) if 1 <= int(n) <= len(r["passages"])}
        if shown - single:
            lost.append(r["id"])
            if r["state"] == "ungrounded" and not single:
                flips.append(r["id"])
    return {"rows": len(rows), "rows_with_list_citation": len(with_list), "lost_sources": len(lost),
            "flips_to_answered": len(flips), "ids_with_list_citation": with_list, "ids_that_lose_sources": lost,
            "ids_that_flip": flips}


# ---------------------------------------------------------------- command line

def _write_json(path: Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=1) + "\n")
    return path


def _container_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True, help="the ONNX classifier (E12 picks it)")
    ap.add_argument("--calibration", required=True, help="results/e5/chosen.json")
    ap.add_argument("--index", required=True, help="the retrieval index folder")
    ap.add_argument("--images", required=True, help="datasets/perf/images.txt")
    ap.add_argument("--images-root", required=True, help="folder the paths in --images are relative to")
    ap.add_argument("--manifest", default=os.environ.get("SPROUT_MANIFEST"))
    ap.add_argument("--writer", required=True, help="the Ollama model that writes answers")
    ap.add_argument("--ollama-url", required=True, help="where the container reaches Ollama on the host")
    ap.add_argument("--out", required=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e8_perf")
    sub = ap.add_subparsers(dest="stage", required=True)
    _container_args(sub.add_parser("measure", help="in the container: cold start, then every stage at 1, 5 and 20"))
    _container_args(sub.add_parser("cold", help="in a fresh container: one more cold-start sample"))
    rec = sub.add_parser("record", help="on the host: write the run record")
    rec.add_argument("--measured", required=True)
    rec.add_argument("--cold", nargs="*", default=[], help="files from `cold` runs")
    rec.add_argument("--machine", required=True, help="the machine, stated: CPU, cores, RAM, OS, Docker limits")
    rec.add_argument("--image", required=True, help="the container image measured (name and digest)")
    rec.add_argument("--variant", required=True, help="what this run is: baseline, or what changed")
    lst = sub.add_parser("list", help="write datasets/perf/images.txt")
    lst.add_argument("--n", type=int, default=120)
    chk = sub.add_parser("floor-check", help="on the host: ONNX vs torch reranker scores near the floor, on rag_dev")
    chk.add_argument("--index", required=True)
    chk.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="cuda needs a GPU window")
    chk.add_argument("--out", required=True)
    mc = sub.add_parser("multi-citations", help="count [1, 2] citations in saved answers; calls no model")
    mc.add_argument("answers", nargs="+", help="E7 generate-*.jsonl files")
    a = ap.parse_args(argv)

    if a.stage == "measure":
        _write_json(a.out, run_measurement(a))
    elif a.stage == "cold":
        _write_json(a.out, run_measurement(a, levels=()))
    elif a.stage == "record":
        read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
        print(build_record(read(a.measured), machine=a.machine, image=a.image, variant=a.variant,
                           cold_files=[read(p) for p in a.cold]))
    elif a.stage == "list":
        from sprout.eval.records import ROOT

        print(write_image_list(ROOT, a.n))
    elif a.stage == "floor-check":
        _floor_check_cli(a)
    else:
        for path in a.answers:
            rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
            print(path, json.dumps({k: v for k, v in count_list_citations(rows).items() if not k.startswith("ids_")}))
    return 0


def _floor_check_cli(a: argparse.Namespace) -> None:
    from sprout.eval.records import ROOT, head_commit, split_info
    from sprout.experiments.e7_rag import dev_questions, load_classes
    from sprout.rag import serving
    from sprout.rag.gpu_models import TorchReranker
    from sprout.rag.retrieve import FastReranker, Retriever

    cfg = serving.retrieval_config()
    recorder = RecordingReranker(FastReranker(cfg.reranker_model))
    retriever = Retriever(Path(a.index), config=cfg, reranker=recorder)
    torch_reranker = TorchReranker(device=a.device)
    report = floor_check(retriever, recorder, torch_reranker, dev_questions(ROOT), load_classes(ROOT),
                         floor=serving.FLOORS["hybrid+rerank"])
    report.update(torch=torch_reranker.info, onnx_reranker=cfg.reranker_model, git_commit=head_commit(ROOT),
                  dataset_hash=split_info("rag_dev", ROOT).hash,
                  note="first-stage candidates come from the fastembed embedder here; E7 chose the floors with the torch embedder on CUDA")
    print(_write_json(a.out, report))


if __name__ == "__main__":
    raise SystemExit(main())
