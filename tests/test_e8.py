"""E8, end-to-end performance: fakes and a tiny exported model only. No GPU, no Ollama, no network, no Claude."""
import io
import json
import subprocess
import threading
import time

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from sprout.eval.check_run import check
from sprout.eval.records import split_info
from sprout.experiments import e8_perf as e8
from sprout.llm.client import FakeLLM, LLMError
from sprout.rag import serving
from sprout.rag.answer import LABEL_NOTE, AnswerConfig, answer
from sprout.rag.retrieve import Passage, RetrievalConfig, Retriever
from sprout.serve.export import export_module
from sprout.serve.identify import Identifier
from sprout.serve.types import Candidate, Identification

P1 = Passage("Rings appear on tomato leaves.", "tomato.pdf", 3, "tomato", 0.9)
P2 = Passage("Remove infected leaves.", "tomato.pdf", 4, "tomato", 0.8)
P3 = Passage("Water at the base.", "tomato.pdf", 5, "tomato", 0.7)
SURE = Identification("answered", Candidate("Tomato___Early_blight", 0.95))


class FakeRetriever:
    crops = {"tomato", "potato"}

    def __init__(self, passages):
        self.passages, self.searches = passages, []

    def search(self, query, **kw):
        self.searches.append(kw)
        return self.passages


def ask(reply, passages=(P1, P2, P3), **kw):
    return answer("What are the signs?", identification=kw.pop("ident", SURE), history=[],
                  retriever=FakeRetriever(list(passages)), llm=FakeLLM([reply]), **kw)


# ---------------------------------------------------------------- [1, 2] citations

@pytest.mark.parametrize("reply", ["Rings appear on leaves [1, 2].", "Rings appear on leaves [1,2].",
                                   "Rings appear on leaves [1; 2]."])
def test_a_list_of_passage_numbers_in_one_bracket_cites_every_passage(reply):
    out = ask(reply)
    assert [(c.source, c.page) for c in out.citations] == [("tomato.pdf", 3), ("tomato.pdf", 4)]
    assert out.grounded and out.state == "answered" and not out.text.startswith("No supporting document")
    assert "[1][2]" in out.text and out.text.endswith(LABEL_NOTE)


def test_numbers_that_were_not_shown_are_dropped_from_a_list_and_a_prose_range_is_left_alone():
    out = ask("Rings appear [1, 9]. Keep it between [10, 20] degrees. Also [3, 4].")
    assert [c.page for c in out.citations] == [3, 5]
    assert "[1]" in out.text and "[9]" not in out.text and "[10, 20]" in out.text and "[3]" in out.text
    assert "[4]" not in out.text


def test_single_brackets_behave_as_before():
    out = ask("Rings appear on leaves [1]. Also see [7] and [2].")
    assert [c.page for c in out.citations] == [3, 4] and "[7]" not in out.text


# ---------------------------------------------------------------- a tiny real path

CLASS_MAP = {"Tomato___healthy": 0, "Tomato___Early_blight": 1, "Potato___Late_blight": 2}
CHUNKS = [
    ("Tomato early blight causes rings on leaves.", "tomato.pdf", 1, "tomato"),
    ("Remove infected tomato leaves and water at the base.", "tomato.pdf", 2, "tomato"),
    ("Always clean pruning tools between plants.", "general.pdf", 1, "general"),
]
GOOD = "Remove infected leaves [1]. Water at the base [2]."


class Embedder:
    def embed_passages(self, texts):
        return np.array([[1.0, 0.0, float(i)] for i, _ in enumerate(texts)])

    def embed_query(self, text):
        return np.array([1.0, 0.0, 0.0])


class Reranker:
    def score(self, query, texts):
        time.sleep(0.002)
        return [0.95 - 0.01 * i for i, _ in enumerate(texts)]


def make_index(tmp_path):
    d = tmp_path / "index"
    d.mkdir(exist_ok=True)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for text, source, page, crop in CHUNKS:
            f.write(json.dumps(dict(text=text, source=source, page=page, crop=crop, diseases=[], title="t")) + "\n")
    return d


def make_retriever(tmp_path):
    return Retriever(make_index(tmp_path), config=serving.retrieval_config(), embedder=Embedder(), reranker=Reranker())


def make_identifier(tmp_path):
    torch.manual_seed(1)
    net = nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Flatten(), nn.Linear(3 * 8 * 8, 3), nn.Identity()).eval()
    model = export_module(net, CLASS_MAP, tmp_path / "tiny.onnx")
    calibration = tmp_path / "chosen.json"
    gate = {"kind": "confidence", "confidence_threshold": 0.0}
    calibration.write_text(json.dumps({"temperature": 1.0, "abstain_threshold": 0.0,
                                       "gate": {"chosen": "confidence", "settings": gate}}), encoding="utf-8")
    return Identifier(model, calibration)


def png(size=(40, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 140, 40)).save(buf, "PNG")
    return buf.getvalue()


class SlowLLM(FakeLLM):
    """Counts how many calls are inside `complete` at once."""

    def __init__(self, reply=GOOD, seconds=0.01, model="tiny-writer"):
        super().__init__([reply], model=model)
        self.seconds, self.now, self.peak, self._lock = seconds, 0, 0, threading.Lock()

    def complete(self, messages, *, max_tokens, timeout_s):
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(self.seconds)
            return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)
        finally:
            with self._lock:
                self.now -= 1


def stack_for(tmp_path, llm=None, retriever=None):
    return e8.Stack(make_identifier(tmp_path), retriever or make_retriever(tmp_path), llm or SlowLLM(), AnswerConfig())


def test_the_plant_comes_from_the_photo_label():
    crops = {"tomato", "potato"}
    assert e8.crop_for(Identification("unsure", Candidate("Tomato___Early_blight", 0.4)), crops) == "tomato"
    assert e8.crop_for(Identification("answered", Candidate("Banana___black_sigatoka", 0.9)), crops) == "general"
    assert e8.crop_for(Identification("unsure", None), crops) is None


def test_the_request_passes_crop_to_answer_even_for_an_unsure_photo():
    spy = FakeRetriever([P1])
    unsure = Identification("unsure", Candidate("Potato___Late_blight", 0.3))
    ident = type("I", (), {"identify": lambda self, b: unsure})()
    e8.serve_request(ident, spy, FakeLLM([GOOD]), AnswerConfig(), b"", "Is it a tomato problem?")
    assert spy.searches[0]["crop"] == "potato"  # the photo's plant, not the "tomato" the words name


def test_one_request_reports_each_stage_and_the_stages_fit_in_the_total(tmp_path):
    with stack_for(tmp_path, SlowLLM(seconds=0.03)) as stack:
        row = stack.request("p/0.png", png(), "What is wrong with my plant?")
    assert row["identification"] == "answered" and row["answer_state"] == "answered"
    for stage in e8.REQUEST_STAGES:
        assert row[stage] is not None and row[stage] > 0, stage
    assert row["generation_ms"] >= 30 and row["retrieval_ms"] >= 2
    assert row["preprocess_ms"] + row["inference_ms"] + row["retrieval_ms"] + row["generation_ms"] <= row["end_to_end_ms"] + 1


def test_a_photo_that_cannot_be_read_skips_the_later_stages_instead_of_recording_zeros(tmp_path):
    with stack_for(tmp_path) as stack:
        row = stack.request("p/bad.png", b"not a picture", "What is wrong?")
    assert row["identification"] == "invalid" and row["answer_state"] == "no_plant"
    assert row["inference_ms"] is None and row["retrieval_ms"] is None and row["generation_ms"] is None
    assert row["end_to_end_ms"] > 0


def test_the_identifier_is_put_back_after_measuring(tmp_path):
    from sprout.serve import export

    stack = stack_for(tmp_path)
    original = (export.preprocess_array, stack.identifier.model.session)
    with stack:
        assert export.preprocess_array is not original[0]
    assert (export.preprocess_array, stack.identifier.model.session) == original
    assert "_decode" not in stack.identifier.__dict__


# ---------------------------------------------------------------- levels, memory, failures

def photos(tmp_path, n):
    folder = tmp_path / "imgs" / "p"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{i}.png").write_bytes(png((20 + i % 5, 20)))
    return [f"p/{i}.png" for i in range(n)]


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return 100.0 + self.n


def test_the_number_in_flight_follows_the_level_and_rows_stay_in_order(tmp_path):
    paths = photos(tmp_path, 12)
    peaks = {}
    for level in (1, 5):
        llm = SlowLLM(seconds=0.03)
        with stack_for(tmp_path, llm) as stack:
            out = e8.run_level(stack, level, e8.work_list(paths), tmp_path / "imgs", read_rss=Counter(), interval=0.005)
        assert [r["path"] for r in out["rows"]] == paths
        assert out["peak_rss_mb"] == max(out["rss_mb"]) and len(out["rss_mb"]) >= 2
        peaks[level] = llm.peak
    assert peaks[1] == 1 and 2 <= peaks[5] <= 5


def test_an_outage_stops_the_measurement_instead_of_recording_fast_failures(tmp_path):
    paths = photos(tmp_path, 3)

    class Down(SlowLLM):
        def complete(self, messages, *, max_tokens, timeout_s):
            raise LLMError("down")

    with stack_for(tmp_path, Down()) as stack, pytest.raises(RuntimeError, match="not a result"):
        e8.run_level(stack, 1, e8.work_list(paths), tmp_path / "imgs", read_rss=Counter())


def test_memory_that_cannot_be_read_refuses_instead_of_inventing_a_number(tmp_path):
    paths = photos(tmp_path, 2)
    with stack_for(tmp_path) as stack, pytest.raises(RuntimeError, match="Linux container"):
        e8.measure(stack, e8.work_list(paths), tmp_path / "imgs", read_rss=lambda: None)


def test_cold_start_times_the_loads_and_the_first_request(tmp_path):
    paths = photos(tmp_path, 1)
    stack, cold = e8.cold_start(lambda: make_identifier(tmp_path), lambda: make_retriever(tmp_path), SlowLLM(),
                                AnswerConfig(), e8.work_list(paths)[0], tmp_path / "imgs", read_rss=Counter())
    assert all(cold[k] > 0 for k in e8.COLD_STAGES) and cold["rss_mb"] > 100
    assert cold["total_ms"] >= cold["identifier_load_ms"] + cold["retriever_warm_ms"] + cold["first_request_ms"] - 1
    assert stack.retriever._retriever.warmed and cold["first_request"]["answer_state"] == "answered"


# ---------------------------------------------------------------- the record

N = 100


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def root(tmp_path):
    repo = tmp_path / "repo"
    (repo / "datasets" / "perf").mkdir(parents=True)
    (repo / "orchestration").mkdir()
    (repo / "orchestration" / "tasks.json").write_text("{}\n")
    (repo / "datasets" / "perf" / "images.txt").write_text("\n".join(f"p/{i}.png" for i in range(N)) + "\n")
    sh(repo, "init", "-q")
    sh(repo, "config", "user.email", "t@t")
    sh(repo, "config", "user.name", "t")
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("measured")
    paths = photos(tmp, N)
    stack, cold = e8.cold_start(lambda: make_identifier(tmp), lambda: make_retriever(tmp), SlowLLM(seconds=0.0),
                                AnswerConfig(), e8.work_list(paths)[0], tmp / "imgs", read_rss=Counter())
    with stack:
        levels = e8.measure(stack, e8.work_list(paths), tmp / "imgs", read_rss=Counter(), interval=0.01)
    return {"format": 1, "started_at": 1.7e9, "n_images": N, "warmup": 1, "questions": list(e8.QUESTIONS),
            "model_sha256": "ab" * 32, "calibration_sha256": "cd" * 32, "model_file": "tiny.onnx", "writer": "tiny-writer",
            "ollama_url": "http://host.docker.internal:11434", "machine": {"logical_cpus": 4}, "cold": cold,
            "levels": levels, "ollama_placement": {"size": 1, "size_vram": 1},
            "answer_config": {"prompt_version": serving.PROMPT_VERSION}}


def record(root, data, **kw):
    args = dict(machine="test box, 4 cores", image="sprout-app:test", variant="baseline", root=root)
    return e8.build_record({**data}, **{**args, **kw})


def test_a_tiny_run_writes_a_record_that_check_run_accepts(root, measured):
    path = record(root, measured)
    assert path.name.startswith("E8-")
    assert check("E8", root=root, min_n=N, split="perf") == []
    rec = json.loads(path.read_text())
    assert rec["n"] == N and rec["split"] == "perf" and set(rec["predictions"]) == {"c1", "c5", "c20"}
    assert rec["config"]["machine_stated"] == "test box, 4 cores"


def test_percentiles_by_stage_are_recorded(root, measured):
    rec = json.loads(record(root, measured).read_text())
    values, samples = rec["measurements"]["values"], json.loads((root / rec["measurements"]["file"]).read_text())
    for level in ("c1", "c5", "c20"):
        for stage in e8.REQUEST_STAGES:
            for q in (50, 95):
                assert values[f"{level}_{stage}_p{q}"] == pytest.approx(float(np.percentile(samples[f"{level}_{stage}"], q)))
        assert samples[f"{level}_peak_rss_mb"] and f"{level}_rss_mb_p95" in values
    for stage in e8.COLD_STAGES:
        assert f"cold_start_{stage}_p50" in values
    assert values["c1_end_to_end_ms_p95"] >= values["c1_end_to_end_ms_p50"]


def test_extra_cold_start_samples_are_added(root, measured):
    extra = {"cold": {**measured["cold"], "total_ms": 5.0}}
    rec = json.loads(record(root, measured, cold_files=[extra]).read_text())
    samples = json.loads((root / rec["measurements"]["file"]).read_text())
    assert len(samples["cold_start_total_ms"]) == 2 and rec["config"]["cold_start_samples"] == 2


def test_tampered_records_fail_the_check(root, measured):
    path = record(root, measured)
    original = path.read_text()
    rec = json.loads(original)
    rec["measurements"]["values"]["c20_end_to_end_ms_p95"] = 1.0  # a faster number than the samples give
    path.write_text(json.dumps(rec))
    assert any("does not match its saved file" in p for p in check("E8", root=root, min_n=N, split="perf"))
    path.write_text(original)
    assert check("E8", root=root, min_n=N, split="perf") == []

    samples = root / rec["measurements"]["file"]
    saved = samples.read_text()
    data = json.loads(saved)
    data["c20_end_to_end_ms"][0] += 1000.0  # a sample changed after the summary was written
    samples.write_text(json.dumps(data))
    assert any("does not match its saved file" in p for p in check("E8", root=root, min_n=N, split="perf"))
    samples.write_text(saved)

    rows = root / rec["predictions"]["c5"]
    kept = rows.read_text().splitlines()
    rows.write_text("\n".join(kept[:-1]) + "\n")  # a request dropped
    assert any("rows" in p or "exactly once" in p for p in check("E8", root=root, min_n=N, split="perf"))


def test_a_run_over_fewer_photos_than_the_minimum_fails(root, measured):
    record(root, measured)
    assert any("below --min-n" in p for p in check("E8", root=root, min_n=N + 1, split="perf"))


def test_a_record_refuses_what_it_cannot_state_or_what_is_not_real(root, measured):
    with pytest.raises(ValueError, match="state what was measured"):
        record(root, measured, machine=" ")
    with pytest.raises(ValueError, match="fake or stub"):
        record(root, {**measured, "writer": "fake-llm"})
    with pytest.raises(ValueError, match="every concurrency level"):
        record(root, {**measured, "levels": {"1": measured["levels"]["1"]}})
    five = measured["levels"]["5"]
    shuffled = {**measured, "levels": {**measured["levels"], "5": {**five, "rows": five["rows"][::-1]}}}
    with pytest.raises(ValueError, match="once each, in order"):
        record(root, shuffled)


def test_a_stage_that_never_ran_is_refused_not_recorded_as_empty(root, measured):
    twenty = measured["levels"]["20"]
    rows = [{**r, "generation_ms": None} for r in twenty["rows"]]
    broken = {**measured, "levels": {**measured["levels"], "20": {**twenty, "rows": rows}}}
    with pytest.raises(ValueError, match="generation_ms stage"):
        record(root, broken)


def test_the_image_list_is_spread_over_the_dev_photos(tmp_path):
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    rows = ["path,label,group,split,sha1,source"] + [
        f"raw/{i:03d}.jpg,A,g{i},field_{'dev' if i % 2 else 'test'},s{i},x" for i in range(400)]
    (manifests / "field.csv").write_text("\n".join(rows) + "\n")
    out = e8.write_image_list(tmp_path, 100)
    lines = out.read_text().splitlines()
    assert len(lines) == 100 and len(set(lines)) == 100 and lines == sorted(lines)
    assert all(int(x[4:7]) % 2 == 1 for x in lines)  # dev photos only
    assert split_info("perf", tmp_path).ids == lines
    with pytest.raises(ValueError, match="at least 100"):
        e8.write_image_list(tmp_path, 50)


# ---------------------------------------------------------------- list citations in saved answers

def test_list_citations_are_counted_on_saved_rows_without_a_model():
    rows = [
        {"id": "a", "state": "answered", "text": "Fine [1][2].", "passages": [1, 2, 3]},
        {"id": "b", "state": "answered", "text": "Rings [1]. Mulch [1, 2].", "passages": [1, 2, 3]},
        {"id": "c", "state": "ungrounded", "text": "No prefix lost here: rings appear [1, 3].", "passages": [1, 2, 3]},
        {"id": "d", "state": "ungrounded", "text": "A range [10, 20] degrees.", "passages": [1, 2, 3]},
    ]
    out = e8.count_list_citations(rows)
    assert out["rows"] == 4 and out["rows_with_list_citation"] == 3
    assert out["ids_that_lose_sources"] == ["b", "c"] and out["ids_that_flip"] == ["c"]
    assert (out["lost_sources"], out["flips_to_answered"]) == (2, 1)


# ---------------------------------------------------------------- the floor check

class Q:
    def __init__(self, id, question, expected):
        self.id, self.question, self.expected, self.history, self.crop = id, question, expected, [], None


class Scripted:
    """A reranker whose scores come from a table keyed by the passage text, so each scenario is exact."""

    def __init__(self, table, default=0.1):
        self.table, self.default = table, default

    def score(self, query, texts):
        return [self.table.get(t, self.default) for t in texts]


def floor_retriever(tmp_path, onnx_table):
    recorder = e8.RecordingReranker(Scripted(onnx_table))
    cfg = RetrievalConfig(floors={"hybrid+rerank": 0.73875}, rerank_pool=3)
    return Retriever(make_index(tmp_path), config=cfg, embedder=Embedder(), reranker=recorder), recorder


def test_floor_check_reports_the_effect_of_a_small_score_shift_near_the_floor(tmp_path):
    first = CHUNKS[0][0]
    # onnx scores the best tomato passage just over the floor, torch just under it: the one question flips
    retriever, recorder = floor_retriever(tmp_path, {first: 0.7400})
    torch_side = Scripted({first: 0.7370})
    questions = [Q("q1", "What are early blight rings on tomato leaves?", "answer"),
                 Q("q2", "Why is my banana yellow?", "abstain")]
    out = e8.floor_check(retriever, recorder, torch_side, questions, list(CLASS_MAP), floor=0.73875)
    rates = out["rates"]
    assert rates["onnx_k5"]["answers_when_answerable"] == 1.0 and rates["torch_k5"]["answers_when_answerable"] == 0.0
    # the banana question names no plant, so it searches every document and meets the same passage
    assert out["questions_whose_decline_changes"]["reranker_only_at_served_k (onnx_k5 vs torch_k5)"] == ["q1", "q2"]
    assert rates["onnx_k5"]["declines_when_unanswerable"] == 0.0 and rates["torch_k5"]["declines_when_unanswerable"] == 1.0
    d = out["score_differences"]["k5"]
    assert d["passages_that_cross_the_floor"] >= 1 and d["max_abs_diff_near_floor"] == pytest.approx(0.003)
    assert d["near_floor_passages"] >= 1 and out["pool"] == {"k5": 15, "k10": 30}


def test_floor_check_refuses_when_its_floor_rule_does_not_match_the_retriever(tmp_path):
    retriever, recorder = floor_retriever(tmp_path, {CHUNKS[0][0]: 0.9})
    with pytest.raises(RuntimeError, match="floor"):
        e8.floor_check(retriever, recorder, Scripted({}), [Q("q1", "tomato rings?", "answer")], list(CLASS_MAP), floor=0.95)
