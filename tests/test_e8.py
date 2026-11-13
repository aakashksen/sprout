"""E8, end-to-end performance: fakes and a tiny exported model only. No GPU, no Ollama, no network, no Claude."""
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from werkzeug.serving import make_server

from sprout.eval.check_run import check
from sprout.eval.records import split_info
from sprout.experiments import e8_perf as e8
from sprout.llm.client import FakeLLM, LLMError
from sprout.rag import serving
from sprout.rag.answer import LABEL_NOTE, answer
from sprout.rag.retrieve import Passage, RetrievalConfig, Retriever
from sprout.serve.export import export_module
from sprout.serve.types import Candidate, Identification

os.environ.setdefault("SPROUT_NO_AUTOBUILD", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import wsgi  # noqa: E402

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


# ---------------------------------------------------------------- a tiny real server

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


class SlowWarm(Retriever):
    delay = 0.0

    def warm(self, *, rerank=False):
        time.sleep(self.delay)
        super().warm(rerank=rerank)


def make_retriever(tmp_path, warm_delay=0.0):
    r = SlowWarm(make_index(tmp_path), config=serving.retrieval_config(), embedder=Embedder(), reranker=Reranker())
    r.delay = warm_delay
    return r


def model_files(tmp_path):
    torch.manual_seed(1)
    net = nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Flatten(), nn.Linear(3 * 8 * 8, 3), nn.Identity()).eval()
    model = export_module(net, CLASS_MAP, tmp_path / "tiny.onnx", source_sha256="b" * 64)
    calibration = tmp_path / "chosen.json"
    gate = {"kind": "confidence", "confidence_threshold": 0.0}
    calibration.write_text(json.dumps({"temperature": 1.0, "abstain_threshold": 0.0, "model_sha256": "b" * 64,
                                       "classes": list(CLASS_MAP), "gate": {"chosen": "confidence", "settings": gate}}), encoding="utf-8")
    return model, calibration


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


class Down(SlowLLM):
    def complete(self, messages, *, max_tokens, timeout_s):
        raise LLMError("down")


class LocalTarget:
    """The real app, built by the container's own entry point with measurement mode on, behind a local HTTP server.
    It stands in for the container (no Docker, and a threaded test server instead of gunicorn)."""

    def __init__(self, tmp_path, llm=None, warm_delay=0.0, **env):
        model, calibration = model_files(tmp_path)
        full = {"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_DATA_DIR": str(tmp_path / "data"), "SPROUT_MEASUREMENT": "1",
                "SPROUT_MODEL_PATH": str(model), "SPROUT_CALIBRATION_PATH": str(calibration), **env}
        self.app = wsgi.build(full, retriever=make_retriever(tmp_path, warm_delay), writer=llm or SlowLLM())
        self.server = make_server("127.0.0.1", 0, self.app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.reads = 0

    def mem_mb(self):
        self.reads += 1
        return 200.0 + self.reads

    def mem_peak_mb(self):
        return None

    def info(self):
        return {"workers": 1, "threads": 1}

    def stop(self):
        self.server.shutdown()


@pytest.fixture
def target(tmp_path):
    made = []

    def make(**kw):
        made.append(LocalTarget(tmp_path / f"t{len(made)}", **kw))
        return made[-1]

    (tmp_path / "t0").mkdir()
    for i in range(1, 4):
        (tmp_path / f"t{i}").mkdir()
    yield make
    for t in made:
        t.stop()


def photos(tmp_path, n):
    folder = tmp_path / "imgs" / "p"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{i}.png").write_bytes(png((20 + i % 5, 20)))
    return [f"p/{i}.png" for i in range(n)]


def client_for(t):
    c = e8.HttpClient(t.base_url, 30)
    c.sign_up()
    return c


def test_one_action_over_http_reports_each_stage_and_the_stages_fit_in_the_total(target):
    t = target(llm=SlowLLM(seconds=0.03))
    row = client_for(t).user_action("p/0.png", png(), "What is wrong with my plant?")
    assert row["identification"] == "answered" and row["answer_state"] == "answered"
    for stage in e8.REQUEST_STAGES:
        assert row[stage] is not None and row[stage] > 0, stage
    assert row["generation_ms"] >= 30 and row["retrieval_ms"] >= 2
    assert row["preprocess_ms"] + row["inference_ms"] + row["retrieval_ms"] + row["generation_ms"] <= row["end_to_end_ms"] + 1


def test_a_photo_the_server_refuses_is_a_failed_upload_with_no_later_stages(target):
    row = client_for(target()).user_action("p/bad.png", b"not a picture", "What is wrong?")
    assert row["identification"] == "http_400" and row["answer_state"] == "upload_failed"
    assert row["inference_ms"] is None and row["retrieval_ms"] is None and row["generation_ms"] is None
    assert row["end_to_end_ms"] > 0


def test_without_measurement_mode_no_stage_times_come_back(tmp_path):
    (tmp_path / "t").mkdir()
    t = LocalTarget(tmp_path / "t", SPROUT_MEASUREMENT="0")
    try:
        row = client_for(t).user_action("p/0.png", png(), "What is wrong?")
    finally:
        t.stop()
    assert row["answer_state"] == "answered" and row["retrieval_ms"] is None  # nothing is guessed


def test_a_server_that_answers_too_slowly_is_a_client_timeout(target):
    t = target(llm=SlowLLM(seconds=0.6))
    c = e8.HttpClient(t.base_url, 0.2)
    c.sign_up()
    row = c.user_action("p/0.png", png(), "What is wrong?")
    assert row["answer_state"] == "client_timeout" and row["answer_state"] in e8.FAILED_STATES


def test_nothing_listening_is_a_failed_connection_not_a_crash():
    c = e8.HttpClient(f"http://127.0.0.1:{e8.free_port()}", 6)  # Windows retries a refused connection for about 2 s
    row = c.user_action("p/0.png", png(), "What is wrong?")
    assert row["answer_state"] == "connection_failed"


def test_requests_run_one_at_a_time_and_rows_stay_in_order(tmp_path, target):
    paths = photos(tmp_path, 5)
    llm = SlowLLM(seconds=0.05)
    out = e8.measure(target(llm=llm), e8.work_list(paths), tmp_path / "imgs", interval=0.005)
    assert [r["path"] for r in out["rows"]] == paths
    assert out["peak_mem_mb"] == max(out["mem_mb"]) and len(out["mem_mb"]) >= 2
    assert llm.peak == 1  # no concurrency: the record makes no claim about it


def test_failed_requests_are_counted_and_kept_out_of_the_timings(tmp_path, target):
    paths = photos(tmp_path, 3)
    block = e8.measure(target(llm=Down()), e8.work_list(paths), tmp_path / "imgs")
    assert [r["answer_state"] for r in block["rows"]] == ["unavailable"] * 3
    data = {"cold": {k: 1.0 for k in e8.COLD_STAGES} | {"mem_mb": 1.0}, "run": block}
    with pytest.raises(ValueError, match="no request ran the"):
        e8.build_measurements(data, [])  # nothing but failures: no number to record
    ok = {**block["rows"][0], "answer_state": "answered", "end_to_end_ms": 9.0, "retrieval_ms": 1.0, "generation_ms": 2.0,
          "preprocess_ms": 3.0, "inference_ms": 4.0}
    mixed = {**data, "run": {**block, "rows": [ok, *block["rows"][1:]]}}
    samples, outcomes = e8.build_measurements(mixed, [])
    assert samples["request_end_to_end_ms"] == [9.0] and samples["request_failed_share"] == [0.6667]
    assert outcomes["failed"] == 2 and outcomes["by_answer_state"] == {"answered": 1, "unavailable": 2}
    assert "2 of 3 failed" in e8.failure_summary(outcomes)


def test_memory_that_cannot_be_read_stops_the_run_instead_of_inventing_a_number(tmp_path, target):
    paths = photos(tmp_path, 2)
    t = target()
    t.mem_mb = lambda: (_ for _ in ()).throw(RuntimeError("no cgroup"))
    with pytest.raises(RuntimeError, match="no cgroup"):
        e8.measure(t, e8.work_list(paths), tmp_path / "imgs")

    ok = target()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("cgroup vanished")
        return 100.0

    ok.mem_mb = flaky
    with pytest.raises(RuntimeError, match="memory sampling failed"):
        e8.measure(ok, e8.work_list(paths * 5), tmp_path / "imgs", interval=0.001)


def test_cold_start_is_timed_from_outside_up_to_ready_and_the_first_request(tmp_path):
    paths = photos(tmp_path, 1)
    made = []

    def start():
        (tmp_path / "t").mkdir(exist_ok=True)
        time.sleep(0.05)  # `docker run` takes a moment
        made.append(LocalTarget(tmp_path / "t", warm_delay=0.4))
        return made[0]

    try:
        _, cold = e8.cold_start(start, e8.work_list(paths)[0], tmp_path / "imgs", poll=0.01)
    finally:
        made[0].stop()
    assert cold["container_start_ms"] >= 50
    assert cold["to_alive_ms"] >= cold["container_start_ms"] and cold["to_ready_ms"] >= cold["to_alive_ms"]
    assert cold["to_ready_ms"] >= 400 - 1  # not ready while the index is still warming
    assert cold["total_ms"] >= cold["to_ready_ms"] + cold["first_request_ms"] - 1
    assert cold["first_request"]["answer_state"] == "answered" and cold["mem_mb"] > 200


def test_a_container_that_never_becomes_ready_is_an_error_not_a_number(tmp_path):
    class Never:
        base_url = f"http://127.0.0.1:{e8.free_port()}"

    with pytest.raises(RuntimeError, match="not ready"):
        e8.cold_start(lambda: Never(), ("p/0.png", "q"), tmp_path, ready_timeout=0.3, poll=0.01)


# ---------------------------------------------------------------- the docker commands

class FakeRun:
    def __init__(self, replies=None):
        self.calls, self.replies = [], replies or {}

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(cmd)
        for key, (code, out) in self.replies.items():
            if key in " ".join(cmd):
                return subprocess.CompletedProcess(cmd, code, out, "boom" if code else "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


INSPECT = json.dumps([{"Config": {"Image": "sprout-app", "Entrypoint": ["/app/deploy/entrypoint.sh"],
                                  "Env": ["SPROUT_WORKERS=2", "SPROUT_SECRET_KEY=hunter2", "SPROUT_MEASUREMENT=1", "PATH=/x"]},
                       "HostConfig": {"NanoCpus": 2000000000, "Memory": 4294967296, "CpuQuota": 0}}])


def test_the_container_runs_with_its_own_entrypoint_and_the_measurement_setting():
    run = FakeRun()
    t = e8.DockerTarget("sprout-app", port=8123, env={"SPROUT_SECRET_KEY": "s"}, mounts=["/m/model.onnx:/models/model.onnx:ro"],
                        docker_args=["--cpus=2"], run=run, name="e8-test")
    t.start()
    cmd = run.calls[0]
    assert cmd[:3] == ["docker", "run", "-d"] and "--entrypoint" not in cmd and cmd[-1] == "sprout-app"
    assert "SPROUT_MEASUREMENT=1" in cmd and "--cpus=2" in cmd and "127.0.0.1:8123:8080" in cmd
    assert "host.docker.internal:host-gateway" in cmd and "/m/model.onnx:/models/model.onnx:ro" in cmd
    t.stop()
    assert run.calls[-1] == ["docker", "rm", "-f", "e8-test"]
    t.stop()
    assert len(run.calls) == 2  # only a container this object started is removed, once


def test_container_memory_comes_from_its_cgroup_and_a_failed_read_is_an_error():
    run = FakeRun({"memory.current": (0, "314572800\n"), "memory.peak": (0, "419430400\n")})
    t = e8.DockerTarget("i", port=1, env={}, mounts=[], docker_args=[], run=run, name="n")
    assert t.mem_mb() == 300.0 and t.mem_peak_mb() == 400.0
    assert run.calls[0] == ["docker", "exec", "n", "cat", "/sys/fs/cgroup/memory.current"]
    broken = e8.DockerTarget("i", port=1, env={}, mounts=[], docker_args=[], run=FakeRun({"cat": (1, "")}), name="n")
    with pytest.raises(RuntimeError, match="cgroup"):
        broken.mem_mb()
    assert broken.mem_peak_mb() is None


def test_server_settings_and_docker_limits_are_read_from_the_container_and_secrets_are_not_kept():
    run = FakeRun({"inspect": (0, INSPECT), "top": (0, "ARGS\ngunicorn: master\ngunicorn: worker\ngunicorn: worker\n")})
    info = e8.DockerTarget("i", port=1, env={}, mounts=[], docker_args=[], run=run, name="n").info()
    assert (info["workers"], info["threads"], info["gunicorn_timeout_s"]) == (2, 4, 60)  # entrypoint defaults where unset
    assert info["gunicorn_processes_running"] == 3
    assert info["docker_limits"]["NanoCpus"] == 2000000000 and info["docker_limits"]["Memory"] == 4294967296
    assert "SPROUT_SECRET_KEY" not in info["env"] and info["env"]["SPROUT_MEASUREMENT"] == "1" and "PATH" not in info["env"]


# ---------------------------------------------------------------- the record

N = 5


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
    (tmp / "t").mkdir()
    paths = photos(tmp, N)
    t = LocalTarget(tmp / "t", llm=SlowLLM(seconds=0.0))
    try:
        _, cold = e8.cold_start(lambda: t, e8.work_list(paths)[0], tmp / "imgs", poll=0.01)
        run = e8.measure(t, e8.work_list(paths), tmp / "imgs", interval=0.01, timeout=30)
    finally:
        t.stop()
    return {"format": 2, "started_at": 1.7e9, "n_images": N, "warmup": 1, "questions": list(e8.QUESTIONS),
            "model_sha256": "ab" * 32, "calibration_sha256": "cd" * 32, "model_file": "tiny.onnx", "writer": "tiny-writer",
            "ollama_url": "http://host.docker.internal:11434", "client": {"machine": {"logical_cpus": 4}, "timeout_s": 30},
            "server": {"workers": 1, "threads": 4, "docker_limits": {"NanoCpus": 0}}, "cold": cold, "run": run,
            "ollama_placement": {"size": 1, "size_vram": 1}, "measurement_mode": "SPROUT_MEASUREMENT=1"}


def record(root, data, **kw):
    args = dict(machine="test box, 4 cores", image="sprout-app:test", variant="baseline", root=root)
    return e8.build_record({**data}, **{**args, **kw})


def test_a_tiny_run_writes_a_record_that_check_run_accepts(root, measured):
    path = record(root, measured)
    assert path.name.startswith("E8-")
    assert check("E8", root=root, min_n=N, split="perf") == []
    rec = json.loads(path.read_text())
    assert rec["n"] == N and rec["split"] == "perf" and set(rec["predictions"]) == {"requests"}
    assert rec["config"]["machine_stated"] == "test box, 4 cores" and rec["config"]["server"]["threads"] == 4
    assert rec["config"]["outcomes"]["failed"] == 0 and "Failed requests" in rec["notes"]
    assert "no concurrency or speed claim" in rec["notes"] and rec["config"]["requests"] == N


def test_percentiles_by_stage_are_recorded(root, measured):
    rec = json.loads(record(root, measured).read_text())
    values, samples = rec["measurements"]["values"], json.loads((root / rec["measurements"]["file"]).read_text())
    for stage in e8.REQUEST_STAGES:
        for q in (50, 95):
            assert values[f"request_{stage}_p{q}"] == pytest.approx(float(np.percentile(samples[f"request_{stage}"], q)))
    assert samples["request_peak_mem_mb"] and "request_mem_mb_p95" in values and values["request_failed_share_p50"] == 0
    for stage in e8.COLD_STAGES:
        assert f"cold_start_{stage}_p50" in values
    assert values["request_end_to_end_ms_p95"] >= values["request_end_to_end_ms_p50"]


def test_a_run_with_a_failed_request_is_refused_not_recorded(root, measured):
    run = measured["run"]
    rows = [{**r, "answer_state": "client_timeout"} if i == 0 else r for i, r in enumerate(run["rows"])]
    with pytest.raises(ValueError, match="1 of 5 failed.*every request to answer"):
        record(root, {**measured, "run": {**run, "rows": rows}})


def test_extra_cold_start_samples_are_added(root, measured):
    extra = {"cold": {**measured["cold"], "total_ms": 5.0}}
    rec = json.loads(record(root, measured, cold_files=[extra]).read_text())
    samples = json.loads((root / rec["measurements"]["file"]).read_text())
    assert len(samples["cold_start_total_ms"]) == 2 and rec["config"]["cold_start_samples"] == 2


def test_tampered_records_fail_the_check(root, measured):
    path = record(root, measured)
    original = path.read_text()
    rec = json.loads(original)
    rec["measurements"]["values"]["request_end_to_end_ms_p95"] = 1.0  # a faster number than the samples give
    path.write_text(json.dumps(rec))
    assert any("does not match its saved file" in p for p in check("E8", root=root, min_n=N, split="perf"))
    path.write_text(original)
    assert check("E8", root=root, min_n=N, split="perf") == []

    samples = root / rec["measurements"]["file"]
    saved = samples.read_text()
    data = json.loads(saved)
    lowest = data["request_end_to_end_ms"].index(min(data["request_end_to_end_ms"]))
    data["request_end_to_end_ms"][lowest] += 1000.0  # a sample changed after the summary was written (moves p50 and p95)
    samples.write_text(json.dumps(data))
    assert any("does not match its saved file" in p for p in check("E8", root=root, min_n=N, split="perf"))
    samples.write_text(saved)

    data = json.loads(saved)
    data["request_failed_share"] = [0.5]  # a failure share changed after the summary was written
    samples.write_text(json.dumps(data))
    assert any("does not match its saved file" in p for p in check("E8", root=root, min_n=N, split="perf"))
    samples.write_text(saved)

    rows = root / rec["predictions"]["requests"]
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
    with pytest.raises(ValueError, match="cold-start file"):
        record(root, {**measured, "run": None})
    run = measured["run"]
    shuffled = {**measured, "run": {**run, "rows": run["rows"][::-1]}}
    with pytest.raises(ValueError, match="once each, in order"):
        record(root, shuffled)


def test_a_stage_that_never_ran_is_refused_not_recorded_as_empty(root, measured):
    run = measured["run"]
    rows = [{**r, "generation_ms": None} for r in run["rows"]]
    broken = {**measured, "run": {**run, "rows": rows}}
    with pytest.raises(ValueError, match="generation_ms stage"):
        record(root, broken)


def test_the_image_list_is_spread_over_the_dev_photos(tmp_path):
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    rows = ["path,label,group,split,sha1,source"] + [
        f"raw/{i:03d}.jpg,A,g{i},field_{'dev' if i % 2 else 'test'},s{i},x" for i in range(400)]
    (manifests / "field.csv").write_text("\n".join(rows) + "\n")
    out = e8.write_image_list(tmp_path, 5)
    lines = out.read_text().splitlines()
    assert len(lines) == 5 and len(set(lines)) == 5 and lines == sorted(lines)
    assert all(int(x[4:7]) % 2 == 1 for x in lines)  # dev photos only
    assert split_info("perf", tmp_path).ids == lines
    with pytest.raises(ValueError, match="at least 5"):
        e8.write_image_list(tmp_path, 4)


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


# ---------------------------------------------------------------- start-up errors while waiting for readiness

def test_a_port_that_accepts_and_hangs_up_means_not_ready_yet(tmp_path):
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    stop = threading.Event()

    def hang_up():
        server.settimeout(0.05)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            conn.close()  # Docker's port is open, the app is not: the client sees RemoteDisconnected

    thread = threading.Thread(target=hang_up, daemon=True)
    thread.start()

    class Booting:
        base_url = f"http://127.0.0.1:{server.getsockname()[1]}"

    try:
        with pytest.raises(RuntimeError, match="not ready"):
            e8.cold_start(lambda: Booting(), ("p/0.png", "q"), tmp_path, ready_timeout=0.5, poll=0.01)
    finally:
        stop.set()
        thread.join()
        server.close()


def test_a_request_that_loses_its_connection_is_a_counted_failure_not_an_exception():
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    thread = threading.Thread(target=lambda: server.accept()[0].close(), daemon=True)
    thread.start()
    status, _, _ = e8.HttpClient(f"http://127.0.0.1:{server.getsockname()[1]}", 2.0).request("GET", "/x")
    thread.join()
    server.close()
    assert status == 0


def test_a_container_that_exited_fails_at_once_with_its_last_log_lines(tmp_path):
    class Dead:
        base_url = f"http://127.0.0.1:{e8.free_port()}"

        def failure(self):
            return "exited with 1; last log lines:\nValueError: model_sha256 does not match"

    begin = time.time()
    with pytest.raises(RuntimeError, match="(?s)exited.*model_sha256 does not match"):
        e8.cold_start(lambda: Dead(), ("p/0.png", "q"), tmp_path, ready_timeout=60, poll=0.01)
    assert time.time() - begin < 5


def test_docker_target_reports_an_exited_container_with_its_logs_and_keeps_it_until_stopped():
    run = FakeRun({"inspect": (0, "false 1\n"), "logs": (0, "boom line\n")})
    t = e8.DockerTarget("sprout-app", port=8123, env={}, mounts=[], docker_args=[], run=run, name="e8-test")
    t.start()
    assert "--rm" not in run.calls[0]  # an auto-removed container would take its logs with it
    message = t.failure()
    assert "exited" in message and "boom line" in message
    assert run.calls[-1][:2] == ["docker", "logs"]
    running = FakeRun({"inspect": (0, "true 0\n")})
    t2 = e8.DockerTarget("sprout-app", port=8124, env={}, mounts=[], docker_args=[], run=running, name="e8-ok")
    assert t2.failure() is None


# ---------------------------------------------------------------- the disease-name fallback check

from sprout.experiments import e8_fallback as fb  # noqa: E402


def test_the_right_disease_rule_reads_the_passage_text_and_known_aliases():
    assert fb.right_disease("Early-blight shows as rings.", "Potato___Early_blight")
    assert not fb.right_disease("Late blight rots tubers.", "Potato___Early_blight")
    assert fb.right_disease("Gray leaf spot of maize", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    assert fb.right_disease("Huanglongbing is spread by psyllids", "Orange___Haunglongbing_(Citrus_greening)")
    assert not fb.right_disease("anything", "Tomato___healthy")


def test_the_follow_up_set_has_every_wording_for_every_disease_class_and_none_for_healthy():
    rows = fb.followups(["Tomato___healthy", "Tomato___Early_blight", "Potato___Late_blight"])
    assert len(rows) == 2 * len(fb.WORDINGS) and {r["label"] for r in rows} == {"Tomato___Early_blight", "Potato___Late_blight"}
    assert len({r["id"] for r in rows}) == len(rows)


def test_the_generic_follow_up_report_counts_only_passages_about_the_right_disease():
    class Fake:
        crops = {"tomato"}

        def search(self, query, *, crop=None, **kw):
            if "early blight" not in query:
                return []
            return [Passage("Early blight rings", "t.pdf", 1, "tomato", 0.9)]

    rows = [{"id": "g1", "label": "Tomato___Early_blight", "question": "How do I treat it?"},
            {"id": "g2", "label": "Tomato___Late_blight", "question": "How do I treat it?"}]
    out = fb.generic_followup_report(Fake(), rows, fb.AnswerConfig())
    s = out["summary"]
    assert (s["found_any_before"], s["found_any_after"], s["right_disease_before"], s["right_disease_after"]) == (0, 1, 0, 1)
    assert out["rows"][0]["searches"] == 2 and out["rows"][0]["crop"] == "tomato"


def test_the_dev_report_lists_the_question_whose_decline_the_fallback_changes():
    from sprout.rag.evalset import Question

    class Fake:
        crops = {"potato"}

        def search(self, query, *, crop=None, **kw):
            return [Passage("Late blight", "p.pdf", 1, "potato", 0.9)] if query.count("late blight") == 2 else []

    qs = [Question(id="d1", category="answerable", split="dev", question="Is late blight bad?", expected="answer", crop="potato"),
          Question(id="d2", category="unanswerable", split="dev", question="Why is my plant sad?", expected="decline")]
    classes = ["Potato___Late_blight", "Potato___Early_blight"]
    out = fb.dev_report(Fake(), qs, classes, fb.AnswerConfig())
    assert [c["id"] for c in out["questions_whose_decline_changes"]] == ["d1"]
    assert out["rates"]["question_only"]["answers_when_answerable"] == 0.0
    assert out["rates"]["with_fallback"]["answers_when_answerable"] == 1.0
