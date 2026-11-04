"""E8: a handful of real requests (about 5) through the serving container on the CPU, to show the parts work together.

This proves the real model, calibration, index and writer answer through the deployed image. The record keeps the stage
times it saw, but with so few requests it supports no concurrency claim and no end-to-end speed claim.

Two halves, because the load client needs nothing from the image and the record needs pandas:
  - `measure` and `cold` run on the HOST (standard library only). They start the serving image exactly as
    docs/deploy.md deploys it (its own entrypoint, gunicorn with its worker and thread counts), plus
    `SPROUT_MEASUREMENT=1`, which lifts the rate limits and makes the upload and chat routes report stage times in a
    `Server-Timing` header. The client sends real HTTP requests one after another, each one user action: upload a photo
    and ask a question about it. Memory is read from the container's cgroup. Raw per-request numbers go to a JSON file.
  - `record` turns that file into a run record with `write_run`, so `check_run E8` can check it.
`floor-check` and `multi-citations` are host-side checks that call no model.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path

from sprout.atomic import atomic_write_text
from sprout.serve import timing

SPLIT = "perf"
MIN_REQUESTS = 5
REQUEST_STAGES = ("preprocess_ms", "inference_ms", "retrieval_ms", "generation_ms", "end_to_end_ms")
# Measured from outside, from the moment `docker run` is issued.
COLD_STAGES = ("container_start_ms", "to_alive_ms", "to_ready_ms", "first_request_ms", "total_ms")
# A request that failed (writer or search unavailable, upload or chat error, a client timeout) is counted and kept out of
# the percentiles, but stays in the record.
FAILED_STATES = ("unavailable", "retrieval_failed", "upload_failed", "chat_failed", "client_timeout", "connection_failed")
CLIENT_TIMEOUT_S = 120.0  # longer than the app's 50 s answer budget, so the app's own timeouts show up as its replies
MEM_INTERVAL_S = 0.5
QUESTIONS = (
    "What is wrong with my plant and how do I treat it?",
    "How can I stop this from spreading to my other plants?",
    "What causes this and how do I prevent it next season?",
    "Is it safe to eat the fruit from a plant with this?",
)
NEAR_FLOOR = 0.05  # a reranker score within this of the floor counts as near it
SERVED_K, E7_K = 5, 10  # the k the app serves with, and the k E7 picked the floors at
ENTRYPOINT_DEFAULTS = {"SPROUT_WORKERS": "1", "SPROUT_THREADS": "4", "SPROUT_TIMEOUT": "60"}  # deploy/entrypoint.sh


def now_ms() -> float:
    return time.perf_counter() * 1000


# ---------------------------------------------------------------- the load client

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # the upload's 302 carries the timing header and the new analysis id


class HttpClient:
    """One signed-in user talking to the server over HTTP, with its own cookies."""

    def __init__(self, base: str, timeout: float = CLIENT_TIMEOUT_S) -> None:
        self.base, self.timeout, self.token = base.rstrip("/"), timeout, ""
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()), _NoRedirect())

    def request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None):
        """(status, headers, body). Status 0: no usable answer came back (refused, reset or hung up). Raises TimeoutError when the server did not answer in time."""
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            r = self._opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:  # includes the 302 the client refuses to follow
            r = e
        except (TimeoutError, socket.timeout) as e:
            raise TimeoutError(str(e)) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(str(e)) from e
            return 0, {}, b""
        except (ConnectionError, http.client.HTTPException):  # port open but the server hung up (RemoteDisconnected)
            return 0, {}, b""
        try:
            body = r.read()
        except (TimeoutError, socket.timeout) as e:
            raise TimeoutError(str(e)) from e
        except (ConnectionError, http.client.HTTPException):
            return 0, {}, b""
        return r.status, {k.lower(): v for k, v in r.headers.items()}, body

    def refresh_token(self, page: str = "/login") -> None:
        _, _, body = self.request("GET", page)
        found = re.search(rb'name="csrf-token" content="([^"]+)"', body)
        if not found:
            raise RuntimeError("the server's login page has no CSRF token")
        self.token = found.group(1).decode()

    def sign_up(self, name: str | None = None) -> None:
        self.refresh_token()
        form = urllib.parse.urlencode({"username": name or f"perf-{uuid.uuid4().hex[:12]}", "password": "password123",
                                       "csrf_token": self.token}).encode()
        status, _, _ = self.request("POST", "/signup", form, {"Content-Type": "application/x-www-form-urlencoded"})
        if status not in (200, 302):
            raise RuntimeError(f"sign-up answered {status}")
        self.refresh_token("/upload")  # signing in rotates it, and /login now redirects

    def _upload(self, filename: str, data: bytes):
        boundary = uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="csrf_token"\r\n\r\n{self.token}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", "/analyses", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})

    def user_action(self, path: str, data: bytes, question: str) -> dict:
        """Upload the photo, then ask the question about it. One row; its end-to-end time is the client's clock."""
        start = now_ms()
        stages: dict[str, float] = {}
        status_text, state = "unknown", "chat_failed"
        analysis = None
        try:
            status, headers, _ = self._upload(Path(path).name, data)
            first = timing.parse_header(headers.get("server-timing", ""))
            status_text = first.pop("ident", f"http_{status}")
            for k, v in first.items():
                stages[k] = stages.get(k, 0.0) + v
            match = re.search(r"/analyses/(\d+)", headers.get("location", "")) if status == 302 else None
            if match is None:
                state = "connection_failed" if status == 0 else "upload_failed"
            else:
                analysis = match.group(1)
                status, headers, body = self.request(
                    "POST", f"/analyses/{analysis}/chat", json.dumps({"message": question}).encode(),
                    {"Content-Type": "application/json", "X-CSRF-Token": self.token})
                second = timing.parse_header(headers.get("server-timing", ""))
                second.pop("ident", None)
                for k, v in second.items():
                    stages[k] = stages.get(k, 0.0) + v
                if status == 200:
                    state = json.loads(body or b"{}").get("state", "chat_failed")
                else:
                    state = "refused" if status == 409 else ("connection_failed" if status == 0 else "chat_failed")
        except TimeoutError:
            state = "client_timeout"
        total = now_ms() - start
        if analysis is not None:  # not part of the action; keeps the server's disk from filling
            try:
                self.request("POST", f"/analyses/{analysis}/delete",
                             urllib.parse.urlencode({"csrf_token": self.token}).encode(),
                             {"Content-Type": "application/x-www-form-urlencoded"})
            except TimeoutError:
                pass
        row = {"path": path, "identification": status_text, "answer_state": state}
        for name in REQUEST_STAGES[:-1]:
            ms = stages.get(name.removesuffix("_ms"))
            row[name] = round(ms, 3) if ms is not None else None  # None: the stage did not run
        row["end_to_end_ms"] = round(total, 3)
        return row


# ---------------------------------------------------------------- the container

class DockerTarget:
    """The serving image run with its own entrypoint, as deployed, plus the measurement setting."""

    def __init__(self, image: str, *, port: int, env: dict[str, str], mounts: list[str], docker_args: list[str],
                 run=subprocess.run, name: str | None = None) -> None:
        self.image, self.port, self.env, self.mounts, self.docker_args = image, port, dict(env), mounts, docker_args
        self.name = name or f"sprout-e8-{uuid.uuid4().hex[:8]}"
        self._run = run
        self.base_url = f"http://127.0.0.1:{port}"
        self.started = False

    def _docker(self, *args: str, check: bool = True):
        out = self._run(["docker", *args], capture_output=True, text=True)
        if check and out.returncode != 0:
            raise RuntimeError(f"docker {args[0]} failed: {out.stderr.strip()[:300]}")
        return out

    def start(self) -> DockerTarget:
        env = {**self.env, "SPROUT_MEASUREMENT": "1"}
        cmd = ["run", "-d", "--name", self.name, "-p", f"127.0.0.1:{self.port}:8080",
               "--add-host", "host.docker.internal:host-gateway", *self.docker_args]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        for m in self.mounts:
            cmd += ["-v", m]
        self._docker(*cmd, self.image)  # no --entrypoint: the image's own gunicorn entrypoint runs
        self.started = True
        return self

    def _cgroup(self, *names: str) -> str | None:
        for n in names:
            out = self._docker("exec", self.name, "cat", n, check=False)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return out.stdout.strip()
        return None

    def mem_mb(self) -> float:
        """The whole container's memory now (every gunicorn worker), from its cgroup."""
        value = self._cgroup("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if value is None:
            raise RuntimeError("the container's memory cannot be read from its cgroup")
        return int(value) / 1048576

    def mem_peak_mb(self) -> float | None:
        value = self._cgroup("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        return int(value) / 1048576 if value else None

    def info(self) -> dict:
        """Docker limits and gunicorn's settings, as the container really has them."""
        data = json.loads(self._docker("inspect", self.name).stdout)[0]
        env = dict(e.split("=", 1) for e in data["Config"]["Env"] if "=" in e)
        host = data["HostConfig"]
        shown = {k: v for k, v in env.items() if k.startswith("SPROUT_") and "SECRET" not in k}
        top = self._docker("top", self.name, "-eo", "args", check=False).stdout
        return {
            "image": data["Config"]["Image"], "entrypoint": data["Config"].get("Entrypoint"),
            "env": shown, "workers": int(env.get("SPROUT_WORKERS", ENTRYPOINT_DEFAULTS["SPROUT_WORKERS"])),
            "threads": int(env.get("SPROUT_THREADS", ENTRYPOINT_DEFAULTS["SPROUT_THREADS"])),
            "gunicorn_timeout_s": int(env.get("SPROUT_TIMEOUT", ENTRYPOINT_DEFAULTS["SPROUT_TIMEOUT"])),
            "gunicorn_processes_running": sum("gunicorn" in line for line in top.splitlines()),
            "docker_limits": {k: host.get(k) for k in ("NanoCpus", "CpuQuota", "CpuPeriod", "CpusetCpus", "Memory", "MemorySwap")},
        }

    def failure(self) -> str | None:
        """None while the container runs; otherwise what it died with, its last log lines read before `stop` removes it."""
        state = self._docker("inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", self.name, check=False)
        if state.returncode != 0:
            return "the container is gone"
        running, _, code = state.stdout.strip().partition(" ")
        if running == "true":
            return None
        logs = self._docker("logs", "--tail", "30", self.name, check=False)
        return f"the container exited with code {code}; last log lines:\n{(logs.stdout + logs.stderr).strip()}"

    def stop(self) -> None:
        if self.started:
            self._docker("rm", "-f", self.name, check=False)
            self.started = False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- memory

class MemorySampler:
    """Memory of the container, sampled while the requests run. A failed read stops the run instead of making up a number."""

    def __init__(self, read, interval: float) -> None:
        self._read, self._interval = read, interval
        self.samples: list[float] = []
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                self.samples.append(self._read())
        except Exception as exc:  # noqa: BLE001 - re-raised on exit
            self._error = exc

    def __enter__(self) -> MemorySampler:
        self.samples.append(self._read())
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"memory sampling failed: {self._error}") from self._error
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
    """What the load client can see (the host, or a Linux VM). The person who ran the stage states the real machine."""
    return {"platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count(),
            "cpu_model": _first_line("/proc/cpuinfo", "model name") or platform.processor(),
            "memory_total": _first_line("/proc/meminfo", "MemTotal")}


# ---------------------------------------------------------------- measuring

def measure(target, work: list[tuple[str, str]], images_root: Path, *, warmup: int = 1,
            interval: float = MEM_INTERVAL_S, timeout: float = CLIENT_TIMEOUT_S) -> dict:
    """Every image once, one request at a time, after `warmup` unrecorded requests. Rows come back in `work` order."""
    client = HttpClient(target.base_url, timeout)
    client.sign_up()
    for path, question in work[:warmup]:  # not recorded: the first request after start-up is the cold start
        client.user_action(path, (images_root / path).read_bytes(), question)
    start = time.perf_counter()
    with MemorySampler(target.mem_mb, interval) as sampler:
        rows = [client.user_action(path, (images_root / path).read_bytes(), question) for path, question in work]
    wall = time.perf_counter() - start
    return {"wall_s": round(wall, 3), "rows": rows, "mem_mb": [round(x, 2) for x in sampler.samples],
            "peak_mem_mb": round(max(sampler.samples), 2), "cgroup_peak_mb": target.mem_peak_mb()}


def cold_start(start_target, first: tuple[str, str], images_root: Path, *, ready_timeout: float = 300.0,
               poll: float = 0.05, timeout: float = CLIENT_TIMEOUT_S):
    """From the moment `docker run` is issued to `/readyz` saying ready (the model loaded and the index warm), then the
    first request. Everything is timed from outside the container."""
    begin = now_ms()
    target = start_target()
    started = now_ms()
    probe = HttpClient(target.base_url, 5.0)
    alive = ready = None
    last = "no answer"
    while now_ms() - begin < ready_timeout * 1000:
        dead = getattr(target, "failure", lambda: None)()
        if dead:
            raise RuntimeError(dead)
        try:
            if alive is None and probe.request("GET", "/healthz")[0] == 200:
                alive = now_ms()
            if alive is not None:
                status, _, body = probe.request("GET", "/readyz")
                last = body.decode(errors="replace")[:200]
                ready_body = json.loads(body or b"{}")
                if status == 200 and ready_body.get("ready") and ready_body.get("chat") == "ready":
                    ready = now_ms()
                    break
        except (TimeoutError, ValueError):  # no answer in time, or a body that is not the readiness JSON yet
            pass
        time.sleep(poll)
    if ready is None:
        raise RuntimeError(f"the container was not ready after {ready_timeout:.0f} s; last /readyz: {last}")
    client = HttpClient(target.base_url, timeout)
    client.sign_up()
    row = client.user_action(first[0], (images_root / first[0]).read_bytes(), first[1])
    done = now_ms()
    if row["answer_state"] in FAILED_STATES:
        raise RuntimeError(f"the first request failed ({row['answer_state']}); this is not a result")
    cold = {"container_start_ms": started - begin, "to_alive_ms": alive - begin, "to_ready_ms": ready - begin,
            "first_request_ms": done - ready, "total_ms": done - begin, "mem_mb": target.mem_mb(), "first_request": row}
    return target, {k: round(v, 3) if isinstance(v, float) else v for k, v in cold.items()}


def work_list(paths: list[str]) -> list[tuple[str, str]]:
    return [(p, QUESTIONS[i % len(QUESTIONS)]) for i, p in enumerate(paths)]


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_list(path: Path) -> list[str]:
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def run_measurement(args: argparse.Namespace, *, requests: bool = True) -> dict:
    """Start the container, measure its cold start, then the requests; always stop the container this call started."""
    from sprout.llm.client import LLMError, OllamaClient

    started = time.time()
    root = Path(args.images_root)
    paths = read_list(args.images)
    work = work_list(paths)
    env = {"SPROUT_SECRET_KEY": secrets.token_urlsafe(32), "SPROUT_OLLAMA_URL": args.ollama_url,
           "SPROUT_WRITER_MODEL": args.writer, "SPROUT_INDEX_DIR": "/models/index",
           "FASTEMBED_CACHE_PATH": "/models/fastembed", **dict(e.split("=", 1) for e in args.env)}
    mounts = [f"{Path(args.model).resolve()}:/models/model.onnx:ro", f"{Path(args.calibration).resolve()}:/models/chosen.json:ro",
              f"{Path(args.index).resolve()}:/models/index:ro", f"{Path(args.fastembed).resolve()}:/models/fastembed:ro"]
    target = DockerTarget(args.image, port=args.port or free_port(), env=env, mounts=mounts, docker_args=list(args.docker_arg))
    try:
        _, cold = cold_start(target.start, work[0], root)
        out = {
            "format": 3, "started_at": started, "n_images": len(paths), "warmup": 1, "questions": list(QUESTIONS),
            "model_sha256": sha256_file(Path(args.model)), "calibration_sha256": sha256_file(Path(args.calibration)),
            "model_file": Path(args.model).name, "writer": args.writer, "ollama_url": args.ollama_url,
            "client": {"machine": machine_info(), "timeout_s": CLIENT_TIMEOUT_S, "memory_interval_s": MEM_INTERVAL_S},
            "server": target.info(), "cold": cold, "run": None,
            "measurement_mode": "SPROUT_MEASUREMENT=1: rate limits lifted, Server-Timing header on the upload and chat routes",
        }
        if requests:
            out["run"] = measure(target, work, root)
        try:
            out["ollama_placement"] = OllamaClient(args.writer, base_url=args.ollama_host_url).placement()
        except LLMError:
            out["ollama_placement"] = None
        return out
    finally:
        target.stop()


# ---------------------------------------------------------------- the record

def _stage_samples(rows: list[dict], stage: str) -> list[float]:
    """Times of requests that did not fail and reached the stage."""
    return [r[stage] for r in rows if r[stage] is not None and r["answer_state"] not in FAILED_STATES]


def build_measurements(data: dict, cold_files: list[dict]) -> tuple[dict[str, list[float]], dict]:
    """Raw samples under the names `check_run` recomputes percentiles from, and what happened to every request."""
    samples: dict[str, list[float]] = {}
    outcomes: dict[str, dict] = {}
    colds = [data["cold"], *(f["cold"] for f in cold_files)]
    for name in COLD_STAGES:
        samples[f"cold_start_{name}"] = [c[name] for c in colds]
    samples["cold_start_mem_mb"] = [c["mem_mb"] for c in colds]
    block = data["run"]
    rows = block["rows"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["answer_state"]] = counts.get(r["answer_state"], 0) + 1
    failed = sum(r["answer_state"] in FAILED_STATES for r in rows)
    outcomes = {"requests": len(rows), "failed": failed, "failed_share": round(failed / len(rows), 4),
                "client_timeouts": counts.get("client_timeout", 0), "by_answer_state": counts, "ran_stage": {}}
    for stage in REQUEST_STAGES:
        got = _stage_samples(rows, stage)
        if not got:
            raise ValueError(f"no request ran the {stage} stage, so it has no number to record")
        samples[f"request_{stage}"] = got
        outcomes["ran_stage"][stage] = len(got)
    samples["request_mem_mb"] = block["mem_mb"]
    samples["request_peak_mem_mb"] = [block["peak_mem_mb"]]
    samples["request_failed_share"] = [outcomes["failed_share"]]
    return samples, outcomes


def failure_summary(outcomes: dict) -> str:
    """So a run that mostly failed cannot look fast."""
    return (f"Failed requests, kept out of the percentiles: {outcomes['failed']} of {outcomes['requests']} failed "
            f"({outcomes['client_timeouts']} client timeouts).")


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
    if not data.get("run"):
        raise ValueError("the measurement file has no requests; it is a cold-start file, not a measurement")
    info = split_info(SPLIT, root)
    samples, ran = build_measurements(data, cold_files or [])
    if ran["failed"]:
        raise ValueError(f"{failure_summary(ran)} A record that shows the parts work together needs every request to answer")
    rows = data["run"]["rows"]
    if [r["path"] for r in rows] != info.ids:
        raise ValueError("the requests are not the perf list's images, once each, in order")
    systems = {"requests": [{"path": r["path"], "identification": r["identification"], "answer_state": r["answer_state"]}
                            for r in rows]}
    config = {
        "machine_stated": machine.strip(), "image": image.strip(), "variant": variant.strip(),
        "load_client": data["client"], "server": data["server"],
        "requests": data["n_images"], "outcomes": ran, "warmup_requests": data["warmup"],
        "questions": data["questions"], "measurement_mode": data["measurement_mode"],
        "model_sha256": data["model_sha256"], "calibration_sha256": data["calibration_sha256"],
        "ollama_url": data["ollama_url"], "ollama_placement": data["ollama_placement"],
        "cold_start_samples": 1 + len(cold_files or []),
        "device": "container on CPU (onnxruntime CPUExecutionProvider, no torch); the writer runs in Ollama on the host",
        "claims": "none about concurrency or end-to-end speed: the requests run one at a time and there are too few for "
                  "percentiles to mean anything. The record shows the real model, calibration, index and writer answered "
                  "together through the deployed image.",
        "memory": "the whole container's memory from its cgroup, sampled every 0.5 s, peak and samples over the requests",
        "stage_times": "from the Server-Timing header (hooks on the real objects); failed and timed-out requests are "
                       "counted in outcomes and request_failed_share, and kept out of every percentile",
        "started_at": datetime.fromtimestamp(data["started_at"], timezone.utc).isoformat(),
        "labels_reviewed": False, "claude_calls": 0,
    }
    models = {"classifier": f"{data['model_file']} sha256 {data['model_sha256'][:12]}", "writer": data["writer"],
              "embedder": "BAAI/bge-small-en-v1.5 (fastembed ONNX)", "reranker": "Xenova/ms-marco-MiniLM-L-6-v2 (fastembed ONNX)"}
    return write_run(
        "E8", SPLIT, config, models, {s: {} for s in systems}, {},
        [{"setting": variant.strip(), "result": "measured", "requests": len(rows)}], systems, info.hash,
        root=root, allow_dirty=allow_dirty, measurements=samples,
        notes="Latency and memory of the deployed app (real Identifier, index and writer) in its container on the CPU, "
              "over HTTP, one request at a time. Requests are the perf list's photos with a fixed question each. Nothing here "
              "is accuracy, and the few requests support no concurrency or speed claim. "
              + failure_summary(ran))


# ---------------------------------------------------------------- the image list

def write_image_list(root: Path, n: int, split: str = "field_dev") -> Path:
    """datasets/perf/images.txt: `n` photos spread evenly over the sorted paths of a real-photo dev split.
    Only paths are listed; labels are never read for timing."""
    import csv

    import numpy as np

    if n < MIN_REQUESTS:
        raise ValueError(f"the perf list needs at least {MIN_REQUESTS} photos for check_run --min-n {MIN_REQUESTS}")
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
    from dataclasses import replace

    import numpy as np

    from sprout.eval.check_run import Scored, recompute_metric
    from sprout.experiments.e7_rag import BEHAVIOUR_METRICS, simulate_identification
    from sprout.rag import serving
    from sprout.rag.answer import AnswerConfig, bounded_history, retrieve

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
        for k in (SERVED_K, E7_K):
            before = len(recorder.calls)
            # the same search answer() makes, including its disease-name second search
            found, _, _ = retrieve(q.question, identification=ident, history=bounded_history(q.history, cfg),
                                   retriever=retriever, cfg=replace(cfg, k=k, mode=serving.MODE, rerank=True), crop=crop)
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
    ap.add_argument("--image", required=True, help="the serving image, built by docker compose build app")
    ap.add_argument("--model", required=True, help="the ONNX classifier (E12 picks it); mounted at /models/model.onnx")
    ap.add_argument("--calibration", required=True, help="results/e5/chosen.json; mounted at /models/chosen.json")
    ap.add_argument("--index", required=True, help="the retrieval index folder, with its vectors cache; /models/index")
    ap.add_argument("--fastembed", required=True, help="a folder already holding the fastembed models; /models/fastembed")
    ap.add_argument("--images", required=True, help="datasets/perf/images.txt")
    ap.add_argument("--images-root", required=True, help="folder the paths in --images are relative to (read on the host)")
    ap.add_argument("--writer", required=True, help="the Ollama model that writes answers")
    ap.add_argument("--ollama-url", default="http://host.docker.internal:11434", help="where the container reaches Ollama")
    ap.add_argument("--ollama-host-url", default="http://127.0.0.1:11434", help="where this host reaches Ollama")
    ap.add_argument("--port", type=int, default=0, help="host port for the container (default: a free one)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra container setting, e.g. SPROUT_WORKERS=2")
    ap.add_argument("--docker-arg", action="append", default=[], metavar="ARG", help="extra `docker run` argument, e.g. --docker-arg=--cpus=2")
    ap.add_argument("--out", required=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e8_perf")
    sub = ap.add_subparsers(dest="stage", required=True)
    _container_args(sub.add_parser("measure", help="start the container; cold start, then each photo once over HTTP, one at a time"))
    _container_args(sub.add_parser("cold", help="start a fresh container: one more cold-start sample"))
    rec = sub.add_parser("record", help="on the host: write the run record")
    rec.add_argument("--measured", required=True)
    rec.add_argument("--cold", nargs="*", default=[], help="files from `cold` runs")
    rec.add_argument("--machine", required=True, help="the machine, stated: CPU, cores, RAM, OS, Docker limits")
    rec.add_argument("--image", required=True, help="the container image measured (name and digest)")
    rec.add_argument("--variant", required=True, help="what this run is: baseline, or what changed")
    lst = sub.add_parser("list", help="write datasets/perf/images.txt")
    lst.add_argument("--n", type=int, default=MIN_REQUESTS)
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
        _write_json(a.out, run_measurement(a, requests=False))
    elif a.stage == "record":
        read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
        path = build_record(read(a.measured), machine=a.machine, image=a.image, variant=a.variant,
                            cold_files=[read(p) for p in a.cold])
        rec = json.loads(path.read_text(encoding="utf-8"))
        o = rec["config"]["outcomes"]
        print(f"{o['failed']} of {o['requests']} requests failed")
        print(path)
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
