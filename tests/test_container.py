import http.cookiejar
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sprout-serve-test"
BANNED = ("torch", "torchvision", "triton", "nvidia", "cuda")


def _requirement_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.append(re.split(r"[=<>~! ;\[]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_dockerignore_excludes_env_and_data():
    patterns = {ln.strip() for ln in (ROOT / ".dockerignore").read_text().splitlines()}
    for must in (".env", ".env.*", "legacy", "datasets", "knowledge/raw", "orchestration", "models",
                 "*.onnx", "*.pt", "*.pth", ".git", ".dvcstore", "var", "uploads"):
        assert must in patterns, must
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text()
    copied = [ln.split()[1] for ln in dockerfile.splitlines() if ln.startswith("COPY ")]
    assert not any(c.startswith(("datasets", "legacy", "orchestration", "knowledge/raw", ".env")) for c in copied)
    assert "knowledge/manifest.csv" in copied  # only the manifest, never the raw documents


def test_image_runs_as_non_root():
    lines = [ln.strip() for ln in (ROOT / "deploy" / "Dockerfile").read_text().splitlines()]
    users = [ln.split()[1] for ln in lines if ln.startswith("USER ")]
    assert users, "no USER line"
    assert users[-1] not in ("root", "0")
    last_user = max(i for i, ln in enumerate(lines) if ln.startswith("USER "))
    assert any(ln.startswith("ENTRYPOINT") for ln in lines[last_user:]), "entrypoint must come after USER"


def test_no_torch_in_serving_requirements():
    names = _requirement_names(ROOT / "requirements" / "serve.txt")
    assert "onnxruntime" in names and "fastembed" in names
    assert not [n for n in names if n.startswith(BANNED)]
    assert "torch" in _requirement_names(ROOT / "requirements" / "train.txt")
    assert "torch" not in (ROOT / "deploy" / "Dockerfile").read_text().lower()


def test_serving_import_path_does_not_import_torch():
    code = (
        "import os, sys; os.environ['SPROUT_NO_AUTOBUILD']='1'; sys.path.insert(0, 'deploy'); "
        "import wsgi, sprout.web, sprout.rag.retrieve, sprout.rag.ingest, sprout.serve.types; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'torchvision')))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_fake_identifier_refused_outside_testing(monkeypatch, tmp_path):
    monkeypatch.setenv("SPROUT_NO_AUTOBUILD", "1")
    monkeypatch.syspath_prepend(str(ROOT / "deploy"))
    import wsgi

    base = {"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_DATA_DIR": str(tmp_path), "SPROUT_FAKE_IDENTIFIER": "1"}
    for env in ("production", "development"):
        with pytest.raises(RuntimeError, match="only allowed"):
            wsgi.build({**base, "SPROUT_ENV": env})
    with pytest.raises(RuntimeError, match="only allowed"):
        wsgi.build(base)  # SPROUT_ENV defaults to production
    app = wsgi.build({**base, "SPROUT_ENV": "testing"})
    assert app.extensions["sprout_identifier"] is not None
    # production with no flag: no model, so not ready
    app = wsgi.build({"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_DATA_DIR": str(tmp_path / "p")})
    assert app.extensions["sprout_identifier"] is None
    assert app.test_client().get("/readyz").status_code == 503


# ---- real: needs Docker -------------------------------------------------------------------

def _docker(*args, check=True, **kw):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check, **kw)


@pytest.fixture(scope="module")
def image():
    if shutil.which("docker") is None:
        pytest.fail("docker is required")
    _docker("build", "-f", "deploy/Dockerfile", "-t", IMAGE, ".", cwd=ROOT)
    return IMAGE


class Browser:
    def __init__(self, base: str):
        self.base = base
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.csrf = ""

    def request(self, method, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            r = self.opener.open(req, timeout=30)
        except urllib.error.HTTPError as e:
            r = e
        return r.status, r.read().decode(), r.url

    def refresh(self, path="/login"):
        _, body, _ = self.request("GET", path)
        self.csrf = re.search(r'name="csrf-token" content="([^"]+)"', body).group(1)

    def form(self, path, fields):
        body = "&".join(f"{k}={v}" for k, v in {**fields, "csrf_token": self.csrf}.items()).encode()
        return self.request("POST", path, body, {"Content-Type": "application/x-www-form-urlencoded"})

    def upload(self, png: bytes):
        b = uuid.uuid4().hex
        body = (f'--{b}\r\nContent-Disposition: form-data; name="csrf_token"\r\n\r\n{self.csrf}\r\n'
                f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="leaf.png"\r\n'
                f"Content-Type: image/png\r\n\r\n").encode() + png + f"\r\n--{b}--\r\n".encode()
        return self.request("POST", "/analyses", body, {"Content-Type": f"multipart/form-data; boundary={b}"})


def _run(image, name, port, **env):
    args = ["run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8080"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    _docker(*args, image)


def _wait(base, path="/healthz"):
    for _ in range(60):
        try:
            return urllib.request.urlopen(base + path, timeout=2).status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            time.sleep(0.5)
    raise AssertionError("container did not answer")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _count_uploads(name):
    return _docker("exec", name, "sh", "-c", "find /data/uploads -type f | wc -l").stdout.strip()


@pytest.mark.real
def test_smoke_in_clean_container(image):
    """upload -> prediction -> question -> history -> deletion over HTTP, in a fresh container.

    The identifier is the FakeIdentifier test double (SPROUT_ENV=testing). No index is mounted, so the question
    step proves the "no supporting document" path, not retrieval. Both are rerun for real at T7.4.
    """
    name, port = f"sprout-smoke-{uuid.uuid4().hex[:8]}", _free_port()
    base = f"http://127.0.0.1:{port}"
    _run(image, name, port, SPROUT_ENV="testing", SPROUT_SECRET_KEY="smoke-secret", SPROUT_FAKE_IDENTIFIER="1")
    try:
        assert _wait(base) == 200
        assert _docker("exec", name, "id", "-u").stdout.strip() != "0"
        missing = _docker("exec", name, "sh", "-c", "ls -d /app/.env /app/datasets /app/legacy /app/orchestration 2>&1",
                          check=False).stdout
        assert missing.count("No such file") == 4
        torch = _docker("exec", name, "sh", "-c", "pip list 2>/dev/null | grep -ci '^torch' || true").stdout
        assert torch.strip() == "0"
        b = Browser(base)
        b.refresh()
        assert b.form("/signup", {"username": "smoke", "password": "password123"})[0] == 200
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (30, 140, 40)).save(buf, "PNG")
        b.refresh("/upload")
        status, body, url = b.upload(buf.getvalue())
        aid = int(re.search(r"/analyses/(\d+)", url).group(1))
        assert status == 200 and "tomato" in body.lower()
        assert _count_uploads(name) == "1"
        status, body, _ = b.request("POST", f"/analyses/{aid}/chat", json.dumps({"message": "how do I treat it?"}).encode(),
                                    {"Content-Type": "application/json", "X-CSRF-Token": b.csrf})
        reply = json.loads(body)
        assert status == 200 and reply["passages"] == [] and reply["note"]
        status, body, _ = b.request("GET", "/history")
        assert status == 200 and f"/analyses/{aid}" in body
        assert b.form(f"/analyses/{aid}/delete", {})[0] == 200
        assert b.request("GET", f"/analyses/{aid}")[0] == 404
        assert _count_uploads(name) == "0"
    finally:
        _docker("rm", "-f", name, check=False)


@pytest.mark.real
def test_production_container_refuses_fake_and_missing_secret(image):
    for env in ({"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_FAKE_IDENTIFIER": "1"}, {}):
        flags = sum((["-e", f"{k}={v}"] for k, v in env.items()), [])
        assert _docker("run", "--rm", *flags, image, check=False).returncode != 0
    name, port = f"sprout-prod-{uuid.uuid4().hex[:8]}", _free_port()
    _run(image, name, port, SPROUT_SECRET_KEY="x" * 32)
    try:
        assert _wait(f"http://127.0.0.1:{port}") == 200
        assert _wait(f"http://127.0.0.1:{port}", "/readyz") == 503  # no model: reported, not faked
    finally:
        _docker("rm", "-f", name, check=False)


CONTAINER_FILES = ("deploy/entrypoint.sh", "deploy/wsgi.py", "deploy/Dockerfile", "requirements/serve.txt",
                   "docker-compose.yml", ".dockerignore")


def test_container_files_are_committed_with_lf_endings():
    """A CRLF entrypoint makes the container fail with 'no such file or directory' on every start."""
    for path in CONTAINER_FILES:
        committed = subprocess.run(["git", "show", f":{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        assert b"\r" not in committed, f"{path} is committed with CR bytes"
        eol = subprocess.run(["git", "check-attr", "eol", "--", path], cwd=ROOT, capture_output=True, text=True)
        assert eol.stdout.strip().endswith("eol: lf"), f"{path} is not pinned to LF in .gitattributes"


@pytest.mark.real
def test_built_container_files_have_no_cr(image):
    out = _docker("run", "--rm", "--entrypoint", "sh", image, "-c",
                  "grep -lc \"$(printf '\r')\" /app/deploy/entrypoint.sh /app/deploy/wsgi.py || true").stdout
    assert out.strip() == ""
