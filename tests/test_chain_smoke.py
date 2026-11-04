"""The full chain in the real container: real ONNX model, real index and reranker, real writer through Ollama on the host.

Marked `real` and not run in the fast suite. It needs Docker, a running Ollama with the writer loaded and listening
beyond loopback (OLLAMA_HOST=0.0.0.0), and these variables (a missing one fails the test, it does not skip it):
  SPROUT_SMOKE_MODEL        the ONNX classifier file
  SPROUT_SMOKE_CALIBRATION  results/e5/chosen.json (or the file E12's choice comes with)
  SPROUT_SMOKE_INDEX        the retrieval index folder (with its vectors cache)
  SPROUT_SMOKE_FASTEMBED    a folder already holding the fastembed models
  SPROUT_SMOKE_PHOTO        a real leaf photo of a plant the index has documents for
"""
import json
import os
import re
import uuid
from pathlib import Path

import pytest
from test_container import (  # noqa: F401
    ROOT,
    Browser,
    _count_uploads,
    _docker,
    _free_port,
    _wait,
    image,
)

WRITER = os.environ.get("SPROUT_SMOKE_WRITER", "qwen3.5:4b-q4_K_M")
NEEDED = ("SPROUT_SMOKE_MODEL", "SPROUT_SMOKE_CALIBRATION", "SPROUT_SMOKE_INDEX", "SPROUT_SMOKE_FASTEMBED",
          "SPROUT_SMOKE_PHOTO")
ANSWER_STATES = ("answered", "ungrounded", "clarify", "no_document")


@pytest.mark.real
def test_full_chain_in_the_container_with_the_real_model_index_and_writer(image):  # noqa: F811
    missing = [k for k in NEEDED if not os.environ.get(k) or not Path(os.environ[k]).exists()]
    assert not missing, f"set these to existing paths first: {missing}"
    models = Path(os.environ["SPROUT_SMOKE_MODEL"])
    name, port = f"sprout-chain-{uuid.uuid4().hex[:8]}", _free_port()
    base = f"http://127.0.0.1:{port}"
    mounts = [
        f"{models}:/models/model.onnx:ro", f"{os.environ['SPROUT_SMOKE_CALIBRATION']}:/models/chosen.json:ro",
        f"{os.environ['SPROUT_SMOKE_INDEX']}:/models/index:ro", f"{os.environ['SPROUT_SMOKE_FASTEMBED']}:/models/fastembed:ro",
    ]
    args = ["run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8080",
            "--add-host", "host.docker.internal:host-gateway", "-e", "SPROUT_SECRET_KEY=" + "chain-secret-" * 3,
            "-e", f"SPROUT_WRITER_MODEL={WRITER}", "-e", "SPROUT_OLLAMA_URL=http://host.docker.internal:11434"]
    for m in mounts:
        args += ["-v", m]
    _docker(*args, image)
    try:
        assert _wait(base) == 200
        ready = None
        for _ in range(240):  # the index and the reranker warm in the background
            ready = json.loads(_docker("exec", name, "python", "-c",
                                       "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/readyz').read().decode())",
                                       check=False).stdout or "{}")
            if ready.get("ready") and ready.get("chat") == "ready":
                break
            import time

            time.sleep(1)
        assert ready.get("ready") is True and ready.get("chat") == "ready", ready
        b = Browser(base)
        b.refresh()
        assert b.form("/signup", {"username": "chain", "password": "password123"})[0] == 200
        b.refresh("/upload")
        status, body, url = b.upload(Path(os.environ["SPROUT_SMOKE_PHOTO"]).read_bytes())
        aid = int(re.search(r"/analyses/(\d+)", url).group(1))
        assert status == 200 and "prediction" in body.lower()
        assert _count_uploads(name) == "1"
        status, body, _ = b.request("POST", f"/analyses/{aid}/chat", json.dumps({"message": "How do I treat this?"}).encode(),
                                    {"Content-Type": "application/json", "X-CSRF-Token": b.csrf})
        reply = json.loads(body)
        assert status == 200 and reply["state"] in ANSWER_STATES, reply
        assert reply["text"] and "product label" in reply["text"] or reply["state"] in ("clarify", "no_document")
        if reply["state"] == "answered":
            assert reply["citations"] and reply["passages"] and reply["grounded"] is True
        assert json.loads(_docker("exec", name, "python", "-c",
                                  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/readyz').read().decode())"
                                  ).stdout)["writer"] == "ok"
        assert f"/analyses/{aid}" in b.request("GET", "/history")[1]
        assert b.form(f"/analyses/{aid}/delete", {})[0] == 200
        assert _count_uploads(name) == "0"
    finally:
        _docker("rm", "-f", name, check=False)
