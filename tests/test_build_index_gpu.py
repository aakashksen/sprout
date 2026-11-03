import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sprout.rag.retrieve import RetrievalConfig, Retriever

TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_index_gpu.py"
spec = importlib.util.spec_from_file_location("build_index_gpu", TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

TEXT = "Early blight shows dark spots with rings on lower leaves of the plant. " * 4


@pytest.fixture
def files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text(TEXT, encoding="utf-8")
    m = tmp_path / "m.csv"
    m.write_text("file,crop,diseases,title,publisher,source_url,date\na.md,potato,early blight,A,P,https://x.edu/a,\n")
    return m, raw, tmp_path / "index"


def test_gpu_share_is_held_only_around_embedding_and_device_is_recorded(files):
    m, raw, index = files
    calls = []

    def embed(texts, model):
        calls.append("embed")
        return np.ones((len(texts), 4), dtype=np.float32), {"device": "cuda", "gpu": "fake"}

    info = tool.build(index, task="T9", manifest=m, raw=raw, embed=embed, ctl=lambda a, t: calls.append(a))
    assert calls == ["gpu-acquire", "embed", "gpu-release"]
    assert json.loads((index / "build-info.json").read_text())["device"] == "cuda"
    assert info["chunks"] == 1
    # the saved vectors are the ones Retriever would have computed itself
    retriever = Retriever(index, config=RetrievalConfig())
    assert retriever._load_vectors().shape == (1, 4)


def test_gpu_share_is_released_when_embedding_fails(files):
    m, raw, index = files
    calls = []

    def embed(texts, model):
        raise RuntimeError("CUDA is not available")

    with pytest.raises(RuntimeError, match="CUDA"):
        tool.build(index, task="T9", manifest=m, raw=raw, embed=embed, ctl=lambda a, t: calls.append(a))
    assert calls == ["gpu-acquire", "gpu-release"]
    assert not (index / "build-info.json").exists()
