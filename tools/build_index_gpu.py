"""Build a document index and embed its chunks on the GPU.

    py -3.12 tools/build_index_gpu.py --index <dir> [--task T6.5] [--chunker structured] [--header]

Runs build_index on knowledge/manifest.csv and knowledge/raw (CPU, text only), then holds a GPU
share from taskctl for the embedding stage only, embeds every chunk with the retrieval model on
CUDA, and saves the vectors where Retriever looks for them. It fails rather than falling back to
the CPU. The device used is written to build-info.json in the index folder.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sprout.rag.ingest import build_index, load_manifest  # noqa: E402
from sprout.rag.retrieve import RetrievalConfig, chunk_search_text, vectors_cache_path  # noqa: E402

BATCH = 64
MAX_LENGTH = 512


def main_folder() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent


def taskctl(action: str, task: str) -> None:
    """gpu-acquire or gpu-release, run from the main folder so every track shares one lock."""
    done = subprocess.run(
        ["py", "-3.12", "tools/taskctl.py", action, task], cwd=main_folder(), capture_output=True, text=True
    )
    print(done.stdout.strip() or done.stderr.strip())
    if done.returncode != 0:
        raise RuntimeError(f"taskctl {action} {task} failed (exit {done.returncode})")


def embed_on_gpu(texts: list[str], model_name: str) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to embed on the CPU")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    print(f"embedding {len(texts)} chunks on {device} ({name}), model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model did not move to CUDA")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    start = time.time()
    with torch.inference_mode():
        for lo in range(0, len(order), BATCH):
            ids = order[lo : lo + BATCH]
            batch = tokenizer([texts[i] for i in ids], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            cls = model(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state[:, 0]
            out[ids] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
    info = {
        "device": "cuda",
        "gpu": name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "embed_seconds": round(time.time() - start, 1),
    }
    return out, info


def build(
    index_dir: Path, *, task: str, manifest: Path, raw: Path, embed=embed_on_gpu, ctl=taskctl,
    chunker: str = "plain", header: bool = False,
) -> dict:
    config = RetrievalConfig()
    report = build_index(load_manifest(manifest), raw, index_dir, chunker=chunker, header=header)
    rows = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [chunk_search_text(r) for r in rows]
    ctl("gpu-acquire", task)
    try:
        vectors, device_info = embed(texts, config.embedding_model)
    finally:
        ctl("gpu-release", task)
    cache = vectors_cache_path(index_dir, config.embedding_model, texts)
    tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, vectors)
    os.replace(tmp, cache)
    info = {
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": task,
        "chunks": report.chunks,
        "failures": len(report.failures),
        "chunker": chunker,
        "header": header,
        "embedding_model": config.embedding_model,
        "vectors_file": cache.name,
        **device_info,
    }
    (index_dir / "build-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--task", default="T6.5")
    parser.add_argument("--manifest", type=Path, default=ROOT / "knowledge" / "manifest.csv")
    parser.add_argument("--raw", type=Path, default=ROOT / "knowledge" / "raw")
    parser.add_argument("--chunker", choices=("plain", "structured"), default="plain")
    parser.add_argument("--header", action="store_true")
    args = parser.parse_args()
    if args.index.name in ("index", "index.old"):
        sys.exit("refusing to build over knowledge/index; give a new name such as knowledge/index-structured")
    print(json.dumps(build(args.index, task=args.task, manifest=args.manifest, raw=args.raw,
                           chunker=args.chunker, header=args.header), indent=2))


if __name__ == "__main__":
    main()
