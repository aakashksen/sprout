"""Extra photos that look like a source's own photos: DINOv2 ViT-S/14 class-token cosine, taking the higher of a
photo and its mirror, against that source's field_dev (as scored) and field_train photos. The embeddings are the
only part that needs a model; the rest is plain arrays, so it is tested with fakes.

Never reads a test split: callers pass only field_dev and field_train rows, and `source_rows` refuses anything else."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

from sprout.atomic import atomic_write
from sprout.data.splits import dataset_hash

THRESHOLD = 0.75
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ALLOWED = ("field_dev", "field_train")


def source_rows(field: pd.DataFrame, source: str, split: str) -> pd.DataFrame:
    if split not in ALLOWED:
        raise ValueError(f"near-copy checks read {ALLOWED} only, not {split!r}")
    return field[(field["source"] == source) & (field["split"] == split)].reset_index(drop=True)


def load_photo(path: Path, mirror: bool) -> np.ndarray:
    """3 x 224 x 224, bicubic resize of the whole photo (no crop), ImageNet normalisation."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (2 * SIZE, 2 * SIZE))
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        if mirror:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(im, dtype=np.float32)
    return ((a / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def embed_paths(paths: list[str], model, device, root: Path, *, mirror: bool = False, batch: int = 32,
                workers: int = 4) -> np.ndarray:
    """Unit-length class-token vectors, one row per path, in order."""
    import torch

    model.to(device).eval()
    out = []
    with ThreadPoolExecutor(max(workers, 1)) as pool, torch.no_grad():
        for s in range(0, len(paths), batch):
            x = np.stack(list(pool.map(lambda p: load_photo(root / p, mirror), paths[s:s + batch])))
            y = torch.nn.functional.normalize(model(torch.from_numpy(x).to(device)).float(), dim=1)
            out.append(y.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def cached_embedding(cache: Path | None, name: str, paths: list[str], compute: Callable[[], np.ndarray]) -> np.ndarray:
    """A saved embedding is reused only when it was made for exactly these paths; written atomically."""
    key = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    file = None if cache is None else Path(cache) / f"{name}.npz"
    if file is not None and file.is_file():
        with np.load(file) as z:
            if str(z["key"]) == key:
                return z["x"]
    x = compute()
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(file, lambda f: np.savez(f, key=np.array(key), x=x))
    return x


def best_cosine(extra: np.ndarray, extra_mirror: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Per extra photo: its highest cosine to any row of `other`, using the higher of photo and mirror."""
    if len(other) == 0:
        return np.full(len(extra), -1.0)
    return np.maximum(extra @ other.T, extra_mirror @ other.T).max(axis=1)


def trim_entries(extra_paths: list[str], vs_eval: np.ndarray, vs_train: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    """The extras at or above the threshold against either side, sorted by path."""
    hit = (vs_eval >= threshold) | (vs_train >= threshold)
    return sorted(({"path": extra_paths[i], "cos_eval": round(float(vs_eval[i]), 6), "cos_train": round(float(vs_train[i]), 6)}
                   for i in np.flatnonzero(hit)), key=lambda e: e["path"])


def trimmed_manifest(extra: pd.DataFrame, entries: list[dict[str, Any]]) -> pd.DataFrame:
    gone = {e["path"] for e in entries}
    return extra[~extra["path"].isin(gone)].reset_index(drop=True)


def build(source: str, eval_rows: pd.DataFrame, train_rows: pd.DataFrame, extra: pd.DataFrame, model, device, root: Path, *,
          threshold: float = THRESHOLD, workers: int = 4, cache: Path | None = None) -> dict[str, Any]:
    """The removal list for `source`: counts, the entries, and the hashes of the extras before and after."""
    if set(eval_rows["split"]) - {"field_dev"} or set(train_rows["split"]) - {"field_train"}:
        raise ValueError("evaluated rows must be field_dev and training rows field_train")
    paths = list(extra["path"])

    def emb(name: str, ps: list[str], mirror: bool = False) -> np.ndarray:
        return cached_embedding(cache, name, ps, lambda: embed_paths(ps, model, device, root, mirror=mirror, workers=workers))

    x, xm = emb("extra", paths), emb("extra_mirror", paths, True)
    ev = emb(f"{source}_eval", list(eval_rows["path"]))
    tr = emb(f"{source}_train", list(train_rows["path"]))
    vs_eval, vs_train = best_cosine(x, xm, ev), best_cosine(x, xm, tr)
    entries = trim_entries(paths, vs_eval, vs_train, threshold)
    kept = trimmed_manifest(extra, entries)
    return {
        "source": source, "threshold": threshold,
        "measure": "DINOv2 ViT-S/14 class token, unit length, 224x224 bicubic resize, higher of the photo and its mirror",
        "device": str(device), "extra_rows": len(extra), "extra_dataset_hash": dataset_hash(extra),
        "evaluated_rows": len(eval_rows), "training_rows": len(train_rows),
        "against_evaluated": int((vs_eval >= threshold).sum()), "against_training": int((vs_train >= threshold).sum()),
        "union": len(entries), "trimmed_rows": len(kept), "trimmed_dataset_hash": dataset_hash(kept),
        "remove": entries,
    }


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=1) + "\n"
