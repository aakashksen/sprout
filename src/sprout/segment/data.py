"""PlantSeg disease masks: safe extraction, the seg manifest, mask run-length encoding and the training dataset.

Mask files hold 0 for background and a non-zero value for diseased pixels; `disease_mask` turns them into booleans.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from sprout.data.field import link_same_scene
from sprout.data.grouping import group_near_duplicates
from sprout.data.masks import iou as mask_iou
from sprout.data.splits import COLUMNS, MANIFEST_DIR

ROOT = Path(__file__).resolve().parents[3]
ZIP_REL = Path("datasets/raw/plantseg_dl/plantseg.zip")
IMAGE_REL = Path("datasets/raw/plantseg/plantseg/images")
MASK_REL = Path("datasets/raw/plantseg_masks")
SEG_SPLITS = ("seg_train", "seg_dev")
SPLIT_DIR = {"Training": "train", "Validation": "val", "Test": "test"}
SEG_COLUMNS = COLUMNS + ["mask"]
DEV_FRACTION = 0.15
MAX_MASK_BYTES = 100_000_000


def _member_target(name: str) -> tuple[str, str] | None:
    """(split folder, file name) for `plantseg/annotations/<split>/<name>.png`; None for anything else.
    Raises for a name that could leave the destination folder."""
    if "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive member name {name!r}")
    parts = PurePosixPath(name).parts
    if ".." in parts:
        raise ValueError(f"unsafe archive member name {name!r}")
    if len(parts) == 4 and parts[:2] == ("plantseg", "annotations") and parts[2] in SPLIT_DIR.values() \
            and parts[3].lower().endswith(".png"):
        return parts[2], parts[3]
    return None


def extract_masks(zip_path: Path, dest: Path) -> int:
    """Copy only the mask PNGs out of the archive into a new folder `dest`, never running anything from it.
    The folder appears only once complete, so an existing `dest` means a finished extraction."""
    dest = Path(dest)
    if dest.exists():
        return sum(1 for _ in dest.glob("*/*.png"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        count = _extract_into(zip_path, tmp)
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return count


def _extract_into(zip_path: Path, tmp: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = _member_target(info.filename)
            if target is None or info.is_dir():
                continue
            if info.file_size > MAX_MASK_BYTES:
                raise ValueError(f"mask {info.filename} is implausibly large")
            out = (tmp / target[0] / target[1]).resolve()
            if tmp.resolve() not in out.parents:
                raise ValueError(f"archive member {info.filename!r} escapes the destination")
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(z.read(info))
            count += 1
    return count


def disease_mask(path: str | Path) -> np.ndarray:
    """Boolean HxW array: True where the mask file marks disease."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    """Run lengths of a boolean mask in row-major order, starting with a run of False (possibly 0)."""
    flat = np.asarray(mask, dtype=bool).ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs
    return {"size": list(np.asarray(mask).shape), "counts": [int(r) for r in runs]}


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    if sum(counts) != h * w:
        raise ValueError("run lengths do not add up to the mask size")
    values = np.arange(len(counts)) % 2 == 1
    return np.repeat(values, counts).reshape(h, w)


def rle_iou(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    if pred["size"] != gt["size"]:
        raise ValueError("prediction and ground-truth masks differ in size")
    return mask_iou(rle_decode(pred), rle_decode(gt))


def exclude_held_out(candidates: list[str], held_out: list[str]) -> tuple[dict[str, str], set[str]]:
    """Near-duplicate groups over candidates plus held-out photos (the field-set rule in docs/field-dedup.md:
    pHash grouping, then pairs confirmed by same_scene_score). Returns the group of each candidate and the
    candidates sharing a group with any held-out photo. These held-out files are opened only for this check."""
    pool = sorted(set(candidates) | set(held_out))
    groups = link_same_scene(pool, group_near_duplicates(pool))
    bad = {groups[p] for p in held_out}
    return {p: groups[p] for p in candidates}, {p for p in candidates if groups[p] in bad}


def split_groups(groups: pd.Series, seed: int = 13, dev_fraction: float = DEV_FRACTION) -> dict[str, str]:
    """Whole groups to seg_dev until it holds about dev_fraction of the images, the rest to seg_train."""
    sizes = groups.value_counts().sort_index()
    order = [sizes.index[i] for i in np.random.default_rng(seed).permutation(len(sizes))]
    target, have, out = dev_fraction * len(groups), 0, {}
    for g in order:
        dev = have < target
        out[g] = "seg_dev" if dev else "seg_train"
        have += int(sizes[g]) if dev else 0
    return out


def build_seg_manifest(meta: pd.DataFrame, field: pd.DataFrame, root: Path = ROOT,
                       image_rel: Path = IMAGE_REL, mask_rel: Path = MASK_REL, seed: int = 13) -> pd.DataFrame:
    """seg.csv rows for PlantSeg images that cannot reach a held-out photo.

    Excluded: any image that is a field_dev or field_test photo (matched by file name and by sha1), and any image
    in a near-duplicate group with a field_dev or field_test photo from PlantDoc or PlantSeg. That second check
    covers PlantSeg images outside field.csv too (unmapped classes, label conflicts, reviewed removals).
    """
    held = field[field["split"].isin(["field_dev", "field_test"])]
    held_names = {Path(p).name for p, s in zip(held["path"], held["source"], strict=True) if s == "plantseg"}
    held_sha = set(held["sha1"])
    rows = []
    for name, disease, label_file, split in zip(meta["Name"], meta["Disease"], meta["Label file"], meta["Split"], strict=True):
        path = image_rel / SPLIT_DIR[split] / name
        sha1 = hashlib.sha1((root / path).read_bytes()).hexdigest()
        if name in held_names or sha1 in held_sha:
            continue
        rows.append((path.as_posix(), disease, sha1, (mask_rel / SPLIT_DIR[split] / label_file).as_posix()))
    df = pd.DataFrame(rows, columns=["path", "label", "sha1", "mask"])
    groups, drop = exclude_held_out([str(root / p) for p in df["path"]], [str(root / p) for p in held["path"]])
    df["group"] = [groups[str(root / p)] for p in df["path"]]
    df = df[[str(root / p) not in drop for p in df["path"]]].copy()
    df["split"] = df["group"].map(split_groups(df["group"], seed))
    return df.sort_values("path")[SEG_COLUMNS].reset_index(drop=True)


def main() -> None:
    """CPU step, no GPU: extract the masks (once) and write datasets/manifests/seg.csv. Opens every PlantSeg image
    and the field_dev and field_test photos for the near-duplicate check, so it takes several minutes."""
    extract_masks(ROOT / ZIP_REL, ROOT / MASK_REL)
    meta = pd.read_csv(ROOT / IMAGE_REL.parent / "Metadata.csv", encoding="utf-8-sig")
    field = pd.read_csv(MANIFEST_DIR / "field.csv")
    seg = build_seg_manifest(meta, field, ROOT)
    seg.to_csv(MANIFEST_DIR / "seg.csv", index=False)
    print(f"{len(meta)} PlantSeg images, {len(meta) - len(seg)} excluded, {seg['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
