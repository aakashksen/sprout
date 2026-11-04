"""Joint training on real and lab photos: which rows train, which select, the lab cap, and the crop and
background options. Only `lab_train` and `field_train` ever train, only `field_dev` selects, and no test split
is ever read."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from sprout.atomic import atomic_write
from sprout.data.masks import BLACK_BELOW, hsv_mask, segmented_path
from sprout.data.splits import load_manifest
from sprout.segment.crop import crop_to_disease
from sprout.train.augment import BACKGROUND_CHANCE, paste_background
from sprout.train.config import TrainConfig
from sprout.train.dataset import train_transform, val_transform

TRAINING = ("lab_train", "field_train", "extra_train")
SELECTION = "field_dev"
MASK_SIZE = 320  # the size E10 predicts at; cached crop masks are stored at this size
LEAF_FRACTION = (0.05, 0.98)  # a leaf mask outside this share of the photo is not trusted


def check_splits(config: TrainConfig) -> None:
    bad = [s for s in (*config.train_splits, config.select_split) if s.endswith("_test")]
    if bad:
        raise ValueError(f"test splits are never read in training: {bad}")
    if not set(config.train_splits) <= set(TRAINING):
        raise ValueError(f"only {TRAINING} may train; got {config.train_splits}")
    if config.select_split != SELECTION:
        raise ValueError(f"selection uses {SELECTION}, not {config.select_split!r}")
    if "extra_train" in config.train_splits and not config.extra_manifest:
        raise ValueError("train_splits names extra_train but extra_manifest is not set")


def load_joint_manifest(config: TrainConfig) -> pd.DataFrame:
    """Lab and real rows in one table with a `source` column ('lab' for lab). Test rows are dropped here."""
    check_splits(config)
    lab = load_manifest(config.manifest).assign(source="lab")
    frames = [lab, load_manifest(config.field_manifest)]
    if config.extra_manifest:
        extra = load_manifest(config.extra_manifest)
        if "source" not in extra or (extra["split"] != "extra_train").any():
            raise ValueError(f"{config.extra_manifest} must have a source column and only extra_train rows")
        frames.append(extra)
    both = pd.concat(frames, ignore_index=True)
    return both[~both["split"].str.endswith("_test")].reset_index(drop=True)


def cap_per_class(rows: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """At most `cap` rows of every class, chosen by a seeded shuffle of the sorted paths."""
    if cap <= 0:
        return rows
    rng = np.random.default_rng(seed)
    keep = []
    for _, part in rows.sort_values("path").groupby("label"):
        idx = part.index.to_numpy()
        keep.extend(rng.permutation(idx)[:cap] if len(idx) > cap else idx)
    return rows.loc[sorted(keep)]


def select_rows(manifest: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training rows (lab capped per class) and the selection rows, minus any held-out source. Extra rows
    that repeat a photo of the held-out source (same sha1 or same group) leave training with it."""
    check_splits(config)
    train = manifest[manifest["split"].isin(config.train_splits)]
    lab = cap_per_class(train[train["source"] == "lab"], config.lab_cap, config.seed)
    train = pd.concat([lab, train[train["source"] != "lab"]]).sort_index()
    sel = manifest[manifest["split"] == config.select_split]
    if config.held_out_source:
        held = manifest[manifest["source"] == config.held_out_source]
        copies = (train["split"] == "extra_train") & (train["sha1"].isin(held["sha1"]) | train["group"].isin(held["group"]))
        train = train[(train["source"] != config.held_out_source) & ~copies]
        sel = sel[sel["source"] != config.held_out_source]
    return train, sel


class MaskStore:
    """Predicted disease masks (E10) per photo path, at MASK_SIZE, kept as packed bits in one file."""

    def __init__(self, masks: dict[str, np.ndarray]):
        self.masks = masks

    def get(self, path: str) -> np.ndarray:
        if path not in self.masks:
            raise KeyError(f"no cached crop mask for {path}")
        return self.masks[path]

    def save(self, path: Path) -> None:
        paths = sorted(self.masks)
        bits = np.stack([np.packbits(self.masks[p].astype(bool)) for p in paths]) if paths else np.zeros((0, 1), np.uint8)
        atomic_write(Path(path), lambda f: np.savez_compressed(f, paths=np.array(paths), bits=bits))

    @classmethod
    def load(cls, path: str | Path) -> "MaskStore":
        with np.load(path) as z:
            paths, bits = z["paths"], z["bits"]
        n = MASK_SIZE * MASK_SIZE
        return cls({str(p): np.unpackbits(b)[:n].reshape(MASK_SIZE, MASK_SIZE).astype(bool) for p, b in zip(paths, bits, strict=True)})


def leaf_mask(path: str, image: Image.Image) -> tuple[np.ndarray | None, str]:
    """True on the leaf, at the photo's size, and where it came from: the official segmented copy of this lab
    photo (matched by file name), else the colour threshold. None when the mask is not believable."""
    seg = segmented_path(path)
    if seg is not None:
        with Image.open(seg) as s:
            rgb = np.asarray(s.convert("RGB").resize(image.size, Image.NEAREST))
        mask, kind = rgb.max(axis=2) >= BLACK_BELOW, "official"
    else:
        small = hsv_mask(image, MASK_SIZE)
        mask = np.asarray(Image.fromarray(small.astype(np.uint8)).resize(image.size, Image.NEAREST), dtype=bool)
        kind = "hsv"
    lo, hi = LEAF_FRACTION
    return (mask, kind) if lo <= mask.mean() <= hi else (None, kind)


class JointDataset(Dataset):
    """Rows with a `source` column. Training mode augments (and may swap a lab photo's background); both modes
    apply the same crop when a mask store is given, so training and scoring see the same kind of input."""

    def __init__(self, rows: pd.DataFrame, class_map: dict[str, int], *, train: bool, transforms=(),
                 masks: MaskStore | None = None):
        self.paths = list(rows["path"])
        self.sources = list(rows["source"])
        self.targets = [class_map[x] for x in rows["label"]]
        self.train, self.transforms, self.masks = train, tuple(transforms), masks

    def __len__(self) -> int:
        return len(self.paths)

    def image(self, i: int) -> Image.Image:
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        if self.train and "background" in self.transforms and self.sources[i] == "lab":
            if random.random() < BACKGROUND_CHANCE:
                leaf, _ = leaf_mask(self.paths[i], image)
                if leaf is not None:
                    image = paste_background(image, leaf, random)
        if self.masks is not None:
            image = crop_to_disease(image, self.masks.get(self.paths[i]))  # a tiny or empty area keeps the photo
        return image

    def __getitem__(self, i: int):
        image = self.image(i)
        x = train_transform(image, self.transforms) if self.train else val_transform(image)
        return x, self.targets[i]


def eval_dataset(rows: pd.DataFrame, class_map: dict[str, int], config: TrainConfig) -> JointDataset:
    masks = MaskStore.load(config.crop_masks) if config.crop_masks else None
    return JointDataset(rows, class_map, train=False, masks=masks)


def joint_datasets(manifest: pd.DataFrame, config: TrainConfig, class_map: dict[str, int]):
    train_rows, sel_rows = select_rows(manifest, config)
    masks = MaskStore.load(config.crop_masks) if config.crop_masks else None
    train = JointDataset(train_rows, class_map, train=True, transforms=config.transforms, masks=masks)
    return train, JointDataset(sel_rows, class_map, train=False, masks=masks)
