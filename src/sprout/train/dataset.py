import random
from functools import partial

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from sprout.data.splits import load_manifest
from sprout.preprocess import preprocess
from sprout.train.augment import augment_extra
from sprout.train.config import TrainConfig


def augment(image: Image.Image, transforms: tuple[str, ...]) -> Image.Image:
    """Random augmentation for training only; draws from Python's `random`."""
    image = image.convert("RGB")
    w, h = image.size
    if "random_resized_crop" in transforms:
        scale = random.uniform(0.5, 1.0)
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        x, y = random.randint(0, w - cw), random.randint(0, h - ch)
        image = image.crop((x, y, x + cw, y + ch))
    if "hflip" in transforms and random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if "color_jitter" in transforms:
        for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            image = enhance(image).enhance(random.uniform(0.7, 1.3))
    return augment_extra(image, transforms)


def train_transform(image: Image.Image, transforms: tuple[str, ...]) -> torch.Tensor:
    return preprocess(augment(image, transforms))


def val_transform(image: Image.Image) -> torch.Tensor:
    return preprocess(image)


class ManifestDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, class_map: dict[str, int], transform):
        self.paths = list(rows["path"])
        self.targets = [class_map[x] for x in rows["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.transform(im), self.targets[i]


def load_training_manifest(config: TrainConfig) -> pd.DataFrame:
    """The manifest `train` reads: the one named manifest, or lab and real rows together for joint training."""
    if config.field_manifest:
        from sprout.train.joint import load_joint_manifest

        return load_joint_manifest(config)
    return load_manifest(config.manifest)


def make_class_map(manifest: pd.DataFrame) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(manifest["label"].unique()))}


def build_datasets(manifest: pd.DataFrame, config: TrainConfig, class_map: dict[str, int]):
    """Separate train and validation datasets; only the train one augments."""
    if config.field_manifest:
        from sprout.train.joint import joint_datasets

        return joint_datasets(manifest, config, class_map)
    tfs = config.transforms
    train = ManifestDataset(
        manifest[manifest["split"] == "lab_train"], class_map, partial(train_transform, transforms=tfs)
    )
    val = ManifestDataset(manifest[manifest["split"] == "lab_dev"], class_map, val_transform)
    return train, val


def build_loaders(manifest: pd.DataFrame, config: TrainConfig, class_map: dict[str, int]):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train, val = build_datasets(manifest, config, class_map)
    gen = torch.Generator().manual_seed(config.seed)
    # Workers are seeded by torch from `gen`, so batches repeat for a given seed and worker count.
    opts = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    train_dl = DataLoader(train, shuffle=True, generator=gen, **opts)
    val_dl = DataLoader(val, shuffle=False, **opts)
    return train_dl, val_dl
