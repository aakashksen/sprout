import random

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from sprout.preprocess import preprocess
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
    return image


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


def make_class_map(manifest: pd.DataFrame) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(manifest["label"].unique()))}


def build_datasets(manifest: pd.DataFrame, config: TrainConfig, class_map: dict[str, int]):
    """Separate train and validation datasets; only the train one augments."""
    tfs = config.transforms
    train = ManifestDataset(
        manifest[manifest["split"] == "lab_train"], class_map, lambda im: train_transform(im, tfs)
    )
    val = ManifestDataset(manifest[manifest["split"] == "lab_dev"], class_map, val_transform)
    return train, val


def build_loaders(manifest: pd.DataFrame, config: TrainConfig, class_map: dict[str, int]):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train, val = build_datasets(manifest, config, class_map)
    gen = torch.Generator().manual_seed(config.seed)
    train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=gen)
    val_dl = DataLoader(val, batch_size=config.batch_size, shuffle=False)
    return train_dl, val_dl
