from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainConfig:
    manifest: str
    out_dir: str
    model: str = "convnext_tiny"
    seed: int = 0
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    transforms: tuple[str, ...] = ("random_resized_crop", "hflip", "color_jitter")
    manifest_hash: str = ""
    num_workers: int = 0
    # Joint training (E4). The defaults keep the lab-only behaviour.
    field_manifest: str = ""  # when set, rows of this manifest join the lab rows
    train_splits: tuple[str, ...] = ("lab_train",)
    select_split: str = "lab_dev"
    lab_cap: int = 0  # most lab photos per class in training; 0 means no cap
    held_out_source: str = ""  # a field `source` removed from training and selection
    extra_manifest: str = ""  # E12: extra real-photo rows (split `extra_train`) that may join training, never selection
    crop_masks: str = ""  # mask cache file; when set, every photo is cut to the disease area
    unfreeze_blocks: int = 0  # DINOv2: transformer blocks trained besides the head
    backbone_lr: float = 0.0  # learning rate of those blocks; 0 means the same as `lr`

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["transforms"] = tuple(d["transforms"])
        if "train_splits" in d:
            d["train_splits"] = tuple(d["train_splits"])
        return cls(**d)
