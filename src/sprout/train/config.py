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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["transforms"] = tuple(d["transforms"])
        return cls(**d)
