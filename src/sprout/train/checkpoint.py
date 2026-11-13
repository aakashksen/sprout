from pathlib import Path

import torch

from sprout.atomic import atomic_write
from sprout.train.config import TrainConfig


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    class_map: dict[str, int],
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "class_map": class_map,
        "config": config.to_dict(),
    }
    atomic_write(path, lambda f: torch.save(state, f))


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["config"] = TrainConfig.from_dict(ckpt["config"])
    return ckpt
