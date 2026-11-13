import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sprout.data.splits import dataset_hash
from sprout.train.checkpoint import load_checkpoint, save_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import build_loaders, load_training_manifest, make_class_map


@dataclass
class RunOutput:
    best_path: Path
    latest_path: Path
    best_metric: float
    epochs_done: int
    history: list[dict]


def build_model(name: str, num_classes: int, pretrained: bool = True, unfreeze_blocks: int = 0) -> nn.Module:
    if name == "tiny_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 8, 7, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )
    if name == "convnext_tiny":
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights="DEFAULT" if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if name == "convnextv2_tiny":
        import timm

        return timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=pretrained, num_classes=num_classes)
    if name == "dinov2_small":
        from sprout.baselines.dino import DinoClassifier, load_dino_encoder

        return DinoClassifier(load_dino_encoder(pretrained), num_classes, unfreeze_blocks)
    raise ValueError(f"unknown model: {name}")


def _optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    backbone = model.backbone_parameters() if hasattr(model, "backbone_parameters") else []
    if not backbone:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return torch.optim.AdamW(
        [{"params": rest, "lr": config.lr}, {"params": backbone, "lr": config.backbone_lr or config.lr}]
    )


def _evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def _restore(ckpt: dict, config: TrainConfig, class_map: dict[str, int], model, optimizer, scheduler, scaler):
    """Check that the checkpoint belongs to this run, load its state, and return (epoch, best_metric)."""
    if ckpt["config"].manifest_hash != config.manifest_hash:
        raise ValueError("checkpoint was trained on a different manifest")
    if dataclasses.replace(ckpt["config"], out_dir=config.out_dir, num_workers=config.num_workers) != config:
        raise ValueError("training settings differ from the checkpoint's; resume with the same settings")
    if ckpt["class_map"] != class_map:
        raise ValueError("checkpoint class map differs from the manifest")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["best_metric"]


def _train_epoch(model, loader, optimizer, scaler, loss_fn, device, epoch: int) -> float:
    """One pass over the training data; returns the mean loss. A non-finite loss stops the run."""
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"epoch {epoch + 1}: train loss is {loss.item()}, not finite; run stopped")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / max(n, 1)


def train(config: TrainConfig, *, resume: Path | None = None) -> RunOutput:
    """Fresh run when `resume` is None, otherwise continue from that checkpoint."""
    out_dir = Path(config.out_dir)
    latest, best = out_dir / "latest.pt", out_dir / "best.pt"
    if resume is None and (latest.exists() or best.exists()):
        raise FileExistsError(f"{out_dir} has checkpoints; pass resume=<path> or use a new out_dir")
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(resume)

    manifest = load_training_manifest(config)
    config = dataclasses.replace(config, manifest_hash=dataset_hash(manifest))
    class_map = make_class_map(manifest)
    train_dl, val_dl = build_loaders(manifest, config, class_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra = {"unfreeze_blocks": config.unfreeze_blocks} if config.unfreeze_blocks else {}
    model = build_model(config.model, len(class_map), **extra).to(device)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    start, best_metric = 0, float("-inf")
    if resume is not None:
        ckpt = load_checkpoint(Path(resume))
        start, best_metric = _restore(ckpt, config, class_map, model, optimizer, scheduler, scaler)

    history: list[dict] = []
    for epoch in range(start, config.epochs):
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_loss = _train_epoch(model, train_dl, optimizer, scaler, loss_fn, device, epoch)
        scheduler.step()
        val_loss, val_acc = _evaluate(model, val_dl, loss_fn, device)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"epoch {epoch + 1}: validation loss is {val_loss}, not finite; run stopped")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seconds": time.perf_counter() - began,
                "peak_mem_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "peak_mem_reserved": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            }
        )
        print(f"epoch {epoch + 1}/{config.epochs} {history[-1]}", flush=True)
        improved = val_acc > best_metric
        if improved:
            best_metric = val_acc
        state = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            best_metric=best_metric,
            class_map=class_map,
            config=config,
        )
        save_checkpoint(latest, **state)
        if improved:
            save_checkpoint(best, **state)
    return RunOutput(best, latest, best_metric, max(start, config.epochs), history)
