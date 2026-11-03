import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sprout.data.splits import dataset_hash, load_manifest
from sprout.train.checkpoint import load_checkpoint, save_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import build_loaders, make_class_map


@dataclass
class RunOutput:
    best_path: Path
    latest_path: Path
    best_metric: float
    epochs_done: int
    history: list[dict]


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
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
    raise ValueError(f"unknown model: {name}")


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


def train(config: TrainConfig, *, resume: Path | None = None) -> RunOutput:
    """Fresh run when `resume` is None, otherwise continue from that checkpoint."""
    out_dir = Path(config.out_dir)
    latest, best = out_dir / "latest.pt", out_dir / "best.pt"
    if resume is None and (latest.exists() or best.exists()):
        raise FileExistsError(f"{out_dir} has checkpoints; pass resume=<path> or use a new out_dir")
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(resume)

    manifest = load_manifest(config.manifest)
    config = dataclasses.replace(config, manifest_hash=dataset_hash(manifest))
    class_map = make_class_map(manifest)
    train_dl, val_dl = build_loaders(manifest, config, class_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, len(class_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    start, best_metric = 0, float("-inf")
    if resume is not None:
        ckpt = load_checkpoint(Path(resume))
        if ckpt["config"].manifest_hash != config.manifest_hash:
            raise ValueError("checkpoint was trained on a different manifest")
        if ckpt["class_map"] != class_map:
            raise ValueError("checkpoint class map differs from the manifest")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start, best_metric = ckpt["epoch"], ckpt["best_metric"]

    history: list[dict] = []
    for epoch in range(start, config.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            n += len(y)
        scheduler.step()
        val_loss, val_acc = _evaluate(model, val_dl, loss_fn, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(n, 1),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
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
