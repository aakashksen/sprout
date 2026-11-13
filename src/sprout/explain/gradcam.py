"""Grad-CAM for the class a model picks (or a chosen class), at the size of the input."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


def target_layer(model: nn.Module, name: str) -> nn.Module:
    """The last layer that still has a spatial grid, for the models `build_model` makes."""
    if name == "convnext_tiny":
        return model.features[-1]
    if name == "tiny_cnn":
        return model[1]
    raise ValueError(f"unknown model: {name}")


def grad_cam(
    model: nn.Module, x: torch.Tensor, layer: nn.Module, target: torch.Tensor | None = None
) -> tuple[np.ndarray, torch.Tensor]:
    """Grad-CAM maps (B, H, W), each scaled to peak at 1 (all zero if nothing supports the class), and the logits.

    Parameters are frozen while it runs so that only the layers after `layer` keep a graph; that
    keeps the memory close to inference. The model is left in eval mode with its flags restored.
    """
    flags = [(p, p.requires_grad) for p in model.parameters()]
    held: list[torch.Tensor] = []

    def grab(_module, _inputs, out):
        out = out.detach().requires_grad_(True)
        held.append(out)
        return out

    handle = layer.register_forward_hook(grab)
    model.eval()
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        with torch.enable_grad():
            logits = model(x)
            idx = logits.argmax(dim=1) if target is None else target
            score = logits.gather(1, idx.view(-1, 1)).sum()
            (grads,) = torch.autograd.grad(score, held[0])
    finally:
        handle.remove()
        for p, flag in flags:
            p.requires_grad_(flag)
    acts = held[0].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.amax(dim=(1, 2), keepdim=True)
    cam = torch.where(peak > 0, cam / peak.clamp_min(1e-12), cam)
    return cam.cpu().numpy(), logits.detach()


def _heat(c: np.ndarray) -> np.ndarray:
    """Blue to red ramp, (H, W) in [0, 1] to (H, W, 3) uint8."""
    r, g, b = (np.clip(1.5 - np.abs(4 * c - k), 0, 1) for k in (3, 2, 1))
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """The image with the map blended over it, at the image's own size. Pixels with no activation stay as they were."""
    base = image.convert("RGB")
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)) / 255.0
    heat = _heat(c).astype(float)
    weight = (alpha * (c > 0))[..., None]
    out = np.asarray(base).astype(float) * (1 - weight) + heat * weight
    return Image.fromarray(out.astype(np.uint8))
