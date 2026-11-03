"""DINOv2 small as a frozen feature extractor (its class token), loaded through torch.hub."""
from __future__ import annotations

import torch
from torch import nn

from sprout.baselines.frozen import freeze

# Primary source: Meta's repository. The commit is pinned so the code that torch.hub runs is the code read here.
SOURCE = {
    "repo": "facebookresearch/dinov2",
    "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    "model_id": "dinov2_vits14",
    "weights_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "license": "Apache License 2.0",
    "license_statement": (
        "README.md of facebookresearch/dinov2, section License: 'DINOv2 code and model weights are released "
        "under the Apache License 2.0.' The same README lists non-commercial licenses for other models "
        "(its X-ray and cell variants), which are not used here."
    ),
    "gated": False,
    "feature": "class token after the final norm, 384 values",
}


def load_dino_encoder(pretrained: bool = True) -> nn.Module:
    """Downloads into the normal torch hub cache on first use. `trust_repo` runs the pinned repository's hubconf.
    `pretrained=False` builds the same network without fetching weights, for loading a saved state."""
    model = torch.hub.load(
        f"{SOURCE['repo']}:{SOURCE['commit']}", SOURCE["model_id"], trust_repo=True, pretrained=pretrained
    )
    return freeze(model)


class DinoClassifier(nn.Module):
    """DINOv2 class token into a linear head. The encoder is frozen except its last `unfreeze_blocks` blocks and
    final norm. It stays in evaluation mode (the network has no dropout), so the one state_dict holds
    everything needed to rebuild and export it."""

    def __init__(self, encoder: nn.Module, num_classes: int, unfreeze_blocks: int = 0, feature_dim: int = 384):
        super().__init__()
        self.encoder = freeze(encoder)
        if unfreeze_blocks:
            for module in (*encoder.blocks[-unfreeze_blocks:], encoder.norm):
                module.requires_grad_(True)
        self.head = nn.Linear(feature_dim, num_classes)

    def train(self, mode: bool = True) -> "DinoClassifier":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.encoder.parameters() if p.requires_grad]
