"""A small two-class (background, disease) segmentation model and mask prediction."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from sprout.preprocess import preprocess

MODEL_SOURCE = {
    "name": "lraspp_mobilenet_v3_large",
    "weights": "torchvision LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1",
    "url": "https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
    "license": "torchvision code is BSD-3-Clause; the weights were trained on a COCO subset (COCO annotations "
               "CC BY 4.0, images under their Flickr terms); the 21-class heads are replaced and retrained here",
}


def build_model(pretrained: bool = True) -> nn.Module:
    """LRASPP with MobileNetV3-Large; its two classifier convolutions are replaced by 2-class ones."""
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
        lraspp_mobilenet_v3_large,
    )

    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None)
    head = model.classifier
    head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, 2, 1)
    head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, 2, 1)
    return model


def to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    return preprocess(image, size)


@torch.no_grad()
def predict_mask(model: nn.Module, image: Image.Image, size: int = 320) -> np.ndarray:
    """Boolean HxW mask (the photo's own size) of predicted disease pixels."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    logits = model(to_tensor(image, size).unsqueeze(0).to(device))["out"]
    model.train(was_training)
    small = (logits.argmax(1)[0] == 1).cpu().numpy().astype(np.uint8)
    w, h = image.size
    return np.asarray(Image.fromarray(small).resize((w, h), Image.NEAREST), dtype=bool)
