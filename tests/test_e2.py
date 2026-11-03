import numpy as np
import torch
from PIL import Image
from torch import nn

from sprout.data.masks import energy_outside
from sprout.explain.gradcam import grad_cam, overlay, target_layer
from sprout.train.loop import build_model


def signal_model() -> nn.Sequential:
    """Reads only the red channel and says class 0 where it is bright."""
    conv = nn.Conv2d(3, 1, 1, bias=False)
    fc = nn.Linear(1, 2)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1))
        fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        fc.bias.zero_()
    return nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)


def test_gradcam_output_matches_image_size():
    torch.manual_seed(0)
    model = build_model("tiny_cnn", 4, pretrained=False)
    x = torch.randn(3, 3, 224, 224)
    cam, logits = grad_cam(model, x, target_layer(model, "tiny_cnn"))
    assert cam.shape == (3, 224, 224) and logits.shape == (3, 4)
    assert cam.min() >= 0 and cam.max() <= 1
    photo = Image.new("RGB", (300, 180), (10, 120, 10))
    assert overlay(photo, cam[0]).size == (300, 180)
    assert all(p.requires_grad for p in model.parameters())


def test_gradcam_highlights_the_only_signal():
    model = signal_model()
    x = torch.zeros(1, 3, 32, 32)
    x[0, 0, 8:16, 8:16] = 1.0
    cam, _ = grad_cam(model, x, model[1])
    inside = np.zeros((32, 32), dtype=bool)
    inside[8:16, 8:16] = True
    assert cam[0][inside].min() > 0.99
    assert cam[0][~inside].max() < 0.1
    assert energy_outside(cam[0], inside) < 0.05
    assert energy_outside(cam[0], ~inside) > 0.95
