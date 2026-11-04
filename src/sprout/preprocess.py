import torch
from PIL import Image

from sprout.preprocess_np import IMAGE_SIZE, MEAN, STD, preprocess_array

__all__ = ["IMAGE_SIZE", "MEAN", "STD", "preprocess"]


def preprocess(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """The one deterministic preprocessing used for validation and inference (`size` is for the segmenter)."""
    return torch.from_numpy(preprocess_array(image, size))
