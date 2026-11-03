import torch
from PIL import Image

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """The one deterministic preprocessing used for validation and inference (`size` is for the segmenter)."""
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    x = data.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std
