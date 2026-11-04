"""The shared preprocessing without torch, so serving can use it. `sprout.preprocess` wraps this."""
import numpy as np
from PIL import Image

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess_array(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize, divide by 255 and normalize in float32; returns (3, size, size)."""
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(size, size, 3).transpose(2, 0, 1)
    x = x.astype(np.float32) / np.float32(255.0)
    mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(STD, dtype=np.float32).reshape(3, 1, 1)
    return np.ascontiguousarray((x - mean) / std)
