from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class Predictions:
    path: list[str]
    label: np.ndarray  # class indices from the class map
    probs: np.ndarray  # (n, num_classes), columns in class-map order


def predict(
    model, manifest_split: pd.DataFrame, class_map: dict[str, int], batch_size: int = 32
) -> Predictions:
    """Softmax outputs for each manifest row, using the shared preprocessing."""
    import torch

    from sprout.preprocess import preprocess

    device = next(model.parameters()).device
    paths = list(manifest_split["path"])
    labels = np.array([class_map[x] for x in manifest_split["label"]], dtype=int)
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i : i + batch_size]:
                with Image.open(p) as img:
                    batch.append(preprocess(img))
            logits = model(torch.stack(batch).to(device))
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(chunks) if chunks else np.zeros((0, len(class_map)))
    return Predictions(paths, labels, probs)
