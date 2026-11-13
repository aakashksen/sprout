"""Hand-made image features (HOG and colour histograms) and a gradient-boosted tree model on them, CPU only."""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import xgboost as xgb
from PIL import Image
from skimage.feature import hog

GRAY = np.array([0.299, 0.587, 0.114])


def classic_features(image: Image.Image, size: int = 128, orientations: int = 9, pixels_per_cell: int = 16,
                     cells_per_block: int = 2, bins: int = 32) -> np.ndarray:
    """HOG on the grey image followed by one normalised histogram per colour channel; no randomness."""
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR))
    edges = hog(arr @ GRAY / 255.0, orientations=orientations, pixels_per_cell=(pixels_per_cell,) * 2,
                cells_per_block=(cells_per_block,) * 2, block_norm="L2-Hys", feature_vector=True)
    colour = [np.histogram(arr[..., k], bins=bins, range=(0, 256))[0] / (size * size) for k in range(3)]
    return np.concatenate([edges, *colour]).astype(np.float32)


def fit_xgboost(x: np.ndarray, y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray, num_classes: int,
                params: dict[str, Any], seed: int) -> tuple[xgb.Booster, dict[str, list[float]], float]:
    """Train on `x`, logging the multiclass log loss on the training and dev rows each round.
    Stops with an error if a loss is not finite. Returns the booster, the loss history and fit seconds."""
    booster_params = {
        "objective": "multi:softprob", "num_class": num_classes, "eval_metric": "mlogloss",
        "max_depth": params["max_depth"], "eta": params["learning_rate"], "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"], "tree_method": params["tree_method"],
        "device": params["device"], "nthread": params["n_jobs"], "seed": seed,
    }
    train, dev = xgb.DMatrix(x, label=y), xgb.DMatrix(dev_x, label=dev_y)
    seen: dict[str, dict[str, list[float]]] = {}
    began = time.perf_counter()
    booster = xgb.train(booster_params, train, int(params["n_estimators"]), evals=[(train, "train"), (dev, "dev")],
                        evals_result=seen, verbose_eval=False)
    seconds = time.perf_counter() - began
    history = {"train_loss": seen["train"]["mlogloss"], "val_loss": seen["dev"]["mlogloss"]}
    for key, values in history.items():
        bad = [i + 1 for i, v in enumerate(values) if not math.isfinite(v)]
        if bad:
            raise RuntimeError(f"xgboost {key} is not finite at round {bad[0]}; run stopped")
    return booster, history, seconds


def xgboost_probs(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    return booster.predict(xgb.DMatrix(x)).astype(np.float64)
