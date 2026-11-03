"""Second, independent copy check: 32x32 gray correlation over rotations and flips, batched.

It uses no hashes and no foreground mask, so watermarks and crops that lower the field rule's
score still correlate. It is deliberately over-inclusive and is used only to drop training
extras, never to build or change `field.csv`.
"""

import numpy as np
from PIL import Image

COPY_SCORE = 0.75
SIZE = 32


def gray_vector(path: str) -> np.ndarray:
    """32x32 gray image as a zero-mean, unit-norm float32 vector (zeros if the image is flat)."""
    with Image.open(path) as img:
        a = np.asarray(
            img.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32
        )
    return _unit(a)


def _unit(a: np.ndarray) -> np.ndarray:
    a = a - a.mean()
    n = float(np.sqrt((a * a).sum()))
    return (a / n if n > 0 else a).reshape(-1)


def variants(vec: np.ndarray) -> np.ndarray:
    """The 8 rotations and flips of one image vector, shape (8, SIZE*SIZE)."""
    a = vec.reshape(SIZE, SIZE)
    return np.stack(
        [np.rot90(b, k).reshape(-1) for b in (a, a[:, ::-1]) for k in range(4)]
    )


def max_correlation(
    paths: list[str],
    ref_paths: list[str],
    device: str = "cpu",
    batch: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """For each path, the best correlation with any reference image over its 8 symmetries.

    Returns (scores, index of the best reference). Every pair is scored, so a copy filed
    under another label is still found.
    """
    import torch

    refs = np.concatenate([variants(gray_vector(p)) for p in ref_paths])
    r = torch.from_numpy(refs).to(device)
    scores = np.zeros(len(paths), dtype=np.float32)
    best = np.zeros(len(paths), dtype=np.int64)
    for start in range(0, len(paths), batch):
        chunk = np.stack([gray_vector(p) for p in paths[start : start + batch]])
        s = (torch.from_numpy(chunk).to(device) @ r.T).view(
            len(chunk), len(ref_paths), 8
        )
        top, which = s.max(dim=2)[0].max(dim=1)
        scores[start : start + len(chunk)] = top.cpu().numpy()
        best[start : start + len(chunk)] = which.cpu().numpy()
    return scores, best


def copies_of(
    paths: list[str],
    ref_paths: list[str],
    device: str = "cpu",
    threshold: float = COPY_SCORE,
) -> dict[str, str]:
    """Paths scoring at or above the threshold against any reference, mapped to that reference."""
    scores, best = max_correlation(paths, ref_paths, device)
    return {p: ref_paths[b] for p, s, b in zip(paths, scores, best) if s >= threshold}
