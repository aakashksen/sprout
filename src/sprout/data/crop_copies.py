"""Crop-aware copy check: DINOv2 picks nearest field photos, patch-token matching with RANSAC confirms.

A crop, zoom or captioned copy of a photo shares no pixel grid with the original, so the full-frame
correlation misses it. DINOv2 patch tokens describe local content, so a copy leaves a set of mutual
nearest-neighbour patches that agree on one scale and shift. Used only to drop training extras.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

SIZE = 224
GRID = 16
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
K_EACH = (
    3  # nearest candidates per evaluation split; the 16 known copies ranked 1st or 2nd
)
MIN_COS = 0.4  # loose floor for a mutual patch match; geometry does the real filtering
HYPOTHESES = 300
TOLERANCE = 1.0  # grid cells
SCALE_RANGE = (0.35, 2.8)
WORKERS = 4
# Rule, chosen on half of a labelled set (see docs/extra-sources.md). An extra is dropped if its best class-token
# cosine to a dev or test photo and its best patch-match inlier count (patches agreeing on one scale and shift)
# reach any pair below. Unsure means drop, so it is looser than the best rule on the choosing half (cosine 0.90 or
# 40 inliers): three rounds of viewing the pairs just under the rule kept finding probable crops, one in ten to
# thirty, so the cosine floor was lowered to 0.80 and a combined clause added.
RULE = ((0.80, 0), (0.0, 30), (0.75, 20))
COS = RULE[0][0]
INLIERS = RULE[1][1]


def _load(path: str, flip: bool = False) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(img, dtype=np.float32)
    return ((a / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def encode(
    paths: list[str],
    encoder,
    device: str,
    batch: int = 64,
    patches: bool = False,
    flip: bool = False,
):
    """Class tokens (N, D) and, if asked, patch tokens (N, GRID*GRID, D) in fp16; all L2-normalised."""
    import torch

    cls, pat = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(paths), batch):
            x = np.stack(
                list(pool.map(lambda p: _load(p, flip), paths[start : start + batch]))
            )
            with torch.no_grad():
                out = encoder.forward_features(torch.from_numpy(x).to(device))
            cls.append(
                torch.nn.functional.normalize(
                    out["x_norm_clstoken"].float(), dim=-1
                ).cpu()
            )
            if patches:
                p = torch.nn.functional.normalize(
                    out["x_norm_patchtokens"].float(), dim=-1
                )
                pat.append(p.half().cpu())
    c = torch.cat(cls)
    return (c, torch.cat(pat)) if patches else c


def candidates(ex_cls, ref_cls, ref_is_dev, k_each: int = K_EACH, device: str = "cpu"):
    """For each extra, indices of its k nearest dev and k nearest test photos by class token, plus the cosines."""
    import torch

    sims = ex_cls.to(device) @ ref_cls.to(device).T
    dev = torch.as_tensor(ref_is_dev, device=device)
    picks, cos = [], []
    for mask in (dev, ~dev):
        s = sims.masked_fill(~mask, -2.0)
        top = s.topk(min(k_each, int(mask.sum())), dim=1)
        picks.append(top.indices.cpu())
        cos.append(top.values.cpu())
    return torch.cat(picks, 1).numpy(), torch.cat(cos, 1).numpy()


def mutual_matches(a, b):
    """Mutual nearest-neighbour patches of two token sets (n, D) and (m, D): indices into a and b, and cosines."""
    import torch

    s = a.float() @ b.float().T
    ab, ba = s.argmax(1), s.argmax(0)
    ia = torch.arange(len(a), device=a.device)
    ok = ba[ab] == ia
    cos = s[ia, ab]
    ok &= cos >= MIN_COS
    return ia[ok].cpu().numpy(), ab[ok].cpu().numpy(), cos[ok].cpu().numpy()


def ransac_inliers(ia: np.ndarray, ib: np.ndarray) -> int:
    """Most matches agreeing on one scale and shift (b = s*a + t) over fixed pseudo-random 2-point hypotheses."""
    n = len(ia)
    if n < 3:
        return n
    pa = np.stack([ia % GRID, ia // GRID], 1).astype(np.float32)
    pb = np.stack([ib % GRID, ib // GRID], 1).astype(np.float32)
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, HYPOTHESES)
    j = rng.integers(0, n, HYPOTHESES)
    da, db = pa[i] - pa[j], pb[i] - pb[j]
    la = np.linalg.norm(da, axis=1)
    keep = la > 1.0
    s = np.linalg.norm(db, axis=1)[keep] / la[keep]
    i = i[keep]
    ok = (s >= SCALE_RANGE[0]) & (s <= SCALE_RANGE[1])
    if not ok.any():
        return 1
    s, i = s[ok], i[ok]
    t = pb[i] - s[:, None] * pa[i]
    pred = s[:, None, None] * pa[None] + t[:, None, :]
    err = np.linalg.norm(pred - pb[None], axis=2)
    return int((err <= TOLERANCE).sum(axis=1).max())


def crop_scores(ex_paths, ref_paths, ref_splits, encoder, device, batch: int = 32):
    """Per extra: best class-token cosine, best inlier count, and the reference of each, over the
    nearest dev and test candidates, for the image as it is and mirrored."""
    ref_cls, ref_pat = encode(ref_paths, encoder, device, patches=True)
    ref_pat = ref_pat.to(device)
    is_dev = [s == "field_dev" for s in ref_splits]
    n = len(ex_paths)
    best_cos, best_inl = np.full(n, -1.0), np.zeros(n, dtype=np.int32)
    best_ref_cos, best_ref_inl = (
        np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
    )
    for flip in (False, True):
        ex_cls = encode(ex_paths, encoder, device, flip=flip)
        idx, cos = candidates(ex_cls, ref_cls, is_dev, device=device)
        for start in range(0, n, batch):
            _, pat = encode(
                ex_paths[start : start + batch],
                encoder,
                device,
                patches=True,
                flip=flip,
            )
            for r, row in enumerate(pat):
                i = start + r
                row = row.to(device)
                for c in range(idx.shape[1]):
                    ia, ib, _ = mutual_matches(row, ref_pat[idx[i, c]])
                    k = ransac_inliers(ia, ib)
                    if k > best_inl[i]:
                        best_inl[i], best_ref_inl[i] = k, idx[i, c]
                    if cos[i, c] > best_cos[i]:
                        best_cos[i], best_ref_cos[i] = cos[i, c], idx[i, c]
    return best_cos, best_inl, best_ref_cos, best_ref_inl


def crop_copies(ex_paths, ref_paths, ref_splits, encoder, device) -> dict[str, str]:
    """Extras that look like a crop, zoom or captioned copy of a dev or test photo, to that photo's split."""
    cos, inl, ref_cos, ref_inl = crop_scores(
        ex_paths, ref_paths, ref_splits, encoder, device
    )
    out = {}
    for i, p in enumerate(ex_paths):
        if cos[i] >= COS:
            out[p] = ref_splits[ref_cos[i]]
        elif any(cos[i] >= c and inl[i] >= k for c, k in RULE):
            out[p] = ref_splits[ref_inl[i]]
    return out
