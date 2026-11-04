"""Independent leak search: ORB keypoints (skimage) + brute-force mutual matching + RANSAC similarity.

Every query photo from the chosen sources in the extra manifest is compared with every field_dev and field_test
photo, as it is and mirrored. No DINOv2, no hashes. A control (a cropped, mirrored field photo of an Apple) is
appended as the last query and must be found; if it is not, the search is not working.

It decodes downloaded images, so run it as `python -I tools/leak_orb.py ...` (isolated mode, so nothing in the
current folder is imported). Paths in the manifests are relative to --root. The test of the real data skips when
the raw images are absent (tests/test_extra_sources.py); this tool needs them and does not skip.

    python -I tools/leak_orb.py EXTRA.csv FIELD.csv OUT.csv --sources plantpathology2021 [--root .] [--start N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import ORB
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

NK, SIDE = 300, 512
MIN_MATCHES, MIN_INLIERS, MAX_HAMMING = 20, 10, 40
HELD_SPLITS = ("field_dev", "field_test")
COLUMNS = ["extra", "ref", "split", "matches", "inliers", "flip"]


def feats(orb: ORB, path: Path, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((SIDE, SIDE))
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(im, dtype=np.float64) / 255
    try:
        orb.detect_and_extract(a)
    except Exception:
        return np.zeros((0, 2)), np.zeros((0, 256), np.float32)
    return orb.keypoints[:, ::-1].copy(), np.where(orb.descriptors, 1.0, -1.0).astype(np.float32)


def pad(d: np.ndarray, n: int = NK) -> np.ndarray:
    out = np.zeros((n, 256), np.float32)
    out[: len(d)] = d[:n]
    return out


def write_control(held: pd.DataFrame, root: Path, out: Path) -> Path:
    apples = held[held.label.str.startswith("Apple")]
    if apples.empty:
        raise SystemExit("no Apple photo in field_dev/field_test to build the control from")
    with Image.open(root / apples.path.iloc[0]) as im:
        im = im.convert("RGB")
        w, h = im.size
        crop = im.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
        crop.transpose(Image.FLIP_LEFT_RIGHT).save(out)
    return out


def search(queries: list[Path], held: pd.DataFrame, root: Path, start: int = 0) -> pd.DataFrame:
    orb = ORB(n_keypoints=NK, fast_threshold=0.05)
    refs = [feats(orb, root / p) for p in held.path]
    ref_kp = [k for k, _ in refs]
    ref_desc = torch.from_numpy(np.stack([pad(d) for _, d in refs]))
    ref_n = torch.tensor([len(d) for _, d in refs])
    rows = []
    for qi, p in enumerate(queries):
        if qi < start:
            continue
        for flip in (False, True):
            qk, qd = feats(orb, p, flip)
            if len(qd) < 10:
                continue
            q = torch.from_numpy(qd)
            for s in range(0, len(held), 400):
                sim = torch.einsum("qd,rnd->rqn", q, ref_desc[s:s + 400])  # larger = closer
                mask = torch.arange(NK)[None, None, :] < ref_n[s:s + 400, None, None]
                sim = sim.masked_fill(~mask, -1e9)
                a, b = sim.argmax(2), sim.argmax(1)  # best ref keypoint per query one, and the reverse
                mutual = torch.gather(b, 1, a) == torch.arange(sim.shape[1])[None]
                good = mutual & (torch.gather(sim, 2, a[..., None])[..., 0] >= 256 - 2 * MAX_HAMMING)
                cnt = good.sum(1)
                for r in torch.nonzero(cnt >= MIN_MATCHES).flatten().tolist():
                    qi_idx = torch.nonzero(good[r]).flatten().numpy()
                    src, dst = qk[qi_idx], ref_kp[s + r][a[r, qi_idx].numpy()]
                    try:
                        _, inl = ransac((src, dst), SimilarityTransform, min_samples=2,
                                        residual_threshold=8, max_trials=200, rng=0)
                    except Exception:
                        continue
                    if inl is not None and inl.sum() >= MIN_INLIERS:
                        rows.append((str(p), held.path.iloc[s + r], held.split.iloc[s + r],
                                     int(cnt[r]), int(inl.sum()), flip))
        print(qi, len(queries), len(rows), flush=True)
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("inliers", ascending=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("extra_manifest", type=Path)
    ap.add_argument("field_manifest", type=Path)
    ap.add_argument("out", type=Path, help="CSV of matches; the control crop is saved beside it")
    ap.add_argument("--sources", required=True, help="comma-separated source names in the extra manifest")
    ap.add_argument("--root", type=Path, default=Path("."), help="folder the manifest paths are relative to")
    ap.add_argument("--start", type=int, default=0, help="skip the first N queries (to resume)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    extra = pd.read_csv(args.extra_manifest)
    extra = extra[extra.source.isin(args.sources.split(","))]
    field = pd.read_csv(args.field_manifest)
    held = field[field.split.isin(HELD_SPLITS)].reset_index(drop=True)
    control = write_control(held, args.root, args.out.with_name("control_crop.jpg"))
    queries = [args.root / p for p in extra.path] + [control]
    df = search(queries, held, args.root, args.start)
    df.to_csv(args.out, index=False)
    print(df.head(15).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
