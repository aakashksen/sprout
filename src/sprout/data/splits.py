import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("lab_train", "lab_dev", "lab_test")
MANIFEST_DIR = Path(__file__).resolve().parents[3] / "datasets" / "manifests"
COLUMNS = ["path", "label", "group", "split", "sha1"]


def make_splits(df: pd.DataFrame, seed: int = 13, ratios=(0.7, 0.15, 0.15)) -> pd.DataFrame:
    """Assign whole groups to splits, balancing each class's image counts."""
    rng = np.random.default_rng(seed)
    sizes = df.groupby("group").size()
    primary = df.groupby("group")["label"].agg(lambda s: s.value_counts().sort_index().idxmax())
    assign: dict[str, str] = {}
    for label in sorted(primary.unique()):
        groups = sorted(primary.index[primary == label])
        order = [groups[i] for i in rng.permutation(len(groups))]
        target = [r * sum(sizes[g] for g in groups) for r in ratios]
        have = [0.0] * len(SPLITS)
        count = [0] * len(SPLITS)
        for k, g in enumerate(order):
            empty = [i for i in range(len(SPLITS)) if count[i] == 0]
            pool = empty if len(order) - k <= len(empty) else range(len(SPLITS))
            i = max(pool, key=lambda s: (target[s] - have[s], -s))
            assign[g] = SPLITS[i]
            have[i] += sizes[g]
            count[i] += 1
    out = df.copy()
    out["split"] = out["group"].map(assign)
    return out


def dataset_hash(manifest: pd.DataFrame) -> str:
    rows = manifest[COLUMNS].astype(str).sort_values("path")
    return hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest()


def load_manifest(name: str) -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_DIR / f"{name}.csv")
    return df[COLUMNS + [c for c in ("source",) if c in df.columns]]


def build_manifest(name: str, root: str) -> pd.DataFrame:
    from sprout.data.grouping import group_near_duplicates
    from sprout.data.inventory import inventory

    df = inventory(root)
    groups = group_near_duplicates(df["path"])
    df["group"] = df["path"].map(groups)
    out = make_splits(df)[COLUMNS]
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANIFEST_DIR / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":
    import sys

    build_manifest(sys.argv[1], sys.argv[2])
