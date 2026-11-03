import sys
from pathlib import Path

from sprout.data.splits import SPLITS, dataset_hash, load_manifest


def check(name: str) -> list[str]:
    m = load_manifest(name)
    problems = []
    if m.groupby("group")["split"].nunique().gt(1).any():
        problems.append("a group appears in more than one split")
    if m.groupby("sha1")["split"].nunique().gt(1).any():
        problems.append("identical files appear in more than one split")
    if m.groupby("label")["split"].nunique().lt(len(SPLITS)).any():
        problems.append("a class is missing from a split")
    missing = sum(not Path(p).exists() for p in m["path"])
    if missing:
        problems.append(f"{missing} manifest files are missing on disk")
    return problems


if __name__ == "__main__":
    name = sys.argv[1]
    issues = check(name)
    m = load_manifest(name)
    print(f"{name}: {len(m)} images, {m['label'].nunique()} classes, hash {dataset_hash(m)}")
    for p in issues:
        print("FAIL:", p)
    sys.exit(1 if issues else 0)
