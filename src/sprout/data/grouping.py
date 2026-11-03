import re
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# Suffixes the dataset adds to copies of one source photo, and the segmented copy marker.
_COPY_SUFFIX = re.compile(r"(_final_masked|_flip(LR|TB)|_(90|180|270)deg|_new\w*)+$", re.I)


def source_id(path) -> str:
    """Filename stem with copy, flip and segmentation suffixes removed."""
    return _COPY_SUFFIX.sub("", Path(path).stem)


def _variants(img: Image.Image) -> list[int]:
    """phash of the image under all 8 flips and 90-degree rotations."""
    small = img.convert("L").resize((64, 64))
    out = []
    for flip in (False, True):
        base = small.transpose(Image.FLIP_LEFT_RIGHT) if flip else small
        for k in range(4):
            h = imagehash.phash(base.rotate(90 * k))
            out.append(int("".join("1" if b else "0" for b in h.hash.flatten()), 2))
    return out


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def group_near_duplicates(paths, max_distance: int = 4) -> dict[str, str]:
    paths = sorted(str(p) for p in paths)
    n = len(paths)
    parent = list(range(n))

    def union(a: int, b: int) -> None:
        parent[_find(parent, a)] = _find(parent, b)

    by_id = defaultdict(list)
    for i, p in enumerate(paths):
        by_id[source_id(p)].append(i)
    for members in by_id.values():
        for m in members[1:]:
            union(members[0], m)

    base, variants = [], []
    for p in paths:
        with Image.open(p) as img:
            v = _variants(img)
        base.append(v[0])
        variants.append(v)

    # Pigeonhole: hashes within max_distance of 64 bits agree exactly on at least
    # one of (max_distance + 1) chunks, so only chunk-sharing pairs are checked.
    chunks = max_distance + 1
    bounds = np.linspace(0, 64, chunks + 1).astype(int)
    for c in range(chunks):
        lo, width = bounds[c], bounds[c + 1] - bounds[c]
        mask = (1 << int(width)) - 1
        buckets = defaultdict(set)
        for j, vs in enumerate(variants):
            for h in vs:
                buckets[(h >> int(lo)) & mask].add(j)
        for i, h in enumerate(base):
            for j in buckets[(h >> int(lo)) & mask]:
                if j > i and _find(parent, i) != _find(parent, j):
                    if any((h ^ v).bit_count() <= max_distance for v in variants[j]):
                        union(i, j)

    ids: dict[int, str] = {}
    out = {}
    for i, p in enumerate(paths):
        root = _find(parent, i)
        ids.setdefault(root, f"g{len(ids):06d}")
        out[p] = ids[root]
    return out
