import re
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# Surveyed in the official release: copies end in " copy" with an optional number (" copy 2" to " copy 5", also " copy3", " copy  3", " copy ", and "0626copy" with no space), and the
# segmented file adds "_final_masked". The leading uuid differs per file, and the code after
# "___" (for example "JR_HL 9885") repeats across classes for unrelated photos, so the
# source ID is the class folder plus that code.
_COPY_SUFFIX = re.compile(r"(?<=[\s0-9])copy\s*\d*\s*$")


def source_id(path) -> str:
    p = Path(path)
    stem = re.sub(r"_final_masked$", "", p.stem).split("___", 1)[-1]
    return f"{p.parent.name}/{_COPY_SUFFIX.sub('', stem).rstrip()}"


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


def _union_same_source(paths: list[str], parent: list[int]) -> None:
    by_id = defaultdict(list)
    for i, p in enumerate(paths):
        by_id[source_id(p)].append(i)
    for members in by_id.values():
        for m in members[1:]:
            parent[_find(parent, members[0])] = _find(parent, m)


def _union_close_hashes(variants: list[list[int]], parent: list[int], max_distance: int) -> None:
    """Join images whose pHash is within max_distance bits of any symmetry of the other.

    Pigeonhole: hashes within max_distance of 64 bits agree exactly on at least
    one of (max_distance + 1) chunks, so only chunk-sharing pairs are checked.
    """
    base = [v[0] for v in variants]
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
                        parent[_find(parent, i)] = _find(parent, j)


def group_near_duplicates(paths, max_distance: int = 4) -> dict[str, str]:
    paths = sorted(str(p) for p in paths)
    parent = list(range(len(paths)))
    _union_same_source(paths, parent)
    variants = []
    for p in paths:
        with Image.open(p) as img:
            variants.append(_variants(img))
    _union_close_hashes(variants, parent, max_distance)

    ids: dict[int, str] = {}
    out = {}
    for i, p in enumerate(paths):
        root = _find(parent, i)
        ids.setdefault(root, f"g{len(ids):06d}")
        out[p] = ids[root]
    return out
