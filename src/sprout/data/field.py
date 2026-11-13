"""Real-photo (field) and not-plant manifests, built from PlantDoc, PlantSeg and COCO."""

import json
import re
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from sprout.data.grouping import group_near_duplicates
from sprout.data.inventory import IMAGE_SUFFIXES, file_sha1, inventory
from sprout.data.splits import COLUMNS, MANIFEST_DIR, SPLITS, make_splits

FIELD_SPLITS = ("field_train", "field_dev", "field_test")
FIELD_COLUMNS = COLUMNS + ["source"]
MAP_COLUMNS = [
    "source",
    "source_class",
    "lab_class",
    "status",
    "reason",
    "n_images",
    "n_field_test",
]
RATIOS = (0.5, 0.25, 0.25)
# Chosen from viewing candidate pairs (see docs/field-dedup.md): hashes propose, correlation decides.
HASH_CANDIDATES = 12
SCENE_MATCH = 0.88
SCENE_MATCH_CLOSE = 0.80
CLOSE_DHASH = 10
RESIDUAL_DISTANCE = 8
RAW = Path("datasets") / "raw"
REVIEW = Path("datasets") / "review"
THUMB_NAME = re.compile(r"\d\d(-[0-9a-f]+)?\.jpg")
CLASS_MAP_PATH = MANIFEST_DIR.parent / "class_map.csv"

PLANTDOC_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

PLANTSEG_MAP = {
    "apple black rot": "Apple___Black_rot",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "bell pepper bacterial spot": "Pepper,_bell___Bacterial_spot",
    "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
# PlantSeg plants whose crop has a lab class; other plants have none.
PLANTSEG_LAB_PLANTS = {
    "Apple", "Bell pepper", "Blueberry", "Cherry", "Citrus", "Corn", "Grape", "Peach",
    "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato",
}  # fmt: skip


def not_plant_images(
    instances_json, image_dir, exclude: str = "potted plant"
) -> list[Path]:
    """COCO images with no annotation of the excluded category."""
    data = json.loads(Path(instances_json).read_text())
    banned = {c["id"] for c in data["categories"] if c["name"] == exclude}
    bad = {a["image_id"] for a in data["annotations"] if a["category_id"] in banned}
    files = sorted(i["file_name"] for i in data["images"] if i["id"] not in bad)
    return [Path(image_dir) / f for f in files]


def build_not_plant(instances_json, image_dir, name: str = "not_plant") -> pd.DataFrame:
    rows = [
        (p.as_posix(), "not_plant", file_sha1(p))
        for p in not_plant_images(instances_json, image_dir)
    ]
    df = pd.DataFrame(rows, columns=["path", "label", "sha1"])
    df["group"] = [f"n{i:06d}" for i in range(len(df))]
    out = make_splits(df, ratios=RATIOS)
    out["split"] = out["split"].str.replace("lab_", "not_plant_")
    out = out[COLUMNS]
    out.to_csv(MANIFEST_DIR / f"{name}.csv", index=False)
    return out


def _gray64(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(
            img.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.uint8
        )


def _hash_variants(path: str) -> tuple[list[int], list[int]]:
    """pHash and dHash of the image under all 8 flips and 90-degree rotations."""
    small = Image.fromarray(_gray64(path))
    ph, dh = [], []
    for flip in (False, True):
        base = small.transpose(Image.FLIP_LEFT_RIGHT) if flip else small
        for k in range(4):
            turned = base.rotate(90 * k)
            for hasher, out in ((imagehash.phash, ph), (imagehash.dhash, dh)):
                bits = hasher(turned).hash.flatten()
                out.append(int("".join("1" if b else "0" for b in bits), 2))
    return ph, dh


def hash_pairs(paths: list[str], max_distance: int) -> pd.DataFrame:
    """Pairs of images within max_distance bits by pHash or dHash, over all 8 symmetries."""
    both = [_hash_variants(p) for p in paths]
    ph = np.array([b[0] for b in both], dtype=np.uint64)
    dh = np.array([b[1] for b in both], dtype=np.uint64)
    return pairs_from_hashes(paths, ph, dh, max_distance)


def pairs_from_hashes(paths: list[str], ph: np.ndarray, dh: np.ndarray, max_distance: int) -> pd.DataFrame:
    """hash_pairs from precomputed (n, 8) pHash and dHash arrays, one row of 8 symmetries per path."""
    rows = []
    for i in range(len(paths) - 1):
        dp = np.bitwise_count(ph[i, 0] ^ ph[i + 1 :]).min(axis=1)
        dd = np.bitwise_count(dh[i, 0] ^ dh[i + 1 :]).min(axis=1)
        for j in np.flatnonzero((dp <= max_distance) | (dd <= max_distance)):
            rows.append((paths[i], paths[i + 1 + j], int(dp[j]), int(dd[j])))
    return pd.DataFrame(rows, columns=["path_a", "path_b", "phash", "dhash"])


def _small(a: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray(a).resize((32, 32), Image.BILINEAR), dtype=np.float32
    )


def _crops(a: np.ndarray):
    yield _small(a)
    for frac in (0.85, 0.7):
        w = int(64 * frac)
        for y in (0, (64 - w) // 2, 64 - w):
            for x in (0, (64 - w) // 2, 64 - w):
                yield _small(a[y : y + w, x : x + w])


def _foreground_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation over pixels that are not white in both images, so blank backgrounds don't match."""
    keep = np.maximum(a, b) < 235
    if keep.sum() < 150:
        return 0.0
    x, y = a[keep] - a[keep].mean(), b[keep] - b[keep].mean()
    norm = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / norm) if norm > 0 else 0.0


def same_scene_score(a: np.ndarray, b: np.ndarray) -> float:
    """Best foreground correlation of two 64x64 gray images over flips, rotations and crops."""
    whole_a = _small(a)
    best = 0.0
    for flip in (b, b[:, ::-1]):
        for k in range(4):
            turned = np.ascontiguousarray(np.rot90(flip, k))
            whole_b = _small(turned)
            best = max(best, _foreground_ncc(whole_a, whole_b))
            for crop in _crops(turned):
                best = max(best, _foreground_ncc(whole_a, crop))
            for crop in _crops(a):
                best = max(best, _foreground_ncc(crop, whole_b))
    return best


def is_same_scene(score: float, dhash: int) -> bool:
    return score >= SCENE_MATCH or (score >= SCENE_MATCH_CLOSE and dhash <= CLOSE_DHASH)


def link_same_scene(
    paths: list[str], groups: dict[str, str], hash_distance: int = HASH_CANDIDATES
) -> dict[str, str]:
    """Merge groups of images that verify as one scene (same photo, crop or flip).

    Hashes only propose pairs. A pair links only when same_scene_score confirms it,
    so a chain of loose hash matches cannot join unrelated images.
    """
    parent = {g: g for g in set(groups.values())}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    gray: dict[str, np.ndarray] = {}
    for r in hash_pairs(paths, hash_distance).itertuples():
        a, b = find(groups[r.path_a]), find(groups[r.path_b])
        if a == b:
            continue
        for p in (r.path_a, r.path_b):
            gray.setdefault(p, _gray64(p))
        if is_same_scene(same_scene_score(gray[r.path_a], gray[r.path_b]), r.dhash):
            parent[max(a, b)] = min(a, b)
    return {p: find(g) for p, g in groups.items()}


def merge_reviewed_links(groups: dict[str, str], links: pd.DataFrame) -> dict[str, str]:
    """Merge the groups of hand-reviewed same-photo pairs. Pairs not in groups are skipped."""
    out = dict(groups)
    for a, b in zip(links["path_a"], links["path_b"], strict=True):
        if a in out and b in out and out[a] != out[b]:
            keep, drop = sorted((out[a], out[b]))
            out = {p: (keep if g == drop else g) for p, g in out.items()}
    return out


GENERIC_NAMES = {
    "image",
    "img",
    "photo",
    "picture",
    "pic",
    "thumb",
    "thumbnail",
    "untitled",
}


def is_generic_name(name: str) -> bool:
    """A name that says nothing about the picture: only digits and separators, or a fixed word."""
    stem = name.lower().split(".")[0]
    return re.fullmatch(r"[\d_\- ]*", stem) is not None or stem in GENERIC_NAMES


def unreviewed_same_names(field: pd.DataFrame, reviews: pd.DataFrame) -> list[str]:
    """Descriptive file names shared by more than one image that have no recorded judgment."""
    names = pd.Series([p.rsplit("/", 1)[-1].lower() for p in field["path"]])
    shared = names[names.duplicated(keep=False)].unique()
    judged = set(reviews["name"])
    return sorted(n for n in shared if not is_generic_name(n) and n not in judged)


def residual_pairs(
    field: pd.DataFrame,
    links: pd.DataFrame | None = None,
    max_distance: int = RESIDUAL_DISTANCE,
) -> pd.DataFrame:
    """Hash-close image pairs the pipeline left in different splits, with their scene score."""
    split = dict(zip(field["path"], field["split"], strict=True))
    label = dict(zip(field["path"], field["label"], strict=True))
    pairs = hash_pairs(list(field["path"]), max_distance)
    across = [split[a] != split[b] for a, b in zip(pairs["path_a"], pairs["path_b"], strict=True)]
    pairs = pairs[pd.Series(across, index=pairs.index, dtype=bool)]
    pairs = pairs.reset_index(drop=True)
    pairs["split_a"] = pairs["path_a"].map(split)
    pairs["split_b"] = pairs["path_b"].map(split)
    pairs["label_a"] = pairs["path_a"].map(label)
    pairs["label_b"] = pairs["path_b"].map(label)
    linked = set()
    if links is not None:
        linked = set(zip(links["path_a"], links["path_b"], strict=True))
    pairs["reviewed_same_photo"] = [
        "yes" if (a, b) in linked or (b, a) in linked else "no"
        for a, b in zip(pairs["path_a"], pairs["path_b"], strict=True)
    ]
    pairs["scene_score"] = [
        round(same_scene_score(_gray64(a), _gray64(b)), 3)
        for a, b in zip(pairs["path_a"], pairs["path_b"], strict=True)
    ]
    return pairs


def assemble_field(
    inv: pd.DataFrame,
    class_map: pd.DataFrame,
    lab: pd.DataFrame,
    seed: int = 13,
    reviewed_links: pd.DataFrame | None = None,
    reviewed_removals: pd.DataFrame | None = None,
):
    """Map, deduplicate, group and split real photos.

    inv has path, label (the source class), sha1, source. class_map has source,
    source_class, lab_class (blank when unmappable) and reason. lab has path and sha1.
    Returns (field manifest, class map with status, n_images and n_field_test filled in,
    conflicting duplicates that were removed).
    """
    cmap = class_map.copy()
    cmap["lab_class"] = cmap["lab_class"].fillna("")
    cmap["reason"] = cmap["reason"].fillna("")
    key = cmap.set_index(["source", "source_class"])["lab_class"]
    df = inv.copy()
    df["lab_class"] = [key.get((s, c), "") for s, c in zip(df["source"], df["label"], strict=True)]
    df = df[df["lab_class"] != ""].sort_values("path")

    groups = group_near_duplicates(list(df["path"]) + list(lab["path"]))
    lab_groups = {groups[p] for p in lab["path"]}
    df["group"] = df["path"].map(groups)
    df = df[~df["group"].isin(lab_groups)]
    linked = link_same_scene(list(df["path"]), dict(zip(df["path"], df["group"], strict=True)))
    if reviewed_links is not None:
        linked = merge_reviewed_links(linked, reviewed_links)
    df["group"] = df["path"].map(linked)

    labels_per_group = df.groupby("group")["lab_class"].nunique()
    clash = df["group"].map(labels_per_group) > 1
    conflicts = df[clash][["path", "lab_class", "source", "group"]].rename(
        columns={"lab_class": "label"}
    )
    conflicts["reason"] = "same image or scene carries conflicting labels"
    df = df[~clash].drop_duplicates("sha1")

    df = df.assign(label=df["lab_class"])
    split_map = dict(zip(SPLITS, FIELD_SPLITS, strict=True))
    out = make_splits(
        df[["path", "label", "group", "sha1", "source"]], seed=seed, ratios=RATIOS
    )
    out["split"] = out["split"].map(split_map)
    if reviewed_removals is not None:
        # after assignment, so removing an image never moves another one
        gone = out["path"].isin(set(reviewed_removals["path"]))
        removed = out[gone][["path", "label", "source", "group"]].merge(
            reviewed_removals[["path", "reason"]], on="path"
        )
        conflicts = pd.concat([conflicts, removed], ignore_index=True)
        out = out[~gone]
    complete = out.groupby("label")["split"].nunique().eq(len(FIELD_SPLITS))
    kept = out[out["label"].isin(complete[complete].index)]
    kept = kept.sort_values("path")[FIELD_COLUMNS].reset_index(drop=True)

    return kept, _with_status(cmap, inv, out, kept, complete), conflicts.reset_index(drop=True)


def _with_status(
    cmap: pd.DataFrame,
    inv: pd.DataFrame,
    out: pd.DataFrame,
    kept: pd.DataFrame,
    complete: pd.Series,
) -> pd.DataFrame:
    """The class map with status (excluded, shared or dropped), reason, n_images and n_field_test filled in."""
    cmap = cmap.copy()
    stats = out.groupby("label").agg(
        images=("path", "size"), groups=("group", "nunique")
    )
    status, reason, n_images, n_test = [], [], [], []
    for r in cmap.itertuples():
        n = int(((inv["source"] == r.source) & (inv["label"] == r.source_class)).sum())
        n_images.append(n)
        n_test.append(0)
        if not r.lab_class:
            status.append("excluded")
            reason.append(r.reason or "no matching lab class")
        elif r.lab_class in complete[complete].index:
            status.append("shared")
            reason.append(r.reason)
            n_test[-1] = int(
                kept[(kept["source"] == r.source) & (kept["label"] == r.lab_class)][
                    "split"
                ]
                .eq("field_test")
                .sum()
            )
        else:
            s = stats.loc[r.lab_class] if r.lab_class in stats.index else None
            have = (
                f"{int(s.images)} images in {int(s.groups)} groups"
                if s is not None
                else "no images"
            )
            status.append("dropped")
            reason.append(f"lab class cannot appear in all three field splits ({have})")
    cmap["status"], cmap["reason"] = status, reason
    cmap["n_images"], cmap["n_field_test"] = n_images, n_test
    return cmap[MAP_COLUMNS]


def stage_plantseg(src: Path, dest: Path) -> pd.DataFrame:
    """Copy PlantSeg images into dest/<disease>/ and return its metadata (Plant, Disease)."""
    meta = pd.read_csv(src / "Metadata.csv", encoding="utf-8-sig")
    where = {
        f.name: f
        for f in (src / "images").glob("*/*")
        if f.suffix.lower() in IMAGE_SUFFIXES
    }
    for r in meta.itertuples():
        out = dest / r.Disease
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(where[r.Name], out / r.Name)
    return meta


def make_class_map(plantdoc_classes, plantseg_meta: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "plantdoc",
            c,
            PLANTDOC_MAP.get(c, ""),
            "" if c in PLANTDOC_MAP else "no matching lab class",
        )
        for c in plantdoc_classes
    ]
    plants = plantseg_meta.drop_duplicates("Disease").set_index("Disease")["Plant"]
    for disease in sorted(plants.index):
        if disease in PLANTSEG_MAP:
            rows.append(("plantseg", disease, PLANTSEG_MAP[disease], ""))
        elif plants[disease] in PLANTSEG_LAB_PLANTS:
            rows.append(("plantseg", disease, "", "disease has no lab class"))
        else:
            rows.append(("plantseg", disease, "", "plant has no lab class"))
    return pd.DataFrame(rows, columns=["source", "source_class", "lab_class", "reason"])


def write_review(field: pd.DataFrame, n: int = 50, seed: int = 13) -> None:
    """Write a seeded sample as thumbnails and a table that name each other.

    A thumbnail's name carries the row's image ID (its sha1) and the table lists the same ID, so a
    thumbnail and a table row from different runs cannot be mistaken for each other. Thumbnails left by
    an earlier run are removed.
    """
    sample = field.sample(n=min(n, len(field)), random_state=seed).reset_index(
        drop=True
    )
    thumbs = REVIEW / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Field sample",
        "",
        f"A random sample of {len(sample)} labelled images from `datasets/manifests/field.csv` "
        f"(seed {seed}). Thumbnails live in the git-ignored `datasets/review/thumbs/`; "
        "each is named `<#>-<first 10 characters of the image ID>.jpg`, and the last column is the full ID (sha1).",
        "",
        "| # | thumbnail | label | source | split | file | image ID |",
        "|---|---|---|---|---|---|---|",
    ]
    written = set()
    for i, r in sample.iterrows():
        name = f"{i:02d}-{r['sha1'][:10]}.jpg"
        with Image.open(r["path"]) as img:
            img = img.convert("RGB")
            img.thumbnail((256, 256))
            img.save(thumbs / name)
        written.add(name)
        lines.append(
            f"| {i} | [thumb](thumbs/{name}) | {r['label']} | {r['source']} | "
            f"{r['split']} | `{Path(r['path']).name}` | `{r['sha1']}` |"
        )
    for old in thumbs.iterdir():
        if THUMB_NAME.fullmatch(old.name) and old.name not in written:
            old.unlink()
    (REVIEW / "field-sample.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_field(sources: dict[str, Path], class_map: pd.DataFrame) -> pd.DataFrame:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, root in sources.items():
        inv = inventory(root)
        inv["source"] = name
        parts.append(inv)
    lab = pd.read_csv(MANIFEST_DIR / "lab.csv")
    links = pd.read_csv(MANIFEST_DIR / "field_reviewed_links.csv")
    removals = pd.read_csv(MANIFEST_DIR / "field_reviewed_removals.csv")
    field, cmap, conflicts = assemble_field(
        pd.concat(parts, ignore_index=True),
        class_map,
        lab,
        reviewed_links=links,
        reviewed_removals=removals,
    )
    field.to_csv(MANIFEST_DIR / "field.csv", index=False)
    conflicts.to_csv(MANIFEST_DIR / "field_conflicts.csv", index=False)
    residual_pairs(field, links).to_csv(
        MANIFEST_DIR / "field_residual_pairs.csv", index=False
    )
    cmap.to_csv(CLASS_MAP_PATH, index=False)
    write_review(field)
    return field


if __name__ == "__main__":
    meta = stage_plantseg(RAW / "plantseg" / "plantseg", RAW / "plantseg_field")
    dirs = {"plantdoc": RAW / "plantdoc", "plantseg": RAW / "plantseg_field"}
    cmap = make_class_map(sorted(p.name for p in dirs["plantdoc"].iterdir()), meta)
    build_field(dirs, cmap)
    build_not_plant(
        RAW / "coco_ann" / "annotations" / "instances_val2017.json",
        RAW / "coco_val" / "val2017",
    )
