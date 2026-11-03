"""Real-photo (field) and not-plant manifests, built from PlantDoc, PlantSeg and COCO."""

import json
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
# Measured on the 4,324 field images: dHash pairs thin out by 6 bits and the noise floor starts near 9.
DHASH_DISTANCE = 6
RAW = Path("datasets") / "raw"
REVIEW = Path("datasets") / "review"
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


def _dhash_variants(path: str) -> list[int]:
    with Image.open(path) as img:
        small = img.convert("L").resize((64, 64))
    out = []
    for flip in (False, True):
        base = small.transpose(Image.FLIP_LEFT_RIGHT) if flip else small
        for k in range(4):
            bits = imagehash.dhash(base.rotate(90 * k)).hash.flatten()
            out.append(int("".join("1" if b else "0" for b in bits), 2))
    return out


def link_same_scene(
    paths: list[str], groups: dict[str, str], max_distance: int = DHASH_DISTANCE
) -> dict[str, str]:
    """Merge groups holding crops or flips of one scene (dHash survives crops, pHash does not)."""
    hashes = np.array([_dhash_variants(p) for p in paths], dtype=np.uint64)
    parent = {g: g for g in set(groups.values())}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for i in range(len(paths) - 1):
        diff = np.bitwise_count(hashes[i, 0] ^ hashes[i + 1 :]).min(axis=1)
        for j in np.flatnonzero(diff <= max_distance):
            a, b = find(groups[paths[i]]), find(groups[paths[i + 1 + j]])
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {p: find(g) for p, g in groups.items()}


def assemble_field(
    inv: pd.DataFrame, class_map: pd.DataFrame, lab: pd.DataFrame, seed: int = 13
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
    df["lab_class"] = [key.get((s, c), "") for s, c in zip(df["source"], df["label"])]
    df = df[df["lab_class"] != ""].sort_values("path")

    groups = group_near_duplicates(list(df["path"]) + list(lab["path"]))
    lab_groups = {groups[p] for p in lab["path"]}
    df["group"] = df["path"].map(groups)
    df = df[~df["group"].isin(lab_groups)]
    df["group"] = df["path"].map(
        link_same_scene(list(df["path"]), dict(zip(df["path"], df["group"])))
    )

    labels_per_group = df.groupby("group")["lab_class"].nunique()
    clash = df["group"].map(labels_per_group) > 1
    conflicts = df[clash][["path", "lab_class", "source", "group"]].rename(
        columns={"lab_class": "label"}
    )
    conflicts["reason"] = "same image or scene carries conflicting labels"
    df = df[~clash].drop_duplicates("sha1")

    df = df.assign(label=df["lab_class"])
    split_map = dict(zip(SPLITS, FIELD_SPLITS))
    out = make_splits(
        df[["path", "label", "group", "sha1", "source"]], seed=seed, ratios=RATIOS
    )
    out["split"] = out["split"].map(split_map)
    complete = out.groupby("label")["split"].nunique().eq(len(FIELD_SPLITS))
    kept = out[out["label"].isin(complete[complete].index)]
    kept = kept.sort_values("path")[FIELD_COLUMNS].reset_index(drop=True)

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
    return kept, cmap[MAP_COLUMNS], conflicts.reset_index(drop=True)


def stage_plantdoc(src: Path, dest: Path) -> None:
    """Flatten PlantDoc's train/ and test/ into dest/<class>/. Long names are shortened."""
    for f in sorted(src.glob("*/*/*")):
        if f.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        out = dest / f.parent.name
        out.mkdir(parents=True, exist_ok=True)
        name = f.name if len(f.name) <= 80 else f"{file_sha1(f)[:12]}{f.suffix.lower()}"
        target = out / name
        if target.exists():
            target = out / f"{f.parent.parent.name}_{name}"
        shutil.copyfile(f, target)


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
    from PIL import Image

    sample = field.sample(n=min(n, len(field)), random_state=seed).reset_index(
        drop=True
    )
    thumbs = REVIEW / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Field sample",
        "",
        f"A random sample of {len(sample)} labelled images from `datasets/manifests/field.csv` "
        f"(seed {seed}). Thumbnails live in the git-ignored `datasets/review/thumbs/`.",
        "",
        "| # | thumbnail | label | source | split | file |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in sample.iterrows():
        with Image.open(r["path"]) as img:
            img = img.convert("RGB")
            img.thumbnail((256, 256))
            img.save(thumbs / f"{i:02d}.jpg")
        lines.append(
            f"| {i} | [thumb](thumbs/{i:02d}.jpg) | {r['label']} | {r['source']} | "
            f"{r['split']} | `{Path(r['path']).name}` |"
        )
    (REVIEW / "field-sample.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_field(sources: dict[str, Path], class_map: pd.DataFrame) -> pd.DataFrame:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, root in sources.items():
        inv = inventory(root)
        inv["source"] = name
        parts.append(inv)
    lab = pd.read_csv(MANIFEST_DIR / "lab.csv")
    field, cmap, conflicts = assemble_field(
        pd.concat(parts, ignore_index=True), class_map, lab
    )
    field.to_csv(MANIFEST_DIR / "field.csv", index=False)
    conflicts.to_csv(MANIFEST_DIR / "field_conflicts.csv", index=False)
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
