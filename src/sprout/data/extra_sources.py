"""Extra real-photo training sources (PlantWild, Plant Pathology 2021), mapped, deduplicated and capped."""

import contextlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from sprout.data.copy_check import COPY_SCORE, copies_of
from sprout.data.field import (
    RESIDUAL_DISTANCE,
    _gray64,
    hash_pairs,
    link_same_scene,
    same_scene_score,
)
from sprout.data.grouping import group_near_duplicates
from sprout.data.inventory import IMAGE_SUFFIXES, file_sha1
from sprout.data.splits import COLUMNS, MANIFEST_DIR, load_manifest

EXTRA_SPLIT = "extra_train"
EXTRA_COLUMNS = COLUMNS + ["source"]
MAP_COLUMNS = [
    "source",
    "source_class",
    "lab_class",
    "status",
    "reason",
    "n_images",
    "n_kept",
]
# Chosen before any E12 result. field_train classes hold at most 136 photos (median about 60),
# so 200 gives thin classes everything while no class can outweigh the real-photo sets.
CAP_PER_CLASS = 200
CLASS_MAP_EXTRA_PATH = MANIFEST_DIR.parent / "class_map_extra.csv"
RAW = Path("datasets") / "raw"
# When a photo matches several field splits, the frozen evaluation splits are named first.
FIELD_PRIORITY = ("field_test", "field_dev", "field_train")
# After the rule ran once, 3 of the 60 hash-close extra/field_dev-or-test pairs scoring highest were viewed:
# two were the same photo at another size (scores 0.83 and 0.785) that the 0.88 rule had missed.
# Extras are plentiful and a leaked copy is costly, so any hash-close pair at or above this score goes too.
GUARD_SCORE = 0.75
UNMAPPED_REASON = "no lab class for this plant or disease"
# Sources from T4.5. A later source never displaces their rows: it loses duplicate groups and
# gets only the room the per-class cap leaves.
ESTABLISHED_SOURCES = ("plantwild", "plantwild_v2")
PP_SOURCE = "plantpathology2021"
# Plant Pathology 2021 is multi-label. Only rows with exactly one label that maps are used.
# "rust" to cedar apple rust is an inference from the name (the mirror says only "rust").
PP_MAP = {
    "scab": "Apple___Apple_scab",
    "healthy": "Apple___healthy",
    "rust": "Apple___Cedar_apple_rust",
}
PP_UNMAPPED_REASON = "no lab class for this apple disease"
PP_MULTI_REASON = "multi-label photo: no single lab class"
JPEG_MAGIC = bytes([0xFF, 0xD8, 0xFF])
_PP_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

# PlantWild class (v1 and v2 use the same names) to lab class. Healthy-leaf classes exist in v1 only.
# "bell pepper leaf spot" follows the PlantDoc mapping, which files it under Bacterial_spot.
PLANTWILD_MAP = {
    "apple black rot": "Apple___Black_rot",
    "apple leaf": "Apple___healthy",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "bell pepper bacterial spot": "Pepper,_bell___Bacterial_spot",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf": "Corn_(maize)___healthy",
    "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
    "peach leaf": "Peach___healthy",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soybean leaf": "Soybean___healthy",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}  # fmt: skip


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract an untrusted zip into a new directory; reject any member that would escape it."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    root = dest.resolve()
    n = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (root / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(
                    f"member escapes the extraction folder: {info.filename}"
                )
        for info in z.infolist():
            if info.is_dir():
                continue
            out = root / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def _map_row(source: str, cls: str) -> tuple[str, str]:
    """(lab class, reason) for one source class; the reason is empty when it maps."""
    if source == PP_SOURCE:
        if cls in PP_MAP:
            return PP_MAP[cls], ""
        return "", PP_MULTI_REASON if "+" in cls else PP_UNMAPPED_REASON
    return PLANTWILD_MAP.get(cls, ""), "" if cls in PLANTWILD_MAP else UNMAPPED_REASON


def make_extra_class_map(classes_by_source: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        (s, c, *_map_row(s, c))
        for s, classes in classes_by_source.items()
        for c in sorted(classes)
    ]
    return pd.DataFrame(rows, columns=["source", "source_class", "lab_class", "reason"])


def _pp_class(label_names) -> str:
    """Source class of one Plant Pathology row: the label, or the sorted labels joined by '+'."""
    return "+".join(sorted(label_names)) or "no label"


def _pp_rows(files, columns):
    import pyarrow.parquet as pq

    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def plant_pathology_counts(files) -> Counter:
    """Rows per source class (single labels, and label sets joined by '+') in the parquet files."""
    return Counter(
        _pp_class(r["label_names"]) for r in _pp_rows(files, ["label_names"])
    )


def extract_plant_pathology(files, dest: Path) -> Counter:
    """Write the images of single-label rows that map into new folder dest/<class>/<image_id>.jpg (the ids carry no suffix in the mirror).

    The parquet files are untrusted: the image bytes are written as they are and never decoded here,
    and an image id that is not a plain file name is rejected. Returns the counts of every row.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    counts: Counter = Counter()
    for r in _pp_rows(files, ["image", "label_names", "image_id"]):
        cls = _pp_class(r["label_names"])
        counts[cls] += 1
        if cls not in PP_MAP:
            continue
        stem = r["image_id"].removesuffix(".jpg")
        if not _PP_SAFE_ID.match(stem):
            raise ValueError(f"unsafe image id: {r['image_id']!r}")
        if not r["image"]["bytes"].startswith(JPEG_MAGIC):
            raise ValueError(f"not a JPEG: {r['image_id']!r}")
        out = dest / cls / f"{stem}.jpg"
        out.parent.mkdir(exist_ok=True)
        with open(out, "xb") as f:
            f.write(r["image"]["bytes"])
    return counts


def extra_inventory(roots: dict[str, Path]) -> pd.DataFrame:
    """One row per image under each root/<class>/: path, label (source class), sha1, source."""
    rows = []
    for source, root in roots.items():
        for cls in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            for f in sorted(cls.iterdir()):
                if f.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append((f.as_posix(), cls.name, file_sha1(f), source))
    return pd.DataFrame(rows, columns=["path", "label", "sha1", "source"])


def _first_split(members: list[str], field_split: dict[str, str]) -> str:
    splits = {field_split[p] for p in members if p in field_split}
    return next((s for s in FIELD_PRIORITY if s in splits), "lab")


def build_extra(
    inv: pd.DataFrame,
    class_map: pd.DataFrame,
    lab: pd.DataFrame,
    field: pd.DataFrame,
    cap: int = CAP_PER_CLASS,
):
    """Map, deduplicate and cap extra real photos for training only.

    inv has path, label (source class), sha1, source; lab and field are the manifests
    (field has split). An image is dropped if it is an exact, near or same-scene copy of
    any lab or field image (rule in docs/field-dedup.md). Among the extras, one image per
    group stays and a group with conflicting lab labels goes whole. Returns (manifest,
    class map with counts, removal counts by reason).
    """
    cmap = class_map.copy()
    cmap["lab_class"] = cmap["lab_class"].fillna("")
    cmap["reason"] = cmap["reason"].fillna("")
    key = cmap.set_index(["source", "source_class"])["lab_class"]
    df = inv.copy()
    df["lab_class"] = [key.get((s, c), "") for s, c in zip(df["source"], df["label"])]
    removed: Counter = Counter()
    removed["unmapped class"] = int((df["lab_class"] == "").sum())
    df["rank"] = (~df["source"].isin(ESTABLISHED_SOURCES)).astype(int)
    df = (
        df[df["lab_class"] != ""]
        .sort_values(["rank", "source", "path"])
        .reset_index(drop=True)
    )

    field_split = dict(zip(field["path"], field["split"]))
    sha_split = dict(zip(field["sha1"], field["split"]))
    lab_sha = set(lab["sha1"])
    drop: dict[str, str] = {}
    for p, s in zip(df["path"], df["sha1"]):
        if s in sha_split:
            drop[p] = f"exact copy of {sha_split[s]}"
        elif s in lab_sha:
            drop[p] = "exact copy of lab"
    left = df[~df["path"].isin(drop)]

    field_paths = list(field["path"])
    groups = group_near_duplicates(list(left["path"]) + field_paths + list(lab["path"]))
    members: dict[str, list[str]] = {}
    for p, g in groups.items():
        members.setdefault(g, []).append(p)
    anchored = set(field_paths) | set(lab["path"])
    for p in left["path"]:
        g = members[groups[p]]
        if any(m in anchored for m in g):
            drop[p] = f"near copy of {_first_split(g, field_split)}"
    left = df[~df["path"].isin(drop)]

    scene = link_same_scene(
        list(left["path"]) + field_paths,
        {p: groups[p] for p in list(left["path"]) + field_paths},
    )
    scene_members: dict[str, list[str]] = {}
    for p, g in scene.items():
        scene_members.setdefault(g, []).append(p)
    for p in left["path"]:
        g = scene_members[scene[p]]
        if any(m in field_split for m in g):
            drop[p] = f"same scene as {_first_split(g, field_split)}"
    left = df[~df["path"].isin(drop)]
    close = hash_pairs(list(left["path"]) + field_paths, RESIDUAL_DISTANCE)
    for a, b in zip(close["path_a"], close["path_b"]):
        if (a in field_split) == (b in field_split):
            continue
        mine, theirs = (b, a) if a in field_split else (a, b)
        if mine not in drop and same_scene_score(_gray64(a), _gray64(b)) >= GUARD_SCORE:
            drop[mine] = f"close match to {field_split[theirs]}"
    left = df[~df["path"].isin(drop)].copy()
    removed.update(Counter(drop.values()))

    left["group"] = left["path"].map(scene)
    per_group = left.groupby("group")["lab_class"].nunique()
    clash = left["group"].map(per_group) > 1
    removed["conflicting labels in one group"] = int(clash.sum())
    left = left[~clash]
    first = ~left.duplicated("group")
    removed["duplicate of another extra image"] = int((~first).sum())
    left = left[first]

    left = left.sort_values(["lab_class", "rank", "sha1"])
    over = left.groupby("lab_class").cumcount() >= cap
    removed["over the per-class cap"] = int(over.sum())
    left = left[~over]

    # own namespace: group ids come from a numbering that field.csv and lab.csv also use
    out = left.assign(
        label=left["lab_class"],
        split=EXTRA_SPLIT,
        group="x" + left["group"].astype(str),
    )
    out = out.sort_values("path")[EXTRA_COLUMNS].reset_index(drop=True)

    n_images, n_kept, status, reason = [], [], [], []
    for r in cmap.itertuples():
        mine = (inv["source"] == r.source) & (inv["label"] == r.source_class)
        n_images.append(int(mine.sum()))
        n_kept.append(int(left[left["path"].isin(inv[mine]["path"])].shape[0]))
        status.append("mapped" if r.lab_class else "excluded")
        reason.append(r.reason or ("" if r.lab_class else UNMAPPED_REASON))
    cmap["status"], cmap["reason"] = status, reason
    cmap["n_images"], cmap["n_kept"] = n_images, n_kept
    return out, cmap[MAP_COLUMNS], dict(removed)


def drop_copies_of_eval_photos(
    inv: pd.DataFrame, class_map: pd.DataFrame, field: pd.DataFrame, device: str
) -> dict[str, str]:
    """Mapped extras at or above COPY_SCORE against any field_dev or field_test photo, to the split."""
    mapped = class_map[class_map["lab_class"].fillna("") != ""]
    keys = set(zip(mapped["source"], mapped["source_class"]))
    todo = [
        p for p, s, c in zip(inv["path"], inv["source"], inv["label"]) if (s, c) in keys
    ]
    held = field[field["split"].isin(["field_dev", "field_test"])]
    split = dict(zip(held["path"], held["split"]))
    return {p: split[r] for p, r in copies_of(todo, list(held["path"]), device).items()}


def drop_crop_copies(
    inv: pd.DataFrame, class_map: pd.DataFrame, field: pd.DataFrame, device: str
) -> dict[str, str]:
    """Mapped extras that are crops, zooms or captioned copies of a field_dev or field_test photo."""
    from sprout.baselines.dino import load_dino_encoder
    from sprout.data.crop_copies import crop_copies

    mapped = class_map[class_map["lab_class"].fillna("") != ""]
    keys = set(zip(mapped["source"], mapped["source_class"]))
    todo = [
        p for p, s, c in zip(inv["path"], inv["source"], inv["label"]) if (s, c) in keys
    ]
    held = field[field["split"].isin(["field_dev", "field_test"])].reset_index(
        drop=True
    )
    encoder = load_dino_encoder().to(device).eval()
    return crop_copies(todo, list(held["path"]), list(held["split"]), encoder, device)


def main(device_scope=None, cap: int = CAP_PER_CLASS, out_name: str = "extra_train") -> dict:
    """device_scope: optional context manager that yields the device name (e.g. holds a GPU share). A different
    `out_name` writes that manifest only and leaves datasets/class_map_extra.csv as it is."""
    roots = {
        "plantwild": RAW / "plantwild" / "plantwild" / "images",
        "plantwild_v2": RAW / "plantwild_v2" / "plantwild_v2",
        PP_SOURCE: RAW / PP_SOURCE,
    }
    shards = sorted((RAW / "plantpathology2021_dl" / "data").glob("*.parquet"))
    pp_counts = (
        plant_pathology_counts(shards)
        if (RAW / PP_SOURCE).exists()
        else extract_plant_pathology(shards, RAW / PP_SOURCE)
    )
    inv = extra_inventory(roots)
    classes = {s: sorted(inv[inv["source"] == s]["label"].unique()) for s in roots}
    classes[PP_SOURCE] = sorted(pp_counts)
    cmap = make_extra_class_map(classes)
    field = load_manifest("field")
    with device_scope() if device_scope else contextlib.nullcontext("cpu") as device:
        copies = drop_copies_of_eval_photos(inv, cmap, field, device)
        crops = drop_crop_copies(inv[~inv["path"].isin(copies)], cmap, field, device)
    out, cmap_out, removed = build_extra(
        inv[~inv["path"].isin(set(copies) | set(crops))],
        cmap,
        load_manifest("lab"),
        field,
        cap,
    )
    for split in ("field_dev", "field_test"):
        removed[f"crop or caption copy of {split}"] = sum(
            v == split for v in crops.values()
        )
    for split in ("field_dev", "field_test"):
        removed[f"correlation >= {COPY_SCORE} with {split}"] = sum(
            v == split for v in copies.values()
        )
    full = inv.groupby(["source", "label"]).size()
    full = full.to_dict() | {(PP_SOURCE, c): n for c, n in pp_counts.items()}
    cmap_out["n_images"] = [
        int(full.get((r.source, r.source_class), 0)) for r in cmap_out.itertuples()
    ]
    out.to_csv(MANIFEST_DIR / f"{out_name}.csv", index=False)
    if out_name == "extra_train":
        cmap_out.to_csv(CLASS_MAP_EXTRA_PATH, index=False)
    summary = {"device": device, "inventory": len(inv), "kept": len(out), "removed": removed}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
