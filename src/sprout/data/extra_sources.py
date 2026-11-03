"""Extra real-photo training sources (PlantWild), mapped, deduplicated and capped."""

import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from sprout.data.field import link_same_scene
from sprout.data.grouping import group_near_duplicates
from sprout.data.inventory import IMAGE_SUFFIXES, file_sha1
from sprout.data.splits import COLUMNS, MANIFEST_DIR, load_manifest

EXTRA_SPLIT = "extra_train"
EXTRA_COLUMNS = COLUMNS + ["source"]
MAP_COLUMNS = ["source", "source_class", "lab_class", "status", "reason", "n_images", "n_kept"]
# Chosen before any E12 result. field_train classes hold at most 136 photos (median about 60),
# so 200 gives thin classes everything while no class can outweigh the real-photo sets.
CAP_PER_CLASS = 200
CLASS_MAP_EXTRA_PATH = MANIFEST_DIR.parent / "class_map_extra.csv"
RAW = Path("datasets") / "raw"
# When a photo matches several field splits, the frozen evaluation splits are named first.
FIELD_PRIORITY = ("field_test", "field_dev", "field_train")
UNMAPPED_REASON = "no lab class for this plant or disease"

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
                raise ValueError(f"member escapes the extraction folder: {info.filename}")
        for info in z.infolist():
            if info.is_dir():
                continue
            out = root / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def make_extra_class_map(classes_by_source: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        (s, c, PLANTWILD_MAP.get(c, ""), "" if c in PLANTWILD_MAP else UNMAPPED_REASON)
        for s, classes in classes_by_source.items()
        for c in sorted(classes)
    ]
    return pd.DataFrame(rows, columns=["source", "source_class", "lab_class", "reason"])


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
    df = df[df["lab_class"] != ""].sort_values(["source", "path"]).reset_index(drop=True)

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

    left = left.sort_values(["lab_class", "sha1"])
    over = left.groupby("lab_class").cumcount() >= cap
    removed["over the per-class cap"] = int(over.sum())
    left = left[~over]

    out = left.assign(label=left["lab_class"], split=EXTRA_SPLIT)
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


def main() -> None:
    roots = {
        "plantwild": RAW / "plantwild" / "plantwild" / "images",
        "plantwild_v2": RAW / "plantwild_v2" / "plantwild_v2",
    }
    inv = extra_inventory(roots)
    classes = {s: sorted(inv[inv["source"] == s]["label"].unique()) for s in roots}
    out, cmap, removed = build_extra(
        inv, make_extra_class_map(classes), load_manifest("lab"), load_manifest("field")
    )
    out.to_csv(MANIFEST_DIR / "extra_train.csv", index=False)
    cmap.to_csv(CLASS_MAP_EXTRA_PATH, index=False)
    print(json.dumps({"inventory": len(inv), "kept": len(out), "removed": removed}, indent=1))


if __name__ == "__main__":
    main()
