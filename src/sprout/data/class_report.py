"""Class report and document-coverage page, built from the manifests and a committed review file.

Reads only CSV rows. No image file is opened, and no `field_test` image or score is read: the
`field_test` figures are row counts by class, which is the plan's frozen list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sprout.web.crops import crop_for_label

ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "datasets" / "manifests"
COVERAGE_CSV = ROOT / "docs" / "document-coverage.csv"
KNOWLEDGE_MANIFEST = ROOT / "knowledge" / "manifest.csv"
MIN_TRAIN, MIN_DEV, MIN_TEST = 80, 20, 20
SOURCES = ("plantdoc", "plantseg")
SPLIT_NAMES = ("train", "dev", "test")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(MANIFESTS / f"{name}.csv")


def source_of_path(path: str) -> str:
    """Source from a raw path such as datasets/raw/plantseg_field/..., or '' if unknown."""
    parts = path.split("/")
    folder = parts[2] if len(parts) > 2 else ""
    return next((s for s in SOURCES if folder.startswith(s)), "")


def class_table(lab: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    """One row per lab class: lab counts, real counts by source and split, and a thin flag."""
    labels = sorted(set(lab["label"]) | set(field["label"]))
    out = pd.DataFrame(index=pd.Index(labels, name="label"))
    for split in SPLIT_NAMES:
        out[f"lab_{split}"] = lab[lab["split"] == f"lab_{split}"].groupby("label").size()
        for source in SOURCES:
            rows = field[(field["source"] == source) & (field["split"] == f"field_{split}")]
            out[f"{source}_{split}"] = rows.groupby("label").size()
    out = out.fillna(0).astype(int)
    for split in SPLIT_NAMES:
        out[f"real_{split}"] = sum(out[f"{s}_{split}"] for s in SOURCES)
    out["thin"] = (
        (out["real_train"] < MIN_TRAIN) | (out["real_dev"] < MIN_DEV) | (out["real_test"] < MIN_TEST)
    )
    return out


def thin_reasons(row: pd.Series) -> str:
    parts = []
    for split, minimum in zip(SPLIT_NAMES, (MIN_TRAIN, MIN_DEV, MIN_TEST), strict=True):
        if row[f"real_{split}"] < minimum:
            parts.append(f"{split} {row[f'real_{split}']} < {minimum}")
    return "; ".join(parts)


def label_conflicts(conflicts: pd.DataFrame) -> dict:
    """Images removed for conflicting labels, by group. Not duplicates across sources."""
    per_group = conflicts.groupby("group")["source"].nunique()
    return {
        "images": len(conflicts),
        "images_by_source": conflicts.groupby("source").size(),
        "groups": len(per_group),
        "single_source_groups": int((per_group == 1).sum()),
        "two_source_groups": int((per_group > 1).sum()),
    }


def cross_source(field: pd.DataFrame, links: pd.DataFrame) -> dict:
    """Counts of images shared across sources, from manifest rows only."""
    n_sources = field.groupby("group")["source"].nunique()
    spanning = set(n_sources.index[n_sources > 1])
    span_rows = field[field["group"].isin(spanning)]
    cross = sum(
        1
        for a, b in zip(links["path_a"], links["path_b"], strict=True)
        if source_of_path(a) != source_of_path(b)
    )
    return {
        "spanning_groups": len(spanning),
        "spanning_images": len(span_rows),
        "spanning_by_label": span_rows.groupby("label").size().sort_index(),
        "links": len(links),
        "links_cross_source": cross,
    }


def healthy_by_crop(table: pd.DataFrame) -> pd.DataFrame:
    """Per crop: its healthy class's real photos, and the real photos of all its classes."""
    real_cols = [f"real_{p}" for p in SPLIT_NAMES]
    crop_of = table.index.str.split("___", n=1).str[0]
    rows = []
    for crop in sorted(set(crop_of)):
        healthy = f"{crop}___healthy"
        total = int(table.loc[crop_of == crop, real_cols].to_numpy().sum())
        if healthy in table.index:
            r = table.loc[healthy]
            h = [int(r[c]) for c in real_cols]
            rows.append((crop, True, sum(h), *h, total))
        else:
            rows.append((crop, False, 0, 0, 0, 0, total))
    return pd.DataFrame(
        rows,
        columns=[
            "crop",
            "has_healthy_class",
            "healthy_real_photos",
            "healthy_real_train",
            "healthy_real_dev",
            "healthy_real_test",
            "all_classes_real_photos",
        ],
    )


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    return head + body


def render_report(
    lab: pd.DataFrame, field: pd.DataFrame, conflicts: pd.DataFrame, links: pd.DataFrame
) -> str:
    table = class_table(lab, field)
    dup = cross_source(field, links)
    conf = label_conflicts(conflicts)
    healthy = healthy_by_crop(table)
    lines = [
        "# Class report",
        "",
        "Generated by `python -m sprout.data.class_report` from `datasets/manifests/` only. The numbers",
        "are counts of manifest rows, not results. No image file was opened. For `field_test` only the",
        "row count per class is used (the frozen list); no `field_test` image or score was read.",
        "",
        f"Real photos are `field.csv` (PlantDoc and PlantSeg). Minimum per class: {MIN_TRAIN} train, "
        f"{MIN_DEV} dev, {MIN_TEST} test real photos. No class is dropped from `field_test`; classes",
        "under the minimum are listed as thin and kept.",
        "",
        f"## Photos per class ({len(table)} classes, {int(table['thin'].sum())} thin)",
        "",
    ]
    cols = ["lab_train", "lab_dev", "lab_test"]
    cols += [f"{s}_{p}" for s in SOURCES for p in SPLIT_NAMES]
    cols += [f"real_{p}" for p in SPLIT_NAMES]
    show = table[cols].reset_index()
    show["thin"] = ["yes" if t else "" for t in table["thin"]]
    lines += [_md_table(show), "", "## Thin classes", ""]
    thin = table[table["thin"]]
    reasons = [thin_reasons(r) for _, r in thin.iterrows()]
    lines += [_md_table(pd.DataFrame({"class": thin.index, "reason": reasons})), ""]
    real_cols = [f"real_{p}" for p in SPLIT_NAMES]
    zero = [c for c in table.index if table.loc[c, real_cols].sum() == 0]
    lines += ["Classes with no real photo at all: " + (", ".join(zero) or "none") + ".", ""]
    lines += ["## Crops and their healthy class", "",
        "`healthy_real_*` counts the crop's healthy class only; `all_classes_real_photos` counts every class of the crop.", "", _md_table(healthy), ""]
    has = healthy["has_healthy_class"]
    with_real = healthy[has & (healthy["healthy_real_photos"] > 0)]["crop"].tolist()
    no_real = healthy[has & (healthy["healthy_real_photos"] == 0)]["crop"].tolist()
    no_class = healthy[~has]["crop"].tolist()
    lines += [
        "Healthy class with real photos: " + (", ".join(with_real) or "none") + ".",
        "Healthy class but no real photos: " + (", ".join(no_real) or "none") + ".",
        "No healthy class in the lab data: " + (", ".join(no_class) or "none") + ".",
        "",
        "## Cross-source duplicates",
        "",
        f"- Groups in `field.csv` holding images from both sources: {dup['spanning_groups']} "
        f"({dup['spanning_images']} images). Each group sits whole in one split.",
        f"- `field_reviewed_links.csv`: {dup['links']} hand-reviewed same-photo links, "
        f"{dup['links_cross_source']} of them across sources.",
        "",
        "Images in cross-source groups, by class:",
        "",
        _md_table(dup["spanning_by_label"].rename("images").reset_index()),
        "",
        "## Label-conflict images",
        "",
        "These are not cross-source duplicates. Each row in `field_conflicts.csv` is an image whose",
        "scene or photo also carries a different label, and it is kept out of `field.csv`.",
        "",
        f"- Images in `field_conflicts.csv`: {conf['images']} ("
        + ", ".join(f"{k} {v}" for k, v in conf["images_by_source"].items())
        + ").",
        f"- Conflict groups: {conf['groups']}. Groups whose images all come from one source: "
        f"{conf['single_source_groups']}. Groups with images from both sources: "
        f"{conf['two_source_groups']}.",
    ]
    return "\n".join(lines)


def load_coverage() -> pd.DataFrame:
    return pd.read_csv(COVERAGE_CSV, keep_default_na=False)


def render_coverage(rows: pd.DataFrame, manifest: pd.DataFrame) -> str:
    counts = rows["symptoms_described"].value_counts()
    crops_with_docs = sorted(set(manifest["crop"]) - {"general"})
    lines = [
        "# Document coverage",
        "",
        "For each class the classifier can output: does a document in `knowledge/manifest.csv`",
        "describe its symptoms? Judged by reading the indexed passages for that crop and condition,",
        "not from the manifest's disease list. The evidence is in `docs/document-coverage.csv`; this",
        "page is rendered from it. A disease named only in a spray table or a passing comparison does",
        "not count. A healthy class has no symptoms to describe and is marked not applicable.",
        "",
        f"Yes: {counts.get('yes', 0)}. No: {counts.get('no', 0)}. Not applicable: "
        f"{counts.get('not applicable', 0)}. Crops with at least one document: "
        f"{', '.join(crops_with_docs)}.",
        "",
        _md_table(rows[["label", "crop", "symptoms_described", "document", "page", "evidence"]]),
    ]
    return "\n".join(lines)


def class_crop(label: str) -> str | None:
    known = set(pd.read_csv(KNOWLEDGE_MANIFEST)["crop"])
    return crop_for_label(label, known)


def main() -> None:
    lab, field = _read("lab"), _read("field")
    report = render_report(lab, field, _read("field_conflicts"), _read("field_reviewed_links"))
    (ROOT / "docs" / "class-report.md").write_text(report + "\n", encoding="utf-8")
    cov = render_coverage(load_coverage(), pd.read_csv(KNOWLEDGE_MANIFEST))
    (ROOT / "docs" / "document-coverage.md").write_text(cov + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
