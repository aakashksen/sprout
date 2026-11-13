import builtins
from pathlib import Path

import pandas as pd

from sprout.data import class_report as cr
from sprout.data.splits import load_manifest


def _manifests():
    field = pd.read_csv(cr.MANIFESTS / "field.csv")
    return load_manifest("lab"), field


def test_counts_match_the_manifests():
    lab, field = _manifests()
    table = cr.class_table(lab, field)
    assert len(table) == lab["label"].nunique() == 38
    assert table["lab_train"].sum() == (lab["split"] == "lab_train").sum()
    assert table["lab_dev"].sum() == (lab["split"] == "lab_dev").sum()
    assert table["lab_test"].sum() == (lab["split"] == "lab_test").sum()
    for source in cr.SOURCES:
        for part in cr.SPLIT_NAMES:
            rows = (field["source"] == source) & (field["split"] == f"field_{part}")
            assert table[f"{source}_{part}"].sum() == rows.sum()
    label = "Tomato___Early_blight"
    expected = ((field["label"] == label) & (field["split"] == "field_dev")).sum()
    assert table.loc[label, "real_dev"] == expected
    class_map = pd.read_csv(cr.MANIFESTS.parent / "class_map.csv")
    shared = class_map[class_map["status"] == "shared"]
    by_class = shared.groupby("lab_class")["n_field_test"].sum()
    assert (table.loc[by_class.index, "real_test"] == by_class).all()


def test_thin_classes_are_reported_not_dropped():
    lab, field = _manifests()
    table = cr.class_table(lab, field)
    assert len(table) == 38
    thin = table[table["thin"]]
    assert "Strawberry___Leaf_scorch" in thin.index
    assert "Apple___Black_rot" in thin.index
    assert table.loc["Tomato___Tomato_Yellow_Leaf_Curl_Virus", "real_train"] < cr.MIN_TRAIN
    assert table.loc["Strawberry___Leaf_scorch", "real_test"] == 5
    text = cr.render_report(lab, field, pd.read_csv(cr.MANIFESTS / "field_conflicts.csv"),
                            pd.read_csv(cr.MANIFESTS / "field_reviewed_links.csv"))
    section = text.split("## Thin classes")[1].split("## Crops")[0]
    for label in thin.index:
        assert label in section
    assert (table["real_test"].sum()) == (field["split"] == "field_test").sum()


def test_report_reads_no_field_test_image(monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    lab, field = _manifests()
    cr.render_report(lab, field, pd.read_csv(cr.MANIFESTS / "field_conflicts.csv"),
                     pd.read_csv(cr.MANIFESTS / "field_reviewed_links.csv"))
    image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    assert not [p for p in opened if Path(p).suffix.lower() in image_suffixes or "raw" in Path(p).parts]
    source = Path(cr.__file__).read_text(encoding="utf-8")
    for banned in ("PIL", "cv2", "torch", "imread"):
        assert banned not in source


def test_every_kept_class_has_a_coverage_row():
    lab, _ = _manifests()
    rows = cr.load_coverage()
    assert sorted(rows["label"]) == sorted(lab["label"].unique())
    assert set(rows["symptoms_described"]) <= {"yes", "no", "not applicable"}
    for _, r in rows.iterrows():
        assert r["evidence"].strip()
        if r["symptoms_described"] == "yes":
            assert r["document"] and r["page"]
        if r["label"].endswith("___healthy"):
            assert r["symptoms_described"] == "not applicable"
    known = set(pd.read_csv(cr.KNOWLEDGE_MANIFEST)["file"])
    for doc in "; ".join(rows["document"]).split("; "):
        assert doc == "" or doc in known
    for label in ("Squash___Powdery_mildew", "Raspberry___healthy", "Soybean___healthy"):
        assert rows.set_index("label").loc[label, "crop"] == label.split("___")[0].lower()
    for _, r in rows.iterrows():
        assert r["crop"] == (cr.class_crop(r["label"]) or "none")


def test_yes_rows_cite_document_and_page_and_no_rows_record_queries():
    rows = cr.load_coverage()
    yes = rows[rows["symptoms_described"] == "yes"]
    no = rows[rows["symptoms_described"] == "no"]
    assert len(yes)  # no class is uncovered after T6.5, so `no` may be empty
    for _, r in yes.iterrows():
        assert r["document"].strip() and r["page"].strip(), r["label"]
    for _, r in no.iterrows():
        assert "Queries" in r["evidence"], r["label"]
    by_label = rows.set_index("label")["symptoms_described"]
    assert by_label["Tomato___Bacterial_spot"] == "yes"
    assert by_label["Tomato___Spider_mites Two-spotted_spider_mite"] == "yes"
    assert by_label["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"] == "yes"


def test_label_conflicts_are_not_reported_as_cross_source_duplicates():
    conflicts = pd.read_csv(cr.MANIFESTS / "field_conflicts.csv")
    conf = cr.label_conflicts(conflicts)
    assert conf["images"] == len(conflicts)
    assert conf["single_source_groups"] + conf["two_source_groups"] == conf["groups"]
    assert conf["single_source_groups"] > 0
    lab, field = _manifests()
    links = pd.read_csv(cr.MANIFESTS / "field_reviewed_links.csv")
    text = cr.render_report(lab, field, conflicts, links)
    dup_section = text.split("## Cross-source duplicates")[1].split("## Label-conflict images")[0]
    assert "field_conflicts" not in dup_section
    assert "## Label-conflict images" in text


def test_crop_table_separates_healthy_photos_from_all_real_photos():
    lab, field = _manifests()
    crops = cr.healthy_by_crop(cr.class_table(lab, field)).set_index("crop")
    assert "real" not in crops.columns
    assert crops.loc["Squash", "healthy_real_photos"] == 0
    squash = field[field["label"].str.startswith("Squash___")]
    assert crops.loc["Squash", "all_classes_real_photos"] == len(squash) > 0
    assert crops.loc["Orange", "all_classes_real_photos"] > 0
    assert crops["all_classes_real_photos"].sum() == len(field)
