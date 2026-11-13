import re
from pathlib import Path

import pandas as pd
from PIL import Image

import sprout.data.field as field_mod
from sprout.data.field import write_review


def _field(tmp_path: Path, n: int, order: list[int] | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n):
        p = tmp_path / "img" / f"photo{i}.png"
        p.parent.mkdir(exist_ok=True, parents=True)
        Image.new("RGB", (300, 200), (i * 20 % 256, 255 - i * 20 % 256, 40)).save(p)
        rows.append((p.as_posix(), f"C{i % 3}", f"g{i}", "field_dev", f"{i:02x}" + "a" * 38, "plantdoc"))
    df = pd.DataFrame(rows, columns=["path", "label", "group", "split", "sha1", "source"])
    return df.iloc[order].reset_index(drop=True) if order else df


def _table(md: str) -> list[dict[str, str]]:
    lines = [ln for ln in md.splitlines() if ln.startswith("| ") and not ln.startswith("| #")]
    out = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells[0].isdigit():
            out.append({"link": re.search(r"\((.+)\)", cells[1]).group(1), "label": cells[2],
                        "file": cells[5].strip("`"), "id": cells[6].strip("`")})
    return out


def test_thumbs_and_table_come_from_the_same_rows(tmp_path, monkeypatch):
    review = tmp_path / "review"
    monkeypatch.setattr(field_mod, "REVIEW", review)
    first = _field(tmp_path, 12)
    write_review(first, n=6)
    # a second call on different rows, as after a rebuild: the old files must not mislead
    second = _field(tmp_path, 12, order=[3, 1, 7, 0, 11, 5, 9, 2, 10, 4, 8])
    write_review(second, n=6)
    rows = _table((review / "field-sample.md").read_text(encoding="utf-8"))
    assert len(rows) == 6
    by_id = dict(zip(second["sha1"], second["path"]))
    for r in rows:
        assert r["id"] in by_id and by_id[r["id"]].endswith(r["file"])
        thumb = review / r["link"]
        assert r["id"][:10] in thumb.name and thumb.is_file()
        with Image.open(thumb) as t, Image.open(by_id[r["id"]]) as src:
            got, want = t.convert("RGB").getpixel((5, 5)), src.convert("RGB").getpixel((5, 5))
            assert max(abs(a - b) for a, b in zip(got, want)) <= 4  # JPEG rounding
    shown = {review / r["link"] for r in rows}
    assert set((review / "thumbs").iterdir()) == shown  # no leftover thumbnail from another run


# ---------------------------------------------------------------------------------------------------------------
# label review tools

import hashlib  # noqa: E402

import pytest  # noqa: E402
from PIL import ImageDraw  # noqa: E402

import sprout.data.label_review as lr  # noqa: E402

FIELD_CSV = Path(__file__).resolve().parents[1] / "datasets" / "manifests" / "field.csv"
FIELD_SHA256 = "edc078417b222f9195b8ea128e2049c851b8e7bf8e44774a0edabd2773cffffd"
SPLIT_WORDS = ("field_", "dev", "test", "train")


def _mixed(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(40):
        p = tmp_path / "raw" / "plantdoc" / f"leaf%20{i}.png"
        p.parent.mkdir(exist_ok=True, parents=True)
        Image.new("RGB", (120, 90), (i * 6 % 256, 120, 60)).save(p)
        split = ("field_train", "field_dev", "field_test", "field_dev")[i % 4]
        rows.append((p.as_posix(), f"Plant{i % 3}___healthy", f"g{i}", split, f"{i:02x}" + "b" * 38,
                     "plantdoc" if i % 2 else "plantseg"))
    return pd.DataFrame(rows, columns=["path", "label", "group", "split", "sha1", "source"])


def test_sheets_do_not_name_the_split(tmp_path, monkeypatch):
    drawn = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        drawn.append(str(text))
        return real(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    field = _mixed(tmp_path)
    rows = lr.shuffled_rows(field)
    side = lr.make_sheets(rows, tmp_path / "sheets")
    assert len(rows) == 30 and set(rows["split"]) == {"field_dev", "field_test"}
    assert rows["split"].iloc[:10].nunique() == 2  # shuffled together
    assert drawn and not any(w in t for t in drawn for w in SPLIT_WORDS)
    assert list(side.columns) == lr.SIDECAR_COLUMNS
    for f in (tmp_path / "sheets").iterdir():
        if f.suffix == ".csv":
            assert not any(w in f.read_text() for w in SPLIT_WORDS)
    assert list(lr.shuffled_rows(field)["image_id"]) == list(rows["image_id"])  # fixed seed


def test_review_inputs_contain_no_scores_or_predictions():
    src = Path(lr.__file__).read_text(encoding="utf-8")
    assert "sprout.eval" not in src and "results/" not in src and "probs" not in src
    assert set(lr.SIDECAR_COLUMNS) == {"sheet", "cell", "image_id", "label", "filename"}


def test_image_id_is_sha1_when_unique_else_path(tmp_path):
    f = _mixed(tmp_path)
    assert lr.id_column(f) == "sha1"
    f.loc[1, "sha1"] = f.loc[0, "sha1"]
    assert lr.id_column(f) == "path"


def test_second_look_sheets_are_larger(tmp_path):
    f = lr.with_ids(_mixed(tmp_path)).head(7)
    side = lr.make_second_look(f, tmp_path / "s")
    assert side["sheet"].nunique() == 2
    with Image.open(tmp_path / "s" / "c0001.jpg") as im:
        assert im.width == 3 * lr.SECOND_CELL and im.height == lr.HEADER + 2 * lr.SECOND_CELL


def _verdicts(tmp_path, side, over=None):
    rows = [{"sheet": s, "cell": c, "verdict": "ok", "reason_code": "", "note": ""}
            for s, c in zip(side["sheet"], side["cell"])]
    for i, kw in (over or {}).items():
        rows[i].update(kw)
    p = tmp_path / "v.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_collector_joins_and_refuses_bad_input(tmp_path):
    field = _mixed(tmp_path)
    side = lr.make_sheets(lr.shuffled_rows(field), tmp_path / "s")
    sidecars = sorted((tmp_path / "s").glob("r*.csv"))
    good = _verdicts(tmp_path, side, {0: {"verdict": "remove", "reason_code": "wrong_plant", "note": "x"}})
    joined = lr.collect([good], sidecars, field)
    assert len(joined) == 30 and joined.iloc[0]["split"] in ("field_dev", "field_test")
    assert joined.iloc[0]["image_id"] == side.iloc[0]["image_id"]
    df = pd.read_csv(good, dtype=str, keep_default_na=False)
    for bad, pat in ((df.iloc[1:], "no verdict"),
                     (pd.concat([df, df.iloc[:1]]), "repeat"),
                     (pd.concat([df, df.iloc[:1].assign(cell="99")]), "not on the sheets"),
                     (df.assign(verdict="remove", reason_code="because"), "reason_code")):
        p = tmp_path / "bad.csv"
        bad.to_csv(p, index=False)
        with pytest.raises(ValueError, match=pat):
            lr.collect([p], sidecars, field)


def _joined(field, ids, verdict, code="wrong_plant"):
    f = lr.with_ids(field).set_index("image_id").loc[ids].reset_index()
    return f.assign(verdict=verdict, reason_code=code, note="n", sheet="c0001", cell=1, began_as=verdict)


def test_every_exclusion_has_id_source_split_reason(tmp_path):
    field = _mixed(tmp_path)
    ids = list(lr.with_ids(field)["image_id"][:3])
    log = lr.log_rows(_joined(field, ids, "remove"), 1)
    lr.check_log(log, field)
    assert set(log.columns) == set(lr.LOG_COLUMNS)
    for col, val in (("source", ""), ("split", ""), ("reason_code", "")):
        with pytest.raises(ValueError):
            lr.check_log(log.assign(**{col: val}), field)
    real = lr.read_log(lr.EXCLUSIONS)
    if len(real):
        lr.check_log(real, pd.read_csv(FIELD_CSV))


def test_every_exclusion_exists_in_field_csv(tmp_path):
    field = _mixed(tmp_path)
    log = lr.log_rows(_joined(field, list(lr.with_ids(field)["image_id"][:2]), "remove"), 1)
    with pytest.raises(ValueError, match="not in field.csv"):
        lr.check_log(log.assign(image_id=["nope1", "nope2"]), field)
    real = lr.read_log(lr.EXCLUSIONS)
    assert set(real["image_id"]) <= set(pd.read_csv(FIELD_CSV)["sha1"])


def test_field_csv_is_unchanged():
    assert hashlib.sha256(FIELD_CSV.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest() == FIELD_SHA256  # the hash is of the CRLF file; git may check it out with LF


def test_exclusions_are_applied_when_the_manifest_is_built(tmp_path):
    field = _mixed(tmp_path)
    fpath, lpath, cpath = tmp_path / "field.csv", tmp_path / "ex.csv", tmp_path / "clean.csv"
    field.to_csv(fpath, index=False)
    before = fpath.read_bytes()
    ids = list(lr.with_ids(field)["image_id"][[1, 2]])
    lr.append_log(lpath, lr.log_rows(_joined(field, ids, "remove"), 1), field)
    h = lr.write_clean(fpath, lpath, cpath)
    clean = pd.read_csv(cpath)
    assert len(clean) == 38 and not set(field["path"][[1, 2]]) & set(clean["path"])
    assert set(clean["clean_version"]) == {1} and h != lr.dataset_hash(field[lr.COLUMNS])
    assert fpath.read_bytes() == before
    with pytest.raises(ValueError, match="version"):  # a round needs a higher version
        lr.append_log(lpath, lr.log_rows(_joined(field, [lr.with_ids(field)["image_id"][5]], "remove"), 1), field)


def test_decide_needs_both_looks_to_remove(tmp_path):
    field = _mixed(tmp_path)
    ids = list(lr.with_ids(field)["image_id"][:4])
    first = pd.concat([_joined(field, ids[:3], "remove"), _joined(field, ids[3:], "uncertain", "")])
    second = pd.concat([_joined(field, ids[:1], "remove"), _joined(field, ids[1:2], "ok", ""),
                        _joined(field, ids[2:3], "uncertain", "")])
    rem, unsure, over = lr.decide(first, second)
    assert list(rem["image_id"]) == ids[:1]
    assert set(unsure["image_id"]) == {ids[2], ids[3]} and list(over["image_id"]) == ids[1:2]


def test_wilson_interval_and_report(tmp_path):
    lo, hi = lr.wilson(20, 400)
    assert lo == pytest.approx(0.0328, abs=1e-3) and hi == pytest.approx(0.0757, abs=1e-3)
    field = _mixed(tmp_path)
    ids = lr.with_ids(field)
    first = _joined(field, list(ids["image_id"]), "ok", "")
    log = lr.log_rows(_joined(field, [ids["image_id"][0], ids["image_id"][1]], "remove"), 1)
    text = lr.write_report(first, log, lr.read_log(tmp_path / "none.csv"), {"remove": 1, "ok": 2, "uncertain": 3}, tmp_path / "r.md")
    assert "## By class" in text and "Wilson 95%" in text


def test_second_look_also_takes_uncertain_rows_whose_note_matches(tmp_path):
    field = _mixed(tmp_path)
    ids = list(lr.with_ids(field)["image_id"][:6])
    first = pd.concat([
        _joined(field, ids[:1], "remove"),
        _joined(field, ids[1:3], "uncertain", "").assign(note=["Framed slide-like image", "leaf looks odd"]),
        _joined(field, ids[3:], "ok", ""),
    ])
    pattern = "slide|poster"
    assert set(lr.proposed(first, None)["image_id"]) == {ids[0]}
    assert set(lr.proposed(first, pattern)["image_id"]) == {ids[0], ids[1]}
    second = pd.concat([_joined(field, ids[:1], "ok", ""), _joined(field, ids[1:2], "remove")])
    removals, unsure, overturned = lr.decide(first, second, pattern)
    assert list(removals["image_id"]) == [ids[1]] and list(removals["began_as"]) == ["uncertain"]
    assert set(unsure["image_id"]) == {ids[2]}  # the confirmed one is no longer uncertain
    assert list(overturned["image_id"]) == [ids[0]]
    # not confirmed: the matched uncertain row stays in the uncertain list
    second = pd.concat([_joined(field, ids[:1], "remove"), _joined(field, ids[1:2], "ok", "")])
    removals, unsure, _ = lr.decide(first, second, pattern)
    assert list(removals["began_as"]) == ["remove"] and set(unsure["image_id"]) == {ids[1], ids[2]}
    log = lr.log_rows(removals, 1)
    assert "began_as" in log.columns
    with pytest.raises(ValueError, match="exactly"):
        lr.decide(first, second.iloc[:1], pattern)


def test_report_states_calibration_and_how_removals_began(tmp_path):
    field = _mixed(tmp_path)
    ids = lr.with_ids(field)
    first = _joined(field, list(ids["image_id"]), "ok", "")
    log = pd.concat([
        lr.log_rows(_joined(field, [ids["image_id"][1]], "remove"), 1),
        lr.log_rows(_joined(field, [ids["image_id"][2]], "remove").assign(began_as="uncertain"), 1),
    ])
    text = lr.write_report(first, log, lr.read_log(tmp_path / "none.csv"), {"remove": 1, "ok": 2, "uncertain": 3}, tmp_path / "r.md")
    assert "of 4 known-bad photos" in text and "2 were `remove`" in text and "1 was `uncertain`" in text
    assert "1 was `ok`" in text and "lower bound" in text
    assert "1 confirmed, 2 called ok" in text and "3 called uncertain" in text
    assert "began as `remove`: 1" in text and "began as `uncertain`: 1" in text


def test_default_uncertain_pattern_lives_in_the_config():
    import json
    import re

    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "e13.json").read_text())
    rx = re.compile(cfg["second_look_uncertain_regex"], re.I)
    assert rx.search("Framed slide-like image") and rx.search("Not tomato") and not rx.search("blurry leaf")
