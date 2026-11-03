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
