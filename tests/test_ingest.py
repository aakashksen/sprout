import csv
import json
from pathlib import Path

import pytest

from sprout.rag.ingest import build_index, load_manifest, missing_from_manifest

HEADER = ["file", "crop", "diseases", "title", "publisher", "source_url", "date"]
TEXT = "Early blight shows dark spots with rings on lower leaves of the plant. " * 8


def make_pdf(path: Path, pages: list[str]) -> None:
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        stream = f"BT /F1 10 Tf 20 700 Td ({text}) Tj ET" if text else ""
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})


def read_chunks(index: Path) -> list[dict]:
    lines = (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_crop_comes_from_manifest_not_filename(tmp_path, raw):
    make_pdf(raw / "grapediseaseidguide.pdf", [TEXT])
    m = tmp_path / "manifest.csv"
    write_manifest(m, [{"file": "grapediseaseidguide.pdf", "crop": "grape", "title": "Grape guide"}])
    build_index(load_manifest(m), raw, tmp_path / "index")
    chunks = read_chunks(tmp_path / "index")
    assert chunks and {c["crop"] for c in chunks} == {"grape"}


def test_markdown_is_ingested(tmp_path, raw):
    (raw / "tomatodoc1.md").write_text("# Tomato\n\n" + TEXT, encoding="utf-8")
    m = tmp_path / "manifest.csv"
    write_manifest(m, [{"file": "tomatodoc1.md", "crop": "tomato", "title": "Tomato notes"}])
    report = build_index(load_manifest(m), raw, tmp_path / "index")
    chunks = read_chunks(tmp_path / "index")
    assert chunks and chunks[0]["crop"] == "tomato"
    assert "blight" in chunks[0]["text"]
    assert report.failures == {}


def test_failed_extraction_creates_no_chunk(tmp_path, raw):
    make_pdf(raw / "scan.pdf", ["", "", ""])
    make_pdf(raw / "good.pdf", [TEXT])
    (raw / "broken.pdf").write_bytes(b"not a pdf")
    m = tmp_path / "manifest.csv"
    write_manifest(
        m,
        [
            {"file": "scan.pdf", "crop": "corn", "title": "Scan"},
            {"file": "good.pdf", "crop": "potato", "title": "Good"},
            {"file": "broken.pdf", "crop": "corn", "title": "Broken"},
        ],
    )
    report = build_index(load_manifest(m), raw, tmp_path / "index")
    chunks = read_chunks(tmp_path / "index")
    assert {c["source"] for c in chunks} == {"good.pdf"}
    assert set(report.failures) == {"scan.pdf", "broken.pdf"}


def test_chunks_keep_source_and_page(tmp_path, raw):
    make_pdf(raw / "a.pdf", ["First page " + TEXT, "Second page " + TEXT])
    m = tmp_path / "manifest.csv"
    write_manifest(m, [{"file": "a.pdf", "crop": "potato", "diseases": "early blight;late blight", "title": "A"}])
    build_index(load_manifest(m), raw, tmp_path / "index")
    chunks = read_chunks(tmp_path / "index")
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(c["source"] == "a.pdf" and c["title"] == "A" for c in chunks)
    assert chunks[0]["diseases"] == ["early blight", "late blight"]
    assert "Second page" in next(c for c in chunks if c["page"] == 2)["text"]


def test_failed_build_keeps_the_old_index(tmp_path, raw):
    make_pdf(raw / "good.pdf", [TEXT])
    m = tmp_path / "manifest.csv"
    write_manifest(m, [{"file": "good.pdf", "crop": "potato", "title": "Good"}])
    index = tmp_path / "index"
    build_index(load_manifest(m), raw, index)
    before = (index / "chunks.jsonl").read_text(encoding="utf-8")

    (raw / "good.pdf").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError):
        build_index(load_manifest(m), raw, index)
    assert (index / "chunks.jsonl").read_text(encoding="utf-8") == before
    assert not (tmp_path / "index.new").exists()


def test_every_file_is_in_the_manifest(tmp_path, raw):
    (raw / "a.md").write_text(TEXT, encoding="utf-8")
    (raw / "b.md").write_text(TEXT, encoding="utf-8")
    m = tmp_path / "manifest.csv"
    write_manifest(m, [{"file": "a.md", "crop": "corn", "title": "A"}, {"file": "gone.md", "crop": "corn", "title": "G"}])
    missing, extra = missing_from_manifest(load_manifest(m), raw)
    assert missing == ["b.md"]
    assert extra == ["gone.md"]


def test_shipped_manifest_is_complete():
    entries = load_manifest(Path(__file__).parents[1] / "knowledge" / "manifest.csv")
    files = [e.file for e in entries]
    assert len(files) == len(set(files)) >= 28
    assert all(e.crop and e.title for e in entries)


ROOT = Path(__file__).parents[1]
CHUNKS = ROOT / "knowledge" / "index" / "chunks.jsonl"
needs_index = pytest.mark.skipif(not CHUNKS.exists(), reason="knowledge/index is git-ignored and absent here")

# Diseases the document discusses only under another name, found by reading its text.
DISEASE_ALIASES = {
    ("Corn1.pdf", "northern corn leaf blight"): "northern leaf blight",
    ("grapediseaseidguide.pdf", "esca"): "black measles",
}


@needs_index
def test_manifest_diseases_appear_in_their_document():
    entries = load_manifest(ROOT / "knowledge" / "manifest.csv")
    text: dict[str, str] = {}
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        text.setdefault(c["source"], "")
        text[c["source"]] += " " + c["text"].lower()
    absent = []
    for e in entries:
        body = " ".join(text.get(e.file, "").split())
        for d in e.diseases:
            term = DISEASE_ALIASES.get((e.file, d), d)
            if term not in body:
                absent.append((e.file, d))
    assert absent == []
