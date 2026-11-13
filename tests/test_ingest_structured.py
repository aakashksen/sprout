"""Structure-aware chunking (whole tables, paragraphs kept together) and the context header."""
import json

from sprout.rag.ingest import ManifestEntry, build_index, split_structured

TABLE = "\n".join(["| Product | Rate | Interval |", "| --- | --- | --- |"] +
                  [f"| Product {i} | {i} oz | {i} days |" for i in range(30)])
DOC = f"# Early blight\n\nFirst paragraph about blight lesions.\n\nSecond paragraph about spores.\n\n# Sprays\n\n{TABLE}\n\nAfter the table."


def test_table_stays_in_one_chunk_even_when_longer_than_a_chunk():
    pieces = split_structured(DOC, max_chars=200)
    tables = [t for t, _ in pieces if "Product 0 " in t]
    assert len(tables) == 1 and "Product 29 " in tables[0] and len(tables[0]) > 200


def test_paragraphs_are_not_cut_in_the_middle():
    text = "# A\n\n" + "\n\n".join(f"Paragraph number {i} says something complete." for i in range(20))
    pieces = split_structured(text, max_chars=120)
    assert len(pieces) > 1
    for chunk, _ in pieces:
        assert all(p.endswith("complete.") for p in chunk.split("\n\n") if p.startswith("Paragraph"))


def test_each_piece_knows_its_section_heading():
    pieces = dict((t.split("\n")[-1][:20], s) for t, s in split_structured(DOC, max_chars=200))
    assert pieces["After the table."] == "Sprays"
    assert any(s == "Early blight" for _, s in split_structured(DOC, max_chars=200))


def test_oversized_table_is_split_by_rows_with_its_header_row():
    pieces = split_structured(DOC, max_chars=200, table_max=300)
    tables = [t for t, _ in pieces if "| Product |" in t or "Product 2" in t]
    assert len(tables) > 1
    assert all(t.startswith("| Product | Rate | Interval |") for t in tables)


def test_build_index_structured_with_header(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text(DOC, encoding="utf-8")
    entry = ManifestEntry("doc.md", "tomato", ["early blight"], "Blight guide", "Univ", "https://x", "2020")
    build_index([entry], raw, tmp_path / "idx", chunker="structured", header=True)
    rows = [json.loads(x) for x in (tmp_path / "idx" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("Product 29 " in r["text"] for r in rows)
    sprays = next(r for r in rows if "Product 0 " in r["text"])
    assert "Blight guide" in sprays["header"] and "tomato" in sprays["header"] and "Sprays" in sprays["header"]
    assert "Blight guide" not in sprays["text"]


def test_plain_build_has_no_header_key_and_is_unchanged(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text(DOC, encoding="utf-8")
    entry = ManifestEntry("doc.md", "tomato", [], "Blight guide", "Univ", "https://x", "2020")
    build_index([entry], raw, tmp_path / "idx")
    rows = [json.loads(x) for x in (tmp_path / "idx" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(set(r) == {"text", "source", "page", "crop", "diseases", "title"} for r in rows)


def test_html_headings_and_tables_become_markdown_like_lines(tmp_path):
    from sprout.rag.ingest import _extract

    page = tmp_path / "p.html"
    page.write_text(
        "<html><body><main><h2>Sprays</h2><p>Use these products on tomatoes in season.</p>"
        "<table><tr><th>Product</th><th>Rate</th></tr><tr><td>Copper</td><td>2 oz</td></tr></table>"
        "</main></body></html>", encoding="utf-8")
    text = _extract(page, structured=True)[0][1]
    assert "# Sprays" in text and "| Product | Rate |" in text and "| Copper | 2 oz |" in text


def test_pdf_columns_become_table_rows():
    from sprout.rag.ingest import _layout_blocks

    text = "Intro line one\nintro line two\n\nName      Rate      Days\nCopper    2 oz      7\nSulfur    3 oz      10\n\nAfter."
    out = _layout_blocks(text)
    assert "| Name | Rate | Days |" in out and "| Sulfur | 3 oz | 10 |" in out
    assert "Intro line one intro line two" in out
