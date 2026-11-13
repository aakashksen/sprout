"""Ingest knowledge documents into a chunk index, using the manifest for all metadata."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

MIN_PAGE_CHARS = 40
MIN_GOOD_PAGE_FRACTION = 0.5
CHUNK_CHARS = 1000
OVERLAP_CHARS = 150
TABLE_MAX_CHARS = 4000  # a table up to this size is never split
CHUNKERS = ("plain", "structured")


@dataclass
class ManifestEntry:
    file: str
    crop: str
    diseases: list[str]
    title: str
    publisher: str
    source_url: str
    date: str


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    crop: str
    diseases: list[str]
    title: str
    header: str = ""  # context header: searched with the text, never shown


@dataclass
class IngestReport:
    chunks: int = 0
    failures: dict[str, str] = field(default_factory=dict)


def load_manifest(path: Path) -> list[ManifestEntry]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [
            ManifestEntry(
                file=r["file"],
                crop=r["crop"].strip().lower(),
                diseases=[d.strip() for d in r["diseases"].split(";") if d.strip()],
                title=r["title"],
                publisher=r["publisher"],
                source_url=r["source_url"],
                date=r["date"],
            )
            for r in csv.DictReader(f)
        ]


def missing_from_manifest(entries: list[ManifestEntry], raw_dir: Path) -> tuple[list[str], list[str]]:
    """Return (files in raw_dir with no manifest row, manifest rows with no file)."""
    listed = {e.file for e in entries}
    present = {p.name for p in Path(raw_dir).iterdir() if p.is_file()}
    return sorted(present - listed), sorted(listed - present)


def _layout_blocks(text: str) -> str:
    """Page text from pypdf's layout mode: runs of columnar lines become table rows, other lines paragraphs."""
    out: list[str] = []
    para: list[str] = []
    rows: list[str] = []

    def end_para() -> None:
        if para:
            out.append(" ".join(para))
            para.clear()

    def end_rows() -> None:
        if len(rows) >= 3:
            out.append("\n".join(rows))
        else:
            out.extend(" ".join(c.strip() for c in r.strip("|").split("|")) for r in rows)
        rows.clear()

    for line in text.splitlines():
        cells = re.split(r"\s{3,}", line.strip())
        if len(cells) >= 3:
            end_para()
            rows.append("| " + " | ".join(cells) + " |")
            continue
        end_rows()
        if line.strip():
            para.append(line.strip())
        else:
            end_para()
    end_rows()
    end_para()
    return "\n\n".join(out)


def _extract_pdf(path: Path, structured: bool = False) -> list[tuple[int, str]]:
    mode = {"extraction_mode": "layout"} if structured else {}
    pages = [(i, (p.extract_text(**mode) or "").strip()) for i, p in enumerate(PdfReader(path).pages, 1)]
    if structured:
        pages = [(i, _layout_blocks(t)) for i, t in pages]
    good = [(i, t) for i, t in pages if len(t) >= MIN_PAGE_CHARS]
    if not pages or len(good) / len(pages) < MIN_GOOD_PAGE_FRACTION:
        raise ValueError(f"only {len(good)} of {len(pages)} pages have extractable text (OCR not available)")
    return good


class _PageText(HTMLParser):
    """Visible text of an HTML page; the <main> or <article> part when there is one."""

    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "head"}
    MAIN = {"main", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.main = 0
        self.all_text: list[str] = []
        self.main_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.MAIN:
            self.main += 1
        self._add("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.MAIN:
            self.main = max(0, self.main - 1)
        self._add("\n")

    def handle_data(self, data):
        self._add(data)

    def _add(self, text: str) -> None:
        if self.skip:
            return
        self.all_text.append(text)
        if self.main:
            self.main_text.append(text)


class _BlockText(_PageText):
    """Like _PageText, but headings become `# ` lines and table rows become `| a | b |` lines."""

    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    BLOCKS = {"p", "div", "section", "ul", "ol", "blockquote", "pre", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.MAIN:
            self.main += 1
        if tag in self.HEADINGS:
            self._add("\n\n# ")
        elif tag in self.BLOCKS or tag == "br":
            self._add("\n\n")
        elif tag == "li":
            self._add("\n- ")
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.MAIN:
            self.main = max(0, self.main - 1)
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self._add("\n| " + " | ".join(self.row) + " |")
            self.row = None
        elif tag in self.HEADINGS or tag in self.BLOCKS:
            self._add("\n\n")

    def handle_data(self, data):
        if self.cell is not None:
            if not self.skip:
                self.cell.append(data)
        else:
            self._add(data)


def _tidy(text: str) -> str:
    lines = [line.strip() if line.lstrip().startswith("|") else " ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _extract_html(path: Path, structured: bool = False) -> list[tuple[int, str]]:
    parser = _BlockText() if structured else _PageText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = "".join(parser.main_text or parser.all_text).strip()
    text = _tidy(text) if structured else text
    if len(text) < MIN_PAGE_CHARS:
        raise ValueError("html page has no usable text")
    return [(1, text)]


def _extract(path: Path, structured: bool = False) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, structured)
    if suffix in (".html", ".htm"):
        return _extract_html(path, structured)
    if suffix == ".md":
        text = path.read_text(encoding="utf-8").strip()
        if len(text) < MIN_PAGE_CHARS:
            raise ValueError("markdown file has no usable text")
        return [(1, text)]
    raise ValueError(f"unsupported file type {suffix}")


def _split(text: str) -> list[str]:
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            space = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            end = space if space > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return out


def _blocks(text: str) -> list[tuple[str, str]]:
    """(kind, text) blocks in order; kind is heading, table or paragraph."""
    out: list[tuple[str, str]] = []
    lines: list[str] = []
    kind = ""

    def end() -> None:
        nonlocal lines, kind
        if lines:
            out.append((kind, "\n".join(lines)))
        lines, kind = [], ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            end()
            continue
        this = "heading" if line.startswith("#") else "table" if line.startswith("|") else "paragraph"
        if this != kind or this == "heading":
            end()
        kind = this
        lines.append(line)
    end()
    return out


def _split_table(table: str, table_max: int) -> list[str]:
    if len(table) <= table_max:
        return [table]
    rows = table.splitlines()
    head = rows[:2] if len(rows) > 1 and re.fullmatch(r"[|\s:-]+", rows[1]) else rows[:1]
    parts: list[list[str]] = [[]]
    for row in rows[len(head):]:
        if parts[-1] and len("\n".join(head + parts[-1] + [row])) > table_max:
            parts.append([])
        parts[-1].append(row)
    return ["\n".join(head + p) for p in parts]


def split_structured(
    text: str, *, max_chars: int = CHUNK_CHARS, table_max: int = TABLE_MAX_CHARS
) -> list[tuple[str, str]]:
    """(chunk text, section heading) pieces. Paragraphs are packed up to max_chars and cut only when one
    alone is longer; a table is its own chunk, split by rows (repeating its header row) only past table_max."""
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    section = cur_section = pending = ""

    def flush() -> None:
        nonlocal cur
        if cur:
            out.append(("\n\n".join(cur), cur_section))
        cur = []

    for kind, block in _blocks(text):
        if kind == "heading":
            pending = block
            section = block.lstrip("# ").strip()
            continue
        if kind == "table":
            flush()
            pending = ""
            out.extend((part, section) for part in _split_table(block, table_max))
            continue
        for piece in [block] if len(block) <= max_chars else _split(block):
            body = f"{pending}\n\n{piece}" if pending else piece
            if cur and sum(map(len, cur)) + len(body) > max_chars:
                flush()
            if not cur:
                cur_section = section
            cur.append(body)
            pending = ""
    flush()
    return out


def context_header(e: ManifestEntry, section: str) -> str:
    parts = [f"Document: {e.title}" + (f" ({e.publisher})" if e.publisher else ""), f"Crop: {e.crop}"]
    if e.diseases:
        parts.append("Diseases: " + ", ".join(e.diseases))
    if section:
        parts.append(f"Section: {section}")
    return ". ".join(parts)


def build_index(
    entries: list[ManifestEntry], raw_dir: Path, index_dir: Path, *, chunker: str = "plain", header: bool = False
) -> IngestReport:
    """Build a new index beside index_dir, validate it, then swap it in.

    Raises RuntimeError and leaves any existing index untouched if the build is invalid.
    `chunker` is "plain" (fixed windows) or "structured" (paragraphs and whole tables); `header` adds each
    chunk's context header (with the section heading when the chunker is structured).
    """
    if chunker not in CHUNKERS:
        raise ValueError(f"unknown chunker {chunker!r}")
    index_dir = Path(index_dir)
    new_dir = index_dir.with_name(index_dir.name + ".new")
    old_dir = index_dir.with_name(index_dir.name + ".old")
    shutil.rmtree(new_dir, ignore_errors=True)
    report = IngestReport()
    chunks: list[Chunk] = []
    for e in entries:
        try:
            pages = _extract(Path(raw_dir) / e.file, chunker == "structured")
        except Exception as exc:
            report.failures[e.file] = str(exc)
            continue
        for page, text in pages:
            pieces = split_structured(text) if chunker == "structured" else [(p, "") for p in _split(text)]
            chunks.extend(
                Chunk(piece, e.file, page, e.crop, e.diseases, e.title, context_header(e, sec) if header else "")
                for piece, sec in pieces
            )
    report.chunks = len(chunks)

    try:
        if not chunks:
            raise RuntimeError("new index has no chunks")
        if report.failures and len(report.failures) == len(entries):
            raise RuntimeError("every document failed extraction")
        new_dir.mkdir(parents=True)
        with (new_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                row = asdict(c)
                if not row["header"]:
                    del row["header"]
                f.write(json.dumps(row) + "\n")
        (new_dir / "failures.json").write_text(json.dumps(report.failures, indent=2), encoding="utf-8")
        lines = (new_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if len(lines) != len(chunks) or any(not json.loads(x)["text"] for x in lines):
            raise RuntimeError("new index failed validation")
    except Exception as exc:
        shutil.rmtree(new_dir, ignore_errors=True)
        raise exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc)) from exc

    shutil.rmtree(old_dir, ignore_errors=True)
    if index_dir.exists():
        index_dir.rename(old_dir)
    new_dir.rename(index_dir)
    shutil.rmtree(old_dir, ignore_errors=True)
    return report
