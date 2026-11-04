"""Ingest knowledge documents into a chunk index, using the manifest for all metadata."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

MIN_PAGE_CHARS = 40
MIN_GOOD_PAGE_FRACTION = 0.5
CHUNK_CHARS = 1000
OVERLAP_CHARS = 150


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


def _extract_pdf(path: Path) -> list[tuple[int, str]]:
    pages = [(i, (p.extract_text() or "").strip()) for i, p in enumerate(PdfReader(path).pages, 1)]
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


def _extract_html(path: Path) -> list[tuple[int, str]]:
    parser = _PageText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = "".join(parser.main_text or parser.all_text).strip()
    if len(text) < MIN_PAGE_CHARS:
        raise ValueError("html page has no usable text")
    return [(1, text)]


def _extract(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in (".html", ".htm"):
        return _extract_html(path)
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


def build_index(entries: list[ManifestEntry], raw_dir: Path, index_dir: Path) -> IngestReport:
    """Build a new index beside index_dir, validate it, then swap it in.

    Raises RuntimeError and leaves any existing index untouched if the build is invalid.
    """
    index_dir = Path(index_dir)
    new_dir = index_dir.with_name(index_dir.name + ".new")
    old_dir = index_dir.with_name(index_dir.name + ".old")
    shutil.rmtree(new_dir, ignore_errors=True)
    report = IngestReport()
    chunks: list[Chunk] = []
    for e in entries:
        try:
            pages = _extract(Path(raw_dir) / e.file)
        except Exception as exc:
            report.failures[e.file] = str(exc)
            continue
        for page, text in pages:
            chunks.extend(Chunk(piece, e.file, page, e.crop, e.diseases, e.title) for piece in _split(text))
    report.chunks = len(chunks)

    try:
        if not chunks:
            raise RuntimeError("new index has no chunks")
        if report.failures and len(report.failures) == len(entries):
            raise RuntimeError("every document failed extraction")
        new_dir.mkdir(parents=True)
        with (new_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c)) + "\n")
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
