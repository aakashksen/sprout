import csv
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from sprout.data import class_report as cr
from sprout.rag.evalset import index_pages, load_questions, normalize
from sprout.rag.ingest import build_index, load_manifest, missing_from_manifest

PAGE = """<html><head><title>T</title><style>p{color:red}</style><script>var secret=1;</script></head>
<body><nav>Home Menu Login</nav><main><h1>Cherry powdery mildew</h1>
<p>Early infections are small, round, paler spots on leaves that can curl slightly.</p>
<script>evil()</script></main><footer>Copyright</footer></body></html>"""

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "knowledge" / "manifest.csv"
RAW = ROOT / "knowledge" / "raw"
QUESTIONS = ROOT / "datasets" / "rag" / "questions.jsonl"
NEW_FILE = re.compile(r"^\d{3}-")


def test_html_page_becomes_text_without_navigation_or_scripts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "p.html").write_text(PAGE, encoding="utf-8")
    m = tmp_path / "m.csv"
    m.write_text("file,crop,diseases,title,publisher,source_url,date\np.html,cherry,powdery mildew,T,P,https://x.edu/p,\n")
    build_index(load_manifest(m), raw, tmp_path / "index")
    text = (tmp_path / "index" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "paler spots on leaves" in text
    assert "Login" not in text and "evil" not in text and "secret" not in text and "Copyright" not in text


def _main_folder() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent


def _new_index() -> Path | None:
    for base in (ROOT, _main_folder()):
        path = base / "knowledge" / "index-t65" / "chunks.jsonl"
        if path.exists():
            return path
    return None


needs_new_index = pytest.mark.skipif(_new_index() is None, reason="knowledge/index-t65 is git-ignored and absent here")


def _manifest_rows() -> list[dict]:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_every_new_document_has_source_url_and_publisher():
    rows = _manifest_rows()
    new = [r for r in rows if NEW_FILE.match(r["file"])]
    assert len(new) >= 100
    for r in new:
        assert r["source_url"].startswith("https://"), r["file"]
        assert r["publisher"].strip(), r["file"]
        assert r["title"].strip(), r["file"]
        assert r["crop"].strip(), r["file"]
    assert len({r["source_url"] for r in new}) == len(new)
    assert len({r["file"] for r in rows}) == len(rows)


@pytest.mark.skipif(not RAW.exists(), reason="knowledge/raw is git-ignored and absent here")
def test_every_file_is_in_the_manifest():
    missing, extra = missing_from_manifest(load_manifest(MANIFEST), RAW)
    assert missing == []
    assert extra == []


def test_no_raw_document_is_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files", "knowledge"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked == ["knowledge/manifest.csv"]
    ignored = subprocess.run(
        ["git", "check-ignore", "knowledge/raw/x.pdf", "knowledge/index-t65/chunks.jsonl"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split()
    assert len(ignored) == 2


def _original_lines(path: Path, original: set[str]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if json.loads(line)["source"] in original]


@needs_new_index
def test_gold_labels_still_hold():
    chunks = _new_index()
    pages = index_pages(chunks)
    original = {r["file"] for r in _manifest_rows() if not NEW_FILE.match(r["file"])}
    gold = [q for q in load_questions(QUESTIONS) if q.gold_source]
    assert gold
    for q in gold:
        assert q.gold_source in original, f"{q.id}: gold document {q.gold_source} is not an original document"
        text = pages[(q.gold_source, q.gold_page)]
        assert all(normalize(t) in text for t in q.answer_terms), q.id
    old = chunks.parent.parent / "index" / "chunks.jsonl"
    if old.exists():
        assert _original_lines(chunks, original) == _original_lines(old, original)


@needs_new_index
def test_every_class_has_a_coverage_row():
    labels = sorted(pd.read_csv(cr.MANIFESTS / "lab.csv")["label"].unique())
    rows = cr.load_coverage()
    assert sorted(rows["label"]) == labels
    assert set(rows["symptoms_described"]) <= {"yes", "no", "not applicable"}
    pages = index_pages(_new_index())
    files = {r["file"] for r in _manifest_rows()}
    for _, r in rows.iterrows():
        assert r["evidence"].strip(), r["label"]
        if r["symptoms_described"] == "yes":
            docs, nums = r["document"].split("; "), [int(n) for n in str(r["page"]).split(";")]
            assert all(d in files for d in docs), r["label"]
            assert all(any((d, n) in pages for d in docs) for n in nums), r["label"]
        if r["symptoms_described"] == "no":
            assert "Queries" in r["evidence"], r["label"]
    assert set(rows[rows["symptoms_described"] == "no"]["label"]) == set()
