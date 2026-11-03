import csv
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from sprout.data import class_report as cr
from sprout.rag.evalset import answer_candidates, index_pages, load_questions, normalize
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


# What a document would have to say to answer each out-of-scope question: the plant or its family
# (a section on cucurbit diseases can answer a watermelon question without the word watermelon)
# and the problem, in the same chunk. Patterns that match only generic words were narrowed after
# reading their hits.
ABSTAIN_SPEC = {
    "not_in_documents-001": (["magnolia", "ornamental tree", "shade tree"], "leaf spot"),
    "not_in_documents-002": (["orchid", "epiphyte"], "rot|mushy"),
    "not_in_documents-003": (["cannabis", "hemp", "marijuana"], "powdery mildew"),
    "not_in_documents-004": (["mango", "tropical fruit", "tropical tree"], "anthracnose"),
    "not_in_documents-005": (["banana", "plantain", "tropical fruit"], "panama|fusarium"),
    "not_in_documents-006": (["peony", "paeonia"], "botrytis|blight"),
    "not_in_documents-007": (["lettuce", "leafy green", "salad green", "leafy vegetable"], r"tip ?burn|slimy|soft rot"),
    "not_in_documents-008": (["lavender", "aromatic herb", "mediterranean herb"], "root rot|dying back|wet soil"),
    "not_in_documents-009": (["avocado", "subtropical", "evergreen fruit tree"], r"tip ?burn|brown(ing)? (leaf )?tips|leaf tip"),
    "not_in_documents-010": (["pineapple", "bromeliad", "tropical fruit"], "heart rot"),
    "not_in_documents-011": (["hibiscus", "mallow", "okra", "flowering shrub"], r"bud drop|buds? (drop|fall)"),
    "not_in_documents-012": (["hosta", "asparagus", "agave", "yucca"], "crown rot"),
    "not_in_documents-013": (["broccoli", "brassica", "cole crop", "crucifer", "cauliflower"], r"(head|curd) (rot|browning)|brown(ing)? (head|curd)"),
    "not_in_documents-014": (["tulip", "tulipa", "flowering bulb"], r"bulb rot|storage rot|rot in storage"),
    "not_in_documents-015": (["celery", "apiaceae", "umbellifer"], "late blight|septoria"),
    "not_in_documents-016": (["boxwood", "buxus", "evergreen shrub", "hedge"], "blight"),
    "not_in_documents-017": (["pomegranate", "punica"], "crack|rot"),
    "not_in_documents-018": (["persimmon", "diospyros"], r"blacken|black spot|black fruit|fruit rot"),
    "not_in_documents-019": (["daylily", "hemerocallis", "lily"], r"bud|rot|mush"),
    "not_in_documents-020": (["hops", "humulus"], "downy mildew"),
    "not_in_documents-021": (["papaya", "tropical fruit", "tropical tree", "subtropical"], r"sunken|dark spots|anthracnose"),
    "not_in_documents-022": (["sugarcane", "sugar cane", "cane crop"], "red rot"),
    "not_in_documents-023": (["sunflower", "helianthus"], r"head rot|brown(ed)? heads?|heads? (turn|rot)"),
    "not_in_documents-024": (["basil", "mint family", "lamiaceae"], "downy mildew"),
    "not_in_documents-025": (["cabbage", "brassica", "cole crop", "crucifer", "broccoli", "cauliflower"], r"clubroot|club root|plasmodiophora"),
    "not_in_documents-026": (["parsley", "apiaceae", "umbellifer"], r"leaf spot|yellowing"),
    "not_in_documents-027": (["bamboo", "ornamental grass"], r"\brust\b"),
    "not_in_documents-028": (["cacao", "cocoa", "theobroma"], r"pod rot|black pod"),
}


def test_answer_candidates_flags_a_family_document_and_ignores_a_passing_mention(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    rows = [
        {"text": "Powdery mildew of squash, zucchini and pumpkin shows white spots.", "source": "s.html", "page": 1},
        {"text": "Plant lettuce, zucchini and beans in rows.", "source": "g.html", "page": 1},
    ]
    chunks.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert answer_candidates(chunks, ["zucchini", "squash", "cucurbit"], "powdery mildew") == [("s.html", 1)]
    assert answer_candidates(chunks, ["lettuce"], r"tip ?burn|slimy") == []


@needs_new_index
def test_abstain_questions_have_no_answer_in_the_index():
    questions = [q for q in load_questions(QUESTIONS) if q.category == "not_in_documents"]
    assert {q.id for q in questions} == set(ABSTAIN_SPEC)
    for q in questions:
        plants, topic = ABSTAIN_SPEC[q.id]
        assert answer_candidates(_new_index(), plants, topic) == [], q.id


UNKNOWN_SOURCE = "unknown: inherited from the legacy project, source not confirmed"


def test_every_row_has_a_url_or_the_exact_unknown_phrase():
    rows = _manifest_rows()
    unknown = [r["file"] for r in rows if r["source_url"] == UNKNOWN_SOURCE]
    for r in rows:
        assert r["source_url"] == UNKNOWN_SOURCE or r["source_url"].startswith(("https://", "http://")), r["file"]
        assert r["publisher"].strip() or r["source_url"] == UNKNOWN_SOURCE, r["file"]
    assert all(not NEW_FILE.match(f) for f in unknown)
    sources = (ROOT / "docs" / "document-sources.md").read_text(encoding="utf-8")
    assert f"{len(unknown)} of the 28 original documents have an unknown source" in sources
    evidence = {r["file"] for r in csv.DictReader((ROOT / "docs" / "original-doc-sources.csv").open(encoding="utf-8"))}
    assert set(unknown) <= evidence
