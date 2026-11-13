import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "check_candidates_doc.py"


def load():
    spec = importlib.util.spec_from_file_location("check_candidates_doc", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLASSES = ["A___x", "B___y"]
CAND = "### T\nURL: u\nPublisher: p\nLicense: l\nApproved: no, needs the person\n"
GOOD = f"# D\n\nintro\n\n## A___x\n\n{CAND}\n## B___y\n\nNo acceptable candidate found. Couldn't confirm one.\n"


def test_complete_doc_has_no_errors():
    assert load().check(GOOD, CLASSES) == []


def test_missing_class_section_is_reported():
    errors = load().check(GOOD.replace("## B___y", "## C___z"), CLASSES)
    assert "B___y: missing section" in errors


def test_missing_line_is_reported():
    errors = load().check(GOOD.replace("Publisher: p\n", ""), CLASSES)
    assert errors == ["A___x / T: missing Publisher line"]


def test_approved_must_say_needs_the_person():
    errors = load().check(GOOD.replace("Approved: no, needs the person", "Approved: yes"), CLASSES)
    assert errors == ["A___x / T: Approved must be 'no, needs the person'"]


def test_empty_section_must_say_no_candidate():
    errors = load().check(GOOD.replace("No acceptable candidate found.", "TODO"), CLASSES)
    assert errors == ["B___y: no candidates and no 'No acceptable candidate found' statement"]


def test_classes_come_from_coverage_csv(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("label,crop,symptoms_described\nA___x,a,no\nQ,a,yes\nB___y,b,no\n", encoding="utf-8")
    assert load().uncovered_classes(csv) == CLASSES


def test_real_doc_passes():
    r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
