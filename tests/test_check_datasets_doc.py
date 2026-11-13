import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "check_datasets_doc.py"


def load():
    spec = importlib.util.spec_from_file_location("check_datasets_doc", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = "# D\n\nintro\n\n## A\n\nURL: u\nLicense: l\nClasses: c\nApproved: yes\n"


def test_complete_doc_has_no_errors():
    assert load().check(GOOD) == []


def test_missing_line_is_reported():
    errors = load().check(GOOD.replace("License: l\n", ""))
    assert errors == ["A: missing License line"]


def test_bad_approved_value_is_reported():
    errors = load().check(GOOD.replace("Approved: yes", "Approved: maybe"))
    assert errors == ["A: Approved must be 'yes' or 'no, needs the person'"]


def test_no_datasets_is_an_error():
    assert load().check("# D\n") == ["no dataset sections found"]


def test_real_doc_passes():
    r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
