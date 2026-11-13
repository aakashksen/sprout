import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_package_imports():
    import sprout

    assert sprout.__name__ == "sprout"


@pytest.mark.real
def test_marked_real_is_never_run_by_default():
    raise AssertionError("a real-marked test ran without -m real")


def test_real_marker_excluded_by_default():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__)), "-q", "--collect-only"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout
    assert "test_marked_real_is_never_run_by_default" not in out
    assert "deselected" in out
