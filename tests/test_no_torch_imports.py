"""Record checking and claims run without loading torch."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")
MODULES = [
    "sprout.eval.check_run",
    "sprout.eval.claims",
    "sprout.eval.records",
    "sprout.eval.metrics",
    "sprout.eval.stats",
    "sprout.eval.predict",
]


@pytest.mark.parametrize("module", MODULES)
def test_importing_does_not_load_torch(module: str) -> None:
    code = f"import sys, {module}; sys.exit(1 if 'torch' in sys.modules else 0)"
    env = {**os.environ, "PYTHONPATH": SRC}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, f"{module} loaded torch\n{done.stderr}"
