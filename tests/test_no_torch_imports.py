"""Modules that do not need torch must not load it, so checks and serving stay light."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ENV = {**os.environ, "PYTHONPATH": str(SRC)}
LIGHT_MODULES = [
    "sprout.data.check_splits",
    "sprout.data.class_report",
    "sprout.data.field",
    "sprout.data.grouping",
    "sprout.data.inventory",
    "sprout.data.masks",
    "sprout.data.splits",
    "sprout.segment.data",
    "sprout.segment.crop",
    "sprout.eval.check_run",
    "sprout.eval.claims",
    "sprout.eval.records",
    "sprout.eval.metrics",
    "sprout.eval.stats",
    "sprout.eval.predict",
    "sprout.serve.followup",
    "sprout.serve.export",
    "sprout.serve.identify",
    "sprout.preprocess_np",
]


def _modules(package: str) -> list[str]:
    root = SRC / package.replace(".", "/")
    return [package] + [f"{package}.{p.stem}" for p in sorted(root.glob("*.py")) if p.stem != "__init__"]


def test_serving_modules_do_not_import_torch():
    names = [*_modules("sprout.web"), *_modules("sprout.rag"), *_modules("sprout.llm"), "sprout.serve.types"]
    code = (
        "import importlib, sys\n"
        f"for n in {names!r}:\n    importlib.import_module(n)\n"
        "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "sys.exit('torch was imported: ' + ', '.join(bad[:3]) if bad else 0)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr[-500:]


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_importing_a_light_module_does_not_load_torch(module):
    code = f"import sys, {module}; sys.exit(1 if 'torch' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV)
    assert done.returncode == 0, f"{module} loaded torch\n{done.stderr}"
