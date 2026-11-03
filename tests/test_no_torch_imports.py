import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _modules(package: str) -> list[str]:
    root = SRC / package.replace(".", "/")
    return [package] + [f"{package}.{p.stem}" for p in sorted(root.glob("*.py")) if p.stem != "__init__"]


def test_serving_modules_do_not_import_torch():
    names = [*_modules("sprout.web"), *_modules("sprout.rag"), "sprout.serve.types"]
    code = (
        "import importlib, sys\n"
        f"for n in {names!r}:\n    importlib.import_module(n)\n"
        "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "sys.exit('torch was imported: ' + ', '.join(bad[:3]) if bad else 0)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(SRC)})
    assert out.returncode == 0, out.stderr[-500:]
