#!/usr/bin/env python3
"""Runs E6 stages in order from the repo root. Default: export, then probs (both resumable).
The latency stage is never in the default: run it on its own when the machine is idle, then run record.
    py -3.12 tools/run_e6.py                 # export, probs
    py -3.12 tools/run_e6.py latency         # alone
    py -3.12 tools/run_e6.py record          # commit results/e6 first
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sprout.experiments.e6_onnx import main  # noqa: E402

if __name__ == "__main__":
    for stage in sys.argv[1:] or ["export", "probs"]:
        code = main(["--stage", stage])
        if code:
            sys.exit(code)
