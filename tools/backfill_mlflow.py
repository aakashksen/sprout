#!/usr/bin/env python3
"""Log existing run records into the local MLflow store (mlruns/), tagged with each record's run_id.
Records are only read. A record whose run_id is already tagged in the store is skipped, so it is safe to rerun."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sprout.eval.mlflow_log import log_record, store_uri  # noqa: E402
from sprout.eval.records import RUNS_REL  # noqa: E402

SIDECAR = re.compile(r".+\.(measurements|judged)\.json")


def backfill(root: Path) -> tuple[Counter, Counter]:
    """(logged now, already there) record counts per experiment."""
    logged, skipped = Counter(), Counter()
    for path in sorted((root / RUNS_REL).glob("*.json")):
        if SIDECAR.fullmatch(path.name):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        (logged if log_record(rec, root, skip_if_logged=True) else skipped)[rec["experiment"]] += 1
    return logged, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding results/runs and mlruns")
    a = ap.parse_args(argv)
    logged, skipped = backfill(a.root)
    print(f"store {store_uri(a.root)}")
    for name in sorted(logged | skipped):
        print(f"{name}: logged {logged[name]}, already there {skipped[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
