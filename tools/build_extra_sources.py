"""Rebuild datasets/manifests/extra_train.csv and datasets/class_map_extra.csv (CPU only; run from the repo root)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sprout.data.extra_sources import main  # noqa: E402

if __name__ == "__main__":
    main()
