"""Fetch the public Hugging Face mirror timm/plant-pathology-2021 into a new git-ignored folder, one file at a time.

Resumable: a file already complete (size matches the hub's) is skipped, a partial one continues.
Refuses to start if the download is over 40 GB or would leave under 40 GB free. No login is used.
Run from the main folder: python -I tools/download_plant_pathology.py [destination]
"""

import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "timm/plant-pathology-2021"
REVISION = "ce9371685ebc24cf250bd02a8e0f6274b0c775d3"
MAX_DOWNLOAD = 40e9
MIN_FREE_AFTER = 40e9
DEFAULT_DEST = Path("datasets") / "raw" / "plantpathology2021_dl"


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else -1


def main(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = sorted(
        (s.rfilename, s.size) for s in info.siblings if s.rfilename.endswith((".parquet", ".md"))
    )  # fmt: skip
    total = sum(size for _, size in files)
    todo = sum(size for name, size in files if _size(dest / name) != size)
    free = shutil.disk_usage(dest).free
    print(f"{len(files)} files, {total / 1e9:.2f} GB; free {free / 1e9:.0f} GB")
    if total > MAX_DOWNLOAD or free - todo < MIN_FREE_AFTER:
        raise SystemExit("too large or too little disk left; stop and ask first")
    start = time.time()
    for name, size in files:
        target = dest / name
        if _size(target) == size:
            continue
        hf_hub_download(
            REPO, name, repo_type="dataset", revision=REVISION, local_dir=dest
        )
        print(f"{name} {size} bytes, {time.time() - start:.0f}s elapsed", flush=True)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEST)
