"""Rebuild datasets/manifests/extra_train.csv and datasets/class_map_extra.csv; run from the repo root.

The copy-correlation and DINOv2 crop steps run on the GPU while holding a share (`taskctl gpu-acquire T4.8`,
released right after). If the GPU is busy it waits for a share; with no GPU at all it runs on the CPU.
Hashing and the rest of the rule stay on the CPU.
"""

import contextlib
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sprout.data.extra_sources import main  # noqa: E402

TASK = "T4.8"
WAIT_MINUTES = 240


def main_folder() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    return Path(common).parent


@contextlib.contextmanager
def gpu_share():
    import torch

    root = main_folder()

    def taskctl(verb: str) -> int:
        cmd = ["py", "-3.12", str(root / "tools" / "taskctl.py"), verb, TASK]
        return subprocess.run(cmd, cwd=root).returncode

    if not torch.cuda.is_available():
        yield "cpu"
        return
    # DINOv2 on the CPU would take hours, so wait for a share instead of falling back
    for _ in range(WAIT_MINUTES):
        if taskctl("gpu-acquire") == 0:
            break
        time.sleep(60)
    else:
        raise SystemExit("no GPU share within the wait; not falling back to the CPU")
    try:
        yield "cuda"
    finally:
        taskctl("gpu-release")


if __name__ == "__main__":
    main(gpu_share)
