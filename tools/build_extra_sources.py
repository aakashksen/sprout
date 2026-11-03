"""Rebuild datasets/manifests/extra_train.csv and datasets/class_map_extra.csv; run from the repo root.

The copy-correlation step runs on the GPU while holding a share (`taskctl gpu-acquire T4.5`,
released right after). If the GPU is busy or missing it runs on the CPU and says so.
Hashing and the rest of the rule stay on the CPU.
"""

import contextlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sprout.data.extra_sources import main  # noqa: E402

TASK = "T4.5"


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

    if not torch.cuda.is_available() or taskctl("gpu-acquire") != 0:
        yield "cpu"
        return
    try:
        yield "cuda"
    finally:
        taskctl("gpu-release")


if __name__ == "__main__":
    main(gpu_share)
