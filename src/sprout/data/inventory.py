import hashlib
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root) -> pd.DataFrame:
    """One row per image under root/<class>/, with columns path, label, sha1."""
    root = Path(root)
    rows = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls.iterdir()):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((f.as_posix(), cls.name, file_sha1(f)))
    return pd.DataFrame(rows, columns=["path", "label", "sha1"])
