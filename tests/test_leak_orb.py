import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "leak_orb.py"


def load():
    spec = importlib.util.spec_from_file_location("leak_orb", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def textured(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    return Image.fromarray(small).resize((384, 384), Image.BICUBIC).convert("RGB")


@pytest.fixture
def tree(tmp_path):
    held, extra = [], []
    for i in range(3):
        textured(i).save(tmp_path / f"f{i}.png")
        held.append((f"f{i}.png", "Apple___healthy", "field_dev" if i else "field_test"))
    textured(1).save(tmp_path / "copy.png")                       # same scene as f1
    textured(1).crop((20, 20, 360, 360)).save(tmp_path / "cropped.png")  # cropped f1
    textured(99).save(tmp_path / "other.png")                     # unrelated
    for name in ("copy.png", "cropped.png", "other.png"):
        extra.append((name, "x", "plantpathology2021"))
    pd.DataFrame(held, columns=["path", "label", "split"]).assign(group="g", sha1="s", source="plantdoc") \
        .to_csv(tmp_path / "field.csv", index=False)
    pd.DataFrame(extra, columns=["path", "label", "source"]).to_csv(tmp_path / "extra.csv", index=False)
    return tmp_path


def test_finds_planted_copies_and_the_control_but_not_an_unrelated_photo(tree):
    out = tree / "out.csv"
    code = load().main([str(tree / "extra.csv"), str(tree / "field.csv"), str(out),
                        "--sources", "plantpathology2021", "--root", str(tree)])
    assert code == 0
    found = pd.read_csv(out)
    hits = {Path(p).name: set(found[found.extra == p].ref) for p in found.extra.unique()}
    assert hits["copy.png"] == {"f1.png"}
    assert "f1.png" in hits["cropped.png"]
    assert "other.png" not in hits
    assert "control_crop.jpg" in hits          # the control is found, so the search works
    assert (tree / "control_crop.jpg").is_file()


def test_source_filter_leaves_other_sources_out(tree):
    out = tree / "out.csv"
    load().main([str(tree / "extra.csv"), str(tree / "field.csv"), str(out),
                 "--sources", "nothing", "--root", str(tree)])
    assert "copy.png" not in " ".join(pd.read_csv(out).extra)

