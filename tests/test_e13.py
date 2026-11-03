import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from sprout.eval.records import split_info, subset_positions

FIELD = ["path,label,group,split,sha1,source"] + [
    f"f/{i}.jpg,{'AB'[i % 2] if i < 10 else 'C'},h{i},field_dev,s{i},{'plantdoc' if i % 3 else 'plantseg'}"
    for i in range(12)
]
LAB = ["path,label,group,split,sha1"] + [f"l/{i}.jpg,{'AB'[i % 2]},g{i},lab_dev,l{i}" for i in range(4)]
LOG = (
    "version,image_id,path,source,split,label,reason_code,note\n"
    "1,s1,f/1.jpg,plantdoc,field_dev,B,wrong_plant,x\n"
    "1,s4,f/4.jpg,plantdoc,field_dev,A,not_plant_photo,y\n"
    "2,s6,f/6.jpg,plantseg,field_dev,A,label_contradicted,z\n"
)


@pytest.fixture
def root(tmp_path):
    for rel, text in {"datasets/manifests/field.csv": FIELD, "datasets/manifests/lab.csv": LAB}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(text) + "\n")
    (tmp_path / "datasets/label_exclusions.csv").write_text(LOG)
    return tmp_path


def _paths(root, subset):
    info = split_info("field_dev", root)
    return [info.ids[i] for i in subset_positions(subset, "field_dev", info, root)]


def test_clean_subset_drops_exclusions_up_to_its_version(root):
    everything = [f"f/{i}.jpg" for i in range(12)]
    assert _paths(root, "clean:0") == everything
    assert _paths(root, "clean:1") == [p for p in everything if p not in ("f/1.jpg", "f/4.jpg")]
    assert "f/6.jpg" not in _paths(root, "clean:2")
    with pytest.raises(ValueError, match="version"):
        _paths(root, "clean:3")


def test_clean_subset_combines_with_the_old_subsets(root):
    shared = _paths(root, "shared_classes")  # labels A and B, not C
    assert len(shared) == 10
    assert _paths(root, "shared_classes+clean:1") == [p for p in shared if p not in ("f/1.jpg", "f/4.jpg")]
    both = _paths(root, "source:plantdoc+clean:1")
    assert both == [p for p in _paths(root, "source:plantdoc") if p not in ("f/1.jpg", "f/4.jpg")]
