import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from test_e12 import CLASSES, world  # noqa: F401  (the fixture)

import sprout.data.splits as splits
from sprout.data import near_extras as ne


class HalvesEncoder(torch.nn.Module):
    """Two numbers per photo: the mean of its left half and of its right half. A mirror swaps them."""

    def forward(self, x):
        return torch.stack([x[:, :, :, :112].mean(dim=(1, 2, 3)), x[:, :, :, 112:].mean(dim=(1, 2, 3))], dim=1)


class ColourEncoder(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


def test_the_higher_of_photo_and_mirror_counts_and_the_threshold_is_inclusive():
    extra = np.array([[1.0, 0.0], [0.75, 0.0], [0.7, 0.0]])
    mirror = np.array([[0.0, 0.0], [0.0, 0.0], [0.8, 0.0]])
    other = np.array([[1.0, 0.0]])
    cos = ne.best_cosine(extra, mirror, other)
    assert cos.tolist() == [1.0, 0.75, 0.8]  # the third is rescued by its mirror
    entries = ne.trim_entries(["c", "b", "a"], cos, np.array([-1.0, -1.0, -1.0]), 0.75)
    assert [e["path"] for e in entries] == ["a", "b", "c"]  # exactly 0.75 goes
    assert ne.trim_entries(["c"], np.array([0.7499]), np.array([-1.0]), 0.75) == []
    assert ne.best_cosine(extra, mirror, np.zeros((0, 2))).tolist() == [-1.0, -1.0, -1.0]


def test_union_counts_either_side_once(tmp_path):
    entries = ne.trim_entries(["a", "b", "c", "d"], np.array([0.9, 0.1, 0.8, 0.1]), np.array([0.9, 0.1, 0.1, 0.76]), 0.75)
    assert [e["path"] for e in entries] == ["a", "c", "d"]
    extra = pd.DataFrame({"path": list("abcd"), "label": "x", "group": "g", "split": "extra_train", "sha1": list("wxyz"),
                          "source": "s"})
    kept = ne.trimmed_manifest(extra, entries)
    assert list(kept["path"]) == ["b"]


def test_embeddings_are_unit_length_mirrors_differ_and_the_cache_is_keyed_by_paths(tmp_path):
    left = tmp_path / "left.png"
    im = Image.new("RGB", (32, 32), (0, 0, 0))
    im.paste((255, 255, 255), (0, 0, 16, 32))
    im.save(left)
    paths = ["left.png"]
    plain = ne.embed_paths(paths, HalvesEncoder(), "cpu", tmp_path, workers=1)
    flip = ne.embed_paths(paths, HalvesEncoder(), "cpu", tmp_path, mirror=True, workers=1)
    assert np.allclose(np.linalg.norm(plain, axis=1), 1) and not np.allclose(plain, flip)
    assert np.allclose(plain[0, ::-1], flip[0], atol=1e-3)  # the mirror swaps the two halves
    calls = []

    def compute():
        calls.append(1)
        return plain

    cache = tmp_path / "cache"
    ne.cached_embedding(cache, "a", paths, compute)
    ne.cached_embedding(cache, "a", paths, compute)
    assert len(calls) == 1  # second call read the file
    ne.cached_embedding(cache, "a", ["other.png"], compute)
    assert len(calls) == 2  # different paths, so recomputed
    assert not list(cache.glob("*.tmp")) and not list(cache.glob(".*.tmp"))


def _field():
    return pd.read_csv(splits.MANIFEST_DIR / "field.csv")


def test_build_lists_extras_near_the_evaluated_and_the_training_photos(world, monkeypatch):  # noqa: F811
    root = world / "repo"
    field, extra = _field(), splits.load_manifest("extra_train")
    ev_all = ne.source_rows(field, "plantdoc", "field_dev")
    tr_all = ne.source_rows(field, "plantdoc", "field_train")
    red, blue = CLASSES[0], CLASSES[1]
    only = lambda rows, label: rows[rows["label"] == label].reset_index(drop=True)  # noqa: E731
    opened = []
    real_open = Image.open

    def spy(fp, *a, **k):
        opened.append(str(fp))
        return real_open(fp, *a, **k)

    monkeypatch.setattr(Image, "open", spy)
    out = ne.build("plantdoc", only(ev_all, red), only(tr_all, red), extra, ColourEncoder(), "cpu", root, workers=1)
    assert out["extra_rows"] == 16 and out["against_evaluated"] == 8 and out["against_training"] == 8
    assert out["union"] == 8 and out["trimmed_rows"] == 8
    assert {e["path"] for e in out["remove"]} == set(extra[extra["label"] == red]["path"])
    assert all(e["cos_eval"] > 0.99 for e in out["remove"])
    # evaluated side red, training side blue: each side catches one colour, the union is everything
    out = ne.build("plantdoc", only(ev_all, red), only(tr_all, blue), extra, ColourEncoder(), "cpu", root, workers=1)
    assert (out["against_evaluated"], out["against_training"], out["union"], out["trimmed_rows"]) == (8, 8, 16, 0)
    assert not [p for p in opened if "field_test" in p]
    assert out["device"] == "cpu" and out["threshold"] == 0.75


def test_test_rows_are_refused(world):  # noqa: F811
    field = _field()
    with pytest.raises(ValueError, match="field_dev"):
        ne.source_rows(field, "plantdoc", "field_test")
    test_rows = field[(field["source"] == "plantdoc") & (field["split"] == "field_test")]
    dev = ne.source_rows(field, "plantdoc", "field_dev")
    with pytest.raises(ValueError, match="field_dev"):
        ne.build("plantdoc", test_rows, dev, splits.load_manifest("extra_train"), ColourEncoder(), "cpu", world / "repo")
