import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprout.data.splits import MANIFEST_DIR, load_manifest
from sprout.experiments import e15_final_model as e15

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "e15.json").read_text(encoding="utf-8"))
RULE = CFG["class_rule"]


def rows(label, split, n, source="s", prefix=""):
    return [{"path": f"{prefix}{label}/{source}/{split}/{i}.jpg", "label": label, "group": f"g{label}{source}{i}",
             "split": split, "sha1": f"{label}{source}{split}{i}", "source": source} for i in range(n)]


def fixture():
    """A: three sources of different sizes. B: one source, fewer photos than a draw. C: too few dev photos."""
    field, extra, lab = [], [], []
    for label, sizes in (("A", {"x": 100, "y": 400, "z": 5}), ("B", {"x": 100}), ("C", {"x": 200})):
        for src, n in sizes.items():
            field += rows(label, "field_train", n // 2, src)
            extra += rows(label, "extra_train", n - n // 2, src)
        field += rows(label, "field_dev", 3 if label == "C" else 12, "x")
        field += rows(label, "field_test", 12, "x")
        lab += rows(label, "lab_train", 300 if label != "B" else 50, "lab")
    lab += rows("L", "lab_train", 200, "lab")  # a lab-only class
    return pd.DataFrame(lab), pd.DataFrame(field), pd.DataFrame(extra)


def pools():
    lab, field, extra = fixture()
    kept, _, _ = e15.derive_classes(lab, field, extra, RULE)
    return e15.build_pools(lab, field, extra, kept), kept


def test_class_rule_gives_the_recorded_list():
    lab, field, extra = fixture()
    kept, dropped, table = e15.derive_classes(lab, field, extra, RULE)
    assert kept == ["A", "B"] and dropped == ["C", "L"]  # C has 3 dev photos, L no real photos
    assert table.loc["C", "dev"] == 3 and table.loc["A", "real_train"] == 505
    # the real manifests
    names = [p for p in ("lab", "field", "extra_train") if not (MANIFEST_DIR / f"{p}.csv").is_file()]
    if names:
        pytest.skip(f"manifests missing: {names}")
    kept, dropped, table = e15.derive_classes(load_manifest("lab"), load_manifest("field"),
                                              load_manifest("extra_train"), RULE)
    assert kept == sorted(CFG["classes"]) and len(kept) == 29 and len(dropped) == 9
    assert table.loc[kept, "real_train"].min() >= 80 and table.loc[kept, ["dev", "test"]].min().min() >= 10


def test_each_pass_draws_170_real_and_70_lab_per_class():
    (real, lab, _), kept = pools()
    got = e15.draw_pass(real, lab, CFG["settings"], 0)
    for c in kept:
        part = got[got["label"] == c]
        assert (part["source"] != "lab").sum() == 170 and (part["source"] == "lab").sum() == 70
    assert set(got["label"]) == set(kept)


def test_real_draws_are_even_across_sources():
    (real, lab, _), _ = pools()
    for p in range(5):
        got = e15.draw_pass(real, lab, CFG["settings"], p)
        a = got[(got["label"] == "A") & (got["source"] != "lab")]["source"].value_counts()
        assert set(a.index) == {"x", "y", "z"} and a.max() - a.min() <= 1 and a.sum() == 170
    # the small source repeats its 5 photos about the same number of times each
    z = a_z = got[(got["label"] == "A") & (got["source"] == "z")]["path"].value_counts()
    assert len(a_z) == 5 and z.max() - z.min() <= 1


def test_lab_draws_change_between_passes():
    (real, lab, _), _ = pools()
    a0 = e15.draw_pass(real, lab, CFG["settings"], 0)
    a1 = e15.draw_pass(real, lab, CFG["settings"], 1)
    again = e15.draw_pass(real, lab, CFG["settings"], 0)
    pick = lambda d: set(d[(d["label"] == "A") & (d["source"] == "lab")]["path"])  # noqa: E731
    assert pick(a0) != pick(a1) and pick(a0) == pick(again)
    assert len(pick(a0)) == 70 and len(pick(a0) | pick(a1)) > 70


def test_field_test_is_never_read(monkeypatch):
    lab, field, extra = fixture()
    (real, labp, sel) = e15.build_pools(lab, field, extra, ["A", "B"])
    for df in (real, labp, sel, e15.draw_pass(real, labp, CFG["settings"], 0)):
        assert not df["path"].str.contains("field_test").any()
    assert set(sel["split"]) == {"field_dev"}
    with pytest.raises(ValueError, match="test"):
        e15.require_not_test("field_test")
    with pytest.raises(ValueError, match="test"):
        e15.require_not_test("lab_test")
    # the class rule counts field_test rows and nothing else: no photo is opened
    import PIL.Image

    monkeypatch.setattr(PIL.Image, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened")))
    e15.derive_classes(lab, field, extra, RULE)


def test_old_models_are_masked_before_scoring():
    names = ["a", "b", "c", "d"]
    keep = ["a", "c"]
    probs = np.array([[0.1, 0.6, 0.2, 0.1],   # top class b is dropped
                      [0.5, 0.1, 0.3, 0.1],
                      [0.1, 0.1, 0.1, 0.7]])  # true class d is dropped: the row leaves
    labels = np.array([2, 0, 3])
    out_p, out_y, kept_rows = e15.restrict(probs, labels, names, keep)
    assert kept_rows.tolist() == [True, True, False]
    assert out_p.shape == (2, 2) and np.allclose(out_p.sum(axis=1), 1)
    assert np.allclose(out_p[0], [1 / 3, 2 / 3])  # 0.1 and 0.2 renormalized
    assert out_y.tolist() == [1, 0]                # labels now index the kept classes
    assert out_p[0].argmax() == 1                   # scored as the best kept class, not as a miss


def test_healthy_threshold_is_stricter():
    t = e15.healthy_threshold(0.88)
    assert 0.88 < t < 1 and abs(t - 0.94) < 1e-9
    names = ["x___healthy", "y"]
    cal = np.array([[0.90, 0.10], [0.10, 0.90], [0.97, 0.03]])
    keep = e15.answered(cal, names, 0.88, t, "___healthy")
    assert keep.tolist() == [False, True, True]  # healthy at 0.90 is now unsure, sick at 0.90 still answered
    assert e15.answered(cal, names, 0.88, 0.88, "___healthy").tolist() == [True, True, True]


def test_new_model_replaces_current_only_if_best_of_three():
    f1 = {"e1": 0.5, "convnext_joint": 0.78, "convnext_joint_extra": 0.80, e15.NEW: 0.80}
    assert e15.choose(f1, "convnext_joint")[0] == e15.NEW  # a tie with E12 goes to the new model
    f1[e15.NEW] = 0.79
    assert e15.choose(f1, "convnext_joint")[0] == "convnext_joint_extra"  # above E4, below E12
    f1["convnext_joint_extra"] = 0.78
    assert e15.choose(f1, "convnext_joint")[0] == e15.NEW
    f1[e15.NEW] = 0.78
    assert e15.choose(f1, "convnext_joint")[0] == e15.NEW  # all tied: the new model
    f1[e15.NEW] = 0.70
    assert e15.choose(f1, "convnext_joint")[0] == "convnext_joint"  # E4 before E12 on a tie
