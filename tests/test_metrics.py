import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import f1_score

from sprout.eval.metrics import (
    confusion,
    ece,
    macro_f1,
    mrr,
    per_class_report,
    recall_at_k,
    selective_metrics,
)
from sprout.eval.predict import predict
from sprout.eval.stats import (
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    percentile,
)
from sprout.train.loop import build_model


def _paired(b, c, both=5):
    a = [True] * both + [True] * b + [False] * c
    bb = [True] * both + [False] * b + [True] * c
    return a, bb


def test_mcnemar_exact_b1_c9():
    assert mcnemar_exact(*_paired(b=1, c=9)) == pytest.approx(0.0215, abs=0.0005)


def test_mcnemar_no_discordant_is_one():
    assert mcnemar_exact([True, False], [True, False]) == 1.0


def test_macro_f1_matches_sklearn():
    y = [0, 0, 1, 1, 2, 2, 2, 1, 0]
    p = [0, 1, 1, 1, 2, 0, 2, 2, 0]
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))


def test_per_class_report_and_confusion():
    y, p = [0, 0, 1, 1], [0, 1, 1, 1]
    assert confusion(y, p, 2).tolist() == [[1, 1], [0, 2]]
    rep = per_class_report(y, p, ["a", "b"])
    assert rep["a"]["recall"] == 0.5
    assert rep["b"]["precision"] == pytest.approx(2 / 3)
    assert rep["b"]["support"] == 2


def test_ece_perfectly_calibrated_is_zero():
    probs = np.array([[1.0, 0.0]] * 4)
    assert ece(probs, np.array([0, 0, 0, 0])) == pytest.approx(0.0)


def test_ece_confident_half_right_is_half():
    probs = np.array([[1.0, 0.0]] * 4)
    assert ece(probs, np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_selective_threshold_above_one_has_zero_coverage():
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert selective_metrics(probs, np.array([0, 1]), 1.01)["coverage"] == 0.0


def test_selective_metrics_values():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    out = selective_metrics(probs, np.array([0, 1, 1]), 0.7)
    assert out["coverage"] == pytest.approx(2 / 3)
    assert out["selective_accuracy"] == 1.0
    assert out["confident_error_rate"] == 0.0


def test_recall_at_k_gold_rank_3():
    assert recall_at_k([3], 2) == 0.0
    assert recall_at_k([3], 3) == 1.0


def test_mrr_ranks_1_and_2():
    assert mrr([1, 2]) == 0.75


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_bootstrap_ci_resamples_whole_groups():
    values = [1] * 4 + [0] * 4
    groups = ["a"] * 4 + ["b"] * 4
    # two groups: group-level resampling can reach 0 and 1; item-level could not
    assert bootstrap_ci(values, groups, n=2000, seed=13) == (0.0, 1.0)


def test_bootstrap_ci_is_seeded():
    v, g = [1, 0, 1, 1, 0, 0], [1, 1, 2, 2, 3, 3]
    assert bootstrap_ci(v, g, n=500) == bootstrap_ci(v, g, n=500)


def test_paired_bootstrap_diff_all_better():
    d, lo, hi = paired_bootstrap_diff([1] * 20, [0] * 20, list(range(20)), n=500)
    assert (d, lo, hi) == (1.0, 1.0, 1.0)


def test_predict_uses_class_map_order(tmp_path):
    rows = []
    for i, label in enumerate(["b", "a"]):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (40, 30), (i * 90, 10, 10)).save(p)
        rows.append({"path": str(p), "label": label})
    out = predict(build_model("tiny_cnn", 2), pd.DataFrame(rows), {"a": 0, "b": 1})
    assert out.label.tolist() == [1, 0]
    assert out.probs.shape == (2, 2)
    assert np.allclose(out.probs.sum(axis=1), 1.0)
