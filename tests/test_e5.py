import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sprout.eval import metrics as M
from sprout.serve import gate as gate_mod
from sprout.serve.calibrate import apply_temperature, choose_threshold, decide, fit_temperature
from sprout.serve.gate import GateSettings, green_ratio, is_plant_image


def overconfident(n=600, k=5, seed=0):
    """Logits that are right 70% of the time but written with a scale of 6, so they claim far more."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < 0.7, labels, (labels + rng.integers(1, k, n)) % k)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), pred] += 4
    return logits * 3, labels


def test_temperature_reduces_ece_on_overconfident_fixture():
    logits, labels = overconfident()
    t = fit_temperature(logits, labels)
    before = M.ece(apply_temperature(logits, 1.0), labels)
    after = M.ece(apply_temperature(logits, t), labels)
    assert t > 1.5
    assert after < before / 2
    assert (apply_temperature(logits, t).argmax(1) == logits.argmax(1)).all()


def test_unsure_below_threshold():
    probs = np.array([0.5, 0.3, 0.2])
    assert decide(probs, 0.6).status == "unsure"
    assert decide(probs, 0.5).status == "answered"
    assert decide(probs, 0.5).confidence == 0.5


def test_top_k_is_limited():
    probs = np.array([0.30, 0.25, 0.2, 0.15, 0.1, 0.0])
    d = decide(probs, 0.9)
    assert [i for i, _ in d.top_k] == [0, 1, 2]
    low = decide(np.array([0.85, 0.11, 0.09, 0.05]), 0.5)
    assert [i for i, _ in low.top_k] == [0, 1]  # 0.10 and below are left out
    assert all(p > 0.10 for _, p in low.top_k)


def test_choose_threshold_rule():
    # rows below 0.75 confidence are wrong, so the best threshold that keeps half the rows is the median
    conf = np.linspace(0.5, 1.0, 100)
    probs = np.stack([conf, 1 - conf], axis=1)
    labels = np.where(conf > 0.75, 0, 1)
    t = choose_threshold(probs, labels, min_coverage=0.5)
    assert 0.75 < t <= conf[50]
    assert M.selective_metrics(probs, labels, t)["coverage"] >= 0.5
    assert choose_threshold(probs, labels, min_coverage=1.0) == conf.min()


def image(color, size=32):
    return Image.new("RGB", (size, size), color)


def test_green_rule_follows_the_old_definition():
    assert green_ratio(image((40, 160, 50))) == 1.0
    assert green_ratio(image((190, 160, 40))) == 0.0  # yellow
    assert green_ratio(image((120, 80, 40))) == 0.0  # brown
    half = image((120, 80, 40))
    half.paste(image((40, 160, 50), 16), (0, 0))
    assert green_ratio(half) == 0.25
