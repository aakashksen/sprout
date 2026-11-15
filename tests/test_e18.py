"""E18's stages on saved data: the noise model, the old flows, and numbers that must not drift."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sprout.experiments import e18_questions as e18
from sprout.experiments import e18_record as rec
from sprout.serve.question_engine import NO, UNSURE, YES, Bank, Params, load_questions

ROOT = Path(__file__).resolve().parents[1]


def test_noise_flips_yes_and_no_but_never_not_sure_and_can_blank_an_answer():
    assert e18.noisy(YES, 0.05, 0.9, 0.10, 0.0) == NO
    assert e18.noisy(NO, 0.05, 0.9, 0.10, 0.0) == YES
    assert e18.noisy(YES, 0.5, 0.9, 0.10, 0.0) == YES
    assert e18.noisy(UNSURE, 0.0, 0.9, 1.0, 0.0) == UNSURE
    assert e18.noisy(YES, 0.9, 0.05, 0.0, 0.20) == UNSURE


def test_calibrate_is_a_temperature_softmax_and_keeps_the_order():
    raw = np.array([[0.7, 0.2, 0.1]])
    out = e18.calibrate(raw, 2.0)
    assert out.sum() == pytest.approx(1) and out.argmax() == 0 and out[0, 0] < 0.7


def test_the_strength_vector_round_trips():
    p = Params(yes_row=(0.6, 0.1, 0.3), no_row=(0.2, 0.5, 0.3), gamma=0.9)
    back = e18.params_from_vector(e18.vector_from_params(p), p)
    assert np.allclose(back.yes_row, p.yes_row) and np.allclose(back.no_row, p.no_row) and back.gamma == pytest.approx(0.9)


def test_the_e11_original_flow_asks_nothing_when_one_class_dominates():
    class D:  # a stand-in for Data
        probs = np.array([[0.95, 0.03, 0.02]])
        labels = np.array([0])
        crops = ["A", "A", "A"]

    p, n = rec.old_e11([], [], lambda i, q: YES, D, 0)
    assert n == 0 and np.allclose(p, D.probs[0])


def test_the_old_live_flow_asks_the_plant_question_first():
    class D:
        probs = np.array([[0.4, 0.1, 0.4, 0.1]])
        labels = np.array([0])
        crops = ["A", "A", "B", "B"]

    p, n = rec.old_live([], [], lambda i, q: YES, D, 0)  # no symptom questions: only the plant question is possible
    assert n == 1 and p[:2].sum() > 0.9


def _dev():
    try:
        return e18.load_dev(ROOT)
    except (OSError, ValueError, KeyError):
        pytest.skip("the saved dev rows are absent")


def test_plant_question_alone_reproduces_the_audit_replay_on_field_dev():
    dev = _dev()
    qs = load_questions(ROOT / "datasets/followup_v2/questions.jsonl")
    bank = Bank(qs, dev.classes, Params.load(ROOT / "configs/e18_params.json"))
    table = e18.grower_table(ROOT, dev.classes, [q.id for q in qs])
    base = (dev.probs.argmax(1) == dev.labels).mean()
    out = e18.simulate(bank, dev, table, use_symptoms=False)
    plant = (out["final"].argmax(1) == dev.labels).mean()
    assert round(base, 4) == 0.8104 and round(plant, 4) == 0.8399  # the 2026-11-14 audit's numbers


def test_questions_do_not_break_a_perfect_grower_run_and_improve_on_the_plant_question():
    dev = _dev()
    qs = load_questions(ROOT / "datasets/followup_v2/questions.jsonl")
    bank = Bank(qs, dev.classes, Params.load(ROOT / "configs/e18_params.json"))
    table = e18.grower_table(ROOT, dev.classes, [q.id for q in qs])
    plant = e18.simulate(bank, dev, table, use_symptoms=False)["final"].argmax(1) == dev.labels
    full = e18.simulate(bank, dev, table)
    assert (full["final"].argmax(1) == dev.labels).mean() > plant.mean() + 0.05
    assert full["symptom_questions"].max() <= bank.params.max_questions
    # a photo that is sure from the start is never asked anything
    sure = np.array([bank.sure(p) for p in dev.probs])
    assert full["symptom_questions"][sure].sum() == 0 and not full["plant_asked"][sure].any()
