"""The E18 question engine: update rule, stopping, plant question first, healthy support, and the bank files."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from sprout.serve.question_engine import NO, UNSURE, YES, Bank, Params, Question, expected_calibration_error, load_questions, run

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy", "Potato___Early_blight", "Potato___Late_blight"]
EB, LB, TH, PEB, PLB = range(5)
PARAMS = Params(yes_row=(0.8, 0.05, 0.15), no_row=(0.1, 0.7, 0.2), gamma=0.8, plant_down=0.02, bar=0.9, healthy_bar=0.95,
                max_questions=4, min_gain=0.02)


def q(id: str, yes: list[str], no: list[str], crop: str = "tomato") -> Question:
    return Question(id, id + "?", "leaf_top", crop, tuple(yes), tuple(no), ({"document": "d", "page": 1, "passage": "p"},))


BANK_Q = [
    q("rings", [CLASSES[EB]], [CLASSES[LB], CLASSES[TH]]),
    q("fuzz", [CLASSES[LB]], [CLASSES[EB], CLASSES[TH]]),
    q("normal", [CLASSES[TH]], [CLASSES[EB], CLASSES[LB]]),
    q("potato-rings", [CLASSES[PEB]], [CLASSES[PLB]], crop="potato"),
]


def bank(params: Params = PARAMS) -> Bank:
    return Bank(BANK_Q, CLASSES, params)


def odds(*v: float) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / a.sum()


# ---- the update rule ----

def test_yes_raises_the_supported_class_and_lowers_the_class_a_no_supports():
    b, p = bank(), odds(0.4, 0.4, 0.1, 0.05, 0.05)
    after = b.update(p, 0, YES, 0)  # "rings": yes supports early blight, no supports late blight and healthy
    assert after[EB] > p[EB] and after[LB] < p[LB] and after[TH] < p[TH]
    assert after.sum() == pytest.approx(1.0)


def test_no_raises_the_classes_a_no_supports():
    b, p = bank(), odds(0.4, 0.4, 0.1, 0.05, 0.05)
    after = b.update(p, 0, NO, 0)
    assert after[LB] > p[LB] and after[TH] > p[TH] and after[EB] < p[EB]


def test_the_update_is_the_stated_likelihood_ratio():
    b, p = bank(), odds(0.5, 0.5, 0, 0, 0)
    after = b.update(p, 0, YES, 0)
    assert after[EB] / after[LB] == pytest.approx(0.8 / 0.1)  # P(yes | yes-role) / P(yes | no-role)
    after = b.update(p, 0, UNSURE, 0)
    assert after[EB] / after[LB] == pytest.approx(0.15 / 0.2)


def test_not_sure_changes_little_and_a_bystander_sits_between_the_two_roles():
    b = bank()
    p = odds(0.3, 0.3, 0.1, 0.15, 0.15)
    after = b.update(p, 0, YES, 0)
    ratio = after / p
    # a class neither list names (potato) is scaled by the mean of the two rows: between the Y and the N class
    assert ratio[LB] < ratio[PEB] < ratio[EB]


def test_later_questions_count_for_less():
    b, p = bank(), odds(0.5, 0.5, 0, 0, 0)
    first = b.update(p, 0, YES, 0)
    third = b.update(p, 0, YES, 2)
    assert first[EB] > third[EB] > 0.5


# ---- picking ----

def test_the_next_question_is_the_one_that_best_splits_the_top_three():
    b = bank()
    p = odds(0.5, 0.5, 0.0001, 0.00005, 0.00005)  # early vs late blight
    assert b.questions[b.pick(p)].id in ("rings", "fuzz")
    g = b.gains(p)
    assert g[3] == pytest.approx(0, abs=1e-6)  # a potato question tells nothing about two tomato diseases


def test_asked_questions_are_not_asked_again_and_no_split_returns_none():
    b = bank()
    p = odds(0.5, 0.5, 0.0001, 0.00005, 0.00005)
    first = b.pick(p)
    second = b.pick(p, [first])
    assert second is not None and second != first
    assert b.pick(p, [0, 1, 2, 3]) is None


# ---- running a check ----

def test_nothing_is_asked_when_the_photo_is_already_sure():
    b = bank()
    tr = run(b, odds(0.97, 0.01, 0.01, 0.005, 0.005), lambda o: None, lambda i: pytest.fail("asked"))
    assert tr.asked == [] and tr.stop == "sure_from_photo"


def test_a_healthy_top_class_needs_the_higher_bar():
    b = bank()
    assert b.sure(odds(0.92, 0.04, 0.02, 0.01, 0.01))  # disease top: bar 0.9
    assert not b.sure(odds(0.01, 0.02, 0.92, 0.025, 0.025))  # healthy top: bar 0.95
    assert b.sure(odds(0.01, 0.01, 0.96, 0.01, 0.01))


def test_the_plant_question_comes_first_and_only_when_two_plants_fit():
    b = bank()
    mixed = odds(0.3, 0.1, 0.05, 0.3, 0.25)
    asked_with: list[list[str]] = []
    tr = run(b, mixed, lambda o: (asked_with.append(o), "Tomato")[1], lambda i: NO)
    assert tr.asked[0] == ("plant", -1) and tr.plant_asked
    assert asked_with[0] == ["Potato", "Tomato"]  # the plants the photo fits, likeliest first
    assert tr.final[3] + tr.final[4] < 0.1  # the other plant is pushed down
    one_plant = odds(0.4, 0.4, 0.1, 0.0005, 0.0005)
    tr = run(b, one_plant, lambda o: pytest.fail("plant asked"), lambda i: YES)
    assert not tr.plant_asked


def test_plant_not_listed_leaves_the_odds_alone_and_the_flow_goes_on():
    b = bank()
    mixed = odds(0.3, 0.1, 0.05, 0.3, 0.25)
    tr = run(b, mixed, lambda o: None, lambda i: UNSURE, use_symptoms=False)
    assert np.allclose(tr.final, mixed) and tr.plant_asked


def test_plant_question_alone_is_the_whole_flow_when_symptom_questions_are_off():
    b = bank()
    tr = run(b, odds(0.3, 0.1, 0.05, 0.3, 0.25), lambda o: "Potato", lambda i: pytest.fail("asked"), use_symptoms=False)
    assert tr.n_symptom == 0 and tr.final[PEB] + tr.final[PLB] > 0.9


def test_stops_when_the_bar_is_reached():
    b = bank()
    tr = run(b, odds(0.5, 0.45, 0.04, 0.005, 0.005), lambda o: "Tomato", lambda i: YES if b.questions[i].id == "rings" else UNSURE)
    assert tr.stop == "sure" and b.sure(tr.final) and tr.n_symptom <= 2


def test_stops_at_the_question_budget():
    p = Params(yes_row=(0.5, 0.3, 0.2), no_row=(0.4, 0.4, 0.2), gamma=1.0, bar=0.999, healthy_bar=0.9995, max_questions=2, min_gain=0.0)
    b = bank(p)
    tr = run(b, odds(0.5, 0.5, 0.0001, 0.00005, 0.00005), lambda o: "Tomato", lambda i: UNSURE)
    assert tr.stop == "max_questions" and tr.n_symptom == 2


def test_stops_when_no_question_splits_the_top_three():
    only_potato = Bank([BANK_Q[3]], CLASSES, PARAMS)
    tr = run(only_potato, odds(0.5, 0.5, 0.0001, 0.00005, 0.00005), lambda o: "Tomato", lambda i: YES)
    assert tr.stop == "no_split" and tr.n_symptom == 0


# ---- healthy support ----

def test_a_no_to_a_disease_symptom_supports_healthy_and_so_does_yes_to_looks_normal():
    b = bank()
    p = odds(0.3, 0.3, 0.3, 0.05, 0.05)
    by_no = b.update(p, 0, NO, 0)  # no rings
    by_yes = b.update(p, 2, YES, 0)  # the plant looks normal
    assert by_no[TH] > p[TH] and by_yes[TH] > p[TH]
    assert by_yes[TH] > by_yes[EB] and by_yes[TH] > by_yes[LB]


def test_healthy_can_win_after_questions():
    b = bank()
    tr = run(b, odds(0.3, 0.3, 0.3, 0.05, 0.05), lambda o: "Tomato", lambda i: YES if b.questions[i].id == "normal" else NO)
    assert int(np.argmax(tr.final)) == TH


# ---- calibration helper ----

def test_ece_is_zero_for_a_perfectly_calibrated_set_and_large_for_an_overconfident_one():
    conf = np.array([0.9] * 10)
    assert expected_calibration_error(conf, np.array([True] * 9 + [False])) == pytest.approx(0, abs=1e-9)
    assert expected_calibration_error(conf, np.array([True] * 5 + [False] * 5)) == pytest.approx(0.4)


# ---- the shipped files ----

def test_the_shipped_bank_is_consistent():
    questions = load_questions(ROOT / "datasets/followup_v2/questions.jsonl")
    classes = json.loads((ROOT / "results/e15/refit/chosen.json").read_text(encoding="utf-8"))["classes"]
    assert len(questions) >= 50 and len({x.id for x in questions}) == len(questions)
    for x in questions:
        assert x.text.endswith("?") and x.passages and x.yes_supports
        assert set(x.yes_supports) | set(x.no_supports) <= set(classes)
        assert not set(x.yes_supports) & set(x.no_supports)
        assert x.kind in {"leaf_top", "leaf_underside", "stem", "fruit", "spread", "timing", "weather"}
    healthy = [x for x in questions if any(c.endswith("healthy") for c in x.yes_supports)]
    assert len(healthy) >= 4  # questions that support a healthy class
    kinds = {x.kind for x in questions}
    assert {"leaf_underside", "stem", "fruit", "spread", "timing", "weather"} <= kinds  # things a photo cannot show


def test_the_bank_covers_the_commonly_confused_top_three_pairs():
    questions = load_questions(ROOT / "datasets/followup_v2/questions.jsonl")
    pairs = [("Tomato___Bacterial_spot", "Tomato___Septoria_leaf_spot"), ("Tomato___Early_blight", "Tomato___Late_blight"),
             ("Tomato___Early_blight", "Tomato___Septoria_leaf_spot"), ("Tomato___Bacterial_spot", "Tomato___Early_blight"),
             ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Northern_Leaf_Blight"),
             ("Potato___Early_blight", "Potato___Late_blight"), ("Apple___Apple_scab", "Apple___Cedar_apple_rust"),
             ("Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight"),
             ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"),
             ("Tomato___Late_blight", "Tomato___Leaf_Mold"), ("Apple___Apple_scab", "Apple___Black_rot"),
             ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___healthy"), ("Apple___Apple_scab", "Apple___healthy"),
             ("Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy"), ("Grape___Black_rot", "Grape___healthy")]
    for a, c in pairs:
        n = sum(1 for x in questions if (a in x.yes_supports and c in x.no_supports) or (c in x.yes_supports and a in x.no_supports))
        assert n >= 1, (a, c)


def test_every_passage_is_in_the_cited_guide_page():
    chunks = ROOT / "knowledge/index/chunks.jsonl"
    if not chunks.is_file():
        pytest.skip("the guide index is git-ignored and absent here")

    def norm(t: str) -> str:
        t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("ﬁ", "fi")
        return re.sub(r"\s+", "", t).casefold()

    pages: dict[tuple[str, int], list[str]] = {}
    for line in chunks.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        pages.setdefault((c["source"], c["page"]), []).append(norm(c["text"]))
    for x in load_questions(ROOT / "datasets/followup_v2/questions.jsonl"):
        for p in x.passages:
            assert any(norm(p["passage"]) in t for t in pages.get((p["document"], p["page"]), [])), (x.id, p["passage"][:60])


def test_the_grower_table_is_complete_and_answers_only_yes_no_or_not_sure():
    classes = json.loads((ROOT / "results/e15/refit/chosen.json").read_text(encoding="utf-8"))["classes"]
    questions = load_questions(ROOT / "datasets/followup_v2/questions.jsonl")
    rows = [json.loads(x) for x in (ROOT / "datasets/followup_v2/grower_answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["true_class"], r["question"]) for r in rows} == {(c, x.id) for c in classes for x in questions}
    assert {r["answer"] for r in rows} <= {"yes", "no", "not sure"}


def test_the_shipped_params_load_and_are_proper_likelihoods():
    p = Params.load(ROOT / "configs/e18_params.json")
    assert sum(p.yes_row) == pytest.approx(1) and sum(p.no_row) == pytest.approx(1)
    assert p.yes_row[YES] > p.no_row[YES] and p.no_row[NO] > p.yes_row[NO]  # a yes favours the yes class, a no the no class
    assert 0 < p.gamma <= 1 and p.healthy_bar > p.bar
