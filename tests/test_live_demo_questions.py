"""The live demo's question flow with the E18 engine in it: same API to the page, no model, no key, no network."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_demo_under_test", ROOT / "tools" / "live_demo.py")
live = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live
spec.loader.exec_module(live)

from sprout.serve.question_engine import Bank, Params, load_questions  # noqa: E402

TL, PL, TE, PE = "Tomato___Late_blight", "Potato___Late_blight", "Tomato___Early_blight", "Potato___Early_blight"


def engine() -> live.Engine:
    eng = live.Engine.__new__(live.Engine)  # no model, no index, no writer
    eng.writer, eng.sources, eng.checks = None, {}, {}
    eng.use_questions(Bank(load_questions(), list(live.CLASSES), Params.load()))
    return eng


def check(top: dict[str, float]) -> live.Check:
    rest = (1 - sum(top.values())) / (len(live.CLASSES) - len(top))
    return live.Check("c1", {c: top.get(c, rest) for c in live.CLASSES}, "asking")


def answer(eng: live.Engine, chk: live.Check, ans: str) -> None:
    assert eng.answer(chk, chk.question["id"], ans, lambda d: None)


def test_the_demo_class_list_is_the_calibration_class_list():
    chosen = json.loads((ROOT / "results/e15/refit/chosen.json").read_text(encoding="utf-8"))
    assert list(live.CLASSES) == chosen["classes"]


def test_a_sure_photo_gets_no_question():
    eng = engine()
    chk = check({TL: 0.97})
    eng.next_step(chk, lambda d: None)
    assert chk.status == "final" and chk.question is None and chk.final["name"] == "Tomato late blight"
    assert chk.final["sure"].endswith("% sure")


def test_the_plant_question_comes_first_with_the_page_api():
    eng = engine()
    chk = check({TL: 0.35, PL: 0.30, TE: 0.15})
    eng.next_step(chk, lambda d: None)
    q = chk.question
    assert q["id"] == "plant" and q["kind"] == "plant" and {o["value"] for o in q["options"]} == {"tomato", "potato"}
    view = eng.view(chk)
    assert set(view) >= {"id", "status", "plant", "message", "odds", "question", "asked", "final"}
    assert set(view["question"]) == {"id", "text", "source", "options"} and len(view["odds"]) == 3


def test_answering_the_plant_pushes_the_other_plant_down_and_the_next_question_is_about_that_plant():
    eng = engine()
    chk = check({TL: 0.35, PL: 0.30, TE: 0.15, PE: 0.1})
    eng.next_step(chk, lambda d: None)
    answer(eng, chk, "potato")
    assert chk.probs[TL] + chk.probs[TE] < 0.1 and chk.plant_known
    assert chk.question is not None and chk.question["id"] != "plant" and chk.question["id"].startswith("pot-")
    assert chk.asked[0]["answer"] == "Potato"


def test_plant_not_listed_leaves_the_odds_and_goes_on():
    eng = engine()
    chk = check({TL: 0.35, PL: 0.30, TE: 0.15, PE: 0.1})
    before = dict(chk.probs)
    eng.next_step(chk, lambda d: None)
    answer(eng, chk, "not sure")
    assert chk.probs == pytest.approx(before) and chk.asked[0]["answer"] == "Not sure"


def test_yes_to_a_question_raises_what_it_supports_and_the_flow_can_end_sure():
    eng = engine()
    chk = check({PL: 0.5, PE: 0.4})
    eng.next_step(chk, lambda d: None)
    assert chk.question["id"].startswith("pot-")
    q = next(x for x in eng.bank.questions if x.id == chk.question["id"])
    target = q.yes_supports[0]
    before = chk.probs[target]
    answer(eng, chk, "yes")
    assert chk.probs[target] > before
    for _ in range(8):
        if chk.status != "asking":
            break
        q = next(x for x in eng.bank.questions if x.id == chk.question["id"])
        answer(eng, chk, "yes" if PL in q.yes_supports else "no")
    assert chk.status == "final" and chk.final_cls == PL


def test_not_sure_and_unknown_answers_count_as_not_sure_and_are_recorded():
    eng = engine()
    chk = check({PL: 0.5, PE: 0.4})
    eng.next_step(chk, lambda d: None)
    answer(eng, chk, "maybe")
    assert chk.asked[0]["answer"] == "Not sure" and "qi" in chk.asked[0]


def test_the_question_budget_and_the_split_rule_end_the_flow_without_a_model():
    eng = engine()
    chk = check({"Apple___Apple_scab": 0.4, "Apple___Black_rot": 0.3, "Apple___healthy": 0.2})
    n = 0
    while chk.status == "asking" and n < 20:
        eng.next_step(chk, lambda d: None) if chk.question is None else answer(eng, chk, "not sure")
        n += 1
    assert chk.status == "final" and eng.symptom_count(chk) <= eng.bank.params.max_questions


def test_a_stale_question_id_is_refused():
    eng = engine()
    chk = check({PL: 0.5, PE: 0.4})
    eng.next_step(chk, lambda d: None)
    assert eng.answer(chk, "not-the-question", "yes", lambda d: None) is False


def test_the_healthy_bar_is_applied_to_a_healthy_answer():
    eng = engine()
    bar = eng.bank.params
    assert bar.healthy_bar > bar.bar
    p = 0.5 * (bar.bar + bar.healthy_bar)
    chk = check({"Apple___healthy": p, "Apple___Apple_scab": 1 - p - 0.001})
    eng.next_step(chk, lambda d: None)
    assert chk.status == "asking" and chk.question is not None  # sure for a disease, not sure enough for a healthy leaf


def test_no_question_is_written_by_a_model():
    assert not hasattr(live.Engine, "write_pool") and not hasattr(live, "K_YES")
