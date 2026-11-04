"""E14H with fakes: the seeded claim draw, the separate read set, four calls at once, resumable grading, parse failures
counting against Haiku, the paired interval and the pass rule, and a record that check_run recomputes."""
import json
import threading

import pytest
import test_e14
from test_e14 import commit, graded_rows, make_source

from sprout.eval.check_run import check_file
from sprout.experiments import e14_judge_check as e14
from sprout.experiments import e14h_haiku_check as h
from sprout.llm.client import FakeLLM, LLMError

MODEL = "haiku-model-x"
root = test_e14.root  # the same temporary repository fixture


def setup_cfg(root, n=8):
    """E14's record on the fixture (10 answers, 20 claims) and an E14H config over it."""
    e14_cfg = make_source(root)
    (root / "configs").mkdir()
    (root / "configs" / "e14.json").write_text(json.dumps(e14_cfg))
    commit(root)
    e14.write_record(e14_cfg, graded_rows(root, e14_cfg), root, "2026-01-01T00:00:00+00:00")
    rec_id = next((root / "results" / "runs").glob("E14-*.json")).stem
    claims = h.load_claims({"e14_run_id": rec_id}, root)
    ids = sorted(claims)
    picks = [ids[0], ids[3], ids[7], ids[12]]
    reads = [{"tag": f"{'AB'[i // 2]}{i % 2 + 1}", "group": "AB"[i // 2], "item": c.split(":")[0], "k": int(c.split(":")[1]),
              "qid": claims[c]["qid"], "system": claims[c]["system"], "local": claims[c]["local"],
              "sonnet": claims[c]["sonnet"], "reader": i % 2 == 0, "close": i == 1} for i, c in enumerate(picks)]
    (root / "configs" / "e14h_reads.json").write_text(json.dumps(reads))
    cfg = {"experiment": "E14H", "split": "rag_dev", "seed": 5, "sample_size": n, "model": MODEL, "workers": 4,
           "max_calls": 50, "e14_config": "configs/e14.json", "e14_run_id": rec_id, "reads": "configs/e14h_reads.json",
           "reference": "sonnet", "device": "none"}
    (root / "configs" / "e14h.json").write_text(json.dumps(cfg))
    commit(root)
    return cfg, claims, reads


def replies(messages):
    return "NOT_SUPPORTED" if "wrong" in messages[-1]["content"].rsplit("Claim: ", 1)[1] else "SUPPORTED"


def run(root, cfg, claims, reads, reply=replies):
    _, blind = e14.load_source(h.load_e14_config(cfg, root), root)
    ck = h.Checkpoint(h.checkpoint_path(cfg, root / "results" / "e14" / "work"))
    llm = FakeLLM(reply=reply, model=cfg["model"])
    h.grade(h.tasks_for(cfg, claims, reads), blind, llm, ck, cfg)
    return ck, llm


def test_the_draw_is_seeded_and_the_read_set_is_graded_apart(root):
    cfg, claims, reads = setup_cfg(root)
    ids = h.draw_sample(claims, cfg["seed"], cfg["sample_size"])
    assert len(ids) == 8 and len(set(ids)) == 8 and set(ids) <= set(claims)
    assert ids == h.draw_sample(claims, cfg["seed"], cfg["sample_size"]) != h.draw_sample(claims, 6, 8)
    tasks = h.tasks_for(cfg, claims, reads)
    assert len(tasks) == 8 + 4 and len({t[0] for t in tasks}) == 12  # a claim in both sets is asked twice
    with pytest.raises(ValueError, match="cannot draw"):
        h.draw_sample(claims, 1, 99)


def test_four_calls_run_at_once_each_is_saved_and_a_rerun_resumes(root):
    cfg, claims, reads = setup_cfg(root)
    live, peak, lock, gate = [0], [0], threading.Lock(), threading.Event()

    def slow(messages):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        gate.wait(0.03)
        with lock:
            live[0] -= 1
        return replies(messages)

    ck, llm = run(root, cfg, claims, reads, slow)
    assert peak[0] == 4 and len(llm.calls) == 12 and len(ck.rows) == 12
    saved = [json.loads(x) for x in ck.path.read_text().splitlines()]
    assert {"id", "reply", "haiku"} <= set(saved[0])
    _, again = run(root, cfg, claims, reads)
    assert again.calls == []


def test_a_failure_keeps_what_arrived_and_makes_no_record(root):
    cfg, claims, reads = setup_cfg(root)
    count = {"n": 0}

    def flaky(messages):
        count["n"] += 1
        if count["n"] == 7:
            raise LLMError("There's an issue with the selected model (claude-haiku-5-5)")
        return replies(messages)

    with pytest.raises(LLMError, match="selected model"):
        run(root, cfg, claims, reads, flaky)
    ck = h.Checkpoint(h.checkpoint_path(cfg, root / "results" / "e14" / "work"))
    assert 4 <= len(ck.rows) < 12
    with pytest.raises(ValueError, match="not graded"):
        h.build_rows(cfg, claims, reads, ck)
    ck2, again = run(root, cfg, claims, reads)
    assert len(again.calls) == 12 - len(ck.rows) and len(ck2.rows) == 12


def test_the_budget_is_checked_before_any_call(root):
    cfg, claims, reads = setup_cfg(root)
    with pytest.raises(LLMError, match="limit"):
        run(root, {**cfg, "max_calls": 5}, claims, reads)


def rows_with(haiku_matches, items=8):
    """Rows whose local judge disagrees with Sonnet on every claim, and whose Haiku matches Sonnet (or opposes it)."""
    out = []
    for i in range(items):
        son = i % 2 == 0
        out.append({"id": f"i{i}:0", "item": f"i{i}", "qid": f"q{i}", "system": "served" if i < 4 else "base", "claim": "c",
                    "local": (not son) if i != 0 else son, "sonnet": son, "haiku": son if haiku_matches else (not son),
                    "haiku_reply": "x"})
    return out


def test_an_unreadable_reply_counts_as_disagreeing_with_sonnet():
    assert h.effective(None, True) is False and h.effective(None, False) is True
    assert h.effective(True, False) is True and h.effective(False, False) is False
    rows = rows_with(True)
    clean = h.sample_metrics(rows, "served")
    assert clean["haiku_agreement"] == 1.0 and clean["haiku_kappa"] == 1.0 and clean["parse_failures"] == 0
    rows[1] = {**rows[1], "haiku": None}
    broken = h.sample_metrics(rows, "served")
    assert broken["parse_failures"] == 1.0 and broken["haiku_agreement"] == 7 / 8
    assert broken["haiku_stricter"] + broken["haiku_kinder"] == 1.0  # the failure is one split claim, not a free match


def test_worked_numbers_for_both_judges():
    # 8 claims. Local judge: agrees with Sonnet on claim 0 only (7 splits). Haiku matches Sonnet everywhere.
    m = h.sample_metrics(rows_with(True), "served")
    assert m["claims"] == 8 and m["answers"] == 8 and m["served_claims"] == 4
    assert m["haiku_agreement"] == 1.0 and m["local_agreement"] == pytest.approx(1 / 8)
    assert m["sonnet_supported"] == 0.5 and m["haiku_supported"] == 0.5
    # local says the opposite of Sonnet on 7 claims: kinder where Sonnet said no (claims 1,3,5,7 -> local yes)
    assert m["local_kinder"] == 4 and m["local_stricter"] == 3 and m["haiku_kinder"] == 0
    assert m["kappa_difference"] == pytest.approx(m["haiku_kappa"] - m["local_kappa"])


def test_the_rule_calls_haiku_worse_only_when_the_whole_interval_is_below_zero():
    assert h.decision({"kappa_difference": (-0.3, -0.01)}) == "local"
    assert h.decision({"kappa_difference": (-0.3, 0.0)}) == "haiku"
    assert h.decision({"kappa_difference": (-0.3, 0.2)}) == "haiku"
    assert h.decision({"kappa_difference": (0.1, 0.4)}) == "haiku"
    with pytest.raises(ValueError, match="undefined"):
        h.decision({})
    bad = rows_with(False, items=40)  # Haiku opposes Sonnet on every claim, the local judge is mixed
    for r in bad:
        r["local"] = r["sonnet"]  # the local judge now agrees with Sonnet everywhere
    i = h.sample_intervals(bad, "served", 2000, 1)
    assert i["kappa_difference"][1] < 0 and h.decision(i) == "local"
    good = rows_with(True, items=40)
    for r in good:
        r["local"] = not r["sonnet"]
    assert h.decision(h.sample_intervals(good, "served", 2000, 1)) == "haiku"


def test_the_interval_resamples_answers_and_is_paired():
    rows = rows_with(True, items=12)
    rows += [{**r, "id": r["id"].replace(":0", ":1")} for r in rows]  # two claims per answer
    a = h.sample_intervals(rows, "served", 1000, 3)
    assert a == h.sample_intervals(rows, "served", 1000, 3)
    assert {"haiku_kappa", "local_kappa", "kappa_difference", "haiku_agreement", "sonnet_supported"} <= set(a)
    assert a["haiku_agreement"] == (1.0, 1.0)  # a perfect judge can't move under resampling


def test_the_read_set_is_scored_against_the_reader():
    read = [{"tag": "A1", "group": "A", "reader": True, "close": False, "local": True, "sonnet": False, "haiku": True},
            {"tag": "A2", "group": "A", "reader": False, "close": True, "local": True, "sonnet": False, "haiku": False},
            {"tag": "B1", "group": "B", "reader": False, "close": True, "local": True, "sonnet": False, "haiku": None}]
    m = h.read_metrics(read)
    assert m["n"] == 3 and m["haiku_right"] == 2 and m["local_right"] == 1 and m["sonnet_right"] == 2  # a failure is wrong
    assert m["clear_n"] == 1 and m["clear_haiku_right"] == 1 and m["served_n"] == 2 and m["other_haiku_right"] == 0
    assert m["parse_failures"] == 1 and m["haiku_with_local"] == 1 and m["haiku_with_sonnet"] == 1


def test_the_record_is_recomputed_by_check_run_and_a_changed_row_is_caught(root):
    cfg, claims, reads = setup_cfg(root)
    ck, _ = run(root, cfg, claims, reads)
    sample, read = h.build_rows(cfg, claims, reads, ck)
    sessions = [{"started_at": "2026-01-01T00:00:00+00:00", "seconds": 12.5, "workers": 4, "saved_after": 12}]
    path = h.write_record(cfg, sample, read, sessions, root, "2026-01-01T00:00:00+00:00")
    rec = json.loads(path.read_text())
    assert rec["experiment"] == "E14H" and rec["split"] == "rag_dev" and rec["n"] == 8 and rec["claude_calls"] == 12
    assert rec["models"]["judge_under_test"] == MODEL and rec["config"]["workers"] == 4
    assert rec["config"]["wall_seconds"] == 12.5 and rec["config"]["rubric_sha256"] == h.rubric_hash()
    assert check_file(path, root, min_n=8) == []

    decided = h.write_decision(path, root)
    on_disk = json.loads((root / h.DECISION_REL).read_text())
    assert on_disk == decided and decided["first_judge"] in ("haiku", "local")
    assert decided["record_id"] == rec["run_id"] and decided["model"] == MODEL
    assert decided["rubric_sha256"] == h.rubric_hash() and decided["interval"]["low"] <= decided["interval"]["high"]

    pred = root / rec["predictions"]["sample"]
    original = pred.read_text()
    lines = original.splitlines()

    def change(**kw):
        pred.write_text("\n".join([json.dumps({**json.loads(lines[0]), **kw}), *lines[1:]]) + "\n")
        return check_file(path, root, min_n=8)

    assert any("does not follow" in p for p in change(haiku=not json.loads(lines[0])["haiku"]))
    assert any("differ from E14" in p for p in change(sonnet=not json.loads(lines[0])["sonnet"]))
    assert any("differ from E14" in p for p in change(local=not json.loads(lines[0])["local"]))
    pred.write_text(original)
    assert check_file(path, root, min_n=8) == []
