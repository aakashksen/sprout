"""E14 with fakes: the sample, what Sonnet is shown, resumable grading, the call cap, the agreement numbers, and a
record that check_run recomputes."""
import hashlib
import json
import math
import subprocess
import threading

import pytest

from sprout.eval import agreement
from sprout.eval.check_run import check_file
from sprout.experiments import e14_judge_check as e14
from sprout.llm.client import ClaudeCodeClient, FakeLLM, LLMError
from sprout.rag.judge import blind_view, rubric

DEV = [f"q{i}" for i in range(1, 9)]
QUESTIONS = [dict(id=q, category="supported", split="dev", question=f"Question {q}?", expected="answer") for q in DEV]
QUESTIONS.append(dict(id="t1", category="supported", split="test", question="Held out?", expected="answer"))
SYSTEMS = {"served": DEV[:4], "base": DEV, "ask": DEV}
LOCAL_NAME, LOCAL_REPLY = "judge-model-zzz", "LOCALREPLY"


def sh(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)


def judged_row(system, qid):
    ok = int(qid[1:]) % 2 == 0
    claims = [{"claim": f"Claim one of {qid}.", "cited": [1], "verdict": True, "judge_reply": LOCAL_REPLY},
              {"claim": f"Claim two of {qid} wrong.", "cited": [2], "verdict": ok, "judge_reply": LOCAL_REPLY}]
    return {"id": qid, "system": system, "question": f"Question {qid}?",
            "passages": [{"n": 1, "source": "a.pdf", "page": 1, "text": "Passage one."},
                         {"n": 2, "source": "b.pdf", "page": 2, "text": "Passage two."}],
            "answer": f"answer {qid}", "claims": claims, "rubric": rubric(), "judge_model": LOCAL_NAME}


def sonnet_reply(messages):
    claim = messages[-1]["content"].rsplit("Claim: ", 1)[1]
    return "NOT_SUPPORTED" if "wrong" in claim else "SUPPORTED"


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "datasets" / "rag").mkdir(parents=True)
    (r / "results" / "runs").mkdir(parents=True)
    (r / "datasets" / "rag" / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    (r / ".gitignore").write_text("results/e14/work/\nmlruns/\n")
    return r


def make_source(root, extra=()):
    rows = [judged_row(s, q) for s, qs in SYSTEMS.items() for q in qs] + [judged_row("base", i) for i in extra]
    runs = root / "results" / "runs"
    (runs / "E7-prompts-v2-test.judgments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (runs / "E7-prompts-v2-test.blind.jsonl").write_text("".join(json.dumps(blind_view(r)) + "\n" for r in rows))
    (runs / "E7-20261104T000000000000Z.json").write_text(json.dumps(
        {"config": {"judgments_file": "E7-prompts-v2-test.judgments.jsonl", "blind_file": "E7-prompts-v2-test.blind.jsonl"}}))
    sha = lambda n: hashlib.sha256((runs / n).read_bytes()).hexdigest()  # noqa: E731
    return {"experiment": "E14", "split": "rag_dev", "seed": 3, "sample_size": 10, "served_system": "served",
            "max_calls": 100, "judge_model": LOCAL_NAME, "grader_model": "sonnet", "device": "none",
            "source": {"e7_run_id": "E7-20261104T000000000000Z",
                       "judgments": "E7-prompts-v2-test.judgments.jsonl", "judgments_sha256": sha("E7-prompts-v2-test.judgments.jsonl"),
                       "blind": "E7-prompts-v2-test.blind.jsonl", "blind_sha256": sha("E7-prompts-v2-test.blind.jsonl")}}


def commit(root):
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "x")


def test_only_rag_dev_answers_are_sampled(root):
    cfg = make_source(root, extra=["t1"])
    with pytest.raises(ValueError, match="outside rag_dev"):
        e14.load_source(cfg, root)
    cfg = make_source(root)
    judged, _ = e14.load_source(cfg, root)
    sample = e14.draw_sample(judged, cfg)
    assert len(sample) == 10 and {e["id"] for e in sample} <= set(DEV)
    assert sum(e["stratum"] == "served" for e in sample) == 4 and sum(e["system"] == "served" for e in sample) == 4
    assert sample == e14.draw_sample(judged, cfg)  # the seed fixes the draw
    assert sample != e14.draw_sample(judged, {**cfg, "seed": 4})
    assert len({e["item"] for e in sample}) == 10


def test_a_superseded_or_changed_source_is_refused(root):
    cfg = make_source(root)
    (root / "results" / "e7").mkdir()
    (root / "results" / "e7" / "superseded.json").write_text(
        json.dumps({"superseded": {cfg["source"]["e7_run_id"]: {"by": "x", "reason": "y"}}, "needs_rerun": {}}))
    with pytest.raises(ValueError, match="superseded"):
        e14.load_source(cfg, root)
    (root / "results" / "e7" / "superseded.json").unlink()
    cfg["source"]["judgments_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        e14.load_source(cfg, root)


def test_sonnet_prompt_omits_the_local_grade(root):
    cfg = make_source(root)
    judged, blind = e14.load_source(cfg, root)
    llm = FakeLLM(reply=sonnet_reply, model="sonnet")
    sample = e14.draw_sample(judged, cfg)
    e14.grade(sample, blind, llm, root / "results" / "e14" / "work", cfg)
    assert len(llm.calls) == 20
    seen = json.dumps(llm.calls)
    for secret in (LOCAL_REPLY, LOCAL_NAME, "judge_reply", "verdict", "served", "gemma"):
        assert secret not in seen
    first = llm.calls[0]
    assert first[0] == {"role": "system", "content": rubric()}  # the same rubric code the local judge used
    assert "Passages:" in first[1]["content"] and "Claim: Claim" in first[1]["content"]
    assert all(set(c) == {"claim", "cited"} for b in blind.values() for c in b["claims"])


def test_prompts_are_json_safe(root):
    cfg = make_source(root)
    _, blind = e14.load_source(cfg, root)
    row = next(iter(blind.values()))
    row["passages"][0]["text"] = 'quote " back\\slash \x00 nul \ud800 lone \x1b[0m tab\tok \u00e9'
    row["claims"][0]["claim"] = 'say "hi" \x07 \\n'
    llm = FakeLLM(["SUPPORTED"], model="sonnet")
    e14.grade_claim(llm, row, 0)
    prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in llm.calls[0])
    json.dumps(prompt)
    prompt.encode("utf-8")
    assert "\x00" not in prompt and "\x1b" not in prompt and "\x07" not in prompt
    assert "tab\tok \u00e9" in prompt and e14.clean("plain text, 1/8 inch.") == "plain text, 1/8 inch."


def test_rows_are_saved_as_they_arrive_and_a_rerun_resumes(root):
    cfg = make_source(root)
    judged, blind = e14.load_source(cfg, root)
    sample = e14.draw_sample(judged, cfg)
    work = root / "results" / "e14" / "work"
    count = {"n": 0}

    def flaky(messages):
        count["n"] += 1
        if count["n"] == 8:
            raise LLMError("usage limit")
        return sonnet_reply(messages)

    with pytest.raises(LLMError, match="usage limit"):
        e14.grade(sample, blind, FakeLLM(reply=flaky, model="sonnet"), work, cfg)
    saved = [json.loads(x) for f in work.glob("claims-*.jsonl") for x in f.read_text().splitlines()]
    assert 6 <= len(saved) <= 8 and all({"id", "reply", "sonnet"} <= set(r) for r in saved)
    again = FakeLLM(reply=sonnet_reply, model="sonnet")
    ck = e14.grade(sample, blind, again, work, cfg)
    assert len(again.calls) == 20 - len(saved)  # nothing already saved is asked again
    assert len(ck.rows) == 20
    third = FakeLLM(reply=sonnet_reply, model="sonnet")
    e14.grade(sample, blind, third, work, cfg)
    assert third.calls == []


def test_at_most_two_calls_run_at_once(root):
    cfg = make_source(root)
    judged, blind = e14.load_source(cfg, root)
    live, peak, lock = [0], [0], threading.Lock()
    gate = threading.Event()

    def slow(messages):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        gate.wait(0.02)
        with lock:
            live[0] -= 1
        return "SUPPORTED"

    e14.grade(e14.draw_sample(judged, cfg), blind, FakeLLM(reply=slow, model="sonnet"), root / "w", cfg)
    assert peak[0] == 2


class Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_call_cap_refuses_and_makes_no_call(root, tmp_path, monkeypatch):
    cfg = make_source(root)
    judged, blind = e14.load_source(cfg, root)
    ran = []
    monkeypatch.setattr("shutil.which", lambda name: "C:/bin/claude.exe")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: ran.append(cmd) or Proc(json.dumps({"result": "SUPPORTED"})))
    counter = tmp_path / "claude_calls.json"
    counter.write_text(json.dumps({"calls": 15}))
    client = ClaudeCodeClient(model="sonnet", cap=30, counter_path=counter)  # 20 claims need 20 calls, 15 left
    work = root / "results" / "e14" / "work"
    with pytest.raises(LLMError, match="cap"):
        e14.grade(e14.draw_sample(judged, cfg), blind, client, work, cfg)
    assert ran == [] and json.loads(counter.read_text())["calls"] == 15 and not list(work.glob("claims-*.jsonl"))
    with pytest.raises(LLMError, match="limit"):  # the experiment's own limit
        e14.grade(e14.draw_sample(judged, cfg), blind, FakeLLM(["SUPPORTED"], model="sonnet"), work, {**cfg, "max_calls": 5})
    # the client itself also refuses once the cap is reached, before it starts a process
    counter.write_text(json.dumps({"calls": 30}))
    with pytest.raises(LLMError, match="cap"):
        client.complete([{"role": "user", "content": "x"}], max_tokens=8, timeout_s=5)
    assert ran == []


def test_a_reply_that_is_not_a_verdict_is_not_graded():
    assert e14.parse_verdict("SUPPORTED") is True and e14.parse_verdict("supported.") is True
    assert e14.parse_verdict("NOT_SUPPORTED") is False and e14.parse_verdict("Not supported") is False
    assert e14.parse_verdict("SUPPORTED, because the passage says so") is None
    assert e14.parse_verdict("") is None and e14.parse_verdict("UNSUPPORTED") is None


def test_kappa_matches_a_worked_example():
    # 50 claims: both supported 20, local only 5, other only 10, both not 15.
    # observed 0.70; local yes 0.5, other yes 0.6; chance 0.5*0.6 + 0.5*0.4 = 0.5; kappa (0.7-0.5)/(1-0.5) = 0.4
    local = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
    other = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    c = agreement._counts(local, other)
    assert c.tolist() == [20, 5, 10, 15]
    assert agreement.kappa(c) == pytest.approx(0.4)
    assert agreement.STATS["agreement"](c) == pytest.approx(0.7)
    assert math.isnan(agreement.kappa(agreement._counts([True, True], [True, True])))  # chance agreement 1: undefined


def test_direction_counts_each_disagreement_once():
    local = [True, True, False, False, True, False]
    other = [False, False, True, False, True, True]
    d = agreement.direction(local, other)
    assert d == {"local_kinder": 2, "local_stricter": 2}
    assert agreement.direction([True, True, True, False], [False, False, True, True]) == {
        "local_kinder": 2,
        "local_stricter": 1,
    }  # uneven, so swapped labels fail
    c = agreement._counts(local, other)
    assert d["local_kinder"] + d["local_stricter"] == len(local) - c[0] - c[3]  # every split claim, once
    rows = [{"id": "q", "system": "s", "local": local, "sonnet": [*other[:5], None]}]
    assert agreement.metrics_from_rows(rows, "s")["claims_not_graded"] == 1.0  # a claim without a verdict is left out
    assert agreement.metrics_from_rows(rows, "s")["claims_compared"] == 5.0


def graded_rows(root, cfg):
    judged, blind = e14.load_source(cfg, root)
    sample = e14.draw_sample(judged, cfg)
    ck = e14.grade(sample, blind, FakeLLM(reply=sonnet_reply, model="sonnet"), root / "results" / "e14" / "work", cfg)
    return e14.build_rows(sample, judged, blind, ck)


def test_build_rows_refuses_ungraded_claims(root):
    cfg = make_source(root)
    judged, blind = e14.load_source(cfg, root)
    sample = e14.draw_sample(judged, cfg)
    ck = e14.Checkpoint(root / "empty.jsonl")
    with pytest.raises(ValueError, match="not graded"):
        e14.build_rows(sample, judged, blind, ck)


def test_record_is_recomputed_by_check_run_and_a_changed_row_is_caught(root):
    cfg = make_source(root)
    commit(root)
    rows = graded_rows(root, cfg)
    path = e14.write_record(cfg, rows, root, "2026-01-01T00:00:00+00:00")
    rec = json.loads(path.read_text())
    assert rec["split"] == "rag_dev" and rec["n"] == 10 and rec["claude_calls"] == 20
    assert rec["config"]["device"] and rec["config"]["seed"] == 3 and rec["models"]["grader"] == "sonnet"
    served = rec["metrics"]["sample"]
    assert served["served_answers"] == 4 and served["served_local_supported"] == pytest.approx(
        sum(c["verdict"] for r in e14.load_source(cfg, root)[0] if r["system"] == "served" for c in r["claims"]) / 8)
    assert check_file(path, root, min_n=10) == []
    assert any("below --min-n" in p for p in check_file(path, root, min_n=200))

    pred = root / rec["predictions"]["sample"]
    original = pred.read_text()
    lines = original.splitlines()
    row = json.loads(lines[0])
    row["sonnet"] = [not row["sonnet"][0], *row["sonnet"][1:]]
    pred.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    assert any("does not recompute" in p for p in check_file(path, root, min_n=10))

    row = json.loads(lines[0])
    row["local"] = [not row["local"][0], *row["local"][1:]]
    pred.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    assert any("local grades differ" in p for p in check_file(path, root, min_n=10))

    row = json.loads(lines[0])
    row["id"] = "t1"  # a rag_test question
    pred.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    assert any("not a rag_dev question" in p for p in check_file(path, root, min_n=10))
    pred.write_text(original)
    assert check_file(path, root, min_n=10) == []
