"""E7 stages with fakes: checkpoints, gold matching, floors, judging files, and records that check_run recomputes."""
import json
import subprocess
from pathlib import Path

import pytest

from sprout.eval.check_run import check_file
from sprout.experiments import e7_rag as e7
from sprout.llm.client import FakeLLM
from sprout.rag.answer import AnswerConfig
from sprout.rag.evalset import Question
from sprout.rag.judge import blind_view, judge_answer, split_claims
from sprout.rag.retrieve import Passage, RetrievalConfig, Retriever

CHUNKS = [
    ("Early blight shows concentric rings on potato leaves. Rings grow in warm wet weather.", "potato.pdf", 1, "potato"),
    ("Store tubers cool and dry after harvest.", "potato.pdf", 2, "potato"),
    ("Thrips are sampled in tomato flowers each week. Seed is treated in hot water.", "tomatodoc.md", 1, "tomato"),
    ("Tomato transplants are hardened before planting out.", "tomatodoc.md", 1, "tomato"),
]
QUESTIONS = [
    dict(id="s1", category="supported", split="dev", question="What do early blight lesions look like on potato leaves?",
         expected="answer", gold_source="potato.pdf", gold_page=1, answer_terms=["concentric rings"]),
    dict(id="s2", category="supported", split="dev", question="How are thrips sampled in tomato flowers?",
         expected="answer", gold_source="tomatodoc.md", gold_page=1, answer_terms=["thrips are sampled"]),
    dict(id="a1", category="not_in_documents", split="dev", question="How do I treat leaf spot on magnolia?",
         expected="abstain", absent_terms=["magnolia"]),
    dict(id="w1", category="wrong_crop", split="dev", question="Early blight rings on my grape leaves, what now?",
         expected="crop_mismatch", crop="grape", trap_terms=["early blight"]),
    dict(id="c1", category="ambiguous", split="dev", question="My plant has yellow leaves.", expected="clarify", missing=["crop"]),
    dict(id="t1", category="supported", split="test", question="Held out.", expected="answer",
         gold_source="potato.pdf", gold_page=2, answer_terms=["tubers"]),
]


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def root(tmp_path):
    tmp_path = tmp_path / "repo"
    tmp_path.mkdir()
    (tmp_path / "datasets" / "rag").mkdir(parents=True)
    (tmp_path / "datasets" / "manifests").mkdir()
    (tmp_path / "datasets" / "rag" / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    (tmp_path / "datasets" / "manifests" / "lab.csv").write_text(
        "path,label\nx,Potato___Early_blight\nx,Potato___Late_blight\nx,Potato___healthy\nx,Tomato___Early_blight\n")
    index = tmp_path / "knowledge" / "index"
    index.mkdir(parents=True)
    with (index / "chunks.jsonl").open("w") as f:
        for text, source, page, crop in CHUNKS:
            f.write(json.dumps(dict(text=text, source=source, page=page, crop=crop, diseases=[], title="t")) + "\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return tmp_path


def retriever(root):
    return Retriever(root / "knowledge" / "index", config=RetrievalConfig(min_score=0.0))


def test_dev_questions_exclude_the_test_split(root):
    assert [q.id for q in e7.dev_questions(root)] == ["s1", "s2", "a1", "w1", "c1"]


def test_plant_and_diagnosis_come_from_the_question_not_the_gold(root):
    qs = {q.id: q for q in e7.dev_questions(root)}
    crops, classes = {"potato", "tomato"}, e7.load_classes(root)
    crop, ident = e7.simulate_identification(qs["s1"], crops, classes)
    assert crop == "potato" and ident.top.label == "Potato___Early_blight"
    assert [a.label for a in ident.alternatives] == ["Potato___Late_blight"]
    assert e7.simulate_identification(qs["a1"], crops, classes) == (None, None)
    assert e7.simulate_identification(qs["c1"], crops, classes) == (None, None)
    assert e7.simulate_identification(qs["w1"], crops, classes) == ("grape", None)  # grape has no class named early blight


def test_single_page_documents_need_an_answer_term_in_the_chunk(root):
    qs = {q.id: q for q in e7.dev_questions(root)}
    single = e7.single_page_sources(root / "knowledge" / "index")
    assert single == {"tomatodoc.md"}
    right = Passage("Thrips are sampled in tomato flowers", "tomatodoc.md", 1, "tomato", 1.0)
    other = Passage("Tomato transplants are hardened", "tomatodoc.md", 1, "tomato", 1.0)
    assert e7.gold_rank([other, right], qs["s2"], single) == 2
    page = Passage("Anything on the right page", "potato.pdf", 1, "potato", 1.0)  # multi-page: page match is enough
    assert e7.gold_rank([page], qs["s1"], single) == 1
    assert e7.gold_rank([Passage("x", "potato.pdf", 2, "potato", 1.0)], qs["s1"], single) is None


def test_checkpoint_resumes_and_drops_a_torn_last_line(tmp_path):
    path = tmp_path / "ck.jsonl"
    path.write_text('{"id": "a", "v": 1}\n{"id": "b", "v"')
    ck = e7.Checkpoint(path)
    assert list(ck.rows) == ["a"]
    ck.add({"id": "b", "v": 2})
    assert list(e7.Checkpoint(path).rows) == ["a", "b"]


def test_retrieval_stage_resumes_without_redoing_questions(root, tmp_path):
    r = retriever(root)
    spec = e7.RetrievalSpec(mode="bm25")
    qs = e7.dev_questions(root)
    work = tmp_path / "work"
    first = e7.run_retrieval("bm25", r, spec, qs[:2], work, root)
    assert [row["rank"] for row in first] == [1, 1]

    class Boom:
        crops, _index_dir = r.crops, r._index_dir

        def search(self, *a, **k):
            raise AssertionError("searched again")

    again = e7.run_retrieval("bm25", Boom(), spec, qs[:2], work, root)
    assert again == first


def test_retrieval_record_passes_check_run(root, tmp_path):
    r = retriever(root)
    qs = e7.dev_questions(root)
    specs = {"bm25": e7.RetrievalSpec(mode="bm25"), "bm25_history": e7.RetrievalSpec(mode="bm25", query_mode="history")}
    rows = {n: e7.run_retrieval(n, r, s, qs, tmp_path / "w", root) for n, s in specs.items()}
    path, comps = e7.write_retrieval_record("baseline", rows, specs, "bm25", {"device": "cuda", "gpu": "x"}, root)
    assert check_file(path, root) == []
    rec = json.loads(path.read_text())
    assert rec["subset"] == "answerable" and rec["n"] == 2 and rec["metrics"]["bm25"]["recall@5"] == 1.0
    assert set(rec["metrics"]) == {"bm25", "bm25_terms", "bm25_history", "bm25_history_terms"}
    assert [(c["a"], c["metric"]) for c in comps] == [("bm25_history", "recall@5"), ("bm25_history", "mrr")]
    assert rec["config"]["device"] == "cuda" and rec["config"]["device_info"]["gpu"] == "x"  # a plain string, as MLflow shows
    assert rec["config"]["method_version"] == e7.METHOD_VERSION


def comp(metric, diff, lo, hi):
    return {"a": "x", "b": "base", "metric": metric, "diff": diff, "ci": [lo, hi], "mcnemar_p": 0.5}


def test_retrieval_rule_v2_keeps_a_gain_on_either_metric_unless_the_other_is_worse():
    # the first pass: hybrid_rerank over dense, recall@5 +0.117 [0.0, 0.233], MRR +0.169 with an interval above zero
    assert e7.better([comp("recall@5", 0.117, 0.0, 0.233), comp("mrr", 0.169, 0.07, 0.27)])
    assert e7.better([comp("recall@5", 0.1, 0.02, 0.2), comp("mrr", 0.0, -0.1, 0.1)])  # other not worse
    assert not e7.better([comp("recall@5", 0.1, 0.02, 0.2), comp("mrr", -0.01, -0.1, 0.1)])  # other worse
    assert not e7.better([comp("recall@5", 0.05, -0.05, 0.15), comp("mrr", 0.02, -0.04, 0.08)])  # neither excludes zero
    assert not e7.better([comp("recall@5", -0.05, -0.1, -0.01), comp("mrr", 0.0, -0.1, 0.1)])
    with pytest.raises(ValueError):
        e7.better([comp("recall@5", 0.1, 0.02, 0.2)])


def test_floor_rule_v2_answers_at_least_80_percent_then_declines_most():
    answerable = [0.9, 0.85, 0.8, 0.7, 0.4]  # 80% answered needs a floor at or under 0.7
    unanswerable = [0.75, 0.6, 0.3, 0.2]
    f = e7.choose_floor(answerable, unanswerable)
    assert f["answers_when_answerable"] >= 0.8 and f["floor"] <= 0.7
    assert f["declines_when_unanswerable"] == 0.75  # 0.75 stays answered; the other three are declined
    assert "at least 80%" in f["rule_applied"]
    assert f["curve"] and all({"floor", "answers", "declines"} <= set(c) for c in f["curve"])


def test_floor_rule_falls_back_to_the_best_mean_when_no_floor_keeps_80_percent():
    # every floor above 0 loses answerable questions at once, so only floor 0 reaches 80%; make it unreachable
    out = e7.choose_floor([], [0.1, 0.9])
    assert "no floor reached" in out["rule_applied"]


def test_floor_record_and_diagnosis(root):
    qs = e7.dev_questions(root)
    rows = [{"id": q.id, "rank": None if q.id == "s2" else 2, "top_score": s, "crop": q.crop, "diagnosed": q.id == "s1",
             "nocrop_top_score": 0.95, "query": q.question}
            for q, s in zip(qs, [0.9, 0.4, 0.2, 0.3, 0.5], strict=True)]
    kept, dropped = e7.floor_groups(qs, rows)
    assert kept == [0.9, 0.4] and dropped == [0.2]
    floors = {"hybrid": e7.choose_floor(kept, dropped)}
    diag = e7.floor_diagnosis(qs, rows, 0.6)
    assert diag["answerable_under_floor"] == 1 and diag["by_category"] == {"supported": 1}
    assert diag["gold_missing_from_top10"] == 1 and diag["no_simulated_diagnosis"] == 1
    assert diag["would_pass_floor_without_plant_filter"] == 1
    path = e7.write_floor_record(qs, {"hybrid": rows}, floors, 0.3, {"device": "cuda"}, root, diagnosis={"hybrid": diag})
    assert check_file(path, root) == []
    rec = json.loads(path.read_text())
    assert rec["config"]["floor_diagnosis"]["hybrid"]["answerable_under_floor"] == 1 and rec["config"]["device"] == "cuda"


def test_checkpoints_start_fresh_when_settings_or_index_change(root, tmp_path):
    r = retriever(root)
    qs = e7.dev_questions(root)[:2]
    work = tmp_path / "w"
    e7.run_retrieval("x", r, e7.RetrievalSpec(mode="bm25"), qs, work, root)
    e7.run_retrieval("x", r, e7.RetrievalSpec(mode="bm25", query_mode="history"), qs, work, root)
    assert len(list(work.glob("x-*.jsonl"))) == 2  # different settings, different files
    (root / "knowledge" / "index" / "build-info.json").write_text(json.dumps({"built": "later"}))
    e7.run_retrieval("x", r, e7.RetrievalSpec(mode="bm25"), qs, work, root)
    assert len(list(work.glob("x-*.jsonl"))) == 3  # a rebuilt index never reuses old rows


def test_term_rank_is_a_lenient_cross_check_of_the_gold_rank(root):
    qs = {q.id: q for q in e7.dev_questions(root)}
    other_doc = Passage("Concentric rings are typical", "elsewhere.pdf", 9, "potato", 1.0)
    assert e7.term_rank([other_doc], qs["s1"]) == 1 and e7.gold_rank([other_doc], qs["s1"], set()) is None


def writer_reply(messages):
    user = messages[1]["content"]
    if "magnolia" in user:
        return "NO_SUPPORT"
    return "Rings appear on potato leaves [1]. Thrips are sampled weekly [2]."


def generate(root, tmp_path, system="abstain", reply=writer_reply):
    r = retriever(root)
    cfg = AnswerConfig(prompt_version="base+abstain", mode="bm25", rerank=False)
    qs = e7.dev_questions(root)
    found = e7.run_answer_retrieval(r, cfg, qs, tmp_path / "w", root)
    cached = e7.cached_retriever(r.crops, found)
    single = e7.single_page_sources(root / "knowledge" / "index")
    llm = FakeLLM(reply=reply, model="qwen-test")
    return qs, e7.run_generation(system, qs, cached, llm, cfg, tmp_path / "w", single, root), llm


def test_generation_replays_retrieval_and_declines_when_told(root, tmp_path):
    qs, (rows, missing), llm = generate(root, tmp_path)
    by = {r["id"]: r for r in rows}
    assert not missing and len(rows) == len(qs)
    assert by["s1"]["answered"] and by["s1"]["cited_gold"] and by["a1"]["answered"] is False
    assert by["a1"]["state"] == "no_document"


def test_an_outage_is_not_recorded_as_an_answer_and_resumes(root, tmp_path):
    from sprout.llm.client import LLMError

    def down(messages):
        raise LLMError("ollama down")

    qs, (rows, missing), _ = generate(root, tmp_path, system="base", reply=down)
    assert rows == [] or all(r["state"] != "unavailable" for r in rows)
    assert missing  # retrieved questions that needed the model are left to resume
    qs, (rows2, missing2), _ = generate(root, tmp_path, system="base")
    assert not missing2 and len(rows2) == len(qs)


def judge_reply(messages):
    return "NOT_SUPPORTED" if "weekly" in messages[-1]["content"] else "SUPPORTED"


def test_judged_rows_keep_everything_a_second_grader_needs(root, tmp_path):
    qs, (rows, _), _ = generate(root, tmp_path)
    judge = FakeLLM(reply=judge_reply, model="judge-test")
    judged = e7.run_judging("abstain", rows, {q.id: q for q in qs}, judge, tmp_path / "w")
    assert {r["id"] for r in judged} == {"s1", "s2", "w1"} or len(judged) >= 2
    one = next(r for r in judged if r["id"] == "s1")
    assert one["question"] and one["passages"][0]["n"] == 1 and one["answer"] and one["rubric"].startswith("You check")
    assert [c["claim"] for c in one["claims"]][0].startswith("Rings appear")
    assert all({"verdict", "judge_reply", "cited"} <= set(c) for c in one["claims"]) and one["judge_model"] == "judge-test"
    blind = blind_view(one)
    assert "judge_model" not in blind and all(set(c) == {"claim", "cited"} for c in blind["claims"])
    assert blind["passages"] == one["passages"] and blind["rubric"] == one["rubric"]
    full, blind_path = e7.write_judgments(tmp_path / "stem", judged)
    assert "verdict" in full.read_text() and "verdict" not in blind_path.read_text()
    assert "judge_reply" not in blind_path.read_text()


def test_judging_resumes_per_answer(root, tmp_path):
    qs, (rows, _), _ = generate(root, tmp_path)
    judge = FakeLLM(reply=judge_reply, model="judge-test")
    e7.run_judging("abstain", rows, {q.id: q for q in qs}, judge, tmp_path / "w")
    calls = len(judge.calls)
    e7.run_judging("abstain", rows, {q.id: q for q in qs}, judge, tmp_path / "w")
    assert len(judge.calls) == calls


def test_generation_record_passes_check_run_and_carries_the_judge(root, tmp_path):
    qs, (rows, _), _ = generate(root, tmp_path)
    judged = e7.run_judging("abstain", rows, {q.id: q for q in qs}, FakeLLM(reply=judge_reply, model="gemma-test"), tmp_path / "w")
    path = e7.write_generation_record(
        "prompts", None, {"abstain": rows}, {"abstain": judged}, {"writer": "qwen-test", "judge": "gemma-test"},
        {"judge": {"model": "gemma-test"}, "writer": "qwen-test"}, root)
    assert check_file(path, root) == []
    rec = json.loads(path.read_text())
    assert rec["n"] == 5 and rec["judged"]["values"]["faithfulness_abstain"] < 1.0
    assert rec["metrics"]["abstain"]["declines_when_unanswerable"] == 1.0


def test_split_claims_skips_the_notes_the_system_adds():
    text = ("This is only a possible match from the photo, not a certain one (Tomato - Early blight, 55%). "
            "Rings appear on leaves [1]. Water early in the day [2].\n\nCheck the product label and your local rules "
            "before using or disposing of any chemical.")
    claims = split_claims(text)
    assert [c.text for c in claims] == ["Rings appear on leaves.", "Water early in the day."]
    assert [c.cited for c in claims] == [[1], [2]]


def test_judge_answer_skips_answers_with_no_claims(root, tmp_path):
    from sprout.rag.answer import Answer

    assert judge_answer(FakeLLM(["SUPPORTED"]), "x", "s", "q", Answer("No supporting document.", [], False, [], "no_document")) is None


def test_answerable_subset_and_new_behaviour_metrics_recompute(root):
    from sprout.eval.records import split_info, subset_positions

    info = split_info("rag_dev", root)
    assert subset_positions("answerable", "rag_dev", info, root) == [0, 1]


def test_prompt_additions_are_kept_only_when_they_help_without_costing_answers():
    def vals(f, u, a, ans):
        return {"faithfulness": f, "declines_when_unanswerable": u, "declines_when_ambiguous": a, "answers_when_answerable": ans}

    v = {"base": vals(0.7, 0.5, 0.1, 0.9), "faithful": vals(0.8, 0.5, 0.1, 0.88),
         "abstain": vals(0.7, 0.9, 0.1, 0.70), "ask": vals(0.7, 0.5, 0.6, 0.9)}
    version, decisions, needs_combined = e7.choose_prompt(v)
    assert [d["kept"] for d in decisions] == [True, False, True]  # abstain costs too many answers
    assert version == "base+faithful+ask" and needs_combined
    ok = e7.choose_prompt({**v, "combined": vals(0.8, 0.5, 0.6, 0.87)})
    assert ok[0] == "base+faithful+ask" and not ok[2]
    worse = e7.choose_prompt({**v, "combined": vals(0.8, 0.5, 0.05, 0.87)})
    assert worse[0] == "base+ask"  # the larger single gain wins when the combination loses one


def test_pair_comparisons_give_both_deciding_metrics(root, tmp_path):
    r = retriever(root)
    qs = e7.dev_questions(root)
    a = e7.run_retrieval("a", r, e7.RetrievalSpec(mode="bm25"), qs, tmp_path / "w", root)
    b = e7.run_retrieval("b", r, e7.RetrievalSpec(mode="bm25", query_mode="history"), qs, tmp_path / "w", root)
    got = e7.pair_comparisons(a, b, root)
    assert [c["metric"] for c in got] == ["recall@5", "mrr"] and set(got[0]) >= {"diff", "ci", "mcnemar_p"}


def test_writer_calls_are_counted_and_declined_rows_skip_the_writer(root, tmp_path):
    qs, (rows, _), llm = generate(root, tmp_path)
    assert sum(r["llm_calls"] for r in rows) == len(llm.calls)
    assert all(r["llm_seconds"] >= 0 for r in rows)
    path = e7.write_generation_record("t", None, {"abstain": rows}, {"abstain": []}, {"writer": "qwen-test", "judge": "gemma-test"},
                                      {"judge": {"model": "gemma-test"}, "device": "cuda"}, root)
    usage = json.loads(path.read_text())["config"]["writer_usage"]["abstain"]
    assert usage["model_calls"] == len(llm.calls) and usage["rows"] == len(qs)
    assert usage["rows_without_model_call"] == sum(r["llm_calls"] == 0 for r in rows)


def test_prompt_gain_intervals_say_when_a_difference_is_noise():
    def judged(flags):
        return [{"id": str(i), "claims": [{"verdict": v} for v in f]} for i, f in enumerate(flags)]

    same = e7.faithfulness_gain(judged([[True, False]] * 20), judged([[True, False]] * 20), n=2000)
    assert same["diff"] == 0 and same["ci"][0] <= 0 <= same["ci"][1]
    better = e7.faithfulness_gain(judged([[True, True]] * 30), judged([[True, False]] * 30), n=2000)
    assert better["ci"][0] > 0
    qs = [Question(id=f"q{i}", category="supported", split="dev", question="q", expected="answer") for i in range(30)]
    hi = [{"id": q.id, "answered": True} for q in qs]
    lo = [{"id": q.id, "answered": i % 2 == 0} for i, q in enumerate(qs)]
    got = e7.behaviour_gain("answers_when_answerable", qs, lo, hi, n=2000)
    assert got["diff"] < 0 and got["ci"][1] < 0 and got["mcnemar_p"] < 0.01


def test_search_latency_is_timed_per_search_and_per_reranker_call_and_records_check_out(root, tmp_path):
    class Rer:
        def score(self, query, texts):
            return [0.5] * len(texts)

    timed = e7.TimedReranker(Rer())
    r = Retriever(root / "knowledge" / "index", config=RetrievalConfig(min_score=0.0), reranker=timed)
    spec = e7.RetrievalSpec(mode="bm25", rerank=True)
    answerable = [q for q in e7.dev_questions(root) if q.expected == "answer"]
    search_ms, rerank_ms = e7.time_search(r, timed, spec, answerable, root, warmup=1)
    assert len(search_ms) == len(rerank_ms) == len(answerable) and all(x >= 0 for x in search_ms + rerank_ms)
    qs = e7.dev_questions(root)
    rows = {"rerank_a": e7.run_retrieval("rerank_a", r, spec, qs, tmp_path / "w", root)}
    path, _ = e7.write_retrieval_record("latency", rows, {"rerank_a": spec}, "rerank_a", {"device": "cuda"}, root,
                                        measurements={"search_ms_a": search_ms, "rerank_ms_a": rerank_ms})
    assert check_file(path, root) == []
    assert "rerank_ms_a_p50" in json.loads(path.read_text())["measurements"]["values"]


def test_the_combined_decision_carries_the_gains_it_was_judged_on():
    def vals(f, u, ans):
        return {"faithfulness": f, "declines_when_unanswerable": u, "declines_when_ambiguous": 0.0, "answers_when_answerable": ans}

    v = {"base": vals(0.72, 0.0, 1.0), "faithful": vals(0.78, 0.0, 1.0), "abstain": vals(0.76, 0.14, 1.0),
         "combined": vals(0.816, 0.43, 1.0)}
    version, decisions, _ = e7.choose_prompt(v)
    combined = decisions[-1]
    assert version == "base+faithful+abstain" and combined["addition"] == "combined" and combined["kept"]
    assert round(combined["target_gains"]["faithful"], 3) == 0.096 and round(combined["target_gains"]["abstain"], 2) == 0.43


def test_superseded_file_names_real_records_and_never_a_record_as_its_own_replacement():
    marks = e7.superseded_records()
    runs = Path(__file__).resolve().parents[1] / "results" / "runs"
    ids = set(marks["superseded"]) | {m["by"] for m in marks["superseded"].values()} | set(marks["needs_rerun"])
    assert ids and all((runs / f"{i}.json").is_file() for i in ids)
    assert all(i != m["by"] for i, m in marks["superseded"].items())
    assert not set(marks["needs_rerun"]) & set(marks["superseded"])  # a record still waiting for its rerun is current


def test_the_blind_view_hides_the_writer_and_the_verdicts_but_rows_can_be_matched_back():
    row = {"id": "supported-001", "system": "claude_sonnet", "question": "q", "passages": [], "answer": "a", "rubric": "r",
           "judge_model": "j", "claims": [{"claim": "c", "cited": [1], "verdict": True, "judge_reply": "SUPPORTED"}]}
    blind = blind_view(row)
    assert "claude_sonnet" not in json.dumps(blind) and "system" not in blind and "judge_model" not in blind
    assert blind["claims"] == [{"claim": "c", "cited": [1]}] and blind["id"] == "supported-001"
    assert blind["item"] != blind_view({**row, "system": "served"})["item"] and blind["item"] == blind_view(row)["item"]


def test_retrieval_replay_records_every_search_answer_makes_including_the_fallback(root):
    class EmptyThenFound:
        crops = {"potato", "tomato"}

        def __init__(self):
            self.calls = []

        def search(self, query, *, crop=None, **kw):
            self.calls.append((query, crop))
            return [] if len(self.calls) == 1 else [Passage("Rings on potato", "p.pdf", 1, "potato", 0.9)]

    from sprout.rag.evalset import Question

    q = Question(id="x", category="answerable", split="dev", question="Tell me about early blight", expected="answer", crop="potato")
    live = EmptyThenFound()
    row = e7.retrieval_for_answers(live, AnswerConfig(), q, live.crops, e7.load_classes(root))
    assert [s["query"] for s in row["searches"]] == [c[0] for c in live.calls] and len(live.calls) == 2
    assert row["query"] == live.calls[1][0] and row["passages"]
    replay = e7.cached_retriever(live.crops, {"x": row})
    assert replay.search(live.calls[0][0], crop="potato") == [] and replay.search(live.calls[1][0], crop="potato")
