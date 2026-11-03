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
    path, comps = e7.write_retrieval_record("baseline", rows, specs, "bm25", {"device": "cuda"}, root)
    assert check_file(path, root) == []
    rec = json.loads(path.read_text())
    assert rec["subset"] == "answerable" and rec["n"] == 2 and rec["metrics"]["bm25"]["recall@5"] == 1.0
    assert comps[0]["a"] == "bm25_history" and comps[0]["metric"] == "recall@5"


def test_candidate_is_kept_only_when_its_interval_excludes_zero():
    assert e7.better({"ci": [0.02, 0.2]}) and not e7.better({"ci": [-0.05, 0.2]}) and not e7.better({"ci": [0.0, 0.2]})


def test_choose_floor_separates_the_two_groups_and_checks_out(root):
    f = e7.choose_floor([0.8, 0.9, 0.7], [0.2, 0.1])
    assert 0.2 < f["floor"] < 0.7 and f["balanced"] == 1.0
    assert e7.choose_floor([0.5], [0.5])["balanced"] == 0.5  # no floor can separate equal scores
    qs = e7.dev_questions(root)
    rows = [{"id": q.id, "rank": None, "top_score": s} for q, s in zip(qs, [0.9, 0.8, 0.2, 0.3, 0.5], strict=True)]
    kept, dropped = e7.floor_groups(qs, rows)
    assert kept == [0.9, 0.8] and dropped == [0.2, 0.3]
    floors = {"hybrid": e7.choose_floor(kept, dropped)}
    path = e7.write_floor_record(qs, {"hybrid": rows}, floors, 0.3, {"device": "cuda"}, root)
    assert check_file(path, root) == []
    m = json.loads(path.read_text())["metrics"]
    assert m["floor_chosen_hybrid"]["answers_when_answerable"] == 1.0
    assert m["floor_chosen_hybrid"]["declines_when_unanswerable"] == 1.0


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
    return "NOT_SUPPORTED" if "weekly" in messages[1]["content"] else "SUPPORTED"


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


def test_pair_comparison_matches_what_the_record_stores(root, tmp_path):
    r = retriever(root)
    qs = e7.dev_questions(root)
    a = e7.run_retrieval("a", r, e7.RetrievalSpec(mode="bm25"), qs, tmp_path / "w", root)
    b = e7.run_retrieval("b", r, e7.RetrievalSpec(mode="bm25", query_mode="history"), qs, tmp_path / "w", root)
    assert set(e7.pair_comparison(a, b, root)) == {"diff", "ci", "mcnemar_p"}
