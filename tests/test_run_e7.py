"""The E7 stage script end to end with fakes: no GPU, no Ollama, no Claude, a throwaway git repo."""
import contextlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_e7 import CHUNKS, QUESTIONS, git, judge_reply, root  # noqa: F401

from sprout.experiments import e7_rag as e7
from sprout.llm.client import FakeLLM
from sprout.rag.retrieve import RetrievalConfig, Retriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_e7 as tool  # noqa: E402


class FakeModels:
    embedder = reranker = None

    def info(self):
        return {"device": "cuda", "gpu": "test gpu", "torch": "x", "peak_gpu_mib": 1}


class Ollama(FakeLLM):
    def unload(self):
        self.unloaded = True


@pytest.fixture
def tool_env(root, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(tool, "ROOT", root)
    monkeypatch.setattr(tool, "WORK", tmp_path / "work")
    monkeypatch.setattr(tool, "CHOSEN", root / "results" / "e7" / "chosen.json")
    monkeypatch.setattr(tool, "DRAFTS", root / "results" / "e7" / "decisions")
    monkeypatch.setattr(tool, "knowledge", lambda: root / "knowledge")
    monkeypatch.setattr(tool, "gpu", contextlib.nullcontext)
    monkeypatch.setattr(tool, "Models", FakeModels)
    monkeypatch.setattr(tool, "make_retriever", lambda spec, models, floors=None: Retriever(
        root / "knowledge" / spec.index, config=RetrievalConfig(min_score=0.0, floors=floors or {})))
    monkeypatch.setattr(tool, "BASELINES", {"dense": e7.RetrievalSpec(mode="bm25"), "hybrid": e7.RetrievalSpec(mode="bm25"),
                                            "hybrid_rerank": e7.RetrievalSpec(mode="bm25", query_mode="history")})
    monkeypatch.setattr(tool, "FLOOR_MODES", {"hybrid": ("bm25", False)})
    monkeypatch.setattr(tool, "load_sources", lambda m: {})
    return root


def commit_all(repo):
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "stage")


def test_dirty_tree_stops_a_stage_before_any_work(tool_env):
    (tool_env / "stray.txt").write_text("x")
    with pytest.raises(SystemExit, match="uncommitted"):
        tool.cmd_baseline(None)
    assert not (tool.WORK / "baseline").exists()


def test_stages_run_in_order_and_each_record_checks_out(tool_env, capsys):
    from sprout.eval.check_run import check_file

    tool.cmd_baseline(None)
    chosen = e7.Chosen.load(tool.CHOSEN)
    assert chosen.decisions[0]["stage"] == "baseline" and (tool.DRAFTS / "baseline.md").is_file()
    commit_all(tool_env)
    tool.cmd_queries(None)  # candidates are scored against the baseline's choice
    commit_all(tool_env)
    tool.cmd_floors(None)
    chosen = e7.Chosen.load(tool.CHOSEN)
    assert "hybrid" in chosen.floors
    for rec in sorted((tool_env / "results" / "runs").glob("E7-*.json")):
        if rec.name.count(".") == 1:
            assert check_file(rec, tool_env) == [], rec.name
    assert "recall@5" in capsys.readouterr().out


def test_generation_judging_and_record_with_the_graded_count_guard(tool_env, monkeypatch, capsys):
    tool.cmd_baseline(None)
    commit_all(tool_env)
    tool.cmd_floors(None)
    commit_all(tool_env)
    tool.cmd_retrieve_answers(None)
    writers, judges = [], []

    def writer_reply(messages):
        return "NO_SUPPORT" if "magnolia" in messages[1]["content"] else "Rings appear on potato leaves [1]."

    def make(model, **kw):
        llm = Ollama(reply=writer_reply if model == tool.WRITER else judge_reply, model=model)
        (writers if model == tool.WRITER else judges).append(llm)
        return llm

    monkeypatch.setattr(tool, "OllamaClient", make)
    args = type("A", (), {"variants": ["base", "faithful"]})()
    tool.cmd_generate(args)
    assert all(w.unloaded for w in writers)  # the model is unloaded afterwards
    tool.cmd_judge(args)
    assert all(j.unloaded for j in judges)
    commit_all(tool_env)
    with pytest.raises(SystemExit) as stop:
        tool.cmd_prompts_record(None)  # five questions cannot give 200 graded answers
    assert stop.value.code == tool.EXIT_TOO_FEW_GRADED
    runs = tool_env / "results" / "runs"
    blind = next(runs.glob("E7-prompts-*.blind.jsonl")).read_text()
    full = next(runs.glob("E7-prompts-*.judgments.jsonl")).read_text()
    assert "verdict" in full and "verdict" not in blind and "passages" in blind
    record = json.loads(next(p for p in runs.glob("E7-*.json") if "prompts" in p.read_text()).read_text())
    assert record["config"]["judge"]["model"] == tool.JUDGE and record["models"]["writer"] == tool.WRITER
    assert "graded answers across variants" in capsys.readouterr().out


def test_unreadable_rag_test_is_never_touched(tool_env):
    assert all(q.split == "dev" for q in e7.dev_questions(tool_env))
    assert any(q["split"] == "test" for q in QUESTIONS) and CHUNKS
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0
