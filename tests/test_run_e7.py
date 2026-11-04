"""The E7 stage script end to end with fakes: no GPU, no Ollama, no Claude, a throwaway git repo."""
import contextlib
import json
import sys
from pathlib import Path

import pytest
from test_e7 import judge_reply, root  # noqa: F401

from sprout.eval.check_run import check_file
from sprout.experiments import e7_rag as e7
from sprout.llm.client import FakeLLM
from sprout.rag.retrieve import RetrievalConfig, Retriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_e7 as tool  # noqa: E402
from test_e7 import git  # noqa: E402


class FakeModels:
    embedder = reranker = None

    def info(self):
        return {"device": "cuda", "gpu": "test gpu", "torch": "x", "peak_gpu_mib": 1}


class Ollama(FakeLLM):
    def __init__(self, *a, vram_share=1.0, **k):
        super().__init__(*a, **k)
        self.vram_share = vram_share

    def placement(self):
        return {"size": 100, "size_vram": int(100 * self.vram_share)}

    def unload(self):
        self.unloaded = True


@pytest.fixture
def tool_env(root, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(tool, "ROOT", root)
    monkeypatch.setattr(tool, "WORK", tmp_path / "work")
    monkeypatch.setattr(tool, "CHOSEN", root / "results" / "e7" / "v2" / "chosen.json")
    monkeypatch.setattr(tool, "DRAFTS", root / "results" / "e7" / "v2" / "decisions")
    monkeypatch.setattr(tool, "knowledge", lambda: root / "knowledge")
    monkeypatch.setattr(tool, "gpu", contextlib.nullcontext)
    monkeypatch.setattr(tool, "Models", FakeModels)
    monkeypatch.setattr(tool, "make_retriever", lambda spec, models, floors=None, reranker=None: Retriever(
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


def test_version_2_starts_fresh_beside_the_first_pass(tool_env):
    assert tool.CHOSEN.parts[-2:] == ("v2", "chosen.json") and tool.WORK.name == "work"
    assert "v2" in Path(tool.__file__).read_text(encoding="utf-8").split("WORK = ", 1)[1].split("\n", 1)[0]


def test_stages_run_in_order_and_each_record_checks_out(tool_env, capsys):
    tool.cmd_baseline(None)
    chosen = e7.Chosen.load(tool.CHOSEN)
    assert chosen.decisions[0]["stage"] == "baseline" and (tool.DRAFTS / "baseline.md").is_file()
    commit_all(tool_env)
    tool.cmd_queries(None)  # candidates are scored against the baseline's choice
    commit_all(tool_env)
    tool.cmd_floors(None)
    chosen = e7.Chosen.load(tool.CHOSEN)
    assert "hybrid" in chosen.floors
    records = [p for p in sorted((tool_env / "results" / "runs").glob("E7-*.json")) if p.name.count(".") == 1]
    assert len(records) == 3
    for rec in records:
        assert check_file(rec, tool_env) == [], rec.name
    out = capsys.readouterr().out
    assert "recall@5" in out and "by category" in out and "Why answerable questions fall under the floor" in out


def test_generation_judging_and_record_with_the_graded_count_guard(tool_env, monkeypatch, capsys):
    tool.cmd_baseline(None)
    commit_all(tool_env)
    tool.cmd_floors(None)
    commit_all(tool_env)
    tool.cmd_retrieve_answers(None)
    clients = {}

    def writer_reply(messages):
        return "NO_SUPPORT" if "magnolia" in messages[-1]["content"] else "Rings appear on potato leaves [1]."

    def make(model, **kw):
        llm = Ollama(reply=writer_reply if model == tool.WRITER else judge_reply, model=model)
        clients.setdefault(model, []).append(llm)
        return llm

    monkeypatch.setattr(tool, "OllamaClient", make)
    args = type("A", (), {"variants": ["base", "faithful", "served"]})()
    tool.cmd_generate(args)
    assert all(w.unloaded for w in clients[tool.WRITER])  # the model is unloaded afterwards
    tool.cmd_judge(args)
    assert all(j.unloaded for j in clients[tool.JUDGE])
    commit_all(tool_env)
    with pytest.raises(SystemExit) as stop:
        tool.cmd_prompts_record(None)  # a handful of questions cannot give 200 graded answers
    assert stop.value.code == tool.EXIT_TOO_FEW_GRADED
    runs = tool_env / "results" / "runs"
    blind = next(runs.glob("E7-prompts-v2-*.blind.jsonl")).read_text()
    full = next(runs.glob("E7-prompts-v2-*.judgments.jsonl")).read_text()
    assert "verdict" in full and "verdict" not in blind and "passages" in blind
    record = json.loads(next(p for p in runs.glob("E7-*.json") if "prompts" in p.read_text()).read_text())
    assert record["config"]["judge"]["model"] == tool.JUDGE and record["models"]["writer"] == tool.WRITER
    assert record["config"]["device"] == "writer cuda; judge cuda"  # a plain string, as MLflow shows
    usage = record["config"]["writer_usage"]
    assert usage["base"]["model_calls"] > 0 and usage["served"]["rows"] == 5
    assert "graded answers across variants" in capsys.readouterr().out
    assert check_file(next(p for p in runs.glob("E7-*.json") if "prompts" in p.read_text()), tool_env) == []


def test_a_model_partly_on_the_cpu_is_recorded_as_such_and_one_wholly_on_the_cpu_stops(tool_env, monkeypatch):
    monkeypatch.setattr(tool, "OllamaClient", lambda model, **kw: Ollama(["ok"], vram_share=0.4, model=model))
    client = tool.OllamaClient(tool.WRITER)
    with tool.ollama_stage(client, "writer"):
        pass
    assert tool.ollama_device("writer")[0].startswith("cuda+cpu (40%")
    cpu_only = Ollama(["ok"], vram_share=0.0, model="m")
    with pytest.raises(RuntimeError, match="not on the GPU"), tool.ollama_stage(cpu_only, "writer"):
        pass
    assert cpu_only.unloaded


def writer_reply(messages):
    return "NO_SUPPORT" if "magnolia" in messages[-1]["content"] else "Rings appear on potato leaves [1]."


def prepare(tool_env, monkeypatch, variants, clients=None):
    """Baseline, floors and cached passages done and committed; the given variants generated and judged."""
    clients = {} if clients is None else clients

    def make(model, **kw):
        llm = Ollama(reply=writer_reply if model == tool.WRITER else judge_reply, model=model)
        clients.setdefault(model, []).append(llm)
        return llm

    monkeypatch.setattr(tool, "OllamaClient", make)
    if not tool.CHOSEN.is_file():
        tool.cmd_baseline(None)
        commit_all(tool_env)
        tool.cmd_floors(None)
        commit_all(tool_env)
        tool.cmd_retrieve_answers(None)
    args = type("A", (), {"variants": variants})()
    tool.cmd_generate(args)
    tool.cmd_judge(args)
    commit_all(tool_env)
    return clients


class Sonnet(FakeLLM):
    made = 0

    def complete(self, messages, *, max_tokens, timeout_s):
        type(self).made += 1
        return super().complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)


def test_claude_calls_are_counted_from_saved_rows_when_the_stage_is_resumed(tool_env, monkeypatch):
    prepare(tool_env, monkeypatch, ["served"])
    Sonnet.made = 0
    monkeypatch.setattr(tool, "ClaudeCodeClient", lambda model, counter_path: Sonnet(reply=writer_reply, model=model))
    real_stage = tool.ollama_stage

    def busy(*a, **k):
        raise RuntimeError("taskctl gpu-acquire T6.4 failed (exit 1)")  # the GPU is busy after the answers are written

    monkeypatch.setattr(tool, "ollama_stage", busy)
    with pytest.raises(RuntimeError, match="gpu-acquire"):
        tool.cmd_claude(type("A", (), {"supersedes": None})())
    first_run_calls = Sonnet.made
    assert first_run_calls > 0
    monkeypatch.setattr(tool, "ollama_stage", real_stage)
    commit_all(tool_env)
    tool.cmd_claude(type("A", (), {"supersedes": "E7-earlier"})())  # resumes: Claude answers are all saved already
    assert Sonnet.made == first_run_calls  # no new calls on the resume
    runs = tool_env / "results" / "runs"
    records = [json.loads(p.read_text()) for p in runs.glob("E7-*.json") if "claude_writer" in p.read_text()]
    assert len(records) == 1 and records[0]["claude_calls"] == first_run_calls == records[0]["config"]["claude_calls"]
    assert records[0]["config"]["supersedes"] == "E7-earlier"
    assert check_file(next(p for p in runs.glob("E7-*.json") if "claude_writer" in p.read_text()), tool_env) == []


def test_prompts_record_checks_the_tree_before_writing_and_never_rewrites_a_committed_file(tool_env, monkeypatch):
    prepare(tool_env, monkeypatch, ["base", "faithful"])
    runs = tool_env / "results" / "runs"
    (tool_env / "stray.txt").write_text("x")
    with pytest.raises(SystemExit, match="uncommitted"):
        tool.cmd_prompts_record(None)
    assert not list(runs.glob("E7-prompts-v2-*"))  # nothing was written before the check
    (tool_env / "stray.txt").unlink()
    with pytest.raises(SystemExit):  # too few graded answers, but the record and files are written
        tool.cmd_prompts_record(None)
    first = {p.name: p.read_bytes() for p in runs.glob("E7-prompts-v2-*")}
    assert len(first) == 2
    commit_all(tool_env)
    args = type("A", (), {"variants": ["abstain"]})()
    tool.cmd_generate(args)
    tool.cmd_judge(args)
    commit_all(tool_env)
    with pytest.raises(SystemExit):
        tool.cmd_prompts_record(None)  # a second run must not trip over its own files
    after = {p.name: p.read_bytes() for p in runs.glob("E7-prompts-v2-*")}
    assert all(after[name] == data for name, data in first.items())  # committed files are untouched
    assert len(after) == 4  # the new variant set got files of its own


def test_rows_made_by_older_answer_code_are_not_reused(tool_env, monkeypatch):
    prepare(tool_env, monkeypatch, ["base"])
    chosen = e7.Chosen.load(tool.CHOSEN)
    old = tool.gen_key(chosen, "base")
    assert tool.present(chosen, "base")
    monkeypatch.setattr(e7, "ANSWER_VERSION", e7.ANSWER_VERSION + 1)
    assert tool.gen_key(chosen, "base") != old and not tool.present(chosen, "base")
