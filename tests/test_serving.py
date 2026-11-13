"""The setup E7 chose is what answer() and the app use by default, from one config."""
import json
from pathlib import Path

from sprout.llm.client import FakeLLM
from sprout.rag import serving
from sprout.rag.answer import AnswerConfig, answer, load_prompt
from sprout.rag.retrieve import Passage
from sprout.serve.types import Candidate, Identification

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_the_chosen_setup():
    cfg = AnswerConfig()
    assert (cfg.mode, cfg.rerank, cfg.k, cfg.query_mode) == ("hybrid", True, 5, "question")
    assert cfg.prompt_version == "base+faithful+abstain" == serving.PROMPT_VERSION
    floors = serving.retrieval_config()
    assert floors.floor_for("hybrid", True) == 0.73875
    assert floors.floor_for("dense", False) == 0.732844 and floors.floor_for("hybrid", False) == 0.806462
    assert floors.floors is not serving.FLOORS  # a copy: callers cannot change the shared values


class Spy:
    crops = {"tomato"}

    def __init__(self):
        self.kw = None

    def search(self, query, **kw):
        self.kw = kw
        return [Passage("Early blight makes rings on tomato leaves.", "t.pdf", 1, "tomato", 0.9)]


def test_answer_with_no_config_uses_hybrid_rerank_and_the_chosen_prompt():
    spy, llm = Spy(), FakeLLM(["Rings appear on leaves [1]."])
    ident = Identification("answered", Candidate("Tomato___Early_blight", 0.95))
    answer("What are the signs?", identification=ident, history=[], retriever=spy, llm=llm)
    assert spy.kw["mode"] == "hybrid" and spy.kw["rerank"] is True and spy.kw["k"] == 5
    system = llm.calls[0][0]["content"]
    assert system == load_prompt("base+faithful+abstain")
    assert "NO_SUPPORT" in system and "end with the number of the passage" in system and "NEED:" not in system


def test_the_config_matches_the_e7_choice_on_record():
    chosen = ROOT / "results" / "e7" / "v2" / "chosen.json"
    data = json.loads(chosen.read_text(encoding="utf-8"))  # tracked, so a missing file fails rather than passes
    assert data["retrieval"]["mode"] == serving.MODE and data["retrieval"]["rerank"] is serving.RERANK
    assert data["retrieval"]["index"] == "index" and data["retrieval"]["query_mode"] == serving.QUERY_MODE
    assert data["floors"] == serving.FLOORS
    prompts = [d for d in data["decisions"] if d["stage"] == "prompts"]
    assert prompts[-1]["prompt"] == serving.PROMPT_VERSION


def test_the_app_builds_its_retriever_with_the_serving_config(monkeypatch, tmp_path):
    import sprout.rag.retrieve as retrieve

    seen = {}

    class SpyRetriever:
        crops = set()

        def __init__(self, index, config=None, **kw):
            seen["config"] = config

        def warm(self, **kw):
            pass

    monkeypatch.setattr(retrieve, "Retriever", SpyRetriever)
    monkeypatch.setenv("SPROUT_NO_AUTOBUILD", "1")
    monkeypatch.syspath_prepend(str(ROOT / "deploy"))
    import wsgi

    index = tmp_path / "idx"
    index.mkdir()
    (index / "chunks.jsonl").write_text("")
    wsgi.build({"SPROUT_SECRET_KEY": "x" * 32, "SPROUT_DATA_DIR": str(tmp_path / "d"), "SPROUT_INDEX_DIR": str(index)})
    assert seen["config"].floors == serving.FLOORS
