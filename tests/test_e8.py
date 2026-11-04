"""E8, end-to-end performance: fakes and a tiny exported model only. No GPU, no Ollama, no network, no Claude."""
import pytest

from sprout.llm.client import FakeLLM
from sprout.rag.answer import LABEL_NOTE, answer
from sprout.rag.retrieve import Passage
from sprout.serve.types import Candidate, Identification

P1 = Passage("Rings appear on tomato leaves.", "tomato.pdf", 3, "tomato", 0.9)
P2 = Passage("Remove infected leaves.", "tomato.pdf", 4, "tomato", 0.8)
P3 = Passage("Water at the base.", "tomato.pdf", 5, "tomato", 0.7)
SURE = Identification("answered", Candidate("Tomato___Early_blight", 0.95))


class FakeRetriever:
    crops = {"tomato", "potato"}

    def __init__(self, passages):
        self.passages, self.searches = passages, []

    def search(self, query, **kw):
        self.searches.append(kw)
        return self.passages


def ask(reply, passages=(P1, P2, P3), **kw):
    return answer("What are the signs?", identification=kw.pop("ident", SURE), history=[],
                  retriever=FakeRetriever(list(passages)), llm=FakeLLM([reply]), **kw)


# ---------------------------------------------------------------- [1, 2] citations

@pytest.mark.parametrize("reply", ["Rings appear on leaves [1, 2].", "Rings appear on leaves [1,2].",
                                   "Rings appear on leaves [1; 2]."])
def test_a_list_of_passage_numbers_in_one_bracket_cites_every_passage(reply):
    out = ask(reply)
    assert [(c.source, c.page) for c in out.citations] == [("tomato.pdf", 3), ("tomato.pdf", 4)]
    assert out.grounded and out.state == "answered" and not out.text.startswith("No supporting document")
    assert "[1][2]" in out.text and out.text.endswith(LABEL_NOTE)


def test_numbers_that_were_not_shown_are_dropped_from_a_list_and_a_prose_range_is_left_alone():
    out = ask("Rings appear [1, 9]. Keep it between [10, 20] degrees. Also [3, 4].")
    assert [c.page for c in out.citations] == [3, 5]
    assert "[1]" in out.text and "[9]" not in out.text and "[10, 20]" in out.text and "[3]" in out.text
    assert "[4]" not in out.text


def test_single_brackets_behave_as_before():
    out = ask("Rings appear on leaves [1]. Also see [7] and [2].")
    assert [c.page for c in out.citations] == [3, 4] and "[7]" not in out.text
