"""The setup E7 chose for serving, in one place. `answer()` and the app read it; nothing else repeats these values.

Retrieval: hybrid search with reranking over the main index, with a score floor per mode. Prompt: the base prompt plus
the `faithful` and `abstain` rules. The floors and the prompt were picked on rag_dev (method v2, see docs/DECISIONS.md),
so they are optimistic. Both prompt rules are distinguishable gains over the base prompt on rag_dev
(record E7-20261104T040010301317Z).
Reranking costs about 30 ms on the GPU and about 600 ms on the CPU per search (record E7-20261103T235630916335Z).
"""

from __future__ import annotations

MODE = "hybrid"
RERANK = True
K = 5
QUERY_MODE = "question"
PROMPT_VERSION = "base+faithful+abstain"
FLOORS = {"dense": 0.732844, "hybrid": 0.806462, "hybrid+rerank": 0.73875}


def retrieval_config():
    """The Retriever settings to serve with (score floors per mode)."""
    from sprout.rag.retrieve import RetrievalConfig

    return RetrievalConfig(floors=dict(FLOORS))
