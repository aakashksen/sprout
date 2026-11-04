"""Per-claim faithfulness judging by a local model that is not the writer, and the files a second grader needs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from sprout.llm.client import LLMClient, LLMError
from sprout.rag.answer import (
    LABEL_NOTE,
    LEGACY_LABEL_NOTE,
    PROMPT_DIR,
    UNGROUNDED,
    Answer,
)

RUBRIC_FILE = "judge.txt"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\d])")
_MARK = re.compile(r"\s*\[(\d+)\]")
_SKIP_PREFIXES = ("This is only a possible match", UNGROUNDED.strip(), LABEL_NOTE, LEGACY_LABEL_NOTE)
_VERDICT = re.compile(r"\W*SUPPORTED\b", re.I)
VERDICT_KEYS = ("verdict", "judge_reply")  # what a second grader is not shown


@dataclass
class Claim:
    text: str
    cited: list[int]


def rubric() -> str:
    return (PROMPT_DIR / RUBRIC_FILE).read_text(encoding="utf-8").strip()


def split_claims(text: str) -> list[Claim]:
    """The factual sentences of an answer, without the notes the system adds itself (uncertainty, label, no-source)."""
    claims = []
    for block in text.split("\n\n"):
        for sentence in _SENTENCE.split(block.strip()):
            sentence = sentence.strip()
            if not sentence or sentence.startswith(_SKIP_PREFIXES) or sentence.endswith("?"):
                continue
            body = _MARK.sub("", sentence).strip()
            if len(body) >= 12:
                claims.append(Claim(body, [int(n) for n in _MARK.findall(sentence)]))
    return claims


def judge_claim(llm: LLMClient, claim: Claim, passages: list, timeout_s: float = 120.0) -> tuple[bool, str]:
    """(supported, raw reply). The claim is checked against the passages it cites, or all shown when it cites none.
    A reply that is neither SUPPORTED nor NOT_SUPPORTED counts as not supported."""
    chosen = [passages[n - 1] for n in claim.cited if 1 <= n <= len(passages)] or passages
    shown = "\n".join(f"[{i}] {p.text}" for i, p in enumerate(chosen, 1))
    messages = [
        {"role": "system", "content": rubric()},
        {"role": "user", "content": f"Passages:\n{shown}\n\nClaim: {claim.text}"},
    ]
    reply = llm.complete(messages, max_tokens=8, timeout_s=timeout_s).text.strip()
    return bool(_VERDICT.match(reply)), reply


def judge_answer(llm: LLMClient, qid: str, system: str, question: str, ans: Answer) -> dict | None:
    """The full judged row for one answer, or None when it has no claims to grade (a decline or a question back).
    Raises LLMError if the judge is unavailable, so the stage retries instead of recording a guess."""
    if ans.state not in ("answered", "ungrounded"):
        return None
    claims = split_claims(ans.text)
    if not claims:
        return None
    graded = []
    for c in claims:
        ok, reply = judge_claim(llm, c, ans.passages)
        graded.append({"claim": c.text, "cited": c.cited, "verdict": ok, "judge_reply": reply})
    return {
        "id": qid, "system": system, "question": question,
        "passages": [{"n": i, "source": p.source, "page": p.page, "text": p.text} for i, p in enumerate(ans.passages, 1)],
        "answer": ans.text, "claims": graded, "rubric": rubric(), "judge_model": llm.model,
    }


def blind_view(row: dict) -> dict:
    """The same row with the judge's verdicts and replies, the judge's name and the writer's system name left out, for
    another grader. `item` is a short hash of the system and question, so rows can be matched back without the name."""
    item = hashlib.sha1(f"{row['system']}|{row['id']}".encode()).hexdigest()[:10]
    return {
        **{k: v for k, v in row.items() if k not in ("judge_model", "claims", "system")},
        "item": item,
        "claims": [{k: v for k, v in c.items() if k not in VERDICT_KEYS} for c in row["claims"]],
    }


__all__ = ["Claim", "LLMError", "blind_view", "judge_answer", "judge_claim", "rubric", "split_claims"]
