"""T7.4: what the disease-name fallback in `answer()` does to retrieval. Retrieval only, on the CPU: no writer, no Claude.

    .venv/Scripts/python -m sprout.experiments.e8_fallback --index <index folder> --out results/e8/fallback_check.json \
        --followups results/e8/generic_followups.jsonl

Two parts. (1) On rag_dev with E7's SIMULATED identifications (the diagnosis is read from the question text, not from a
photo), decline rates by category without and with the fallback, and every question whose decline changes. A question
"declines" here when the search returns no passage; the writer's own NO_SUPPORT declines are not in this.
(2) A small set of generic follow-ups ("How do I treat it?") for each disease class among the 29, with the diagnosis
given as `answered` (also simulated): how many get a passage about the right disease among the served k.
rag_test is never read.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from sprout.atomic import atomic_write_text
from sprout.rag.answer import AnswerConfig, bounded_history, retrieve
from sprout.rag.query import disease_of_label
from sprout.serve.types import Candidate, Identification

WORDINGS = (
    "How do I treat it?",
    "What should I do about this?",
    "Is it contagious?",
    "What causes this?",
    "How can I stop it coming back?",
)
# "Right disease" is fixed here, before any result: a served passage counts when its text, lowercased with hyphens read
# as spaces, contains the diagnosed disease's name, or one of the alternate names below for the labels whose class name
# lists two names or spells one differently from the documents.
ALIASES = {
    "cercospora leaf spot gray leaf spot": ("cercospora leaf spot", "gray leaf spot", "grey leaf spot"),
    "haunglongbing (citrus greening)": ("huanglongbing", "haunglongbing", "citrus greening"),
    "tomato yellow leaf curl virus": ("tomato yellow leaf curl", "yellow leaf curl"),
}
NOTE = (
    "The identification is SIMULATED in both parts: in part 1 E7 reads the diagnosis from the question text, in part 2 "
    "the diagnosis is the label the follow-up was written for, given as 'answered' at confidence 0.9. No photo and no "
    "classifier ran. Retrieval only (CPU, no writer, no Claude); rag_test was not read."
)


def names_of(label: str) -> tuple[str, ...]:
    disease = disease_of_label(label)
    return (disease, *ALIASES.get(disease, ())) if disease else ()


def right_disease(text: str, label: str) -> bool:
    flat = re.sub(r"\s+", " ", text.lower().replace("-", " "))
    return any(n in flat for n in names_of(label))


def followups(classes: list[str]) -> list[dict]:
    """Every class with a disease name, each asked every wording. Deterministic."""
    return [{"id": f"gf-{i:03d}-{j}", "label": label, "question": w}
            for i, label in enumerate(c for c in classes if disease_of_label(c)) for j, w in enumerate(WORDINGS)]


def _identified(label: str) -> Identification:
    return Identification("answered", Candidate(label, 0.9))


def generic_followup_report(retriever, rows: list[dict], cfg: AnswerConfig) -> dict:
    """For each follow-up: what the question alone finds, and what `retrieve` finds with the fallback."""
    from sprout.experiments.e7_rag import RecordingRetriever

    out = []
    for r in rows:
        rec = RecordingRetriever(retriever)
        found, query, crop = retrieve(r["question"], identification=_identified(r["label"]), history=[], retriever=rec,
                                      cfg=cfg)
        first = [p for p in rec.searches[0]["passages"]]
        out.append({**r, "crop": crop, "query_used": query, "searches": len(rec.searches),
                    "before_passages": len(first), "before_right": any(right_disease(p["text"], r["label"]) for p in first),
                    "after_passages": len(found), "after_right": any(right_disease(p.text, r["label"]) for p in found)})
    n = len(out)
    summary = {"follow_ups": n, "k": cfg.k,
               "found_any_before": sum(o["before_passages"] > 0 for o in out),
               "found_any_after": sum(o["after_passages"] > 0 for o in out),
               "right_disease_before": sum(o["before_right"] for o in out),
               "right_disease_after": sum(o["after_right"] for o in out)}
    by_label: dict[str, dict] = {}
    for o in out:
        b = by_label.setdefault(o["label"], {"follow_ups": 0, "right_disease_before": 0, "right_disease_after": 0})
        b["follow_ups"] += 1
        b["right_disease_before"] += o["before_right"]
        b["right_disease_after"] += o["after_right"]
    return {"summary": summary, "by_label": by_label, "rows": out}


def dev_report(retriever, questions, classes: list[str], cfg: AnswerConfig) -> dict:
    """Decline rates on the dev questions before and after the fallback, with E7's simulated identification."""
    from sprout.eval.check_run import Scored, recompute_metric
    from sprout.experiments.e7_rag import (
        BEHAVIOUR_METRICS,
        RecordingRetriever,
        simulate_identification,
    )

    before, after, detail = [], [], []
    for q in questions:
        crop, ident = simulate_identification(q, retriever.crops, classes)
        rec = RecordingRetriever(retriever)
        found, query, _ = retrieve(q.question, identification=ident, history=bounded_history(q.history, cfg),
                                   retriever=rec, cfg=cfg, crop=crop)
        first = rec.searches[0]["passages"]
        before.append(not first)
        after.append(not found)
        if bool(first) != bool(found):
            detail.append({"id": q.id, "category": q.category, "expected": q.expected, "question": q.question,
                           "declined_before": not first, "declined_after": not found, "fallback_query": query,
                           "top_score_after": found[0].score if found else None})
    ids = [q.id for q in questions]
    expected = [q.expected for q in questions]
    rates = {name: {m: recompute_metric(m, Scored(groups=ids, answered=~np.array(flags), expected=expected))
                    for m in BEHAVIOUR_METRICS} for name, flags in (("question_only", before), ("with_fallback", after))}
    return {"split": "rag_dev", "questions": len(ids), "categories": dict(Counter(q.category for q in questions)),
            "decline_means": "the search returned no passage (the writer's own declines are not in this)",
            "rates": rates, "questions_whose_decline_changes": detail}


def main(argv: list[str] | None = None) -> int:
    from sprout.eval.records import ROOT, head_commit, split_info
    from sprout.experiments.e7_rag import dev_questions, load_classes
    from sprout.rag import serving
    from sprout.rag.retrieve import Retriever

    ap = argparse.ArgumentParser(prog="e8_fallback")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--followups", required=True, help="where the generic follow-up set is written (committed)")
    ap.add_argument("--classes-config", default="configs/e15.json", help="its `classes` are the 29")
    a = ap.parse_args(argv)
    classes29 = json.loads((ROOT / a.classes_config).read_text(encoding="utf-8"))["classes"]
    rows = followups(classes29)
    atomic_write_text(Path(a.followups), "".join(json.dumps(r) + "\n" for r in rows))
    retriever = Retriever(Path(a.index), config=serving.retrieval_config())
    cfg = AnswerConfig()
    report = {
        "note": NOTE, "git_commit": head_commit(ROOT), "dataset_hash": split_info("rag_dev", ROOT).hash,
        "right_disease_rule": "a served passage counts when its text names the diagnosed disease (aliases: "
                              + json.dumps(ALIASES) + "); fixed before the run",
        "served": {"mode": cfg.mode, "rerank": cfg.rerank, "k": cfg.k, "floor": serving.FLOORS["hybrid+rerank"]},
        "rag_dev": dev_report(retriever, dev_questions(ROOT), load_classes(ROOT), cfg),
        "generic_follow_ups": generic_followup_report(retriever, rows, cfg),
    }
    atomic_write_text(Path(a.out), json.dumps(report, indent=1) + "\n")
    print(a.out)
    print(json.dumps({"rag_dev": report["rag_dev"]["rates"], "generic": report["generic_follow_ups"]["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
