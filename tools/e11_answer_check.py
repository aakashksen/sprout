"""Checks the saved E11 answers against the answerer reading the list rather than the photo, and re-runs the flow
with the first-attempt wording for sheets s001-s070. Reads saved data only; writes docs/e11-answer-check.md and
changes no record or answer.

    PYTHONPATH=src .venv/Scripts/python tools/e11_answer_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from sprout.atomic import atomic_write_text
from sprout.experiments import e11_followup as E
from sprout.serve import followup as F

ROOT = E.ROOT
ATTEMPT1 = ROOT / "datasets" / "followup" / "attempt1"
REPORT = ROOT / "docs" / "e11-answer-check.md"
SHEETS = range(1, 71)


def yes_rates(answers: list[dict], plan: list[dict], truth: dict[str, str], bank: list[F.Question]) -> dict:
    """Yes and unknown rates by kind of question: a decoy, a real candidate about a class that is not the photo's
    class, a real candidate about the photo's class (the true class). The class label is used here only to describe
    the answers after they were saved."""
    by_q = {q.id: q for q in bank}
    kind_of: dict[tuple[str, str], str] = {}
    for e in plan:
        for q in e["decoys"]:
            kind_of[(e["path"], q)] = "decoy"
        for q in e["needed"]:
            true = truth[e["path"]] in by_q[q].classes
            kind_of[(e["path"], q)] = "candidate_true_class" if true else "candidate_wrong_class"
    count: dict[str, dict[str, int]] = {}
    for a in answers:
        c = count.setdefault(kind_of[(a["path"], a["question"])], {"yes": 0, "no": 0, "unknown": 0})
        c[a["answer"]] += 1
    out = {}
    for kind, c in count.items():
        n = sum(c.values())
        out[kind] = {"n": n, **c, "yes_rate": c["yes"] / n, "unknown_rate": c["unknown"] / n,
                     "yes_rate_of_answered": c["yes"] / max(c["yes"] + c["no"], 1)}
    return out


def attempt1_answers(key: dict, answers: list[dict]) -> list[dict]:
    """The saved answers with sheets s001-s070 replaced by the first attempt's."""
    first: dict[tuple[str, str], str] = {}
    for n in SHEETS:
        sheet = f"s{n:03d}"
        given = json.loads((ATTEMPT1 / f"{sheet}.answers.json").read_text(encoding="utf-8"))
        for cell, info in key[sheet].items():
            for j, qid in enumerate(info["questions"], 1):
                first[(info["path"], qid)] = str(given[cell][str(j)]).strip().lower()
    return [{**a, "answer": first.get((a["path"], a["question"]), a["answer"])} for a in answers]


def rescore(cfg: dict, answers: list[dict]) -> dict:
    """Flow summaries per view and system on these answers, with nothing written."""
    by_path: dict[str, dict[str, str]] = {}
    for a in answers:
        by_path.setdefault(a["path"], {})[a["question"]] = a["answer"]
    meta = json.loads((ROOT / E.META_REL).read_text(encoding="utf-8"))
    models, names = {}, None
    for key in E._model_keys(cfg):
        m = cfg["models"][key]
        rec, models[key] = E._saved(ROOT, m["record"], m["system"])
        names = names or rec["config"]["class_names"]
    real_write, out, seen = E.write_run, {}, None
    E.write_run = lambda *a, **k: Path("none")
    try:
        for view in cfg["views"]:
            _, flow = E.run_view(cfg, view, ROOT, models, names, by_path, meta, F.load_questions(), seen, "")
            seen = seen or flow
            out[view["name"]] = flow
    finally:
        E.write_run = real_write
    return out


def _row(label: str, k: dict) -> str:
    return f"| {label} | {k['n']} | {k['yes_rate']:.3f} | {k['yes_rate_of_answered']:.3f} | {k['unknown_rate']:.3f} |"


def _flow_line(view: str, system: str, f: dict) -> str:
    w, a = f["whole_split"], f["asked_subset"]
    return (f"| {view} | {system} | {f['unknown_rate']:.3f} | {f['better']} | {f['worse']} | "
            f"{w['accuracy_before']:.4f} -> {w['accuracy_after']:.4f} [{w['accuracy_diff_ci'][0]:+.4f}, "
            f"{w['accuracy_diff_ci'][1]:+.4f}] | {w['macro_f1_before']:.4f} -> {w['macro_f1_after']:.4f} | "
            f"{a['accuracy_before']:.4f} -> {a['accuracy_after']:.4f} |")


def main() -> int:
    cfg = E.load_config(ROOT / "configs" / "e11.json")
    answers = [json.loads(x) for x in (ROOT / E.ANSWERS_REL).read_text(encoding="utf-8").splitlines() if x.strip()]
    plan = json.loads((ROOT / E.PLAN_REL).read_text(encoding="utf-8"))["plan"]
    field = pd.read_csv(ROOT / "datasets" / "manifests" / "field.csv")
    truth = dict(zip(field["path"], field["label"], strict=True))
    bank = F.load_questions()
    rates = yes_rates(answers, plan, truth, bank)
    key = json.loads((ROOT / E.KEY_REL).read_text(encoding="utf-8"))
    changed = sum(a["answer"] != b["answer"] for a, b in zip(answers, attempt1_answers(key, answers), strict=True))
    first = rescore(cfg, attempt1_answers(key, answers))
    final = rescore(cfg, answers)
    lines = [
        "# E11 answer check and wording sensitivity",
        "",
        "Computed by `tools/e11_answer_check.py` from the saved answers, `plan.json`, the manifest and the sheet key. "
        "It changes no answer and no record. The class labels are used only to describe the answers after they were "
        "saved. Development numbers on cleaned `field_dev`.",
        "",
        "## Does the answerer say yes more to the true class?",
        "",
        "Yes rate by kind of question, over all 6,190 saved answers. A decoy is a question about a class outside the "
        "photo's top 3; a candidate is a question the flow could ask. The true class is the photo's label.",
        "",
        "| kind | answers | yes rate | yes rate of answered | unknown rate |",
        "|---|---|---|---|---|",
        _row("decoy", rates["decoy"]),
        _row("candidate, wrong class", rates["candidate_wrong_class"]),
        _row("candidate, true class", rates["candidate_true_class"]),
        "",
        "## Sensitivity to the first-attempt wording (sheets s001-s070)",
        "",
        f"For this report only, not a new selection. {changed} of 6,190 answers differ when s001-s070 use the first "
        "attempt's answers (kept in the git-ignored `datasets/followup/attempt1/`); the flow is run on them with "
        "nothing written. E11's records use the one-wording answers.",
        "",
        "| answers | view | system | unknown rate | better | worse | whole view accuracy | macro-F1 | unsure photos accuracy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, flows in (("saved (one wording)", final), ("attempt 1 on s001-s070", first)):
        for view, flow in flows.items():
            for system, f in flow.items():
                lines.append(_flow_line(view, system, f).replace("| " + view, f"| {tag} | {view}", 1))
    atomic_write_text(REPORT, "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
