"""results/claims.json: every published number must equal a value in a run record that passes check_run."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sprout.eval.check_run import check_file, test_once_problems
from sprout.eval.records import ROOT, RUNS_REL

CLAIMS_REL = Path("results/claims.json")
SIGNOFF_REL = Path("orchestration/label_signoff.json")
SIGNED_SPLITS = ("lab_test", "field_test", "rag_test")
COMPARISON_FIELDS = ("diff", "ci", "mcnemar_p")


def _system(c: dict[str, Any], rec: dict[str, Any]) -> tuple[str | None, str | None]:
    systems = list(rec.get("predictions") or {})
    name = c.get("system")
    if name is None:
        return (systems[0], None) if len(systems) == 1 else (None, "names no system, and the run has several")
    return (name, None) if name in systems else (None, f"system {name} is not in the run")


def _cited(c: dict[str, Any], rec: dict[str, Any]) -> tuple[object, str | None]:
    """(the record's value, a problem)."""
    kind, metric = c.get("kind", "metric"), c.get("metric")
    if kind == "comparison":
        for saved in rec.get("comparisons") or []:
            if (isinstance(saved, dict) and saved.get("a") == c.get("a") and saved.get("b") == c.get("b")
                    and saved.get("metric") == metric):
                if c.get("field") not in COMPARISON_FIELDS:
                    return None, f"field must be one of {', '.join(COMPARISON_FIELDS)}"
                return saved.get(c["field"]), None
        return None, f"comparison of {metric} for {c.get('a')} vs {c.get('b')} not in run"
    if kind in ("measurement", "judged"):
        table = (rec.get("measurements" if kind == "measurement" else "judged") or {}).get("values") or {}
        return (table[metric], None) if metric in table else (None, f"{kind} {metric} not in run")
    if kind not in ("metric", "interval"):
        return None, f"unknown kind {kind}"
    system, problem = _system(c, rec)
    if system is None:
        return None, problem
    table = (rec.get("metrics") if kind == "metric" else rec.get("intervals") or {}).get(system) or {}
    return (table[metric], None) if metric in table else (None, f"{kind} {metric} not in run")


def _review_problems(config: Any) -> list[str]:
    review = config.get("judge_review") if isinstance(config, dict) else None
    if not isinstance(review, dict):
        return ["a judged claim needs config.judge_review: the person's check of a random sample"]
    rate, size = review.get("agreement_rate"), review.get("sample_size")
    out = []
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        out.append("config.judge_review.agreement_rate must be a number from 0 to 1")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        out.append("config.judge_review.sample_size must be a positive integer")
    if review.get("sample") != "random":
        out.append("config.judge_review.sample must be 'random'")
    return out


def signoff_problems(root: Path, split: str) -> list[str]:
    """E9 records never hold the label signature. The person signs each test split, in words, in a tracked file, so a
    later sign-off needs no rerun of the once-only test and changes no record."""
    path = root / SIGNOFF_REL
    if not path.is_file():
        return [f"{SIGNOFF_REL.as_posix()} is missing, so the {split} labels are not signed"]
    row = (json.loads(path.read_text(encoding="utf-8")) or {}).get(split)
    if not isinstance(row, dict) or row.get("signed") is not True:
        return [f"{split} labels are not signed in {SIGNOFF_REL.as_posix()}"]
    missing = [k for k in ("by", "at", "words") if not str(row.get(k) or "").strip()]
    return [f"{split} is signed true in {SIGNOFF_REL.as_posix()} but names no {', '.join(missing)}: "
            "a signature needs by, at and words"] if missing else []


def check_claims(root: Path = ROOT) -> list[str]:
    claims = json.loads((root / CLAIMS_REL).read_text(encoding="utf-8")).get("claims")
    if not isinstance(claims, list):
        return ["claims.json must have a 'claims' key holding a list"]
    problems: list[str] = []
    once = None
    for c in claims:
        cid = c.get("id", "?")
        path = root / RUNS_REL / f"{c.get('run_id')}.json"
        if not path.is_file():
            problems.append(f"claim {cid}: no run record {c.get('run_id')}")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        saved, problem = _cited(c, rec)
        if problem:
            problems.append(f"claim {cid}: {problem}")
        elif saved is None:
            problems.append(f"claim {cid}: {c.get('metric')} is undefined (None) in the run")
        elif saved != c.get("value"):
            problems.append(f"claim {cid}: value differs from the {c.get('kind', 'metric')} in the run")
        if c.get("kind") == "judged":
            problems += [f"claim {cid}: {m}" for m in _review_problems(rec.get("config"))]
        if rec.get("experiment") == "E9":
            if rec.get("split") in SIGNED_SPLITS:
                problems += [f"claim {cid}: {m}" for m in signoff_problems(root, rec["split"])]
        elif not rec.get("labels_reviewed"):
            problems.append(f"claim {cid}: run has labels_reviewed false")
        once = test_once_problems(root) if once is None else once
        problems += [f"claim {cid}: run: {m}" for m in check_file(path, root, once=once)]
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["check"]:
        print("usage: python -m sprout.eval.claims check")
        return 2
    problems = check_claims()
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("claims ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
