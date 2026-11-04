"""E9: the final test. Each test split is scored once, with configurations frozen and committed before.

Stages (see docs/PLAN.md T8.1; every stage that touches a test split refuses unless the configs are committed, the
tree is clean and the split has not been used):
    lab-test, field-test        score every frozen system on one image test split and write its record
    wrong-disease-dev           the wrong-disease check on the final E7 served rag_dev answers (Sonnet), with its rate
    blind-sample, unblind       a seeded sample for a reader who does not see the grader's call, then the error report
    freeze                      record the check's prompt commit and sha256 in configs/e9_wrong_disease.json
    rag-retrieve, rag-generate, rag-judge-local, rag-sonnet, rag-record
                                rag_test: served retrieval, the local writer, both judges, the wrong-disease check
No stage reads a test split unless it is the one being run. Tests use fakes and tiny fixtures only.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from sprout.eval.records import ROOT, git, is_dirty

CONFIG_REL = "configs/e9.json"
FROZEN_REL = "configs/e9_wrong_disease.json"
PROMPT_REL = "src/sprout/rag/prompts/wrong_disease.txt"
IMAGE_TESTS = ("lab_test", "field_test")
CALLS = ("FITS", "WRONG")


class Refused(RuntimeError):
    """A guard stopped the stage before it did anything."""


# ---------------------------------------------------------------- guards

def require_committed(root: Path, rels: list[str]) -> None:
    """Every file is tracked and identical to HEAD, so the frozen setting was committed before it was used."""
    for rel in rels:
        tracked = git(root, "ls-files", "--error-unmatch", "--", rel).returncode == 0
        same = git(root, "diff", "--quiet", "HEAD", "--", rel).returncode == 0
        if not (tracked and same and (root / rel).is_file()):
            raise Refused(f"{rel} is not committed (or differs from HEAD); freeze and commit it before the test run")


def require_clean(root: Path) -> None:
    if is_dirty(root):
        raise Refused("the working tree has uncommitted changes; commit first")


def labels_reviewed(root: Path = ROOT, tasks_file: str = "orchestration/tasks.json") -> bool:
    """True only when H2 is marked done in the task list, which is where the person's sign-off is recorded."""
    path = Path(root) / tasks_file
    tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks", []) if path.is_file() else []
    return any(t.get("id") == "H2" and t.get("status") == "done" for t in tasks)


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((Path(root) / CONFIG_REL).read_text(encoding="utf-8"))


def load_decision(cfg: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """E12's decision, read by path. Without it the chosen model is unknown, so nothing runs."""
    path = Path(root) / cfg["decision_file"]
    if not path.is_file():
        raise Refused(f"{cfg['decision_file']} does not exist: E12 has not chosen a model, so E9 cannot start")
    decision = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("chosen_model", "chosen_model_sha256", "class_names") if k not in decision]
    if missing:
        raise Refused(f"{cfg['decision_file']} lacks {missing}")
    return decision


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the wrong-disease prompt

def prompt_text(root: Path = ROOT) -> str:
    return (Path(root) / PROMPT_REL).read_text(encoding="utf-8").strip()


def wd_messages(question: str, diagnosis: str, answer: str, root: Path = ROOT) -> list[dict[str, str]]:
    """What the grader is shown: the question, the diagnosis and the answer. No passages, no faithfulness grade."""
    from sprout.experiments.e14_judge_check import clean

    body = f"Question: {clean(question)}\n\nDiagnosis: {clean(diagnosis)}\n\nAnswer: {clean(answer)}"
    return [{"role": "system", "content": prompt_text(root)}, {"role": "user", "content": body}]


def parse_call(reply: str) -> str | None:
    """FITS or WRONG for exactly that one word (case and punctuation aside); anything longer is unusable, not a guess."""
    words = " ".join(re.findall(r"[A-Za-z]+", reply)).upper()
    return words if words in CALLS else None


# ---------------------------------------------------------------- image test splits

def run_images(split: str, cfg: dict[str, Any], predictors: dict[str, Any], root: Path = ROOT, *,
               override_note: str = "") -> list[Path]:
    if split not in IMAGE_TESTS:
        raise Refused(f"E9 scores lab_test or field_test here, not {split!r}")
    require_committed(root, [CONFIG_REL])
    require_clean(root)
    load_decision(cfg, root)
    return []
