"""Fail unless each uncovered class in docs/document-coverage.csv has a section in docs/document-candidates.md,
and each candidate has URL, Publisher, License and Approved lines (Approved: no, needs the person)."""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "document-candidates.md"
COVERAGE = ROOT / "docs" / "document-coverage.csv"
FIELDS = ("URL", "Publisher", "License", "Approved")
APPROVED = re.compile(r"no, needs the person\b")
NONE_FOUND = "No acceptable candidate found"


def uncovered_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r["label"] for r in csv.DictReader(f) if r["symptoms_described"].strip() == "no"]


def _sections(text: str) -> dict[str, str]:
    out = {}
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        name, _, body = section.partition("\n")
        out[name.strip()] = body
    return out


def _candidate_errors(cls: str, candidate: str) -> list[str]:
    title, _, body = candidate.partition("\n")
    errors = []
    for field in FIELDS:
        m = re.search(rf"^{field}:[ \t]*(.*)$", body, flags=re.M)
        if not m or not m.group(1).strip():
            errors.append(f"{cls} / {title.strip()}: missing {field} line")
        elif field == "Approved" and not APPROVED.match(m.group(1).strip()):
            errors.append(f"{cls} / {title.strip()}: Approved must be 'no, needs the person'")
    return errors


def _class_errors(cls: str, section: str | None) -> list[str]:
    if section is None:
        return [f"{cls}: missing section"]
    candidates = re.split(r"^### ", section, flags=re.M)[1:]
    if not candidates:
        return [] if NONE_FOUND in section else [f"{cls}: no candidates and no '{NONE_FOUND}' statement"]
    return [e for cand in candidates for e in _candidate_errors(cls, cand)]


def check(text: str, classes: list[str]) -> list[str]:
    if not classes:
        return ["no uncovered classes given"]
    sections = _sections(text)
    return [e for cls in classes for e in _class_errors(cls, sections.get(cls))]


def main() -> int:
    classes = uncovered_classes(COVERAGE)
    if not classes:
        print("no uncovered classes; nothing to check")
        return 0
    errors = check(DOC.read_text(encoding="utf-8"), classes)
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
