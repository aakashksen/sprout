"""Fail unless every dataset in docs/datasets.md has URL, License, Classes and Approved lines."""
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "datasets.md"
FIELDS = ("URL", "License", "Classes", "Approved")
APPROVED = re.compile(r"(yes|no, needs the person)\b")


def check(text: str) -> list[str]:
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    if not sections:
        return ["no dataset sections found"]
    errors = []
    for section in sections:
        name, _, body = section.partition("\n")
        name = name.strip()
        for field in FIELDS:
            m = re.search(rf"^{field}:[ \t]*(.*)$", body, flags=re.M)
            if not m or not m.group(1).strip():
                errors.append(f"{name}: missing {field} line")
            elif field == "Approved" and not APPROVED.match(m.group(1).strip()):
                errors.append(f"{name}: Approved must be 'yes' or 'no, needs the person'")
    return errors


def main() -> int:
    errors = check(DOC.read_text(encoding="utf-8"))
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
