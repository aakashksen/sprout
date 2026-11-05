"""Fail unless every number in docs/REPORT.md and the resume bullets traces to a run record, no placeholder
remains, and results/claims.json has claims.

A number is any digits in the text, except: single digits and 10 (counts of things, not metrics) unless they carry
a %; years 2000 to 2030; text inside URLs and paths; run ids (E17-20261105T...Z); hardware model names (HX 370); dates; task and gate ids (T8.2, E12, #90); seeds; top-k and
@k; k=N; Python versions; line numbering and section numbers (not starting with 0). Everything else must equal a
value in results/runs/*.json (or a value in claims.json), after rounding to the digits written. A figure with a
% also matches the record's value times 100, so a fraction stored as 0.123 backs "12%".
This guards against mistakes and casual edits. It does not decide that a cited record is the right one."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_REL = Path("docs/REPORT.md")
BULLETS_REL = Path("docs/RESUME-TARGETS.md")
CLAIMS_REL = Path("results/claims.json")
RUNS_REL = Path("results/runs")

PLACEHOLDER = re.compile(r"\[(?!x\])[a-z][a-z0-9_]*\](?!\()")
NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(%?)(?!\.\d)")
EXEMPT = [re.compile(p, re.I) for p in (
    r"\S*(?:://|[/\\])\S*",
    r"\bE\d+[A-Z]*-\d{8}T\d{12}Z\b",
    r"\b(?:HX|RTX|GTX)\s?\d+\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"#\d+",
    r"\bseeds?\s*[=:]?\s*\d+(?:\s*(?:,|and)\s*\d+)*",
    r"\btop-\d+",
    r"@\d+",
    r"\bk\s*=\s*\d+",
    r"\b(?:python|version|v)\s*\d+(?:\.\d+)+",
)]
LINE_NUMBER = re.compile(r"^\s*(?:\d+[.)]\s|#+\s+[1-9]\d?(?:\.\d{1,2})*\.?\s+(?=[A-Za-z]))")
SMALL_INTS = set(range(11))


def run_values(root: Path) -> list[float]:
    out: list[float] = []

    def walk(x: object) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            out.append(abs(float(x)))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for path in sorted((root / RUNS_REL).glob("*.json")):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def claim_values(claims: list) -> list[float]:
    return [abs(float(c["value"])) for c in claims
            if isinstance(c, dict) and isinstance(c.get("value"), (int, float)) and not isinstance(c["value"], bool)]


def traced(token: str, percent: bool, values: list[float]) -> bool:
    text = token.replace(",", "")
    decimals = len(text.split(".")[1]) if "." in text else 0
    target, half = float(text), 0.5 * 10 ** -decimals + 1e-9
    return any(abs(v - target) <= half or (percent and abs(v * 100 - target) <= half) for v in values)


def untraced(line: str, values: list[float]) -> list[str]:
    start = LINE_NUMBER.match(line)
    rest = line[start.end():] if start else line
    for pattern in EXEMPT:
        rest = pattern.sub(" ", rest)
    out = []
    for m in NUMBER.finditer(rest):
        token, percent = m.group(0).rstrip("%"), bool(m.group(1))
        is_int = re.fullmatch(r"\d+", token) is not None
        if is_int and not percent and (int(token) in SMALL_INTS or 2000 <= int(token) <= 2030):
            continue
        if not traced(token, percent, values):
            out.append(m.group(0))
    return out


def bullet_lines(text: str) -> list[tuple[int, str]]:
    out, inside = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            inside = line.startswith("## SPROUT")
        elif inside and re.match(r"\d+\.\s", line):
            out.append((n, line))
    return out


def read(root: Path, rel: Path, problems: list[str]) -> str | None:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError:
        problems.append(f"{rel.as_posix()} is missing or unreadable")
        return None


def check(root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    claims: list = []
    try:
        data = json.loads((root / CLAIMS_REL).read_text(encoding="utf-8"))
        found = data.get("claims") if isinstance(data, dict) else data
        claims = found if isinstance(found, list) else []
    except (OSError, ValueError):
        problems.append(f"{CLAIMS_REL.as_posix()} is missing or not valid JSON")
    else:
        if not claims:
            problems.append(f"{CLAIMS_REL.as_posix()} has no claims")
    values = run_values(root) + claim_values(claims)

    report = read(root, REPORT_REL, problems)
    bullets = read(root, BULLETS_REL, problems)
    sections = []
    if report is not None:
        sections.append((REPORT_REL, list(enumerate(report.splitlines(), 1))))
    if bullets is not None:
        sections.append((BULLETS_REL, bullet_lines(bullets)))
    for rel, lines in sections:
        for n, line in lines:
            for ph in PLACEHOLDER.findall(line):
                problems.append(f"{rel.as_posix()}:{n}: placeholder {ph} remains")
            for token in untraced(line, values):
                problems.append(f"{rel.as_posix()}:{n}: {token} is not in any run record")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("report numbers ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
