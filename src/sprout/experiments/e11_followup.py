"""E11: do follow-up questions about visible symptoms improve the model's answer when it is unsure?

Run from the repo root with PYTHONPATH set to its src folder, one stage per command (configs/e11.json holds every
setting and rule and was committed before any answer existed):
    .venv/Scripts/python -m sprout.experiments.e11_followup --stage <stage>
  plan      CPU  which cleaned field_dev photos the saved E4 models are unsure about, and which questions each
                 could be asked -> datasets/followup/plan.json (tracked). Reads probabilities only.
  sheets    CPU  contact sheets of those photos and their questions, shuffled by the seed, into the git-ignored
                 datasets/followup/sheets/, and the key that maps cells back to photos into datasets/followup/key/
  collect   CPU  joins the answer files a vision agent wrote for each sheet -> datasets/followup/answers.jsonl
                 (tracked) and answers.meta.json. Commit these before the next stage.
  record    CPU  runs the flow on the saved answers and writes one E11 record per view; refuses unless the answers
                 were committed first
Nothing runs a model: the probabilities are E4's saved ones. A reviewer sees only a sheet image and its question
text; nothing in plan, sheets or collect reads a label. field_test is never read.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from sprout.atomic import atomic_write_text
from sprout.serve import followup as F

ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP = Path("datasets/followup")
PLAN_REL = FOLLOWUP / "plan.json"
ANSWERS_REL = FOLLOWUP / "answers.jsonl"
META_REL = FOLLOWUP / "answers.meta.json"
SHEETS_REL = FOLLOWUP / "sheets"
KEY_REL = FOLLOWUP / "key" / "key.json"
RECORD_GLOB = "results/runs/E11-*.json"
SPLIT = "field_dev"
KIND = "a vision agent reading the photo, not a grower"


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["split"] != SPLIT:
        raise ValueError(f"E11 reads {SPLIT} only, not {cfg['split']!r}")
    return cfg


# ----------------------------------------------------------------------------------------------------------------
# plan


def build_plan(
    rows: dict[str, list[tuple[str, Any]]],
    names: list[str],
    bank: list[F.Question],
    unsure: dict[str, Any],
    sheets: dict[str, Any],
) -> list[dict[str, Any]]:
    """One entry per photo some model is unsure about and could be asked about, sorted by path.

    `rows` maps a model to its (path, probabilities) pairs; nothing else about a photo is used. `needed` holds
    every question the flow could ask about the photo's top 3 under any model that finds it unsure. `decoys`
    are other questions, drawn with the seed from classes in none of those top 3s, so that every photo has
    exactly `questions_per_photo` questions and the list does not show which classes were in its top 3."""
    needed: dict[str, set[str]] = {}
    classes: dict[str, set[str]] = {}
    for model_rows in rows.values():
        for path, probs in model_rows:
            top = F.top3(probs, names)
            if not F.is_unsure(top, unsure["min_probability"], unsure["min_plausible"]):
                continue
            ids = {q.id for q in F.askable(top, bank)}
            if ids:
                needed.setdefault(path, set()).update(ids)
                classes.setdefault(path, set()).update(c for c, _ in top)
    k = sheets["questions_per_photo"]
    plan = []
    for path in sorted(needed):
        ids = sorted(needed[path])
        if len(ids) > k:
            raise ValueError(f"{path} needs {len(ids)} questions, more than questions_per_photo ({k})")
        pool = sorted(q.id for q in bank if q.id not in needed[path] and not set(q.classes) & classes[path])
        if len(pool) < k - len(ids):
            raise ValueError(f"not enough decoy questions for {path}")
        decoys = sorted(random.Random(f"{sheets['seed']}:{path}:decoys").sample(pool, k - len(ids)))
        plan.append({"path": path, "needed": ids, "decoys": decoys})
    return plan


# ----------------------------------------------------------------------------------------------------------------
# sheets and answers

INSTRUCTIONS = (
    "You are looking at a sheet of {n} photos of plants. Each photo has a yellow number in its corner. Below, "
    "each photo has its own numbered questions about what is visible in it. You do not know what the plant or "
    "its problem is, and you should not try to guess: answer each question only from what you can see in that "
    "photo.\n\nFor every question write exactly one of `yes`, `no` or `unknown`. Use `unknown` when the photo "
    "cannot show it (too small, too blurry, not in the picture, or you cannot tell). Answer every question of "
    "every photo, and answer each photo on its own.\n\nCopy `{sheet}.template.json` to `{sheet}.answers.json` "
    "and replace every `null` with your answer, keeping the same keys: the outer key is the photo number, the "
    "inner key is the question number.\n"
)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _draw_cell(sheet: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, size: int, number: int, image: Path) -> None:
    box = (size - 2, size - 2)
    with Image.open(image) as img:
        img = img.convert("RGB")
        img.thumbnail(box, Image.LANCZOS)
        sheet.paste(img, (x + 1 + (box[0] - img.width) // 2, y + 1 + (box[1] - img.height) // 2))
    badge = _font(max(14, size // 14))
    w = int(draw.textlength(str(number), font=badge)) + 12
    draw.rectangle((x + 1, y + 1, x + 1 + w, y + 1 + badge.size + 8), fill=(255, 230, 0))
    draw.text((x + 7, y + 4), str(number), fill=(0, 0, 0), font=badge)
    draw.rectangle((x, y, x + size - 1, y + size - 1), outline=(150, 150, 150))


def build_sheets(
    plan: list[dict[str, Any]], bank: list[F.Question], out_dir: Path, key_path: Path, root: Path, sheets: dict[str, Any]
) -> dict[str, int]:
    """Contact sheets (image, instructions with questions, blank answer template) of the planned photos in an
    order shuffled with the seed, plus the key from sheet and cell back to the photo, kept apart from them.

    A sheet carries a number badge per photo and nothing else about it: no path, file name or id. Each photo's
    questions are in a shuffled order and numbered from 1; their ids stay in the key."""
    text = {q.id: q.text for q in bank}
    order = sorted(plan, key=lambda r: r["path"])
    random.Random(sheets["seed"]).shuffle(order)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("s*.*"):
        old.unlink()
    per, cols, size = sheets["photos_per_sheet"], sheets["columns"], sheets["cell_px"]
    key: dict[str, dict[str, Any]] = {}
    for n, start in enumerate(range(0, len(order), per), 1):
        name = f"s{n:03d}"
        chunk = order[start : start + per]
        grid = Image.new("RGB", (cols * size, -(-len(chunk) // cols) * size), (255, 255, 255))
        draw = ImageDraw.Draw(grid)
        lines = [f"# Sheet {name}\n", INSTRUCTIONS.format(n=len(chunk), sheet=name)]
        template: dict[str, dict[str, None]] = {}
        key[name] = {}
        for i, entry in enumerate(chunk, 1):
            _draw_cell(grid, draw, ((i - 1) % cols) * size, ((i - 1) // cols) * size, size, i, root / entry["path"])
            ids = entry["needed"] + entry["decoys"]
            random.Random(f"{sheets['seed']}:{entry['path']}:order").shuffle(ids)
            lines.append(f"\n## Photo {i}\n")
            lines += [f"{j}. {text[q]}" for j, q in enumerate(ids, 1)]
            template[str(i)] = {str(j): None for j in range(1, len(ids) + 1)}
            key[name][str(i)] = {"path": entry["path"], "questions": ids}
        grid.save(out_dir / f"{name}.jpg", quality=90)
        atomic_write_text(out_dir / f"{name}.md", "\n".join(lines) + "\n")
        atomic_write_text(out_dir / f"{name}.template.json", json.dumps(template, indent=1) + "\n")
    atomic_write_text(key_path, json.dumps(key, indent=1) + "\n")
    return {"photos": len(order), "sheets": len(key)}


def collect_answers(sheets_dir: Path, key_path: Path, dest: Path, answerer: str) -> dict[str, Any]:
    """Join each sheet's answer file to its photos through the key and save them, sorted, as `path, question,
    answer` rows. Every cell and question must be answered with yes, no or unknown. Returns the counts, which
    are also saved beside the answers with who answered."""
    key = json.loads(key_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for sheet, cells in key.items():
        file = sheets_dir / f"{sheet}.answers.json"
        if not file.is_file():
            raise ValueError(f"no answers for sheet {sheet}: {file.name} is missing")
        given = json.loads(file.read_text(encoding="utf-8"))
        if set(given) != set(cells):
            raise ValueError(f"{file.name} must answer photos {sorted(cells)}")
        for cell, info in cells.items():
            answers = given[cell]
            if not isinstance(answers, dict) or set(answers) != {str(j) for j in range(1, len(info["questions"]) + 1)}:
                raise ValueError(f"{file.name} photo {cell} must answer questions 1 to {len(info['questions'])}")
            for j, qid in enumerate(info["questions"], 1):
                value = answers[str(j)]
                if value is None:
                    raise ValueError(f"{file.name} photo {cell} question {j} is unanswered")
                value = str(value).strip().lower()
                if value not in F.ANSWERS:
                    raise ValueError(f"{file.name} photo {cell} question {j}: answers must be yes, no or unknown")
                rows.append({"path": info["path"], "question": qid, "answer": value})
    rows.sort(key=lambda r: (r["path"], r["question"]))
    meta = {
        "answerer": answerer,
        "kind": KIND,
        "sheets": len(key),
        "photos": len({r["path"] for r in rows}),
        "rows": len(rows),
        **{a: sum(r["answer"] == a for r in rows) for a in F.ANSWERS},
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(dest, "".join(json.dumps(r) + "\n" for r in rows))
    atomic_write_text(dest.with_name("answers.meta.json"), json.dumps(meta, indent=1) + "\n")
    return meta


# ----------------------------------------------------------------------------------------------------------------
# the order in which answers and records reached git


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def commit_order_problems(root: Path = ROOT) -> list[str]:
    """Why the saved answers could have been influenced by a score, or [].

    The answers must be committed, unchanged since, and every commit that touched them must be strictly before
    the commit that first added an E11 record. With no record yet, only the first two hold."""
    answers = ANSWERS_REL.as_posix()
    commits = _git(root, "log", "--format=%H", "--", answers).stdout.split()
    if not commits:
        return [f"{answers} is not committed"] if (root / answers).exists() else [f"no answers saved at {answers}"]
    if _git(root, "status", "--porcelain", "--", answers).stdout.strip():
        return [f"{answers} has uncommitted changes"]
    records = _git(root, "log", "--reverse", "--format=%H", "--", RECORD_GLOB).stdout.split()
    if not records:
        return []
    first = records[0]
    late = [c for c in commits if c == first or _git(root, "merge-base", "--is-ancestor", c, first).returncode != 0]
    return [f"{answers} changed at or after the first E11 record's commit ({c[:7]})" for c in late]
