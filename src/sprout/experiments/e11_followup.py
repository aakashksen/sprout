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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sprout.atomic import atomic_write_text
from sprout.eval import metrics as M
from sprout.eval.check_run import Scored, recompute_comparison, recompute_interval
from sprout.eval.records import (
    ROOT,
    RUNS_REL,
    split_info,
    standard_image_metrics,
    subset_positions,
    write_run,
)
from sprout.serve import followup as F

FOLLOWUP = Path("datasets/followup")
PLAN_REL = FOLLOWUP / "plan.json"
ANSWERS_REL = FOLLOWUP / "answers.jsonl"
META_REL = FOLLOWUP / "answers.meta.json"
SHEETS_REL = FOLLOWUP / "sheets"
KEY_REL = FOLLOWUP / "key" / "key.json"
RECORD_GLOB = "results/runs/E11-*.json"
SPLIT = "field_dev"
KIND = "a vision agent reading the photo, not a grower"
COMPARED = ("accuracy", "macro_f1")


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
    are other questions, drawn with the seed from classes in none of those top 3s, preferring the same crops
    (the crop is plain to see, so decoys from another crop would give themselves away), so that every photo has
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
        outside = [q for q in bank if q.id not in needed[path] and not set(q.classes) & classes[path]]
        crops = {c.partition("___")[0] for c in classes[path]}
        same = sorted(q.id for q in outside if any(c.partition("___")[0] in crops for c in q.classes))
        other = sorted(q.id for q in outside if q.id not in same)
        want = k - len(ids)
        if len(same) + len(other) < want:
            raise ValueError(f"not enough decoy questions for {path}")
        rng = random.Random(f"{sheets['seed']}:{path}:decoys")
        decoys = rng.sample(same, min(want, len(same)))
        decoys = sorted(decoys + rng.sample(other, want - len(decoys)))
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


# ----------------------------------------------------------------------------------------------------------------
# the saved models, the plan stage


def _saved(root: Path, record_id: str, system: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rec = json.loads((root / RUNS_REL / f"{record_id}.json").read_text(encoding="utf-8"))
    if system not in rec["predictions"]:
        raise ValueError(f"{record_id} has no saved predictions for {system}; E11 does not run models")
    lines = (root / rec["predictions"][system]).read_text(encoding="utf-8").splitlines()
    return rec, {r["path"]: r for r in map(json.loads, (x for x in lines if x.strip()))}


def _view_paths(root: Path, subset: str) -> list[str]:
    info = split_info(SPLIT, root)
    return [info.ids[i] for i in subset_positions(subset, SPLIT, info, root)]


def _model_subset(cfg: dict[str, Any], key: str) -> str:
    source = cfg["models"][key].get("source")
    return cfg["subset"] if source is None else f"source:{source}+{cfg['subset']}"


def _model_keys(cfg: dict[str, Any]) -> list[str]:
    return [k for k in cfg["models"] if k != "note"]


def _rows_for(saved: dict[str, dict[str, Any]], paths: list[str], what: str) -> list[dict[str, Any]]:
    missing = [p for p in paths if p not in saved]
    if missing:
        raise ValueError(f"{what} lacks {len(missing)} of the rows being scored, e.g. {missing[0]}")
    return [saved[p] for p in paths]


def make_plan(
    cfg: dict[str, Any], root: Path = ROOT, bank: list[F.Question] | None = None, names: list[str] | None = None
) -> dict[str, Any]:
    """The plan for every saved model. Only each row's path and probabilities are taken from the saved rows."""
    bank = F.load_questions() if bank is None else bank
    unsure = cfg["unsure"]
    rows: dict[str, list[tuple[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for key in _model_keys(cfg):
        model = cfg["models"][key]
        rec, saved = _saved(root, model["record"], model["system"])
        names = names or rec["config"]["class_names"]
        paths = _view_paths(root, _model_subset(cfg, key))
        rows[key] = [(r["path"], r["probs"]) for r in _rows_for(saved, paths, f"{model['record']} {model['system']}")]
        tops = [F.top3(p, names) for _, p in rows[key]]
        unsure_tops = [t for t in tops if F.is_unsure(t, unsure["min_probability"], unsure["min_plausible"])]
        summary[key] = {
            "rows": len(tops),
            "unsure": len(unsure_tops),
            "unsure_without_a_question": sum(not F.askable(t, bank) for t in unsure_tops),
            "unsure_with_an_uncovered_class": sum(bool(F.uncovered(t, bank)) for t in unsure_tops),
        }
    plan = build_plan(rows, names, bank, unsure, cfg["sheets"])
    return {"summary": summary, "photos": len(plan), "plan": plan}


# ----------------------------------------------------------------------------------------------------------------
# the flow and its statistics


def flow_row(
    probs: Any, names: list[str], bank: list[F.Question], rule: F.Rule, unsure: dict[str, Any],
    answers: dict[str, str],
) -> dict[str, Any]:
    """Run the question flow on one photo. Photos the model is sure about, or that no question can help,
    come back unchanged with nothing asked."""
    top = F.top3(probs, names)
    out: dict[str, Any] = {
        "probs": [float(p) for p in probs],
        "asked": [],
        "answers": [],
        "in_scope": False,
        "unsure": F.is_unsure(top, unsure["min_probability"], unsure["min_plausible"]),
        "uncovered": F.uncovered(top, bank),
    }
    if not out["unsure"] or not F.askable(top, bank):
        return out
    out["in_scope"] = True
    for _ in range(rule.max_questions):
        question = F.next_question(top, asked=out["asked"], bank=bank)
        if question is None:
            break
        if question.id not in answers:
            raise ValueError(f"no saved answer for question {question.id}")
        top = F.update(top, question, answers[question.id], rule=rule)
        out["asked"].append(question.id)
        out["answers"].append(answers[question.id])
    out["probs"] = F.rerank(probs, names, top)
    return out


def ece_diff(before, after, labels, groups, n: int = 10000, seed: int = 13, bins: int = 15) -> tuple[float, float, float]:
    """ECE(after) - ECE(before) with a 95% interval, resampling whole groups of paired photos."""
    labels = np.asarray(labels)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    g = int(inverse.max()) + 1
    edges = np.linspace(0.0, 1.0, bins + 1)

    def per_group(probs: Any) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        conf, hit = probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(float)
        cell = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1) + bins * inverse
        count = np.bincount(cell, minlength=g * bins).reshape(g, bins).astype(float)
        gap = np.bincount(cell, weights=hit - conf, minlength=g * bins).reshape(g, bins)
        return np.stack([count, gap])  # a bin's share of ECE is |sum of (hit - confidence)| over its rows / N

    a, b = per_group(before), per_group(after)
    rng = np.random.default_rng(seed)
    diffs, done = [], 0
    while done < n:
        step = min(250, n - done)
        draw = rng.integers(0, g, size=(step, g))
        total = a[0][draw].sum(axis=(1, 2))
        ece_a = np.abs(a[1][draw].sum(axis=1)).sum(axis=1) / total
        ece_b = np.abs(b[1][draw].sum(axis=1)).sum(axis=1) / total
        diffs.append(ece_b - ece_a)
        done += step
    lo, hi = np.percentile(np.concatenate(diffs), [2.5, 97.5])
    return float(M.ece(after, labels) - M.ece(before, labels)), float(lo), float(hi)


def paired_numbers(before: Any, after: Any, labels: Any, groups: list[str], names: list[str]) -> dict[str, Any]:
    """Top-1, macro-F1 and calibration before and after, with the paired differences and their intervals."""
    labels = np.asarray(labels, dtype=int)
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    sb = Scored(groups=groups, labels=labels, class_names=names, probs=before)
    sa = Scored(groups=groups, labels=labels, class_names=names, probs=after)
    out: dict[str, Any] = {"n": len(labels)}
    mb, ma = standard_image_metrics(before, labels, names), standard_image_metrics(after, labels, names)
    for metric in ("accuracy", "macro_f1", "ece"):
        out[f"{metric}_before"], out[f"{metric}_after"] = mb[metric], ma[metric]
    for metric in COMPARED:
        got = recompute_comparison(sa, sb, metric)
        out[f"{metric}_diff"], out[f"{metric}_diff_ci"] = got["diff"], got["ci"]
        if metric == "accuracy":
            out["mcnemar_p"] = got["mcnemar_p"]
    out["ece_diff"], *ci = ece_diff(before, after, labels, groups)
    out["ece_diff_ci"] = ci
    return out


def _summary(
    flows: list[dict[str, Any]], before: np.ndarray, after: np.ndarray, labels: np.ndarray, groups: list[str],
    names: list[str],
) -> dict[str, Any]:
    scope = np.array([f["in_scope"] for f in flows], dtype=bool)
    top_before, top_after = before.argmax(axis=1), after.argmax(axis=1)
    right_before, right_after = top_before == labels, top_after == labels
    given = [a for f in flows for a in f["answers"]]
    asked_groups = [g for g, s in zip(groups, scope, strict=True) if s]
    return {
        "n_rows": len(flows),
        "unsure": sum(f["unsure"] for f in flows),
        "unsure_without_a_question": sum(f["unsure"] and not f["in_scope"] for f in flows),
        "unsure_with_an_uncovered_class": sum(f["unsure"] and bool(f["uncovered"]) for f in flows),
        "questions_asked": len(given),
        "questions_per_asked_photo": (len(given) / scope.sum()) if scope.any() else 0.0,
        "answers": {a: given.count(a) for a in F.ANSWERS},
        "unknown_rate": (given.count("unknown") / len(given)) if given else 0.0,
        "better": int((~right_before & right_after).sum()),
        "worse": int((right_before & ~right_after).sum()),
        "wrong_both_times_but_changed": int((~right_before & ~right_after & (top_before != top_after)).sum()),
        "whole_split": paired_numbers(before, after, labels, groups, names),
        "asked_subset": paired_numbers(before[scope], after[scope], labels[scope], asked_groups, names)
        if scope.any()
        else {"n": 0},
    }


def _read_answers(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    lines = (root / ANSWERS_REL).read_text(encoding="utf-8").splitlines()
    answers: dict[str, dict[str, str]] = {}
    for row in map(json.loads, (x for x in lines if x.strip())):
        answers.setdefault(row["path"], {})[row["question"]] = row["answer"]
    meta = json.loads((root / META_REL).read_text(encoding="utf-8"))
    return answers, meta


def _describe(system: str, f: dict[str, Any]) -> str:
    w, a = f["whole_split"], f["asked_subset"]
    nan = float("nan")
    return (
        f"{system}: whole view accuracy {w['accuracy_before']:.4f} -> {w['accuracy_after']:.4f}, macro-F1 "
        f"{w['macro_f1_before']:.4f} -> {w['macro_f1_after']:.4f}, ECE {w['ece_before']:.4f} -> {w['ece_after']:.4f}; "
        f"{a['n']} unsure photos accuracy {a.get('accuracy_before', nan):.4f} -> {a.get('accuracy_after', nan):.4f}; "
        f"better {f['better']}, worse {f['worse']}, {f['questions_asked']} questions, unknown rate {f['unknown_rate']:.3f}"
    )


def run_view(
    cfg: dict[str, Any],
    view: dict[str, Any],
    root: Path,
    models: dict[str, dict[str, dict[str, Any]]],
    names: list[str],
    answers: dict[str, dict[str, str]],
    meta: dict[str, Any],
    bank: list[F.Question],
    seen: dict[str, Any] | None,
    note: str,
) -> tuple[Path, dict[str, Any]]:
    """Score one view (one subset, one or two models) before and after the flow and write its record."""
    info = split_info(SPLIT, root)
    pos = subset_positions(view["subset"], SPLIT, info, root)
    paths, groups = [info.ids[i] for i in pos], [info.groups[i] for i in pos]
    rule = F.Rule(**{k: cfg["rule"][k] for k in ("yes_factor", "no_factor", "max_questions")})
    predictions: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    flow: dict[str, Any] = {}
    described: dict[str, str] = {}
    for system, key in view["systems"].items():
        saved = _rows_for(models[key], paths, f"model {key}")
        labels = np.array([r["label"] for r in saved], dtype=int)
        before = np.array([r["probs"] for r in saved], dtype=float)
        flows = [flow_row(r["probs"], names, bank, rule, cfg["unsure"], answers.get(r["path"], {})) for r in saved]
        after = np.array([f["probs"] for f in flows], dtype=float)
        flow[system] = _summary(flows, before, after, labels, groups, names)
        rows_before = [{"path": r["path"], "label": int(r["label"]), "probs": [float(p) for p in r["probs"]]} for r in saved]
        rows_after = [
            {"path": r["path"], "label": int(r["label"]), "probs": f["probs"], "asked": f["asked"], "answers": f["answers"]}
            for r, f in zip(saved, flows, strict=True)
        ]
        scored = {}
        for tag, rows, probs in (("before", rows_before, before), ("after", rows_after, after)):
            name = f"{system}_{tag}"
            predictions[name] = rows
            metrics[name] = standard_image_metrics(probs, labels, names)
            scored[tag] = Scored(groups=groups, labels=labels, class_names=names, probs=probs)
            intervals[name] = {m: recompute_interval(m, scored[tag]) for m in COMPARED}
            model = cfg["models"][key]
            described[name] = f"{key}: saved probabilities of {model['record']} {model['system']}" + (
                ", re-ranked by follow-up answers" if tag == "after" else "")
        for metric in COMPARED:
            got = recompute_comparison(scored["after"], scored["before"], metric)
            comparisons.append({"a": f"{system}_after", "b": f"{system}_before", "metric": metric, "diff": got["diff"],
                                "ci": got["ci"], "mcnemar_p": got["mcnemar_p"]})
    config: dict[str, Any] = {
        "class_names": names,
        "view": view["name"],
        "models": cfg["models"],
        "rule": cfg["rule"],
        "unsure": cfg["unsure"],
        "healthy": cfg["healthy"],
        "answers": {k: meta[k] for k in ("answerer", "kind", "sheets", "photos", "rows", *F.ANSWERS)},
        "answerer_note": "The answers are an agent's reading of each photo, not a grower's.",
        "flow": flow,
        "device": cfg["device"],
        "label": "development number, field_dev only; from E4's model, not a later E12 model",
        "limits": cfg["limits"],
    }
    if seen is not None:
        config["seen_source_number"] = seen["joint"]["whole_split"]
    parts = [_describe(system, f) for system, f in flow.items()]
    notes = (
        f"Development numbers on field_dev ({view['subset']}) from E4's saved probabilities, not E12's. "
        f"Answers by {meta['answerer']} ({meta['kind']}). " + "; ".join(parts) + (f" {note}" if note else "")
    )
    setting = "rule fixed in configs/e11.json before any answer: yes x{yes_factor}, no x{no_factor}, {max_questions} questions; not tuned"
    attempts = [{"setting": setting.format(**cfg["rule"]), "result": "; ".join(parts), "device": cfg["device"]}]
    path = write_run("E11", SPLIT, config, described, metrics, intervals, attempts, predictions, info.hash, root=root,
                     notes=notes, subset=view["subset"], comparisons=comparisons)
    return path, flow


def confirm_same_rows(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], paths: list[str], what: str) -> None:
    for p in paths:
        if p not in a or p not in b or a[p]["label"] != b[p]["label"] or not np.array_equal(a[p]["probs"], b[p]["probs"]):
            raise ValueError(f"saved rows of {what} do not match at {p}")


def run_all(cfg: dict[str, Any], root: Path = ROOT, *, bank: list[F.Question] | None = None, note: str = "") -> list[Path]:
    """Write one record per view. Refuses unless the answers were committed first and unchanged since."""
    problems = commit_order_problems(root)
    if problems:
        raise RuntimeError("the answers must be saved and committed before scoring: " + "; ".join(problems))
    bank = F.load_questions() if bank is None else bank
    answers, meta = _read_answers(root)
    models: dict[str, dict[str, dict[str, Any]]] = {}
    names: list[str] | None = None
    seen_model = cfg["models"]["seen"]
    for key in _model_keys(cfg):
        model = cfg["models"][key]
        rec, models[key] = _saved(root, model["record"], model["system"])
        names = names or rec["config"]["class_names"]
        if key != "seen":  # the seen model's rows in this record must be the very rows the seen record holds
            _, same = _saved(root, model["record"], seen_model["system"])
            confirm_same_rows(models["seen"], same, _view_paths(root, _model_subset(cfg, key)),
                              f"{model['record']} {seen_model['system']} against {seen_model['record']}")
    out, seen = [], None
    for view in cfg["views"]:
        path, flow = run_view(cfg, view, root, models, names, answers, meta, bank, seen, note)
        seen = seen or flow
        out.append(path)
    return out


# ----------------------------------------------------------------------------------------------------------------
# command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("plan", "sheets", "collect", "record"))
    ap.add_argument("--config", default="configs/e11.json")
    ap.add_argument("--answerer", default="", help="who answered, for the collect stage")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        if a.stage == "plan":
            plan = make_plan(cfg, ROOT)
            atomic_write_text(ROOT / PLAN_REL, json.dumps(plan, indent=1) + "\n")
            print(json.dumps(plan["summary"], indent=1))
            print(f"{plan['photos']} photos planned -> {PLAN_REL}")
        elif a.stage == "sheets":
            plan = json.loads((ROOT / PLAN_REL).read_text(encoding="utf-8"))["plan"]
            counts = build_sheets(plan, F.load_questions(), ROOT / SHEETS_REL, ROOT / KEY_REL, ROOT, cfg["sheets"])
            print(f"{counts['photos']} photos on {counts['sheets']} sheets in {SHEETS_REL}; key in {KEY_REL}")
        elif a.stage == "collect":
            if not a.answerer:
                raise ValueError("--answerer is required: say who answered")
            print(json.dumps(collect_answers(ROOT / SHEETS_REL, ROOT / KEY_REL, ROOT / ANSWERS_REL, a.answerer), indent=1))
        else:
            for path in run_all(cfg, ROOT, note=a.note):
                print(path)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
