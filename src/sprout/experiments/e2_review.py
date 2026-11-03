"""The by-hand parts of E2: which mistakes to categorize, which masks to judge, and how the answers are tallied.

Nothing here decides a category or a verdict. The sheets are written blank; a person fills them in
after looking at the images, and the summaries only count what was filled in.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sprout.data.masks import hsv_mask
from sprout.eval.predict import Predictions

CATEGORIES = ("blur", "background", "similar_symptoms", "wrong_label", "unsupported_condition")
MISTAKE_COLUMNS = ["path", "label", "predicted", "confidence", "category", "note"]
MASK_COLUMNS = ["index", "path", "good"]
VERDICTS = ("yes", "no")


def select_mistakes(preds: Predictions, class_names: list[str], n: int, seed: int) -> pd.DataFrame:
    """A seeded sample of n wrong predictions, in path order, with a blank category column."""
    guess = preds.probs.argmax(axis=1)
    wrong = sorted(i for i in range(len(guess)) if guess[i] != preds.label[i])
    rng = random.Random(seed)
    chosen = sorted(rng.sample(wrong, min(n, len(wrong))), key=lambda i: preds.path[i])
    rows = [
        (preds.path[i], class_names[preds.label[i]], class_names[guess[i]], round(float(preds.probs[i].max()), 4), "", "")
        for i in chosen
    ]
    return pd.DataFrame(rows, columns=MISTAKE_COLUMNS)


def select_mask_sample(paths: list[str], n: int, seed: int) -> pd.DataFrame:
    chosen = random.Random(seed).sample(sorted(paths), min(n, len(paths)))
    return pd.DataFrame({"index": range(len(chosen)), "path": chosen, "good": ""})


def write_mask_images(sheet: pd.DataFrame, out_dir: Path, size: int = 224) -> None:
    """For each sampled photo: the photo on the left, the photo with everything outside the mask blacked out on the right."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, path in zip(sheet["index"], sheet["path"], strict=True):
        with Image.open(path) as im:
            photo = im.convert("RGB").resize((size, size))
            mask = hsv_mask(im, size)
        cut = np.asarray(photo) * mask[..., None]
        both = np.concatenate([np.asarray(photo), cut], axis=1)
        Image.fromarray(both).save(out_dir / f"{index:03d}.jpg", quality=85)


def read_sheet(path: Path, columns: list[str]) -> pd.DataFrame:
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(sheet.columns) != columns:
        raise ValueError(f"{path} must have the columns {columns}")
    return sheet


def mistake_counts(sheet: pd.DataFrame) -> dict[str, int]:
    """Count per category plus 'not viewed' for blank rows; any other value is an error."""
    bad = sorted(set(sheet["category"]) - set(CATEGORIES) - {""})
    if bad:
        raise ValueError(f"unknown categories {bad}; use one of {list(CATEGORIES)}")
    counts = {c: int((sheet["category"] == c).sum()) for c in CATEGORIES}
    counts["not viewed"] = int((sheet["category"] == "").sum())
    return counts


def mask_verdict(sheet: pd.DataFrame, min_good: float) -> dict:
    """Share of judged masks marked good, and whether it reaches min_good. Blank rows are not judged."""
    bad = sorted(set(sheet["good"]) - set(VERDICTS) - {""})
    if bad:
        raise ValueError(f"unknown verdicts {bad}; use yes or no")
    judged = int((sheet["good"] != "").sum())
    good = int((sheet["good"] == "yes").sum())
    complete = judged == len(sheet) and judged > 0
    share = good / judged if judged else None
    return {
        "sampled": len(sheet),
        "judged": judged,
        "good": good,
        "share_good": share,
        "min_good": min_good,
        "reliable": bool(complete and share >= min_good),
    }


def mistakes_markdown(sheet: pd.DataFrame, header: str) -> str:
    counts = mistake_counts(sheet)
    lines = [
        "# E2 error analysis",
        "",
        header,
        "",
        "Categories: " + ", ".join(CATEGORIES) + ". Paths, categories and short notes only; no images are embedded.",
        "",
        "| Category | Mistakes |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in counts.items()],
        f"| total | {len(sheet)} |",
        "",
        "| Path | True class | Predicted | Confidence | Category | Note |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r.path} | {r.label} | {r.predicted} | {r.confidence} | {r.category or 'not viewed'} | {r.note} |"
            for r in sheet.itertuples()
        ],
        "",
    ]
    return "\n".join(lines)
