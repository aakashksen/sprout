"""Label review of the real photos: contact sheets, a collector for the reviewers' verdicts, the exclusion log,
the cleaned manifest and the rate report.

Reviewers see an image, a cell number, its label and its file name, and nothing else: no split name, no score,
no prediction. The split is added from field.csv only when verdicts are joined. This module reads no run
results and imports nothing from the evaluation package.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from sprout.atomic import atomic_write_text
from sprout.data.splits import COLUMNS, dataset_hash

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ROOT / "datasets"
FIELD_CSV = DATASETS / "manifests" / "field.csv"
CLEAN_CSV = DATASETS / "manifests" / "field_clean.csv"
EXCLUSIONS = DATASETS / "label_exclusions.csv"
UNCERTAIN = DATASETS / "label_uncertain.csv"
REPORT = ROOT / "docs" / "label-review.md"
SHEETS = DATASETS / "review" / "label_sheets"
JOINED = DATASETS / "review" / "label_joined"  # holds splits, so reviewers never get this folder

SEED = 13
REVIEW_SPLITS = ("field_dev", "field_test")
TRAIN_SPLIT = "field_train"
TRAIN_SAMPLE = 400
VERDICTS = ("ok", "remove", "uncertain")
REASONS = ("not_plant_photo", "wrong_plant", "label_contradicted")
SIDECAR_COLUMNS = ["sheet", "cell", "image_id", "label", "filename"]
REVIEW_COLUMNS = ["sheet", "cell", "verdict", "reason_code", "note"]
LOG_COLUMNS = ["version", "image_id", "path", "source", "split", "label", "reason_code", "note", "began_as"]
FIRST_GRID, FIRST_CELL = (6, 5), 256  # columns, rows; pixels per cell (1536 wide, under the 1568 px reading size)
SECOND_GRID, SECOND_CELL = (3, 2), 500
CAPTION = 46
HEADER = 28
BAD_RATE = 0.05  # the plan's line for reviewing all of field_train
CONFIG = ROOT / "configs" / "e13.json"
CALIBRATION = (
    "Limit, from a calibration of the first pass: of 4 known-bad photos placed on the sheets, 2 were `remove`, "
    "1 was `uncertain` (\"framed slide-like image\") and 1 was `ok`. The sheets are conservative, so the removal "
    "rates below are a lower bound on bad photos."
)


def id_column(field: pd.DataFrame) -> str:
    """The manifest's sha1 when every row has a different one, else the path."""
    sha = field["sha1"]
    return "sha1" if sha.notna().all() and sha.is_unique else "path"


def with_ids(field: pd.DataFrame) -> pd.DataFrame:
    out = field.copy()
    out["image_id"] = out[id_column(field)].astype(str)
    return out


def shuffled_rows(field: pd.DataFrame, splits=REVIEW_SPLITS, seed: int = SEED) -> pd.DataFrame:
    """The rows of these splits in one seeded random order, so no sheet is one split."""
    part = with_ids(field)
    part = part[part["split"].isin(splits)].sort_values("path")
    return part.sample(frac=1, random_state=seed).reset_index(drop=True)


def train_sample(field: pd.DataFrame, n: int = TRAIN_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    part = with_ids(field)
    part = part[part["split"] == TRAIN_SPLIT].sort_values("path")
    return part.sample(n=min(n, len(part)), random_state=seed).reset_index(drop=True)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _fit(draw: ImageDraw.ImageDraw, text: str, width: int, size: int, floor: int = 11) -> tuple[str, ImageFont.FreeTypeFont]:
    """The text in the largest font from size down to floor that fits, cut with '..' if even floor is too wide."""
    for s in range(size, floor - 1, -1):
        font = _font(s)
        if draw.textlength(text, font=font) <= width:
            return text, font
    font = _font(floor)
    while len(text) > 3 and draw.textlength(text + "..", font=font) > width:
        text = text[:-1]
    return text + "..", font


def _draw_cell(sheet: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, size: int, number: int, row) -> None:
    box = (size - 2, size - CAPTION - 2)
    with Image.open(row["path"]) as img:
        img = img.convert("RGB")
        img.thumbnail(box, Image.LANCZOS)
        sheet.paste(img, (x + 1 + (box[0] - img.width) // 2, y + 1 + (box[1] - img.height) // 2))
    badge = _font(max(18, size // 16))
    w = int(draw.textlength(str(number), font=badge)) + 12
    draw.rectangle((x + 1, y + 1, x + 1 + w, y + 1 + badge.size + 8), fill=(255, 230, 0))
    draw.text((x + 7, y + 4), str(number), fill=(0, 0, 0), font=badge)
    label, font = _fit(draw, row["label"], size - 8, max(15, size // 22))
    draw.text((x + 4, y + size - CAPTION + 1), label, fill=(0, 0, 0), font=font)
    name, font = _fit(draw, unquote(Path(row["path"]).name), size - 8, 14, floor=10)
    draw.text((x + 4, y + size - CAPTION + 24), name, fill=(90, 90, 90), font=font)


def _make(rows: pd.DataFrame, out_dir: Path, prefix: str, grid: tuple[int, int], size: int) -> pd.DataFrame:
    cols, nrows = grid
    per = cols * nrows
    out_dir.mkdir(parents=True, exist_ok=True)
    side: list[tuple] = []
    for s, start in enumerate(range(0, len(rows), per), 1):
        name = f"{prefix}{s:04d}"
        chunk = rows.iloc[start : start + per]
        height = HEADER + math.ceil(len(chunk) / cols) * size
        sheet = Image.new("RGB", (cols * size, height), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"sheet {name}", fill=(0, 0, 0), font=_font(18))
        mine = []
        for i, (_, row) in enumerate(chunk.iterrows(), 1):
            x, y = ((i - 1) % cols) * size, HEADER + ((i - 1) // cols) * size
            _draw_cell(sheet, draw, x, y, size, i, row)
            draw.rectangle((x, y, x + size - 1, y + size - 1), outline=(150, 150, 150))
            mine.append((name, i, row["image_id"], row["label"], Path(row["path"]).name))
        sheet.save(out_dir / f"{name}.jpg", quality=90)
        pd.DataFrame(mine, columns=SIDECAR_COLUMNS).to_csv(out_dir / f"{name}.csv", index=False)
        side += mine
    return pd.DataFrame(side, columns=SIDECAR_COLUMNS)


def make_sheets(rows: pd.DataFrame, out_dir: Path = SHEETS, prefix: str = "r") -> pd.DataFrame:
    """First-pass sheets (6 x 5 cells of 256 px), each with a sidecar CSV. Returns all sidecar rows."""
    return _make(rows, out_dir, prefix, FIRST_GRID, FIRST_CELL)


def make_second_look(rows: pd.DataFrame, out_dir: Path = SHEETS, prefix: str = "c") -> pd.DataFrame:
    """Larger sheets (3 x 2 cells of 500 px) of the photos proposed for removal."""
    return _make(rows, out_dir, prefix, SECOND_GRID, SECOND_CELL)


# ----------------------------------------------------------------------------------------------------------------
# verdicts


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def collect(reviews: list[Path], sidecars: list[Path], field: pd.DataFrame) -> pd.DataFrame:
    """Join reviewers' verdicts to image IDs. Refuses unknown cells, missing cells, duplicates and bad codes.

    `sidecars` lists every sheet the verdicts must cover. The result has the split from field.csv.
    """
    side = pd.concat([_read_csv(p) for p in sidecars], ignore_index=True)
    side["cell"] = side["cell"].astype(int)
    got = pd.concat([_read_csv(p) for p in reviews], ignore_index=True)
    absent = [c for c in REVIEW_COLUMNS if c not in got.columns]
    if absent:
        raise ValueError(f"reviewer files lack columns {absent}")
    got["cell"] = got["cell"].astype(int)
    got["verdict"] = got["verdict"].str.strip().str.lower()
    known = set(zip(side["sheet"], side["cell"]))
    unknown = [(s, c) for s, c in zip(got["sheet"], got["cell"]) if (s, c) not in known]
    if unknown:
        raise ValueError(f"{len(unknown)} verdicts name cells that are not on the sheets, e.g. {unknown[0]}")
    dup = got[got.duplicated(["sheet", "cell"], keep=False)]
    if len(dup):
        raise ValueError(f"{len(dup)} verdict rows repeat a cell, e.g. {(dup.iloc[0]['sheet'], dup.iloc[0]['cell'])}")
    missing = sorted(known - set(zip(got["sheet"], got["cell"])))
    if missing:
        raise ValueError(f"{len(missing)} cells have no verdict, e.g. {missing[0]}")
    bad = got[~got["verdict"].isin(VERDICTS)]
    if len(bad):
        raise ValueError(f"verdict must be one of {VERDICTS}, got {bad.iloc[0]['verdict']!r}")
    code = got["reason_code"].str.strip()
    if (got["verdict"].eq("remove") & ~code.isin(REASONS)).any():
        raise ValueError(f"a remove verdict needs a reason_code from {REASONS}")
    if (got["verdict"].ne("remove") & code.ne("") & ~code.isin(REASONS)).any():
        raise ValueError(f"reason_code must be blank or one of {REASONS}")
    if (got["verdict"].eq("ok") & code.ne("")).any():
        raise ValueError("an ok verdict must not carry a reason_code")
    got["reason_code"] = code
    joined = got.merge(side[["sheet", "cell", "image_id"]], on=["sheet", "cell"], how="left")
    ids = with_ids(field).set_index("image_id")
    unlisted = sorted(set(joined["image_id"]) - set(ids.index))
    if unlisted:
        raise ValueError(f"{len(unlisted)} image IDs are not in field.csv, e.g. {unlisted[0]}")
    info = ids.loc[joined["image_id"], ["path", "source", "split", "label"]].reset_index(drop=True)
    out = pd.concat([joined.reset_index(drop=True), info], axis=1)
    return out[["sheet", "cell", "image_id", "path", "source", "split", "label", "verdict", "reason_code", "note"]]


def proposed(first: pd.DataFrame, also_uncertain: str | None) -> pd.DataFrame:
    """First-pass removals, plus uncertain rows whose note matches the (case-insensitive) pattern."""
    hit = first["verdict"] == "remove"
    if also_uncertain:
        hit |= (first["verdict"] == "uncertain") & first["note"].map(lambda n: re.search(also_uncertain, n, re.I) is not None)
    return first[hit]


def decide(
    first: pd.DataFrame, second: pd.DataFrame, also_uncertain: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(removals, uncertain, overturned) from the first pass and the larger second look at its proposed photos.

    A photo is removed only if the second look says remove (its reason and note are kept; `began_as` says whether
    the first pass said remove or uncertain). Uncertain photos not confirmed stay in the uncertain list. A
    first-pass removal the second look calls ok is overturned.
    """
    asked = proposed(first, also_uncertain)
    if set(second["image_id"]) != set(asked["image_id"]):
        raise ValueError("the second look must cover exactly the photos proposed for removal")
    began = dict(zip(asked["image_id"], asked["verdict"]))
    removals = second[second["verdict"] == "remove"].assign(began_as=lambda d: d["image_id"].map(began))
    pool = pd.concat(
        [first[first["verdict"] == "uncertain"], second[second["verdict"] == "uncertain"]], ignore_index=True
    ).drop_duplicates("image_id")
    uncertain = pool[~pool["image_id"].isin(removals["image_id"])].assign(
        began_as=lambda d: d["image_id"].map(began).fillna("uncertain")
    )
    look = second.set_index("image_id")["verdict"]
    overturned = asked[(asked["verdict"] == "remove") & asked["image_id"].map(look).eq("ok")]
    return removals, uncertain, overturned


def second_outcomes(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, int]:
    """How the second look judged the photos the first pass proposed for removal."""
    ids = first[first["verdict"] == "remove"]["image_id"]
    return second[second["image_id"].isin(ids)]["verdict"].value_counts().to_dict()


def log_rows(verdicts: pd.DataFrame, version: int) -> pd.DataFrame:
    out = verdicts.assign(version=version)
    return out[LOG_COLUMNS].reset_index(drop=True)


def check_log(log: pd.DataFrame, field: pd.DataFrame, need_reason: bool = True) -> None:
    """Every entry must name a field.csv row and carry matching path, source, split and label, and a reason."""
    absent = [c for c in LOG_COLUMNS if c not in log.columns]
    if absent:
        raise ValueError(f"log lacks columns {absent}")
    ids = with_ids(field).set_index("image_id")
    unknown = sorted(set(log["image_id"]) - set(ids.index))
    if unknown:
        raise ValueError(f"{len(unknown)} logged image IDs are not in field.csv, e.g. {unknown[0]}")
    if log["image_id"].duplicated().any():
        raise ValueError("an image is logged twice")
    row = ids.loc[log["image_id"]].reset_index(drop=True)
    for col in ("path", "source", "split", "label"):
        if (row[col].astype(str).to_numpy() != log[col].astype(str).to_numpy()).any():
            raise ValueError(f"logged {col} differs from field.csv")
    code = log["reason_code"].astype(str)
    if not (code.isin(REASONS) | (code.eq("") & (not need_reason))).all():
        raise ValueError(f"reason_code must be one of {REASONS}")
    if (log["source"].astype(str) == "").any() or (log["split"].astype(str) == "").any():
        raise ValueError("every entry needs a source and a split")


def read_log(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, dtype={"version": int}, keep_default_na=False)


def append_log(path: Path, rows: pd.DataFrame, field: pd.DataFrame, need_reason: bool = True) -> pd.DataFrame:
    """Add one round of entries. A round has a version above every earlier one; no image is logged twice."""
    old = read_log(path)
    version = int(rows["version"].iloc[0]) if len(rows) else None
    if version is not None:
        if rows["version"].nunique() != 1 or (len(old) and version <= old["version"].max()):
            raise ValueError("a round needs one version, above every version already logged")
        again = set(old["image_id"]) & set(rows["image_id"])
        if again:
            raise ValueError(f"{len(again)} images are already logged, e.g. {sorted(again)[0]}")
    both = pd.concat([old, rows], ignore_index=True)
    check_log(both, field, need_reason)
    atomic_write_text(Path(path), both.to_csv(index=False))
    return both


# ----------------------------------------------------------------------------------------------------------------
# the cleaned manifest


def exclusion_paths(log: pd.DataFrame, version: int | None = None) -> set[str]:
    """Paths of the entries up to this version (all entries when None)."""
    rows = log if version is None else log[log["version"] <= version]
    return set(rows["path"])


def apply_exclusions(field: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """field.csv without the excluded rows, with the exclusion version in a `clean_version` column."""
    check_log(log, field)
    version = int(log["version"].max()) if len(log) else 0
    out = field[~field["path"].isin(exclusion_paths(log))].reset_index(drop=True)
    return out.assign(clean_version=version)


def write_clean(field_path: Path = FIELD_CSV, log_path: Path = EXCLUSIONS, out_path: Path = CLEAN_CSV) -> str:
    """Write field_clean.csv from field.csv and the exclusion log. Returns its dataset hash. field.csv is only read."""
    field = pd.read_csv(field_path)
    clean = apply_exclusions(field, read_log(log_path))
    atomic_write_text(Path(out_path), clean.to_csv(index=False))
    return dataset_hash(clean[COLUMNS])


# ----------------------------------------------------------------------------------------------------------------
# the rate report


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, mid - half), min(1.0, mid + half))


def _rate_table(reviewed: pd.DataFrame, removed: pd.DataFrame, uncertain: pd.DataFrame, by: list[str]) -> str:
    n = reviewed.groupby(by).size().rename("reviewed")
    r = removed.groupby(by).size().rename("removed")
    u = uncertain.groupby(by).size().rename("uncertain")
    t = pd.concat([n, r, u], axis=1).fillna(0).astype(int)
    lines = ["| " + " | ".join(by) + " | reviewed | removed | removed % | uncertain |", "|" + "---|" * (len(by) + 4)]
    for key, row in t.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        lines.append(
            f"| {' | '.join(map(str, key))} | {row['reviewed']} | {row['removed']} | "
            f"{100 * row['removed'] / row['reviewed']:.1f} | {row['uncertain']} |"
        )
    return "\n".join(lines)


def write_report(
    first: pd.DataFrame, log: pd.DataFrame, uncertain: pd.DataFrame, outcomes: dict[str, int], path: Path = REPORT
) -> str:
    """Write docs/label-review.md. `first` is every joined first-pass verdict, dev, test and train sample alike."""
    dev_test = first[first["split"].isin(REVIEW_SPLITS)]
    sample = first[first["split"] == TRAIN_SPLIT]
    gone = log[log["image_id"].isin(first["image_id"])]
    began = gone["began_as"].value_counts()
    unsure = uncertain[uncertain["image_id"].isin(first["image_id"])]
    text = [
        "# Label review of the real photos",
        "",
        "Agent review, not a human review: vision agents looked at each photo with its label and file name and "
        "nothing else (no split name, no score, no prediction). A removal counts only after a second look at "
        "larger size. The rules: not a plant photo, wrong plant, or a label clearly contradicted by the photo; "
        "no relabeling and no fine disease judgments. Removed photos stay in `field.csv`; `field_clean.csv` leaves "
        f"them out. Exclusion version: {int(log['version'].max()) if len(log) else 0}.",
        "",
        f"Reviewed: {len(dev_test)} photos of `field_dev` and `field_test`, shuffled together. "
        f"Removed: {int(gone['split'].isin(REVIEW_SPLITS).sum())}. "
        f"Uncertain (kept): {int(unsure['split'].isin(REVIEW_SPLITS).sum())}. "
        f"First-pass removals: {sum(outcomes.values())}. At the larger second look: {outcomes.get('remove', 0)} "
        f"confirmed, {outcomes.get('ok', 0)} called ok (overturned), {outcomes.get('uncertain', 0)} called uncertain "
        "(kept, in `label_uncertain.csv`).",
        "",
        CALIBRATION,
        "",
        "How the removals began (all reviewed photos): "
        f"began as `remove`: {int(began.get('remove', 0))}; began as `uncertain`: {int(began.get('uncertain', 0))} "
        "(an uncertain photo went to the larger sheets when its note matched the configured pattern, and was "
        "removed only if the second look said remove).",
        "",
        "## By split",
        "",
        _rate_table(dev_test, gone[gone["split"].isin(REVIEW_SPLITS)], unsure[unsure["split"].isin(REVIEW_SPLITS)], ["split"]),
        "",
        "## By source",
        "",
        _rate_table(dev_test, gone[gone["split"].isin(REVIEW_SPLITS)], unsure[unsure["split"].isin(REVIEW_SPLITS)], ["source", "split"]),
        "",
        "## By class",
        "",
        _rate_table(dev_test, gone[gone["split"].isin(REVIEW_SPLITS)], unsure[unsure["split"].isin(REVIEW_SPLITS)], ["label"]),
        "",
        "## By reason",
        "",
        "| reason | removed |",
        "|---|---|",
        *[f"| {r} | {int((gone[gone['split'].isin(REVIEW_SPLITS)]['reason_code'] == r).sum())} |" for r in REASONS],
        "",
        "## field_train, estimated from a seeded sample",
        "",
    ]
    k = int(gone["split"].eq(TRAIN_SPLIT).sum())
    n = len(sample)
    if n:
        lo, hi = wilson(k, n)
        verdict = (
            f"The point estimate is above {BAD_RATE:.0%}" if k / n > BAD_RATE else f"The point estimate is not above {BAD_RATE:.0%}"
        )
        text += [
            f"A random sample of {n} of the `field_train` photos (seed {SEED}), reviewed the same way and "
            f"kept apart from the numbers above: {k} removed, {k / n:.1%}, Wilson 95% interval "
            f"[{lo:.1%}, {hi:.1%}]. Uncertain: {int(unsure['split'].eq(TRAIN_SPLIT).sum())}. {verdict}; "
            f"the interval's lower end is {'above' if lo > BAD_RATE else 'not above'} it.",
        ]
    else:
        text += ["No `field_train` sample has been reviewed yet."]
    atomic_write_text(Path(path), "\n".join(text) + "\n")
    return "\n".join(text)


# ----------------------------------------------------------------------------------------------------------------
# command line


def _sidecars(prefix: str, sheets_dir: Path) -> list[Path]:
    found = sorted(p for p in sheets_dir.glob(f"{prefix}[0-9][0-9][0-9][0-9].csv"))
    if not found:
        raise FileNotFoundError(f"no sheets with prefix {prefix!r} in {sheets_dir}")
    return found


def _pattern(given: str | None) -> str | None:
    """None when the option is absent, the given regex, or the default in configs/e13.json for a bare flag."""
    if given == "":
        return json.loads(CONFIG.read_text(encoding="utf-8"))["second_look_uncertain_regex"]
    return given


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheets", help="first-pass sheets of field_dev and field_test, shuffled together (prefix r)")
    s = sub.add_parser("train-sample", help="first-pass sheets of a seeded field_train sample (prefix s)")
    s.add_argument("--n", type=int, default=TRAIN_SAMPLE)
    c = sub.add_parser("collect", help="join reviewers' verdict CSVs to image IDs and splits")
    c.add_argument("--prefix", required=True, help="r, s or c: which sheets the verdicts must cover")
    c.add_argument("--reviews", nargs="+", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    sl = sub.add_parser("second-look", help="larger sheets (prefix c) of the first pass's proposed removals")
    sl.add_argument("--first", nargs="+", type=Path, required=True, help="joined first-pass files from collect")
    sl.add_argument("--also-uncertain", nargs="?", const="", default=None, metavar="REGEX",
                    help="also send uncertain rows whose note matches; bare flag uses the default in configs/e13.json")
    f = sub.add_parser("finish", help="apply both looks: log removals, write field_clean.csv and the report")
    f.add_argument("--first", nargs="+", type=Path, required=True, help="joined first-pass files (r and s)")
    f.add_argument("--second", type=Path, required=True, help="joined second-look file")
    f.add_argument("--version", type=int, required=True)
    f.add_argument("--also-uncertain", nargs="?", const="", default=None, metavar="REGEX",
                   help="the same pattern given to second-look")
    a = ap.parse_args(argv)
    field = pd.read_csv(FIELD_CSV)
    if a.cmd == "sheets":
        side = make_sheets(shuffled_rows(field))
        print(f"{side['sheet'].nunique()} sheets, {len(side)} photos in {SHEETS}")
    elif a.cmd == "train-sample":
        side = make_sheets(train_sample(field, a.n), prefix="s")
        print(f"{side['sheet'].nunique()} sheets, {len(side)} photos in {SHEETS}")
    elif a.cmd == "collect":
        joined = collect(a.reviews, _sidecars(a.prefix, SHEETS), field)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(a.out, joined.to_csv(index=False))
        print(joined["verdict"].value_counts().to_dict())
    elif a.cmd == "second-look":
        first = pd.concat([_read_csv(p) for p in a.first], ignore_index=True)
        asked = proposed(first, _pattern(a.also_uncertain))
        ids = with_ids(field).set_index("image_id")
        rows = ids.loc[asked["image_id"]].reset_index().sample(frac=1, random_state=SEED).reset_index(drop=True)
        side = make_second_look(rows)
        print(f"{asked['verdict'].value_counts().to_dict()}: {side['sheet'].nunique()} sheets, {len(side)} photos "
              f"in {SHEETS}")
    elif a.cmd == "finish":
        first = pd.concat([_read_csv(p) for p in a.first], ignore_index=True)
        second = _read_csv(a.second)
        removals, unsure, overturned = decide(first, second, _pattern(a.also_uncertain))
        append_log(EXCLUSIONS, log_rows(removals, a.version), field)
        append_log(UNCERTAIN, log_rows(unsure, a.version), field, need_reason=False)
        write_report(first, read_log(EXCLUSIONS), read_log(UNCERTAIN), second_outcomes(first, second))
        print(f"removed {len(removals)}, uncertain {len(unsure)}, overturned {len(overturned)}; "
              f"clean hash {write_clean()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
