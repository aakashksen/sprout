"""Check run records against docs/TESTING.md: every saved number must recompute from saved files."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sprout.eval import metrics as M
from sprout.eval.records import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    ROOT,
    RUNS_REL,
    SplitInfo,
    git,
    split_info,
    subset_positions,
)
from sprout.eval.stats import (
    bootstrap_ci,
    macro_f1_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_macro_f1_diff,
    percentile,
)

REQUIRED = (
    "run_id experiment git_commit dirty dataset_hash split subset config models n metrics measurements judged "
    "intervals bootstrap per_class confusion comparisons attempts claude_calls labels_reviewed predictions started_at finished_at notes"
).split()
TOLERANCE = 1e-9
NOT_PLANT = "not_plant"
SIDECAR = re.compile(r".+\.(measurements|judged)\.json")
FAKE = re.compile(r"fake|stub", re.I)
TIMESTAMP = re.compile(r"\d{8}T\d{12}Z")
THRESHOLDED = ("coverage", "selective_accuracy", "confident_error_rate")


@dataclass
class Scored:
    """One system's saved rows, aligned to the split (or declared subset) in manifest order."""

    groups: list[str]
    labels: np.ndarray | None = None
    probs: np.ndarray | None = None
    ranks: list[int | None] | None = None
    accepted: np.ndarray | None = None  # gate decision per image
    is_plant: np.ndarray | None = None  # from the manifest's labels
    answered: np.ndarray | None = None  # RAG: answered rather than declined
    expected: list[str] | None = None  # RAG gold behaviour from the questions file


def _num(x: float) -> float | None:
    return None if x != x else float(x)


def item_series(name: str, s: Scored) -> np.ndarray | None:
    """Per-item values whose mean is the metric, for metrics that have an interval or comparison."""
    if s.probs is not None and s.labels is not None:
        correct = (s.probs.argmax(axis=1) == s.labels).astype(float)
        if name == "accuracy":
            return correct
        m = re.fullmatch(r"coverage@([\d.]+)", name)
        if m:
            return (s.probs.max(axis=1) >= float(m.group(1))).astype(float)
    if name == "end_to_end_accuracy" and s.probs is not None and s.labels is not None and s.accepted is not None:
        return ((s.probs.argmax(axis=1) == s.labels) & s.accepted).astype(float)
    if s.ranks is not None:
        m = re.fullmatch(r"recall@(\d+)", name)
        if m:
            return np.array([r is not None and r <= int(m.group(1)) for r in s.ranks], dtype=float)
        if name == "mrr":
            return np.array([0.0 if r is None else 1.0 / r for r in s.ranks])
    return None


def _rate(flags: np.ndarray, among: np.ndarray) -> float | None:
    return float(flags[among].mean()) if among.any() else None


def recompute_metric(name: str, s: Scored, others: dict[str, Scored] | None = None) -> float | None:
    """The metric from saved rows, using sprout.eval.metrics. None when it is undefined for these rows
    (nothing to divide by, or the rows lack the needed fields). Raises KeyError for an unknown metric."""
    if name == "plant_rejection_rate" and s.accepted is not None and s.is_plant is not None:
        return _rate(~s.accepted, s.is_plant)
    if name == "not_plant_acceptance_rate" and s.accepted is not None and s.is_plant is not None:
        return _rate(s.accepted, ~s.is_plant)
    if name in ("answers_when_answerable", "declines_when_unanswerable") and s.answered is not None:
        gold = np.array(s.expected or [])
        if name == "answers_when_answerable":
            return _rate(s.answered, gold == "answer")
        return _rate(~s.answered, gold == "abstain")
    changed = re.fullmatch(r"changed_vs@([A-Za-z0-9_-]+)", name)
    if changed:
        other = (others or {}).get(changed.group(1))
        if other is None:
            raise KeyError(name)
        if s.probs is None or other.probs is None:
            return None
        return float((s.probs.argmax(axis=1) != other.probs.argmax(axis=1)).mean())
    series = item_series(name, s)
    if series is not None:
        return float(series.mean())
    if s.probs is not None and s.labels is not None:
        if name == "macro_f1":
            return M.macro_f1(s.labels, s.probs.argmax(axis=1))
        if name == "ece":
            return M.ece(s.probs, s.labels)
        m = re.fullmatch(r"(selective_accuracy|confident_error_rate)@([\d.]+)", name)
        if m:
            return _num(M.selective_metrics(s.probs, s.labels, float(m.group(2)))[m.group(1)])
    raise KeyError(name)


def recompute_interval(
    name: str, s: Scored, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float] | None:
    """95% interval resampling whole near-duplicate groups; macro_f1 is recomputed inside each resample."""
    if name == "macro_f1" and s.probs is not None and s.labels is not None:
        return macro_f1_ci(s.labels, s.probs.argmax(axis=1), s.groups, s.probs.shape[1], n, seed)
    series = item_series(name, s)
    return None if series is None else bootstrap_ci(series, s.groups, n, seed)


def recompute_comparison(
    a: Scored, b: Scored, metric: str, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any] | None:
    """System a minus system b over the same resampled groups, as saved in a record's `comparisons`.
    McNemar is always on per-image correctness (accuracy, or recall@k for retrieval)."""
    if a.groups != b.groups:
        return None
    if metric == "macro_f1":
        if a.probs is None or b.probs is None or a.labels is None or not np.array_equal(a.labels, b.labels):
            return None
        pa, pb = a.probs.argmax(axis=1), b.probs.argmax(axis=1)
        diff, lo, hi = paired_macro_f1_diff(a.labels, pa, pb, a.groups, a.probs.shape[1], n, seed)
        p = mcnemar_exact(pa == a.labels, pb == b.labels)
        return {"diff": diff, "ci": [lo, hi], "mcnemar_p": p}
    if metric != "accuracy" and not re.fullmatch(r"recall@\d+", metric):
        return None
    sa, sb = item_series(metric, a), item_series(metric, b)
    if sa is None or sb is None or len(sa) != len(sb):
        return None
    diff, lo, hi = paired_bootstrap_diff(sa, sb, a.groups, n, seed)
    return {"diff": diff, "ci": [lo, hi], "mcnemar_p": mcnemar_exact(sa > 0.5, sb > 0.5)}


def _same(fresh: float | None, saved: object) -> bool:
    if fresh is None or saved is None:
        return fresh is None and saved is None
    return isinstance(saved, (int, float)) and abs(fresh - saved) <= TOLERANCE


def _same_pair(fresh, saved: object) -> bool:
    return isinstance(saved, (list, tuple)) and len(saved) == 2 and all(_same(f, v) for f, v in zip(fresh, saved))


def _strings(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def _timestamp(name: str) -> str:
    found = TIMESTAMP.findall(name)
    return found[-1] if found else name


def records_for(experiment: str, root: Path = ROOT) -> list[Path]:
    """Every record of the experiment, oldest first."""
    pat = re.compile(rf"{re.escape(experiment)}-{TIMESTAMP.pattern}\.json")
    found = [p for p in (root / RUNS_REL).glob(f"{experiment}-*.json") if pat.fullmatch(p.name)]
    return sorted(found, key=lambda p: (_timestamp(p.name), p.name))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _score(
    rows: list[dict[str, Any]], info: SplitInfo, positions: list[int], rec: dict[str, Any], tag: str, out: list[str]
) -> Scored | None:
    """Align a system's rows to the split in manifest order; None (with problems added) when they don't fit."""
    if len({frozenset(r) for r in rows}) != 1:
        out.append(tag + "every row must have the same fields")
        return None
    key = "path" if "path" in rows[0] else "id"
    ids = [r.get(key) for r in rows]
    wanted = [info.ids[i] for i in positions]
    if len(set(ids)) != len(ids) or set(ids) != set(wanted):
        out.append(tag + "prediction ids do not match the split's ids exactly once each")
        return None
    by_id = dict(zip(ids, rows))
    ordered = [by_id[i] for i in wanted]
    scored = Scored(groups=[info.groups[i] for i in positions])
    if "accepted" in rows[0]:
        flags = [r.get("accepted") for r in ordered]
        if not all(isinstance(f, bool) for f in flags):
            out.append(tag + "accepted must be true or false in every row")
            return None
        scored.accepted = np.array(flags)
        if info.labels is not None:
            scored.is_plant = np.array([info.labels[i] != NOT_PLANT for i in positions])
    if "answered" in rows[0]:
        flags = [r.get("answered") for r in ordered]
        if not all(isinstance(f, bool) for f in flags) or info.expected is None:
            out.append(tag + "answered must be true or false in every row of a RAG split")
            return None
        scored.answered = np.array(flags)
        scored.expected = [info.expected[i] for i in positions]
    if "probs" in rows[0]:
        names = rec["config"].get("class_names") if isinstance(rec["config"], dict) else None
        probs = np.array([r["probs"] for r in ordered], dtype=float)
        labels = np.array([r["label"] for r in ordered], dtype=int)
        if not isinstance(names, list) or probs.ndim != 2 or probs.shape[1] != len(names):
            out.append(tag + "config.class_names must list the classes, one per probability column")
            return None
        if info.labels is not None and [names[x] for x in labels] != [info.labels[i] for i in positions]:
            out.append(tag + "saved labels do not match the manifest's labels")
            return None
        scored.probs, scored.labels = probs, labels
    if "rank" in rows[0]:
        scored.ranks = [r["rank"] for r in ordered]
    return scored


def _measurement_problems(block: Any, root: Path, kind: str) -> list[str]:
    """Saved summary values must recompute from the saved sample file."""
    if not block:
        return []
    path = root / block["file"] if isinstance(block, dict) and "file" in block else None
    if path is None or not path.is_file():
        return [f"{kind} file is missing"]
    samples = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for name, saved in block.get("values", {}).items():
        if kind == "judged":
            fresh = float(np.mean([bool(x) for x in samples[name]])) if name in samples and samples[name] else None
        else:
            m = re.fullmatch(r"(.+)_p(\d+)", name)
            fresh = percentile(samples[m.group(1)], int(m.group(2))) if m and m.group(1) in samples else None
        if fresh is None:
            out.append(f"{kind} value {name} has no samples in {block['file']}")
        elif not _same(fresh, saved):
            out.append(f"{kind} value {name} does not match its saved file")
    return out


def _table_problems(rec: dict[str, Any], name: str, s: Scored, tag: str) -> list[str]:
    """Per-class report and confusion matrix: saved for every classification system, and recomputed."""
    saved_pc, saved_cm = rec["per_class"].get(name), rec["confusion"].get(name)
    if s.probs is None or s.labels is None:
        return [tag + f"{k} given for rows that are not classifications" for k, v in
                (("per_class", saved_pc), ("confusion", saved_cm)) if v]
    names, pred = rec["config"]["class_names"], s.probs.argmax(axis=1)
    out = []
    fresh_pc = M.per_class_report(s.labels, pred, names)
    if not isinstance(saved_pc, dict) or set(saved_pc) != set(fresh_pc) or not all(
        set(saved_pc[c]) == set(v) and all(_same(float(v[k]), saved_pc[c][k]) for k in v)
        for c, v in fresh_pc.items()
    ):
        out.append(tag + "per_class does not recompute from predictions")
    if saved_cm != M.confusion(s.labels, pred, len(names)).tolist():
        out.append(tag + "confusion does not recompute from predictions")
    return out


def _judge_problems(config: Any, models: Any) -> list[str]:
    judge = config.get("judge") if isinstance(config, dict) else None
    model = judge.get("model") if isinstance(judge, dict) else None
    if not (isinstance(model, str) and model.strip()):
        return ["config.judge.model is missing: judged values must name their judge"]
    out = []
    if FAKE.search(model):
        out.append("the judge model matches fake or stub")
    if isinstance(models, dict) and model == models.get("writer"):
        out.append("the judge must be a different model from the writer")
    return out


def _nonfinite(x: Any, path: str = "", loss_only: bool = False) -> list[str]:
    """Paths of NaN or infinite numbers inside nested JSON values (json.load accepts NaN and Infinity)."""
    if isinstance(x, float):
        return [] if math.isfinite(x) or (loss_only and "loss" not in path.rsplit(".", 1)[-1]) else [path]
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in _nonfinite(v, f"{path}.{k}" if path else str(k), loss_only)]
    if isinstance(x, list):
        return [p for i, v in enumerate(x) for p in _nonfinite(v, f"{path}[{i}]" if path else f"[{i}]", loss_only)]
    return []


def _finite_problems(rec: dict[str, Any]) -> list[str]:
    out = []
    for key in ("metrics", "intervals", "comparisons"):
        bad = _nonfinite(rec.get(key), key)
        if bad:
            out.append(f"non-finite value in {key}: {', '.join(bad[:3])}")
    bad = _nonfinite(rec.get("attempts"), "attempts", loss_only=True)
    if bad:
        out.append(f"attempts history holds a non-finite loss: {', '.join(bad[:3])}")
    return out


def _stamp_problems(rec: dict[str, Any]) -> list[str]:
    """finished_at must be the write time that run_id carries."""
    found = TIMESTAMP.findall(str(rec.get("run_id")))
    try:
        stamp = datetime.strptime(found[-1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        ok = datetime.fromisoformat(rec["finished_at"]) == stamp
    except (IndexError, ValueError, TypeError):
        ok = False
    return [] if ok else ["finished_at disagrees with the timestamp in run_id"]


def check_record(rec: dict[str, Any], root: Path, min_n: int = 0) -> list[str]:
    problems = [f"missing field {k}" for k in REQUIRED if k not in rec]
    if problems:
        return problems
    if rec["dirty"]:
        problems.append("record was made on a dirty tree")
    problems += _stamp_problems(rec)
    problems += _finite_problems(rec)
    if any(FAKE.search(s) for s in _strings(rec["models"])):
        problems.append("a model name matches fake or stub")
    if not rec["attempts"]:
        problems.append("attempts is empty")
    if git(root, "merge-base", "--is-ancestor", rec["git_commit"], "HEAD").returncode != 0:
        problems.append("git_commit is not an ancestor of HEAD")
    if min_n and rec["n"] < min_n:
        problems.append(f"n={rec['n']} is below --min-n {min_n}")

    try:
        info = split_info(rec["split"], root)
    except (ValueError, OSError) as exc:
        return problems + [f"cannot read split {rec['split']}: {exc}"]
    if rec["dataset_hash"] != info.hash:
        problems.append("dataset_hash does not match the manifest")
    positions = list(range(len(info.ids)))
    if rec["subset"] is not None:
        try:
            positions = subset_positions(rec["subset"], rec["split"], info, root)
        except ValueError as exc:
            return problems + [str(exc)]
    if not positions or rec["n"] == 0:
        problems.append("no scored rows: the split or subset is empty")
    what = f"subset {rec['subset']}" if rec["subset"] else "split"
    if rec["n"] != len(positions):
        problems.append(f"n={rec['n']} differs from the {what}'s {len(positions)} items")

    paths, metrics, intervals = rec["predictions"], rec["metrics"], rec["intervals"]
    if not isinstance(paths, dict) or not paths:
        return problems + ["predictions must map each system to its predictions file"]
    if not isinstance(metrics, dict) or set(metrics) != set(paths):
        return problems + ["metrics must hold one entry per system, matching predictions"]
    if not set(intervals) <= set(paths):
        problems.append(f"intervals name systems not in predictions: {sorted(set(intervals) - set(paths))}")
        intervals = {}

    boot = rec["bootstrap"]
    if not (isinstance(boot, dict) and isinstance(boot.get("n"), int) and isinstance(boot.get("seed"), int)):
        return problems + ["bootstrap must record n and seed"]
    if boot["n"] != BOOTSTRAP_N:
        problems.append(f"bootstrap n={boot['n']}, but intervals need {BOOTSTRAP_N:,} resamples")
    n_boot, seed = boot["n"], boot["seed"]

    usable: dict[str, Scored] = {}
    for name, rel in paths.items():
        tag = f"system {name}: "
        if not (root / rel).is_file():
            problems.append(tag + "predictions file is missing")
            continue
        rows = _read_rows(root / rel)
        if len(rows) != rec["n"]:
            problems.append(tag + f"predictions hold {len(rows)} rows but n={rec['n']}")
        if not metrics[name] and rec["split"] != "perf":
            problems.append(tag + "metrics is empty")
        if not rows:
            problems.append(tag + "no scored rows: the predictions file is empty")
            continue
        scored = _score(rows, info, positions, rec, tag, problems)
        if scored is None:
            continue
        usable[name] = scored

    for name in set(paths) - set(usable):
        if f"system {name}: " not in " ".join(problems):
            problems.append(f"system {name}: was not scored")

    for name, scored in usable.items():
        tag = f"system {name}: "
        for metric, saved in metrics[name].items():
            try:
                fresh = recompute_metric(metric, scored, usable)
            except KeyError:
                problems.append(tag + f"metric {metric} cannot be recomputed from predictions")
                continue
            if not _same(fresh, saved):
                problems.append(tag + f"metric {metric} does not recompute from predictions")
        for metric, saved in (intervals.get(name) or {}).items():
            got = recompute_interval(metric, scored, n_boot, seed)
            if got is None:
                problems.append(tag + f"interval {metric} cannot be recomputed from predictions")
            elif not _same_pair(got, saved):
                problems.append(tag + f"interval {metric} does not recompute from predictions")
        problems += _table_problems(rec, name, scored, tag)

    for i, c in enumerate(rec["comparisons"]):
        tag = f"comparison {i}: "
        if not (isinstance(c, dict) and {"a", "b", "metric", "diff", "ci", "mcnemar_p"} <= set(c)):
            problems.append(tag + "needs a, b, metric, diff, ci and mcnemar_p")
        elif c["a"] == c["b"] or c["a"] not in usable or c["b"] not in usable:
            problems.append(tag + "a and b must be two different systems with usable predictions")
        else:
            fresh = recompute_comparison(usable[c["a"]], usable[c["b"]], c["metric"], n_boot, seed)
            if fresh is None:
                problems.append(tag + f"{c['metric']} cannot be recomputed for a comparison")
            elif not (_same(fresh["diff"], c["diff"]) and _same_pair(fresh["ci"], c["ci"])
                      and _same(fresh["mcnemar_p"], c["mcnemar_p"])):
                problems.append(tag + f"{c['metric']} for {c['a']} vs {c['b']} does not recompute from predictions")

    problems += _measurement_problems(rec["measurements"], root, "measurements")
    problems += _measurement_problems(rec["judged"], root, "judged")
    if rec["judged"]:
        problems += _judge_problems(rec["config"], rec["models"])
    return problems


def _load_records(experiment: str, root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in records_for(experiment, root)]


def _row_map(root: Path, rel: str) -> dict[str, str]:
    rows = _read_rows(root / rel) if (root / rel).is_file() else []
    return {str(r.get("path", r.get("id"))): json.dumps(r, sort_keys=True) for r in rows}


def _is_view_of(view: dict[str, Any], full: dict[str, Any], root: Path) -> bool:
    """A declared subset drawn from the same commit, models, config and predictions as a full record."""
    if view.get("subset") is None or full.get("subset") is not None:
        return False
    drop = lambda c: {k: v for k, v in (c or {}).items() if k != "override_note"}  # noqa: E731
    if any(view.get(k) != full.get(k) for k in ("git_commit", "models", "split")) or drop(view.get("config")) != drop(
        full.get("config")
    ):
        return False
    for system, rel in (view.get("predictions") or {}).items():
        if system not in (full.get("predictions") or {}):
            return False
        theirs = _row_map(root, full["predictions"][system])
        if not all(theirs.get(k) == v for k, v in _row_map(root, rel).items()):
            return False
    return True


def _valid_note(rec: dict[str, Any]) -> bool:
    note = (rec.get("config") or {}).get("override_note")
    return isinstance(note, str) and bool(note.strip())


def _history(root: Path) -> list[tuple[str, str]]:
    """(commit, path) for every file ever added under results/runs, oldest first. Raises when git can't say."""
    out = git(root, "log", "--reverse", "--no-renames", "--diff-filter=A", "--name-only",
              "--format=commit %H", "--", RUNS_REL.as_posix())
    if out.returncode != 0:
        raise RuntimeError(f"git log failed: {out.stderr.strip()}")
    pairs, commit = [], ""
    for line in out.stdout.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.strip():
            pairs.append((commit, line.strip()))
    return pairs


def _uses(root: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every test-split record that has ever existed (committed history plus working tree), in order of use:
    committed ones by the commit that added them, then uncommitted ones by the timestamp in their name."""
    recs: dict[str, tuple[tuple, dict[str, Any]]] = {}
    for index, (commit, rel) in enumerate(_history(root)):
        name = rel.rsplit("/", 1)[-1]
        if not name.endswith(".json") or SIDECAR.fullmatch(name):
            continue
        if name[:-5] in recs:
            continue  # the first add decides the order
        path = root / rel
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            shown = git(root, "show", f"{commit}:{rel}")
            if shown.returncode != 0:
                raise RuntimeError(f"git show failed for {rel}")
            text = shown.stdout
        recs[name[:-5]] = ((0, index, _timestamp(name)), json.loads(text))
    for p in (root / RUNS_REL).glob("*.json"):
        if p.stem not in recs and not SIDECAR.fullmatch(p.name):
            recs[p.stem] = ((1, 0, _timestamp(p.name)), json.loads(p.read_text(encoding="utf-8")))
    return [(stem, rec) for stem, (_, rec) in sorted(recs.items(), key=lambda kv: (kv[1][0], kv[0]))
            if isinstance(rec, dict) and str(rec.get("split", "")).endswith("_test")]


def test_once_problems(root: Path) -> list[str]:
    """A test split is scored once. Every record that has ever existed on it counts, whatever its file name or
    experiment, deleted or not; a later one needs a non-empty string config.override_note. A declared subset
    of a full record's own predictions (same commit, models, config) is a view of that use, not another."""
    by_split: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for stem, rec in _uses(root):
        by_split.setdefault(rec["split"], []).append((stem, rec))
    out = []
    for split, recs in by_split.items():
        uses = [(s, r) for s, r in recs if not any(_is_view_of(r, f, root) for t, f in recs if t != s)]
        out += [f"{stem}: more than one {split} record and no override_note in config"
                for stem, r in uses[1:] if not _valid_note(r)]
    return out


def check_file(path: Path, root: Path, min_n: int = 0, once: list[str] | None = None) -> list[str]:
    """Every problem with one record file. check_run and claims both call this, so they agree."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if rec.get("run_id") != path.stem or not TIMESTAMP.search(path.stem):
        problems.append("the record's run_id must equal its file name and end in a write timestamp")
    problems += check_record(rec, root, min_n)
    once = test_once_problems(root) if once is None else once
    return problems + [m.split(": ", 1)[1] for m in once if m.startswith(f"{path.stem}: ")]


def check(experiment: str, *, root: Path = ROOT, min_n: int = 0, split: str | None = None) -> list[str]:
    recs = _load_records(experiment, root)
    if split:
        recs = [(p, r) for p, r in recs if r.get("split") == split]
    if not recs:
        return [f"no run record for {experiment}"]
    once = test_once_problems(root)
    return [f"{p.name}: {m}" for p, _ in recs for m in check_file(p, root, min_n, once)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--split")
    a = ap.parse_args(argv)
    problems = check(a.experiment, min_n=a.min_n, split=a.split)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"{a.experiment} ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
