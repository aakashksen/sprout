"""E14: is the local faithfulness judge reliable?

Sonnet grades, claim by claim, a seeded sample of the answers the local judge (`gemma4`) graded in E7. It gets the same
question, passages and rubric as the local judge and is built from the blind file only, which has no grade, verdict,
judge name or system name. Its grades measure the judge; they set no E7 number. Only rag_dev answers are used.

Stages (see docs/PLAN.md T6.6):
    plan    draw the sample and print how many Sonnet calls grading needs; makes no call
    grade   make the calls, saving each reply as it arrives; a rerun skips replies already saved
    record  write the run record and print the report; needs every claim graded and a clean tree
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sprout.atomic import atomic_write_text
from sprout.eval import agreement
from sprout.eval.records import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    ROOT,
    RUNS_REL,
    split_info,
    write_run,
)
from sprout.experiments.e7_rag import SPLIT, Checkpoint, superseded_records
from sprout.llm.client import ClaudeCodeClient, LLMClient, LLMError
from sprout.rag.judge import Claim, blind_view, judge_claim, rubric
from sprout.rag.retrieve import Passage

WORK_REL = Path("results/e14/work")
IN_FLIGHT = 2
CALL_TIMEOUT_S = 180.0
DEVICE = "none: no GPU or local model is used; Sonnet runs through the claude command and the local grades are read from a saved file"
CAVEAT = (
    "These are grades of the judge, not E7 numbers. Sonnet's grades measure how far the local judge can be trusted; "
    "they set no E7 result, and Claude is still not the judge of E7."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["split"] != SPLIT:
        raise ValueError(f"E14 grades {SPLIT} answers only, not {cfg['split']!r}")
    if cfg["sample_size"] < 200:
        raise ValueError("the plan needs at least 200 answers")
    return cfg


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def item_of(system: str, qid: str) -> str:
    return blind_view({"system": system, "id": qid, "claims": []})["item"]


def load_source(cfg: dict[str, Any], root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """(judged rows with the local grades, blind rows by item). Refuses a source that is not the final E7 record's,
    belongs to a superseded record, holds anything outside rag_dev, or whose blind file is not the judged file."""
    src = cfg["source"]
    runs = root / RUNS_REL
    judged_path, blind_path = runs / src["judgments"], runs / src["blind"]
    for path, key in ((judged_path, "judgments_sha256"), (blind_path, "blind_sha256")):
        if not path.is_file():
            raise ValueError(f"{path.name} is missing")
        if sha256(path) != src[key]:
            raise ValueError(f"{path.name} differs from the file the config names (sha256)")
    record = json.loads((runs / f"{src['e7_run_id']}.json").read_text(encoding="utf-8"))
    if record["config"].get("judgments_file") != src["judgments"] or record["config"].get("blind_file") != src["blind"]:
        raise ValueError(f"{src['e7_run_id']} did not produce these judgment files")
    old = superseded_records(root)
    if src["e7_run_id"] in old.get("superseded", {}) or src["e7_run_id"] in old.get("needs_rerun", {}):
        raise ValueError(f"{src['e7_run_id']} is superseded or needs a rerun")
    dev = set(split_info(SPLIT, root).ids)
    judged = _read(judged_path)
    outside = sorted({r["id"] for r in judged} - dev)
    if outside:
        raise ValueError(f"answers to questions outside rag_dev ({outside[:3]}) cannot be sampled")
    blind = {r["item"]: r for r in _read(blind_path)}
    for r in judged:
        b = blind.get(item_of(r["system"], r["id"]))
        if b is None or b["answer"] != r["answer"] or [c["claim"] for c in b["claims"]] != [c["claim"] for c in r["claims"]]:
            raise ValueError(f"the blind file does not match the judged answer {r['system']}/{r['id']}")
    return judged, blind


def draw_sample(judged: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Every answer of the served system, plus a seeded random draw from the rest to reach `sample_size`."""
    served = sorted((r for r in judged if r["system"] == cfg["served_system"]), key=lambda r: r["id"])
    rest = sorted((r for r in judged if r["system"] != cfg["served_system"]), key=lambda r: (r["system"], r["id"]))
    want = cfg["sample_size"] - len(served)
    if want < 0 or want > len(rest):
        raise ValueError(f"cannot draw {want} other answers from {len(rest)}")
    chosen = random.Random(cfg["seed"]).sample(rest, want)
    out = [{"item": item_of(r["system"], r["id"]), "id": r["id"], "system": r["system"],
            "stratum": "served" if r in served else "other"} for r in served + chosen]
    return sorted(out, key=lambda e: e["item"])


# ---------------------------------------------------------------- what Sonnet is shown

def clean(text: str) -> str:
    """Text safe to send as a prompt and to save as JSON: no control characters or lone surrogates."""
    text = text.encode("utf-8", "replace").decode("utf-8")
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def parse_verdict(reply: str) -> bool | None:
    """True for SUPPORTED, False for NOT_SUPPORTED, None for anything else. Stricter than the local judge's reader:
    a long reply is not silently counted as 'not supported'."""
    words = " ".join(re.findall(r"[A-Za-z]+", reply)).upper()
    return {"SUPPORTED": True, "NOT SUPPORTED": False}.get(words)


def grade_claim(llm: LLMClient, blind_row: dict[str, Any], j: int) -> dict[str, Any]:
    """One Sonnet call for claim j of a blind row, through the rubric code the local judge used."""
    c = blind_row["claims"][j]
    passages = [Passage(clean(p["text"]), p["source"], p["page"], "", 0.0) for p in blind_row["passages"]]
    _, reply = judge_claim(llm, Claim(clean(c["claim"]), list(c["cited"])), passages, CALL_TIMEOUT_S)
    return {"id": f"{blind_row['item']}:{j}", "reply": reply, "sonnet": parse_verdict(reply)}


# ---------------------------------------------------------------- grading

def run_key(cfg: dict[str, Any], model: str) -> str:
    parts = [cfg["seed"], cfg["sample_size"], cfg["served_system"], cfg["source"]["judgments_sha256"], model, rubric()]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:10]


def _pending(sample: list[dict], blind: dict[str, dict], done: dict[str, dict]) -> list[tuple[str, int]]:
    return [(e["item"], j) for e in sample for j in range(len(blind[e["item"]]["claims"]))
            if f"{e['item']}:{j}" not in done]


def check_budget(llm: LLMClient, need: int, limit: int, spent: int) -> None:
    """Refuse, before any call, a run that would pass the experiment's limit or the shared Claude call cap."""
    if spent + need > limit:
        raise LLMError(f"E14 call limit {limit} would be passed ({spent} made, {need} needed); no call was made")
    made, cap = getattr(llm, "calls_made", None), getattr(llm, "cap", None)
    if callable(made) and cap is not None and made() + need > cap:
        raise LLMError(f"Claude call cap {cap} would be passed ({made()} made, {need} needed); no call was made")


def grade(sample: list[dict], blind: dict[str, dict], llm: LLMClient, work: Path, cfg: dict[str, Any]) -> Checkpoint:
    """Grade every claim not already saved, two calls at a time, saving each reply as it arrives. An error stops new
    calls, keeps what arrived, and is raised."""
    ck = Checkpoint(work / f"claims-{run_key(cfg, llm.model)}.jsonl")
    todo = _pending(sample, blind, ck.rows)
    check_budget(llm, len(todo), cfg["max_calls"], len(ck.rows))
    failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
        queue, live = iter(todo), set()
        while True:
            while failure is None and len(live) < IN_FLIGHT:
                nxt = next(queue, None)
                if nxt is None:
                    break
                live.add(pool.submit(grade_claim, llm, blind[nxt[0]], nxt[1]))
            if not live:
                break
            done, live = wait(live, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    ck.add(f.result())
                except LLMError as exc:
                    failure = failure or exc
    if failure is not None:
        raise failure
    return ck


# ---------------------------------------------------------------- the record

def build_rows(sample: list[dict], judged: list[dict], blind: dict[str, dict], ck: Checkpoint) -> list[dict[str, Any]]:
    by_item = {item_of(r["system"], r["id"]): r for r in judged}
    rows, missing = [], 0
    for e in sample:
        j, n = by_item[e["item"]], len(blind[e["item"]]["claims"])
        got = [ck.rows.get(f"{e['item']}:{k}") for k in range(n)]
        missing += sum(g is None for g in got)
        if None not in got:
            rows.append({"id": e["id"], "item": e["item"], "system": e["system"],
                         "claims": [c["claim"] for c in j["claims"]], "local": [bool(c["verdict"]) for c in j["claims"]],
                         "sonnet": [g["sonnet"] for g in got], "sonnet_reply": [g["reply"] for g in got], "calls": n})
    if missing:
        raise ValueError(f"{missing} claims are not graded yet; run the grade stage")
    return rows


def report(rows: list[dict], metrics: dict, intervals: dict, served: str) -> str:
    def f(name: str) -> str:
        v, ci = metrics.get(name), intervals.get(name)
        if v is None:
            return "undefined"
        return f"{v:.3f}" + (f" (95% {ci[0]:.3f} to {ci[1]:.3f})" if ci else "")

    lines = [CAVEAT, f"answers {len(rows)}, claims compared {int(metrics['claims_compared'])}, "
                     f"claims without a usable Sonnet verdict {int(metrics['claims_not_graded'])}"]
    for tag, p in (("all sampled", ""), (f"served ({served})", agreement.SERVED)):
        lines += [f"{tag}: agreement {f(p + 'agreement')}; kappa {f(p + 'kappa')}",
                  f"  supported share, local judge {f(p + 'local_supported')}, Sonnet {f(p + 'sonnet_supported')}",
                  f"  disagreements: local supported but Sonnet not {int(metrics[p + 'local_kinder'])}, "
                  f"Sonnet supported but local not {int(metrics[p + 'local_stricter'])}"]
    lines.append(f"served answers with at least one unsupported claim: local judge "
                 f"{int(metrics['served_answers_with_unsupported_local'])} of {int(metrics['served_answers'])}, "
                 f"Sonnet {int(metrics['served_answers_with_unsupported_sonnet'])}")
    for r in rows:
        if r["system"] == served and r["id"] == "supported-031":
            for c, a, b, reply in zip(r["claims"], r["local"], r["sonnet"], r["sonnet_reply"], strict=True):
                lines.append(f"supported-031 claim: local {a}, Sonnet {b} ({reply!r}): {c}")
    return "\n".join(lines)


def write_record(cfg: dict[str, Any], rows: list[dict], root: Path, started: str, note: str = "") -> Path:
    info = split_info(SPLIT, root)
    served = cfg["served_system"]
    metrics = agreement.metrics_from_rows(rows, served)
    intervals = agreement.intervals_from_rows(rows, served, BOOTSTRAP_N, BOOTSTRAP_SEED)
    calls = sum(r["calls"] for r in rows)
    config = {**{k: v for k, v in cfg.items() if k != "device"}, "device": DEVICE, "claude_calls": calls,
              "judge": {"model": cfg["judge_model"]}, "started_at": started, "labels_reviewed": False,
              "rubric": rubric(), "caveat": CAVEAT}
    attempts = [{"stage": "grade", "answers": len(rows), "claude_calls": calls,
                 "replies_without_a_verdict": int(metrics["claims_not_graded"])}]
    return write_run("E14", SPLIT, config, {"judge_under_test": cfg["judge_model"], "grader": cfg["grader_model"]},
                     {"sample": metrics}, {"sample": intervals}, attempts, {"sample": rows}, info.hash, root=root,
                     notes=CAVEAT + (" " + note if note else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "grade", "record"])
    ap.add_argument("--config", default="configs/e14.json")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        judged, blind = load_source(cfg, ROOT)
        sample = draw_sample(judged, cfg)
        work = ROOT / WORK_REL
        llm = ClaudeCodeClient(model=cfg["grader_model"])
        if a.stage == "plan":
            work.mkdir(parents=True, exist_ok=True)
            atomic_write_text(work / f"sample-{run_key(cfg, llm.model)}.json", json.dumps(sample, indent=1) + "\n")
            claims = sum(len(blind[e["item"]]["claims"]) for e in sample)
            print(f"{len(sample)} answers ({sum(e['stratum'] == 'served' for e in sample)} served), {claims} claims = "
                  f"{claims} Sonnet calls; limit {cfg['max_calls']}; calls counted so far {llm.calls_made()} of {llm.cap}")
        elif a.stage == "grade":
            t = time.time()
            ck = grade(sample, blind, llm, work, cfg)
            print(f"graded; {len(ck.rows)} claim replies saved in {time.time() - t:.0f}s this session")
        else:
            ck = Checkpoint(work / f"claims-{run_key(cfg, llm.model)}.jsonl")
            rows = build_rows(sample, judged, blind, ck)
            path = write_record(cfg, rows, ROOT, datetime.now(timezone.utc).isoformat(), a.note)
            rec = json.loads(path.read_text(encoding="utf-8"))
            print(path)
            print(report(rows, rec["metrics"]["sample"], rec["intervals"]["sample"], cfg["served_system"]))
    except (LLMError, RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
