"""E14H: can Haiku take the local judge's place? The rule is in docs/DECISIONS.md, "T8.1 - Haiku takes the local
judge's place", and was fixed before any Haiku grade.

Haiku grades, one claim per call and through E14's rubric code, a seeded random draw of E14's saved claims (each already
holding the local judge's and Sonnet's grade) and, as a separate set, the 49 claims a reader called in
docs/e14-adjudication.md. Haiku is clearly worse only if the paired 95% interval for
(kappa(Haiku, Sonnet) - kappa(local judge, Sonnet)), resampling answers, lies entirely below zero. A reply that is not a
verdict counts as a disagreement with Sonnet.

Stages:
    plan    draw the sample and print how many Haiku calls grading needs; makes no call
    grade   make the calls with several in flight, saving each reply as it arrives; a rerun skips saved replies
    record  write the run record and results/e14/haiku_decision.json; needs every claim graded and a clean tree
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
from sprout.experiments import e14_judge_check as e14
from sprout.experiments.e7_rag import Checkpoint
from sprout.llm.client import ClaudeCodeClient, LLMClient, LLMError
from sprout.rag.judge import rubric

EXPERIMENT = "E14H"
WORK_REL = Path("results/e14/work")
DECISION_REL = Path("results/e14/haiku_decision.json")
BOTH, JUDGE_ONLY, SONNET_ONLY, NEITHER = agreement.BOTH, agreement.LOCAL_ONLY, agreement.OTHER_ONLY, agreement.NEITHER
JUDGES = ("haiku", "local")
CLAIM_KEYS = ("item", "qid", "system", "claim", "local", "sonnet")
CAVEAT = (
    "Sonnet is a reference and not truth, and Haiku and Sonnet come from one family, so agreement may flatter Haiku. "
    "Passing shows Haiku is not clearly worse at matching Sonnet, not that it is right. Neither figure is accuracy."
)


def rubric_hash() -> str:
    return hashlib.sha256(rubric().encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["split"] != e14.SPLIT:
        raise ValueError(f"E14H grades {e14.SPLIT} claims only, not {cfg['split']!r}")
    return cfg


def load_e14_config(cfg: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / cfg["e14_config"]).read_text(encoding="utf-8"))


def load_claims(cfg: dict[str, Any], root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Every claim in E14's record, by `<item>:<j>`, with the local and Sonnet grades the record holds."""
    rec = json.loads((root / RUNS_REL / f"{cfg['e14_run_id']}.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (root / rec["predictions"]["sample"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = {}
    for r in rows:
        for j, (text, local, sonnet) in enumerate(zip(r["claims"], r["local"], r["sonnet"], strict=True)):
            if sonnet is None:
                raise ValueError(f"{r['item']}:{j} has no Sonnet verdict, so it has nothing to be compared with")
            out[f"{r['item']}:{j}"] = {"item": r["item"], "qid": r["id"], "system": r["system"], "claim": text,
                                       "local": bool(local), "sonnet": bool(sonnet)}
    return out


def load_reads(cfg: dict[str, Any], root: Path = ROOT) -> list[dict[str, Any]]:
    return json.loads((root / cfg["reads"]).read_text(encoding="utf-8"))


def draw_sample(claims: dict[str, dict[str, Any]], seed: int, n: int) -> list[str]:
    if n > len(claims):
        raise ValueError(f"cannot draw {n} claims from {len(claims)}")
    return sorted(random.Random(seed).sample(sorted(claims), n))


def tasks_for(cfg: dict[str, Any], claims: dict[str, dict], reads: list[dict]) -> list[tuple[str, str, int]]:
    """(checkpoint key, blind item, claim index). The 49 read claims are graded apart from the sample, so a claim in
    both is asked twice."""
    out = []
    for cid in draw_sample(claims, cfg["seed"], cfg["sample_size"]):
        item, j = cid.rsplit(":", 1)
        out.append((f"sample|{cid}", item, int(j)))
    out += [(f"read|{r['tag']}", r["item"], r["k"]) for r in reads]
    return out


# ---------------------------------------------------------------- grading

def checkpoint_path(cfg: dict[str, Any], work: Path) -> Path:
    key = hashlib.sha1(json.dumps([cfg["seed"], cfg["sample_size"], cfg["model"], rubric_hash(), cfg["e14_run_id"]]).encode())
    return work / f"haiku-{key.hexdigest()[:10]}.jsonl"


def grade_task(llm: LLMClient, blind_row: dict[str, Any], j: int, key: str) -> dict[str, Any]:
    got = e14.grade_claim(llm, blind_row, j)
    return {"id": key, "reply": got["reply"], "haiku": got["sonnet"]}


def grade(tasks: list[tuple[str, str, int]], blind: dict[str, dict], llm: LLMClient, ck: Checkpoint, cfg: dict[str, Any]) -> None:
    """Grade every task not already saved, `workers` at a time, saving each reply as it arrives. An error stops new
    calls, keeps what arrived, and is raised."""
    todo = [t for t in tasks if t[0] not in ck.rows]
    e14.check_budget(llm, len(todo), cfg["max_calls"], len(ck.rows))
    workers, failure = cfg["workers"], None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue, live = iter(todo), set()
        while True:
            while failure is None and len(live) < workers:
                nxt = next(queue, None)
                if nxt is None:
                    break
                live.add(pool.submit(grade_task, llm, blind[nxt[1]], nxt[2], nxt[0]))
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


# ---------------------------------------------------------------- numbers

def effective(haiku: bool | None, sonnet: bool) -> bool:
    """An unreadable reply is a disagreement with Sonnet, so it can't help Haiku."""
    return (not sonnet) if haiku is None else haiku


def _counts(rows: list[dict], judge: str) -> np.ndarray:
    pick = (lambda r: effective(r["haiku"], r["sonnet"])) if judge == "haiku" else (lambda r: r["local"])
    return agreement._counts([pick(r) for r in rows], [r["sonnet"] for r in rows])


def _num(x: float) -> float | None:
    return None if x != x else float(x)


def _stats(c: np.ndarray) -> dict[str, Any]:
    return {"agreement": agreement.STATS["agreement"](c), "kappa": agreement.kappa(c),
            "supported": agreement.STATS["local_supported"](c)}


def sample_metrics(rows: list[dict], served: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for prefix, part in (("", rows), ("served_", [r for r in rows if r["system"] == served])):
        if not part:
            continue
        counts = {j: _counts(part, j) for j in JUDGES}
        for j in JUDGES:
            for name, v in _stats(counts[j]).items():
                out[f"{prefix}{j}_{name}"] = _num(v)
            out[f"{prefix}{j}_kinder"] = float(counts[j][JUDGE_ONLY])  # supported by the judge, not by Sonnet
            out[f"{prefix}{j}_stricter"] = float(counts[j][SONNET_ONLY])
        out[prefix + "sonnet_supported"] = _num(agreement.STATS["sonnet_supported"](counts["haiku"]))
        out[prefix + "claims"] = float(len(part))
        out[prefix + "answers"] = float(len({r["item"] for r in part}))
        out[prefix + "parse_failures"] = float(sum(r["haiku"] is None for r in part))
        k = [out[f"{prefix}{j}_kappa"] for j in JUDGES]
        out[prefix + "kappa_difference"] = None if None in k else k[0] - k[1]
    return out


def sample_intervals(rows: list[dict], served: str, n: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% bootstrap intervals resampling whole answers, with Haiku and the local judge scored on the same resample,
    so the difference of their kappas is paired. A statistic undefined in some resamples is taken over the rest."""
    out = {}
    for prefix, part in (("", rows), ("served_", [r for r in rows if r["system"] == served])):
        if not part:
            continue
        items = sorted({r["item"] for r in part})
        index = {it: i for i, it in enumerate(items)}
        per = np.zeros((len(items), 2, 4))
        for r in part:
            for k, j in enumerate(JUDGES):
                per[index[r["item"]], k] += _counts([r], j)
        rng = np.random.default_rng(seed)
        pooled = np.empty((n, 2, 4))
        for start in range(0, n, 1000):
            m = min(1000, n - start)
            pooled[start : start + m] = per[rng.integers(0, len(items), size=(m, len(items)))].sum(axis=1)
        series = {}
        for k, j in enumerate(JUDGES):
            for name, v in _stats(pooled[:, k, :]).items():
                series[f"{j}_{name}"] = np.asarray(v)
        series["sonnet_supported"] = np.asarray(agreement.STATS["sonnet_supported"](pooled[:, 0, :]))
        series["kappa_difference"] = series["haiku_kappa"] - series["local_kappa"]
        for name, values in series.items():
            if np.isfinite(values).any():
                lo, hi = np.nanpercentile(values, [2.5, 97.5])
                out[prefix + name] = (float(lo), float(hi))
    return out


def decision(intervals: dict[str, tuple[float, float]]) -> str:
    """'local' only when the paired interval lies entirely below zero; otherwise 'haiku'."""
    if "kappa_difference" not in intervals:
        raise ValueError("the paired interval is undefined, so the rule cannot be applied")
    return "local" if intervals["kappa_difference"][1] < 0 else "haiku"


def read_metrics(rows: list[dict]) -> dict[str, float | None]:
    """Each judge against the reader's call on the 49 claims where the local judge and Sonnet split."""
    def right(r: dict, who: str) -> bool:
        return r["haiku"] is not None and r["haiku"] == r["reader"] if who == "haiku" else r[who] == r["reader"]

    out: dict[str, float | None] = {"claims": float(len(rows)), "reader_supported": float(sum(r["reader"] for r in rows)),
                                    "haiku_supported": float(sum(r["haiku"] is True for r in rows)),
                                    "parse_failures": float(sum(r["haiku"] is None for r in rows)),
                                    "haiku_with_local": float(sum(r["haiku"] is not None and r["haiku"] == r["local"] for r in rows)),
                                    "haiku_with_sonnet": float(sum(r["haiku"] is not None and r["haiku"] == r["sonnet"] for r in rows))}
    for tag, part in (("", rows), ("clear_", [r for r in rows if not r["close"]]),
                      ("served_", [r for r in rows if r["group"] == "A"]), ("other_", [r for r in rows if r["group"] == "B"])):
        out[tag + "n"] = float(len(part))
        for who in ("haiku", "local", "sonnet"):
            out[f"{tag}{who}_right"] = float(sum(right(r, who) for r in part))
    return out


# ---------------------------------------------------------------- the record

def build_rows(cfg: dict[str, Any], claims: dict[str, dict], reads: list[dict], ck: Checkpoint) -> tuple[list[dict], list[dict]]:
    sample, missing = [], 0
    for cid in draw_sample(claims, cfg["seed"], cfg["sample_size"]):
        got = ck.rows.get(f"sample|{cid}")
        if got is None:
            missing += 1
            continue
        sample.append({"id": cid, **claims[cid], "haiku": got["haiku"], "haiku_reply": got["reply"]})
    read = []
    for r in reads:
        got = ck.rows.get(f"read|{r['tag']}")
        if got is None:
            missing += 1
            continue
        read.append({"id": f"{r['item']}:{r['k']}", **{k: r[k] for k in ("tag", "group", "item", "qid", "system", "local", "sonnet", "reader", "close")},
                     "claim": claims[f"{r['item']}:{r['k']}"]["claim"], "haiku": got["haiku"], "haiku_reply": got["reply"]})
    if missing:
        raise ValueError(f"{missing} claims are not graded yet; run the grade stage")
    return sample, read


def numbers(cfg: dict[str, Any], sample: list[dict], read: list[dict], served: str) -> tuple[dict, dict]:
    m, i = sample_metrics(sample, served), sample_intervals(sample, served, BOOTSTRAP_N, BOOTSTRAP_SEED)
    m["haiku_clearly_worse"] = float(decision(i) == "local")
    return {"sample": m, "read": read_metrics(read)}, {"sample": i, "read": {}}


def write_record(cfg: dict[str, Any], sample: list[dict], read: list[dict], sessions: list[dict], root: Path, started: str) -> Path:
    e14_cfg = load_e14_config(cfg, root)
    metrics, intervals = numbers(cfg, sample, read, e14_cfg["served_system"])
    calls = len(sample) + len(read)
    config = {**{k: v for k, v in cfg.items() if k != "device"}, "device": cfg["device"], "claude_calls": calls,
              "served_system": e14_cfg["served_system"], "source": e14_cfg["source"], "local_judge": e14_cfg["judge_model"],
              "rubric_sha256": rubric_hash(), "wall_seconds": round(sum(s["seconds"] for s in sessions), 1),
              "sessions": sessions, "started_at": started, "labels_reviewed": False, "caveat": CAVEAT}
    attempts = [{"stage": "grade", "claude_calls": calls, "workers": cfg["workers"],
                 "replies_without_a_verdict": int(metrics["sample"]["parse_failures"] + metrics["read"]["parse_failures"])}]
    return write_run(EXPERIMENT, e14.SPLIT, config, {"judge_under_test": cfg["model"], "reference": cfg["reference"],
                     "local_judge": e14_cfg["judge_model"]}, metrics, intervals, attempts, {"sample": sample, "read": read},
                     split_info(e14.SPLIT, root).hash, root=root,
                     notes="Haiku replies that were not a verdict count as disagreements with Sonnet. " + CAVEAT)


def write_decision(path: Path, root: Path) -> dict[str, Any]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    interval = rec["intervals"]["sample"]["kappa_difference"]
    out = {"first_judge": decision({"kappa_difference": tuple(interval)}), "record_id": rec["run_id"],
           "model": rec["config"]["model"], "rubric_sha256": rec["config"]["rubric_sha256"],
           "interval": {"statistic": "kappa(Haiku, Sonnet) - kappa(local judge, Sonnet), answers resampled",
                        "low": interval[0], "high": interval[1], "point": rec["metrics"]["sample"]["kappa_difference"],
                        "level": 0.95}}
    atomic_write_text(root / DECISION_REL, json.dumps(out, indent=1) + "\n")
    return out


def problems(rec: dict[str, Any], root: Path, min_n: int) -> list[str]:
    """What check_run finds wrong with an E14H record: every number recomputes from the saved rows, and every claim,
    local grade and Sonnet grade is the one E14's record holds."""
    from sprout.eval.check_run import _header_problems, _read_rows, _same, _same_pair

    out = _header_problems(rec, root, min_n)
    info = split_info(rec["split"], root)
    if rec["split"] != e14.SPLIT:
        return out + ["E14H grades rag_dev claims only"]
    if rec["dataset_hash"] != info.hash:
        out.append("dataset_hash does not match the manifest")
    paths, cfg = rec["predictions"], rec["config"]
    if set(paths) != {"sample", "read"} or set(rec["metrics"]) != {"sample", "read"}:
        return out + ["E14H holds sample and read in predictions and metrics"]
    sample, read = _read_rows(root / paths["sample"]), _read_rows(root / paths["read"])
    claims = load_claims(cfg, root)
    if [r["id"] for r in sample] != draw_sample(claims, cfg["seed"], cfg["sample_size"]) or len(sample) != rec["n"]:
        out.append("the sample rows are not the seeded draw")
    for r in sample + read:
        if {k: r[k] for k in CLAIM_KEYS} != {k: claims[r["id"]][k] for k in CLAIM_KEYS}:
            out.append(f"{r['id']}: claim or grades differ from E14's record")
        if r["haiku"] != e14.parse_verdict(r["haiku_reply"]):
            out.append(f"{r['id']}: haiku does not follow from the saved reply")
    if [(r["tag"], r["item"], r["qid"], r["system"], r["local"], r["sonnet"], r["reader"], r["close"], r["group"]) for r in read] != [
            (x["tag"], x["item"], x["qid"], x["system"], x["local"], x["sonnet"], x["reader"], x["close"], x["group"]) for x in load_reads(cfg, root)]:
        out.append("the read rows differ from the reader's calls in " + cfg["reads"])
    if rec["claude_calls"] != len(sample) + len(read) or cfg["model"] != rec["models"]["judge_under_test"]:
        out.append("claude_calls or the model name does not match the rows")
    if cfg["rubric_sha256"] != rubric_hash():
        out.append("the rubric differs from the one the record names")
    if rec["bootstrap"] != {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}:
        return out + ["bootstrap must be the standard n and seed"]
    fresh_m, fresh_i = numbers(cfg, sample, read, cfg["served_system"])
    for system in ("sample", "read"):
        for name, saved in rec["metrics"][system].items():
            if name not in fresh_m[system] or not _same(fresh_m[system][name], saved):
                out.append(f"metric {system}.{name} does not recompute from predictions")
    saved_i = rec["intervals"].get("sample") or {}
    if set(saved_i) != set(fresh_i["sample"]) or any(not _same_pair(fresh_i["sample"][k], v) for k, v in saved_i.items()):
        out.append("intervals do not recompute from predictions")
    return out


# ---------------------------------------------------------------- command line

def _sessions(work: Path) -> Path:
    return work / "haiku-sessions.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "grade", "record"])
    ap.add_argument("--config", default="configs/e14h.json")
    ap.add_argument("--counter", help="the shared Claude call counter (results/claude_calls.json of the main folder)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(ROOT / a.config)
        claims, reads = load_claims(cfg, ROOT), load_reads(cfg, ROOT)
        _, blind = e14.load_source(load_e14_config(cfg, ROOT), ROOT)
        work = ROOT / WORK_REL
        ck = Checkpoint(checkpoint_path(cfg, work))
        if a.stage == "record":
            sample, read = build_rows(cfg, claims, reads, ck)
            sessions = [json.loads(x) for x in _sessions(work).read_text(encoding="utf-8").splitlines() if x.strip()]
            path = write_record(cfg, sample, read, sessions, ROOT, sessions[0]["started_at"])
            print(path)
            print(json.dumps(write_decision(path, ROOT), indent=1))
            return 0
        if not a.counter:
            raise ValueError("--counter is required: the shared call counter, so these calls are counted")
        llm = ClaudeCodeClient(model=cfg["model"], counter_path=Path(a.counter))
        tasks = tasks_for(cfg, claims, reads)
        left = sum(t[0] not in ck.rows for t in tasks)
        if a.stage == "plan":
            print(f"{len(tasks)} Haiku calls ({cfg['sample_size']} sample + {len(reads)} read), {left} not yet saved; "
                  f"limit {cfg['max_calls']}; counted so far {llm.calls_made()} of {llm.cap}")
            return 0
        started, t = datetime.now(timezone.utc).isoformat(), time.time()
        try:
            grade(tasks, blind, llm, ck, cfg)
        finally:
            seconds = round(time.time() - t, 1)
            with _sessions(work).open("a", encoding="utf-8") as f:
                f.write(json.dumps({"started_at": started, "seconds": seconds, "workers": cfg["workers"],
                                    "saved_after": len(ck.rows)}) + "\n")
        print(f"graded; {len(ck.rows)} replies saved, {seconds}s this session")
    except (LLMError, RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
