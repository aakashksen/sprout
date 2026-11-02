#!/usr/bin/env python3
"""The plan's state machine. A task reaches "done" only through this tool, which reruns the
acceptance commands itself and requires a PASS from the independent verifier at the same commit.
Standard library only, so it works before the venv exists."""
import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parent.parent))
TASKS = ROOT / "orchestration" / "tasks.json"
EVIDENCE = ROOT / "orchestration" / "evidence"
VERDICTS = ROOT / "orchestration" / "verdicts"
MAX_ATTEMPTS = 3
TIMEOUT_S = int(os.environ.get("CL_VERIFY_TIMEOUT_S", "1800"))


def load():
    return json.loads(TASKS.read_text(encoding="utf-8"))


def save(plan):
    TASKS.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def find(plan, task_id):
    for task in plan["tasks"]:
        if task["id"] == task_id:
            return task
    sys.exit(f"no such task: {task_id}")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def head():
    return git("rev-parse", "HEAD")


def dirty():
    # The state files change as work proceeds; everything else must be committed.
    return git("status", "--porcelain", "--", ".", ":!orchestration")


def python():
    venv = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv) if venv.exists() else sys.executable


def ready(plan, owner="agent"):
    done = {t["id"] for t in plan["tasks"] if t["status"] == "done"}
    return [
        t for t in plan["tasks"]
        if t["status"] == "pending" and t["owner"] == owner and set(t["deps"]) <= done
    ]


def cmd_next(args):
    plan = load()
    active = [t for t in plan["tasks"] if t["status"] == "in_progress"]
    if active:
        print(json.dumps({"state": "resume", "task": active[0]}, indent=2))
        return
    tasks = ready(plan)
    if tasks:
        print(json.dumps({"state": "ready", "task": tasks[0]}, indent=2))
        return
    open_tasks = [t for t in plan["tasks"] if t["status"] != "done"]
    if not open_tasks:
        print(json.dumps({"state": "complete"}))
        return
    print(json.dumps({
        "state": "waiting",
        "human": [{"id": t["id"], "title": t["title"]} for t in ready(plan, "human")],
        "blocked": [{"id": t["id"], "reason": t.get("reason")} for t in open_tasks if t["status"] == "blocked"],
    }, indent=2))


def cmd_start(args):
    plan = load()
    task = find(plan, args.id)
    if task["owner"] != "agent":
        sys.exit(f"{args.id} belongs to the person; use human-done after they finish")
    done = {t["id"] for t in plan["tasks"] if t["status"] == "done"}
    missing = set(task["deps"]) - done
    if missing:
        sys.exit(f"{args.id} is waiting on: {sorted(missing)}")
    if task["status"] not in ("pending", "in_progress"):
        sys.exit(f"{args.id} is {task['status']}")
    task["status"] = "in_progress"
    task["started_at"] = task.get("started_at") or now()
    save(plan)
    print(f"started {args.id}")


def cmd_verify(args):
    """Runs the acceptance commands from the plan. Output goes to a log the verifier can read."""
    plan = load()
    task = find(plan, args.id)
    commit = head()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    log_path = EVIDENCE / f"{args.id}.log"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    passed = bool(task["acceptance"])
    lines = [f"task {args.id} commit {commit} at {now()}"]
    if dirty():
        passed = False
        lines.append("FAILED: uncommitted changes:\n" + dirty())
    for command in task["acceptance"]:
        if not passed:
            lines.append(f"SKIPPED: {command}")
            continue
        argv = [python() if part == "{py}" else part for part in shlex.split(command)]
        argv[0] = shutil.which(argv[0]) or argv[0]  # npm is npm.cmd on Windows
        lines.append(f"$ {command}")
        try:
            run = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
            lines += [run.stdout[-6000:], run.stderr[-3000:], f"exit {run.returncode}"]
            passed = run.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as err:
            lines.append(f"could not finish: {err}")
            passed = False
    lines.append("RESULT " + ("PASS" if passed else "FAIL"))
    log_path.write_text("\n".join(lines), encoding="utf-8")
    task["verified"] = {"commit": commit, "passed": passed, "at": now()}
    save(plan)
    print(f"{args.id}: {'PASS' if passed else 'FAIL'} (log: {log_path.relative_to(ROOT)})")
    sys.exit(0 if passed else 1)


def cmd_done(args):
    plan = load()
    task = find(plan, args.id)
    if task["owner"] != "agent":
        sys.exit(f"{args.id} belongs to the person; use human-done")
    commit = head()
    if dirty():
        sys.exit("refused: uncommitted changes")
    verified = task.get("verified") or {}
    if not verified.get("passed") or verified.get("commit") != commit:
        sys.exit("refused: acceptance commands have not passed at this commit; run verify")
    path = VERDICTS / f"{args.id}.json"
    if not path.exists():
        sys.exit(f"refused: no verifier verdict at {path.relative_to(ROOT)}")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    if verdict.get("verdict") != "PASS" or verdict.get("commit") != commit:
        sys.exit("refused: the verifier has not passed this exact commit")
    task["status"] = "done"
    task["done_at"] = now()
    save(plan)
    print(f"done {args.id}")


def cmd_fail(args):
    plan = load()
    task = find(plan, args.id)
    task["attempts"] = task.get("attempts", 0) + 1
    task["reason"] = args.reason
    task["status"] = "blocked" if task["attempts"] >= MAX_ATTEMPTS else "in_progress"
    save(plan)
    print(f"{args.id}: attempt {task['attempts']}/{MAX_ATTEMPTS}, now {task['status']}")


def cmd_block(args):
    plan = load()
    task = find(plan, args.id)
    task["status"] = "blocked"
    task["reason"] = args.reason
    save(plan)
    print(f"blocked {args.id}")


def cmd_unblock(args):
    plan = load()
    task = find(plan, args.id)
    if task["status"] != "blocked":
        sys.exit(f"{args.id} is {task['status']}")
    task["status"] = "pending"
    task["attempts"] = 0
    task.pop("verified", None)
    task["note"] = args.note
    save(plan)
    print(f"unblocked {args.id}")


def cmd_reopen(args):
    plan = load()
    task = find(plan, args.id)
    task["status"] = "pending"
    task["attempts"] = 0
    task["reason"] = args.reason
    task.pop("verified", None)
    save(plan)
    print(f"reopened {args.id}")


def cmd_human_done(args):
    plan = load()
    task = find(plan, args.id)
    if task["owner"] != "human":
        sys.exit(f"{args.id} is an agent task")
    done = {t["id"] for t in plan["tasks"] if t["status"] == "done"}
    missing = set(task["deps"]) - done
    if missing:
        sys.exit(f"{args.id} is waiting on: {sorted(missing)}")
    task["status"] = "done"
    task["note"] = args.note
    task["done_at"] = now()
    save(plan)
    print(f"done {args.id}")


def cmd_status(args):
    plan = load()
    counts = {}
    for task in plan["tasks"]:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    print("  ".join(f"{key}: {value}" for key, value in sorted(counts.items())))
    for task in plan["tasks"]:
        if args.all or task["status"] != "done":
            who = "YOU" if task["owner"] == "human" else "agent"
            reason = f"  ({task['reason']})" if task.get("reason") and task["status"] == "blocked" else ""
            print(f"[{task['status']:<11}] {task['id']:<6} {who:<5} {task['title']}{reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("next").set_defaults(run=cmd_next)
    status = sub.add_parser("status")
    status.add_argument("--all", action="store_true")
    status.set_defaults(run=cmd_status)
    for name, run in (("start", cmd_start), ("verify", cmd_verify), ("done", cmd_done)):
        item = sub.add_parser(name)
        item.add_argument("id")
        item.set_defaults(run=run)
    for name, run, field in (("fail", cmd_fail, "--reason"), ("block", cmd_block, "--reason"),
                             ("unblock", cmd_unblock, "--note"), ("human-done", cmd_human_done, "--note"),
                             ("reopen", cmd_reopen, "--reason")):
        item = sub.add_parser(name)
        item.add_argument("id")
        item.add_argument(field, required=True)
        item.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
