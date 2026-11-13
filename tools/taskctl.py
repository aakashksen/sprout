#!/usr/bin/env python3
"""The plan's state machine. A task reaches "done" only through this tool, which reruns the
acceptance commands itself and requires a PASS from the independent verifier at the same commit.
Tasks can run side by side in their own git worktrees and land one at a time.
Standard library only, so it works before the venv exists."""
import argparse
import contextlib
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parent.parent))
TASKS = ROOT / "orchestration" / "tasks.json"
EVIDENCE = ROOT / "orchestration" / "evidence"
VERDICTS = ROOT / "orchestration" / "verdicts"
GPU_NEEDS = ROOT / "orchestration" / "gpu_needs.json"
STATE_LOCK = ROOT / "orchestration" / ".state.lock"
LAND_LOCK = ROOT / "orchestration" / ".land.lock"
WT_ROOT = Path(os.environ.get("CL_WT_ROOT", ROOT.parent / "_worktrees" / ROOT.name))
GPU_LOCK = Path(os.environ.get("CL_GPU_LOCK", ROOT.parent / ".gpu.lock"))
# Git-ignored data a worktree needs. Directories are linked, never copied, so large data exists once.
SHARED = ("datasets/raw", "knowledge/raw", "models", "mlruns", ".dvc/cache", ".dvcstore")
SHARED_FILES = (".env",)
# Outputs every worktree must write to the main folder, so they outlive the worktree.
SHARED_OUTPUTS = ("models", "mlruns")
# Ignored files that are safe to lose when a worktree goes away.
DISPOSABLE = ("__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "node_modules", ".egg-info", ".pyc", "dist", "build")
GPU_BUDGET_MB = int(os.environ.get("CL_GPU_BUDGET_MB", "6400"))  # 8 GB less what the desktop keeps
MAX_ATTEMPTS = 3
INDEX_LOCK_STALE_S = 120
INDEX_LOCK_TRIES = 12
INDEX_LOCK_WAIT_S = 15
TIMEOUT_S = int(os.environ.get("CL_VERIFY_TIMEOUT_S", "1800"))


def load():
    return json.loads(TASKS.read_text(encoding="utf-8"))


def save(plan):
    temp = TASKS.with_suffix(".tmp")
    temp.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, TASKS)


@contextlib.contextmanager
def file_lock(path, wait_s=900, stale_s=3600):
    """A lock file, since two tracks can call this tool at once."""
    deadline = time.time() + wait_s
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - path.stat().st_mtime > stale_s:
                    path.unlink()
                    continue
            if time.time() > deadline:
                sys.exit(f"timed out waiting for {path.name}")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(handle)
        with contextlib.suppress(OSError):
            path.unlink()


@contextlib.contextmanager
def editing():
    with file_lock(STATE_LOCK, wait_s=60, stale_s=120):
        plan = load()
        yield plan
        save(plan)


def find(plan, task_id):
    for task in plan["tasks"]:
        if task["id"] == task_id:
            return task
    sys.exit(f"no such task: {task_id}")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True).stdout.strip()


def git_ok(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True)


def clear_stale_index_lock(cwd=None, older_than_s=INDEX_LOCK_STALE_S):
    """Git holds index.lock for moments, and a killed git leaves an empty one behind. An old empty one is stale."""
    lock = Path(cwd or ROOT) / git("rev-parse", "--git-path", "index.lock", cwd=cwd).strip()
    try:
        info = lock.stat()
        if info.st_size == 0 and time.time() - info.st_mtime > older_than_s:
            lock.unlink()
            return True
    except OSError:
        pass
    return False


def move_forward(ref, cwd=None, tries=INDEX_LOCK_TRIES, wait_s=INDEX_LOCK_WAIT_S):
    """Fast-forwards to ref, waiting out index.lock: a land takes minutes, and shouldn't be lost to a lock at its end."""
    for attempt in range(tries):
        done = git_ok("merge", "--ff-only", "-q", ref, cwd=cwd)
        if done.returncode == 0 or "index.lock" not in done.stderr or attempt == tries - 1:
            return done
        clear_stale_index_lock(cwd)
        time.sleep(wait_s)
    return done


def head(cwd=None):
    return git("rev-parse", "HEAD", cwd=cwd)


def dirty(cwd=None):
    # The state files change as work proceeds; everything else must be committed.
    return git("--no-optional-locks", "status", "--porcelain", "--", ".", ":!orchestration", cwd=cwd)


def python():
    venv = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv) if venv.exists() else sys.executable


def gpu_table():
    return json.loads(GPU_NEEDS.read_text(encoding="utf-8")) if GPU_NEEDS.exists() else {}


def gpu_need(task_id):
    return gpu_table().get(task_id)


def gpu_holders():
    """Who holds GPU memory now: (label, reserved MB, exclusive)."""
    if not GPU_LOCK.exists():
        return []
    rows = []
    for line in GPU_LOCK.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            rows.append((parts[0], int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else GPU_BUDGET_MB, len(parts) > 2 and parts[2] == "exclusive"))
    return rows


def ready(plan, owner="agent"):
    done = {t["id"] for t in plan["tasks"] if t["status"] == "done"}
    tasks = [
        t for t in plan["tasks"]
        if t["status"] == "pending" and t["owner"] == owner and set(t["deps"]) <= done
    ]
    return tasks


def brief(task):
    return {key: task.get(key) for key in ("id", "title", "status", "worktree", "attempts", "reason") if task.get(key) is not None}


def cmd_next(args):
    plan = load()
    active = [t for t in plan["tasks"] if t["status"] == "in_progress"]
    tasks = ready(plan)
    if tasks:
        print(json.dumps({"state": "ready", "task": tasks[0], "ready": [brief(t) for t in tasks], "active": [brief(t) for t in active]}, indent=2))
        return
    if active:
        print(json.dumps({"state": "active", "active": [brief(t) for t in active],
                          "human": [brief(t) for t in ready(plan, "human")]}, indent=2))
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


def link_shared(worktree):
    for rel in SHARED_OUTPUTS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    for rel in SHARED:
        target, link = ROOT / rel, worktree / rel
        if not target.is_dir():
            continue
        if not link.exists():
            make_link(link, target)
            continue
        for child in target.iterdir():
            if not (link / child.name).exists():
                make_link(link / child.name, child)
    for rel in SHARED_FILES:
        if (ROOT / rel).exists() and not (worktree / rel).exists():
            shutil.copy2(ROOT / rel, worktree / rel)


def make_link(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        os.link(target, link)
    elif os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def is_link(path):
    return os.path.isjunction(path) or path.is_symlink()


def unlink_shared(worktree):
    """Removes only the links, so removing the worktree can't reach the shared data."""
    for rel in SHARED:
        link = worktree / rel
        if is_link(link):
            os.rmdir(link) if os.path.isjunction(link) else link.unlink()
        elif link.is_dir():
            for child in link.iterdir():
                if is_link(child):
                    os.rmdir(child) if os.path.isjunction(child) else child.unlink()


def rescue_ignored(worktree):
    """Moves git-ignored work out of a worktree before it's removed. Raises rather than overwrite anything."""
    listed = git("--no-optional-locks", "status", "--porcelain", "--ignored", "--untracked-files=all", cwd=worktree)
    paths = [line[3:].strip().strip('"').rstrip("/") for line in listed.splitlines() if line.startswith("!! ")]
    keep = [rel for rel in paths if not any(part in DISPOSABLE or part.endswith(DISPOSABLE) for part in Path(rel).parts)]
    moves, conflicts = [], []
    for rel in keep:
        source = worktree / rel
        files = [source] if source.is_file() else [f for f in source.rglob("*") if f.is_file()]
        for file in files:
            target = ROOT / file.relative_to(worktree)
            if target.exists() and target.read_bytes() != file.read_bytes():
                conflicts.append(str(file.relative_to(worktree)))
            elif not target.exists():
                moves.append((file, target))
    if conflicts:
        raise FileExistsError("; ".join(conflicts[:5]))
    for file, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(target))
    return len(moves)


def remove_worktree(task):
    path = task.get("worktree")
    if not path:
        return
    unlink_shared(Path(path))
    rescued = rescue_ignored(Path(path))
    if rescued:
        print(f"moved {rescued} git-ignored files from the worktree to the main folder")
    git_ok("worktree", "remove", "--force", path)
    git_ok("worktree", "prune")
    git_ok("branch", "-D", f"task/{task['id']}")
    task.pop("worktree", None)


def cmd_start(args):
    with editing() as plan:
        task = find(plan, args.id)
        if task["owner"] != "agent":
            sys.exit(f"{args.id} belongs to the person; use human-done after they finish")
        done = {t["id"] for t in plan["tasks"] if t["status"] == "done"}
        missing = set(task["deps"]) - done
        if missing:
            sys.exit(f"{args.id} is waiting on: {sorted(missing)}")
        if task["status"] not in ("pending", "in_progress"):
            sys.exit(f"{args.id} is {task['status']}")
        if args.worktree and not task.get("worktree"):
            path = WT_ROOT / args.id
            path.parent.mkdir(parents=True, exist_ok=True)
            added = git_ok("worktree", "add", "-q", "-b", f"task/{args.id}", str(path), git("branch", "--show-current"))
            if added.returncode:
                sys.exit(f"could not create a worktree: {added.stderr.strip()}")
            link_shared(path)
            task["worktree"] = str(path)
        task["status"] = "in_progress"
        task["started_at"] = task.get("started_at") or now()
        where = task.get("worktree") or str(ROOT)
    print(f"started {args.id} in {where}")


def run_acceptance(task, cwd, log_name):
    commit = head(cwd)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    log_path = EVIDENCE / log_name
    env = {**os.environ, "PYTHONPATH": str(Path(cwd) / "src")}
    passed = bool(task["acceptance"])
    lines = [f"task {task['id']} commit {commit} at {now()} in {cwd}"]
    changes = dirty(cwd)
    if changes:
        passed = False
        lines.append("FAILED: uncommitted changes:\n" + changes)
    for command in task["acceptance"]:
        if not passed:
            lines.append(f"SKIPPED: {command}")
            continue
        argv = [python() if part == "{py}" else part for part in shlex.split(command)]
        argv[0] = shutil.which(argv[0]) or argv[0]  # npm is npm.cmd on Windows
        lines.append(f"$ {command}")
        try:
            run = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
            lines += [run.stdout[-6000:], run.stderr[-3000:], f"exit {run.returncode}"]
            passed = run.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as err:
            lines.append(f"could not finish: {err}")
            passed = False
    lines.append("RESULT " + ("PASS" if passed else "FAIL"))
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return commit, passed, log_path


def cmd_verify(args):
    """Runs the acceptance commands from the plan. Output goes to a log the verifier can read."""
    task = find(load(), args.id)
    cwd = task.get("worktree") or ROOT
    commit, passed, log_path = run_acceptance(task, cwd, f"{args.id}.log")
    with editing() as plan:
        find(plan, args.id)["verified"] = {"commit": commit, "passed": passed, "at": now()}
    print(f"{args.id}: {'PASS' if passed else 'FAIL'} (log: {log_path.relative_to(ROOT)})")
    sys.exit(0 if passed else 1)


def verdict_commit(task_id):
    path = VERDICTS / f"{task_id}.json"
    if not path.exists():
        sys.exit(f"refused: no verifier verdict at {path.relative_to(ROOT)}")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    if verdict.get("verdict") != "PASS":
        sys.exit("refused: the verifier has not passed this exact commit")
    return verdict.get("commit")


def verifier_passed(task, commit):
    """The independent verifier is required unless the person exempted the task in tasks.json."""
    if task.get("verifier") is False:
        return
    if verdict_commit(task["id"]) != commit:
        sys.exit("refused: the verifier has not passed this exact commit")


def require_decision_entry(task_id, base):
    """A task is not finished until docs/DECISIONS.md names it, in an entry or as `<id>: no new decisions`.
    The id must stand alone, so T4.1 is not found inside T4.10 or T4.1.5."""
    path = Path(base) / "docs" / "DECISIONS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not re.search(r"(?<!\w)" + re.escape(task_id) + r"(?!\w|\.\d)", text):
        sys.exit(f"refused: docs/DECISIONS.md does not mention {task_id}; add an entry for what it decided, "
                 f"or a line `{task_id}: no new decisions`")


def cmd_done(args):
    plan = load()
    task = find(plan, args.id)
    if task["owner"] != "agent":
        sys.exit(f"{args.id} belongs to the person; use human-done")
    if task.get("worktree"):
        sys.exit(f"refused: {args.id} is in a worktree; use land")
    commit = head()
    if dirty():
        sys.exit("refused: uncommitted changes")
    verified = task.get("verified") or {}
    if not verified.get("passed") or verified.get("commit") != commit:
        sys.exit("refused: acceptance commands have not passed at this commit; run verify")
    verifier_passed(task, commit)
    require_decision_entry(args.id, ROOT)
    with editing() as plan:
        task = find(plan, args.id)
        task["status"] = "done"
        task["done_at"] = now()
    print(f"done {args.id}")


def cmd_land(args):
    """Merges a verified worktree task into the build branch. The build branch only ever moves forward
    to a commit that passed this task's acceptance commands after merging everything else that landed."""
    task = find(load(), args.id)
    worktree = task.get("worktree")
    if not worktree:
        sys.exit(f"refused: {args.id} has no worktree; use done")
    if dirty(worktree):
        sys.exit("refused: uncommitted changes in the worktree")
    verified = task.get("verified") or {}
    if not verified.get("passed") or verified.get("commit") != head(worktree):
        sys.exit("refused: acceptance commands have not passed at this commit; run verify")
    verifier_passed(task, verified["commit"])
    require_decision_entry(args.id, worktree)
    with file_lock(LAND_LOCK, wait_s=3600, stale_s=7200):
        if dirty():
            sys.exit("refused: the build branch has uncommitted changes")
        branch = git("branch", "--show-current")
        if git_ok("merge-base", "--is-ancestor", branch, "HEAD", cwd=worktree).returncode:
            merged = git_ok("merge", "--no-edit", "-q", branch, cwd=worktree)
            if merged.returncode:
                git_ok("merge", "--abort", cwd=worktree)
                print(f"MERGE CONFLICT: {args.id} conflicts with {branch}. Nothing landed; send it back to the implementer.\n{merged.stdout[-1500:]}")
                sys.exit(4)
            _, passed, log_path = run_acceptance(task, worktree, f"{args.id}.merged.log")
            if not passed:
                print(f"MERGED CODE FAILED: {args.id} no longer passes after merging {branch}. Nothing landed. See {log_path.relative_to(ROOT)}")
                sys.exit(5)
        unlink_shared(Path(worktree))
        try:
            rescued = rescue_ignored(Path(worktree))
        except FileExistsError as err:
            link_shared(Path(worktree))
            sys.exit(f"refused: nothing landed, because these ignored files differ from the main folder's copies: {err}")
        if rescued:
            print(f"moved {rescued} git-ignored files from the worktree to the main folder")
        forward = move_forward(f"task/{args.id}")
        if forward.returncode:
            link_shared(Path(worktree))
            sys.exit(f"refused: could not move {branch} forward: {forward.stderr.strip()}")
        with editing() as plan:
            task = find(plan, args.id)
            remove_worktree(task)
            task["status"] = "done"
            task["done_at"] = now()
    print(f"landed {args.id} at {head()}")


def cmd_fail(args):
    with editing() as plan:
        task = find(plan, args.id)
        task["attempts"] = task.get("attempts", 0) + 1
        task["reason"] = args.reason
        task["status"] = "blocked" if task["attempts"] >= MAX_ATTEMPTS else "in_progress"
    print(f"{args.id}: attempt {task['attempts']}/{MAX_ATTEMPTS}, now {task['status']}")


def cmd_block(args):
    with editing() as plan:
        task = find(plan, args.id)
        task["status"] = "blocked"
        task["reason"] = args.reason
    print(f"blocked {args.id}")


def cmd_unblock(args):
    with editing() as plan:
        task = find(plan, args.id)
        if task["status"] != "blocked":
            sys.exit(f"{args.id} is {task['status']}")
        task["status"] = "pending"
        task["attempts"] = 0
        task.pop("verified", None)
        task["note"] = args.note
    print(f"unblocked {args.id}")


def cmd_reopen(args):
    with editing() as plan:
        task = find(plan, args.id)
        task["status"] = "pending"
        task["attempts"] = 0
        task["reason"] = args.reason
        task.pop("verified", None)
    print(f"reopened {args.id}")


def gpu_problem(task_id):
    """Why this task's GPU job must not start, or None. Fails closed if the GPU can't be read."""
    need = gpu_need(task_id)
    if need is None:
        return None
    argv = shlex.split(os.environ.get("CL_NVIDIA_SMI", "nvidia-smi"))
    argv[0] = shutil.which(argv[0]) or argv[0]
    query = ["--query-gpu=memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run([*argv, *query], capture_output=True, text=True, timeout=30, check=True).stdout
        free, used, total = (int(part) for part in out.splitlines()[0].split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as err:
        return f"GPU CHECK FAILED: {task_id} needs about {need} MB of GPU memory, but the GPU couldn't be read: {err}"
    if free >= need:
        return None
    return (f"GPU BUSY: {task_id} needs about {need} MB of free GPU memory, and only {free} MB is free "
            f"({used} of {total} MB in use). Not started. Free up {need - free} MB or more.")


def cmd_gpu(args):
    problem = gpu_problem(args.id)
    if problem:
        print(problem)
        sys.exit(3)
    print(f"{args.id}: GPU ok")


def cmd_gpu_acquire(args):
    """Takes a share of the GPU just before a job starts. Jobs run together while their reserved memory fits."""
    label = f"{ROOT.name}/{args.id}"
    need = gpu_need(args.id) or 0
    exclusive = args.id in gpu_table().get("_exclusive", [])
    with file_lock(GPU_LOCK.with_suffix(".guard"), wait_s=60, stale_s=120):
        holders = gpu_holders()
        if any(h[0] == label for h in holders):
            print(f"{args.id}: already holds a GPU share")
            return
        reserved = sum(h[1] for h in holders)
        if holders and (exclusive or any(h[2] for h in holders)):
            who = ", ".join(h[0] for h in holders)
            print(f"GPU BUSY: {args.id} can't share the GPU with {who} because one of them needs it alone. Not started.")
            sys.exit(3)
        if reserved + need > GPU_BUDGET_MB:
            print(f"GPU BUSY: {args.id} needs about {need} MB, and {reserved} of {GPU_BUDGET_MB} MB is reserved by {', '.join(h[0] for h in holders)}. Not started.")
            sys.exit(3)
        problem = None if holders else gpu_problem(args.id)
        if problem:
            print(problem)
            sys.exit(3)
        with GPU_LOCK.open("a", encoding="utf-8") as f:
            f.write(f"{label} {need}{' exclusive' if exclusive else ''} {now()}\n")
    print(f"{args.id}: has a GPU share ({need} MB reserved)")


def cmd_gpu_release(args):
    label = f"{ROOT.name}/{args.id}"
    with file_lock(GPU_LOCK.with_suffix(".guard"), wait_s=60, stale_s=120):
        lines = GPU_LOCK.read_text(encoding="utf-8").splitlines() if GPU_LOCK.exists() else []
        keep = [] if args.force else [row for row in lines if row.split()[:1] != [label]]
        if not lines:
            print("no GPU share is held")
        elif len(keep) == len(lines):
            sys.exit(f"refused: {label} holds no share; held by {', '.join(row.split()[0] for row in lines)}")
        elif keep:
            GPU_LOCK.write_text("\n".join(keep) + "\n", encoding="utf-8")
            print(f"released {label}")
        else:
            GPU_LOCK.unlink()
            print(f"released {label}; the GPU is clear")


def cmd_human_done(args):
    with editing() as plan:
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
    start = sub.add_parser("start")
    start.add_argument("id")
    start.add_argument("--worktree", action="store_true", help="work in a separate git worktree, so tasks can run side by side")
    start.set_defaults(run=cmd_start)
    for name, run in (("verify", cmd_verify), ("done", cmd_done), ("land", cmd_land), ("gpu", cmd_gpu), ("gpu-acquire", cmd_gpu_acquire)):
        item = sub.add_parser(name)
        item.add_argument("id")
        item.set_defaults(run=run)
    release = sub.add_parser("gpu-release")
    release.add_argument("id")
    release.add_argument("--force", action="store_true")
    release.set_defaults(run=cmd_gpu_release)
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
