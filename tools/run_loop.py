#!/usr/bin/env python3
"""Optional unattended loop: gives each task batch a fresh Claude Code session, so context does not
pile up. Stops when the plan is complete, needs the person, or stops making progress."""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parent.parent))
CTL = Path(__file__).resolve().parent / "taskctl.py"
PROMPT = "Use the run-plan skill with a limit of {n} tasks. Follow it exactly and stop when it says to stop."


def plan_state():
    out = subprocess.run([sys.executable, str(CTL), "next"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def done_count():
    plan = json.loads((ROOT / "orchestration" / "tasks.json").read_text(encoding="utf-8"))
    return sum(t["status"] == "done" for t in plan["tasks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument("--tasks-per-session", type=int, default=3)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--stall-limit", type=int, default=2)
    args = parser.parse_args()
    base = shlex.split(os.environ.get("CL_CLAUDE_CMD", "claude"))
    base[0] = shutil.which(base[0]) or base[0]  # may be claude.cmd on Windows
    stalled = 0
    for session in range(1, args.max_sessions + 1):
        state = plan_state()
        if state["state"] == "complete":
            print("plan complete")
            return 0
        if state["state"] == "waiting":
            print("stopped: the plan needs you\n" + json.dumps(state, indent=2))
            return 2
        before = done_count()
        print(f"session {session}/{args.max_sessions}: {done_count()} tasks done", flush=True)
        prompt = PROMPT.format(n=args.tasks_per_session)
        run = subprocess.run([*base, "-p", prompt, "--model", args.model], cwd=ROOT)
        stalled = stalled + 1 if done_count() == before else 0
        if run.returncode != 0:
            print(f"Claude Code exited with {run.returncode}; often a usage limit or a denied permission")
        if stalled >= args.stall_limit:
            print(f"stopped: {stalled} sessions in a row finished no task")
            return 3
    print("stopped: reached --max-sessions")
    return 4


if __name__ == "__main__":
    sys.exit(main())
