"""The gate that everything else relies on. Standard library only, so it runs before the venv exists."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CTL = REPO / "tools" / "taskctl.py"
LOOP = REPO / "tools" / "run_loop.py"


def task(task_id, owner="agent", deps=(), acceptance=("git --version",)):
    return {"id": task_id, "phase": 0, "title": task_id, "owner": owner, "deps": list(deps),
            "acceptance": list(acceptance), "status": "pending"}


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        (self.root / "orchestration" / "verdicts").mkdir(parents=True)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")
        (self.root / "a.txt").write_text("a")
        self.git("add", "a.txt")
        self.git("commit", "-qm", "init")
        self.write([task("T1"), task("T2", deps=["T1"]), task("H1", owner="human")])

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=True).stdout.strip()

    def write(self, tasks):
        (self.root / "orchestration" / "tasks.json").write_text(json.dumps({"tasks": tasks}))

    def state(self, task_id):
        plan = json.loads((self.root / "orchestration" / "tasks.json").read_text())
        return next(t for t in plan["tasks"] if t["id"] == task_id)

    def ctl(self, *args):
        env = {**os.environ, "CL_ROOT": str(self.root)}
        return subprocess.run([sys.executable, str(CTL), *args], capture_output=True, text=True, env=env)

    def verdict(self, task_id, verdict="PASS", commit=None):
        path = self.root / "orchestration" / "verdicts" / f"{task_id}.json"
        path.write_text(json.dumps({"task": task_id, "verdict": verdict, "commit": commit or self.git("rev-parse", "HEAD")}))


class Gate(Sandbox):
    def test_next_respects_dependencies_and_skips_human_tasks(self):
        self.assertEqual(json.loads(self.ctl("next").stdout)["task"]["id"], "T1")
        self.assertNotEqual(self.ctl("start", "T2").returncode, 0)
        self.assertNotEqual(self.ctl("start", "H1").returncode, 0)

    def test_done_is_refused_without_verify(self):
        self.ctl("start", "T1")
        self.verdict("T1")
        out = self.ctl("done", "T1")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("have not passed", out.stderr)
        self.assertEqual(self.state("T1")["status"], "in_progress")

    def test_done_is_refused_without_a_verifier_pass(self):
        self.ctl("start", "T1")
        self.assertEqual(self.ctl("verify", "T1").returncode, 0)
        self.assertIn("no verifier verdict", self.ctl("done", "T1").stderr)
        self.verdict("T1", "FAIL")
        self.assertIn("not passed this exact commit", self.ctl("done", "T1").stderr)

    def test_done_is_refused_when_a_commit_landed_after_verifying(self):
        self.ctl("start", "T1")
        self.ctl("verify", "T1")
        old = self.git("rev-parse", "HEAD")
        (self.root / "b.txt").write_text("b")
        self.git("add", "b.txt")
        self.git("commit", "-qm", "later")
        self.verdict("T1", commit=old)
        self.assertNotEqual(self.ctl("done", "T1").returncode, 0)

    def test_verify_fails_on_uncommitted_work_and_failed_command(self):
        self.ctl("start", "T1")
        (self.root / "dirty.txt").write_text("x")
        self.assertNotEqual(self.ctl("verify", "T1").returncode, 0)
        (self.root / "dirty.txt").unlink()
        self.write([task("T1", acceptance=["git no-such-command"])])
        self.ctl("start", "T1")
        self.assertNotEqual(self.ctl("verify", "T1").returncode, 0)

    def test_full_path_to_done_unlocks_the_next_task(self):
        self.ctl("start", "T1")
        self.assertEqual(self.ctl("verify", "T1").returncode, 0)
        self.verdict("T1")
        self.assertEqual(self.ctl("done", "T1").returncode, 0)
        self.assertEqual(json.loads(self.ctl("next").stdout)["task"]["id"], "T2")

    def test_three_failures_block_and_unblock_resets(self):
        self.ctl("start", "T1")
        for _ in range(3):
            self.ctl("fail", "T1", "--reason", "nope")
        self.assertEqual(self.state("T1")["status"], "blocked")
        self.assertEqual(self.ctl("unblock", "T1", "--note", "fixed").returncode, 0)
        self.assertEqual(self.state("T1")["status"], "pending")
        self.assertEqual(self.state("T1")["attempts"], 0)

    def test_reopen_sends_a_finished_task_back(self):
        self.ctl("start", "T1")
        self.ctl("verify", "T1")
        self.verdict("T1")
        self.ctl("done", "T1")
        self.assertEqual(self.ctl("reopen", "T1", "--reason", "labels wrong").returncode, 0)
        self.assertEqual(self.state("T1")["status"], "pending")
        self.assertNotIn("verified", self.state("T1"))
        self.assertNotEqual(self.ctl("start", "T2").returncode, 0)

    def test_waiting_reports_the_persons_tasks(self):
        self.write([task("H1", owner="human"), task("T1", deps=["H1"])])
        state = json.loads(self.ctl("next").stdout)
        self.assertEqual(state["state"], "waiting")
        self.assertEqual(state["human"][0]["id"], "H1")
        self.assertEqual(self.ctl("human-done", "H1", "--note", "ok").returncode, 0)
        self.assertEqual(json.loads(self.ctl("next").stdout)["task"]["id"], "T1")


class Loop(Sandbox):
    def loop(self, script_body, *args):
        stub = self.root / "stub.py"
        stub.write_text(script_body)
        env = {**os.environ, "CL_ROOT": str(self.root), "CL_CLAUDE_CMD": f'"{sys.executable}" "{stub}"'}
        return subprocess.run([sys.executable, str(LOOP), *args], capture_output=True, text=True, env=env)

    def test_stops_after_sessions_that_finish_nothing(self):
        run = self.loop("import sys\nprint('ran')", "--stall-limit", "2")
        self.assertEqual(run.returncode, 3, run.stdout + run.stderr)
        self.assertIn("2 sessions in a row", run.stdout)

    def test_stops_when_the_person_is_needed(self):
        self.write([task("H1", owner="human")])
        run = self.loop("print('should not run')")
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("H1", run.stdout)

    def test_reports_complete(self):
        self.write([{**task("T1"), "status": "done"}])
        self.assertEqual(self.loop("print('no')").returncode, 0)


class PlanMatchesState(unittest.TestCase):
    """tasks.json drives the loop and PLAN.md explains it, so they must not drift apart."""

    def setUp(self):
        self.tasks = json.loads((REPO / "orchestration" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
        self.plan = (REPO / "docs" / "PLAN.md").read_text(encoding="utf-8")

    def test_every_task_has_a_plan_section_and_the_reverse(self):
        headings = set(re.findall(r"^### ([TH]\d+\.\d+|H\d+):", self.plan, re.M))
        self.assertEqual(headings, {t["id"] for t in self.tasks})

    def test_dependencies_exist_and_point_backwards(self):
        seen = set()
        for item in self.tasks:
            self.assertTrue(set(item["deps"]) <= seen, f"{item['id']} depends on a later or missing task")
            seen.add(item["id"])

    def test_every_agent_task_has_acceptance_commands(self):
        for item in self.tasks:
            if item["owner"] == "agent":
                self.assertTrue(item["acceptance"], item["id"])
            self.assertIn(item["owner"], ("agent", "human"))

    def test_every_agent_declares_a_model(self):
        for path in (REPO / ".claude" / "agents").glob("*.md"):
            self.assertRegex(path.read_text(encoding="utf-8"), r"(?m)^model: (haiku|sonnet)\s*$", path.name)


if __name__ == "__main__":
    unittest.main()
