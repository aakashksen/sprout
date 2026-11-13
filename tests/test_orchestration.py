"""The gate that everything else relies on. Standard library only, so it runs before the venv exists."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CTL = REPO / "tools" / "taskctl.py"
LOOP = REPO / "tools" / "run_loop.py"


# Every task the sandbox uses has a line, so the decision-log check passes unless a test removes it.
NO_DECISIONS = "".join(f"{name}: no new decisions\n" for name in ("T1", "T2", "T3"))


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
        (self.root / "docs").mkdir()
        (self.root / "docs" / "DECISIONS.md").write_text(NO_DECISIONS)
        self.git("add", "a.txt", "docs/DECISIONS.md")
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

    def test_only_an_exempt_task_skips_the_verifier(self):
        self.ctl("start", "T1")
        self.assertEqual(self.ctl("verify", "T1").returncode, 0)
        self.assertIn("no verifier verdict", self.ctl("done", "T1").stderr)
        path = self.root / "orchestration" / "tasks.json"
        tasks = json.loads(path.read_text())["tasks"]
        next(t for t in tasks if t["id"] == "T1")["verifier"] = False
        self.write(tasks)
        self.assertNotIn("verdict", self.ctl("done", "T1").stderr)

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


class DecisionLog(Sandbox):
    """A task is not finished until docs/DECISIONS.md says what it decided, or that it decided nothing."""

    def log(self, text):
        (self.root / "docs" / "DECISIONS.md").write_text(text)
        self.git("add", "docs/DECISIONS.md")
        self.git("commit", "-qm", "decisions")

    def finish(self, task_id):
        self.write([task(task_id)])
        self.ctl("start", task_id)
        self.assertEqual(self.ctl("verify", task_id).returncode, 0)
        self.verdict(task_id)
        return self.ctl("done", task_id)

    def test_done_is_refused_without_an_entry(self):
        self.log("### T9 - something else\n")
        out = self.finish("T1")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("docs/DECISIONS.md", out.stderr)
        self.assertIn("T1", out.stderr)
        self.assertEqual(self.state("T1")["status"], "in_progress")
        self.assertNotIn("done_at", self.state("T1"))

    def test_done_passes_with_an_entry(self):
        self.log("### T1 - chose a thing over another\n- Chose: it\n")
        out = self.finish("T1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.state("T1")["status"], "done")

    def test_done_passes_with_the_no_decisions_line(self):
        self.log("T1: no new decisions\n")
        out = self.finish("T1")
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_a_longer_task_id_does_not_count(self):
        self.log("### T4.10 - another task\nT4.10: no new decisions\nT14.1: no new decisions\nXT4.1 is not it\n")
        out = self.finish("T4.1")
        self.assertNotEqual(out.returncode, 0)
        self.assertEqual(self.state("T4.1")["status"], "in_progress")
        self.log("T4.10 and also T4.1, which ends a sentence.\n")
        self.assertEqual(self.ctl("verify", "T4.1").returncode, 0)
        self.verdict("T4.1")
        self.assertEqual(self.ctl("done", "T4.1").returncode, 0)

    def test_a_missing_log_file_refuses(self):
        (self.root / "docs" / "DECISIONS.md").unlink()
        self.git("add", "-A")
        self.git("commit", "-qm", "no log")
        out = self.finish("T1")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("docs/DECISIONS.md", out.stderr)

    def test_tasks_already_done_are_not_rechecked(self):
        self.write([{**task("T1"), "status": "done"}, task("T2")])
        self.log("T9: no new decisions\n")
        self.assertEqual(json.loads(self.ctl("next").stdout)["task"]["id"], "T2")
        self.assertEqual(self.state("T1")["status"], "done")


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


class GpuGuard(Sandbox):
    def setUp(self):
        super().setUp()
        (self.root / "orchestration" / "gpu_needs.json").write_text(json.dumps({"T1": 5000}))
        self.lock = self.root / "gpu.lock"

    def smi(self, reply):
        stub = self.root / "smi.py"
        stub.write_text(reply)
        return f'"{sys.executable}" "{stub}"'

    def run_ctl(self, smi, *args):
        env = {**os.environ, "CL_ROOT": str(self.root), "CL_NVIDIA_SMI": smi, "CL_GPU_LOCK": str(self.lock)}
        return subprocess.run([sys.executable, str(CTL), *args], capture_output=True, text=True, env=env)

    def test_busy_gpu_refuses_the_job_and_says_how_much(self):
        out = self.run_ctl(self.smi("print('1200, 6988, 8188')"), "gpu-acquire", "T1")
        self.assertEqual(out.returncode, 3)
        self.assertIn("5000 MB", out.stdout)
        self.assertIn("1200 MB", out.stdout)
        self.assertFalse(self.lock.exists())

    def test_free_gpu_takes_the_turn_and_release_frees_it(self):
        self.assertEqual(self.run_ctl(self.smi("print('7000, 1188, 8188')"), "gpu-acquire", "T1").returncode, 0)
        self.assertTrue(self.lock.exists())
        self.assertEqual(self.run_ctl("unused", "gpu-release", "T1").returncode, 0)
        self.assertFalse(self.lock.exists())

    def test_jobs_share_the_gpu_while_reserved_memory_fits(self):
        (self.root / "orchestration" / "gpu_needs.json").write_text(json.dumps({"T1": 3000, "T2": 3000, "T3": 3000, "T4": 100, "_exclusive": ["T4"]}))
        smi = self.smi("print('7000, 1188, 8188')")
        self.assertEqual(self.run_ctl(smi, "gpu-acquire", "T1").returncode, 0)
        self.assertEqual(self.run_ctl(smi, "gpu-acquire", "T2").returncode, 0)
        full = self.run_ctl(smi, "gpu-acquire", "T3")
        self.assertEqual(full.returncode, 3)
        self.assertIn("reserved", full.stdout)
        self.assertEqual(self.run_ctl(smi, "gpu-acquire", "T4").returncode, 3)
        self.assertEqual(self.run_ctl("unused", "gpu-release", "T1").returncode, 0)
        self.assertEqual(self.run_ctl(smi, "gpu-acquire", "T3").returncode, 0)
        self.assertNotEqual(self.run_ctl("unused", "gpu-release", "T9").returncode, 0)

    def test_an_exclusive_job_runs_alone(self):
        (self.root / "orchestration" / "gpu_needs.json").write_text(json.dumps({"T1": 100, "T2": 100, "_exclusive": ["T1"]}))
        smi = self.smi("print('7000, 1188, 8188')")
        self.assertEqual(self.run_ctl(smi, "gpu-acquire", "T1").returncode, 0)
        self.assertEqual(self.run_ctl(smi, "gpu-acquire", "T2").returncode, 3)

    def test_unreadable_gpu_refuses(self):
        out = self.run_ctl(self.smi("raise SystemExit(1)"), "gpu-acquire", "T1")
        self.assertEqual(out.returncode, 3)
        self.assertIn("GPU CHECK FAILED", out.stdout)
        self.assertEqual(self.run_ctl("no-such-nvidia-smi-tool", "gpu", "T1").returncode, 3)

    def test_task_without_a_gpu_need_is_not_checked(self):
        self.write([task("T1"), task("T9")])
        self.assertEqual(self.run_ctl("no-such-nvidia-smi-tool", "gpu-acquire", "T9").returncode, 0)

    def test_gpu_tasks_always_start_so_their_cpu_work_can_run(self):
        (self.root / "orchestration" / "gpu_needs.json").write_text(json.dumps({"T1": 5000, "T2": 5000}))
        self.write([task("T1"), task("T2"), task("T3")])
        self.assertEqual(self.run_ctl("unused", "start", "T1").returncode, 0)
        state = json.loads(self.run_ctl("unused", "next").stdout)
        self.assertEqual([t["id"] for t in state["ready"]], ["T2", "T3"])
        self.assertEqual(self.run_ctl("unused", "start", "T2").returncode, 0)


class Tracks(Sandbox):
    def setUp(self):
        super().setUp()
        self.wt_root = Path(self.tmp.name + "-wt")
        self.lock = self.root / "gpu.lock"
        (self.root / ".gitignore").write_text("orchestration/\ndatasets/raw/\n")
        self.git("add", ".gitignore")
        self.git("commit", "-qm", "ignore")
        self.branch = self.git("branch", "--show-current")
        self.write([task("T1"), task("T2"), task("T3", deps=["T1"]), task("H1", owner="human")])

    def tearDown(self):
        shutil.rmtree(self.wt_root, ignore_errors=True)
        super().tearDown()

    def ctl(self, *args):
        env = {**os.environ, "CL_ROOT": str(self.root), "CL_WT_ROOT": str(self.wt_root), "CL_GPU_LOCK": str(self.lock)}
        return subprocess.run([sys.executable, str(CTL), *args], capture_output=True, text=True, env=env)

    def build(self, task_id, name, text):
        """Plays the implementer: commits a file in the task's worktree and returns its commit."""
        path = Path(self.state(task_id)["worktree"])
        (path / name).write_text(text)
        subprocess.run(["git", "add", name], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", task_id], cwd=path, check=True, capture_output=True)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()

    def pass_gates(self, task_id, commit):
        self.assertEqual(self.ctl("verify", task_id).returncode, 0)
        self.verdict(task_id, commit=commit)

    def test_next_offers_every_independent_task(self):
        state = json.loads(self.ctl("next").stdout)
        self.assertEqual([t["id"] for t in state["ready"]], ["T1", "T2"])
        self.ctl("start", "T1", "--worktree")
        state = json.loads(self.ctl("next").stdout)
        self.assertEqual([t["id"] for t in state["ready"]], ["T2"])
        self.assertEqual([t["id"] for t in state["active"]], ["T1"])

    def test_two_tracks_land_together_without_touching_the_main_folder(self):
        self.ctl("start", "T1", "--worktree")
        self.ctl("start", "T2", "--worktree")
        c1 = self.build("T1", "one.txt", "1")
        c2 = self.build("T2", "two.txt", "2")
        self.assertFalse((self.root / "one.txt").exists())
        self.pass_gates("T1", c1)
        self.pass_gates("T2", c2)
        self.assertEqual(self.ctl("land", "T1").returncode, 0)
        out = self.ctl("land", "T2")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertTrue((self.root / "one.txt").exists() and (self.root / "two.txt").exists())
        self.assertEqual(self.state("T2")["status"], "done")
        self.assertNotIn("worktree", self.state("T2"))
        self.assertNotIn("task/T2", self.git("branch"))
        self.assertEqual(json.loads(self.ctl("next").stdout)["task"]["id"], "T3")

    def test_land_needs_verify_and_a_verifier_pass(self):
        self.ctl("start", "T1", "--worktree")
        commit = self.build("T1", "one.txt", "1")
        self.assertIn("have not passed", self.ctl("land", "T1").stderr)
        self.ctl("verify", "T1")
        self.assertIn("no verifier verdict", self.ctl("land", "T1").stderr)
        self.verdict("T1", "FAIL", commit=commit)
        self.assertNotEqual(self.ctl("land", "T1").returncode, 0)
        self.verdict("T1", commit="0" * 40)
        self.assertNotEqual(self.ctl("land", "T1").returncode, 0)
        self.assertFalse((self.root / "one.txt").exists())
        self.assertEqual(self.state("T1")["status"], "in_progress")

    def worktree_log(self, task_id, text):
        path = Path(self.state(task_id)["worktree"])
        (path / "docs" / "DECISIONS.md").write_text(text)
        subprocess.run(["git", "add", "docs/DECISIONS.md"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "decisions"], cwd=path, check=True, capture_output=True)

    def test_land_is_refused_without_an_entry_and_moves_nothing(self):
        self.ctl("start", "T1", "--worktree")
        self.worktree_log("T1", "T9: no new decisions\n")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        before = self.git("rev-parse", "HEAD")
        out = self.ctl("land", "T1")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("docs/DECISIONS.md", out.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), before)
        self.assertFalse((self.root / "one.txt").exists())
        self.assertEqual(self.state("T1")["status"], "in_progress")
        self.assertTrue(Path(self.state("T1")["worktree"]).exists())
        self.assertIn("task/T1", self.git("branch"))

    def test_land_goes_by_the_worktrees_copy_of_the_log(self):
        (self.root / "docs" / "DECISIONS.md").write_text("T9: no new decisions\n")
        self.git("add", "docs/DECISIONS.md")
        self.git("commit", "-qm", "main log lacks T1")
        self.ctl("start", "T1", "--worktree")
        self.worktree_log("T1", "### T1 - chose a thing\n")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        out = self.ctl("land", "T1")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(self.state("T1")["status"], "done")

    def test_land_word_boundary(self):
        self.write([task("T4.1"), task("T4.10")])
        self.ctl("start", "T4.1", "--worktree")
        self.worktree_log("T4.1", "### T4.10 - someone else's entry\n")
        commit = self.build("T4.1", "one.txt", "1")
        self.pass_gates("T4.1", commit)
        self.assertIn("docs/DECISIONS.md", self.ctl("land", "T4.1").stderr)
        self.worktree_log("T4.1", "### T4.10 - someone else's entry\nT4.1: no new decisions\n")
        commit = self.build("T4.1", "two.txt", "2")
        self.pass_gates("T4.1", commit)
        self.assertEqual(self.ctl("land", "T4.1").returncode, 0)

    def test_done_refuses_a_worktree_task(self):
        self.ctl("start", "T1", "--worktree")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        self.assertIn("use land", self.ctl("done", "T1").stderr)

    def test_a_merge_conflict_lands_nothing_and_keeps_both_sides(self):
        self.ctl("start", "T1", "--worktree")
        self.ctl("start", "T2", "--worktree")
        c1 = self.build("T1", "same.txt", "one")
        c2 = self.build("T2", "same.txt", "two")
        self.pass_gates("T1", c1)
        self.pass_gates("T2", c2)
        self.assertEqual(self.ctl("land", "T1").returncode, 0)
        out = self.ctl("land", "T2")
        self.assertEqual(out.returncode, 4)
        self.assertIn("MERGE CONFLICT", out.stdout)
        self.assertEqual((self.root / "same.txt").read_text(), "one")
        self.assertEqual(self.state("T2")["status"], "in_progress")
        worktree = self.state("T2")["worktree"]
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(subprocess.run(["git", "status", "--porcelain"], cwd=worktree, capture_output=True, text=True).stdout, "")

    def test_code_that_breaks_after_merging_does_not_land(self):
        check = "import pathlib,sys; sys.exit(pathlib.Path('bad.txt').exists())"
        self.write([task("T1"), task("T2", acceptance=[f'"{sys.executable}" -c "{check}"'])])
        self.ctl("start", "T1", "--worktree")
        self.ctl("start", "T2", "--worktree")
        c1 = self.build("T1", "bad.txt", "breaks T2")
        c2 = self.build("T2", "two.txt", "2")
        self.pass_gates("T1", c1)
        self.pass_gates("T2", c2)
        self.assertEqual(self.ctl("land", "T1").returncode, 0)
        out = self.ctl("land", "T2")
        self.assertEqual(out.returncode, 5, out.stdout + out.stderr)
        self.assertFalse((self.root / "two.txt").exists())
        self.assertEqual(self.state("T2")["status"], "in_progress")

    def test_shared_data_is_linked_and_survives_removing_the_worktree(self):
        data = self.root / "datasets" / "raw" / "big"
        data.mkdir(parents=True)
        (data / "image.txt").write_text("keep me")
        self.ctl("start", "T1", "--worktree")
        linked = Path(self.state("T1")["worktree"]) / "datasets" / "raw" / "big" / "image.txt"
        self.assertEqual(linked.read_text(), "keep me")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        self.assertEqual(self.ctl("land", "T1").returncode, 0)
        self.assertEqual((data / "image.txt").read_text(), "keep me")
        self.assertFalse(linked.exists())

    def ignore_outputs(self):
        (self.root / ".gitignore").write_text("orchestration/\ndatasets/raw/\nmodels/\nmlruns/\nresults/calls.json\n__pycache__/\n")
        self.git("add", ".gitignore")
        self.git("commit", "-qm", "ignore outputs")

    def test_models_written_in_a_worktree_survive_landing(self):
        self.ignore_outputs()
        self.assertEqual(self.ctl("start", "T1", "--worktree").returncode, 0)
        worktree = Path(self.state("T1")["worktree"])
        (worktree / "models" / "run").mkdir(parents=True)
        (worktree / "models" / "run" / "best.pt").write_text("weights")
        (self.root / "models" / "run" / "best.pt").read_text()  # one folder, seen from both sides
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        self.assertEqual(self.ctl("land", "T1").returncode, 0)
        self.assertEqual((self.root / "models" / "run" / "best.pt").read_text(), "weights")

    def test_a_real_output_folder_from_before_the_fix_is_moved_on_landing(self):
        self.ignore_outputs()
        self.ctl("start", "T1", "--worktree")
        worktree = Path(self.state("T1")["worktree"])
        for folder in ("models", "mlruns"):
            link = worktree / folder
            os.rmdir(link) if os.path.isjunction(link) else link.unlink()
            link.mkdir()
            (link / "kept.txt").write_text(folder)
        (worktree / "results").mkdir()
        (worktree / "results" / "calls.json").write_text("7")
        (worktree / "__pycache__").mkdir()
        (worktree / "__pycache__" / "x.pyc").write_text("junk")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        out = self.ctl("land", "T1")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual((self.root / "models" / "kept.txt").read_text(), "models")
        self.assertEqual((self.root / "mlruns" / "kept.txt").read_text(), "mlruns")
        self.assertEqual((self.root / "results" / "calls.json").read_text(), "7")
        self.assertFalse((self.root / "__pycache__").exists())

    def test_a_differing_ignored_file_refuses_to_land_and_deletes_nothing(self):
        self.ignore_outputs()
        self.ctl("start", "T1", "--worktree")
        worktree = Path(self.state("T1")["worktree"])
        (self.root / "results").mkdir()
        (self.root / "results" / "calls.json").write_text("main")
        (worktree / "results").mkdir()
        (worktree / "results" / "calls.json").write_text("worktree")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        out = self.ctl("land", "T1")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("nothing landed", out.stderr)
        self.assertEqual((worktree / "results" / "calls.json").read_text(), "worktree")
        self.assertEqual((self.root / "results" / "calls.json").read_text(), "main")
        self.assertEqual(self.state("T1")["status"], "in_progress")
        self.assertFalse((self.root / "one.txt").exists())

    def test_tracked_data_folder_links_only_the_ignored_children(self):
        (self.root / ".gitignore").write_text("orchestration/\ndatasets/raw/*\n!datasets/raw/pointer.dvc\n")
        data = self.root / "datasets" / "raw"
        data.mkdir(parents=True)
        (data / "pointer.dvc").write_text("tracked")
        (data / "lab").mkdir()
        (data / "lab" / "a.txt").write_text("image")
        self.git("add", ".gitignore", "datasets/raw/pointer.dvc")
        self.git("commit", "-qm", "pointer")
        self.ctl("start", "T1", "--worktree")
        worktree = Path(self.state("T1")["worktree"])
        self.assertEqual((worktree / "datasets/raw/lab/a.txt").read_text(), "image")
        self.assertEqual(subprocess.run(["git", "status", "--porcelain"], cwd=worktree, capture_output=True, text=True).stdout, "")
        commit = self.build("T1", "one.txt", "1")
        self.pass_gates("T1", commit)
        self.assertEqual(self.ctl("land", "T1").returncode, 0)
        self.assertEqual((data / "lab" / "a.txt").read_text(), "image")

    def test_state_survives_many_calls_at_once(self):
        tasks = [task(f"T{n}") for n in range(1, 13)]
        self.write(tasks)
        env = {**os.environ, "CL_ROOT": str(self.root), "CL_WT_ROOT": str(self.wt_root), "CL_GPU_LOCK": str(self.lock)}
        running = [subprocess.Popen([sys.executable, str(CTL), "start", t["id"]], env=env, stdout=subprocess.DEVNULL) for t in tasks]
        self.assertEqual([p.wait() for p in running], [0] * 12)
        self.assertEqual({self.state(t["id"])["status"] for t in tasks}, {"in_progress"})


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

    def test_gpu_needs_name_real_agent_tasks(self):
        needs = json.loads((REPO / "orchestration" / "gpu_needs.json").read_text(encoding="utf-8"))
        agent_tasks = {t["id"] for t in self.tasks if t["owner"] == "agent"}
        exclusive = needs.pop("_exclusive", [])
        self.assertTrue(needs)
        self.assertTrue(set(needs) <= agent_tasks, set(needs) - agent_tasks)
        self.assertTrue(set(exclusive) <= agent_tasks)

    def test_only_the_named_tasks_skip_the_verifier(self):
        exempt = {t["id"] for t in self.tasks if t.get("verifier") is False}
        self.assertEqual(exempt, {"T5.1", "T5.2", "T8.2"})

    def test_every_agent_declares_a_model(self):
        for path in (REPO / ".claude" / "agents").glob("*.md"):
            self.assertRegex(path.read_text(encoding="utf-8"), r"(?m)^model: (haiku|sonnet)\s*$", path.name)


class ReadOnlyGit(unittest.TestCase):
    """A read-only status must not take .git/index.lock: a killed process leaves it behind and blocks everyone."""

    def test_status_calls_pass_no_optional_locks(self):
        import importlib.util
        from unittest import mock

        spec = importlib.util.spec_from_file_location("taskctl_under_test", CTL)
        ctl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ctl)
        seen = []
        real = subprocess.run

        def spy(cmd, *a, **kw):
            seen.append(cmd)
            return real(cmd, *a, **kw)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ctl.subprocess, "run", spy):
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            ctl.dirty(tmp)
            ctl.git("--no-optional-locks", "status", "--porcelain", "--ignored", cwd=tmp)
            self.assertFalse((Path(tmp) / ".git" / "index.lock").exists())
        self.assertTrue(seen and all(c[:2] == ["git", "--no-optional-locks"] for c in seen if "status" in c), seen)

    def test_no_source_file_runs_a_plain_git_status(self):
        for path in [CTL, REPO / "src" / "sprout" / "eval" / "records.py"]:
            for line in path.read_text(encoding="utf-8").splitlines():
                if '"status",' in line and "git(" in line:
                    self.assertIn('"--no-optional-locks", "status"', line, f"{path.name}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
