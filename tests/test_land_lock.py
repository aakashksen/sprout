"""A land takes minutes, so a stale index.lock at its last step must not lose it."""
import importlib.util
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("taskctl_under_test", ROOT / "tools" / "taskctl.py")
taskctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(taskctl)


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "build")
    (tmp_path / "a.txt").write_text("a")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "first")
    git(tmp_path, "checkout", "-q", "-b", "task/x")
    (tmp_path / "a.txt").write_text("b")
    git(tmp_path, "commit", "-q", "-am", "second")
    git(tmp_path, "checkout", "-q", "build")
    return tmp_path


def lock_at(repo, age_s, body=b""):
    lock = repo / ".git" / "index.lock"
    lock.write_bytes(body)
    os.utime(lock, (time.time() - age_s, time.time() - age_s))
    return lock


def test_a_stale_lock_is_cleared_and_the_move_succeeds(repo):
    lock = lock_at(repo, 600)
    assert taskctl.move_forward("task/x", cwd=repo, wait_s=0).returncode == 0
    assert not lock.exists()
    assert (repo / "a.txt").read_text() == "b"


def test_a_recent_lock_is_left_alone_and_reported(repo):
    lock = lock_at(repo, 0)
    done = taskctl.move_forward("task/x", cwd=repo, tries=2, wait_s=0)
    assert done.returncode != 0 and "index.lock" in done.stderr
    assert lock.exists()


def test_a_lock_with_content_is_left_alone(repo):
    lock = lock_at(repo, 600, b"x")
    assert taskctl.move_forward("task/x", cwd=repo, tries=2, wait_s=0).returncode != 0
    assert lock.exists()


def test_another_failure_is_not_retried(repo, monkeypatch):
    calls = []
    real = taskctl.git_ok
    monkeypatch.setattr(taskctl, "git_ok", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert taskctl.move_forward("no-such-ref", cwd=repo, wait_s=0).returncode != 0
    assert len(calls) == 1
