import json

import pytest
import test_records

from sprout.eval import check_cache, check_run

make = test_records.make
root = test_records.root  # the fixture


@pytest.fixture
def calls(monkeypatch):
    """Names of the records that were fully checked, not served from the cache."""
    seen = []
    real = check_run.check_file

    def spy(path, *a, **kw):
        seen.append(path.name)
        return real(path, *a, **kw)

    monkeypatch.setattr(check_run, "check_file", spy)
    return seen


def run(root, **kw):
    return check_run.check("E1", root=root, use_cache=True, **kw)


def test_unchanged_record_is_skipped(root, calls):
    make(root)
    assert run(root) == []
    assert len(calls) == 1
    assert run(root) == []
    assert len(calls) == 1


def test_changed_record_is_rechecked(root, calls):
    path = make(root)
    other = make(root, split="lab_dev")
    assert run(root) == []
    calls.clear()
    rec = json.loads(path.read_text())
    rec["notes"] = "edited"
    path.write_text(json.dumps(rec))
    assert run(root) == []
    assert calls == [path.name]
    assert other.name not in calls


def test_changed_checker_rechecks_everything(root, calls, monkeypatch):
    make(root)
    make(root, split="lab_dev")
    run(root)
    calls.clear()
    monkeypatch.setattr(check_cache, "code_fingerprint", lambda: "a different checker")
    run(root)
    assert len(calls) == 2


def test_checker_fingerprint_follows_imports():
    files, outside = check_cache._closure(check_cache.ENTRY_MODULE)
    names = {p.name for p in files}
    assert {"check_run.py", "records.py", "metrics.py", "stats.py", "check_cache.py", "atomic.py"} <= names
    assert "numpy" in outside


def test_changed_manifest_rechecks_everything(root, calls):
    make(root)
    make(root, split="lab_dev")
    run(root)
    calls.clear()
    manifest = root / "datasets/manifests/lab.csv"
    manifest.write_text(manifest.read_text() + "\n")
    run(root)
    assert len(calls) == 2


def test_changed_exclusion_list_rechecks_everything(root, calls):
    make(root)
    run(root)
    calls.clear()
    log = root / "datasets/label_exclusions.csv"
    log.write_text("version,image_id\n1,abc\n")
    run(root)
    assert len(calls) == 1


def test_changed_prediction_file_rechecks_the_record(root, calls):
    path = make(root)
    run(root)
    calls.clear()
    pred = next(root.glob("results/runs/E1-*.a.predictions.jsonl"))
    pred.write_text(pred.read_text() + "\n")
    run(root)
    assert calls == [path.name]


def test_failed_record_is_not_cached(root, calls):
    path = make(root)
    rec = json.loads(path.read_text())
    rec["metrics"]["a"]["accuracy"] = 0.0
    path.write_text(json.dumps(rec))
    assert run(root) != []
    assert run(root) != []
    assert len(calls) == 2


def test_all_ignores_the_cache(root, calls):
    make(root)
    run(root)
    calls.clear()
    assert check_run.check("E1", root=root, use_cache=False) == []
    assert len(calls) == 1


def test_cli_all_flag(root, calls, monkeypatch):
    make(root)
    monkeypatch.setattr(check_run, "ROOT", root)
    assert check_run.main(["E1"]) == 0
    assert check_run.main(["E1"]) == 0
    assert len(calls) == 1
    assert check_run.main(["E1", "--all"]) == 0
    assert len(calls) == 2


def test_cache_lives_in_the_git_directory(root):
    make(root)
    run(root)
    assert check_cache.cache_path(root).is_file()
    assert check_cache.cache_path(root).is_relative_to((root / ".git").resolve())
