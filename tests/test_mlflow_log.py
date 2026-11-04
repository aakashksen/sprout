import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import test_records

from sprout.eval import mlflow_log, records

make = test_records.make
root = test_records.root  # the fixture

TOOL = Path(__file__).resolve().parents[1] / "tools" / "backfill_mlflow.py"
spec = importlib.util.spec_from_file_location("backfill_mlflow", TOOL)
backfill_mlflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill_mlflow)


def runs_of(root, experiment="E1"):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_log.store_uri(root))
    exp = client.get_experiment_by_name(experiment)
    return client, ([] if exp is None else client.search_runs([exp.experiment_id]))


def fingerprint(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "results/runs").iterdir()}


def test_write_run_logs_to_mlflow(root):
    path = make(root)
    rec = json.loads(path.read_text())
    _, runs = runs_of(root)
    assert [r.data.tags["run_id"] for r in runs] == [rec["run_id"]]
    assert runs[0].data.metrics["a.accuracy"] == rec["metrics"]["a"]["accuracy"]


def test_logging_failure_warns_and_keeps_the_record(root, monkeypatch):
    def boom(*a, **k):
        raise OSError("store unreachable")

    monkeypatch.setattr(records, "log_record", boom)
    with pytest.warns(UserWarning, match="not logged to MLflow"):
        path = make(root)
    assert json.loads(path.read_text())["run_id"] == path.stem


def test_only_finite_metrics_and_the_device_are_logged(root):
    rec = json.loads(make(root).read_text())
    rec["run_id"] = "E1-20990101T000000000000Z"
    rec["metrics"]["a"].update(nan=float("nan"), inf=float("inf"), none=None, yes=True)
    rec["config"]["device"] = "cuda"
    rec["config"]["table"] = {f"Cherry_(sour){i}": i for i in range(30)}
    rec["config"]["small"] = {"Cherry_(sour)": 1}
    mlflow_log.log_record(rec, root)
    client, runs = runs_of(root)
    run = next(r for r in runs if r.data.tags["run_id"] == rec["run_id"])
    assert not {"a.nan", "a.inf", "a.none", "a.yes"} & set(run.data.metrics)
    assert run.data.params["device"] == "cuda"
    assert run.data.params["config.small.Cherry_(sour)".replace("(", "_").replace(")", "_")] == "1"
    assert client is not None


def test_backfill_logs_once_and_changes_no_record(root, monkeypatch):
    monkeypatch.setattr(records, "log_record", lambda *a, **k: None)  # records that predate logging
    make(root, split="lab_dev")
    make(root, split="lab_test", experiment="E2")
    monkeypatch.undo()
    before = fingerprint(root)
    logged, skipped = backfill_mlflow.backfill(root)
    assert dict(logged) == {"E1": 1, "E2": 1} and not skipped
    logged, skipped = backfill_mlflow.backfill(root)
    assert not logged and dict(skipped) == {"E1": 1, "E2": 1}
    assert fingerprint(root) == before
    assert len(runs_of(root, "E1")[1]) == 1
