"""Log a run record to the local MLflow file store. The record stays the source of truth: this only copies it."""
from __future__ import annotations

import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

STORE = "mlruns"
PARAM_LIMIT = 100
METRIC_LIMIT = 1000
VALUE_CHARS = 250
BAD_NAME = re.compile(r"[^A-Za-z0-9_\-. /]")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _params(rec: dict[str, Any]) -> dict[str, str]:
    """Scalars from the record and its config (two levels deep); lists and long structures are left in the record."""
    out: dict[str, Any] = {k: rec.get(k) for k in ("split", "subset", "n", "dataset_hash", "git_commit")}
    out.update({f"model.{k}": v for k, v in (rec.get("models") or {}).items()})
    out["device"] = rec.get("device") or (rec.get("config") or {}).get("device")
    for key, value in (rec.get("config") or {}).items():
        if isinstance(value, dict):
            out.update({f"config.{key}.{k}": v for k, v in value.items()})
        else:
            out[f"config.{key}"] = value
    return {k: str(v)[:VALUE_CHARS] for k, v in out.items() if isinstance(v, (str, int, float, bool)) and v != ""}


def _metrics(rec: dict[str, Any]) -> list[tuple[str, float, int]]:
    """(name, value, step): each system's finite metrics, and any per-epoch history an attempt saved."""
    out = [(BAD_NAME.sub("_", f"{system}.{name}"), float(v), 0)
           for system, values in (rec.get("metrics") or {}).items() for name, v in values.items() if _finite(v)]
    for attempt in rec.get("attempts") or []:
        for row in attempt.get("history") or [] if isinstance(attempt, dict) else []:
            out += [(BAD_NAME.sub("_", k), float(v), int(row["epoch"]))
                    for k, v in row.items() if k != "epoch" and _finite(v) and "epoch" in row]
    return out


def _millis(stamp: Any) -> int | None:
    try:
        return int(datetime.fromisoformat(stamp).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def store_uri(root: Path) -> str:
    return (root / STORE).as_uri()


def log_record(rec: dict[str, Any], root: Path, *, skip_if_logged: bool = False) -> bool:
    """Create one MLflow run for a record, tagged with its run_id. Returns False if it was already logged
    and skip_if_logged is set. Raises on any MLflow problem; write_run turns that into a warning."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # mlflow-skinny has no database backend
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=store_uri(root))
    experiment = client.get_experiment_by_name(rec["experiment"])
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(rec["experiment"])
    if skip_if_logged and client.search_runs([experiment_id], f"tags.run_id = '{rec['run_id']}'", max_results=1):
        return False
    started, finished = _millis(rec.get("started_at")), _millis(rec.get("finished_at"))
    tags = {"run_id": rec["run_id"], "git_commit": str(rec.get("git_commit")), "split": str(rec.get("split"))}
    run = client.create_run(experiment_id, start_time=started, tags=tags, run_name=rec["run_id"])
    params = [Param(k, v) for k, v in _params(rec).items()]
    stamp = finished or started or 0
    metrics = [Metric(k, v, stamp, step) for k, v, step in _metrics(rec)]
    for i in range(0, len(params), PARAM_LIMIT):
        client.log_batch(run.info.run_id, params=params[i:i + PARAM_LIMIT])
    for i in range(0, len(metrics), METRIC_LIMIT):
        client.log_batch(run.info.run_id, metrics=metrics[i:i + METRIC_LIMIT])
    client.log_batch(run.info.run_id, tags=[RunTag("record_file", f"results/runs/{rec['run_id']}.json")])
    client.set_terminated(run.info.run_id, end_time=finished)
    return True
