# Code map

Package `sprout` is in `src/sprout/`. Run Python with `PYTHONPATH=src` (pytest sets it). Paths below are from the repo root.

## Parts

| Part | Entry point | What it holds |
|---|---|---|
| Data | `src/sprout/data/splits.py`, `check_splits.py` | Manifests (`datasets/manifests/*.csv`), group-wise splits, dataset hash, lab and field loading (`field.py`), masks, class report |
| Training | `src/sprout/train/loop.py` (`train`) | Config (`config.py`), loaders and class map (`dataset.py`), augmentation, checkpoints, joint training |
| Baselines | `src/sprout/baselines/` | `classic.py`, `dino.py`, `frozen.py` (E3 comparisons) |
| Segmentation | `src/sprout/segment/` | Leaf crop model and data (E10) |
| Explain | `src/sprout/explain/gradcam.py` | Grad-CAM |
| Serving | `src/sprout/serve/gate.py` | Input gate, calibration, shared result types |
| RAG | `src/sprout/rag/retrieve.py` | Ingest (`ingest.py`), retrieval, benchmark questions (`evalset.py`) |
| Web app | `deploy/wsgi.py`, `src/sprout/web/pages.py` | Flask pages, auth, uploads, database, templates and static files |
| Experiments | `src/sprout/experiments/e1_lab.py` ... `e5_confidence.py`, `e10_segment.py`, `e12_more_sources.py`, `e14_judge_check.py`, `e9_final.py` | One module per experiment, each with `main()`; settings in `configs/e*.json` |
| Records and checks | `src/sprout/eval/records.py` (`write_run`), `check_run.py` | Run records, metrics (`metrics.py`), bootstrap (`stats.py`), claims (`claims.py`), top-k baselines |
| Grader agreement | `src/sprout/eval/agreement.py` | Cohen's kappa, direction of disagreement and bootstrap intervals for E14, shared with `check_run.py` |
| Wrong-disease numbers | `src/sprout/eval/answer_checks.py` | The strict FITS/WRONG reading, rates, Wilson intervals, the blind sample and the rag_test judged samples, shared by E9 and `check_run.py` |
| Check cache | `src/sprout/eval/check_cache.py` | Which records already passed `check_run` |
| Experiment logging | `src/sprout/eval/mlflow_log.py` | Copies each record into `mlruns/` |
| Orchestration | `tools/taskctl.py` | Task status, verify, land; plan in `orchestration/tasks.json` |
| Doc checks | `tools/check_datasets_doc.py`, `tools/check_candidates_doc.py` | Keep `docs/datasets.md` and `docs/document-candidates.md` in shape |
| Backfill | `tools/backfill_mlflow.py` | Logs existing records into MLflow |

## Where to change common things

- A new experiment: add `src/sprout/experiments/eN_*.py` and `configs/eN.json`; save results only through `write_run` (also logs to MLflow).
- A new metric: define it in `src/sprout/eval/metrics.py`, add its recompute case in `check_run.py` (`recompute_metric`), and list it in `docs/TESTING.md`.
- A rule a record must satisfy: `check_run.py` (`check_record` and the `_..._problems` helpers). The cache invalidates itself when this file changes.
- What a split is: `src/sprout/data/splits.py` and the manifests; `src/sprout/eval/records.py` (`split_info`) reads them for checks.
- Training settings: `src/sprout/train/config.py` (fields) and `configs/e*.json` (values).
- What goes in MLflow: `src/sprout/eval/mlflow_log.py`.
- Dependencies: `pyproject.toml`.
- Plan tasks and acceptance commands: `orchestration/tasks.json` (edit through the orchestrator only); numbers in docs only through `results/claims.json`.

## Checking

```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_<area>.py -q   # one named file
.venv/Scripts/python -m ruff check .
PYTHONPATH=src .venv/Scripts/python -m sprout.eval.check_run E1 --min-n 1000 --split lab_dev   # --all ignores the cache
```

Tests are in `tests/` (`test_records.py` and `test_check_cache.py` for records and checks, `test_train.py` for training).
Fast tests use fakes and tiny fixtures; tests that need a GPU, Ollama or a network are marked `real`. Run the full suite one process at a time.
