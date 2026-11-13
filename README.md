# SPROUT V2

Plant disease identification that is measured on real web photos, not just lab images, plus a care assistant whose advice points back to its sources. The repository holds the training and evaluation code, the Flask app, the design and plan documents, the run records behind every reported number, and the original code (`legacy/`).

## What it does, and what has been measured

- **Identify:** a ConvNeXt V2 Tiny classifier over 29 plant-and-condition classes (14 plants), with a photo gate, an abstention threshold and a stricter bar for "healthy" labels. A healthy result never rules out disease.
- **Care assistant:** hybrid retrieval with a reranker over extension and agency documents; an answer cites its documents, says so when none supports it, and tells the user to check the product label and local rules for chemical or disposal advice.
- **Headline numbers** (from `docs/REPORT.md`, provisional): on `field_test` (1006 real web photos of the 29 classes) macro-F1 0.8111, accuracy 0.8131, top-3 accuracy 0.9642; on `lab_test` (6443 lab photos) macro-F1 0.9859. A lab-only model scores macro-F1 0.2387 on the same web photos, which is the reason the project exists.
- **Results:** `docs/REPORT.md` has every number, the splits, the failures and the limits. All test numbers are provisional because the photo labels are not signed. Real photos here are web photos; no phone photos were tested. The headline is a seen-source number; a web source the model has not trained on scores far lower (E4), and the final model has no held-out-source run.
- **Not claimed:** no speed or concurrency claim (the container check is 5 requests, and the quantized model is smaller but slower on CPU than PyTorch), no "deployed" (the app is containerized, not deployed), no claim about unsupported plants or diseases (never measured), and the wrong-disease check on answers is not validated.
- **Resume bullets** are in `docs/RESUME-TARGETS.md`; every number in them is in `results/claims.json` and tied to a run record. `python -m sprout.eval.claims check` verifies them and `tools/check_report_numbers.py` checks the report and bullets.

## Run the app

Needs Docker, and Ollama running on the host for the answer writer. No API keys are needed.

```
docker compose build app
SPROUT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" docker compose up app
```

The model and index are mounted from `models/` (git-ignored; not in the repository), the writer runs in Ollama on the host, and nothing is deployed anywhere. `docs/deploy.md` has the settings. There is no hosted demo and no sample photos are included, because the photos are third-party web photos that cannot be redistributed. Datasets, models and the document index are not in the repository either; `docs/datasets.md` and `docs/DESIGN.md` say how they were obtained and built.

## Set up the environment

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

The default Windows torch wheel is CPU-only, so install the CUDA build first. `cu128` matches the driver used here (CUDA 13.1 reported by `nvidia-smi`). CI uses the CPU wheel. Tests need no GPU, network or paid service; tests marked `real` are skipped unless asked for. `docs/TESTING.md` has the test layers.

## What's where

| Path | What it is |
|---|---|
| `src/sprout/` | The package: `serve/` (classifier, gate, calibration), `rag/` (retrieval and answers), `web/` (Flask app), `train/`, `eval/`, `experiments/` |
| `tests/` | Unit and integration tests |
| `configs/` | Experiment configurations |
| `deploy/`, `docker-compose.yml` | Serving image and entry point |
| `tools/` | Scripts for data work, checks and experiment runs |
| `docs/REPORT.md` | Every result, with limits |
| `docs/DESIGN.md`, `docs/PLAN.md`, `docs/DECISIONS.md` | What was built, the task plan, and the decisions with their evidence |
| `docs/TESTING.md`, `docs/deploy.md`, `docs/datasets.md` | Test layers, deployment settings, dataset licenses |
| `docs/RESUME-TARGETS.md`, `results/claims.json` | Target bullets, and the numbers allowed into them, each tied to a run |
| `results/runs/` | Experiment records, the only source of numbers |
| `legacy/PlantProject/` | The original code, unchanged, without user data |

## How this project was built

The work was done task by task from `docs/PLAN.md` using an agent-assisted workflow: `CLAUDE.md` and `.claude/` hold its rules and commands, `orchestration/tasks.json` holds task state (changed only through `tools/taskctl.py`), and each task passed a test run and an independent verifier before it landed. `docs/HANDOFF.md` and `docs/HUMAN-GATES.md` describe the state and the parts only the owner can do. Don't train at the same time as another GPU job: the project assumes one 8 GB GPU, and a task stops immediately if there is not enough free memory.
