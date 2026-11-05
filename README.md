# SPROUT V2

Plant disease identification that is measured on real web photos, not just lab images, plus a care assistant whose advice points back to its sources. This folder holds the whole rebuild: the old code, the design, the plan, and the system that runs the plan.

## What it does, and what has been measured

- **Identify:** a ConvNeXt V2 Tiny classifier over 29 plant-and-condition classes (14 plants), with a photo gate, an abstention threshold and a stricter bar for "healthy" labels. A healthy result never rules out disease.
- **Care assistant:** hybrid retrieval with a reranker over extension and agency documents; an answer cites its documents, says so when none supports it, and tells the user to check the product label and local rules for chemical or disposal advice.
- **Results:** `docs/REPORT.md` has every number, the splits, the failures and the limits. All test numbers are provisional because the photo labels are not signed. Real photos here are web photos; no phone photos were tested. The headline is a seen-source number; a web source the model has not trained on scores far lower (E4), and the final model has no held-out-source run.
- **Not claimed:** no speed or concurrency claim (the container check is 5 requests, and the quantized model is smaller but slower on CPU than PyTorch), no "deployed" (the app is containerized, not deployed), no claim about unsupported plants or diseases (never measured), and the wrong-disease check on answers is not validated.
- **Resume bullets** are in `docs/RESUME-TARGETS.md`; every number in them is in `results/claims.json` and tied to a run record. `python -m sprout.eval.claims check` verifies them and `tools/check_report_numbers.py` checks the report and bullets.

## Run the app

```
docker compose build app
SPROUT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" docker compose up app
```

The model and index are mounted from `models/`, the writer runs in Ollama on the host, and nothing is deployed anywhere. `docs/deploy.md` has the settings. There is no hosted demo and no sample photos are included, because the photos are third-party web photos that cannot be redistributed.

## Start it

```
cd <local path>
claude --model opus --remote-control
```

Say yes when Claude Code asks whether to trust the folder, or its permissions won't load. Then type `/run-plan 40`. It keeps going until it
finishes or needs you. `/plan-status` shows where things stand.

Don't use `tools/run_loop.py` for this project: each of its sessions exits when its turn ends, which would kill a training run in the background.

No API keys are needed. Needs Docker Desktop and Ollama running.

**Don't train at the same time as ContextLensV2.** They share one 8 GB GPU. If there isn't enough free GPU memory for a task, the run stops immediately and tells you how much it needs.

## Set up the environment

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev]"
```

The default Windows torch wheel is CPU-only, so install the CUDA build first. `cu128` matches this driver (CUDA 13.1 reported by `nvidia-smi`). CI uses the CPU wheel.

## How a task gets finished

```
taskctl next ──> up to 3 tracks, each in its own git worktree: implementer builds, test first, and commits
                     │
                     ▼   (experiments: the orchestrator runs them and commits the results)
              taskctl verify      reruns the plan's acceptance commands, keeps a log
                     │
                     ▼
              verifier agent      fresh eyes, no build context, tries to break it
                     │
         ┌──── FAIL ─┴─ PASS ────┐
         ▼                       ▼
  back to implementer       taskctl land      refused unless both passed; merges, retests, then moves the build branch forward
  (3 tries, then blocked)        │
                                 ▼
                        phase ends? reviewer checks the whole phase
```

## What's where

| Path | What it is |
|---|---|
| `legacy/PlantProject/` | The original code, unchanged, without user data |
| `docs/DESIGN.md` | What's being built and why |
| `docs/PLAN.md` | Every task, with files, interfaces, tests |
| `docs/TESTING.md` | Test layers, metrics, experiments |
| `docs/HUMAN-GATES.md` | The parts only you can do |
| `docs/RESUME-TARGETS.md` | Target bullets, and the evidence each number needs |
| `orchestration/tasks.json` | Live task state, changed only through `tools/taskctl.py` |
| `results/runs/` | Experiment records, the only source of numbers |
| `results/claims.json` | Numbers allowed onto the resume, each tied to a run |
