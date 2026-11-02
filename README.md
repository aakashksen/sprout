# SPROUT V2

Plant disease identification that's measured on real photos, not just lab images, plus a care assistant whose advice points back to
its sources. This folder holds the whole rebuild: the old code, the design, the plan, and the system that runs the plan.

## Start it

```
cd <local path>
claude --model opus --remote-control
```

Say yes when Claude Code asks whether to trust the folder, or its permissions won't load. Then type `/run-plan 40`. It keeps going until it
finishes or needs you. `/plan-status` shows where things stand.

Don't use `tools/run_loop.py` for this project: each of its sessions exits when its turn ends, which would kill a training run in the background.

No API keys are needed. Needs Docker Desktop and Ollama running.

**Don't train at the same time as ContextLensV2.** They share one 8 GB GPU. If the other project is training, this one finishes its non-GPU tasks and then stops and tells you.

## How a task gets finished

```
taskctl next ──> implementer builds it, test first, and commits
                     │
                     ▼   (experiments: the orchestrator runs them and commits the results)
              taskctl verify      reruns the plan's acceptance commands, keeps a log
                     │
                     ▼
              verifier agent      fresh eyes, no build context, tries to break it
                     │
         ┌──── FAIL ─┴─ PASS ────┐
         ▼                       ▼
  back to implementer       taskctl done      refused unless both passed at this exact commit
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
