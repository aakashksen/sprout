VERDICT: DEFECTS

Scope: T2.1, T2.2, T2.3 on `build`. The 83 phase tests pass (`CUDA_VISIBLE_DEVICES=-1`). `claims check` on the real repo prints "claims ok". No tests are skipped or xfailed, and I found no broad exception handlers in `src/sprout/train` or `src/sprout/eval`.

## Integration run (CPU, temporary git repo)
I copied `src/` into a fresh temporary git repo under the scratchpad. That makes the package's `ROOT` and `MANIFEST_DIR` point at the temp repo, so the real `results/runs/` was untouched. The repo had a 30-image lab manifest.

- `train` (tiny_cnn, 2 epochs) ran on CPU, then I loaded `best.pt`, ran `predict` on `lab_dev`, and called `write_run`.
- `python -m sprout.eval.check_run E0` printed "E0 ok".
- `python -m sprout.eval.claims check` printed "claims ok".
- The interfaces line up:
  - `predict` returns paths in the same order as `split_info`.
  - The dataset hash from `train` equals the hash from `split_info`.
  - `Predictions` goes into `write_run` unchanged.
  - The `recompute_*` helpers produce values that `check_run` accepts.
- A later experiment still has to supply four things itself. None of them is a defect.
  - `config.class_names`, derived from the checkpoint's `class_map`.
  - The model rebuilt with `build_model` and `load_state_dict`.
  - A working directory equal to the repo root, because manifest paths are relative.
  - `recompute_metric` and `recompute_interval`, imported from `check_run`.
- The two CLIs take no `--root`. They always use the package's `ROOT`, which is why I copied `src/` into the temp repo.
- Setting `CUDA_VISIBLE_DEVICES=` (empty) did NOT hide the GPU on this machine (`torch.cuda.is_available()` stayed True). I used `-1` instead.
- Commands outside the temp directory: none. A first scratch script was refused. I then created one empty directory, `scratchpad/repo`, and left it in place. The real repo was only read, apart from running pytest and `claims check`.

## Defects

### D1. Resume replaces the declared config with the one actually trained, so the saved config can be false (medium)
- Where: `src/sprout/train/loop.py`, resume block (lines ~63-77). The checks cover only the manifest hash and the class map.
- Reproduce: train 2 epochs with `lr=0.05, batch_size=4, epochs=2`. Then call `train(replace(cfg, lr=0.001, batch_size=8, epochs=4), resume=latest)`.
- Result: the new `latest.pt` stores `config.lr=0.001, batch_size=8, epochs=4`. The optimizer state, restored from the old run, still has `lr=0.05` (`initial_lr=0.05`). The scheduler still has `T_max=2`. So after epoch 2 the cosine schedule goes back up, and the `epochs=4` in the config describes a schedule that never ran.
- This covers the verifier's second note (extended `epochs` keeps the old `T_max`). It is broader: `lr` is silently ignored, and nothing is compared (model, seed, batch size, transforms, `lr`, `epochs`).
- Impact: it cannot change a metric, because metrics are recomputed from predictions. It can put a false training config on a record. T3.1 (E1) first, if it resumes a run. T4.1 (E4) next, because its `attempts` must list each setting truthfully.
- Fix: on resume, either reject any config difference except `epochs`, or rebuild the scheduler for the new `epochs` and use the new `lr`.

### D2. No repo test guards optimizer or scheduler restore (low to medium, test gap)
- Where: `tests/test_train.py::test_resume_requires_explicit_path` (lines 82-91) checks only epochs and history.
- Evidence: the verifier replaced `optimizer.load_state_dict` and `scheduler.load_state_dict` in `loop.py` with `pass` and all tests stayed green. Reading the test confirms it. The round-trip test covers `save_checkpoint`/`load_checkpoint` only.
- My ruling: this is a real defect. The plan lists "checkpoints save optimizer, scheduler..." as the point of resume, and D1 shows the restore path is where the config gets misreported.
- Impact: a regression would not produce a false metric. It would produce an unfaithful "resumed" run that is recorded as a resume. T3.1 first, since that is the first long GPU run that might be resumed.
- Fix: after a resume, assert that the optimizer step count and the scheduler `last_epoch` continue, and that `lr` follows the restored schedule.

## Ruled not defects
- `build_loaders` reseeds on every `train()` call, so a resumed run repeats epoch 0's shuffle and augmentation draws (the verifier's first note). It means a resume is not bit-exact with an uninterrupted run. The seed is in the config, nothing is lost, and no metric depends on it. It becomes relevant only if a claim requires an exact reproduction. Low priority for T3.1.
- `preprocess` does no EXIF transpose. T2.1 does not list it, and the verifier correctly deferred it to the later prediction task.
- `accepted` flags against `gate_threshold` belong to T4.2. Git-history rewriting is out of scope for the checker. Both were decided before this review.

## Leakage and honesty
- The training loop selects `best` on `lab_dev` accuracy and never reads `lab_test`. Class maps are built from labels over the whole manifest, which is label-only and harmless.
- `results/claims.json` has no claims, and no docs cite numbers from this phase.
- The test-once rule is enforced by `check_run` and `claims`, and the phase's tests cover it.
