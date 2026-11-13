VERDICT: DEFECTS

Scope: T3.1 to T3.5 on `build`, `CUDA_VISIBLE_DEVICES=-1`, no GPU, nothing written to `results/runs/`. The five phase test files (test_topk_metrics, test_class_report, test_e1, test_e2, test_e3) give 44 passed, 1 deselected, with no skips or xfails. `check_run` printed "ok" for E1 lab_dev, E2 field_dev and E3 field_dev. The E3 lab_dev run had not finished when I stopped waiting (the E2 lab_dev check did not start), so those two checks were not run.

## What holds
- E1, E2 and E3 tell one story. All records share the same 38-class `config.class_names` and the same `checkpoint_sha256` 808b7f91d7e7. E2 and E3 name `E1-20261103T092124608963Z` as their parent.
- The two E1 records have byte-identical `e1` prediction files (same sha256), so the restore retrain reproduced the first run. The E1 `e1` rows against the E2 and E3 `e1` rows on the same paths have no label differences. The largest probability difference is about 4e-6, which is float32 noise.
- No code outside `data/field.py` and the class report reads `field_test` or `field_train`. The class report uses `field_test` row counts only, as T3.4 allows. No broad exception handlers and no skipped or weakened tests in the phase. The one edit to an older test, `tests/test_records.py`, only supplies the new required metrics.
- `results/error-analysis-e2.md` matches its CSVs (16 of 40 masks good; 5/12/15/9/9 categories). `results/claims.json` is empty, and no number appears in README or RESUME-TARGETS, so nothing overstates a run.

## Defects

1. `src/sprout/train/dataset.py:48-49` (`build_datasets`) hardcodes `lab_train` and `lab_dev`. This is the first concrete gap for T4.1.
   - `TrainConfig` (`src/sprout/train/config.py`) has no field for train or selection splits, a per-class cap, or a source filter. `load_manifest` (`splits.py:55`) returns only the five `COLUMNS`, so `source` is dropped.
   - There is no joint manifest, and `train()` loads one named CSV.
   - Reproduce: `build_datasets(load_manifest('field'), TrainConfig('field','x'), make_class_map(...))` gives 31 classes, `len(train)=0` and `len(val)=0`. The class map is built from whatever labels are present, so a field-only manifest silently gives 31 classes, not E1's 38.
   - Impact: E4 arms 1 to 6 and both held-out-source runs cannot be trained. Selection on `field_dev`, or on only the other source's part of it, is impossible as built.
   - False number on record: no. It fails loudly (an empty loader). The risk is that a quick patch selects on `lab_dev`, which would put a lab-selected model into a field_dev comparison.
   - First needed by T4.1. T4.0 is unaffected.

2. `src/sprout/train/loop.py:20-34` (`build_model`) knows only `tiny_cnn` and `convnext_tiny`.
   - Reproduce: `build_model('dinov2_vits14', 38)` raises `ValueError('unknown model')`.
   - `baselines/dino.py` loads DINOv2 through `torch.hub` as a frozen encoder only, with no classifier head, no partial unfreezing, and a network fetch.
   - `train/checkpoint.py` stores `TrainConfig`, not the hub repo and commit.
   - Impact: T4.1 arms 4 and 5 and the later ONNX export have nothing to load a trained DINOv2 from. Nothing in the repo shows that DINOv2 exports to ONNX with `preprocess`. It also needs 224-pixel input to be a multiple of 14, which it is.
   - False number on record: no.
   - First needed by T4.1 (arm 4) and T6 export.

3. `docs/class-report.md`, table "Crops and their healthy class". The `real` column counts only the crop's healthy-class photos, but the header does not say so. Squash and Orange show `real = 0` although Squash___Powdery_mildew has 136 real train photos and Orange___Haunglongbing has 47 in the same report. Corn shows 0 for the same reason.
   - Impact: a reader could conclude that the crops have no real photos. The text below the table is correct.
   - False number on record: no. It is a mislabeled column in a doc.
   - Task: T3.4 follow-up. Fix before the report guides the T4.1 cap.

## Checked and not defects
- `check_run` with arms of different architectures. There is one shared `config.class_names`, so arms must share the 38-class map. They do under a joint manifest.
- `standard_image_metrics` and the `source:<name>` subset. `subset_positions` rebuilds `source:<name>` from `field.csv` for `field_*` splits, and that order matches `split_info`. A held-out-source record can therefore be written and verified. It cannot also declare `shared_classes`, but `field_dev` has only the 31 shared classes, so this changes nothing.
- `docs/data-counts.md` cites the first E1 record (`...074728`), whose checkpoint was deleted. Its figures equal the restored record's (accuracy 0.99791, macro-F1 0.99661) and the predictions are identical, so this is a provenance note only.
- `check_run` exempts E1, E2 and E3 by name, as decided.
