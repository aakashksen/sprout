# SPROUT V2: implementation plan

> **For agentic workers:** run this with the `/run-plan` skill in this folder. It uses the project's own `implementer`, `verifier`, and `reviewer` agents and the `tools/taskctl.py` gate. Do not use another execution workflow.

**Goal:** Rebuild SPROUT so its model is measured honestly on real photos, improved with evidence, and served with calibrated confidence and cited advice.

**Architecture:** Training lives in a tested Python package with permanent data manifests and tracked experiments. The app loads an exported ONNX model and shares preprocessing with evaluation. The RAG pipeline is rebuilt around document metadata and citations, with its own evaluation.

**Tech stack:** Python 3.12, PyTorch, timm, scikit-learn, XGBoost, MLflow, DVC, ONNX Runtime, Grad-CAM, ChromaDB, sentence-transformers, Ollama, Flask, Docker.

**Spec:** `docs/DESIGN.md`. Testing rules: `docs/TESTING.md`. Original issue list: `docs/backlog.md`.

## Final stretch, the person's direction (2026-11-04)
This overrides older text in this plan where they differ. Background and evidence are in `docs/HANDOFF.md`.
- **Order.** T4.9 (final model), then T5.1, T7.4, T8.1, T8.2. T5.2 and the prep lanes run alongside T4.9.
- **Parallel lanes.** Anything that doesn't need the final checkpoint runs at the same time on the CPU: T5.2, T7.4's bug fixes and page wording, the Haiku judge, the wrong-disease check on `rag_dev`, T8.1's code, and T8.2's report template. Existing `prep/*` branches are material to take from, not to merge blindly.
- **Checks.** The independent verifier runs on T4.9, T7.4 and T8.1. T5.1, T5.2 and T8.2 use their acceptance commands (`"verifier": false` in `tasks.json`). No more phase reviews.
- **Judges.** Haiku through `claude -p`, in parallel, is the judge. Sonnet is a second named figure on final numbers. No local-model judging. Don't call either one "upper" or "conservative".
- **User's steps.** Label sign-off, deploy and the read-through come at the end and aren't blockers for agent work. Until H3 is done, the bullet says "containerized".

## Global constraints
- Python 3.12 in `.venv`; package `sprout` under `src/`, installed with `pip install -e ".[dev]"`.
- `.venv/Scripts/python -m pytest -q` needs no GPU, network, or spending. Real calls use `@pytest.mark.real`.
- Numbers come only from `results/runs/`.
- 8 GB VRAM shared with other projects, one GPU job at a time, mixed precision on.
- Product calls to Claude use Sonnet, capped at 5,000 as a runaway guard, counted in `results/claude_calls.json`. No API keys. DeepSeek is removed.
- Evaluation and the app share one preprocessing function. Class order comes from the saved class map, never a hardcoded list.
- `legacy/` is read-only.

## Source locations (read only; copy, never modify)
- PlantVillage: the official 38-class release, https://github.com/spMohanty/PlantVillage-Dataset, approved by the person under CC BY-SA 3.0 (2026-11-02). Use `raw/color` and `raw/segmented` at commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Original filenames carry a source ID, such as `00075aa8-...___FREC_Scab 3335.JPG`; the segmented copy of that file is `00075aa8-...___FREC_Scab 3335_final_masked.jpg`.
- The local 39-class copy (`...\9layerplantdatacnn\data\noaug\noaug data`, files renamed `image (N).JPG`) is not used: its 39th class, `Background_without_leaves`, can't be traced to a source.
- Augmented local version: the sibling `data\aug\aug data`. Not used.
- Old checkpoint: `...\pytorchConvNextMulticlassClassification\plant_disease_convnext_best.pth`.
- Knowledge documents: `<local path> Project Showcase\PlantProject\PlantRagChatbot\data`.
- `<local path>` is empty (checked in T0.2).
- Dataset terms set by the person (2026-11-02, recorded in `docs/datasets.md`): COCO and PlantDoc images are local only. Never commit or redistribute them or copies derived from them, including thumbnails and Grad-CAM overlays.

## Review focus
Likely real-world failures. Each has a test in the task that owns it.
1. A phone photo with EXIF rotation gets the same result as the upright version (T5.2).
2. A severely yellow or brown leaf isn't rejected by a green-pixel rule, and a green non-plant isn't accepted (T4.2).
3. A corrupt, huge, or non-image file renamed `.jpg` gives a clear error, not a crash or an unbounded allocation (T7.1).
4. A question about tomato can't be answered from an apple document, and a question no document covers says so (T6.2, T6.4).
5. If Ollama or the model file is unavailable, the app reports not ready and chat says it's unavailable, rather than making up an answer (T6.4, T7.1).

## How every task runs
The steps are the same throughout and aren't repeated below. For each listed test: write it, see it fail, write the smallest code that passes, see it pass, commit. A task's **Accept** commands are stored in `orchestration/tasks.json` and run by `taskctl verify`. For Experiment tasks, the implementer builds and tests the runner; the orchestrator then runs it, and commits the results before verifying.

Item numbers like (#3) refer to `docs/backlog.md`.

---

## Phase 0: Foundation

### T0.1: Project setup
**Files:** `pyproject.toml`, `src/sprout/__init__.py`, `.env.example`, `.github/workflows/ci.yml`, `tests/test_smoke.py`, `docs/backlog.md` already exists.
**Do:** Create `.venv` with `py -3.12 -m venv .venv`, and install with `.venv/Scripts/python -m pip install -e ".[dev]"` (pytest, ruff; add other packages to the extras as tasks need them). Set `pythonpath = ["src"]` for pytest, exclude `legacy` from ruff, and register the `real` marker, excluded by default (`addopts = "-m 'not real'"`). Install CUDA PyTorch from `https://download.pytorch.org/whl/` with the index matching this driver, since the default Windows wheel is CPU-only. Put the command in the README. CI runs ruff and pytest, with CPU torch.
**Tests:** `test_package_imports`; `test_real_marker_excluded_by_default`.
**Accept:** `{py} -m pytest tests/test_smoke.py tests/test_orchestration.py -q`; `{py} -m ruff check .`; `{py} -c "import torch; assert torch.cuda.is_available()"`.

### T0.2: Check the datasets
**Agent:** use the `researcher` for lookups. **Files:** `docs/datasets.md`, `tools/check_datasets_doc.py`.
**Do:** For PlantVillage (and which published version this local copy is), PlantDoc, one further real-photo source if one exists, and a public `not_plant` image set: record from primary sources the URL, license, size, classes, and access steps. Find segmented PlantVillage images (local or official repo). Apply the standing rule in `CLAUDE.md` and write an `Approved:` line for each.
**Accept:** `{py} tools/check_datasets_doc.py` (fails unless each dataset has URL, License, Classes, and Approved lines).

### H1: Check the dataset choices
Doesn't block the build; blocks T8.1. See `docs/HUMAN-GATES.md`.

---

## Phase 1: Data you can trust

### T1.1: Lab data, grouping, and manifests
**Files:** `src/sprout/data/inventory.py`, `src/sprout/data/grouping.py`, `src/sprout/data/splits.py`, `src/sprout/data/check_splits.py`, `datasets/manifests/lab.csv`, `.dvc` setup, `tests/test_lab_data.py`.
**Produces:** `inventory(root) -> DataFrame`; `group_near_duplicates(paths, max_distance=4) -> dict[path, group_id]` with perceptual hashes (imagehash) compared across flips and 90° rotations; `make_splits(df, seed=13, ratios=(.7,.15,.15)) -> DataFrame` stratified by class and split by group; `load_manifest(name) -> DataFrame` with `path, label, group, split, sha1`; `dataset_hash(manifest) -> str`.
**Do:** Get the official release at the pinned commit (see Source locations) inside this project at the git-ignored `datasets/raw/_pv_upstream`, never outside the project folder, and copy `raw/color` into `datasets/raw/lab/` and `raw/segmented` into `datasets/raw/lab_segmented/` with `run_in_background` or a long timeout. Track both with `.venv/Scripts/python -m dvc` (local remote). The `.dvc` pointer files are committed. Grouping uses the source IDs in the original filenames together with perceptual hashes: images share a group if either says so. Normalize the filename to its source ID, including any copy or flip suffixes the data actually contains, and record what you found. A source ID isn't proven to identify one physical leaf, so reports still say "near-duplicate-grouped". Resolves #2, #3, #4, #14. Write counts by class and split to `docs/data-counts.md`, noting that the 38 classes include healthy classes, so they aren't "38 diseases".
**Tests:** `test_duplicate_images_share_a_group` (an image with a resized, a flipped, and a rotated copy); `test_same_source_id_shares_a_group`; `test_no_group_in_two_splits`; `test_split_is_deterministic`; `test_every_class_in_every_split`; `test_hash_changes_when_a_label_changes`.
**Accept:** `{py} -m pytest tests/test_lab_data.py -q`; `{py} -m sprout.data.check_splits lab`.

### T1.2: Real-photo and not-plant sets
**Files:** `src/sprout/data/field.py`, `datasets/manifests/field.csv`, `datasets/manifests/not_plant.csv`, `datasets/class_map.csv`, `datasets/review/field-sample.md`, `tests/test_field_data.py`.
**Do:** Download only sets marked `Approved: yes`, and keep the images under the git-ignored `datasets/raw/`. If PlantDoc isn't approved, block this task with the reason. `not_plant` is COCO 2017 validation, with every image annotated `potted plant` excluded using the official instance annotations. Add a `source` column to `field.csv`, so T4.1 can tell PlantDoc images from other real-photo sets. Write `class_map.csv`, mapping real-photo classes to lab classes, and marking unmappable ones excluded with a reason. Deduplicate against each other and against lab. Split by group, stratified by class, into `field_train`, `field_dev`, `field_test` (.5,.25,.25), which must not be re-split later. Every shared class must appear in `field_test`, and those that can't are dropped from the shared list with a recorded reason. Write a random sample of 50 with labels to the review file. Thumbnails go in the git-ignored `datasets/review/thumbs/`, and the committed review file only links to them.
**Tests:** `test_not_plant_excludes_potted_plant_images`; `test_no_coco_or_plantdoc_image_is_tracked_by_git`; `test_unmapped_classes_excluded`; `test_no_field_image_duplicates_a_lab_image`; `test_no_group_in_two_field_splits`; `test_field_test_has_at_least_200_images`; `test_every_shared_class_is_in_field_test`; `test_shared_class_list_is_recorded`.
**Accept:** `{py} -m pytest tests/test_field_data.py -q`; `{py} -m sprout.data.check_splits field`.

### H2: Review labels
Doesn't block the build; blocks T8.1. See `docs/HUMAN-GATES.md`.

---

## Phase 2: Training and evaluation pipeline

### T2.1: Training pipeline
**Files:** `src/sprout/train/config.py`, `src/sprout/train/dataset.py`, `src/sprout/train/loop.py`, `src/sprout/train/checkpoint.py`, `src/sprout/preprocess.py`, `tests/test_train.py`.
**Produces:** `TrainConfig` (dataclass saved in every checkpoint: model, seed, lr, batch size, epochs, transforms, manifest hash); `preprocess(image: PIL.Image) -> Tensor`, the one shared inference preprocessing; `train(config, *, resume: Path | None = None) -> RunOutput`. Resolves #1, #5–#13.
**Do:** Separate train and validation datasets, so augmentation really applies to training only. Fresh vs resume is explicit, never inferred from a file existing. Checkpoints save model, optimizer, scheduler, scaler, epoch, best metric, class map, and config; `best` and `latest` are separate files. Uses CPU when CUDA isn't available.
**Tests:**
- `test_train_transform_differs_from_val_transform` (the legacy bug: an image through each gives different results for train, the same for val)
- `test_val_dataset_has_no_random_augmentation`
- `test_resume_requires_explicit_path`
- `test_checkpoint_round_trip_includes_class_map_and_optimizer`
- `test_best_and_latest_are_separate`
- `test_two_step_run_on_tiny_model_reduces_loss_on_cpu`
- `test_same_seed_same_split_and_first_batch`
**Accept:** `{py} -m pytest tests/test_train.py -q`.

### T2.2: Metrics and statistics
**Files:** `src/sprout/eval/metrics.py`, `src/sprout/eval/stats.py`, `src/sprout/eval/predict.py`, `tests/test_metrics.py`.
**Produces:** `macro_f1`, `per_class_report`, `confusion`, `ece(probs, labels, bins=15)`, `selective_metrics(probs, labels, threshold)`, `recall_at_k`, `mrr`, `percentile`, `bootstrap_ci(values, groups, n=10000, seed=13)`, `paired_bootstrap_diff`, `mcnemar_exact(a_correct, b_correct)`; `predict(model, manifest_split) -> Predictions` with `path, label, probs`.
**Tests:** `mcnemar_exact` with b=1, c=9 gives p=0.0215 (two-sided, ±0.0005); `macro_f1` matches scikit-learn on a fixed case; `ece` is 0 for perfectly calibrated bins and 0.5 for always 100% confident at 50% right; `bootstrap_ci` resamples whole groups; `selective_metrics` at threshold 1.01 gives coverage 0; `recall_at_k` with gold at rank 3 is 0 at k=2 and 1 at k=3; `mrr` of ranks 1 and 2 is 0.75.
**Accept:** `{py} -m pytest tests/test_metrics.py -q`.

### T2.3: Run records and claims
**Before starting:** read how ContextLensV2 solved this, read only, and never change that folder: `<local path>` (`records.py`, `check_run.py`, `claims.py`, `stats.py`, `series.py`) and `<local path>`. Use the same design, so its defects aren't repeated: one record maps each system to its predictions, and `check_run` recomputes the intervals and comparisons.
**Files:** `src/sprout/eval/records.py`, `src/sprout/eval/check_run.py`, `src/sprout/eval/claims.py`, `results/claims.json`, `tests/test_records.py`.
**Produces:** `write_run(experiment, split, config, models, metrics, intervals, attempts, predictions, dataset_hash, *, root=ROOT, allow_dirty=False, notes="") -> Path`, with the fields in `docs/TESTING.md`; `allow_dirty` exists for tests only. `python -m sprout.eval.check_run <E#> [--min-n N] [--split S]`, enforcing all rules in `docs/TESTING.md`. The dirty check ignores `orchestration/`, as `taskctl` does. Image splits read their manifests; `rag_dev` and `rag_test` read `datasets/rag/questions.jsonl`; `perf` hashes its fixed image list and uses that list's size. `python -m sprout.eval.claims check`.
**Tests:** one failing case for each `check_run` rule, plus `test_real_runs_cannot_use_the_fake` (`check_run` rejects a record whose models name `FakeIdentifier` from T5.0); `test_write_run_refuses_dirty_tree`; `test_claim_without_matching_run_fails`; `test_claim_value_must_equal_metric_in_run`.
**Accept:** `{py} -m pytest tests/test_records.py -q`; `{py} -m sprout.eval.claims check`.

---

## Phase 3: The honest baseline

### T3.1: Experiment E1, clean lab baseline
**Files:** `src/sprout/experiments/e1_lab.py`, `configs/e1.json`, `tests/test_e1.py`.
**Do:** Train ConvNeXt-Tiny from ImageNet weights with the fixed pipeline, logging to MLflow. Evaluate on `lab_dev`. Also evaluate the old checkpoint on `lab_dev`, labeled `legacy, may have seen these images; 39-class checkpoint mapped onto 38 shared classes`, because its split history is unknown (#5) and it has an extra output. Its class order comes from the hardcoded list in `legacy/PlantProject/application.py`; confirm the order gives a sensible score, or record that it couldn't be established. Map its 39 classes onto the 38 official classes by name, and record the mapping and how the `Background_without_leaves` output is handled. Save a per-class report and confusion matrix. Reconcile the old 99.97%, 98.5%, and 97.5% figures in `docs/data-counts.md` (#15).
**Tests:** runner on a tiny fixture and tiny model writes a valid record; `test_legacy_result_is_labeled`; `test_legacy_classes_map_onto_38_shared_classes`.
**Accept:** `{py} -m pytest tests/test_e1.py -q`; `{py} -m sprout.eval.check_run E1 --min-n 1000 --split lab_dev`.

### T3.2: Experiment E2, domain shift
**Files:** `src/sprout/data/masks.py`, `src/sprout/explain/gradcam.py`, `src/sprout/experiments/e2_field.py`, `results/e2-gradcam/`, `results/error-analysis-e2.md`, `tests/test_e2.py`.
**Do:** Run the E1 model on `field_dev`, over shared classes. Save Grad-CAM images for correct and wrong predictions on lab and real photos, with real-photo overlays in the git-ignored `results/e2-gradcam/field/`, and measure what share of Grad-CAM energy falls outside the leaf, using masks from `sprout.data.masks`. Check mask quality on a sample, and if masks are unreliable for real photos, report the metric only for lab images and record why. Hand-categorize 50 mistakes: blur, background, similar symptoms, wrong label, unsupported condition (#20, #21).
**Tests:** `test_gradcam_output_matches_image_size`; `test_gradcam_highlights_the_only_signal` (a tiny model on a synthetic image); `test_only_shared_classes_are_scored`.
**Accept:** `{py} -m pytest tests/test_e2.py -q`; `{py} -m sprout.eval.check_run E2 --min-n 200 --split field_dev`.

### T3.3: Experiment E3, baselines
**Files:** `src/sprout/baselines/classic.py`, `src/sprout/baselines/frozen.py`, `src/sprout/baselines/dino.py`, `src/sprout/experiments/e3_baselines.py`, `tests/test_e3.py`.
**Do:** Write one record per split. On `lab_dev` and `field_dev`, compare: HOG and color histograms with XGBoost; frozen ConvNeXt with a trained head; DINOv2 (small) with a trained head; and the E1 fine-tuned model. Use paired statistics. Record training time and model size (#18).
**Tests:** `test_features_are_stable`; `test_xgboost_learns_a_separable_fixture`; `test_frozen_backbone_has_no_grad`; `test_record_includes_all_four_systems`.
**Accept:** `{py} -m pytest tests/test_e3.py -q`; `{py} -m sprout.eval.check_run E3 --min-n 200 --split field_dev`.

### T3.4: Class report and document coverage
**Files:** `src/sprout/data/class_report.py`, `docs/class-report.md`, `docs/document-coverage.md`, `tests/test_class_report.py`.
**Do:** Written before any joint training, for the person. Photos per class, by source and split, from the manifests only, with no `field_test` image or score read. Cross-source duplicates, from `field_conflicts.csv`, `field_reviewed_links.csv`, and groups spanning sources. Classes under the minimum of about 80 train, 20 dev, and 20 test real photos. No class is dropped from the frozen `field_test`: thin classes are reported as thin. Show which crops have a healthy class with real photos. Add `docs/document-coverage.md`: per kept class, whether a document in `knowledge/manifest.csv` describes its symptoms, checked by reading the passages rather than the manifest's disease list alone. Raspberry, soybean, and squash have no documents today. The counts come from code, so they can be regenerated.
**Tests:** `test_counts_match_the_manifests`; `test_thin_classes_are_reported_not_dropped`; `test_report_reads_no_field_test_image`; `test_every_kept_class_has_a_coverage_row`.
**Accept:** `{py} -m pytest tests/test_class_report.py -q`.

### T3.5: Top-k and plant-given metrics
**Files:** `src/sprout/eval/metrics.py`, `src/sprout/eval/check_run.py`, `src/sprout/eval/records.py`, `docs/TESTING.md`, `tests/test_topk_metrics.py`.
**Produces:** `top_k_accuracy(probs, labels, k)`; `plant_given_top_k(probs, labels, class_names, k)`, which restricts outputs to the true plant's classes; `options_per_plant(class_names)`; a declared subset `source:<name>` read from `field.csv`'s `source` column.
**Do:** Score top-1, top-3, and top-5 alongside macro-F1 and calibration in every image evaluation from here on, plus plant-given top-1 and top-3. Plant comes from the part of the class name before `___`. `check_run` recomputes all of them from saved predictions, and a record with plant-given numbers must also state the options per plant. Plants average 2.7 diseases, so plant-given top-3 is easy: every report that shows it says so and states the options per plant. A plant with three or fewer classes scores 1.0 by construction, and the report must make that plain. Recompute these for the E1, E2, and E3 records from their saved predictions, and save them in a new file rather than editing a record.
**Tests:** `test_top_k_on_a_fixed_case`; `test_plant_given_restricts_to_the_true_plant`; `test_plant_with_three_classes_scores_one_at_top_3`; `test_saved_top_k_is_recomputed_and_tampering_fails`; `test_source_subset_rebuilds_from_the_manifest`.
**Accept:** `{py} -m pytest tests/test_topk_metrics.py -q`.

---

## Phase 4: Making it better on real photos

### T4.0: Experiment E10, disease-area segmentation
**Files:** `src/sprout/segment/data.py`, `src/sprout/segment/model.py`, `src/sprout/segment/crop.py`, `src/sprout/experiments/e10_segment.py`, `configs/e10.json`, `datasets/manifests/seg.csv`, `tests/test_e10.py`.
**Produces:** `predict_mask(model, image) -> mask`; `crop_to_disease(image, mask, margin) -> image`, which returns the full image when the predicted area is tiny.
**Do:** Use `num_workers=8` at most in real training; tests use what they need. Train a small segmentation model on PlantSeg's masks, from the `plantseg.zip` already downloaded, with no new download. Use only images not in `field_dev` or `field_test`. Exclude by group as well, and check PlantSeg images outside `field.csv` for near-duplicates of `field_dev` and `field_test` with the rule in `docs/field-dedup.md`, so no held-out photo reaches training. Write `seg.csv` with splits `seg_train` and `seg_dev`, and teach `check_run` the `seg_dev` split, recomputing per-image IoU from saved rows. Keep only best and latest. Report IoU honestly, and that PlantSeg's masks mark disease on PlantSeg-style photos and may not carry over to PlantDoc.
**Tests:** `test_no_field_dev_or_field_test_photo_in_seg_manifest`; `test_near_duplicate_of_held_out_photo_is_excluded`; `test_crop_includes_margin`; `test_tiny_mask_falls_back_to_full_image`; `test_saved_iou_is_recomputed`.
**Accept:** `{py} -m pytest tests/test_e10.py -q`; `{py} -m sprout.eval.check_run E10 --min-n 50 --split seg_dev`.

### T4.1: Experiment E4, improvements
**Consumes:** `docs/class-report.md` from T3.4, `standard_image_metrics` from T3.5 (`check_run` requires them on image records from E4 on), `crop_to_disease` from T4.0. Use `num_workers=4` at most in real training, and `num_workers=0` in tests.
**Files:** `src/sprout/train/joint.py`, `src/sprout/train/augment.py`, `src/sprout/experiments/e4_improve.py`, `configs/e4.json`, `tests/test_e4.py`.
**Do:** Train on real and lab photos together from the start: `field_train` (PlantDoc and PlantSeg) plus `lab_train` capped per class, so lab can't dominate. Fix the cap in `configs/e4.json` from the class report, before any E4 result is seen. PlantDoc images are cleared for this for local use: don't republish them, and record whether the resulting weights were trained on them. One change at a time, all listed in `attempts`, selected on `field_dev`. Write the arm list in `configs/e4.json` and commit it before running. At most 6 training runs: (1) ConvNeXt-Tiny joint baseline; (2) ConvNeXt with stronger color and blur augmentation; (3) ConvNeXt with background randomization of lab photos, with leaves cut out using the official segmented images, matched to `lab_train` by filename so nothing leaks, and pasted on varied backgrounds, falling back to masks from `sprout.data.masks` (HSV and GrabCut on `lab_train` images only), and recording this arm as not run if good masks can't be made; (4) DINOv2 small with a head trained on the joint set, since E3 had DINOv2 ahead of ConvNeXt on real photos; (5) DINOv2 with its last blocks fine-tuned, only if arm 4 shows it can help, otherwise recorded as not run; (6) the best of those, with and without the T4.0 crop, as a separate arm. The combined augmentation-and-background ConvNeXt arm was dropped before running to make room for DINOv2; record that in `attempts`. Also score a held-out-source run as a secondary number: train with one real source left out of both training and selection, so selection uses only the other source's part of `field_dev`, and test on the held-out source's part of `field_dev`, once for each source, so 2 more runs, each in `attempts`. It stays on `field_dev` only and is labeled a development number in reports. Report before and after against E1 and the E3 baselines with paired statistics, top-1, top-3, top-5, macro-F1, calibration, plant-given top-1 and top-3, and Grad-CAM energy outside the leaf on lab images. Long runs resume explicitly from `latest.pt` if interrupted, with both attempts listed, and a partial run never counts as a result. If nothing beats the baseline, record that, and keep the baseline. If the crop doesn't help, say so and leave it out of serving (#19).
**From phase review:** `build_datasets` in `src/sprout/train/dataset.py` hardcodes `lab_train` and `lab_dev`, `TrainConfig` has no train or selection splits, per-class cap, or source filter, and `load_manifest` drops the `source` column, so joint training and held-out-source runs can't run as built. Selection must use `field_dev`, never `lab_dev`. `build_model` in `src/sprout/train/loop.py` only knows `tiny_cnn` and `convnext_tiny`, and DINOv2 exists only as a frozen encoder with no head, partial unfreezing, or checkpointing; arms 4 and 5 need those, in a form T5.1 can export.
**Tests:** `test_background_paste_keeps_leaf_pixels_and_changes_background`; `test_field_test_is_never_read` (the loader raises if asked); `test_lab_cap_limits_every_class`; `test_field_dev_is_never_trained_on`; `test_held_out_source_is_absent_from_training_and_selection`; `test_crop_arm_uses_full_image_on_tiny_mask`; `test_attempts_lists_each_run`.
**Accept:** `{py} -m pytest tests/test_e4.py -q`; `{py} -m sprout.eval.check_run E4 --min-n 200 --split field_dev`.

### T4.2: Experiment E5, calibration, abstention, and input checks
**Files:** `src/sprout/serve/calibrate.py`, `src/sprout/serve/gate.py`, `src/sprout/experiments/e5_confidence.py`, `tests/test_e5.py`.
**Produces:** `fit_temperature(logits, labels) -> float`; `choose_threshold(probs, labels, min_coverage=0.5) -> float`, on `field_dev`; `decide(probs, threshold) -> Decision` with `status: answered|unsure`, top-k alternatives (k≤3, each above 10%); `is_plant_image(image, probs) -> bool`.
**Do:** The official release has no background class, so compare these input checks: the old 5% green-pixel rule, model confidence alone, and model confidence plus an out-of-distribution score. Measure both rejection rates on `field_dev` and `not_plant` (#22–#24, #26). `check_run` trusts each row's `accepted` flag, so save the confidences and threshold and add a check that the flags follow from them.
**Tests:** `test_temperature_reduces_ece_on_overconfident_fixture`; `test_threshold_is_chosen_without_test_data`; `test_unsure_below_threshold`; `test_top_k_is_limited`; `test_yellow_brown_leaf_fixture_is_not_rejected_by_chosen_gate`; `test_green_non_plant_fixture_is_measured_not_silently_passed`.
**Accept:** `{py} -m pytest tests/test_e5.py -q`; `{py} -m sprout.eval.check_run E5 --min-n 200 --split field_dev`.

### T4.7: Experiment E13, clean the real-photo labels
**Agent:** the implementer builds the tools; the orchestrator runs the vision review with `sonnet` agents, up to 3 at once. **Files:** `src/sprout/data/field.py`, `src/sprout/data/label_review.py`, `src/sprout/experiments/e13_clean_labels.py`, `datasets/manifests/field_clean.csv`, `datasets/label_exclusions.csv`, `datasets/label_uncertain.csv`, `docs/label-review.md`, `configs/e13.json`, `tests/test_label_review.py`, `tests/test_e13.py`.
**Why:** an agent review (`orchestration/evidence/H2-agent-review-photos.md`) found `datasets/review/thumbs` doesn't match `datasets/review/field-sample.md`, and with the right images 7 of 50 were wrong and 9 unclear, over the limit of 5. The person approved this fix.
**Do:** (1) Find why thumbs and table diverged, fix it with a test, and regenerate them. (2) Check all `field_dev` and `field_test` photos by eye, in contact sheets where that's faster, with the two splits shuffled together and not named on the sheets, so both get the same standard. Reviewers see only the image, its label, and its filename, and never test scores or model predictions. Remove only clear errors: not a plant photo (slides, charts, products, text), wrong plant, or a label clearly contradicted by the image, such as healthy on an obviously diseased leaf. No relabeling, no fine disease judgments from small images, and the filename is evidence only. Look at each proposed removal again at larger size before it counts. (3) Log each removal with image ID, source, split, and reason in `datasets/label_exclusions.csv`, versioned, and each uncertain one in `datasets/label_uncertain.csv` without removing it. Measure the removal rate by split, source, and class. Estimate `field_train`'s rate from a seeded sample, written separately. (4) Apply the list when the manifest is built, writing `field_clean.csv` with its own hash. Raw data isn't deleted, and `field.csv` stays byte-identical so old records stay valid. (5) Re-score the existing joint ConvNeXt, DINOv2, and baseline models on cleaned `field_dev` in new E13 records, beside the old numbers. No new training, and no `field_test` score, which waits for T8.1. Old records aren't touched. (6) If the `field_train` estimate is over about 5% bad, say so, review all of `field_train` the same way, and retrain joint ConvNeXt on the cleaned train set as a new run with otherwise unchanged settings. (7) `docs/DECISIONS.md`: why, how many, and what changed in dev scores.
**Tests:** `test_thumbs_and_table_come_from_the_same_rows`; `test_field_csv_is_unchanged`; `test_exclusions_are_applied_when_the_manifest_is_built`; `test_every_exclusion_has_id_source_split_reason`; `test_every_exclusion_exists_in_field_csv`; `test_review_inputs_contain_no_scores_or_predictions`; `test_sheets_do_not_name_the_split`; `test_field_test_is_never_scored`; `test_old_records_still_pass_check_run`.
**Accept:** `{py} -m pytest tests/test_label_review.py tests/test_e13.py -q`; `{py} -m sprout.eval.check_run E13 --min-n 200 --split field_dev`.

### T4.3: Experiment E11, follow-up questions on top-3
**Files:** `src/sprout/serve/followup.py`, `src/sprout/experiments/e11_followup.py`, `datasets/followup/questions.jsonl`, `datasets/followup/answers.jsonl`, `configs/e11.json`, `tests/test_e11.py`.
**Produces:** `next_question(top3, *, asked) -> Question | None`; `update(top3, question, answer) -> top3`, which only uses symptoms described in a document.
**Do:** When the model is unsure between its top-3, ask the person what they can see, and use the answer to re-rank. Questions come only from symptom text in the existing documents, each citing its document and page, and only for classes `docs/document-coverage.md` marks as covered. If a class among the top-3 has no document, don't ask about it, and say so. Labeled test set, on `field_dev` photos only: answers must come from looking at the photo, not from knowing its class, so no simulated user who already knows the answer. A reviewer agent views each photo and answers the questions without being shown the label, the prediction, or the file name, and those answers are saved before any scoring. Record who answered, that it's an agent's reading and not a grower's, and how many it couldn't answer. Unknown is a valid answer. Report top-1, macro-F1, and calibration before and after, with paired statistics; how many questions it took; and how often the flow made the answer worse. Also report it on the held-out-source models, side by side with the seen-source number. `field_test` isn't read. If it doesn't help, record that, and leave it out of the product.
**Tests:** `test_questions_cite_a_document_and_page`; `test_uncovered_class_gets_no_question`; `test_answers_were_saved_blind_to_the_label`; `test_unknown_answer_leaves_ranking_unchanged`; `test_field_test_is_never_read`.
**Accept:** `{py} -m pytest tests/test_e11.py -q`; `{py} -m sprout.eval.check_run E11 --min-n 50 --split field_dev`.

### T4.4: Candidate documents for uncovered classes
**Agent:** use the `researcher` for lookups. **Files:** `docs/document-candidates.md`, `tools/check_candidates_doc.py`.
**Do:** For each of the 13 classes `docs/document-coverage.md` marks as uncovered, list candidate documents from primary publishers, such as university extension services and government agencies: URL, publisher, title, date, what the page says about symptoms, and its license or terms as stated by the publisher. Say "couldn't confirm" where that's so. Download nothing. Nothing is added to `knowledge/`: the person approves licenses first.
**Accept:** `{py} tools/check_candidates_doc.py` (fails unless each of the 13 classes has a section, and each candidate has URL, Publisher, License, and Approved lines, with Approved saying `no, needs the person`).

### T4.5: More real-photo sources for training
**Agent:** use the `researcher` for lookups. **Files:** `docs/datasets.md`, `src/sprout/data/extra_sources.py`, `datasets/manifests/extra_train.csv`, `datasets/class_map_extra.csv`, `docs/extra-sources.md`, `tests/test_extra_sources.py`.
**Do:** Gather more real photos, starting with PlantWild, plus Plant Pathology 2021 if it's publicly downloadable without login. Use only data publicly accessible without login, paywall, or getting around a block. Record each source in `docs/datasets.md`. Raw content stays in the git-ignored `datasets/raw/`, and is never committed or published. Stop and tell the coordinator before any download over 40 GB, or under 40 GB free. Map classes to lab classes, with unmappable ones excluded and a reason recorded. Deduplicate against each other, against `lab`, and against every `field.csv` split with the rule in `docs/field-dedup.md`, so no `field_dev` or `field_test` photo or copy reaches training. Cap per class. All of it goes into training only: `field_dev` and `field_test` are frozen and don't change. Report what each source added per class, and which thin classes it helped.
**Tests:** `test_no_extra_image_duplicates_a_field_dev_or_field_test_photo`; `test_extra_sources_only_add_training_rows`; `test_field_manifest_is_unchanged`; `test_cap_limits_every_class`; `test_unmapped_classes_excluded`.
**Accept:** `{py} -m pytest tests/test_extra_sources.py -q`; `{py} -m pytest tests/test_extra_sources.py -q -m real`.

### T4.8: Plant Pathology 2021 as a further training source
**Agent:** the T4.5 implementer's pipeline. **Files:** `src/sprout/data/extra_sources.py`, `tools/build_extra_sources.py`, `datasets/manifests/extra_train.csv`, `datasets/class_map_extra.csv`, `docs/extra-sources.md`, `docs/datasets.md`, `tests/test_extra_sources.py`.
**Do:** The person approved fetching the public mirror `timm/plant-pathology-2021` on Hugging Face (about 16 GB, no login), recorded as third-party in `docs/datasets.md`, with what is known of its terms: the mirror's Apache-2.0 tag is the uploader's, and the Kaggle terms couldn't be confirmed. Raw content stays git-ignored and is never committed or published. Labels are multi-label: use only rows whose single label maps to a lab class (scab, healthy, rust; rust's mapping to `Apple___Cedar_apple_rust` is an inference, so check it by viewing samples), and exclude the rest with a recorded reason. Run the same copy checks as T4.5 against `field_dev` and `field_test`, and the same cap per class. Report what it adds per class after the cap, including if that's little. `field.csv` doesn't change. E12 waits for this because its data must stop changing first.
**Tests:** `test_no_extra_is_a_near_copy_of_a_field_dev_or_field_test_photo` still holds with the new rows; `test_only_single_label_rows_that_map_are_used`; `test_extra_sources_only_add_training_rows`; `test_cap_limits_every_class`.
**Accept:** `{py} -m pytest tests/test_extra_sources.py -q`; `{py} -m pytest tests/test_extra_sources.py -q -m real`.

### T4.6: Experiment E12, best configuration with more sources
**Files:** `src/sprout/experiments/e12_more_sources.py`, `configs/e12.json`, `tests/test_e12.py`.
**Do:** Rerun E4's best configuration, `convnext_joint`, with the T4.5 photos added to training, one run, with settings otherwise unchanged and fixed in `configs/e12.json` before running. Score on `field_dev`, including held-out-source runs in the same way as E4, side by side with E4's seen-source and held-out numbers, with paired statistics. Selection stays on `field_dev` and these are development numbers. If it clearly helps, it becomes the chosen model, and E5's calibration and thresholds are refit on it; if not, record that and keep E4's model.
**Tests:** `test_field_test_is_never_read`; `test_settings_match_e4_best_apart_from_the_extra_rows`; `test_held_out_source_is_absent_from_training_and_selection`.
**Accept:** `{py} -m pytest tests/test_e12.py -q`; `{py} -m sprout.eval.check_run E12 --min-n 200 --split field_dev`.

---

### T4.9: Experiment E15, final model on 29 classes with all photos
**Files:** `src/sprout/experiments/e15_final_model.py`, `configs/e15.json`, `tests/test_e15.py`, plus the training loader and class list it needs.
**Classes:** keep a class only with at least 80 real training photos and at least 10 real photos each in `field_dev` and `field_test`, which leaves 29. Confirm the list against the manifests in the record. Dropped: Grape Esca, Grape leaf blight, Peach bacterial spot, Tomato spider mites, Tomato target spot, Corn healthy, Potato healthy, Strawberry leaf scorch, Cherry powdery mildew. The lab-only baseline stays for the lab-to-real comparison.
**Photos:** no lab cap and no extras cap. The pool holds all usable lab and real photos on those classes, including the Plant Pathology rows the old cap of 200 removed; rebuild `extra_train` without the cap by the same copy checks. Each pass draws 170 real photos per class, evenly across sources within the class, and 70 lab photos per class, fresh each pass. Small classes repeat photos; nothing is deleted. The record states how many different photos training actually saw, and that is the number any bullet uses.
**Model:** ConvNeXt V2 Tiny from `timm`, `convnextv2_tiny.fcmae_ft_in22k_in1k`, 224 px, E4's settings otherwise, 10 passes. One run, no tuning search; one learning-rate change only if it clearly trains badly, recorded with the reason. Confirm the weights' license in the record. Before training, confirm an untrained V2 exports to ONNX and quantizes, because T5.1 depends on it; if it can't, stop and tell the coordinator.
**Choosing:** rescore E4 and E12 from their saved predictions on the same 29 classes: keep rows whose true class is one of the 29, mask the dropped classes' outputs, renormalize, then score. This rule is fixed before any score is seen. The new model replaces the current one only if its macro-F1 on cleaned `field_dev` rows is at least as high. Otherwise keep the best of the three, which still serves 29 classes. The record states this, and E12's G3 failure stays recorded as history. Add G1 and G2 to "Try later" in `docs/backlog.md` if they aren't there.
**Thresholds:** refit E5's calibration and thresholds on the chosen model, on the CPU, with a stricter confidence threshold for any healthy label. Report what that costs in unsure answers, and confident sick-called-healthy counts beside E4's and E12's. Healthy photos aren't deleted or reweighted.
**Tests:** `test_class_rule_gives_the_recorded_list`; `test_each_pass_draws_170_real_and_70_lab_per_class`; `test_real_draws_are_even_across_sources`; `test_lab_draws_change_between_passes`; `test_field_test_is_never_read`; `test_old_models_are_masked_before_scoring`; `test_healthy_threshold_is_stricter`.
**Accept:** `{py} -m pytest tests/test_e15.py -q`; `{py} -m sprout.eval.check_run E15 --min-n 200 --split field_dev`.

## Phase 5: Efficient serving

### T5.0: Identification contract and fake identifier
**Files:** `src/sprout/serve/types.py`, `tests/test_identify_types.py`.
**Produces:** `Identification` with `status: answered|unsure|not_plant|invalid`, `top`, `alternatives`, `guidance`; `class IdentifierProtocol`; `FakeIdentifier(results)`. This lets the chatbot and the app be built while training runs. T5.2 implements the real one, and T7.4 and T8.1 use only the real one.
**Tests:** `test_fake_returns_each_status`; `test_identification_round_trips_as_json`. T2.3 owns `test_real_runs_cannot_use_the_fake`, since it builds `check_run`.
**Accept:** `{py} -m pytest tests/test_identify_types.py -q`.

### T5.1: Experiment E6, ONNX and quantization
**The person's direction (2026-11-04), overrides the Do below:** run the model T4.9 chose and its quantized version through the normal stats and record system, and compare their stats directly: size, CPU p50 and p95, `field_dev` scores, predictions changed, and whether T4.9's thresholds hold. Skip the separate middle ONNX comparison. This is bullet 3's speed number.
**Files:** `src/sprout/serve/export.py`, `src/sprout/experiments/e6_onnx.py`, `tests/test_e6.py`.
**Produces:** `export_onnx(checkpoint, path)`; `quantize(path, mode, calibration_images=None) -> Path`, with calibration images from `lab_dev` for static mode; `OnnxModel(path).predict(image) -> probs`.
**Do:** Export whichever model E4 chose, ConvNeXt or DINOv2; if it's DINOv2, confirm it exports to ONNX with the same preprocessing, and report plainly if it can't. If E4 kept the T4.0 crop, export the segmentation model too and include it in every latency number. Compare PyTorch CPU, ONNX, and quantized ONNX: size, p50 and p95 latency on CPU, macro-F1 on `field_dev`, and share of predictions that changed. State the CPU in the record. If quantization costs a measurable loss, report it, and choose by the numbers.
**Tests:** `test_onnx_matches_torch_within_tolerance` (tiny model); `test_quantized_runs_and_is_smaller`; `test_class_map_travels_with_the_model`.
**Accept:** `{py} -m pytest tests/test_e6.py -q`; `{py} -m sprout.eval.check_run E6 --min-n 200 --split field_dev`.

### T5.2: Inference service
**The person's direction (2026-11-04):** runs alongside T4.9, tested on fixtures. The chosen model is bound once T4.9 and T5.1 land; `Identifier` refuses a calibration fitted to another model. A class outside the 29 can't be named.
**Consumes:** `Identification` from T5.0.
**Files:** `src/sprout/serve/identify.py`, `tests/test_identify.py`.
**Produces:** `Identifier(model_path, calibration_path).identify(image_bytes) -> Identification` with `status: answered|unsure|not_plant|invalid`, `top`, `alternatives`, `guidance`. Uses the shared `preprocess`, fixes EXIF orientation, and bounds dimensions (#25, #30, #31, #63).
**From phase review:** see `orchestration/verdicts/phase-4.md`. (D1) `results/e5/chosen.json` names no model, and `src/sprout/serve/gate.py` reads only `gate.settings`, so a different model would silently load thresholds fitted to another. Record the model's sha256 in the calibration file, and make `Identifier` refuse a mismatch. E6 must also state whether thresholds fitted on the PyTorch model hold on the quantized one: coverage was already 0.497 against a 0.5 target. (D3) Training and evaluation never apply EXIF orientation, while serving does. About 0.6% of field_dev photos carry a rotation tag (6 of 1,040). Serving keeps applying it, because a sideways phone photo is what the app must handle; record this and report the small difference between evaluation and serving.
**Tests:** `test_exif_rotation_gives_same_result_as_upright`; `test_preprocess_matches_evaluation_exactly`; `test_invalid_bytes_returns_invalid`; `test_oversize_image_is_rejected_before_decode_fully`; `test_unsure_includes_specific_photo_guidance`.
**Accept:** `{py} -m pytest tests/test_identify.py -q`.

---

## Phase 6: A chatbot you can trust

### T6.1: Document ingestion
**Files:** `knowledge/manifest.csv`, `src/sprout/rag/ingest.py`, `tests/test_ingest.py`.
**Do:** Copy documents to `knowledge/raw/` (git-ignored). Write a manifest: file, crop, diseases, title, publisher, source URL, date where available. It replaces filename guessing. Ingest PDF and Markdown. Check extraction, OCR where necessary, keep page numbers, and exclude failures from the index and list them. Build the new index beside the old one, validate it, and then switch. Resolves #32–#36, #44, #45.
**Tests:** `test_crop_comes_from_manifest_not_filename` (the "guide" bug); `test_markdown_is_ingested`; `test_failed_extraction_creates_no_chunk`; `test_chunks_keep_source_and_page`; `test_failed_build_keeps_the_old_index`; `test_every_file_is_in_the_manifest`.
**Accept:** `{py} -m pytest tests/test_ingest.py -q`.

### T6.2: Retrieval
**Files:** `src/sprout/rag/retrieve.py`, `tests/test_retrieve.py`.
**Do:** Use `fastembed` (ONNX) for embeddings and reranking, so serving needs no PyTorch. Choose models it supports, and record them.
**Produces:** `Retriever.search(query, *, crop=None, disease=None, k=8, mode="hybrid", rerank=True) -> list[Passage]` with `Passage: text, source, page, crop, score`; modes `dense`, `bm25`, `hybrid`. Filters each passage by score, with general guidance as a controlled fallback. The search query is separate from the conversation instructions. Embedding model and distance metric are stated in config (#37–#43).
**Tests:** `test_crop_filter_excludes_other_crops`; `test_general_docs_only_as_fallback`; `test_low_score_passages_are_dropped_individually`; `test_bm25_finds_exact_disease_name`; `test_passage_carries_source_and_page`; `test_empty_result_is_empty_not_error`.
**Accept:** `{py} -m pytest tests/test_retrieve.py -q`.

### T6.3: RAG evaluation set
**Files:** `datasets/rag/questions.jsonl`, `datasets/review/rag-sample.md`, `src/sprout/rag/evalset.py`, `tests/test_evalset.py`.
**Do:** At least 150 questions, split into dev and test by source document: supported (with gold source and page), not in the documents, wrong-crop traps, ambiguous symptoms, and multi-turn follow-ups, at least 20 each. Gold labels come from reading the passages. Write a sample of 30 for review (#53).
**Tests:** `test_gold_pages_exist_and_contain_the_answer_terms`; `test_category_counts`; `test_no_document_in_both_splits_for_supported_questions`.
**Accept:** `{py} -m pytest tests/test_evalset.py -q`.

### T6.5: More care documents, with coverage
**Agent:** use the `researcher` for lookups. **Files:** `knowledge/manifest.csv`, `docs/document-coverage.csv`, `docs/document-coverage.md`, `docs/document-sources.md`, `tests/test_more_documents.py`.
**Do:** Gather 200 to 400 public plant-care documents covering every class, starting with the uncovered ones in `docs/document-coverage.md`. Use only documents publicly accessible without login, paywall, or getting around a block, such as university extension services and government agencies. Keep each document's URL, publisher, title, and date in the manifest. Raw content stays in the git-ignored `knowledge/raw/`, and is never committed or published. Stop and tell the coordinator before any download over 40 GB, or under 40 GB free. Rebuild the index through `build_index`, listing failures. Regenerate the per-disease coverage report, reading passages as T3.4 did, and report any class still uncovered. `datasets/rag/questions.jsonl` and its gold labels must keep passing: new documents can't change a question's gold page, and `rag_test`'s documents stay out of dev choices.
**Tests:** `test_every_new_document_has_source_url_and_publisher`; `test_every_file_is_in_the_manifest`; `test_no_raw_document_is_tracked_by_git`; `test_gold_labels_still_hold`; `test_every_class_has_a_coverage_row`.
**Accept:** `{py} -m pytest tests/test_more_documents.py -q`.

### T6.4: Experiment E7, answers
Uses `Identification` from T5.0 and a fake identifier, so it doesn't wait for training. Confirm against the real `Identifier` in T7.4.
**Files:** `src/sprout/llm/client.py`, `src/sprout/rag/answer.py`, `src/sprout/rag/prompts/`, `src/sprout/experiments/e7_rag.py`, `tests/test_answer.py`.
**Produces:** `LLMClient.complete(messages, *, max_tokens, timeout_s) -> LLMResult`; `FakeLLM`, `OllamaClient`, `ClaudeCodeClient(model="sonnet")`; `class LLMError`; `answer(question, *, identification, history, retriever, llm) -> Answer` with `text, citations, grounded: bool, asks_for: list[str]`.
**Do:** `OllamaClient` uses `http://127.0.0.1:11434`, never `localhost`, which adds about 2 seconds to every call on this machine. `ClaudeCodeClient` runs `claude -p` through `shutil.which`, sends the prompt on stdin, and uses `--output-format json --tools "" --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1`, from a temp folder. It treats usage limits or a nonzero exit as `LLMError`, and refuses past the cap. Answers separate instructions, evidence, and user text, carry the diagnosis's uncertainty, use bounded history, and ask for missing details that matter. On dev: compare dense, hybrid, and hybrid + rerank for retrieval, then these upgrades, each measured one at a time, keeping only what helps: structure-aware chunking with whole tables and OCR; a context header on each chunk; and search queries built from the top-3 diseases and the conversation. No graph RAG. Then prompt versions for faithfulness and abstention, one change at a time, on one local model. Run the best on `ClaudeCodeClient` over at most 100 questions as a comparison. Writer: `qwen3.5:4b-q4_K_M`. Judge: `gemma4:e4b-it-qat`. Claude is never the judge of E7. T8.1's second figure is the one exception, approved by the person on 2026-11-04. Writes per-claim judgments to a file. Both are installed; don't pull others. Resolves #37, #46–#51, #54, #55, #57.
**Tests:** `test_answer_cites_real_passages_only`; `test_no_passages_says_no_supporting_document`; `test_retrieval_failure_is_visible`; `test_llm_outage_returns_unavailable_not_invented_text`; `test_tomato_question_cannot_cite_apple`; `test_followup_gets_bounded_history`; `test_uncertain_diagnosis_is_not_stated_as_fact`; `test_chemical_advice_says_check_the_label`; `test_call_cap_refuses_and_makes_no_call`; `test_claude_command_has_no_tools` (stubbed process); `test_ollama_client_uses_loopback_ip_not_localhost`.
**Accept:** `{py} -m pytest tests/test_answer.py -q`; `{py} -m sprout.eval.check_run E7 --min-n 50 --split rag_dev`.

### T6.6: Experiment E14, is the local judge reliable?
**Files:** `src/sprout/experiments/e14_judge_check.py`, `configs/e14.json`, `tests/test_e14.py`.
**Do:** Have Sonnet, through the existing `ClaudeCodeClient(model="sonnet")`, grade a seeded random sample of at least 200 chatbot answers the local `gemma4` judge already graded in T6.4, from `rag_dev` only, never `rag_test`. Sonnet grades per claim on the same question, passages, and rubric, without seeing the local judge's grade. This check is what the rule that Claude is never the judge in T6.4 leaves open: Sonnet's grades here measure the judge and set no E7 number. Save rows as they arrive, inside the call cap, so a crash resumes. Report agreement, Cohen's kappa with an interval, and which direction disagreements lean, including whether the local judge flatters results. Add a `T6.6` entry to `docs/DECISIONS.md`. If agreement is poor, say so and propose using Sonnet as the judge for final numbers; that choice goes to the coordinator.
**Tests:** `test_only_rag_dev_answers_are_sampled`; `test_sonnet_prompt_omits_the_local_grade`; `test_rows_are_saved_as_they_arrive_and_a_rerun_resumes`; `test_call_cap_refuses_and_makes_no_call`; `test_kappa_matches_a_worked_example`; `test_direction_counts_each_disagreement_once`.
**Accept:** `{py} -m pytest tests/test_e14.py -q`; `{py} -m sprout.eval.check_run E14 --min-n 200 --split rag_dev`.

---

## Phase 7: The application

### T7.1: Backend rebuild
Built against `IdentifierProtocol` and `FakeIdentifier`, so it doesn't wait for training. The app takes any identifier by injection.
**Files:** `src/sprout/web/` (split from `legacy/PlantProject/application.py`), `tests/test_web.py`.
**Known issues from T6.2, to fix when serving wires in the retriever:** passage vectors aren't saved, so each new `Retriever` re-embeds all 2,258 chunks on CPU, which takes minutes; and `min_score` 0.3 filters nothing with reranking off, because an off-topic query scored about 0.78 with real `bge-small` cosines, so it only works with reranking on.
**Do:** Move to an app factory with routes in modules, and one explicit database setup. Review the hardcoded care recommendations against sources, keeping those with sources and labeling or removing the rest (#52). Replace process-local state with database records scoped to the user and analysis. Config comes from the environment, with no default secret in production mode. Add CSRF, secure cookies when on HTTPS, safe redirects, upload checks, private uploads per user, payload validation, rate limits, API timeouts, and separate liveness and readiness. Store the model prediction and the user's selection separately. Resolves #27–#29, #52, #58, #61, #63–#71, #76–#79.
**Tests:** `test_prediction_and_user_selection_are_stored_separately`; `test_refresh_does_not_overwrite_selection`; `test_user_cannot_see_another_users_upload_or_history`; `test_open_redirect_is_refused`; `test_csrf_required_on_post`; `test_non_image_upload_rejected`; `test_rate_limit_applies`; `test_ready_false_when_model_missing`; `test_two_app_instances_share_state`; `test_delete_removes_row_and_file`.
**Accept:** `{py} -m pytest tests/test_web.py -q`.

### T7.2: Frontend
**Files:** `src/sprout/web/templates/`, `src/sprout/web/static/`, `tests/test_frontend.py`.
**Do:** Carry over the existing templates. Show confidence, unsure, and not-plant states. Show alternatives and the user's choice. Show clickable sources, and a clear "no supporting document" label. Render user text as text, sanitize generated markup, and consolidate the chat code. Handle errors, loading, duplicate submits, and retries. Make the report button actually store the report. Resolves #62, #83–#86. Use Playwright for browser tests.
**Tests:** `test_script_in_user_message_is_not_executed`; `test_script_in_model_markdown_is_removed`; `test_unsure_result_shows_guidance`; `test_sources_are_links`; `test_report_is_stored`; `test_double_click_sends_once`.
**Accept:** `{py} -m pytest tests/test_frontend.py -q`.

### T7.3: Container and CI
**Files:** `deploy/Dockerfile`, `deploy/entrypoint.sh`, `docker-compose.yml`, `requirements/serve.txt`, `requirements/train.txt`, `.dockerignore`, `docs/deploy.md`, `tests/test_container.py`.
**Do:** A small serving image using ONNX Runtime without PyTorch, with a locked dependency set, non-root user, no `.env`, and documented model and index provisioning. Add `docs/deploy.md` with steps for Google Cloud Run, and persistent storage choices. Set workers from measured memory. Smoke test: upload → prediction → question → history → deletion, in a clean container. Resolves #60, #72–#75, #80, #81, #87–#89.
**Tests:** real-marked `test_smoke_in_clean_container`; `test_dockerignore_excludes_env_and_data`; `test_image_runs_as_non_root`; `test_no_torch_in_serving_requirements`.
**Accept:** `docker compose build app`; `{py} -m pytest tests/test_container.py -q -m "real or not real"`.

### T7.4: Experiment E8, end-to-end performance
**The person's direction (2026-11-04), overrides the Do and phase-review items below:** about 5 requests through the real Docker image with the real model and real answers, to prove it works together, in a record with `--min-n 5`. Keep the wrong-plant-name fix (`crop=` from the photo label), the lost-citation fix for `[1,2]`, a quick floor check on the serving reranker, and the page wording that a healthy label doesn't rule out disease. Drop the load levels, the bottleneck fix and rerun, and rerank speed work. No concurrency or end-to-end speed claim comes from this.
This is the integration point: it runs the real `Identifier` from T5.2 and the real answers from T6.4 in the container, and fixes any mismatch. No fakes.
**Files:** `src/sprout/experiments/e8_perf.py`, `tests/test_e8.py`.
**Do:** In the container, on CPU, measure cold start, preprocessing, inference, retrieval, generation, end-to-end p50 and p95, and memory, under 1, 5, and 20 concurrent requests. State the machine. Fix the top bottleneck if there's a clear one, then rerun, and report before and after (#82).
**From phase review:** see `orchestration/verdicts/phase-6.md`. (1) The `hybrid+rerank` floor of 0.73875 was tuned on the torch reranker, but serving uses the fastembed ONNX one, and the rerank pool differs (30 candidates at k=10, 15 at k=5; floors were chosen at k=10, serving uses k=5). Compare per-passage scores near the floor on the real path, and report what happens to served decline rates. (2) `answer()` takes the first plant named when there is no photo or crop, so pass `crop=` from the photo label. (3) `[1,2]` citations lose their sources, so a grounded answer can show the no-document prefix. Count how often this happens on the served path and fix it if it does. (4) CPU rerank is about 600 ms per search.
**Tests:** tiny fixture writes a valid record; `test_percentiles_by_stage_are_recorded`.
**Accept:** `{py} -m pytest tests/test_e8.py -q`; `{py} -m sprout.eval.check_run E8 --min-n 5 --split perf`.

### H3: Deploy
Optional for the build. See `docs/HUMAN-GATES.md`.

---

## Phase 8: Final results

### T8.1: Experiment E9, final test
**The person's direction (2026-11-04), overrides older text below:** scored on the 29 classes. The model chosen in T4.9 is primary, and E4 and E12 are reported beside it where saved models allow. Haiku is the judge on all claims and Sonnet is a second named figure, with no local judging. The wrong-disease check is the cheap version: run on `rag_dev`, freeze the prompt, run once on `rag_test`; no blind read, and the report says the check wasn't validated. `prep/T8.1` is work in progress with a known failing test, so finish it here. The test runs once.
**Files:** `src/sprout/experiments/e9_final.py`, `tests/test_e9.py`.
**Do:** Freeze configurations chosen on dev, and commit them before touching test. Run once on `lab_test`, `field_test`, and `rag_test`, one record each: clean baseline, improved model, XGBoost, DINOv2, with and without abstention, and the quantized final model. Report lab and real-photo results separately, with `lab_test` also scored on shared classes, with top-1, top-3, top-5, macro-F1, calibration, and plant-given top-1 and top-3, stating the options per plant. RAG uses the local model as the writer; any Claude comparison stays within the call cap.
**Answer quality, decided by the person on 2026-11-04:** final answer-quality numbers on `rag_test` report two judges: the local judge as the upper figure and Sonnet as the conservative one, on the same claims with the same rubric. **Changed by the person on 2026-11-04, through the coordinator:** Haiku through `claude -p` takes the local judge's place, in parallel, if the check in `docs/DECISIONS.md` doesn't find it clearly worse; the writer stays local, and Sonnet stays as the conservative figure. The choice is read from the check's committed decision file, never typed into a config. If Haiku's supported share is below Sonnet's, the report says so and doesn't call it the upper figure. Neither figure is called accuracy. Add a wrong-disease check: a grader, Sonnet, sees the question and the diagnosis and judges whether the advice fits the right disease and plant, about one call per answer. Run it on `rag_dev` first and report its rate. Estimate its error with a blind read of a dev sample, as in `docs/e14-adjudication.md`, reporting both directions. Then freeze the prompt, record its commit and hash, and run it on `rag_test` once. Don't tune it after seeing `rag_test`.
`field_test` means its cleaned rows from T4.7, and the report states how many were removed and why. The person signs label review per split, in words, in a small tracked file; results don't hold the signature, and `claims check` reads the file, so a later sign-off doesn't require rerunning this once-only task. All splits start unsigned. H2's `done` status does not sign the photo labels: its note says they are not signed.
**From phase review:** see `orchestration/verdicts/phase-4.md`. (D4) Report every model on one basis, the cleaned rows, and name it: E4 and E5 records are on uncleaned rows, so the same model reads 0.7653 and 0.7712 on dev depending on the record.
**Tests:** `test_refuses_when_config_not_committed`; `test_refuses_second_test_run_without_override_note`; `test_labels_reviewed_is_false_before_h2_and_true_after`; `test_both_judges_are_reported_and_neither_is_called_accuracy`; `test_first_judge_follows_the_committed_haiku_check`; `test_wrong_disease_prompt_shows_question_and_diagnosis`; `test_wrong_disease_check_refuses_rag_test_when_its_prompt_hash_differs_from_the_frozen_one`.
**Accept:** `{py} -m pytest tests/test_e9.py -q`; `{py} -m sprout.eval.check_run E9 --min-n 200 --split field_test --all`; `{py} -m sprout.eval.check_run E9 --min-n 1000 --split lab_test --all`; `{py} -m sprout.eval.check_run E9 --min-n 50 --split rag_test --all`.

### T8.2: Report, README, and bullets
**The person's direction (2026-11-04):** doesn't wait for H3; the bullet says "containerized" until H3 is done. The model is named as what actually won (ConvNeXt V2 if chosen). State the class rule and the 9 dropped classes, that real photos are web photos and no phone photos were tested, and that E12's held-out gain rests on one run and seed. Photo labels stay `labels_reviewed: false` until the person signs.
**From phase review:** the report states each item under "Items for T8.2" in `orchestration/verdicts/phase-4.md`, and these: (D2) photos of plants or diseases the model doesn't support were never measured, and E5's only negatives are COCO objects, so a confident diagnosis on an unsupported plant is untested; seven lab classes have no real-photo row. (D5) Move the ORB leak search from `orchestration/evidence/leak_orb.py` into `tools/` with its hard-coded paths and lint errors fixed; state that the leak test skips when raw images are absent. (D6) Fix the stale lines: `docs/extra-sources.md:13` says Plant Pathology 2021 is not used; `docs/datasets.md` lines 3, 53, 78, 83 say nothing was downloaded and PlantSeg access is not done; PlantSeg's approval has no who or when; pretrained ConvNeXt and DINOv2 weight licenses aren't recorded. Also the E12 choice per its gates entry in `docs/DECISIONS.md`.
**Files:** `docs/REPORT.md`, `README.md`, `results/claims.json`, `docs/RESUME-TARGETS.md`, `docs/interview-notes.md`, `tools/check_report_numbers.py`.
**Do:** Write a model and evaluation report: data, splits, baselines, every experiment including failures, limits, and failure examples, with no COCO or PlantDoc images embedded. Always report seen-source and held-out-source numbers side by side, and record the crop as tested and left out. State E4's held-out-source scores beside the joint score: the joint score mostly reflects sources seen in training, and a new source scores much lower. Remeasure the effect of same-scene `field_train` look-alikes on the final model on `field_dev`, in a record, and state it. Plant-given top-3 is easy, since plants average 2.7 diseases: state that and the options per plant wherever it appears. Report thin classes as thin. Include `docs/document-coverage.md`'s gaps. Report PlantDoc and PlantSeg separately as secondary numbers, with the headline on all of each split, and state E2's sample finding on fruit photos and doubtful labels; `docs/field-dedup.md` records why there's no leaf-only subset. State that the run-record checker guards against mistakes and casual tampering, not someone determined to rewrite git history. If any weights were trained on PlantDoc images, the report flags plainly that the person must check before publishing those weights. Fill `results/claims.json` from run records, then fill the bullets, rewording any that results don't support. Correct the README's claims (#90–#93). `docs/interview-notes.md` explains each decision, tradeoff, and failure, so the person can defend them. The duplicate search on the field set was the last one; a few same-scene images could remain across splits, as a known limit of web-scraped data.
**Tests:** `tools/check_report_numbers.py` fails if a number in `docs/REPORT.md` or the bullets isn't in a run record, if any `[placeholder]` remains, or if `results/claims.json` is empty.
**Accept:** `{py} -m sprout.eval.claims check`; `{py} tools/check_report_numbers.py`; `{py} -m pytest -q`.

### H4: Final sign-off
See `docs/HUMAN-GATES.md`.

---

## Later stages, not started

Set by the person on 2026-11-03. 
- **Stage 2 and 3:** T4.3, T4.4, T4.5, T4.6, and T6.5 are in the plan. By the person's later rule, public data needs no one-by-one approval if it requires no login, paywall, or getting around a block.
