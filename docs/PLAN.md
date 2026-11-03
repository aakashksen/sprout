# SPROUT V2: implementation plan

> **For agentic workers:** run this with the `/run-plan` skill in this folder. It uses the project's own `implementer`, `verifier`, and `reviewer` agents and the `tools/taskctl.py` gate. Do not use another execution workflow.

**Goal:** Rebuild SPROUT so its model is measured honestly on real photos, improved with evidence, and served with calibrated confidence and cited advice.

**Architecture:** Training lives in a tested Python package with permanent data manifests and tracked experiments. The app loads an exported ONNX model and shares preprocessing with evaluation. The RAG pipeline is rebuilt around document metadata and citations, with its own evaluation.

**Tech stack:** Python 3.12, PyTorch, timm, scikit-learn, XGBoost, MLflow, DVC, ONNX Runtime, Grad-CAM, ChromaDB, sentence-transformers, Ollama, Flask, Docker.

**Spec:** `docs/DESIGN.md`. Testing rules: `docs/TESTING.md`. Original issue list: `docs/backlog.md`.

## Global constraints
- Python 3.12 in `.venv`; package `sprout` under `src/`, installed with `pip install -e ".[dev]"`.
- `.venv/Scripts/python -m pytest -q` needs no GPU, network, or spending. Real calls use `@pytest.mark.real`.
- Numbers come only from `results/runs/`.
- 8 GB VRAM shared with other projects, one GPU job at a time, mixed precision on.
- Product calls to Claude capped at 1,000, on Haiku, counted in `results/claude_calls.json`. No API keys. DeepSeek is removed.
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
**Do:** Train a small segmentation model on PlantSeg's masks, from the `plantseg.zip` already downloaded, with no new download. Use only images not in `field_dev` or `field_test`. Exclude by group as well, and check PlantSeg images outside `field.csv` for near-duplicates of `field_dev` and `field_test` with the rule in `docs/field-dedup.md`, so no held-out photo reaches training. Write `seg.csv` with splits `seg_train` and `seg_dev`, and teach `check_run` the `seg_dev` split, recomputing per-image IoU from saved rows. Keep only best and latest. Report IoU honestly, and that PlantSeg's masks mark disease on PlantSeg-style photos and may not carry over to PlantDoc.
**Tests:** `test_no_field_dev_or_field_test_photo_in_seg_manifest`; `test_near_duplicate_of_held_out_photo_is_excluded`; `test_crop_includes_margin`; `test_tiny_mask_falls_back_to_full_image`; `test_saved_iou_is_recomputed`.
**Accept:** `{py} -m pytest tests/test_e10.py -q`; `{py} -m sprout.eval.check_run E10 --min-n 50 --split seg_dev`.

### T4.1: Experiment E4, improvements
**Consumes:** `docs/class-report.md` from T3.4, metrics from T3.5, `crop_to_disease` from T4.0.
**Files:** `src/sprout/train/joint.py`, `src/sprout/train/augment.py`, `src/sprout/experiments/e4_improve.py`, `configs/e4.json`, `tests/test_e4.py`.
**Do:** Train on real and lab photos together from the start: `field_train` (PlantDoc and PlantSeg) plus `lab_train` capped per class, so lab can't dominate. Fix the cap in `configs/e4.json` from the class report, before any E4 result is seen. PlantDoc images are cleared for this for local use: don't republish them, and record whether the resulting weights were trained on them. One change at a time, all listed in `attempts`, selected on `field_dev`: the joint baseline; stronger color and blur augmentation; background randomization of lab photos, with leaves cut out using the official segmented images, matched to `lab_train` by filename so nothing leaks, and pasted on varied backgrounds, falling back to masks from `sprout.data.masks` (HSV and GrabCut on `lab_train` images only) and recording this arm as not run if good masks can't be made; both; and, as a separate arm, with and without the T4.0 crop. Include at most 6 training runs for these arms. Also score a held-out-source run as a secondary number: train with one real source left out, and test on that source's part of `field_dev`, once for each source, so 2 more runs, each in `attempts`. Report before and after against E1 and the E3 baselines with paired statistics, top-1, top-3, top-5, macro-F1, calibration, plant-given top-1 and top-3, and Grad-CAM energy outside the leaf on lab images. If nothing beats the baseline, record that, and keep the baseline. If the crop doesn't help, say so and leave it out of serving (#19).
**Tests:** `test_background_paste_keeps_leaf_pixels_and_changes_background`; `test_field_test_is_never_read` (the loader raises if asked); `test_lab_cap_limits_every_class`; `test_field_dev_is_never_trained_on`; `test_held_out_source_is_absent_from_training`; `test_crop_arm_uses_full_image_on_tiny_mask`; `test_attempts_lists_each_run`.
**Accept:** `{py} -m pytest tests/test_e4.py -q`; `{py} -m sprout.eval.check_run E4 --min-n 200 --split field_dev`.

### T4.2: Experiment E5, calibration, abstention, and input checks
**Files:** `src/sprout/serve/calibrate.py`, `src/sprout/serve/gate.py`, `src/sprout/experiments/e5_confidence.py`, `tests/test_e5.py`.
**Produces:** `fit_temperature(logits, labels) -> float`; `choose_threshold(probs, labels, min_coverage=0.5) -> float`, on `field_dev`; `decide(probs, threshold) -> Decision` with `status: answered|unsure`, top-k alternatives (k≤3, each above 10%); `is_plant_image(image, probs) -> bool`.
**Do:** The official release has no background class, so compare these input checks: the old 5% green-pixel rule, model confidence alone, and model confidence plus an out-of-distribution score. Measure both rejection rates on `field_dev` and `not_plant` (#22–#24, #26). `check_run` trusts each row's `accepted` flag, so save the confidences and threshold and add a check that the flags follow from them.
**Tests:** `test_temperature_reduces_ece_on_overconfident_fixture`; `test_threshold_is_chosen_without_test_data`; `test_unsure_below_threshold`; `test_top_k_is_limited`; `test_yellow_brown_leaf_fixture_is_not_rejected_by_chosen_gate`; `test_green_non_plant_fixture_is_measured_not_silently_passed`.
**Accept:** `{py} -m pytest tests/test_e5.py -q`; `{py} -m sprout.eval.check_run E5 --min-n 200 --split field_dev`.

---

## Phase 5: Efficient serving

### T5.0: Identification contract and fake identifier
**Files:** `src/sprout/serve/types.py`, `tests/test_identify_types.py`.
**Produces:** `Identification` with `status: answered|unsure|not_plant|invalid`, `top`, `alternatives`, `guidance`; `class IdentifierProtocol`; `FakeIdentifier(results)`. This lets the chatbot and the app be built while training runs. T5.2 implements the real one, and T7.4 and T8.1 use only the real one.
**Tests:** `test_fake_returns_each_status`; `test_identification_round_trips_as_json`. T2.3 owns `test_real_runs_cannot_use_the_fake`, since it builds `check_run`.
**Accept:** `{py} -m pytest tests/test_identify_types.py -q`.

### T5.1: Experiment E6, ONNX and quantization
**Files:** `src/sprout/serve/export.py`, `src/sprout/experiments/e6_onnx.py`, `tests/test_e6.py`.
**Produces:** `export_onnx(checkpoint, path)`; `quantize(path, mode, calibration_images=None) -> Path`, with calibration images from `lab_dev` for static mode; `OnnxModel(path).predict(image) -> probs`.
**Do:** If E4 kept the T4.0 crop, export the segmentation model too and include it in every latency number. Compare PyTorch CPU, ONNX, and quantized ONNX: size, p50 and p95 latency on CPU, macro-F1 on `field_dev`, and share of predictions that changed. State the CPU in the record. If quantization costs a measurable loss, report it, and choose by the numbers.
**Tests:** `test_onnx_matches_torch_within_tolerance` (tiny model); `test_quantized_runs_and_is_smaller`; `test_class_map_travels_with_the_model`.
**Accept:** `{py} -m pytest tests/test_e6.py -q`; `{py} -m sprout.eval.check_run E6 --min-n 200 --split field_dev`.

### T5.2: Inference service
**Consumes:** `Identification` from T5.0.
**Files:** `src/sprout/serve/identify.py`, `tests/test_identify.py`.
**Produces:** `Identifier(model_path, calibration_path).identify(image_bytes) -> Identification` with `status: answered|unsure|not_plant|invalid`, `top`, `alternatives`, `guidance`. Uses the shared `preprocess`, fixes EXIF orientation, and bounds dimensions (#25, #30, #31, #63).
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

### T6.4: Experiment E7, answers
Uses `Identification` from T5.0 and a fake identifier, so it doesn't wait for training. Confirm against the real `Identifier` in T7.4.
**Files:** `src/sprout/llm/client.py`, `src/sprout/rag/answer.py`, `src/sprout/rag/prompts/`, `src/sprout/experiments/e7_rag.py`, `tests/test_answer.py`.
**Produces:** `LLMClient.complete(messages, *, max_tokens, timeout_s) -> LLMResult`; `FakeLLM`, `OllamaClient`, `ClaudeCodeClient(model="haiku")`; `class LLMError`; `answer(question, *, identification, history, retriever, llm) -> Answer` with `text, citations, grounded: bool, asks_for: list[str]`.
**Do:** `OllamaClient` uses `http://127.0.0.1:11434`, never `localhost`, which adds about 2 seconds to every call on this machine. `ClaudeCodeClient` runs `claude -p` through `shutil.which`, sends the prompt on stdin, and uses `--output-format json --tools "" --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1`, from a temp folder. It treats usage limits or a nonzero exit as `LLMError`, and refuses past the cap. Answers separate instructions, evidence, and user text, carry the diagnosis's uncertainty, use bounded history, and ask for missing details that matter. On dev: compare dense, hybrid, and hybrid + rerank for retrieval, then prompt versions for faithfulness and abstention, one change at a time, on one local model. Run the best on `ClaudeCodeClient` over at most 100 questions as a comparison. Writer: `qwen3.5:4b-q4_K_M`. Judge: `gemma4:e4b-it-qat`. Claude is never the judge. Writes per-claim judgments to a file. Both are installed; don't pull others. Resolves #37, #46–#51, #54, #55, #57.
**Tests:** `test_answer_cites_real_passages_only`; `test_no_passages_says_no_supporting_document`; `test_retrieval_failure_is_visible`; `test_llm_outage_returns_unavailable_not_invented_text`; `test_tomato_question_cannot_cite_apple`; `test_followup_gets_bounded_history`; `test_uncertain_diagnosis_is_not_stated_as_fact`; `test_chemical_advice_says_check_the_label`; `test_call_cap_refuses_and_makes_no_call`; `test_claude_command_has_no_tools` (stubbed process); `test_ollama_client_uses_loopback_ip_not_localhost`.
**Accept:** `{py} -m pytest tests/test_answer.py -q`; `{py} -m sprout.eval.check_run E7 --min-n 50 --split rag_dev`.

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
This is the integration point: it runs the real `Identifier` from T5.2 and the real answers from T6.4 in the container, and fixes any mismatch. No fakes.
**Files:** `src/sprout/experiments/e8_perf.py`, `tests/test_e8.py`.
**Do:** In the container, on CPU, measure cold start, preprocessing, inference, retrieval, generation, end-to-end p50 and p95, and memory, under 1, 5, and 20 concurrent requests. State the machine. Fix the top bottleneck if there's a clear one, then rerun, and report before and after (#82).
**Tests:** tiny fixture writes a valid record; `test_percentiles_by_stage_are_recorded`.
**Accept:** `{py} -m pytest tests/test_e8.py -q`; `{py} -m sprout.eval.check_run E8 --min-n 100 --split perf`.

### H3: Deploy
Optional for the build. See `docs/HUMAN-GATES.md`.

---

## Phase 8: Final results

### T8.1: Experiment E9, final test
**Files:** `src/sprout/experiments/e9_final.py`, `tests/test_e9.py`.
**Do:** Freeze configurations chosen on dev, and commit them before touching test. Run once on `lab_test`, `field_test`, and `rag_test`, one record each: clean baseline, improved model, XGBoost, DINOv2, with and without abstention, and the quantized final model. Report lab and real-photo results separately, with `lab_test` also scored on shared classes, with top-1, top-3, top-5, macro-F1, calibration, and plant-given top-1 and top-3, stating the options per plant. RAG uses the local model; any Claude comparison stays within the call cap.
**Tests:** `test_refuses_when_config_not_committed`; `test_refuses_second_test_run_without_override_note`.
**Accept:** `{py} -m pytest tests/test_e9.py -q`; `{py} -m sprout.eval.check_run E9 --min-n 200 --split field_test`.

### T8.2: Report, README, and bullets
**Files:** `docs/REPORT.md`, `README.md`, `results/claims.json`, `docs/RESUME-TARGETS.md`, `docs/interview-notes.md`, `tools/check_report_numbers.py`.
**Do:** Write a model and evaluation report: data, splits, baselines, every experiment including failures, limits, and failure examples, with no COCO or PlantDoc images embedded. Plant-given top-3 is easy, since plants average 2.7 diseases: state that and the options per plant wherever it appears. Report thin classes as thin. Include `docs/document-coverage.md`'s gaps. Report PlantDoc and PlantSeg separately as secondary numbers, with the headline on all of each split, and state E2's sample finding on fruit photos and doubtful labels; `docs/field-dedup.md` records why there's no leaf-only subset. State that the run-record checker guards against mistakes and casual tampering, not someone determined to rewrite git history. If any weights were trained on PlantDoc images, the report flags plainly that the person must check before publishing those weights. Fill `results/claims.json` from run records, then fill the bullets, rewording any that results don't support. Correct the README's claims (#90–#93). `docs/interview-notes.md` explains each decision, tradeoff, and failure, so the person can defend them. The duplicate search on the field set was the last one; a few same-scene images could remain across splits, as a known limit of web-scraped data.
**Tests:** `tools/check_report_numbers.py` fails if a number in `docs/REPORT.md` or the bullets isn't in a run record, if any `[placeholder]` remains, or if `results/claims.json` is empty.
**Accept:** `{py} -m sprout.eval.claims check`; `{py} tools/check_report_numbers.py`; `{py} -m pytest -q`.

### H4: Final sign-off
See `docs/HUMAN-GATES.md`.

---

## Later stages, not started

Set by the person on 2026-11-03. Don't start these without the person's word, and download nothing new for them.
- **Stage 2:** fill the document gaps from `docs/document-coverage.md`, and build a follow-up question bot on top of top-3.
- **Stage 3:** only if stage 1 clearly beats the baseline: more photo sources for thin classes, with licenses checked by the person first.
