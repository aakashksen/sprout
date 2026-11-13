VERDICT: DEFECTS

Phase 4 (T4.0 to T4.8: E10, E4, E5, E11, E12, E13, extra sources). Reviewed on `build` at 2122505. No code, GPU, Claude or Ollama was used. The only images opened were EXIF headers of `field_dev`, `field_train` and extra images; no `field_test` image or prediction row was opened.

## What held up

- `check_run` passes for E10 (seg_dev), E4, E5, E11, E12, E13 on the committed records (run from `.venv` with `PYTHONPATH=src`).
- No `field_test` read in any phase-4 experiment: `grep field_test` over `src/sprout` shows only manifest-row counts (class report), copy-check inputs (T4.5/T4.8, which drop extras), the E13 label review (disclosed), and "never read" statements. E4, E5, E11 and E12 records say `field_test` was not read, and E12's `field_test_read` is false.
- Row sets are consistent where they are compared. E12 and E13 score E4's saved rows filtered to the cleaned rows (1,028 / 610 / 418), so every E12-versus-E4 pair is on identical rows. E4's own records are on 1,040 / 619 / 421.
- `results/e12/decision.json` `chosen_model_sha256` (362a11f3...) equals the sha256 of `models/e4/convnext_joint/best.pt`.
- Train, evaluation and E5 all go through `sprout.preprocess.preprocess` (`src/sprout/train/dataset.py:10`, `src/sprout/eval/predict.py:21`, `src/sprout/experiments/e5_confidence.py:169`).
- Every source in use has a section in `docs/datasets.md`: PlantVillage, PlantDoc, PlantSeg, COCO, segmented PlantVillage, PlantWild (CC BY-NC-ND 4.0, publishing needs the person), Plant Pathology 2021 (third-party, terms unconfirmed).
- The phase's own tests: the run of the eight phase-4 test files (`-x`, one process) printed 109 passing dots (about 75% of the run, no F) and then ended with no summary line; the 900 s `timeout` most likely killed it in the slow real-manifest copy test. The exit code 0 was `tail`'s. So: no failure seen, but the run is incomplete and is not evidence that the suite passes.

## Defects

### D1. E5's `chosen.json` has no link to the model it was fitted on (owner: T5.1 / T5.2; T8.2 must state)
- File: `results/e5/chosen.json` (no checkpoint, sha or record id; keys are seed, fit_percent, fit_rows, temperature, min_coverage, abstain_threshold, gates, gate). `src/sprout/serve/gate.py:13` and `:70-73` (`load_settings`) read only `gate.settings`.
- Repro: `.venv/Scripts/python -c "import json;print(list(json.load(open('results/e5/chosen.json'))))"`. `results/e12/decision.json` does carry the sha, so the model choice is bound and the calibration is not. The binding exists only in `configs/e5.json` (`model.checkpoint`, `e4_record`).
- Impact: `Identifier(model_path, calibration_path)` (T5.2) can pair any model with T=1.1927, abstain 0.885 and the gate. This matters in three live cases: the coordinator's conditional E12 override (E5 must be refit; a stale file would load without error), T8.1 retraining, and T5.1's ONNX or quantized model. The fit used `log(probs)` of E4's saved float32 PyTorch rows as logits, and the 0.885 threshold sits where coverage is only just 0.5, so small probability shifts from quantization change coverage and confident error (the T4.2 verdict notes coverage was already 0.497 on the scoring half). Nothing in the repo measures the gate on the exported model.
- Ask: T5.1/T5.2 should write the model sha and E5 record id into the calibration file and refuse a mismatch. T5.1 should re-score the gate on the ONNX and quantized probabilities. T4.2's note that `decide` can return an empty `top_k` is still open for T5.2.

### D2. Real photos of unsupported plants and diseases are never measured; the unsure and not-plant claims rest on COCO objects only (owner: T8.1; T8.2 must state)
- Evidence: `datasets/class_map.csv` has 5,922 PlantSeg images in 94 `excluded` classes and 2 `dropped` PlantDoc classes. `datasets/class_map_extra.csv` has 159 excluded source classes (19,834 images). None of these are in `field.csv`. Command: `.venv/Scripts/python -I -c "import pandas as pd;c=pd.read_csv('datasets/class_map.csv');print(c.groupby(['source','status']).n_images.sum())"`. E5's negatives are COCO 2017 val only (`configs/e5.json`: `not_plant_split`). Seven lab classes have no `field.csv` row at all (Corn healthy, Potato healthy, Grape Esca, Grape leaf blight, Peach bacterial spot, Tomato spider mites, Tomato Target Spot), so a real photo called one of those is unscored. Corn healthy and Potato healthy do have extras in training.
- Impact: DESIGN's anti-goal ("a confident diagnosis for a photo it should have called unsure, and that case wasn't measured") is not tested for the most likely real failure, a diseased leaf of a plant or disease outside the 38 classes. The E5 gate rates (95-98% of COCO objects rejected) say nothing about it. The PlantSeg-excluded images are not `field_test`, so they could serve as dev-side negatives.
- Ask: T8.1 should report confident-wrong rates on held-out out-of-scope plant photos, drawn without touching `field_test`, and the accuracy for the 7 classes with no real photos (as "not measured" if still none).

### D3. EXIF orientation is applied nowhere in training or evaluation, but T5.2 will apply it (owner: T5.1 / T5.2)
- Evidence: `grep -rn -i exif src tools` finds nothing. `predict.py:31-33` and `dataset.py` open the file and call `preprocess` directly. Real rotation tags (3, 6, 8): `field_dev` 6 of 1,040 (tag 6; three more carry an invalid tag 0), `field_train` 14 of 2,069, extras 4 of 3,484. Script: `<temp>`, run as `.venv/Scripts/python -I`.
- Impact: small here (about 0.6% of dev). But the plan's T5.2 test `test_preprocess_matches_evaluation_exactly` and `test_exif_rotation_gives_same_result_as_upright` can both pass only if the order is made explicit: the served path would rotate while every recorded number was computed on unrotated pixels. Phone photos, the intended input, usually carry the tag, and the training data nearly never does.
- Ask: T5.2 decides it in one place (the shared `preprocess` entry point or an `open_image` helper used by train, eval and serve) and records the effect on the 9 dev images.

### D4. Mixed row sets across phase-4 numbers: uncleaned E4 and E5, cleaned E11/E12/E13 (owner: T8.1 / T8.2)
- Evidence: E4 records n=1,040 / 619 / 421, subset `shared_classes`. E5 `E5-20261103T170832277859Z` n=517 (half of the uncleaned 1,040), and its `chosen.json` fit uses 523 uncleaned dev rows. E11, E12 and E13 use `shared_classes+clean:1` with 1,028 / 610 / 418. The same E4 model is 0.7653 macro-F1 in `docs/DECISIONS.md:306` and 0.7712 in E12/E13 (`DECISIONS.md:358`). The 12 removed dev rows include 6 not-plant photos, which E5 counts as plant photos when it limits plant rejection to 5% (I did not count how many of the 6 fall in the fit half).
- No wrong comparison was found: each pair is on the same rows, and DECISIONS states both. The risk is a final report that quotes E4/E5 and E12/E11 side by side without saying so.
- Ask: T8.1/T8.2 pick one basis (`clean:1`) and restate E4 and E5 on it. If E5 is refit for any reason, refit on `clean:1`.

### D5. The independent ORB leak search for T4.8 cannot be reproduced from the repo (owner: T8.2, or a small follow-up before T8.1)
- Evidence: `docs/extra-sources.md:164` reports the search with specific results (no pair reached 20 mutual matches; the positive control gave 82 inliers; run in two parts). `git ls-files | grep -i orb` returns nothing, and `tools/` has no such script. The T4.8 verdict (note 1) says `leak_orb.py` and its logs are in a scratchpad. Its dependency (scikit-image) is not declared by the repo either.
- Related: `tests/test_extra_sources.py:277` (`test_no_extra_is_a_near_copy_of_a_field_dev_or_field_test_photo`, the plan's main leak guard) skips when `datasets/raw` images are absent, so on a clean checkout or CI it checks nothing. It passes here only because the raw images are present.
- Impact: the "no extra is a copy of a held-out photo" claim for the 137 Plant Pathology rows rests partly on a check nobody can rerun. The pipeline's own checks are in the repo.
- Ask: commit the script to `tools/` with its thresholds and the control, or say in T8.2 that the claim is one-off.

### D6. Stale statements in docs (owner: whoever next edits them; T8.2 for the report)
- `docs/extra-sources.md:13` (Sources table) says Plant Pathology 2021 is "not used", but the manifest contains 137 of its rows (`extra_train.csv`, `source=plantpathology2021`), E12 trained on them, and the T4.8 section at line 130 says so. `docs/DECISIONS.md:132` ("T4.5 - Plant Pathology 2021 not used") is the earlier decision; its later reversal is in the T4.8 entries.
- `docs/datasets.md:3` ("Nothing was downloaded or installed"), `:53` (PlantSeg "Access: ... (not done)"), `:78` ("Size: Not downloaded") and `:83` ("No segmented PlantVillage images are on this machine") are no longer true: `datasets/raw/lab_segmented` exists, arm 3 of E4 used it (`DECISIONS.md:318`), and PlantSeg is downloaded. `:54` records PlantSeg `Approved: yes` with no who or when. `tools/check_datasets_doc.py` passes because it checks only for the lines' presence.
- The licenses of the pretrained weights used (ImageNet ConvNeXt-Tiny, DINOv2 small) are not in `docs/datasets.md` or elsewhere in the docs (`grep -il imagenet docs/*.md` shows no license line). This is relevant to the person's pre-publishing check of the weights.
- Impact: a reader of the README tree gets the wrong source list. Low.

### D7. Dropped E5 worker test; DECISIONS says the reason is "not recorded" (owner: none; low)
- `tests/test_e5.py`: `test_score_images_runs_with_worker_processes` removed in `933e5bf`, whose message gives the reason (WinError 6, covered by the pickle test). `docs/DECISIONS.md` ("T4.2 - Dataset class defined at module level") still says the reason is not recorded. No real weakening found, but the entry contradicts the commit.

## Items for T8.2's report (not new defects; each must be stated)

1. Reviewers saw `field_test` photos with labels to clean them, and 19 of the removals are `field_test` rows (`docs/label-review.md`, T4.7 verdict note 5). No score used it. T8.1 must restate this.
2. Only 7 of 400 sampled `field_train` rows were reviewed and removed; `field_clean.csv` (4,106 rows) is cleaned for dev and test and only sample-reviewed for train. The sheets are conservative (about 2 of 4 known-bad found), so every removal rate is a lower bound.
3. The E12 gates G1 to G3 and the override reasoning were fixed after the dev numbers were seen (`DECISIONS.md`, "Gates for choosing E12's model"). If E12 is chosen, T8.2 states the override. E5 is refit from `models/e12/convnext_joint_extra/field_dev.predictions.jsonl`, which is git-ignored (D1).
4. E4 and E12 are one seed each. The convnext arms differ by 0.01 to 0.02 macro-F1, so the arm ranking is not shown to exceed noise. The E12 margin of 0.02 is a judgment (`DECISIONS.md`).
5. About 40 of 1,040 `field_dev` photos are same-scene look-alikes of `field_train` photos (T4.1 verdict), so seen-source dev numbers are optimistic. No equivalent check was run for `field_test`, which T8.1 should run before quoting a test number.
6. The extras survivors are, by construction, the photos least like `field_dev` and `field_test` (the crop rule over-removed, known). Plant Pathology scab rows (106) come from one orchard style and healthy has none (T4.8 verdict note 3), so E12's Apple healthy precision fall (0.889 to 0.633) is consistent with a style shortcut and should not be read as general.
7. The disease crop (E10, IoU 0.54 on PlantSeg-style photos, no PlantDoc transfer measured) made E4 significantly worse (-0.029 macro-F1), so serving has no segmentation model. T5.1 exports only the classifier.
8. Weights trained on PlantDoc, PlantWild (no-derivatives term) and Plant Pathology 2021 (terms unconfirmed) need the person's check before publishing. E10's PlantSeg weights are CC BY-NC.
