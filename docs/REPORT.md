# SPROUT model and evaluation report

Written for T8.2 from the run records in `results/runs/`. Every number here is a value in one of them, and `tools/check_report_numbers.py` fails if one is not. The bullet numbers also go through `results/claims.json`.

**Read this first.**
- The photo labels are not signed. `orchestration/label_signoff.json` has `signed: false` for `lab_test`, `field_test` and `rag_test`, and every record has `labels_reviewed: false`. The person's H2 note says the text samples were fine and the photo labels are not signed. Whether H2 covers `rag_test` was never settled; this report treats `rag_test` as unsigned too. Every test number below is therefore provisional, and `python -m sprout.eval.claims check --final` refuses all of them.
- Every claim in `results/claims.json` is marked provisional: `lab_f1`, `field_f1_before`, `field_f1_after`, `field_top3_after`, `field_gain_f1`, `field_accuracy`, `field_ece`, `field_ece_calibrated`, `abstain_coverage`, `abstain_accuracy`, `abstain_error`, `xgboost_f1`, `dinov2_f1`, `vs_dinov2_f1`, `vs_e4_f1`, `vs_e12_f1`, `quantized_f1`, `quantized_vs_float_f1`, `rag_answers`, `rag_declines`, `rag_declines_crop` and `rag_declines_ambiguous`. No claim can rest on a development record, because `claims check` refuses a record with `labels_reviewed` false unless it is an E9 test split with a provisional mark.
- Real photos here are web photos (PlantDoc, PlantSeg, PlantWild and Plant Pathology 2021). No phone photos were tested, so no number below is a statement about a phone photo.
- Development numbers (`field_dev`, `rag_dev`) picked the models and settings, so they are optimistic. Test numbers were produced once each, with the configuration frozen first.

## Summary
The model that won T4.9 is ConvNeXt V2 Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k`, fine-tuned here on 29 classes). It was picked by a rule fixed before it had a score: the highest macro-F1 on the cleaned `field_dev` rows among it (0.81), the E12 ConvNeXt (0.7904) and the E4 ConvNeXt (0.76). Training saw 19260 different photos (6342 real web photos and 12918 lab photos).

On `field_test` (1006 photos of the 29 classes, provisional) it scores macro-F1 0.8111, accuracy 0.8131 and top-3 accuracy 0.9642. The clean lab-trained model on the same photos scores macro-F1 0.2387. On `lab_test` (6443 photos) the new model scores macro-F1 0.9859 against the lab model's 0.9984, so joint training cost a little on lab photos.

Seen-source and held-out-source numbers are side by side under "Domain shift and held-out sources". The short form: the field test photos come from the same web sources as the training photos, and a source the model has not seen scores far lower (E4 on `field_dev`: 0.7653 joint, against 0.3778 with PlantDoc held out and 0.5185 with PlantSeg held out). The final model has no held-out-source run.

The limits are in the sections below: one seed per training run, web photos only, the final model's dev gain over E12 has an interval that includes zero, unsupported plants were never measured, the photo gate refuses about 3.6% of real plant photos, and the answer system's wrong-disease check is not validated and only partly run.

## Data and sources
- **Lab photos:** the official 38-class PlantVillage release (CC BY-SA 3.0), plus its segmented set for one E4 arm. The local 39-class copy was not used.
- **Real photos:** PlantDoc (CC BY 4.0 per its LICENSE file) and PlantSeg (CC BY-NC 4.0, from its newest Zenodo record) make `field.csv`. PlantWild (CC BY-NC-ND 4.0) and Plant Pathology 2021 (from a third-party mirror its uploader tags apache-2.0; the authors' terms are unconfirmed) were added as extra training sources. The person approved the dataset choices at H1 and the later additions through the coordinator, for local, non-commercial, unpublished use. No raw image is committed.
- **Not-plant photos:** COCO 2017 validation. Its terms page could not be reached; the person approved it directly for local non-commercial evaluation.
- **Pretrained weights:** `docs/pretrained-weights.md`. The ConvNeXt V2 weights show `cc-by-nc-4.0` on their model card; the original ConvNeXt-Tiny weights have no license statement a primary source gives.
- Licenses and approvals are in `docs/datasets.md`; the stale lines the phase 4 review found there (D6) are corrected, and the ORB leak search (D5) is now `tools/leak_orb.py`.

## Classes kept and dropped
The final model serves 29 classes. The rule: a class is kept only with at least 80 real training photos and at least 10 real photos each in `field_dev` and `field_test`. Test rows were counted for this and no `field_test` photo was opened.

The 9 dropped classes are Grape Esca, Grape leaf blight, Peach bacterial spot, Tomato spider mites, Tomato target spot, Corn healthy, Potato healthy, Strawberry leaf scorch and Cherry powdery mildew. Seven of them (all but the last two) have no real photo at all; Strawberry leaf scorch and Cherry powdery mildew have 5 to 6 real photos each in `field_dev` and `field_test`. A photo of a dropped class gets no correct answer from the final model: it can only be called something else or left unsure. On `field_test`, 10 photos of 2 dropped classes were left out of scoring, and on `lab_test` 1694 photos of the 9.

Scores for E4 and E12 on the 29 classes are their saved predictions with the other outputs set to zero and each row renormalized, by a rule fixed before any score was seen.

## Splits and leakage controls
- Lab photos are grouped by source ID plus perceptual hash. A source ID is not proven to identify one physical leaf, so the splits are near-duplicate-grouped, not leak-free.
- Real photos are web scrapes, so one photo appears several times. Pairs are proposed by hashes and merged only when a foreground correlation confirms them, and some pairs the rule missed were linked by eye. `docs/field-dedup.md` has the rule and how its thresholds were chosen. The rule's false-merge rate is about 3% from a small viewed sample, an estimate.
- Extras are checked against every real split by correlation, a crop-aware check and DINOv2 similarity. A separate ORB keypoint search (`tools/leak_orb.py`, reproducible from the repo) found no copy of the 137 Plant Pathology rows among the `field_dev` and `field_test` photos, with a positive control found. The test that guards the real manifest, `test_no_extra_is_a_near_copy_of_a_field_dev_or_field_test_photo`, skips when the raw images are absent, so on a clean checkout or CI it checks nothing.
- The duplicate search on the field set was the last one. A few same-scene images could remain across splits, as a known limit of web-scraped data. The section "Same-scene look-alikes" measures how many reach `field_dev`.
- **Not checked for `field_test`:** no same-scene look-alike search was run between `field_test` and `field_train` (phase 4 review item 5). The `field_dev` measurement is the only one, so the seen-source test numbers may be somewhat optimistic by the same mechanism.

## Row sets and label cleaning
- Real-photo numbers here use the cleaned rows (exclusion version 1) wherever they can: `field_dev` 1028 rows on the shared classes (1018 on the 29) and `field_test` 1016 of 1035 rows (19 removed, then 10 more left out by the class rule). The 19 removed from `field_test` are 11 not-plant photos, 6 wrong plants and 2 contradicted labels. Older E4 and E5 records are on the uncleaned 1040 rows or on half of them; where a table below uses them, the row count is stated.
- **Reviewers saw `field_test` photos with their labels and file names while cleaning, and 19 of the removals are `field_test` rows.** No model output was seen. This is a use of the test split.
- Only a seeded sample of 400 `field_train` photos was reviewed. `field_clean.csv` is cleaned for dev and test and only sample-reviewed for train. The review sheets are conservative (they found about half of the known-bad photos), so every removal rate is a lower bound.
- Cleaning moved every dev score by under 1 point and left the ranking unchanged (E13 records).

## Baselines
All on `field_dev` unless stated, development numbers, one seed.
- **E1, the lab-trained ConvNeXt-Tiny:** `lab_dev` accuracy 0.9978; on `field_dev` accuracy 0.224 and macro-F1 0.199 (31 shared classes, all 38 outputs kept, 1040 photos), ECE 0.4617. A restore after the first checkpoint was lost reproduced the first run byte for byte. The legacy checkpoint's 1.0 is not performance: its split was random per image with no grouping.
- **E2, the fall:** a hand sample of 50 mistakes sorted into background, similar symptoms, wrong label, unsupported condition and blur (counts in `results/error-analysis-e2.md`). These describe the sample, not a proven cause. PlantDoc and PlantSeg score about the same, so fruit photos do not appear to be driving the fall; that is a weak signal.
- **E3:** XGBoost on HOG and colour histograms 0.0413 accuracy on `field_dev` (0.8974 on `lab_dev`); frozen ConvNeXt 0.2519; DINOv2 small with a linear head 0.4067. One untuned head setting for the frozen systems.
- On `field_test` (29 classes, provisional): clean lab model macro-F1 0.2387, XGBoost 0.0432, DINOv2 head 0.7171.

## Experiments, including failures
One seed for every training run, so the ranking among arms is not shown to exceed noise.
- **E4 (joint training on real and lab photos, lab capped at 60 per class):** `field_dev` accuracy 0.7644, macro-F1 0.7653, ECE 0.0794, against E1's 0.224. A DINOv2 head scored macro-F1 0.7469; the interval on the difference spans zero (-0.0537 to 0.015). The ConvNeXt won by the pre-set rule, not by a proven margin. The DINOv2 fine-tune was not run, by rule.
- **Stronger augmentation and pasted backgrounds:** no measurable difference (accuracy -0.0087 and 0), not adopted. The pasted backgrounds are garish, a design weakness.
- **E10 segmentation and the disease crop:** mean IoU 0.5432 against 0.2209 for the whole image. Using the crop in E4 made it significantly worse (macro-F1 0.7359 against 0.7653). The crop was tested and left out of serving.
- **E11, follow-up questions to the user:** nothing clearly better or worse (accuracy 0.7695 to 0.7636), so the feature stays out of the product. The answers came from vision agents, not growers.
- **E12 (more sources: PlantWild, Plant Pathology):** seen-source macro-F1 0.7836 against E4's 0.7712 (+0.0124, interval -0.0168 to 0.0404), so the pre-set rule said keep E4. See "E12 and the model choice".
- **E13 (label cleaning):** see "Row sets and label cleaning".
- **E15, the final model (T4.9):** ConvNeXt V2 Tiny, macro-F1 0.81 on the 1018 cleaned dev photos against 0.76 (E4) and 0.7904 (E12), accuracy 0.8104. Paired with E12 the macro-F1 difference is +0.0196 with an interval of -0.0082 to 0.0474, which includes zero; against E4 it is +0.05 (0.0188 to 0.0816). It won by the fixed rule (highest macro-F1), not by a measured margin over E12. The choosing rule was fixed before the V2 model had a score. E4's and E12's rescored scores had already been seen in a dry run with an untrained model. The first wording of the rule let V2 replace E4 even if E12 was higher; it was corrected before any V2 score existed. All three models were picked on `field_dev`; the test numbers are held out, one run each.
- **Test, `field_test`, provisional, the same 1006 photos for each system:** final model macro-F1 0.8111, E12 0.7682, E4 0.7524, DINOv2 head 0.7171, clean lab model 0.2387, XGBoost 0.0432. The final model's macro-F1 difference is +0.0587 (0.0299 to 0.0891) over E4, +0.0429 (0.0167 to 0.0702) over E12 and +0.5725 (0.5367 to 0.608) over the lab model. The intervals resample photos, not training runs.
- **Failures worth keeping:** the lost E1 checkpoint (a worktree cleanup deleted it, then fixed); the first dedup rule that chained unrelated images; the crop rule that over-removed extras; the E7 decline bug that hid prompt gains (below); and the lab-then-fine-tune plan, replaced by joint training after the fall, with no run of the replaced plan to compare.

## Domain shift and held-out sources
Seen-source and held-out-source numbers, `field_dev`, cleaned rows, macro-F1, side by side. A held-out model is trained with that web source removed (for E12, its near copies among the extras were counted but not removed).

| View | Rows | E4 seen-source | E4 held-out source | E12 held-out source |
|---|---|---|---|---|
| All sources | 1028 | 0.7712 | | |
| PlantDoc | 610 | 0.7398 | 0.3789 | 0.668 |
| PlantSeg | 418 | 0.7757 | 0.5182 | 0.7523 |

- **E4's joint score mostly reflects sources seen in training, and a new source scores much lower.** On the uncleaned rows E4's joint macro-F1 is 0.7653 against 0.3778 with PlantDoc held out and 0.5185 with PlantSeg held out. Both gaps are significant.
- The held-out gap stays large on photos with no look-alike, so it reflects source shift and the loss of about half the real training data, not leakage (the T4.1 verifier's reading).
- The final model has no held-out-source run. Its test score (0.8111) is a seen-source number: PlantDoc and PlantSeg photos are in `field_train`, and `field_test` is drawn from the same sources. Do not read it as a new-source or new-photo score.
- PlantDoc's white-background stock photos are mostly healthy, a style cue that will not transfer to phone photos, and PlantSeg has no healthy photos, so some within-source skill is style.
- Real photos here are web photos. No phone photos were tested. The EXIF difference is under "Image handling".

## The disease crop
Tested, and left out of serving. E10's segmentation (trained on PlantSeg only) gave mean IoU 0.5432 on `seg_dev`; as E4's crop arm it lowered macro-F1 from 0.7653 to 0.7359 and accuracy from 0.7644 to 0.7404 (McNemar p 0.037). Real-photo Grad-CAM masks also failed their check (16 of 40 sampled masks good, against a bar of all 40 judged and at least 80% good), so Grad-CAM energy outside the leaf is reported for lab photos only, and the crop arm has no such number. Serving exports the classifier only.

## Same-scene look-alikes
Records `E17-20261105T024504962312Z` and `E17-20261105T024509606999Z`, from the final model's saved `field_dev` predictions, on the CPU, with no retraining (the plan's words did not force one). Each of the 1018 cleaned dev photos was compared with the 2047 `field_train` photos of the 29 classes by DINOv2 ViT-S/14 class-token cosine, taking the higher of the photo and its mirror. 77 photos have a neighbour at 0.9 or above and 15 at 0.95 or above. Extras were not searched.
- Accuracy on the 77 photos with a close neighbour is 0.8961; on the 941 without one it is 0.8034; on all 1018 it is 0.8104. Macro-F1 is 0.9051 on the 77, 0.8065 on the 941 and 0.81 on all.
- So removing the look-alikes lowers the dev accuracy from 0.8104 to 0.8034, under one point, and the interval on the 941 (0.7774 to 0.8285) still holds the full-set value. The 77 are partly easier because of ordinary visual similarity, not only copying, so this is an upper bound on the leak effect. The T4.1 verifier found a similar size for E4 (0.7644 to 0.7552).
- Not measured: the same search for `field_test`, and the E12 gate G1 retrain (in `docs/backlog.md`), which would show how much of E12's held-out gain survives removing PlantDoc's near copies.

## E12 and the model choice
- The pre-set rule said keep E4 (seen-source gain +0.0124, interval -0.0168 to 0.0404). The held-out gains are far larger: PlantDoc held out 0.3789 to 0.668 (+0.2891, 0.2451 to 0.3299), PlantSeg held out 0.5182 to 0.7523 (+0.2341, 0.18 to 0.2927).
- The person chose E12's model despite the rule, with G1 and G2 skipped for now and G3 still deciding. This overrode the pre-set rule after dev numbers were seen. The held-out gain rests on one run and one seed; its intervals resample photos, not training noise. The look-alike retrain (G1) and the second seed (G2) were not run; both are in `docs/backlog.md` under "Try later".
- **G3 failed:** at each model's fitted thresholds E12 confidently called 2 sick photos healthy and E4 called none (two PlantDoc tomato yellow leaf curl virus photos called healthy at 0.886 and 0.922). That is thin evidence, and it is the harm the gate was written to catch. E4 stayed as the serving model until T4.9 replaced both.
- The verifier's dissent: it read the evidence as keep E4. The seen gain is noise-sized, PlantDoc's held-out gain is mostly healthy leaves, and it shows web-to-web transfer, not phone photos. The healthy and sick split goes wherever the gain is quoted: E12 lowered Apple healthy precision from 0.889 to 0.633, which fits a style shortcut from Plant Pathology photos (scab from one orchard style, and no healthy ones).
- No phone claim, and no number here is called new-photo performance.
- T4.9 then replaced the question: the final model was picked from all three by the highest macro-F1, so the E12 choice stands as history.

## Confidence, abstention and calibration
- **Fit (E5, refit by T4.9 for the final model):** temperature 1.3006 and an abstention threshold of 0.929437, fitted on one half of `field_dev` (by group, seed 0) and scored on the other, so the settings are not scored on the rows that chose them. They are bound to the checkpoint by its sha256, and the identifier refuses a mismatch (phase 4 review D1).
- **Test, provisional:** at the served thresholds the final model answers 50.4% of `field_test` photos, with accuracy 0.9507 on those against 0.8131 overall. The share of answered photos that are wrong is 4.93%. ECE falls from 0.0752 to 0.0434 with temperature scaling.
- **Healthy is stricter.** A photo whose top label ends in `___healthy` must clear 0.964719, halfway from the abstention threshold to 1. The cost, as unsure answers: on the dev scoring part of 517 photos, 259 are unsure with abstention alone and 276 with the healthy bar; on `field_test` the answered share is 0.5318 with abstention alone and 0.504 with the healthy bar. At the fitted thresholds there were 0 confident sick-called-healthy answers on the dev scoring part. The result page says that a healthy label does not rule out disease, at any confidence.
- **Quantized model:** the thresholds fitted on the PyTorch model hold on the dynamic int8 file: on the 517 dev rows both answer 241 with 0 sick called healthy (confident-wrong 5 against 6). On `field_test` the quantized copy answers 0.4771 at the served thresholds.

## Gate
The photo gate T4.9 chose is `confidence_ood`: top probability at least 0.15453240806843455 and normalized entropy at most 0.5993013217342331. On E4's model in E5 it let through 0.0166 of COCO objects on the scoring half of `not_plant_dev`.
- **False rejection of real plant photos, from existing data, no tuning (record `E16-20261105T024129550839Z`):** T4.9's gate settings applied unchanged to the final model's saved `field_dev` probabilities and the saved green ratios refused 37 of 1018 cleaned plant photos, 3.63%, and 19 of 517 (3.68%) on the half its settings were not fitted on. For the other two gates in the record, the green-pixel rule refused 34 and the confidence-only gate 38.
- A real Apple scab photo, PlantDoc's `12-20_apple_scab1.jpg`, was refused as not a plant in E8. It is a `field_dev` photo and it is among the 37 refused here, so that refusal is the gate behaving as measured. Tomato bacterial spot (6 of 43), tomato mosaic virus (4 of 23) and soybean healthy (3 of 15) have the most refusals.
- Development numbers; the gate and the model were chosen on `field_dev`. How often a real phone photo is refused is unknown.

## Plant-given and top-3 numbers
Plant-given top-3 is easy: the 29 classes cover 14 plants, about two classes per plant, and eight plants have a single class. The options per plant, as the records state them: Apple 4, Blueberry 1, Cherry 1, Corn 3, Grape 2, Orange 1, Peach 1, Pepper 2, Potato 2, Raspberry 1, Soybean 1, Squash 1, Strawberry 1, Tomato 8. Wherever plant-given top-3 appears (1.0 on `lab_test`, 0.9821 on `field_test`, provisional), read it with these numbers. Plain top-3 accuracy over all 29 classes (0.9642 on `field_test`) is the figure the bullets use.

## Secondary numbers
The headline is all of each split. PlantDoc and PlantSeg are reported separately as secondary numbers (final model, `field_dev`, cleaned, development numbers): PlantDoc 610 photos, accuracy 0.8016, macro-F1 0.7863; PlantSeg 408 photos, accuracy 0.8235, macro-F1 0.8023 (records `E17-20261105T024510557987Z` and `E17-20261105T024511349975Z`). Per-source numbers for the test split were not computed. There is no leaf-only subset: `docs/field-dedup.md` records why (the class names exclude nothing, and the PlantSeg metadata has no plant-part field). E2's 50-mistake sample found fruit photos and doubtful labels; E1's `field_dev` accuracy is 0.2294 on PlantDoc and 0.2162 on PlantSeg, close, so fruit photos do not look like the main driver of the fall.

## Thin classes and coverage gaps
- **Thin classes:** after the class rule every kept class has at least 10 photos in each of `field_dev` and `field_test`, but ten photos is a thin test. In `field_dev` Tomato healthy has 13 photos, Cherry healthy 14, Pepper healthy 14, Soybean healthy 15, Apple black rot 15 and Grape healthy 17. A per-class score on 10 to 20 photos has a wide interval, so read per-class numbers as thin. The class report is `docs/class-report.md`.
- **Seven lab classes have no real-photo row** (Corn healthy, Potato healthy, Grape Esca, Grape leaf blight, Peach bacterial spot, Tomato spider mites, Tomato target spot), so a real photo of one of them is unscored, and the final model cannot name them.
- **Document coverage (`docs/document-coverage.md`):** a document describes the symptoms of every disease class the older 38-class model could output (26 yes, none no). The 12 healthy classes are marked not applicable, so no document says what a healthy plant looks like or that a leaf can look healthy and still be sick. A disease named only in a spray table does not count. Eleven of the original manifest documents carry "source not confirmed" (`docs/document-sources.md`), and the crop of a mixed page (for example tomato and pepper bacterial spot) is its leading crop only, so a pepper question can miss a tomato-tagged page; that was not measured.

## Unsupported plants and diseases
Never measured. E5's only negatives are COCO objects, so a confident diagnosis for a photo of a plant or disease the model does not support is untested. Thousands of PlantSeg and extra-source photos of excluded diseases exist but are in no evaluation. The most likely real failure, a diseased leaf of a plant outside the 14, is unmeasured, and the gate's COCO rejection rates say nothing about it. The final model cannot name a dropped class, so such a photo would be called one of the 29 or left unsure.

## Image handling
Every recorded number was computed on photos opened without EXIF orientation. Serving applies it. Of the 1006 scored `field_test` rows, 207 carry an orientation tag and 14 carry one that serving would rotate. The effect of the difference is not measured. Phone photos usually carry the tag and the training data nearly never does, so the effect could be larger on phone photos than on these web photos.

## Retrieval and answers
- **E7 (101 `rag_dev` questions, 60 answerable):** hybrid search plus the cross-encoder reranker was kept over dense search: recall@5 0.833 against 0.717, MRR 0.712 against 0.543. The MRR interval (0.084 to 0.262) is what keeps it; the recall@5 interval touches zero. Hybrid without reranking was no better than dense, so the reranker, not the hybrid step, did the work. The selection rules were revised once after the first pass, so the E7 numbers are not pre-registered.
- Structure-aware chunking, the context header, shared-crop pages, and queries built from the conversation or the top three diseases were each tested and dropped.
- **Floors:** per-mode floors picked on `rag_dev`, 0.73875 for hybrid with reranking. On the serving path (ONNX reranker) the scores differ from the torch scores by about 1e-6 at most and no passage crosses the floor, so the floors hold on the ONNX path (`results/e8/floor_check.json`, retrieval only; the answer rates were not rerun on it).
- **Fallback:** a generic follow-up such as "What is wrong with my plant and how do I treat it?" found no passage in every real try, so `answer()` searches again with the diagnosed disease when the question finds nothing. Measured on dev with a simulated identification and no writer: of 95 generic follow-ups, a passage named the right disease in 70, a floor because the text rule misses aliases (Northern leaf blight got 0 of 5 from that gap). Unsure photos and healthy labels get no second search.
- **On `rag_test` (87 questions, provisional):** it answers 91.5% of the answerable questions and declines 100% of the questions with no document, 100% of the wrong-plant questions and 30.8% of the ambiguous ones. 27 of the 87 questions are generic follow-ups, 9 of them answered, and the fallback fired on 4. The claim-level faithfulness figures mostly do not cover generic follow-ups: 53 of the 273 claims judged come from them.

## Answer judging
- **E14:** the local judge was checked against Sonnet on 1228 claims: agreement 0.874, kappa 0.63 (0.576 to 0.684); on served answers 0.901 and 0.653. The local judge was the kinder grader, and it is no longer used.
- **Haiku check (E14H):** Haiku was not clearly worse than the local judge at matching Sonnet, so it is the first judge. Its supported share (0.665) came out below Sonnet's (0.705) on the 200-claim check. Haiku and Sonnet come from one family, so their agreement may flatter Haiku. Sonnet is a reference, not truth.
- **Two judges on `rag_test` (273 claims, same claims and rubric, provisional):** the Haiku judge supports 78.8% and the Sonnet judge 81.7%, so Haiku's share came out below Sonnet's. Neither figure is accuracy, an upper figure or a conservative one, and neither is the share of correct claims. The shares are not claims in `results/claims.json`: a judged claim needs the person's review of a random sample, which has not happened.
- **Wrong-disease check: not validated, and partial.** A Sonnet grader was run on `rag_dev` (54 answers, 1 flagged), frozen, and run once on `rag_test`, with no blind read, so how often it flags a good answer or passes a wrong one is not known. On `rag_test` it graded 51 of 52 answers that have claims and flagged 1. One request was refused by Sonnet's safeguard (request id `req_011Cgirq4g3x6XmNV3EEmW3i`, answer `supported-051`) and is counted in neither direction. No rewording and no other model was used.
- The `rag_test` run stopped once at that refusal and was resumed with changed code; the record `E9-20261105T021656097995Z` tells how and supersedes `E9-20261105T011744079881Z`, which said "one run" without the stop and stays as history.
- The `rag_dev` answers behind the wrong-disease check were written before the fallback existed, so the dev check does not reflect the retrieval served now. The judge's prompt shows no question or diagnosis, so a wrong-disease transfer cannot be caught by swapping judges.

## Safety behavior of answers
- A model reply that begins `NEED:` unprompted used to return advice with no citation and no label note (phase 6 review, defect 1). T7.4 fixed it: a clarifying reply carries no advice (`test_a_clarifying_reply_cannot_carry_advice`).
- A citation like `[1, 2]` is read as a citation of both passages; ranges such as `[1-2]` are not read and fall on the safe side as no source.
- A healthy diagnosis always carries the sentence that a healthy result does not rule out disease, whatever the writer wrote.
- Treatment advice must cite a document; chemical or disposal guidance must say to check the product label and local rules, and care advice with no supporting document is labeled as such.
- Serving matches the chosen setup: `tests/test_serving.py` checks `serving.py` against `results/e7/v2/chosen.json`.
- E8's record keeps no answer text. The E9 `rag_test` record keeps the answer text, its citations, the query that produced the passages and whether the fallback fired.

## Latency and serving
- **E6 on the final model** (CPU: AMD Ryzen AI 9 HX 370, 4 threads, 24 logical CPUs, 100 photos after 10 warm-ups; record `E6-20261104T223942238389Z`, the weights-only baseline; `E6-20261104T222414977796Z` is history). The machine was not fully idle: CPU load before each timing was 5% (PyTorch), 17% (dynamic) and 12% (static), so the latencies are indicative. They time one photo at a time from resize to probabilities with the photo already decoded, so image decoding is not in them.
- **Quantization made the file smaller with accuracy held, and it is slower on CPU than PyTorch.** The PyTorch weights alone are 111627887 bytes and the dynamic int8 file 28475011, roughly a quarter of the size. On `field_dev` macro-F1 is 0.81 for PyTorch and 0.8082 quantized (3.14% of top predictions changed). On the CPU the PyTorch p50 is 97.4 ms (p95 126.9) and the dynamic int8 file's p50 is 198.5 ms (p95 229.1). On `field_test` the quantized copy scores macro-F1 0.8042 against 0.8111; the difference is -0.0069 and its interval (-0.0213 to 0.0074) includes zero.
- **Static quantization was tested and is unusable:** it is fast (p50 44.3 ms) but macro-F1 falls to 0.455 and it answers nothing at the thresholds. Only one static configuration was tried.
- **The unquantized ONNX file's latency is unmeasured**, by direction, so "ONNX is slower than PyTorch" is not established, only that the dynamic file is.
- **E8 proves 5 requests only.** One at a time through the deployed container: 2 answered, 2 had no document and 1 was refused as not a plant. Stage times from 5 requests mean nothing as percentiles. It supports no end-to-end speed claim and no concurrency claim. The container's memory peaked at 947 MB. The same laptop ran the writer in Ollama on its GPU. Retrieval and generation times come from the Server-Timing header, not the client's clock.
- The app was containerized, not deployed. H3 (Google Cloud) is not done, so nothing here says deployed.

## Limits
- One seed for every training run (E4, E12, the final model). Arm differences of 0.01 to 0.02 are not shown to exceed noise.
- Web photos only; phone photos untested. No new-source score for the final model.
- Photo labels are unsigned; every test number is provisional, and `labels_reviewed` stays false.
- Reviewers saw `field_test` labels while cleaning, and only 400 `field_train` photos were sample-reviewed.
- The extras' survivors are the photos least like dev and test (the crop rule over-removed). Plant Pathology scab rows come from one orchard style and have no healthy photos, so E12's Apple healthy fall is consistent with a style shortcut and should not be read as general.
- Phase 4 review items: the `field_dev` look-alike effect is measured and `field_test`'s is not; E4 and E5 rows differ from the cleaned basis, and this report uses the cleaned basis; E4 and E12 are one seed each; weights trained on PlantDoc, PlantWild or Plant Pathology need the person's check (below).
- Phase 6 review items: the judge's prompt shows no question or diagnosis; the floors were tuned on torch scores and the ONNX comparison is retrieval only.
- The generic-follow-up fallback depends on a text rule that misses aliases.
- Dev numbers are optimistic: models, thresholds and floors were picked on dev.
- The run-record checker guards against mistakes and casual tampering, not someone determined to rewrite git history.

## Weights and publishing
**Weights trained on PlantDoc images exist (E4, E12 and the final model), and so do weights trained on PlantWild, Plant Pathology 2021 and PlantSeg. The person must check before publishing any of them.** PlantDoc: the person approved local use and asked to check before publishing weights. PlantWild is CC BY-NC-ND 4.0 with a no-derivatives term. Plant Pathology 2021 came from a third-party mirror, and the authors' terms are unconfirmed. PlantSeg is CC BY-NC 4.0. The ConvNeXt V2 weights are `cc-by-nc-4.0` on their model card and the original ConvNeXt-Tiny weights have no license a primary source states. The serving reranker upload `Xenova/ms-marco-MiniLM-L-6-v2` has no license tag. Nothing has been published or pushed.

## Failure examples
Described in words, with no COCO or PlantDoc image embedded.
- A PlantDoc apple scab photo with almost no green pixels was refused as not a plant in E8 and by the gate in the dev measurement.
- Two PlantDoc tomato yellow leaf curl virus photos were called healthy at 0.886 and 0.922 by the E12 model (G3).
- The E2 sample: a leaf photographed against soil or wood, a hand holding a leaf, a fruit photo filed under a leaf disease, and two diseases with similar spots on one plant.
- E12 lowered Apple healthy precision from 0.889 to 0.633, so more sick apple photos were called healthy.
- A generic follow-up, "What is wrong with my plant and how do I treat it?", found no passage until the fallback added the diagnosed disease name; Northern leaf blight found no right-disease passage in any of 5 wordings.
- In the E8 check, one photo identified as `unsure` and one answered corn rust photo each got no document.

## How the numbers are checked
- `tools/check_report_numbers.py` fails when a number in this report or the bullets is not in a run record, when a placeholder remains, or when `results/claims.json` is empty. It guards against mistakes and casual tampering, not someone determined to rewrite git history. It does not decide that a cited record is the right one.
- `python -m sprout.eval.claims check` re-verifies each claim against its record. Every claim is a provisional claim on an E9 test split whose labels are unsigned; `claims check --final` refuses them all, and H4 should run it after the person signs.
- README corrections for #90 to #93: the old claims in the README were replaced by this report and the bullets in `docs/RESUME-TARGETS.md`.
- Photo labels stay `labels_reviewed: false` until the person signs. Signing changes no record and needs no rerun.
