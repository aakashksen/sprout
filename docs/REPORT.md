# SPROUT model and evaluation report

OUTLINE ONLY. Every part below is still to be written from run records, and `tools/check_report_numbers.py` fails until each `[placeholder]` is gone. Real-photo numbers stay marked provisional until the person signs the label review. No measured number belongs in this outline: each is a `[placeholder]` until T8.2 fills it from `results/claims.json`.

## Summary
The headline, with seen-source and held-out-source numbers side by side, for the model that actually won in T4.9: [winning model, named as chosen]. The number of different photos training saw: [n_distinct_photos]. [placeholder]

## Data and sources
Licenses and standing for each source, including PlantSeg's approval and the pretrained weight licenses. [placeholder]

## Classes kept and dropped
The class rule: a class is kept only with at least 80 real training photos and at least 10 real photos each in `field_dev` and `field_test`, which leaves [n_classes_kept] classes. The 9 dropped: Grape Esca, Grape leaf blight, Peach bacterial spot, Tomato spider mites, Tomato target spot, Corn healthy, Potato healthy, Strawberry leaf scorch, Cherry powdery mildew. Scores for E4 and E12 are rescored on the kept classes by the rule fixed before any score was seen. [placeholder]

## Splits and leakage controls
Grouping, why there's no leaf-only subset (`docs/field-dedup.md`), the ORB search, that the leak test skips without raw images, and the same-scene limit of web-scraped data. [placeholder]

## Row sets and label cleaning
The cleaned-row basis, named for every model, how many `field_test` rows were removed and why, and that reviewers saw `field_test` labels during cleaning. [placeholder]

## Baselines
E1, E2 and E3. [placeholder]

## Experiments, including failures
Every experiment, with one seed each for E4 and E12 stated where that's so. [placeholder]

## Domain shift and held-out sources
Real photos here are web photos. No phone photos were tested. E4's held-out scores beside the joint score: the joint score mostly reflects sources seen in training, and a new source scores much lower. [placeholder]

## The disease crop
Tested, and left out of serving. [placeholder]

## Same-scene look-alikes
The effect of same-scene `field_train` look-alikes on the final model on `field_dev`, remeasured in a record. [placeholder]

## E12 and the model choice
The choice per its gates entry in `docs/DECISIONS.md`, the override of the pre-set rule if the gates pass, the verifier's dissent, the healthy and sick split wherever the gain is quoted, and no phone claim. E12's held-out gain rests on one run and one seed. [placeholder]

## Confidence, abstention and calibration
E5, the binding between thresholds and model, and whether the thresholds hold on the quantized model. [placeholder]

## Plant-given and top-3 numbers
Why plant-given top-3 is easy, with the options per plant stated wherever it appears. [placeholder]

## Secondary numbers
PlantDoc and PlantSeg reported separately, with the headline on all of each split, plus E2's sample finding on fruit photos and doubtful labels. [placeholder]

## Thin classes and coverage gaps
Thin classes reported as thin, the lab classes with no real-photo row, and the gaps in `docs/document-coverage.md`. [placeholder]

## Unsupported plants and diseases
Never measured: E5's only negatives are COCO objects, so a confident diagnosis on an unsupported plant is untested. [placeholder]

## Image handling
The EXIF difference between evaluation and serving. [placeholder]

## Retrieval and answers
E7, and whether its floors hold on the ONNX path. [placeholder]

## Answer judging
E14, the Haiku check and its limits, and the two judges on `rag_test`, Haiku and Sonnet by name, neither called upper, conservative or accuracy. The wrong-disease check, run on `rag_dev`, prompt frozen, run once on `rag_test`: stated as not validated, with no blind read. [placeholder]

## Safety behavior of answers
The unprompted clarify reply, multi-citation handling, and the serving-match test. [placeholder]

## Latency and serving
E6 and E8, stating the CPU, whether it was quiet, and any stage that wasn't timed end to end. E8 supports no end-to-end speed claim and no concurrency claim. [placeholder]

## Limits
Each item under "Items for T8.2" in `orchestration/verdicts/phase-4.md` and those in `orchestration/verdicts/phase-6.md`, where not already stated above. [placeholder]

## Weights and publishing
The person must check before publishing any weights: the PlantDoc, PlantWild, Plant Pathology and PlantSeg terms, the ConvNeXt weight license no primary source states, and the reranker upload with no license tag. [placeholder]

## Failure examples
Described in words, with no COCO or PlantDoc images embedded. [placeholder]

## How the numbers are checked
What the run-record checker guards against (mistakes and casual tampering, not someone determined to rewrite git history), the label sign-off state (photo labels stay `labels_reviewed: false` until the person signs), and the README corrections for #90 to #93. [placeholder]
