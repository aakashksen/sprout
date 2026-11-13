# Testing and evaluation

## Layers

| Layer | Runs | Needs | Who |
|---|---|---|---|
| **Unit and integration** | Every commit, `.venv/Scripts/python -m pytest -q` | Nothing: no GPU, network, or spending. Uses fakes and tiny fixtures | Implementer |
| **Acceptance** | `taskctl verify <id>` | Whatever the task lists | The gate |
| **Independent verification** | After each task | The task text and code, no build notes | `verifier` |
| **Phase review** | End of each phase | The whole phase together | `reviewer` |
| **Real runs** | `pytest -m real`, and experiments | GPU, Ollama, or Docker | Named tasks only |
| **CI** | Push, once the person publishes | The first layer | GitHub Actions |

## Data sets, and what each is for

| Name | Source | Use |
|---|---|---|
| `lab_train`, `lab_dev`, `lab_test` | PlantVillage, grouped by near-duplicate | Training, choosing, final lab score |
| `field_train` | Real photos | Training together with capped lab photos, in E4 only |
| `field_dev` | Real photos | Choosing, including abstention thresholds |
| `field_test` | Real photos | Final real-world score, once |
| `not_plant` | Public non-plant images | Rejection rate |

Real-photo results are only scored on classes both datasets share, and the list is reported. The model keeps all its outputs, as it would in real use, so choosing a class outside the list counts as wrong. Masking those outputs is a secondary number, labeled as such. Whenever lab and real-photo numbers are compared, the lab number is also scored on shared classes. Classes with under 10 test images are shown but not called out individually.

Choose settings on dev. Use test once, on the chosen configuration. If test results influence a decision, record that and draw a new split. `lab` and `field` numbers never get merged or blurred together.

## Metrics, each defined once in `sprout/eval/metrics.py`

| Metric | Definition |
|---|---|
| **Accuracy, macro-F1** | Macro-F1 is the unweighted mean over classes present in the evaluation set |
| **Top-1, top-3, top-5** | Share of images whose true class is among the k highest outputs (ties go to the lower index, so top-1 is accuracy). Every image evaluation reports all three, with macro-F1 and ECE. Names: `top_1_accuracy`, `top_3_accuracy`, `top_5_accuracy` |
| **Plant-given top-1 and top-3** | Outputs restricted to the classes of the true plant (the part of the class name before `___`), across all model outputs. Plants average 2.7 diseases, so this is easy: a plant with three or fewer classes scores 1.0 at top-3 by construction. Any report showing it states the options per plant and which plants are trivial. A record that has it states `config.options_per_plant` (`write_run` fills it in) and `check_run` recomputes it. Undefined, not zero, when the true label has no plant (a `not_plant` image). Names: `plant_given_top_1`, `plant_given_top_3` |
| **Per-class precision, recall, F1** | Saved for every evaluation, with a confusion matrix |
| **ECE** | Expected calibration error, 15 equal-width bins |
| **Confident error rate** | Share of answered predictions that are wrong; report it with coverage (share not abstained) |
| **Selective accuracy** | Accuracy on answered predictions, at the chosen threshold |
| **Rejection rates** | Valid plant photos wrongly rejected, and non-plant images wrongly accepted. Rejected photos stay in end-to-end results. |
| **Latency** | p50 and p95 ms per image, on CPU, stating the machine |
| **Recall@k, MRR** | RAG: gold document or page within the top k |
| **Answer faithfulness** | Share of answer claims supported by the cited passage, judged by a rubric prompt on a different model from the writer, with a random sample reviewed by the person |
| **Abstention accuracy** | RAG: declines when the docs can't answer, answers when they can. Report both sides |
| **Decline rates** | RAG, from `answered` rows: `answers_when_answerable` (gold `answer`), `declines_when_unanswerable` (`abstain`), `declines_when_crop_mismatch` (`crop_mismatch`), `declines_when_ambiguous` (`clarify`). The subset `answerable` scores retrieval (recall@k, MRR) on the questions that have a gold page |

## Comparing systems
Run every system on the same images with the same seed. Report a 95% bootstrap confidence interval (10,000 resamples, by near-duplicate group where groups exist) and an exact McNemar test for paired correct/incorrect outcomes. Don't call anything an improvement if the interval for the difference includes zero. Say "no measurable difference".

## Run records
Every experiment writes `results/runs/<experiment>-<timestamp>.json`:
`run_id, experiment, git_commit, dirty, dataset_hash, split, subset, config, models, n, metrics, measurements, judged, intervals, bootstrap, per_class, confusion, comparisons, attempts, claude_calls, labels_reviewed, predictions, started_at, finished_at, notes`.
One record per split. `predictions` maps each system name to its saved predictions file, so one record covers every system scored on the same images. `bootstrap` records the resample count (10,000) and seed. `comparisons` lists paired system comparisons (`a, b, metric, diff, ci, mcnemar_p`) on accuracy, macro-F1 or recall@k, resampled over the same near-duplicate groups; `check_run` recomputes them, and also intervals on accuracy, macro-F1, recall@k, MRR and coverage. `write_run` takes `subset`, `comparisons`, `measurements` and `judged` (raw samples) as extra keyword arguments. Classification records name their classes in `config.class_names`, and `write_run` saves each system's per-class precision, recall, F1 and support (`per_class`) and confusion matrix (`confusion`); `check_run` recomputes both. Prediction rows hold paths or ids and numbers, never image data: `path, label, probs` for classifiers, `path, accepted` for an input gate (`plant_rejection_rate` and `not_plant_acceptance_rate`, each undefined when its denominator is empty), `id, rank, answered` for RAG (`answers_when_answerable` against gold `answer`, `declines_when_unanswerable` against gold `abstain`; comparisons on `recall@k` and `mrr`, McNemar on first-rank hits for `mrr`), and `path` for perf. `changed_vs@<system>` is the share of predictions that differ from another system of the same record. `metrics` must recompute from saved predictions. `measurements` (latency, memory, size, training time) keep their raw samples in a saved file. `judged` values (faithfulness) keep every per-claim judgment in a saved file. `subset` names a declared part of a split, such as `shared_classes`, `first_100` or `source:<name>` (field rows whose `source` column in `field.csv` is that name), and how to rebuild its IDs. `attempts` lists every setting tried, including failures; a single run lists itself.

`check_run` fails when: a model name matches fake or stub; `dirty` is true; `dataset_hash` doesn't match the manifest; `git_commit` isn't an ancestor of HEAD; `n` differs from the size of the split or declared subset, or is below `--min-n`; `attempts` is empty; a saved metric doesn't recompute from predictions, or a measurement or judged value doesn't match its saved file; or, for a test split, a second record exists without an override note (`config.override_note`). Every record that has ever existed under `results/runs/` counts, in the working tree or in git history, deleted or not, and the earlier use is the one git added first. `override_note` must be a non-empty string. A record's file name must equal its `run_id`, and `finished_at` must match the timestamp in `run_id`. Every system with probabilities on a labeled image split must report `accuracy`, `macro_f1`, `ece`, `top_1_accuracy`, `top_3_accuracy`, `top_5_accuracy`, `plant_given_top_1` and `plant_given_top_3` (use `standard_image_metrics`); a missing one fails. Exempt: experiments E1, E2 and E3, named explicitly because their records predate the rule or were running without it (`results/topk-baselines.json` covers them); `not_plant` splits, which have no class labels; and systems without probabilities, such as a gate's accept/reject rows. Segmentation, RAG and perf splits are not image classification. A record with no scored rows, or a metric that cannot be recomputed, fails. `end_to_end_accuracy` counts a rejected photo as wrong (recomputed from `accepted` and `probs`). A declared subset (such as `shared_classes`) made from the same commit, models, config and saved predictions as a full record on that split is a view of that one use, not a second.

## Provisional results
Dev experiments may run before the person reviews labels (H2), with `labels_reviewed: false`. They guide choices but aren't claims. The final run (E9) needs H1 and H2. If H2 finds wrong labels, fix them and rerun what they affect.

## Experiments

| ID | Question | Task |
|---|---|---|
| E1 | With leakage fixed and augmentation actually on, what's the honest lab score? How does the old checkpoint do? | T3.1 |
| E2 | How far does it fall on real photos, and what is Grad-CAM showing? | T3.2 |
| E3 | How do XGBoost, a frozen head, and DINOv2 compare? | T3.3 |
| E10 | How well does a segmentation model trained on PlantSeg's masks find the disease area? | T4.0 |
| E4 | What actually raises real-photo performance: joint training, augmentation, background, or a disease crop? | T4.1 |
| E11 | Do follow-up questions on top-3, answered from the photo by someone blind to its class, improve the answer? | T4.3 |
| E12 | Does adding more real-photo sources to training help, on seen and held-out sources? | T4.6 |
| E13 | What changes on `field_dev` once clearly bad photos and labels are removed? | T4.7 |
| E14 | Does Sonnet agree with the local judge on `rag_dev` answers, and does the local judge flatter results? | T6.6 |
| E5 | Does calibration plus abstention cut confident errors, and at what coverage? Which input gate works best? | T4.2 |
| E6 | Does ONNX quantization keep accuracy while speeding up CPU? | T5.1 |
| E7 | Which retrieval setup and prompts give faithful, cited answers? | T6.4 |
| E8 | What's the end-to-end latency in the container under load? | T7.4 |
| E9 | Final scores on untouched test sets | T8.1 |

## Top-k baselines for existing records
`results/topk-baselines.json` holds top-1, top-3, top-5 and plant-given scores for every E1, E2 and E3 record, recomputed from their saved predictions without editing a record. It names each run id and is regenerated, identically for the same records, with:

```
PYTHONPATH=src .venv/Scripts/python -m sprout.eval.topk_baselines
```

Run it again after E3 lands. Records on `shared_classes` keep all 38 outputs; plant-given restricts to the true plant's classes across all 38.
