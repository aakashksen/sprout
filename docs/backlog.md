# Original issue list

Short index of the 93-item audit the plan refers to as #N. "Skipped" items aren't in the plan.

**Training (1-13):** 1 shared-transform bug; 2 permanent manifests; 3 untouched test set; 4 duplicates before splitting; 5 checkpoint provenance; 6 explicit fresh vs resume; 7 complete resume checkpoints; 8 class map with checkpoint; 9 unified checkpoint names; 10 best vs latest; 11 reproducible config; 12 CPU/GPU handling; 13 scripts instead of notebook.

**Model results (14-23):** 14 reconcile dataset counts; 15 one results record; 16 per-class metrics; 17 macro-F1 and confusion matrix; 18 baselines; 19 limited tuning; 20 real-world photos; 21 error analysis; 22 confidence and abstention; 23 input rejection rates.

**Prediction behavior (24-31):** 24 green-pixel gate; 25 uncertainty; 26 concise top-k; 27 prediction vs user selection; 28 label/confidence mismatch; 29 refresh overwrites selection; 30 consistent preprocessing and orientation; 31 photo guidance.

**Ingestion and retrieval (32-45):** 32 crop metadata bug; 33 document manifest; 34 Markdown ingestion; 35 no filename-only chunks; 36 extraction quality; 37 sources through generation; 38 metadata in retrieval; 39 separate search query; 40 tune chunking and thresholds; 41 per-passage filter; 42 explicit embedding and metric; 43 evaluate hybrid and rerank; 44 safe index rebuild; 45 remove unused dependencies.

**Answers (46-57):** 46 visible retrieval failure; 47 source links; 48 structured roles; 49 bounded history; 50 diagnosis uncertainty in chat; 51 ask for missing details; 52 review hardcoded recommendations; 53 fixed evaluation set; 54 prompt comparison; 55 retrieval vs generation errors; 56 skipped, since the 80% figure was an estimate and there's no experiment to revisit; 57 correct model name.

**App and security (58-68):** 58 session secret; 59 tracked runtime data, left out of this repo; 60 secrets out of Docker; 61 CSRF and cookies; 62 chat rendering; 63 image validation; 64 upload access and retention; 65 payload validation; 66 limits; 67 redirect validation; 68 debug output and errors.

**State and deployment (69-82):** 69 process-local state; 70 scoped conversations; 71 database setup; 72 persistent storage; 73 model and index provisioning; 74 missing dependencies; 75 locked dependencies; 76 startup race; 77 liveness vs readiness; 78 arbitrary startup delay; 79 timeouts and retries; 80 worker memory; 81 container permissions; 82 measured performance.

**Frontend and presentation (83-93):** 83 chat logic; 84 report button; 85 request handling; 86 deletion; 87 regression tests; 88 smoke test; 89 CI; 90 README; 91 model report; 92 demo and examples; 93 resume claims.

# Try later

Things to try if time allows. None of these is a plan task, and none blocks anything. Fixed rules and code stay in place.

- **E12 gate G1, look-alike retrain:** retrain with PlantDoc held out and its extras at DINOv2 class-token cosine 0.75 or above removed, to learn how much of the PlantDoc gain survives. Rule in `docs/DECISIONS.md`, T4.6 gates entry; code on branch `gates/E12`, `tools/extra_trim.py` and `configs/e12_gates.json`.
- **E12 gate G2, second seed:** rerun E12 and the E4 arms it's compared with on one new seed, paired, for the seen-source and both held-out runs, because the held-out gain rests on one seed. Same entry and code.

## Next: choosing the final diagnosis from the top 3 (the person, 2026-11-04)
The app currently takes the model's top guess (81% on `field_test`) and shows the others. Nothing narrows the top 3 to one answer. The person wants this fixed through the pipeline. Options to work through, not started:
1. Ask which plant it is, and drop other plants' classes. Measured on `field_test`: top-1 85.1%, top-3 98.2%. Confirm the app can take the plant as an input.
2. Ask symptom questions from the documents and re-rank what's left (`src/sprout/serve/followup.py`, 52 questions). E11 showed no gain with AI agents answering from web photos; never tested with real people. Try a gentler "no" on half the saved answers, score on the other half, and rerun on the final model.
3. Ask for a second photo, such as a close-up or the underside, and combine the predictions. No test data yet.
4. Train a specialist model for the look-alike pairs, such as tomato early and late blight. Those classes have the fewest photos.
Resume bullets drafted around this flow hold only once it's in the app.
