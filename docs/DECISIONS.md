# Decisions

One entry per real choice, including failed attempts and rejected ideas. Each names its task. Numbers come only from run records in `results/runs/` (quoted with the run ID). A number labeled "verifier's measurement" comes from a file in `orchestration/verdicts/` and is not a result. Photo counts are manifest counts, not metrics. All model scores here are development numbers on `field_dev` or `lab_dev`; none is a held-out test score, and a lab score is not real-world performance. Where the repo does not say why, the entry says "reason not recorded".

A task with no entry gets a line `<id>: no new decisions` at the end of this file. `tools/taskctl.py` checks for this before a task is marked done or landed.

---

## Data

### T0.2, T1.1 - Official 38-class PlantVillage, not the local 39-class copy (2026-11-02)
- **Chose:** the official release (CC BY-SA 3.0), pinned to commit `7f7ecc7e`, fetched into the project.
- **Considered:** the local copy (39 folders, 55,448 files). Its 39th class, `Background_without_leaves`, is not in the official repo, so its origin could not be traced, and its license could not be confirmed.
- **Evidence:** `docs/datasets.md` (both sections), `docs/data-counts.md` (the official `raw/color` holds 54,305 images).
- **Would change if:** the 39th class's origin and license were traced. The official release also has no background class, which is why `not_plant` comes from elsewhere (next entry).

### T0.2, T1.2 - COCO 2017 validation for `not_plant`; Open Images rejected (2026-11-02)
- **Chose:** COCO 2017 val, minus every image annotated `potted plant` (4,828 of 5,000 kept, from `docs/datasets.md`). Local, non-commercial evaluation only; never committed or redistributed.
- **Considered:** Open Images. Its images carry per-image licenses and Google makes no license representation, so no dataset-level license allows the use.
- **Evidence:** `docs/datasets.md`. COCO's license could not be confirmed from a primary source (the terms page returned 404). It was used because the person approved it, which is a decision, not a license finding.
- **Would change if:** a primary-source COCO license appeared, or the person withdrew the approval.

### T0.2, T1.2, T4.1 - PlantDoc: local evaluation first, then local fine-tuning (2026-11-02)
- **Chose:** the person first cleared PlantDoc for local evaluation only. Later the same day they cleared it for T4.1 fine-tuning, still local, with no redistribution (commit `84d937f`). The person must check before any weights trained on it are published. E4 records state `weights trained on it: True`, and `False` for the PlantDoc held-out run.
- **Considered:** keeping it evaluation-only. Why that was relaxed is not recorded beyond the person's decision.
- **Evidence:** `docs/datasets.md` ("Decisions by the person"), notes in `E4-20261103T143743951448Z` and `E4-20261103T151651745350Z`. The dataset license is CC BY 4.0, but per-image rights of the scraped photos were not checked.
- **Would change if:** the person decides on publishing weights; then H1 applies.

### T0.2 - PlantSeg license taken from the newest Zenodo record (2026-11-02)
- **Chose:** record 17719108, CC BY-NC 4.0 (non-commercial, fine for this research use), after commit `8d95415` corrected the license record from a direct check.
- **Considered:** the older record 16665770 (v6), which showed CC BY 4.0. The phase 0 review flagged that `Approved: yes` had been written while the newest version's license was unchecked.
- **Evidence:** `docs/datasets.md`, `orchestration/verdicts/phase-0.md`. The file has 7,774 images, not the 11,400 plus 8,000 healthy in the paper's abstract (counted after download).
- **Would change if:** any commercial use. It would need rechecking.

### T1.1 - Group by source ID plus perceptual hash, and disclose the chain group (2026-11-02)
- **Chose:** two images share a group if either their source ID matches or their pHash is within 4 bits (over flips and 90-degree rotations). The source ID is the class folder plus the code after `___`, with copy suffixes stripped. Reports say "near-duplicate-grouped", not "leak-free".
- **Considered:** source ID alone. The uuid is never shared and the code is reused across classes, so it is not a leaf ID; hashing added 2,295 merges beyond it. The distance of 4 was left as the plan sets it; other values were not compared.
- **Evidence:** `docs/data-counts.md`. Chaining is real: group `g000682` has 1,290 images in 20 classes (986 are `Soybean___healthy`), kept whole in `lab_train`. Twelve viewed images were different leaves on plain gray with similar outlines, not duplicates. Whole-group placement is safe against leakage but shrinks the dev and test shares of Soybean and Raspberry. The verifier recomputed the counts and found 0 mismatches.
- **Would change if:** a better grouping method replaced single-link merging; the lab manifest hash would then change.

### T1.2 - Duplicate rule for PlantDoc and PlantSeg: three attempts (2026-11-02)
- **Chose:** hashes only propose pairs (pHash or dHash within 12 bits). A pair merges only if a foreground correlation confirms it (score at least 0.88, or at least 0.80 with dHash at most 10). A group with more than one lab label is removed whole.
- **Failed:** (1) single-link on dHash at most 6 (commit `1f19c31`) joined unrelated images through chains of loose matches. (2) Plain correlation on gray images matched white-background stock leaves at 0.88 to 0.93, so non-white foreground pixels only are compared.
- **Evidence:** `docs/field-dedup.md`; the band table there comes from viewing 71 sampled pairs, so its error rates are hand estimates, not measurements. The 0.80 rule favors missed copies over wrong merges, because a wrong merge only moves two images together while a missed copy leaks.
- **Would change if:** a measured leak turned up in `field_dev`; see the next entry for the one measurement we have (T4.1).

### T1.2 - Reviewed links, same-name review, and a stated limit; the search ends here (2026-11-02)
- **Chose:** after the rule, 32 hash-close pairs crossed splits. Six were judged the same photo by eye and linked in `field_reviewed_links.csv`. All 25 pairs left over were viewed and listed as different. Seven repeated file names were viewed; two pairs were the same photo with conflicting labels, so four images were removed. Then the duplicate search was closed.
- **Considered:** more searching. Not done; the stated limit is that a few same-scene images could remain across splits, as with any web-scraped data.
- **Evidence:** `docs/field-dedup.md`: 4,144 images kept (`field_train` 2,069, `field_dev` 1,040, `field_test` 1,035), 219 removed. Verifier's measurement for the limit is under T4.1 below.
- **Would change if:** the final model's look-alike effect on `field_dev` is remeasured (T8.2 plans this) and is larger than expected.

### T3.2 - No leaf-only subset (2026-11-03)
- **Chose:** the headline stays all of each split. No leaf-only secondary subset.
- **Considered:** a subset defined by source class names, to drop fruit photos. Rejected because the rule excludes nothing: none of the 48 shared class-map rows names fruit or a nutrient deficiency, PlantSeg files fruit under disease names such as `apple scab`, and `Metadata.csv` has no plant-part field.
- **Evidence:** `docs/field-dedup.md`; in `E2-20261103T092315605470Z` (development numbers) E1 scores 142/619 = 0.2294 on PlantDoc rows and 91/421 = 0.2162 on PlantSeg rows. They are close, a weak signal that fruit photos are not driving the fall. Fruit photos were not counted.
- **Would change if:** plant-part labels existed, or the H2 label review found the fruit photos matter.

### T3.4 - Thin classes are reported, not dropped (2026-11-03)
- **Chose:** keep every class in the frozen `field_test` and mark classes under the minimum (80 train, 20 dev, 20 test real photos) as thin. Report: 27 of 38 classes thin, 7 with no real photo (`docs/class-report.md`, manifest counts).
- **Considered:** dropping thin classes from scoring. Not chosen; `field_test` is frozen and the plan says thin classes are shown as thin. Anything beyond that is reason not recorded.
- **Evidence:** `docs/class-report.md`. The phase 3 review found its crops table mislabeled (healthy-only real counts looked like totals); fixed in `222936b`.
- **Would change if:** T4.5 sources fill thin classes; the report is regenerated from manifests.

### T3.4 - Document coverage: 10 yes corrected to 13 yes (2026-11-03)
- **Chose:** coverage counts only a real symptom description found by reading the passages; a name in a spray table or comparison does not count. After correction: yes 13, no 13, not applicable 12 (the first version, `03486ed`, had 10 yes).
- **Changed rows:** Grape leaf blight (`winegrapes8.pdf` p12), Tomato bacterial spot and Tomato spider mites (`tomatodoc2.md` p1) became yes. Each remaining "no" row records its search queries. Tomato mosaic virus stays a borderline "no": one generic virus sentence, nothing specific.
- **Evidence:** `docs/document-coverage.md`, commit `35f0022`. The person's account is that verification found the false "no" rows; the rejected verdict is not kept in the repo, only the final PASS.
- **Would change if:** a document describing a "no" class is added (T6.5 regenerates the report).

### T4.4 - Candidate documents for the 13 uncovered classes; nothing downloaded or added (2026-11-03)
- **Chose:** list candidates from primary publishers (extension services, agencies) with URL, publisher, date, symptom text and the publisher's stated terms. 23 candidates in 13 sections, every one marked `Approved: no, needs the person`; nothing added to `knowledge/`.
- **Considered:** adding them directly. Held back by the dataset rule in force at the time (person approves licenses first).
- **Evidence:** `docs/document-candidates.md`. Only NDSU late blight (a Creative Commons non-commercial share-alike statement) and Iowa State Septoria (republish as written with credit) state open terms; the rest are all-rights-reserved or state none. `Tomato___Target_Spot` has no usable candidate (fetch failures recorded there).
- **Would change if:** the person's 2026-11-03 rule (public, no login or paywall, so no one-by-one approval) is applied; T6.5 is built for that.

### T4.5, T4.6, T6.5 - Stage 2 and 3 additions (2026-11-03)
- **Chose:** more real photos for training only (T4.5: PlantWild, and Plant Pathology 2021 if public), a single E12 rerun of the best E4 configuration with them (T4.6), and 200 to 400 more care documents (T6.5). `field_dev` and `field_test` stay frozen.
- **Considered:** the earlier text said stage 3 runs "only if stage 1 clearly beats the baseline, with licenses checked by the person first". The person's rule of 2026-11-03 replaced the second half: local, non-commercial, unpublished use needs no one-by-one approval for public data with no login, paywall or block; raw content is never committed.
- **Evidence:** `docs/PLAN.md` ("Later stages"), commits `b95985f`, `f150227`, `3493a8c`. Why PlantWild was picked over other sources is reason not recorded. No results exist yet.
- **Would change if:** E12 does not clearly help; the plan keeps E4's model then.

### T4.5 - Cap of 200 images per lab class, fixed before any E12 result (2026-11-03)
- **Chose:** at most 200 extra photos per lab class, kept by sha1 order, so selection is deterministic and unrelated to image content.
- **Considered:** no cap, which lets one source's healthy-leaf classes outweigh the real-photo sets; a cap near the median `field_train` class (about 60), which wastes the thin classes' only extra photos; sampling by file name, which can follow the source's own ordering.
- **Evidence:** `docs/extra-sources.md`. `field_train` classes hold at most 136 photos (median about 60), so 200 lets every thin class keep everything. after the copy checks only 45 images were cut and 1 class ended at exactly 200 (Apple healthy).
- **Would change if:** E12 shows the capped healthy classes push the model toward "healthy" on white-background leaves; a lower cap for those classes would then be tried in a new run.

### T4.5 - Close-match guard of 0.75 on top of the field-dedup rule (2026-11-03)
- **Chose:** besides the rule in `docs/field-dedup.md`, drop any extra photo with a hash-close pair (within 8 bits) to a field photo that scores at least 0.75 on `same_scene_score` (`GUARD_SCORE`). In the final build it removes nothing, because the two copy checks below run first.
- **Considered:** the rule alone, which kept 4,727 images and was missing same-photo pairs; viewing and listing every hash-close pair by hand, as T1.2 did for field, which was not worth it for training-only extras; a higher threshold (0.80), which would keep the 0.785 pair that was the same photo.
- **Evidence:** the 60 hash-close pairs between kept extras and `field_dev` or `field_test` were scored; the top 3 were viewed: two were the same photo at another size (0.83 and 0.785), one a different corn leaf. The threshold comes from those 3 pairs only. Pairs below 0.75 were not viewed one by one, so a few copies could remain.
- **Would change if:** a later check finds copies of `field_dev` or `field_test` photos below 0.75, or finds the guard dropped many non-copies; the threshold or a reviewed-pairs list would then replace it.

### T4.5 - One image kept per duplicate group among the extras (2026-11-03)
- **Chose:** within the extras, a group of exact, near or same-scene copies keeps one image (first by source and path). A group with conflicting lab labels is removed whole. In the final build 438 images were dropped as duplicates and 218 as conflicts.
- **Considered:** keeping every member of a group, as `field.csv` does so a split stays whole; that matters for splits and not for training-only rows, and crops of one photo would count several times in training. Picking a label by majority, which T1.2 also avoided.
- **Evidence:** `docs/extra-sources.md` removal table. Same-scene merging has a known false-merge rate of about 3% (`docs/field-dedup.md`), so a few distinct photos may be lost.
- **Would change if:** E12 shows classes starved by the removals.

### T4.5 - Second copy check for extras: 32x32 correlation at 0.75, deliberately over-inclusive (2026-11-03)
- **Chose:** after verification found at least 6 kept extras that were watermark or crop copies of `field_dev`/`field_test` photos, drop any mapped extra whose best 32x32 grayscale correlation (mean-centred, over the 8 rotations and flips, against every `field_dev` and `field_test` photo whatever its class) is at least 0.75 (`src/sprout/data/copy_check.py`). It removed 2,890 images (1,374 for `field_test`, 1,516 for `field_dev`), most of them exact or near copies that the other steps would also have caught. It does not see crops, zooms or captions, which the next entry handles.
- **Why over-inclusive is fine here, and why `field.csv`'s rule does not change:** losing a few extras costs little, while one leaked evaluation photo inflates E12's `field_dev` score. The field rule decides which field photos share a split (a wrong merge co-locates or removes real evaluation photos), so it is tuned against false merges and stays as in `docs/field-dedup.md`. This check only removes training extras and never touches `field.csv`.
- **Considered:** loosening the field rule, which would change `field.csv` and the frozen splits; a threshold of 0.80, the first choice (the verifier counted about 213 kept extras above it), which still left one copy at 0.798 (an apple branch with a white strip added); hand review of every pair, which is impractical for hundreds of pairs.
- **Evidence:** the 20 kept pairs just under 0.80 were viewed: 1 copy. The 20 just under 0.75 (0.747 to 0.750) were viewed: 0 copies. Pairs below 0.747 were not viewed. `tests/test_extra_sources.py`: a synthetic watermark and crop filed under another label is caught, a numpy reference matches, and the real manifest has no extra at or above 0.75 (including the six verifier pairs by name).
- **Would change if:** a later view finds copies below 0.747, or E12 shows classes starved by the drops.

### T4.5 - Crop-aware copy check with DINOv2 and patch matching; rule loosened until audits found nothing (2026-11-03)
- **Chose:** a second pass for extras that survived the full-frame correlation (`src/sprout/data/crop_copies.py`): DINOv2 small class tokens pick the 3 nearest `field_dev` and 3 nearest `field_test` photos for each extra, for the image and its mirror; mutual nearest-neighbour patch tokens plus a fixed-seed scale-and-shift RANSAC count agreeing patches. Drop if class-token cosine is at least 0.80, or at least 30 inliers, or cosine at least 0.75 with at least 20 inliers. It removed 2,173 images (1,095 for `field_test`, 1,078 for `field_dev`). Final manifest: 3,347 images. Deliberately over-inclusive for extras, for the same reason as the correlation check; `field.csv` and its rule do not change.
- **Considered:** local feature matching with OpenCV ORB or SIFT, which is not installed (the T4.2 verifier found the same) and would be a new dependency; the verifier's 80% and 60% crop correlation plus Sobel edge correlation, which the coordinator replaced with this approach and which is a pixel-grid measure; a rule tuned only to the 13 listed leaks, which would not generalise.
- **Evidence:** the rule was chosen on half of a labelled set (31 positives: 16 known copies, 11 viewed copies and 4 undecided, counted as drop; 45 viewed non-copies) and measured on the other half. The rule best on the choosing half (cosine 0.90 or 40 inliers) caught 13 of 15 held-back positives (missed one undecided pair and one potato zoom at 39 inliers), so about 13% missed, 95% upper bound about 40%, a very small sample. Fresh audits of pairs just under the rule found one probable crop each in three rounds, so the rule was loosened to the shipped one; the last 20 pairs sampled from the 1,349 extras under the final rule showed no copies (95% upper bound about 14% of that pool). The shipped rule catches all 15 held-back positives, but it was set after seeing them, so that is not an unbiased measure. The cost: it also drops 13 of 22 held-back negatives (near-threshold samples, not the drop rate overall) and cuts the three corn disease classes to 17 to 23 extras each, since DINOv2 rates corn leaf close-ups as similar.
- **Would change if:** E12 shows the corn or other classes starved by the drops (then a stricter pair-specific check could replace the cosine floor), or a later view finds copies below the rule. 90-degree rotated crops are not covered.

### T4.5 - 3 candidates per split and mirrored copies in the crop check (2026-11-03)
- **Chose:** 3 nearest candidates per evaluation split by class token, and the mirrored image as a second pass.
- **Considered:** 5 candidates, which costs about 60% more time. In the 16 known copies the true photo was always the 1st or 2nd nearest of its split, and the 24 extras that only the 4th or 5th candidate flagged at 30 or more inliers were all non-copies. Skipping the mirror, which would leave flipped crops to the full-frame correlation only.
- **Evidence:** ranks of the 16 known copies by class-token cosine (all 1st or 2nd within their split); viewed montage of the 24.
- **Would change if:** a copy is found that was not among the 3 nearest.

### T4.5 - Extra groups get their own `x` namespace (2026-11-03)
- **Chose:** group ids in `extra_train.csv` start with `x`.
- **Considered:** leaving them. In the previous build 98 ids also appeared in `field.csv`. They come from a `g` plus running number scheme that restarts in each manifest build, so they were reused numbers for different images and not shared groups, but a join in E12 could have merged them.
- **Evidence:** `test_extra_groups_have_their_own_namespace` checks a fixture and the real manifests.
- **Would change if:** E12 gives every row a global group by construction.

### T4.5 - `bell pepper leaf spot` mapped to Bacterial_spot; grape leaf spot and grapevine leafroll excluded (2026-11-03)
- **Chose:** PlantWild `bell pepper leaf spot` goes to `Pepper,_bell___Bacterial_spot`, the way PlantDoc's `Bell_pepper leaf spot` already does. `grape leaf spot` and `grapevine leafroll disease` are excluded with the reason "no lab class for this plant or disease". 150 of 204 source classes are excluded in all.
- **Considered:** excluding `bell pepper leaf spot`, which drops the class's photos although PlantDoc's matching class is already mapped; mapping `grape leaf spot` to `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`, which is a specific disease and the match is not certain.
- **Evidence:** `datasets/class_map_extra.csv`, `PLANTWILD_MAP` in `src/sprout/data/extra_sources.py`. Class names were not checked against the photos themselves.
- **Would change if:** a look at the photos shows the pepper leaf spot class is not bacterial spot; then it moves to excluded.

### T4.5 - Plant Pathology 2021 not used (2026-11-03)
- **Chose:** not used. The Kaggle competition needs login, and the only public copy is a third-party Hugging Face mirror (`timm/plant-pathology-2021`, 16 GB, apple only, license stated by the uploader).
- **Considered:** the mirror, which covers three usable apple classes (healthy, rust, scab) that PlantWild already feeds, at 16 GB and with a license that is not the authors'.
- **Evidence:** `docs/extra-sources-lookup.md` and `docs/datasets.md`.
- **Would change if:** the coordinator overrules it, or the person decides a third-party mirror's license is enough.

### T4.5 - Hashing on CPU; the two new copy checks on the GPU (2026-11-03)
- **Chose:** hashing and the field rule stay on the CPU. The new full-frame correlation and the DINOv2 crop-aware check run on the GPU while holding a share (`taskctl gpu-acquire T4.5` just before, release just after; device `cuda` recorded in the build output; the build tool waits for a share instead of using the CPU, which would take hours for DINOv2). The first build ran everything on CPU. The slow step is hashing images with PIL and imagehash (pHash over flips and rotations for the lab, field and extra images); the scene check is correlations on 64x64 arrays for a few thousand pairs.
- **Considered:** moving the field rule's scene correlation to the GPU. It would be the only step that could move, and it is small, so the gain is tiny. Moving it risks small float differences that could flip a pair at the 0.88 / 0.80 / 0.75 thresholds, and the person's instruction was to keep the field-dedup rule unchanged and move a step only if results stay identical. A GPU gain for the hashing would need a different hash implementation, which would change the rule.
- **Evidence:** the build stayed inside the memory limits. The new correlation step gave the same 2,890 drops on CPU and on the GPU on the real data, and a fixture test compares CPU with a numpy reference (and with the GPU when present). `docs/extra-sources.md` records the device.
- **Would change if:** a rebuild is needed often or the extra set grows by an order of magnitude; the correlation could then move, checked to give identical kept sets.

### T6.5 - Document set: public extension and agency pages, 237 documents, saved raw outside git (2026-11-03)
- **Chose:** 237 documents (209 added to the original 28) from university extension services and agencies, fetched one at a time with a plain client name and a 2 second pause, `robots.txt` respected, nothing run. Raw files stay in the ignored `knowledge/raw/`; only the manifest (URL, publisher, title, date) is committed. A page whose URL redirected to a different page is recorded under the URL actually saved.
- **Considered:** keeping only PDFs (lost: most extension content is HTML, and ingest now reads HTML with the standard-library parser, no new dependency); working around the five Incapsula bot-block pages (lost: the rule is never to get around a block); following PDF links from landing pages (lost: those links were not in the researcher's list).
- **Evidence:** `docs/document-sources.md` (drops with reasons, counts, reuse limits), `tests/test_more_documents.py`, commits `42edd78` and the second-round commit. Twelve first-round documents also appear in `docs/document-candidates.md` with stated reuse restrictions.
- **Would change if:** the documents are ever published (reuse terms would need checking one by one), or a host asks for removal.

### T6.5 - Crop of a multi-crop document is its leading crop, or `general` (2026-11-03)
- **Chose:** the crop mentioned most often when it leads clearly, else `general`. Mixed pages such as tomato and pepper bacterial spot carry one crop only.
- **Considered:** one manifest row per crop for the same file (lost: file names must be unique and the tests assume one row per file).
- **Evidence:** manifest `crop` column. Pepper retrieval misses a tomato-tagged mixed page; this was not measured.
- **Would change if:** retrieval for a crop is found to miss a mixed document that its questions need.

### T6.5 - Three earlier test expectations changed on purpose (2026-11-03)
- **Chose:** `test_shipped_manifest_is_complete` now asks for at least 28 files instead of exactly 28; the class report and web tests now expect raspberry, soybean and squash to map to their own crop, since they have documents.
- **Considered:** leaving the tests alone (lost: they would fail for the exact reason the task exists). No check was loosened beyond that.
- **Evidence:** commit `9821a72`.
- **Would change if:** the person wants the 28-file count kept as a regression guard.

### T6.5 - Index embedded on the GPU, built beside the current index (2026-11-03)
- **Chose:** `tools/build_index_gpu.py` embeds with `BAAI/bge-small-en-v1.5` through transformers on CUDA, holds a `taskctl` GPU share only for that stage, fails without CUDA, and writes the device to `build-info.json`. The new index is `knowledge/index-t65`; the current `knowledge/index` stays until it is renamed.
- **Considered:** the fastembed ONNX path on the GPU (lost: installed onnxruntime has no CUDA provider; swapping it risks the shared venv). Cosine between the two paths was at least 0.99999 on a 31-chunk sample.
- **Evidence:** `build-info.json` in the index folder (device `cuda`, RTX 4070 Laptop). Reranking is query-time, so it is not part of the build.
- **Would change if:** vectors from the CUDA path ever differ from the CPU path on a retrieval result.

### T6.5 - Coverage after round two: all classes describe symptoms (2026-11-03)
- **Chose:** the per-class review in `docs/document-coverage.csv` was redone by reading indexed passages; cherry powdery mildew moved from no to yes after round-two pages 201 and 204 were read. Totals: yes 26, no 0, not applicable 12.
- **Considered:** trusting the manifest disease lists (lost: T3.4 showed they overstate coverage).
- **Evidence:** `docs/document-coverage.md`; `tests/test_more_documents.py::test_every_class_has_a_coverage_row`.
- **Would change if:** a class's cited passage is found to describe a different disease.

### T6.5 - Two abstain questions answered by the new documents were replaced (2026-11-03)
- **Chose:** every `not_in_documents` question (28, both splits) was checked against `knowledge/index-t65` by looking for a chunk that names the plant, or a family term, together with the problem, then reading the hits. Two were answered, so their text was replaced and their ID, split and category kept. `results/runs/` holds no E7 or `rag_test` record, so no result shaped this.
  - `not_in_documents-008` (test): "How do I control powdery mildew on my zucchini?" became "Why are my lavender plants dying back from the roots in wet soil?". The new squash and cucurbit pages describe powdery mildew on zucchini and its relatives.
  - `not_in_documents-011` (dev): "Why are my pumpkin vines wilting with white mold?" became "Why is my hibiscus dropping its buds before they open?". The new pumpkin and winter squash pages cover powdery mildew (white growth) and bacterial wilt of pumpkin.
  - The replacements name plants that appear nowhere in the index (`lavender`, `hibiscus`). The other 26 were kept; plant names that appear only in lists or passing mentions (for example cranberry in a blueberry pest list, papaya in a host list, sunflower in a rotation note) do not answer them.
- **Considered:** dropping the two (lost: the category needs at least 20 per split pair); adding gold pages from new documents (lost: no gold document may be new). Watermelon fusarium wilt (012) was first kept on the wrong belief that the watermelon pages cover powdery mildew only; verification attempt 2 found document 268 (Clemson cucurbit diseases) describes Fusarium wilt (Fusarium oxysporum f. sp. niveum) in a section that never says watermelon, so a same-chunk check on the plant name missed it. It was replaced in the second round below.
- **Evidence:** `tests/test_more_documents.py::test_abstain_questions_have_no_answer_in_the_index` (index-dependent, skipped when the ignored index is absent) and its fast fixture test; `answer_candidates` in `src/sprout/rag/evalset.py`.
- **Would change if:** a later document round names one of these plants together with its problem; the test then fails and the question must be replaced the same way.

### T6.5 - Abstain questions rechecked with family terms; two more replaced (2026-11-03)
- **Chose:** every `not_in_documents` question was rerun against `knowledge/index-t65` with the plant's family terms as well (cucurbit, melon, squash, brassica, Apiaceae, Vaccinium and so on), reading every hit. Generic matches were narrowed (`rust`, `crown rot`, and dropping words such as "houseplant", "vegetable" and "herb" that only match menus and index pages). Two more were replaced, keeping ID, split and category; `results/runs/` still has no E7 or `rag_test` record.
  - `not_in_documents-012` (test): "How do I stop fusarium wilt in watermelon?" became "How do I treat crown rot on my hosta plants?". Document 268 answers the old one.
  - `not_in_documents-019` (dev): "How do I treat fruit rot in cranberry bogs?" became "Why are my daylily flower buds turning to mush before they open?". Cranberry shares its genus with blueberry, and the blueberry pages (019, 238) describe fruit rot, so a family link exists. This one is a precaution, not a clear answer.
  - Replacement plants (`hosta`, `daylily`, with `lavender` and `hibiscus` from before) appear nowhere in the index, and neither does a crop in their family.
- **Kept after reading:** the other 24. Family hits were menu lists (for example "Carrots", "Asparagus" in a growing-problems menu), index pages, host lists (hops downy mildew in a list of vine-crop pages; sunflower in a Sclerotinia host list) or title lists (Brassica diseases in a vegetable pathology index).
- **Fixed:** the cranberry pattern had lost its backslashes (`bbogs?b`); the test patterns are now raw strings.
- **Would change if:** a later round names any of these plants or families together with its problem; the test fails and the question is replaced the same way.

### T6.5 - Unconfirmed sources are stated, not guessed (2026-11-03)
- **Chose:** 7 of the 18 original manifest rows without a source URL were filled from `docs/original-doc-sources.csv`; 11 carry `unknown: inherited from the legacy project, source not confirmed`. A test accepts only a URL or that exact phrase on every row, and `docs/document-sources.md` states the count.
- **Considered:** leaving them blank (lost: a blank cannot be told from an oversight); guessing the likeliest publisher (lost: not allowed).
- **Evidence:** `docs/original-doc-sources.csv`, `docs/document-sources.md`.
- **Would change if:** a source for any of the 11 is found.

---

## Evaluation and records

### T2.3 - Test-once rule and its bypass rounds (2026-11-02)
- **Chose:** a held-out split may be scored once. A second record needs a non-empty `config.override_note`. The rule uses git order: the earlier use is the record git added first, every record that ever existed counts (deleted or not, any file name), and an empty record fails closed (`340b9f5`).
- **Failed:** the first versions (`462e0a3`, `340b9f5`) could be bypassed. The verifier's scripts tried moving the first record away and back, and deleting and re-adding it after a claim. Each bypass got a test (`test_deleted_earlier_test_record_still_counts`, `test_claim_on_a_renamed_copy_of_a_record_fails`, `test_test_once_counts_files_with_any_name`) and `84f37bb` orders uses by first add.
- **Evidence:** `orchestration/verdicts/T2.3.json`, `tests/test_records.py`, `docs/TESTING.md`.
- **Limit, stated on purpose:** the checker guards against mistakes and casual tampering, not someone determined to rewrite git history. T8.2 must say so.
- **Would change if:** records moved to a store that cannot be rewritten.

### T3.5 - Top-3 and plant-given metrics; why plant-given top-3 is easy (2026-11-03)
- **Chose:** every image evaluation reports top-1, top-3, top-5, macro-F1, ECE, plant-given top-1 and top-3. Plant-given top-3 is never shown alone: plants average 2.7 diseases, so a plant with three or fewer classes scores 1.0 by construction, and a report must state options per plant.
- **Evidence (`E4-20261103T143743951448Z`, E1 on `field_dev` shared classes):** top-3 0.4538 against plant-given top-3 0.8538. For `convnext_joint`, 0.9327 against 0.9856.
- **Considered:** editing the E1, E2, E3 records to add the new metrics. Rejected; they are exempt by name and recomputed from saved predictions into `results/topk-baselines.json`.
- **Weak spot (verifier's note):** the exemption keys on the record's own `experiment` field, so a hand-edited field would skip the check. Treated as deliberate falsification, not a defect.
- **Would change if:** a record's experiment name is checked against its file name.

### T3.1 - ECE recomputed from saved float64 probabilities (2026-11-02)
- **Chose:** score the saved float64 probabilities (commit `145f879`).
- **Failed:** `ece` computed during training did not recompute from float32 on CPU, so `check_run` could not reproduce the saved value.
- **Evidence:** the two-line fix in `src/sprout/experiments/e1_lab.py`; `E1-20261103T092124608963Z` passes `check_run`.
- **Would change if:** metrics were computed in float64 everywhere.

### T3.2 - Non-finite loss guards (2026-11-03)
- **Chose:** training stops on a NaN or infinite loss, and `check_run` rejects a record with a non-finite value or loss history (commit `5b21f5a`).
- **Considered:** nothing recorded. The person's account is that a NaN gate failure in another project prompted this; that is not written in this repo.
- **Evidence:** tests in `tests/test_train.py` and `tests/test_records.py`; the T3.2 verifier broke the guards and 6 tests failed.
- **Would change if:** a run needs to tolerate occasional non-finite values (none does today).

### T2.1, T2.3 - Atomic writes for records and checkpoints (2026-11-03)
- **Chose:** write run records and checkpoints through a temp file and a rename, and record `last` (commit `af92f26`, `src/sprout/atomic.py`).
- **Considered:** plain writes. A crash mid-write could leave a partial file; the project rule is that a partial run never counts as a result. Detailed reason beyond the commit title is not recorded.
- **Evidence:** `tests/test_atomic_writes.py`.
- **Would change if:** records move to a database.

### Cleanup - check_run skips records that already fully passed (2026-11-03)
- **Chose:** `check_run` stores a key for each record that fully passed and skips it next time (`src/sprout/eval/check_cache.py`). The key covers the record, the prediction and sample files it names, `--min-n`, every sprout module `check_run` imports (found by walking the imports), the versions of the third-party libraries those modules import, and all manifests plus the RAG and perf lists. Any change to the checker or those data files rechecks everything. Failures are never stored. Ancestry of `git_commit` and the test-once rule are still computed on every run because they depend on HEAD and on other records. The file sits in git's common directory (untracked, shared by worktrees, written atomically). `--all` ignores it, and `claims` call `check_file`, which never uses it. T8.1 and T8.2 must run the full check.
- **Considered:** a timestamp or mtime key (files change time without changing content, and a checker edit would not show); a cache in `results/` (a new git-ignored path, and each worktree starts empty so a land would save nothing); hashing only `check_run.py` (misses `records.py`, `metrics.py` and `stats.py`, so a metric fix would leave stale passes).
- **Evidence:** the six tests in `tests/test_check_cache.py`. On the 7 experiment/split `check_run` acceptance commands that have records today, full checks took 603 s (`--all`); the same commands with a warm cache took 44 s, of which E10 (26 s) was uncached because its first cached run crashed with exit code 3221226505 and was not stored. Timings are from one busy machine, not a run record.
- **Would change if:** a check starts to depend on something not in the key (a new data file, an environment variable); then add it to the key, or drop the cache.

### Cleanup - every run record is logged to MLflow by write_run (2026-11-03)
- **Chose:** `write_run` copies each saved record into `mlruns/` (`src/sprout/eval/mlflow_log.py`): tag `run_id`, finite metrics per system, scalar config as parameters, the device when the record has one, per-epoch history when an attempt saved one. A logging failure warns and the record is untouched. E1's own logging is removed so nothing logs twice (its record is unchanged). `tools/backfill_mlflow.py` logs older records and skips ones already tagged. `is_dirty` ignores `mlruns/`.
- **Considered:** each experiment calling MLflow itself (only E1 did, about 3 runs; every new experiment would need its own code); a database store (not available in mlflow-skinny); letting a logging error stop the run (the record, not MLflow, is the source of truth).
- **Evidence:** `tests/test_mlflow_log.py`. The backfill on the main folder's store logged 16 records (E1 2, E10 1, E2 2, E3 2, E4 7, E5 2) and a second run logged none. The store held one older untagged E1 run from E1's own logging; it was left alone, so E1 shows 3 runs for 2 records.
- **Would change if:** MLflow is dropped from the resume bullets, or a tracking server replaces the local file store.

---

## Experiments

### T3.1 - E1 lab baseline; the legacy 1.0 is not performance; old figures reconciled (2026-11-02)
- **Chose:** train ConvNeXt-Tiny from ImageNet weights with separate train and validation transforms, and score on `lab_dev`. `E1-20261103T092124608963Z`: accuracy 0.9979, macro-F1 0.9966, ECE 0.0011 (n 8,152, lab photos, development numbers; `lab_dev` also picks the best epoch). The legacy checkpoint scores 1.0 on the same rows, labeled "legacy, may have seen these images".
- **Considered:** comparing legacy to E1. Rejected: the old notebook split 80/20 at random per image with no grouping and no seed, and its data folder is not proven to be the same photos, so the checkpoint may have seen `lab_dev`. The record makes no comparison.
- **Old figures:** 99.97 is the best `val_acc` in the old `training_history.json` (epoch 11 of 16; no split or dataset recorded). 97.5 is a README sentence with no run behind it. 98.5 has no source anywhere. None is reproduced, none is quoted as a result.
- **Evidence:** `docs/data-counts.md` (reconciliation), the verifier's rescoring of 237 `lab_dev` images agreed with the record (verifier's measurement).
- **Would change if:** the old split history were found.

### T3.1 - Workers and persistent workers for DataLoader (2026-11-02)
- **Chose:** `num_workers` in `TrainConfig` with `persistent_workers` when above 0 (commit `592fa2c`); E1 used 8, and from T4.0 real training uses at most 4 and tests use 0.
- **Considered:** single-process loading, the earlier default. Why it was too slow is not written down; the commit adds per-epoch time and GPU memory to the history. The cap of 4 comes from the memory rule: each Python process importing torch commits about 3 GB here and many at once nearly exhausted Windows (rule in `3493a8c`).
- **Evidence:** worker count is excluded from the resume-config check; batches repeat for a given seed and worker count.
- **Would change if:** the machine's memory or the data pipeline changed.

### T3.2 - E1 checkpoint lost, retrained as a restore (2026-11-03)
- **Chose:** retrain once with the same config and seed, as a restore, not a search. Both records stay: `E1-20261103T074728015380Z` (original) and `E1-20261103T092124608963Z` (restore).
- **What happened:** `ctl land` removed a worktree's git-ignored `models/`, deleting the first checkpoint. Fixed in `109bc46`.
- **Result:** the saved prediction files are byte-identical (`cmp`, and equal sha256 per the phase 3 review), and metrics and dataset hash are equal. The restore reproduced the first run. Downstream records (E2, E3) point at the restore.
- **Considered:** changing settings while retraining. That would be a search, not a restore.
- **Would change if:** training on another GPU or driver; identical output is not guaranteed there.

### T3.2 - E2: the fall on real photos (2026-11-03)
- **Result (development numbers, 31 shared classes, all 38 outputs kept):** E1 `lab_dev` accuracy 0.9978 (`E2-20261103T092359497734Z`, n 6,784) against `field_dev` accuracy 0.2240, macro-F1 0.199, ECE 0.4617 (`E2-20261103T092315605470Z`, n 1,040). With non-shared outputs masked: 0.2394, shown as secondary only.
- **Hand sample of 50 mistakes:** background 12, similar symptoms 15, wrong label 9, unsupported condition 9, blur 5. These describe the sample, not a proven cause (`results/error-analysis-e2.md`).
- **Considered:** declaring a cause. The verifier found errors spread over many classes and no proven cause.
- **Led to:** training on real photos (T4.1 plan change below).

### T3.2 - Masks judged unreliable for real photos (2026-11-03)
- **Chose:** report Grad-CAM energy outside the leaf for lab photos only. Real-photo masks failed the check: 16 of 40 sampled masks good, against a bar of all 40 judged and at least 80% good (`results/e2-mask-check.csv`).
- **Considered:** using the HSV-threshold masks anyway. The verifier's reading is that they drop lesions and keep soil and wood, so the energy number would be wrong.
- **Would change if:** a mask method passes the 40-image check on real photos.

### T3.3 - E3 baselines (2026-11-03)
- **Result (`E3-20261103T115245769685Z`, `field_dev`, development numbers):** accuracy XGBoost 0.0413, E1 0.2240, frozen ConvNeXt 0.2519, DINOv2 small 0.4067. On `lab_dev` (`E3-20261103T115944716117Z`) XGBoost 0.8974. So XGBoost collapses on real photos; E1 against frozen ConvNeXt is "no measurable difference" (-0.0279, interval -0.0589 to +0.0029); DINOv2 is ahead of E1 by +0.1827.
- **Decision:** add DINOv2 arms to T4.1 (commit `3c49726`), written down before any E4 run.
- **Caveats (verifier):** one seed, and one untuned head setting (lr 0.001, 30 epochs) for the two frozen systems. The fine-tuned model did not beat a frozen backbone with a linear head on real photos; that is the finding.

### T4.0 - E10 segmentation, kept as a separate arm (2026-11-03)
- **Result (`E10-20261103T135121438366Z`, `seg_dev` n 1,028):** mean IoU 0.5432 against 0.2209 for the whole-image baseline. `seg_dev` also picks the epoch, so it is slightly optimistic; there is no separate seg test split.
- **Chose:** train only on PlantSeg images outside `field_dev` and `field_test` and their near-duplicates, and use the crop only as its own arm (E4 arm 6), never by default.
- **Considered:** cropping everywhere. PlantSeg masks mark disease on PlantSeg-style photos and may not carry over to PlantDoc; no transfer was measured.
- **Verifier's measurement:** about 40% of dev images are under IoU 0.5.
- **Would change if:** the crop beat its uncropped twin (it did not; see T4.1).

### T4.1 - Plan change: joint training instead of lab first, then fine-tune (2026-11-03)
- **Chose:** train on `field_train` (PlantDoc and PlantSeg) plus `lab_train` capped per class, from the start (commit `3e46ee3`).
- **Considered:** the earlier plan trained on lab, then fine-tuned on `field_train` after augmentation and background fixes. Replaced because real photos are what the model fell on, and lab-only training fell far short (`E2-20261103T092315605470Z`: 0.2240). No lab-then-fine-tune run was made to compare.
- **Evidence:** `docs/DESIGN.md` decisions table.
- **Would change if:** joint training hurt lab performance enough to matter; that was not scored here.

### T4.1 - Lab cap 60, arm list and rules fixed before any run (2026-11-03)
- **Chose:** cap lab at 60 per class and write arms, rules and cap into `configs/e4.json` (commit `429b16f`) before any E4 result. At most 6 training runs plus 2 held-out runs. The combined augmentation-plus-background arm was dropped to make room for DINOv2 and recorded as `dropped`.
- **Why 60:** median 59 real photos per class, so 60 gives 2,280 lab against 2,069 real photos (lab 52% of rows); 80 would make lab 59%, 100 would make it 65% (from `configs/e4.json`, using `docs/class-report.md`).
- **Rules:** best epoch by `field_dev` accuracy; best arm by `field_dev` macro-F1; arm 5 runs only if arm 4 beats arm 1; the baseline is kept unless the paired interval excludes zero; the crop stays out unless it beats its uncropped twin the same way.
- **Evidence:** the verifier confirmed `configs/e4.json` was unchanged since `429b16f` and training rows were exactly 4,349 (2,069 + 2,280) with no dev or test overlap (verifier's measurement).

### T4.1 - Joint ConvNeXt selected over lab-only and the DINOv2 head (2026-11-03)
- **Result (`E4-20261103T153432358868Z` and the arm records, `field_dev`, development numbers):** `convnext_joint` accuracy 0.7644, macro-F1 0.7653, ECE 0.0794. E1 was 0.2240 and 0.199; joint beats E1 by +0.5404 accuracy (interval +0.5058 to +0.5743, `E4-20261103T143743951448Z`). `dinov2_head` macro-F1 0.7469, difference from joint -0.0184 (interval -0.0537 to +0.0150): no measurable difference (`E4-20261103T151153536379Z`).
- **Chose:** joint ConvNeXt, by the pre-set rule (highest macro-F1). The DINOv2 head is not clearly worse and is better calibrated (ECE 0.0318), so the choice is a rule outcome, not a proven win.
- **Caveat:** the arm and epoch were chosen on `field_dev`, so the numbers are optimistic; one seed only (verifier).
- **Would change if:** more seeds separate the two, or E12 changes the picture.

### T4.1 - DINOv2 fine-tune not run, by rule (2026-11-03)
- **Chose:** record arm 5 as `not_run`: `dinov2_head` macro-F1 0.7469 is not above `convnext_joint` 0.7653 (`E4-20261103T153432358868Z`, attempts).
- **Considered:** running it anyway. The pre-committed rule says no.
- **Would change if:** a later plan reopens DINOv2.

### T4.1 - Stronger augmentation and background paste: no clear gain, not adopted (2026-11-03)
- **Result against joint, `field_dev` (development numbers):** strong augmentation accuracy -0.0087 (interval -0.0279 to +0.0096), macro-F1 -0.0190 (`E4-20261103T145445487577Z`). Background paste accuracy 0.0000 (-0.0182 to +0.0174), macro-F1 -0.0107 (`E4-20261103T150434839939Z`). Both "no measurable difference".
- **Chose:** keep plain augmentation. The background arm ran because good masks existed (official segmented images matched by file name).
- **Verifier's note:** the pasted backgrounds are garish rather than natural, which is a design weakness; one seed.
- **Would change if:** natural backgrounds or more seeds showed a gain.

### T4.1 - Crop kept out of serving (2026-11-03)
- **Result (`E4-20261103T153432358868Z`, `field_dev`):** `best_with_crop` macro-F1 0.7359 (0.736) against joint 0.7653, difference -0.0294 (interval -0.0560 to -0.0044); accuracy 0.7404 against 0.7644, difference -0.0240 (-0.0466 to -0.0019), McNemar p 0.037. Significantly worse.
- **Chose:** leave the crop out, as the rule says. The cropped arm has no Grad-CAM energy number because the leaf mask would not line up.
- **Caveat:** E10 masks were trained on PlantSeg only.
- **Would change if:** a crop model trained on more varied photos beats its uncropped twin.

### T4.1 - Held-out sources score far lower than the joint number (2026-11-03)
- **Result (`field_dev`, development numbers, each against joint on the same rows):** without PlantDoc, macro-F1 0.3778, accuracy 0.4330 against 0.7322 and 0.7447 (`E4-20261103T151651745350Z`). Without PlantSeg, 0.5185 and 0.5463 against 0.7752 and 0.7933 (`E4-20261103T152355622205Z`). Both differences are significant (McNemar p 5.8e-41 and 3.4e-25).
- **Chose:** report seen-source and held-out numbers side by side everywhere (T8.2). The joint score mostly reflects sources seen in training.
- **Leakage check (verifier's measurement, `orchestration/verdicts/T4.1.json`, not a result):** about 40 of 1,040 `field_dev` images are same-scene look-alikes of a training image; removing every image with a close neighbour moves joint accuracy from 0.7644 to 0.7552. The held-out gap stays large on images with no look-alike, so it points to source shift and the loss of about half the real training data, not leakage. Also: PlantSeg has no healthy images, and PlantDoc white-background stock photos are mostly healthy, a style cue that will not transfer to phone photos.
- **Would change if:** E12 (more sources) narrows the held-out gap.

### T4.2 - E5 calibration design: fit on one half of `field_dev`, score on the other (2026-11-03)
- **Chose (design only; no results yet, on branch `task/T4.2`, `configs/e5.json` commit `5589897`, fixed before any run):** cut `field_dev` and `not_plant_dev` by near-duplicate group, 50% to fitting (seed 0). Temperature, abstention threshold and gate settings are chosen on the fitting half only and scored on the scoring half. Gates compared: the old 5% green-pixel rule, confidence alone, and confidence plus an entropy score, with plant rejection kept to 5% on the fitting half.
- **Considered:** fitting and scoring on all of `field_dev`. The config gives no reason beyond the method text; the evident point is that a threshold should not be scored on the rows that chose it.
- **Limits stated in the config:** the model was selected on `field_dev`, so these are development numbers, and seen-source calibration may not carry to new sources (see held-out entry above).
- **Would change if:** E12 changes the chosen model; calibration is refit then.

### T4.2 - Dataset class defined at module level for DataLoader workers (2026-11-03)
- **Failed:** the E5 scoring dataset was built inside a function, so Windows worker processes (spawn) could not pickle it (commit `4dfb3cd`, on branch `task/T4.2`, not yet on `build`).
- **Chose:** move it to module level. Later `6fba542` and `933e5bf` set E5 scoring to `num_workers` 0 and dropped the worker-spawning test; why is reason not recorded.
- **Would change if:** scoring `not_plant_dev` becomes slow enough to need workers.

### T4.7 - Label cleaning of the real photos: tooling choices (2026-11-03)
Tooling choices first, then the results of the review.
- **Why thumbnails and table diverged:** the table (`field-sample.md`, tracked in git) is exactly the seed-13 sample of the 4,148-row `field.csv` committed at `08cb625`; the thumbnails (git-ignored) are exactly the sample of the final 4,144-row `field.csv` (4 rows removed in `3c825e8`, the sample's positions shift). Checked by sampling every committed `field.csv` and comparing: table 50/50 with the old file, thumbs 50/50 with the current file. Thumbnails were written 22:14:21, the table file's mtime is 22:14:42, after the rebuild; the code writes both in one call, so one call cannot diverge. Most likely cause (not proven; no log of the git operation): the rebuild wrote both, then a git checkout or merge restored the committed old table while the ignored thumbnails stayed. The root problem is two files with different lifetimes and thumbnails named only by position.
- **Fix:** each thumbnail is named `<#>-<sha1 prefix>.jpg`, the table lists the full image ID, and a rerun removes the earlier run's thumbnails. A stale thumbnail now fails to match rather than showing another photo. Regenerated with `write_review` on the current `field.csv`, not a rebuild.
- **Sheet size:** 6 x 5 cells of 256 px (1536 px wide, so the reader does not shrink it), 30 photos per sheet, 70 sheets for the 2,075 `field_dev` + `field_test` photos. Considered 4 x 4 at 380 px (130 sheets) and 5 x 5 at 300 px (83 sheets); a slide, a cup of tea, a hand holding a leaf, and a mildew-covered maple leaf were all recognisable and the label and file name legible at 256 px (looked at all three on the same 40 photos), so the largest count won. Second look: 3 x 2 cells of 500 px.
- **Image ID:** the manifest `sha1`; it is unique across all 4,144 `field.csv` rows (path is the fallback if that ever stops holding).
- **How cleaned rows are requested:** a new subset name `clean:<N>` (rows not in `datasets/label_exclusions.csv` up to version N), combinable with the old ones, e.g. `shared_classes+clean:1`. The split stays `field_dev`, so the plan's accept command `check_run E13 --min-n 200 --split field_dev` is unchanged, and `check_run` recomputes the scored rows from `field.csv` plus the log, so a record cannot claim cleaned rows it did not use. `field.csv` and the old records are untouched. Considered a new split name (`field_dev_clean`) and a different manifest argument: both change the record's `dataset_hash` rule and need new code in every consumer; the subset needs one new rule. `datasets/manifests/field_clean.csv` (same columns, plus `clean_version`) is a convenience copy for T4.3, T4.6 and T8.1 that train or sample from rows; it has its own hash.
- **E13 scoring:** every earlier system has saved per-image probabilities on the same `field_dev` rows, so E13 filters them (exact, paired, no GPU, no image opened) instead of rerunning models. Systems are refused, not rerun, if they lack saved rows.
- **Also decided:** a removal needs both looks (first pass and a larger second look); uncertain photos are logged and kept; `field_train` is estimated from a seeded sample of 400, with its removals logged in the same file, so `field_clean.csv` also lacks those sampled train rows until a full train review is done.
- **Test-once guards:** nothing counts or blocks sheets. `test_once_problems` counts run records on `*_test` splits only, and the joint-training loaders refuse test splits; sheets show image, label and file name only and produce no record. The cleaning does use `field_test` photos to decide removals (no model output seen), which is stated here and in `docs/label-review.md`.
- **Why clean the labels:** the H2 agent review found 7 wrong and 9 unclear photos in a sample of 50 (over the limit of 5), so the real-photo labels were reviewed and clear errors removed before the dev scores are trusted.
- **How many (exclusion version 1, `datasets/label_exclusions.csv`):** 38 removed: `field_dev` 12, `field_test` 19, `field_train` sample 7. Of the 2,075 `field_dev` + `field_test` photos, 31 removed (1.5%). By reason among the 31: not a plant photo 17, wrong plant 11, label contradicted 3 (the 7 sampled train rows are in the file too, so the file's reasons differ). 30 of the 38 began as first-pass `remove` and 8 as `uncertain` photos sent to the second look by the note pattern in `configs/e13.json`. 278 photos were uncertain at first pass; 283 rows are in `datasets/label_uncertain.csv` after the second look. `field_clean.csv` has 4,106 rows (4,144 minus 38).
- **Second look checked against the data:** the first pass proposed 46 removals. The larger second look confirmed 30, called 3 `ok` (the "overturned 3" in the `finish` output; it counts only those) and called 13 `uncertain` (kept and logged as uncertain, not removed). So 16 of 46 were not confirmed, 13 of them by being downgraded rather than overturned. Of 70 matched `uncertain` photos it confirmed 8 as removals, called 47 `ok` and left 15 uncertain.
- **Dev scores, old (all rows) -> new (cleaned rows), accuracy / macro-F1, from E13 records `E13-20261103T184414193293Z` (shared classes, 1,040 -> 1,028 rows), `...T184438106036Z` (PlantDoc, 619 -> 610) and `...T184452515675Z` (PlantSeg, 421 -> 418); same saved predictions filtered, so old and new are the same models:** convnext_joint 0.7644 / 0.7653 -> 0.7695 / 0.7712; dinov2_head 0.7462 / 0.7469 -> 0.7500 / 0.7498; dinov2_small (E3 head) 0.4067 / 0.3819 -> 0.4076 / 0.3835; frozen_convnext 0.2519 / 0.2127 -> 0.2539 / 0.2134; xgboost 0.0413 / 0.0295 -> 0.0418 / 0.0297; e1 (lab model) 0.2240 / 0.1990 -> 0.2257 / 0.1978. Held-out by source: PlantDoc view convnext_joint 0.7447 / 0.7322 -> 0.7525 / 0.7398, heldout_plantdoc 0.4330 / 0.3778 -> 0.4377 / 0.3789, e1 0.2294 / 0.2251 -> 0.2311 / 0.2242; PlantSeg view convnext_joint 0.7933 / 0.7752 -> 0.7943 / 0.7757, heldout_plantseg 0.5463 / 0.5185 -> 0.5478 / 0.5182, e1 0.2162 / 0.1982 -> 0.2177 / 0.1998. Cleaning moved every score by under 1 point, the ranking of systems did not change, and the records' paired comparisons (convnext_joint against each other system) are on the cleaned rows. Each record states exclusion version 1, the clean manifest hash and `device: none` (no model was run).
- **Train decision:** the `field_train` sample (400 photos, seed 13) had 7 removals, 1.8%, Wilson 95% interval [0.9%, 3.6%]: the point estimate and the whole interval are under the 5% line, so no full `field_train` review and no retrain. Caveat: the calibration (of 4 known-bad photos, 2 were `remove`, 1 `uncertain`, 1 `ok`) says the sheets are conservative, so every removal rate here is a lower bound on bad photos; with that miss rate the true train rate could be higher than the interval, and the decision is on the review as run, not a proof the train set is clean. The 7 sampled train rows are absent from `field_clean.csv`, but E4's models were trained on `field.csv`'s train rows, so E4's training and its records are unchanged.
- **Who saw what:** reviewers saw `field_test` photos (19 of the removals are test photos) with their label and file name and no model output; the test labels are therefore cleaned by looking at the photos, which is stated here because it is a use of the test split. No `field_test` score exists.
- **Considered and rejected:** one look only (the second look did not confirm 16 of the 46 first-pass removals); removing every uncertain photo (283 rows, mostly clean-looking leaves under a disease label, where small spots can't be ruled out, so this would be taste rather than clear error); relabeling (ruled out by the approved instruction); reviewing all of `field_train` now (the sample's interval is under the 5% line, though the verifier's sampling and the calibration leave a true rate around 3-4% plausible, not proven under 5%).
- **Would change if:** the second look overturns many first-pass removals (the first pass is too loose), or the `field_train` estimate is over about 5% (review all of `field_train`).

---

### T4.3 - E11 design fixed before any answer or result (2026-11-03)
- **Chose (design only; no answers and no results yet; `configs/e11.json`, commit `146f760`, with one later wording change to the decoy rule before any answer):** score E4's saved `field_dev` probabilities on the cleaned rows (`clean:1`, 1,028 photos), no GPU. A photo is "unsure" when at least 2 of the model's top 3 classes have probability 0.10 or more (the product's floor for showing an alternative); it is asked about only if at least one question can be asked about its top 3. Update rule: "yes" multiplies the probability of the top-3 classes the question is tied to by 3, "no" by 0.5, the top 3 is renormalised to the probability it had, at most 2 questions per photo, the next question being the one that splits the top-3 probability closest to half. These numbers were set from the argument that a symptom a document calls characteristic is stronger evidence seen than absent (a photo may not show it), not from any data, and are not tuned.
- **Considered:** tuning the two factors on a fitting half by near-duplicate group, as E5 did. Rejected for now: one setting is cheaper to explain and there is no result yet to tune against; if the flow does not help, the finding is that it does not help at this setting, and any later tuning would need a fitting half and a new record. Also considered the E5 temperature on the probabilities before the 0.10 cut; not used, so the raw E4 numbers decide who is unsure.
- **Healthy classes (12, `not applicable` in `docs/document-coverage.csv`):** no question is ever about one, because no document describes symptoms for them. A healthy class can still be in the top 3: a "no" to a symptom question lowers the diseases the symptom belongs to, and a "yes" to another class's symptom lowers it, both relatively. All 26 disease classes are covered, so no top-3 class is reported uncovered (planned count: 0); 6 to 48 unsure photos per model have no askable question (for example three healthy classes, or a top 3 whose only covered classes share every question) and are left alone and counted.
- **Both effects are reported:** the whole cleaned split and the unsure subset, since a gain on a subset is not a gain overall; a record covers the whole view and its asked subset is in `config.flow` with its own paired interval (accuracy, macro-F1, and ECE through a group bootstrap in `e11_followup.ece_diff`, which `check_run` does not recompute; a test recomputes it from the saved rows).
- **Would change if:** the flow clearly helps (then fit the factors on a half and score the other, and recheck on a model E12 may choose) or clearly hurts (then it stays out of the product).

### T4.3 - 52 questions drafted from document passages, two per disease class (2026-11-03)
- **Chose:** 52 yes/no questions (`datasets/followup/questions.jsonl`), two for each of the 26 covered disease classes, each with the document file, page (HTML is page 1) and the passage it is taken from, word for word, so a verifier can compare. Drafted by the builder, not by a model run on its own; a test (`test_every_passage_is_in_the_cited_page`) checks each passage is on the cited page of the document in `knowledge/raw`, and another that every cited document has a known source (none of the 11 unknown-source originals is used). Questions avoid the class and crop names and disease-name words (a test), avoid sizes, timing and anything a photo cannot show, and quote leaf symptoms except two on fruit or berries (orange fruit shape, esca berry spots), where "unknown" is an allowed answer.
- **Per class (count 2 each):** apple scab, apple black rot, cedar apple rust, cherry powdery mildew, corn gray leaf spot, corn common rust, corn northern leaf blight, grape black rot, grape esca, grape leaf blight, orange HLB, peach bacterial spot, pepper bacterial spot, potato early blight, potato late blight, squash powdery mildew, strawberry leaf scorch, tomato bacterial spot, early blight, late blight, leaf mold, Septoria, spider mites, target spot, yellow leaf curl, mosaic virus.
- **Considered:** one question shared by several classes (for example "white dusty coating" for both powdery mildews). Rejected: each question must cite the passage for the class it is tied to, and a shared question would need one passage per class. Several questions in different classes therefore look alike in meaning (white dusty coating, rings inside spots), which is a limit of the answerer, not of the rule.
- **Would change if:** a verifier finds a passage that does not support its question, or a question a photo cannot answer.

### T4.3 - Blind answers: what the sheets hide, and what they cannot (2026-11-03)
- **Chose:** the answerer is a vision agent, which is not a grower, and the record says so (`config.answerer_note`, `config.answers`). Sheets show 6 photos each at 500 px with a yellow number badge and, in a separate text file, that photo's questions in a seed-shuffled order numbered from 1. No label, prediction, class name, file name, path, split, or question id appears (a test scans every sheet file, including the image bytes, for those). Sheet order is shuffled by seed 13 (a test); the key from sheet and cell back to the photo is in the git-ignored `datasets/followup/key/`, apart from the git-ignored `datasets/followup/sheets/` the reviewers get. The plan, sheet and collect code never reads a label (a test inspects its source); the plan stage takes only each row's path and probabilities. "unknown" is a valid answer and is counted.
- **Answers before scoring:** `datasets/followup/answers.jsonl` (tracked) must be committed and unchanged before the record stage will run, and `test_answers_were_saved_blind_to_the_label` / `e11_followup.commit_order_problems` check that every commit touching it is a strict ancestor of the commit that first adds an E11 record. If the answers and the first record are squashed into one commit, that check fails by design.
- **What the sheets cannot hide:** which questions a photo gets depends on the model's top 3, and the flow cannot work otherwise. Every photo therefore gets exactly 10 questions: every question the flow could ask about its top 3, under any model that finds it unsure (between 2 and 10), plus decoys about other classes, preferring the same crop. Audit of the plan (it looks at the labels of the planned photos only to describe the leak, and nothing was changed after it): the true class has a question on 427 of 619 sheets' lists (69%), 397 of them real candidates and 30 decoys; so the list does hint at plausible classes. It does not say which three, and the answerer does not see a label. The answers are a reading under that hint, not a free reading.
- **Considered:** asking all 52 questions for every photo, which removes the hint but means 32,188 answers (619 photos); rejected as too many for the reviewers. Fewer decoys, rejected for the leak above.
- **Would change if:** an answerer's yes rate on decoys is far above its rate on questions about classes that are plainly absent, which would show it answers from the list rather than the photo.

### T4.3 - Seen-source and held-out-source models side by side (2026-11-03)
- **Chose:** three records on `field_dev` cleaned rows: the seen-source model (E4 `convnext_joint`, 1,028 photos), and for each held-out source the held-out model (`heldout_plantdoc` on its 610 PlantDoc rows, `heldout_plantseg` on its 418 PlantSeg rows) beside `convnext_joint` on the same rows. The saved rows are E4's; the joint model's rows inside each held-out record were confirmed identical to the seen record's before use (checked on the 610 PlantDoc and 418 PlantSeg rows). One shared answer file serves all three, since the sheet for a photo carries the questions for every model that finds it unsure. If E12 later replaces the model, this result stays E4's and its records say so.
- **Considered:** one record mixing the two held-out models, which would give no single model for the rows. Rejected.
- **Would change if:** E12 chooses a different model, in which case E11 is a new run on its probabilities.

### T4.3 - Result: follow-up questions do not help; left out of the product (2026-11-03)
- **Result (records `E11-20261103T224508347561Z`, `...T224526379096Z`, `...T224537576002Z`, cleaned `field_dev`, E4's model, development numbers, answers by three blind Sonnet vision agents, not growers):** seen-source joint model, 1,028 photos: accuracy 0.7695 -> 0.7636 (difference -0.0058, interval [-0.0185, +0.0068], McNemar p 0.46), macro-F1 0.7712 -> 0.7618 (-0.0094, [-0.0284, +0.0106]), ECE 0.0769 -> 0.0663 (-0.0106, [-0.0227, +0.0122]). Nothing is clearly better or worse, and no interval excludes zero.
- **Who was asked and what changed (seen model):** 340 photos were unsure, 292 had a question to ask (48 had none), 584 questions were asked (2 per photo). The top-1 answer changed on 70 photos: 20 became right, 26 became wrong, 24 changed from one wrong class to another. Answers: 86 yes, 339 no, 159 unknown (unknown rate 27.2%).
- **Held-out sources beside the seen number (accuracy / macro-F1, before -> after, whole view; asked photos):** PlantDoc rows (610), joint model 0.7525 -> 0.7492 / 0.7398 -> 0.7394 (178 asked: 11 better, 13 worse, 17 changed wrong to wrong, unknown 22.8%); PlantDoc held-out model 0.4377 -> 0.4492 / 0.3789 -> 0.3947, accuracy difference +0.0115 [-0.0115, +0.0347] (342 asked: 30 better, 23 worse, 75 wrong to wrong, unknown 16.1%), ECE 0.2116 -> 0.1805 with an interval that excludes zero [-0.0551, -0.0049]. PlantSeg rows (418), joint 0.7943 -> 0.7847 / 0.7757 -> 0.7703 (114 asked: 9 better, 13 worse, unknown 34.2%); PlantSeg held-out model 0.5478 -> 0.5455 / 0.5182 -> 0.5345, accuracy -0.0024 [-0.0269, +0.0235] (187 asked: 13 better, 14 worse, 18 wrong to wrong, unknown 32.4%). The held-out models start far lower (0.44 and 0.55 against 0.77), so there is more to gain, and macro-F1 rises a little on both held-out models while accuracy moves +0.0115 and -0.0024, but every accuracy and macro-F1 interval spans zero. Only ECE on PlantDoc held-out (and ECE of the asked photos there) has an interval that excludes zero; with this many comparisons that is not evidence of a gain, and it comes with no accuracy gain.
- **Chose:** it does not help at the settings fixed before any answer, so by the plan it stays out of the product. Not tuned afterwards, and the factors 3 and 0.5 were not changed after seeing the result.
- **Why it does not help (from `docs/e11-answer-check.md`, saved answers only):** the answerer does say yes more to the true class: yes rate 0.233 on real-candidate questions about the photo's true class (0.320 of answered), against 0.078 on candidate questions about a wrong class and 0.030 on decoys. But even for the true class only 23% are yes, 50% are no and 27% unknown, so "no" is as common as "yes" for the class that is right, and a "no" lowers the right class. The wrong-class yes rate (0.078) is above the decoy rate (0.030); that may be real look-alike symptoms among confusable classes, or reading from the list, and the data cannot separate the two.
- **Answering history, stated because it was a restart:** sheets s001 to s070 were answered first with looser wording about no versus unknown (unknown rate 5.3% and 9.0% for the two agents, against 24.8% on s071 to s104). Nothing had been scored. Those answers were set aside in the git-ignored `datasets/followup/attempt1/` and all of s001 to s070 were redone under one wording (unknown 20.5%, 17.5%, 24.8% for the three agents), which is the answer file in `datasets/followup/answers.jsonl`. It was a restart for consistency, not a selection on a result.
- **Sensitivity, for this report only, not a new selection (`docs/e11-answer-check.md`):** with the attempt-1 answers for s001 to s070 (704 of 6,190 answers differ), the seen model goes 0.7695 -> 0.7607 (unknown 16.4%), the PlantDoc and PlantSeg held-out models 0.4377 -> 0.4557 and 0.5478 -> 0.5287, and the conclusion is the same: no clear gain, joint model no better, PlantSeg held-out worse by 1.9 points with an interval [-0.0477, +0.0096] that still spans zero.
- **Would change if:** a different update rule (for example weighting "no" far lower, since it is common for the true class) were fitted on one half by group and scored on the other, or a better answerer (a grower) were used. Neither is planned.

### T4.6 - Extra rows enter as a third training split; held-out runs also drop their copies (2026-11-04)
- **Chose:** `TrainConfig.extra_manifest` loads `extra_train.csv` (split `extra_train`, `source` plantwild, plantwild_v2 or plantpathology2021) beside lab and field; `extra_train` joins `TRAINING`, selection stays `field_dev`, test splits are still refused. The extras are not capped again (T4.5 capped them at 200 per class). In a held-out-source run, `select_rows` removes the source's rows as before and also any extra that shares a sha1 or a group with a row of that source (any loaded split).
- **Considered:** adding the extras as more `field_train` rows, which would change `field.csv` (frozen, and its hash is in every record); a source filter on names, which would miss a copy filed under another source.
- **Evidence:** on the real manifests no extra has source plantdoc or plantseg and none shares a sha1 or group with either, so the copy rule removes 0 of 3,484 extras in both held-out runs (`test_held_out_source_is_absent_from_training_and_selection`, which also plants copies in a fixture and shows they leave). Near copies (not byte or group copies) of PlantDoc or PlantSeg photos are covered only by T4.5's checks against every `field.csv` split (full-frame correlation, crop-aware check, the field-dedup rule, the close-match guard); E12 adds no image-level check of its own. A one-off audit on 2026-11-04 (T4.5's 32x32 correlation over 8 symmetries, `max_correlation`, CPU, `field_test` not opened) scored every extra against the held-out source's `field_train` and `field_dev` photos: the best score was 0.83 (PlantDoc) and 0.82 (PlantSeg); 73 extras reach 0.75 against PlantDoc and 36 against PlantSeg. The first ten of each by file name pair different plants or classes (for example an apple-leaf extra with a strawberry PlantDoc photo), the known false match of white-background leaves, so they look like non-copies, but the images were not viewed and the audit is not part of the code. Lab and field group numbers restart per build, so only extra groups (prefix `x`) are compared.
- **Would change if:** a held-out run's score on a source looks too good, which would suggest near copies got through; an image-level check of the extras against that source would then be added and the run repeated.

### T4.6 - Settings copied into configs/e12.json and tested against E4 (2026-11-04)
- **Chose:** `configs/e12.json` states model, lr, epochs, batch size, transforms, seed, lab cap and the splits explicitly (ConvNeXt-Tiny, 1e-4, 10, 32, random_resized_crop+hflip+color_jitter, seed 0, cap 60). `test_settings_match_e4_best_apart_from_the_extra_rows` takes E4's best arm by E4's own rule from E4's records (`convnext_joint`), builds both train configs, and requires them equal except `extra_manifest`, `train_splits` (plus `extra_train`), `out_dir`, `num_workers` and `manifest_hash`; it also compares with the `train` block saved in the E4 records. The held-out runs are compared with E4's held-out arms the same way.
- **Considered:** reading the numbers from E4's config at run time, which hides a later edit of `configs/e4.json` from the E12 record.
- **Evidence:** the test above.
- **Would change if:** E4's best arm changes (a new E4 record that beats `convnext_joint`); the test would then fail and the config would be revisited.

### T4.6 - Scored on the cleaned rows from E4's saved predictions (2026-11-04)
- **Chose:** each run is predicted on every `field_dev` photo of its subset and scored on `<subset>+clean:1` (1,028 rows for `shared_classes`; 610 PlantDoc and 418 PlantSeg for the held-out parts), in the row order `check_run` reads. E4's systems in the same record (`e1`, `convnext_joint`, and the matching held-out arm) are E4's saved rows for the same photos, filtered by path; no E4 model is rerun. The records compared are the ones E13 used (`E4-20261103T143743951448Z`, `E4-20261103T151651745350Z`, `E4-20261103T152355622205Z`). No E12 branch was added to `check_run`: its generic recompute of metrics, intervals and comparisons covers these records.
- **Considered:** reading E13's filtered rows, which holds the same rows but adds a second record to trust.
- **Evidence:** a real-data check before any training showed the E4 rows cover all 1,028, 610 and 418 cleaned photos, and that with E4's own probabilities in place of a new model the paired difference is exactly 0.
- **Would change if:** the exclusion list gets a version 2 (then `clean_version` in the config changes and the runs are scored again from the saved `field_dev.predictions.jsonl`).

### T4.6 - What "clearly helps" means, fixed before any E12 run (2026-11-04)
- **Chose:** the extras clearly help if (1) on the seen-source run, the paired 95% interval of the macro-F1 difference (E12 minus E4 `convnext_joint`) on `shared_classes+clean:1` lies wholly above zero, and (2) on each held-out run the macro-F1 difference (E12 held-out minus E4 held-out, same rows) is at least -0.02. `decide` applies this to the records' own paired comparisons and writes `results/e12/decision.json` with the chosen model path (`models/e12/convnext_joint_extra/best.pt` if they help, else `models/e4/convnext_joint/best.pt`), its sha256, the side-by-side table, and the per-class table for Apple and Corn classes. It refuses unless all three records pass `check_run`.
- **Considered:** an interval on the held-out difference, which a held-out part of 418 to 610 photos cannot make narrow enough to show a two-point loss; accuracy instead of macro-F1, which hides the thin classes the extras were meant to help; "any positive difference", which would follow noise.
- **Evidence:** this is a rule, not a result. The margin of 0.02 is a judgment, not derived from a noise measurement: no repeat run with another seed exists to measure it.
- **Would change if:** the person wants a stricter or looser margin; it is one number in the config, but must be changed before the runs.

### T4.6 - E12 results: the rule says keep E4, and the held-out gain is large (2026-11-04)
- **Result under the rule fixed before the runs:** the extras do not clearly help, and E4's model stays. Seen-source macro-F1 on `shared_classes+clean:1`, 1,028 rows: E4 0.771, E12 0.784, difference +0.012 (paired 95% -0.017 to +0.040), so part (1) fails. Records `E12-20261104T101951153850Z`, `E12-20261104T103235489613Z`, `E12-20261104T104854427501Z`; `results/e12/decision.json`.
- **What the rule did not weigh:** held-out runs gained far more than the margin it tested for. PlantDoc held out, 610 rows: E4 0.379, E12 0.668, +0.289 (0.245 to 0.330). PlantSeg held out, 418 rows: E4 0.518, E12 0.752, +0.234 (0.180 to 0.293). The rule used the held-out runs only as a check against harm.
- **Costs seen on seen-source rows:** apple classes fell: scab F1 -0.031, cedar rust -0.058, healthy -0.069, the last with precision 0.889 to 0.633, which fits the warning that capped healthy classes could pull toward healthy. Black rot rose +0.044. Corn barely moved, at most -0.020.
- **Limits:** one run with one seed; intervals resample photos, not training noise. These are development numbers, and field_dev picked the best epoch. The extras and PlantDoc are all web photos, so a held-out web source is not the same as a new phone photo, and near copies of a held-out source's training photos were not removed by T4.5.
- **Would change if:** the coordinator or the person chooses the E12 model despite the rule, which would be a choice made after seeing these numbers and must say so, with E5 refit on the new model.

### T4.6 - Gates for choosing E12's model, fixed before their runs (2026-11-04)
- **Decided by the coordinator, which the person may overrule:** the pre-set rule said keep E4, and the record stays on E4 until all three gates pass. Passing would override that rule after seeing dev numbers, for this reason: the project's goal is holding up on photos from sources it hasn't trained on, and the rule tested held-out runs only for harm. No test split has been seen, so this remains dev selection. Any gate failing means keep E4. Gates don't change after seeing results.
- **G1, retrain:** with PlantDoc held out and its extras at DINOv2 class-token cosine 0.75 or above removed, the PlantDoc gain stays at least half of +0.289 (so at least 0.1445) and its paired 95% interval stays clear of zero. Sick-only macro-F1 and healthy accuracy are reported separately. If nearly all the remaining gain is healthy leaves and the sick-only gain's interval touches zero, that alone doesn't fail G1, but it must be stated, and the PlantDoc gain can't be described as better disease detection.
- **G2, second seed:** E12 and the E4 arms it's compared with rerun on one new seed, paired, for the seen-source and both held-out runs. Both held-out gains stay clear of zero, and the seen-source gain's point estimate is not below zero. About 90 minutes of GPU is the limit for this gate.
- **G3, safety:** after an E5 refit on E12, on the same rows, E12's confident sick-called-healthy answers are not more than E4's. A tie at low counts passes, and is reported as such, saying if E4's count is too low to mean much. Unthresholded rates are reported beside it for seen and held-out, both directions (from T4.6's verifier: seen 1.0% and 3.3%; PlantDoc held out 0 of 404 and 11 of 404; PlantSeg held out 18.9% and 3.3%).
- **The verifier's dissent:** it read this evidence as keep E4: the seen gain is noise-sized, PlantDoc's gain is mostly healthy leaves (healthy accuracy 0.10 to 0.71, sick 0.61 to 0.64), and this shows web-to-web transfer rather than phone photos.
- **If all pass:** E12 is primary and E4 is reported beside it on all the same measures in T8.1. The healthy and sick split goes wherever the gain is quoted. Phone photos stay untested, so no phone claim, and no number is called new-photo performance. T8.2 states the override.

### T4.6 - Loader workers: default 2, set by flag or environment, E4 used 4 (2026-11-04)
- **Chose:** `--num-workers` or `SPROUT_NUM_WORKERS`, default 2, at most 4 (T4.1's limit), recorded in the record's `config.device` next to E4's 4. The sample order comes from a torch `Generator` seeded with the run seed and does not depend on the count. The random augmentation draws happen inside the workers, seeded from that generator plus the worker's number, so they do depend on the count, in E4 as well. So E12 with 2 workers is not a bit-for-bit repeat of E4's draws even apart from the extra rows; the extra rows change the batches anyway, so no draw sequence could match. A `--resume` must use the count the run started with and refuses otherwise (the checkpoint stores it).
- **Considered:** per-sample seeding so results never depend on the count, which would change E4's augmentation stream relative to E4 and so break "settings unchanged"; fixing the count at 4, which the person's class jobs on this machine make unwelcome.
- **Evidence:** `test_workers_come_from_flag_then_environment_then_the_default`, and the resume test. No measurement of how much the count changes a score exists; it is part of the run-to-run noise that the single-seed comparison cannot separate from the effect of the extras.
- **Would change if:** a second seed or a 4-worker rerun shows the count moves macro-F1 by more than the margin.

### T4.6 - E5 refit needs probabilities on all of field_dev (2026-11-04)
- **Chose:** each run saves the probabilities of every `field_dev` photo it predicted to `models/e12/<run>/field_dev.predictions.jsonl` (1,040 rows for the seen-source run), beside the checkpoint and written atomically, because the record holds only the 1,028 cleaned rows. If `decide` says the extras help, E5 is refit by pointing `configs/e5.json` (`model.checkpoint`, `model.name`, `model.e4_record`) at the E12 model and running its stages `probs-not-plant` (GPU), `green`, `fit` and `record`.
- **Considered:** writing a second E12 record on all 1,040 rows, which duplicates the seen-source run.
- **Evidence:** E5's `load_field` requires a record that covers every `field_dev` photo, so it cannot read an E12 record as it is. This loader does not read the saved file either: that change to E5 is not made here and is needed before the refit.
- **Would change if:** the extras do not help (no refit).

## Serving, language model, and workflow

### Limits dropped by the person (2026-11-03)
- **Chose:** the person, through the coordinator, approved dropping three limits. Claude calls are Sonnet with a cap of 5,000. The cap on real training's DataLoader workers goes from 4 to 8 in E4, E5, and E10, and tests use what they need. Data leaving this machine in a new way no longer means stop and ask (the coordinator committed that `CLAUDE.md` change). Public data is no longer limited to what needs no login, which was the coordinator's wording, not the person's.
- **Considered:** keeping the limits; the person rejected that. Temporary load limits while the person is on the laptop are separate and sit on top of these caps.
- **Evidence:** the person's instruction, relayed by the coordinator. `num_workers` is a setting, and no recorded config changes with the cap; E4 and E5's first batches were seed-stable across worker counts in the tests that cover it. The worker crashes on this machine (`WinError 6`) came under load, so a higher count may need `num_workers` 0 again.
- **Would change if:** crashes or memory alerts return at higher worker counts. Plant Pathology 2021 follows as T4.8.

### T6.4 - Local model `qwen3.5:4b-q4_K_M`; a different judge (2026-11-02)
- **Chose:** `qwen3.5:4b-q4_K_M` as the local writer (100% on GPU, 63.5 tokens/s, 5/6 tool choices, 10/10 labels, 6/6 cited answers). `gemma4:e4b-it-qat` is the separate judge; Claude is never the judge.
- **Considered and lost:** `gemma4:e4b-it-qat` as writer (9/10 labels), `qwen3.5:9b` (85% on GPU, 29.8 tokens/s), `granite4.2:8b` (92% on GPU, 27.7 tokens/s, 5/6 cards). The larger models were about half as fast and left less GPU room.
- **Evidence:** `docs/model-choice.md`: a small selection test on 6 + 10 + 6 prompts, never a result or resume claim.

### T6.4 - Claude comparisons: Sonnet with a 5,000-call cap, not Haiku with lower caps (2026-11-03)
- **Chose:** comparisons use Sonnet; the cap is a runaway guard of 5,000 calls counted in `results/claude_calls.json`, and `ClaudeCodeClient` refuses past it.
- **Replaced:** Haiku with a cap of 300, raised to 1,000 on 2026-11-02 (`f851db6`), then Sonnet and 5,000 on 2026-11-03 (`3493a8c`), all as the person's limits. Why Haiku was dropped is reason not recorded.
- **Evidence:** `docs/DESIGN.md` limits, `docs/PLAN.md`. Only benchmark questions and public document text go to the cloud model.
- **Would change if:** only the person can change the cap.

### T6.4 - No graph RAG (2026-11-03)
- **Chose:** the plan names it as excluded ("No graph RAG"); upgrades to test one at a time are structure-aware chunking, a context header per chunk, and search queries from the top-3 diseases.
- **Considered:** graph RAG. Why it was excluded is reason not recorded.
- **Evidence:** `docs/PLAN.md` T6.4 (commit `f150227`).
- **Would change if:** nothing recorded.

### T6.4 - How E7 is built and scored (built 2026-11-03; stage numbers are added from the records as each stage runs)
- **Chose:**
  - Everything is scored on `rag_dev`; `rag_test` is not read by any E7 code. Retrieval is scored on the answerable questions (declared subset `answerable`, 60 questions) with recall@1/3/5/10 and MRR; the deciding metric is recall@5, and a candidate is kept only when the paired bootstrap interval of its gain over the current choice excludes zero (otherwise "no measurable difference" and the simpler setup stays).
  - Gold matching: a passage is the gold hit when source and page equal the question's gold. A document the index holds as one page (every HTML page and Markdown file is "page 1"; `tomatodoc2.md` is one 177k-character page) would match any of its chunks, so for those the passage must also contain one of the question's answer terms. Citations show document, page and the opening 200 characters of the passage, so a one-page document still points at the place.
  - The plant for a question comes from the question text (or its stated crop), never from the gold label. The photo diagnosis is simulated: the plant's class that the question names is the top disease (0.9), with two other classes of that plant as alternatives. A plant with no such class has no diagnosis. This is an upper bound on what the top-3 query upgrade can show, since a real identifier is sometimes wrong.
  - Records: `write_run` (also logs to MLflow). `check_run` gained the subset `answerable` and the metrics `declines_when_crop_mismatch` and `declines_when_ambiguous`; every number in an E7 record recomputes from its saved rows (`tests/test_e7.py` runs `check_file` on each kind).
  - An answer counts as `answered` only when it is grounded (cites a shown passage). A model outage or search failure is never written as an answer; the stage leaves the question to resume.
- **Considered:** scoring recall over all 115 dev questions (lost: abstain and ambiguous questions have no gold, so every system's recall would be diluted by the same constant); matching one-page documents by source alone (lost: any chunk of a 177k-character page would count); taking the plant from the gold source (lost: it leaks the answer into retrieval).
- **Evidence:** `src/sprout/experiments/e7_rag.py`, `tests/test_e7.py`, `tests/test_run_e7.py`.
- **Would change if:** the set is reviewed (H2) and gold labels move, or a one-page document gets real section anchors.

### T6.4 - Score floor per mode (built 2026-11-03, numbers pending the `floors` stage)
- **Chose:** `RetrievalConfig.floors` holds a floor per `mode` or `mode+rerank`; a mode without one falls back to `min_score`. The `floors` stage measures the best passage's score for every dev question in dense, hybrid and hybrid+rerank, picks for each the floor that best keeps answerable questions and drops `not_in_documents` and wrong-crop ones (balanced), and records the result next to the old 0.3. Floors are picked on `rag_dev` itself, so the rates are optimistic and stated so in the record.
- **Considered:** one floor for all modes (lost: the T6.5 verifier showed 0.3 filters almost nothing without reranking, because dense cosine is clipped at 0 and hybrid/BM25 scores are normalised to the top hit, so BM25's best passage always scores 1).
- **Evidence:** pending record from `tools/run_e7.py floors`.
- **Would change if:** hybrid/BM25 scores stay uninformative; a raw (not top-normalised) BM25 score would be the next thing to try, and was not built because the plan asks for no extra retrieval features.

### T6.4 - Shared crops: `crops_from_title`, off until measured (2026-11-03)
- **Chose:** an option that lets a chunk answer for every known crop named in its document's title or disease list, in addition to its manifest crop. A mixed page titled "Bacterial spot of tomato and pepper" then answers pepper questions while its manifest crop stays tomato. Off by default; the `crops` stage measures it.
- **Considered:** a `crops` list in the manifest (lost: would change the committed manifest and tests that assume one crop per row); tagging a chunk with any crop its text mentions (lost: apple documents mention tomato ringspot virus, which would let a tomato question cite apple). `test_tomato_question_cannot_cite_apple` and `test_crops_from_title_never_adds_a_crop_the_title_lacks` guard this.
- **Evidence:** pending record from `tools/run_e7.py crops`.
- **Would change if:** the stage shows no recall@5 gain, in which case the option stays off.

### T6.4 - Structure-aware chunks, context header, OCR (2026-11-03)
- **Chose:** `build_index(..., chunker="structured")` keeps paragraphs whole, makes each table (HTML tables, Markdown tables, columnar PDF lines in pypdf layout mode) one chunk up to 4,000 characters and splits longer ones by rows repeating the header row. `header=True` stores a context header (document, publisher, crop, diseases, section heading) in a separate `header` field that is embedded, matched and reranked but never shown. Variant indexes are built beside the main one (`knowledge/index-structured`, `index-header`, `index-structured-header`) with `tools/build_index_gpu.py`, which refuses to build over `knowledge/index`. Both are kept only if their stage shows a recall@5 gain.
- **OCR: not added.** No OCR engine is installed (no `tesseract`, `easyocr` or `rapidocr`), and the 237-document build has 0 failed extractions, so there is nothing for OCR to recover and no way to measure it. The smallest addition would be `rapidocr-onnxruntime` (onnxruntime is already installed; it adds one small package and its model files). It becomes worth adding when a build lists scanned PDFs among its failures.
- **Considered:** page-aware table detection with a PDF table library (lost: new dependency for an unmeasured gain).
- **Evidence:** `tests/test_ingest_structured.py`; stage numbers pending.
- **Would change if:** the `chunking` or `header` stage shows a gain.

### T6.4 - Queries from the top-3 diseases and the conversation (built 2026-11-03)
- **Chose:** `build_query` modes `question` (default), `history` (the user's last two turns plus the question), `diseases` (top-3 disease names with confidence at least 0.05) and `full` (both). Assistant turns never go into the query, so a model's own words are not searched. The `queries` stage measures `history`, `diseases` and `full` each against the base.
- **Considered:** adding assistant turns (lost: echoes the model's earlier claims back into retrieval).
- **Evidence:** `tests/test_answer.py::test_query_modes_add_history_and_top3_diseases_but_not_assistant_text`; stage numbers pending.
- **Would change if:** a stage shows no gain; the default stays `question`.

### T6.4 - Prompts: a base plus one rule at a time (built 2026-11-03)
- **Chose:** `prompts/base.txt` plus three single-rule additions (`faithful`: every fact sentence ends with its passage number; `abstain`: reply `NO_SUPPORT` when no passage answers; `ask`: reply `NEED: ...` when plant or symptoms are missing). Each addition is generated and judged alone against the base on the one local writer (`qwen3.5:4b-q4_K_M`, judge `gemma4:e4b-it-qat`), then the kept ones are combined and re-checked. An addition is kept when its target metric rises and `answers_when_answerable` does not fall by more than 0.05. With about 80 graded answers per variant there is no interval, so the rule is direction only, and the record says so.
- **Considered:** a fixed chain v1 to v4 (lost: a dropped change would stay inside every later version, so changes would not be measured one at a time).
- **Evidence:** `tests/test_e7.py::test_prompt_additions_are_kept_only_when_they_help_without_costing_answers`; stage numbers pending.
- **Would change if:** T6.6's second grader disagrees with the local judge on the saved claims.

### T6.4 - Answers carry sources, uncertainty and labels in code, not only in the prompt (2026-11-03)
- **Chose:** `answer()` cites only passages that were shown (other `[n]` markers are removed), labels an answer with no citation as general advice without a supporting document, says "possible match, not certain" when the diagnosis is under 0.8 or `unsure`, appends "check the product label and your local rules" to any chemical or disposal advice that lacks it, returns a visible "search failed" or "unavailable" text instead of inventing one, and bounds history (4 turns, 500 characters each). A citation whose manifest source is `unknown: inherited from the legacy project, source not confirmed` shows that note and an empty link, never a blank. Text from the user or documents cannot close the data tags the prompt uses.
- **Considered:** leaving these to the prompt (lost: a prompt rule can fail silently on a small model, and these are the safety rules).
- **Evidence:** `tests/test_answer.py` (22 tests).
- **Would change if:** a stage shows the label or the prefix hurts faithfulness judgments; they are not part of the judged claims (`split_claims` skips them).

### T6.4 - Models on the GPU, or an error (2026-11-03)
- **Chose:** `sprout.rag.gpu_models.TorchEmbedder` and `TorchReranker` run on CUDA, fail with an error when CUDA is missing (never the CPU), and record device, GPU, torch and CUDA versions in the run record. Torch is imported only inside their constructors; `sprout.rag` and `sprout.llm` stay torch-free at import (`tests/test_no_torch_imports.py` now covers `llm`). The reranker uses the HF weights `cross-encoder/ms-marco-MiniLM-L6-v2`, the model behind the ONNX one serving uses. A smoke run on the RTX 4070 Laptop GPU gave a minimum cosine of 0.99999988 against the saved index vectors and peaked at 224 MiB allocated.
- **Considered:** fastembed on the GPU (lost: the installed onnxruntime has no CUDA provider, as T6.5 found).
- **Evidence:** smoke run on 2026-11-03 under `gpu-acquire T6.4`; `tests/test_retrieve_upgrades.py::test_cuda_is_required_not_replaced_by_cpu`.
- **Would change if:** serving should use the same reranker weights through ONNX; the two should then be compared.

### T6.4 - Graded answers are saved for a second grader (2026-11-03)
- **Chose:** each judged answer is saved with question ID, the passages shown, the answer text, every claim with its cited passage numbers, the judge's verdict and raw reply per claim, the rubric text and the judge model (`results/runs/E7-<stage>-<date>.judgments.jsonl`). A second file with the verdicts, replies and judge name removed (`...blind.jsonl`) is what another grader gets. Unparseable judge replies count as not supported. The prompts stage prints the number of graded answers and exits with code 3 if it is under 200.
- **Considered:** one file with a flag (lost: easy to leak the verdict to the second grader).
- **Evidence:** `tests/test_e7.py::test_judged_rows_keep_everything_a_second_grader_needs`, `tests/test_run_e7.py`.
- **Would change if:** T6.6 needs another field.


### T6.4 - Method revised after the first pass: four selection rules were hiding gains (2026-11-03)
The first pass of E7 (records `E7-20261103T211314827721Z` to `E7-20261103T231812228057Z`, kept untouched as evidence) showed that several rules made real differences look like nothing. The rules below were changed after seeing those numbers, at the coordinator's direction, so none of this is a pre-registered result. Version 2 reruns everything from scratch in `results/e7/work/v2` and `results/e7/v2`.
- **Chose, retrieval rule:** keep a change when the paired interval of either recall@5 or MRR excludes zero and the other point estimate is not worse. In the first pass hybrid_rerank had recall@1 0.600 against dense 0.417, recall@3 0.800 against 0.617, recall@5 0.833 against 0.717, recall@10 0.933 against 0.817 and MRR 0.712 against 0.543, yet dense was kept because the recall@5 interval [0.0, 0.233] touched zero at n=60. MRR comparisons use the same paired bootstrap; McNemar for MRR is on first-rank hits (documented in `check_run`).
- **Chose, upgrades against the chosen base:** header, crops, queries and chunking are each measured against the base the new rule picks (the first pass measured them against dense). Checkpoints are keyed by method version, settings, index build and question set, so a changed method or a rebuilt index never reuses old rows. Results are also shown per question category and with a lenient "any document holding an answer term" rank (`_terms` systems), because the gold-page rule is strict: the index grew from 28 to 237 documents, so other documents may answer legitimately.
- **Chose, floor rule:** among floors that answer at least 80% of answerable questions, the highest `declines_when_unanswerable` (ties: more answers, then the lower floor); if none reaches 80%, the highest mean of the two, and the record says so. The first pass picked a dense floor of 0.80 that declined every unanswerable question and answered only 65% of answerable ones. Only `abstain` questions count as unanswerable here; wrong-crop and ambiguous declines are reported beside it. Floors are picked on rag_dev, so the rates are optimistic.
- **Why answerable questions fell under the dense floor (first-pass data, 21 of 60):** 13 were follow-ups (multi_turn) and 8 single questions. The gold page was within the top 10 for 16 of the 21 and missing for 5, so retrieval usually found it but cosine scored short follow-ups low (the query was only "How does it come back each year?"); the floor, not retrieval, removed them. Dense cosine scores overlap: at a floor of 0.75 dense answered 76.7% of answerable and declined 78.6% of abstain questions, at 0.70 it answered 85% and declined 21%. With reranking, 12 of 60 fell under the first-pass floor, 11 of them follow-ups. The version-2 floors stage records, per mode, how many under-floor questions have the gold page in the top 10, no simulated diagnosis, a plant filter applied, or would pass the floor without the plant filter.
- **Chose, prompt starvation fix:** the first pass ran the prompt variants behind the dense floor, so 60 of 101 questions never reached the writer (0 model calls for those rows) and only 41 answers per variant could be judged; the abstain and faithful additions could not show an effect. Version 2 runs the prompt variants with no floor (`answers-nofloor`, every question reaches the writer), and a `served` variant (chosen prompt, chosen floors) shows the app's behaviour. The faithful addition's +0.018 on about 41 answers was noise; version 2 prints a bootstrap interval for every prompt gain and labels a gain whose interval includes zero "NOT distinguishable from noise". The direction rule is kept so a prompt is still chosen, but it is reported as noise when it is. The `ask` addition lowered answers to 0.533 because 7 answers asked a question back and 2 had no citation; whether that is a real effect is read from McNemar's exact p on the paired answered flags, printed in the prompts draft.
- **Device and speed:** `config.device` is now a plain string (`cuda`, or `writer cuda; judge cuda+cpu (NN% of the model in GPU memory)`), with details in `config.device_info`; the first pass stored a dict, which MLflow's `device` parameter drops. Ollama placement comes from `/api/ps` after a one-token load call, and a model wholly on the CPU stops the stage. Each generated row stores `llm_calls` and `llm_seconds`, and the record has `writer_usage` (calls, seconds, rows without a call). The first pass's `generate` took about 5 minutes for 404 rows because 60 of every 101 rows had no passage, so the writer ran for 41 rows per variant (about 164 calls).
- **Question count:** rag_dev has 101 questions (46 supported, 14 multi_turn, 14 wrong_crop, 14 not_in_documents, 13 ambiguous), not the 115 I wrote earlier (an addition slip); 60 are answerable.
- **Rerank cost:** a `latency` stage times the reranker and the whole search on CUDA torch, CPU torch and CPU ONNX (serving's path) for the same candidates, and records ranks too so the paths can be checked for agreement. It is meant to run with the GPU otherwise idle.
- **Other selection rules checked for the same fault** (a rule that makes nothing look helpful): (1) the strict gold-page rank, handled by the `_terms` cross-check and the category breakdown; (2) the pooled recall averaging a follow-up-only gain away, handled by the category breakdown; (3) the 0.05 answer-drop limit for prompts, which in the first pass was measured on floor-starved answers and is now measured with every question reaching the writer; (4) judge replies that are neither SUPPORTED nor NOT_SUPPORTED count as not supported, which lowers every variant alike; (5) the crops option touches only 9 of 237 documents, so little effect is expected and the stage prints the count; (6) `answered` means grounded, so a good model answer with no citation counts as not answered in all variants alike.
- **Considered:** keeping the first-pass rules and only rerunning (lost: they hid a clear reranking gain and starved the prompt stage).
- **Evidence:** first-pass records above; `tests/test_e7.py` (the rule tests use the first pass's numbers), `tests/test_run_e7.py`.
- **Would change if:** a pre-registered rule is wanted for a final claim; these numbers are not a claim.

### T6.4 - Result: hybrid + rerank is the retrieval base (method v2, 2026-11-03)
- **Chose:** hybrid search with the cross-encoder reranker over `knowledge/index` (237 documents), question as the query. The v2 rule (a paired interval of recall@5 or MRR excludes zero, and the other point estimate is not worse) kept it over dense.
- **Considered:** dense and hybrid without reranking. Dense is the simpler setup and was kept by the first pass; hybrid without reranking was never better than dense in either pass.
- **Evidence (v2, record `E7-20261103T234741564452Z`, 60 answerable rag_dev questions; recall@1 / @3 / @5 / @10 / MRR):** dense 0.417 / 0.617 / 0.717 / 0.817 / 0.543; hybrid 0.367 / 0.583 / 0.717 / 0.783 / 0.507; hybrid + rerank 0.600 / 0.800 / 0.833 / 0.933 / 0.712. Against dense, hybrid + rerank gains recall@5 +0.117 (95% CI 0.000 to +0.233, McNemar p=0.092) and MRR +0.170 (CI +0.084 to +0.262, p=0.007); hybrid alone is recall@5 +0.000 (CI -0.150 to +0.150) and MRR -0.036 (CI -0.154 to +0.078). The MRR interval is what keeps it under the revised rule. Hybrid without reranking is much worse on follow-up questions (recall@5 0.357 on the 14 multi-turn questions against 0.714 for dense and for hybrid + rerank). Cross-check with the lenient "any document holding an answer term" rank: dense 0.833, hybrid 0.733, hybrid + rerank 0.900 at recall@5.
- **First pass, same numbers, other rule:** the recall@5 interval [0.000, +0.233] touched zero, so the old rule kept dense. The rule was revised after seeing this (see the method entry above); the numbers did not change, the rule did, and this result is therefore not independent of that choice.
- **Would change if:** a larger or reviewed question set shows hybrid + rerank no better than dense on MRR.

### T6.4 - Result: reranking costs about 30 ms on the GPU and about 600 ms on the CPU (2026-11-03)
- **Chose:** nothing; a measurement for T7.4 (which measures the whole path end to end).
- **Evidence (record `E7-20261103T235630916335Z`, 60 searches after 3 warm-up searches, 15 candidates reranked per search, k=5):** the reranker call takes p50 30.1 ms / p95 42.5 ms on CUDA (torch), p50 631.1 / p95 1063.4 ms on the CPU (torch) and p50 604.2 / p95 975.6 ms on the CPU through ONNX (the path serving uses). A whole search, including embedding the query on CUDA in every path, takes p50 57.7 / p95 79.5 ms on CUDA, p50 682.4 / p95 1085.9 ms on CPU torch and p50 630.0 / p95 1078.5 ms on CPU ONNX. All three paths reach the same recall@5 (0.833), so they agree on ranking. The laptop had 2,043 MiB of GPU memory in use by other work when the stage began; the CPU is a 24-thread AMD64. One machine, one run.
- **Considered:** nothing else was timed. Serving on the CPU would add about 0.6 s to every chat search.
- **Would change if:** another machine, a busier GPU, or a smaller candidate pool gives other numbers.

### T6.4 - Result: structure-aware chunking dropped (method v2, 2026-11-03)
- **Chose:** the plain chunker (base kept). `index-structured` (whole tables, paragraphs kept together) is not used.
- **Evidence (record `E7-20261103T235742665344Z`, against hybrid + rerank):** base 0.600 / 0.800 / 0.833 / 0.933 / 0.712; structured 0.500 / 0.700 / 0.783 / 0.900 / 0.620. Structured minus base: recall@5 -0.050 (CI -0.117 to +0.017), MRR -0.092 (CI -0.185 to -0.002, p=0.180 for first-rank McNemar), so it is worse on MRR and does not help on recall. It is worse on follow-ups (recall@5 0.571 against 0.714) and slightly on single questions (0.848 against 0.870).
- **First pass (against dense):** structured recall@5 0.683, MRR 0.425 against 0.717 and 0.543; the same conclusion.
- **Considered:** keeping it because the questions rarely need a table; no gold question asks about a table, so this test cannot show a benefit for tables, only that chunking differently did not help these questions.
- **Would change if:** questions that need whole tables (spray schedules, rates) are added and shown to benefit.

### T6.4 - Result: context header dropped, borderline (method v2, 2026-11-03)
- **Chose:** no header (base kept).
- **Evidence (record `E7-20261103T235854416532Z`, against hybrid + rerank):** base 0.600 / 0.800 / 0.833 / 0.933 / 0.712; header 0.600 / 0.833 / 0.900 / 0.900 / 0.719. Header minus base: recall@5 +0.067 (CI 0.000 to +0.150, McNemar p=0.219) and MRR +0.007 (CI -0.070 to +0.083). The recall@5 interval has a lower bound of exactly 0.000, so under the rule (the lower bound must be above zero) it is dropped; it is the closest call in E7. By category the header helps single questions (recall@5 0.957 against 0.870) and not follow-ups (MRR 0.518 against 0.617). Recall@10 falls from 0.933 to 0.900.
- **First pass (against dense):** header recall@5 0.767 against 0.717, recall@10 0.917 against 0.817, MRR 0.550 against 0.543: it looked promising there, and on the reranked base the gain shrinks.
- **Considered:** keeping it on the point estimate (lost: the rule is the rule, and a +0.067 on 60 questions with MRR flat is not distinguishable from noise).
- **Would change if:** a larger set puts the recall@5 interval above zero.

### T6.4 - Result: shared-crop pages dropped (method v2, 2026-11-03)
- **Chose:** `crops_from_title` stays off.
- **Evidence (record `E7-20261104T000006095961Z`, against hybrid + rerank):** recall@5 -0.033 (CI -0.083 to 0.000), MRR -0.002 (CI -0.003 to 0.000). The option widens only 9 of 237 documents (a title naming a second crop), so it could change little. Two questions lost a hit. `test_tomato_question_cannot_cite_apple` still holds.
- **First pass (against dense):** no change in recall@5 (0.717 for both) and MRR 0.533 against 0.543.
- **Considered:** a `crops` list in the manifest (see the earlier crops entry).
- **Would change if:** questions asked about a crop that only a mixed page covers are added.

### T6.4 - Result: queries from the conversation and the top-3 diseases dropped (method v2, 2026-11-03)
- **Chose:** the question alone is the query.
- **Evidence (record `E7-20261104T000146288157Z`, against hybrid + rerank; recall@5 / MRR):** base 0.833 / 0.712; history 0.783 / 0.640; diseases 0.817 / 0.653; full (both) 0.750 / 0.618. Every variant is lower, and the MRR intervals of all three sit below zero (history -0.073, CI -0.136 to -0.021; diseases -0.059, CI -0.116 to -0.010; full -0.094, CI -0.164 to -0.033). Adding history lowers recall@5 on follow-ups from 0.714 to 0.500 and adding the diseases leaves it at 0.714 but lowers their MRR from 0.617 to 0.488.
- **First pass (against dense):** history 0.683 / 0.528, diseases 0.700 / 0.507, full 0.667 / 0.456 against base 0.717 / 0.543: also at or below the base.
- **Caveat:** the diagnosis here is simulated from the question text (see the method entry), a best case for the disease terms; it did not help even then. Whole earlier user turns pull the search toward the first question's topic, not the follow-up's.
- **Considered:** a rewritten follow-up query (a language-model step); not built, since the plan asks for none.
- **Would change if:** a real identifier's top-3 and a query-rewriting step are tested.

### T6.4 - Result: score floor 0.73875 for hybrid + rerank (method v2, 2026-11-04)
- **Chose:** floors per mode: dense 0.732844, hybrid 0.806462, hybrid + rerank 0.73875, replacing the single 0.3. They live in `sprout.rag.serving` and are picked on rag_dev, so the rates below are optimistic.
- **Evidence (record `E7-20261104T000358595403Z`; answers when answerable / declines when not in documents / declines for a wrong crop / declines when ambiguous):** hybrid + rerank at 0.73875: 0.817 / 0.857 / 0.714 / 0.385. At the old 0.3 the same mode gives 0.833 / 0.643 / 0.571 / 0.154. Dense at 0.732844: 0.800 / 0.571 / 0.143 / 0.462; without reranking the floor cannot separate the classes (dense cosine scores overlap). Each floor is the rule's choice: the highest decline rate among floors answering at least 80% of answerable questions.
- **Why answerable questions fall under the floor (hybrid + rerank, 11 of 60):** 10 follow-ups and 1 single question; for 7 the gold page was in the top 10 anyway, for 4 it was missing; 4 had no simulated diagnosis; the plant filter was applied to all 11 but only 3 would pass the floor without it. So most losses are short follow-ups scoring low on their own, not the plant filter or the diagnosis.
- **First pass:** a dense floor of 0.801 answered 65% of answerable questions and declined every unanswerable one, and 21 of 60 answerable questions fell under it (13 follow-ups, 8 single). The rule was revised after seeing this.
- **Considered:** a single floor for all modes (lost: the scales differ), and the first pass's balanced rule.
- **Would change if:** the floors do not hold on new questions or on a real identifier's outputs.

### T6.4 - Result: prompt `base+faithful+abstain`; the gains are real, the earlier decline bug had hidden them (method v2, final 2026-11-04)
- **Chose:** `base+faithful+abstain` as the default prompt (`sprout.rag.serving.PROMPT_VERSION`). `ask` was dropped by the rule.
- **Supersedes:** the prompts records `E7-20261104T013649302918Z` and the ones before it (`E7-20261104T004412525032Z`, `E7-20261104T010519096337Z`, `E7-20261104T012545627616Z`, and the first pass `E7-20261103T231812228057Z`). They were made before the decline fixes, so a model reply that declined in prose or ended in `NO_SUPPORT` was counted as an answer and kept its sources. The final record is `E7-20261104T040010301317Z`, from a rerun of every writer and judge stage on the fixed code. The older records are kept untouched and listed in `results/e7/superseded.json`.
- **Correction to the earlier entries:** I wrote that the faithful and abstain gains were not distinguishable from noise, and that the prompt choice was direction-only. That was true of the records then, and it was wrong about the prompts: the decline bug hid the gains. On the fixed code both gains are distinguishable from noise. The earlier statement is superseded; the numbers below are the final ones.
- **Evidence (record `E7-20261104T040010301317Z`; all 101 dev questions reach the writer with no score floor, except `served`; writer `qwen3.5:4b-q4_K_M`, judge `gemma4:e4b-it-qat`; faithfulness / answers when answerable / declines when not in documents / declines when ambiguous; graded answers):**
  - base 0.743 / 0.983 / 0.357 / 0.000; 88
  - faithful 0.821 / 0.983 / 0.643 / 0.000; 79
  - abstain 0.825 / 0.950 / 1.000 / 0.000; 71
  - ask 0.859 / 0.867 / 1.000 / 0.692; 58
  - combined (faithful + abstain) 0.851 / 0.967 / 1.000 / 0.077; 71
  - served (chosen prompt and floors) 0.877 / 0.783 / 1.000 / 0.462; 54
  - 421 graded answers across the six variants, so T6.6's 200 are available.
- **Gains over base, with intervals:**
  - **faithful:** faithfulness +0.077 (95% CI +0.014 to +0.140), distinguishable from noise; answers unchanged.
  - **abstain:** declines when not in documents +0.643 (CI +0.357 to +0.857, McNemar p=0.004, n=14), distinguishable from noise; answers when answerable -0.033 (CI -0.083 to 0.000, p=0.500).
  - **ask:** declines when ambiguous +0.692 (CI +0.462 to +0.923, p=0.004, n=13) is real, but answers when answerable fall -0.117 (CI -0.200 to -0.050, p=0.016); the rule (answers may fall by at most 0.05) drops it. Its faithfulness (0.859) is on fewer answers (58) and is not comparable.
  - **combined:** the table's target gains are faithfulness +0.108 and unanswerable declines +0.643 over base, answers -0.017: kept. (The earlier "+0.000" for combined was a display error, fixed.)
- **What the bug hid:** on the same questions the first fixed-code run reads base declines on not-in-documents 0.357 against the earlier 0.000, abstain 1.000 against 0.143, and the faithful gain +0.077 (CI +0.014 to +0.140) against +0.058 (CI -0.002 to +0.119). The earlier records treated a prose decline as an answer, which also inflated the wrong-crop and not-in-documents "answered" counts. The offline counts I made from stored text (base 5 and abstain 12 not-in-documents flips) match the rerun.
- **Served variant:** answers when answerable 0.783 is just under the 0.80 the floor was chosen to keep (the floor alone answers 0.817, 49 of 60): the writer itself declines 2 of the answerable questions that passed the floor, by `NO_SUPPORT` or a prose decline. Served makes 62 writer calls; 39 rows have no call because no passage passed the floor. Declines when not in documents 1.000 and when ambiguous 0.462, optimistic since the floor and prompt were picked on rag_dev.
- **Device and time:** writer `cuda`, judge `cuda`; the writer took 262 to 347 s per full variant of 101 calls.
- **Considered:** `ask` in the default prompt (lost on answers); keeping `base` (lost: both kept rules are distinguishable gains).
- **Would change if:** T6.6's second grader disagrees with the local judge on the claims in the blind file (faithfulness is a local-judge value), or a larger set shows the answer cost of abstain to be larger.

### T6.4 - Result: Sonnet against the local writer under the chosen prompt, 100 dev questions (method v2, final 2026-11-04)
- **Chose:** reported as a comparison only; the local writer stays the default.
- **Evidence (record `E7-20261104T040723229114Z`; both writers use `base+faithful+abstain` with the chosen floors over the first 100 dev questions, judged by `gemma4:e4b-it-qat`, never by Claude):** local writer faithfulness 0.877, answers when answerable 0.783, declines when not in documents 1.000, wrong crop 1.000, ambiguous 0.417; Sonnet faithfulness 0.798, answers 0.767, declines 1.000, wrong crop 1.000, ambiguous 0.667. Each writer made 62 calls (counted from the saved rows); the other 38 questions had no passage above the floor.
- **Supersedes:** `E7-20261104T014326302860Z` (local 0.851, Sonnet 0.785), made before the decline fixes, and the two base-prompt comparisons `E7-20261104T011437447083Z` (call count read 0) and `E7-20261104T012205498018Z` (local 0.799, Sonnet 0.671). All three are kept untouched and listed in `results/e7/superseded.json`.
- **Calls:** this record counts 62 calls, from rows. `results/claude_calls.json` reads 187: one smoke call and three runs of 62 (base prompt, chosen prompt before the decline fixes, chosen prompt on the fixed code).
- **Not interpreted:** the gap in faithfulness is a number from a local judge, with no interval here. T6.6 will have Sonnet check at least 200 of the judge's graded answers, including whether the judge favours the local writer's style.
- **Would change if:** the person changes the model or the cap, or T6.6 shows the judge is biased.

### T6.4 - Serving uses the chosen setup by default (2026-11-04)
- **Chose:** `sprout/rag/serving.py` holds the retrieval mode (hybrid), reranking (on), k (5), the query mode (question), the floors and the prompt. `AnswerConfig` and `deploy/wsgi.py` read it, so `answer()` and the app use it with no arguments. `knowledge/index` stays the index. Reranking costs about 600 ms on the CPU and 30 ms on the GPU per search; T7.4 measures it end to end.
- **Evidence:** `tests/test_serving.py` (defaults, an `answer()` call with no config, agreement with `results/e7/v2/chosen.json`, and the app's retriever config).
- **Would change if:** serving cannot afford 600 ms per search on the CPU; the next step would be dropping reranking and using the dense floor, which separates far less.

### T6.4 - Safety fixes after verification attempt 1: the label note no longer depends on the writer's words (2026-11-04)
The verifier failed T6.4 on the chemical and disposal rule. Its `CHEMICAL` word list missed bleach, chlorine, TSP, neem oil, sulfur and burning or destroying plants, and any text containing "product label" skipped the note, so local rules went unmentioned (saved answers `supported-039` and `supported-046` give bleach and TSP mixes with no note).
- **Chose, the note:** every answer that gives advice (grounded, or ungrounded and labeled as general advice) ends with the fixed sentence "Check the product label and your local rules before using any chemical or disposing of plant material.", added by code and never conditional on the text. If the writer's prose already says it, or copies the sentence, the fixed note still appears, once. Declines, questions back, outages and failed searches do not get it. It is slightly over-inclusive, which is the safe side for this rule.
- **Considered:** a wider word list (lost: it always misses the next product, and the fault was the approach, not the list); requiring both "label" and "local" in the text (lost: still text matching).
- **E7 numbers and the note:** unchanged by the note. `split_claims` skips the new wording and the old one (`LEGACY_LABEL_NOTE`), so the claims judged, and therefore faithfulness, are the same; `test_the_note_wordings_are_skipped_by_the_claim_splitter_so_e7_numbers_do_not_change` checks it without a rerun.
- **Other fixes, each with tests in `tests/test_answer.py`:**
  - A `NO_SUPPORT` token anywhere in the reply (also `no_support`, `NO-SUPPORT`) is a decline; the first version only read the start of the reply, so a trailing token leaked into the user text.
  - A prose decline in the first sentence ("The provided evidence does not contain...", "I cannot find...", "There is no information...") drops all citations and returns the fixed no-document text. Only the first sentence is read, so a useful answer that later notes a gap ("The passages do not give a spray rate") keeps its sources. This is a word pattern, used only on the safe side (it removes sources and the model's prose; it never adds a claim).
  - With no diagnosis and no crop, `answer()` reads the plant from the question and the user's own earlier turns (never the assistant's), with the same `detect_crop` the E7 harness uses (now in `sprout.rag.query`). A tomato question can no longer cite a potato passage. Chosen over declining to cite across crops because it is smaller and is the behaviour E7 already measured. A question with no recognisable plant still searches every crop.
  - The judge reads a verdict strictly (`SUPPORTED` as the first word; `NOT SUPPORTED`, `UNSUPPORTED`, `SUPPORTED_NOT` and odd text count as not supported).
  - The blind file for T6.6 no longer shows the writer's system name (`base`, `served`, `claude_sonnet`) or the judge's name; rows carry an `item` hash to match them back. The verifier noted the name could bias a grader. Files written earlier still show it; the newly named files will not.
- **Do the fixes change a recorded number? Yes, for the prompts and Sonnet records.** Re-reading the saved answer texts with the new rules, answered rows that become declines (answerable / wrong crop / not in documents / ambiguous): base 1 / 6 / 5 / 0; faithful 1 / 12 / 9 / 0; abstain 3 / 5 / 12 / 0 (declines on not-in-documents would go from 0.143 to 1.000); ask 0 / 0 / 1 / 0; combined 2 / 6 / 8 / 1; final served 1 / 5 / 1 / 1; Sonnet in the final comparison 0 / 0 / 0 / 2. Model prose declines were counted as answers and kept their citations, which is why the base prompt showed 0.000 declines on not-in-documents questions. **Update: the reruns are done** (`E7-20261104T040010301317Z` and `E7-20261104T040723229114Z`; see the results entries above), and the abstain numbers moved as predicted. Before the rerun, `E7-20261104T013649302918Z` (prompts, with the served variant) and `E7-20261104T014326302860Z` (Sonnet comparison) needed the generation, judging and record stages rerun; the abstain rule probably looks much better than the table in the earlier entry says. The flips were counted from stored text, not rerun. Rows made by older answer code are not reused (`ANSWER_VERSION` is in the generation key); a rerun of the writer stages, and 62 new Sonnet calls for the comparison, are needed, or the stored rows can be reprocessed offline (not built).
- **Superseded records are marked in the data:** `results/e7/superseded.json` lists each superseded record with its replacement and reason (the first pass, the base-prompt served rows, the comparison records, the call-count error) and, under `needs_rerun`, the two records above. Old records are not edited. `tools/run_e7.py status` prints the counts; `sprout.experiments.e7_rag.superseded_records()` is the reader reports use; a test checks every id names a real record.
- **Safety behaviours that depend on matching words in model text** (plan and `CLAUDE.md`), each tried with wrong or odd wordings in tests:
  1. chemical and disposal note: now unconditional (above);
  2. citations: only `[n]` markers naming a shown passage count. `[1, 2]`, `(1)`, `[Passage 1]` and `[1-2]` give no source, so the answer is labeled as having no supporting document (safe side);
  3. decline: the `NO_SUPPORT` token anywhere, and a first-sentence prose pattern (above);
  4. asking for details: only a line starting with `NEED:` (any case). `NEEDS: crop` is read as an ordinary answer, labeled as unsourced if it has no citation;
  5. tag injection: `</evidence>` and the other prompt tags are removed case-insensitively, with spaces; look-alike characters are not handled;
  6. the faithfulness judge's verdict: read strictly (above);
  7. the claim splitter skips the system's own notes by their start; an answer beginning with those exact words would not be judged, which does not affect safety.
  Behaviours that do not read model text: the uncertainty prefix (confidence and status), the "no supporting document" label (citation presence), outage and search-failure messages, the web layer's label note (shown whenever passages are).
- **Not fixed (named for later tasks):** floors were chosen at k=10 while serving uses k=5; the local judge missed a wrong-disease transfer (`supported-031`) and counts decline sentences as claims; unconfirmed-source citations will appear often.
- **Would change if:** a real model shows another habit for declining or citing that these patterns miss; the fixed note and the "no source" label are the safety net.
### T6.4 - Not changed: navigation text in documents 241, 203, 239
- **Chose:** left as they are. They are measured only through the stages (a navigation chunk that ranks high shows up as a lower recall); no cleaner was built, since the plan asks for none and the structured chunker's heading and table handling is the intended fix to test first.
- **Would change if:** the answer reads show navigation text cited as evidence.

### T6.6 - E14: how the local judge is checked (code choices, 2026-11-04; results are filled in after the run)
- **Chose, the sample:** the final E7 prompts judgments (`E7-prompts-v2-1e90d37a`, from record `E7-20261104T040010301317Z`: 421 graded answers, 2,326 claims, all `rag_dev`). The sample is all 54 `served` answers (`base+faithful+abstain`, the app's configuration and the one behind 0.877) plus a seeded random draw (seed 14, `random.Random`) from the other 367 to reach 220 answers, about 1,228 claims. `served` is also reported as its own stratum, so its numbers do not depend on the draw. `supported-031` is in the served set. The sample is fixed by the seed and the sorted list, so a rerun draws the same answers.
- **Chose, what is left out:** the Sonnet-written answers (`E7-claude-v2-*`). The plan asks about the local judge's reliability on the chatbot answers it graded in T6.4 for the app's own writer; the Sonnet-writer answers measure a different writer, and Sonnet grading its own prose would mix in self-preference. Judgment files of superseded records are refused: the config names the E7 record and the files' sha256, and `load_source` checks the record produced them and is not in `results/e7/superseded.json`. Any answer outside `rag_dev` stops the run.
- **Chose, what Sonnet sees:** the blind file only (no verdict, reply, judge name or system name), through the same `judge_claim` and `judge.txt` rubric the local judge used, one call per claim, the claim checked against the passages it cites (all shown when it cites none). A test checks that no local reply, judge name, system name or the words verdict and judge_reply reach the prompt. Text is cleaned of control characters and lone surrogates before sending so the prompt is valid JSON and UTF-8; ordinary text is unchanged.
- **Chose, reading Sonnet's reply:** strictly. Only `SUPPORTED` and `NOT SUPPORTED` (any case or punctuation) count; any other reply, such as an explanation, has no grade and the claim is left out of every agreement number and counted in `claims_not_graded`. The local judge's reader counts such a reply as not supported, which would hide the difference. Considered: retrying a bad reply. Lost: it spends calls and the count of unusable replies is itself a finding.
- **Chose, the numbers:** per claim, because the claim is what both graders judged. Agreement and Cohen's kappa overall and for `served`, with 95% bootstrap intervals that resample whole questions (the same question under several prompts moves together; 10,000 resamples, seed 13, as everywhere). Direction: each claim the graders split on is counted once, as local-supported-but-Sonnet-not (the local judge is kinder) or the reverse. Also the share supported by each grader, and for `served` the number of answers with at least one unsupported claim under each grader, to set beside the hand read of about 7 of 20. Served faithfulness under Sonnet's grades is `served_sonnet_supported`; it is a measure of the judge, not an E7 number. Kappa is undefined when one grader gives only one grade; it is then reported as undefined.
- **Chose, calls:** `ClaudeCodeClient(model="sonnet")` and the existing counter; two calls at a time; each reply saved (fsynced) as it arrives, keyed by answer and claim; a rerun asks only for claims without a saved reply; the run is refused before any call if the needed calls would pass `max_calls` (1,800) in `configs/e14.json` or the shared cap. A failed call stops new calls and keeps what arrived. Calls in flight when it fails still finish, so a few more than the failure point can be saved.
- **Chose, the record:** one system, `sample`, whose rows are graded answers (not split items), so `check_run` has an E14 branch: it checks rows are `rag_dev` questions, local grades equal the named judgment file (sha256 in the config), `claude_calls` equals the calls the rows record, and recomputes every metric and interval with `sprout.eval.agreement`, the module E14 also uses. Considered: one record row per question. Lost: one question has several answers, and the unit graded is the answer.
- **Evidence:** `tests/test_e14.py` (worked kappa example 0.4, direction, resume, cap, blind prompt, tampered rows).
- **Would change if:** agreement is poor, then the plan's proposal is to use Sonnet as the judge for final numbers, which the coordinator decides. The sample size or the strict reading would change if many replies come back unusable.
- **Results, record `E14-20261104T085533959094Z` (dev only, a measure of the judge, not an E7 number):** 220 answers, 1,228 claims, 1,228 Sonnet calls, 0 unusable replies. All sampled: agreement 0.874 (95% 0.849 to 0.897), kappa 0.630 (0.576 to 0.684). Served: agreement 0.901 (0.871 to 0.930), kappa 0.653 (0.553 to 0.739). Direction: local supported but Sonnet not 141, the reverse 14; served 28 and 1. Supported share on served: local 0.877 (0.834 to 0.917), Sonnet 0.785 (0.729 to 0.836). Served answers with at least one unsupported claim: local 24 of 54, Sonnet 37 of 54.
- **Reading:** the disagreements lean one way, so the local judge is the kinder grader by about 9 points on served answers. This run doesn't show which grader is right on the split claims: Sonnet is a second grader, not the truth. Hand reads had about 7 of 20 served answers with an unsupported claim, lower than either grader's share, on a small sample. On `supported-031`, the wrong-disease transfer sits in two claims drawn from a passage about a different disease. Sonnet passed one and failed the other only because it cited the wrong passage, not because of the disease. The judge's prompt shows no question or diagnosis, so neither grader can see this kind of error. A blind read by the verifier of the split claims (`docs/e14-adjudication.md`, one reader, most calls close): served 29, local right 15, Sonnet right 14; a random 20 non-served where local said supported and Sonnet not, local right 6, Sonnet right 14. Of Sonnet's 48 no's read, 21 were overly literal and 27 real catches, several of them citation faults. So the local judge errs kind and Sonnet errs literal, and neither number is the truth.

### T8.1 - Two judges and a wrong-disease check on final answer numbers (2026-11-04)
- **Decided by the person ("both yes", relayed by the coordinator):** (1) final answer-quality numbers report both judges, local as the upper figure and Sonnet as the conservative one, neither called accuracy. The rule that Claude is never the judge changes only for this second figure and still holds for E7. (2) T8.1 adds a wrong-disease check: the grader sees the question and diagnosis and judges whether the advice fits the right disease and plant, about one Sonnet call per answer. It runs on `rag_dev` first with its rate reported, its error is estimated by a blind read of a dev sample in both directions, then its prompt is frozen with commit and hash recorded, and it runs on `rag_test` once with no tuning afterwards.
- **Considered:** keeping the local judge alone and reporting E14's gap as a caveat. Lost: the final figure would be known to flatter. Replacing the local judge with Sonnet. Lost: Sonnet errs literal (21 of its 48 no's read were over-literal), so it isn't the truth either.
- **Evidence:** record `E14-20261104T085533959094Z` (local kinder 141 to 14; served supported 0.877 local and 0.785 Sonnet, dev), `docs/e14-adjudication.md`, and the `supported-031` walkthrough, where a per-claim prompt with no question or diagnosis can't see a wrong-disease transfer.
- **Would change if:** the person changes it. The blind read could show the wrong-disease check is too unreliable to report, and that would be reported rather than tuned on `rag_test`.
- **Cost:** a weakened faithfulness resume bullet is possible; the person accepted that risk by approving this.

### T8.1 - Code choices for the final test (prepared 2026-11-04, before E12 decides and before any test split is read)
Built with fakes and tiny fixtures only; no test photo, question or prediction was opened, and the real stage commands have not been run.
- **Records:** one record per test split. `lab_test` gets a full record and a second record that is a declared subset (`shared_classes`) of the same predictions, same commit and config, so `check_run`'s test-once rule counts them as one use. `field_test` gets one record on its cleaned rows (`subset: clean:N`, N from `configs/e9.json`), and only cleaned rows reach a model; the record states rows before and after and the removals by reason and source from `datasets/label_exclusions.csv`. Each record holds the baseline, the improved model, XGBoost, DINOv2, the quantized model, the alternate candidate, and calibrated copies of the improved, alternate and quantized models. "Without abstention" is the plain system; "with abstention" is the calibrated copy, which also reports `coverage@t`, `selective_accuracy@t` and `confident_error_rate@t` at E5's threshold `t`. All image metrics are the standard set from T3.5, so `check_run` recomputes them.
- **Which model is which:** the improved model, its hash and its class order come from `results/e12/decision.json` (read by path; the stage refuses if it is missing). Each candidate's calibration file is named in the config (see the primary-model entry below). The quantized model is the export `results/e6/export.json` names for the variant in the config, and the export must have been made from the chosen checkpoint (hash chain). The baseline is E1, XGBoost is E3's booster, DINOv2 is E4's `dinov2_head` (the joint-trained head), not E3's lab-trained one; the config can point elsewhere. The config's quantized `variant` (`onnx_dynamic`) is a placeholder to set from E6's final numbers before it is committed.
- **Test-once:** a stage refuses unless the configs are tracked and equal to HEAD, the tree is clean, and no record has ever existed on that split (committed history and working tree, via `check_run`'s own list). A later run needs `--override-note`, which goes into the record's config, which is what `check_run` demands of a later record.
- **Label signature (changed 2026-11-04 by the coordinator and the person):** E9 records hold no signature. `labels_reviewed` is written false at run time, and `check_run` flags an E9 record whose two copies of it disagree; `claims check` ignores the field for E9 and reads the signature file instead. The person signs each split (`lab_test`, `field_test`, `rag_test`) in words in the tracked file `orchestration/label_signoff.json` (all three start `signed: false`, with `by`, `at` and `words`); `claims check` reads it for E9 records and refuses a claim on a split that is not signed true with all three fields. Signing is a commit of that file: it changes no record and needs no rerun of a once-only test. H2's status is not read anywhere: it is `done`, but its note says the photo labels are not signed. Every E9 record, its notes and the reports say where the signature lives. The plan's test `test_labels_reviewed_is_false_before_h2_and_true_after` is replaced by `test_a_false_signoff_row_refuses_a_claim`, `test_a_signed_row_passes_the_claim_without_touching_the_record` and `test_signing_needs_no_rerun_and_adds_no_record`, plus `test_an_edited_labels_reviewed_is_caught_and_claims_ignore_it_for_e9`. Considered: reading H2 (rejected by the person, see above); keeping the flag in the record and rerunning after sign-off (the test splits are once-only).
- **Primary model and its neighbour:** `results/e12/decision.json`'s `chosen_model` is the primary (system `improved`); E12's gates are pending, so today it names E4's `convnext_joint`. The other candidate in `configs/e9.json` is scored beside it as `alternate`, on the same rows, with every measure, its own calibration and abstention, and a paired comparison, whichever is chosen. The record's `config.primary` and `config.alternate` say which is which, and the model labels say "primary" and "alternate". Each candidate has its own calibration file (E4's `results/e5/chosen.json`; E12's `results/e12/e5_refit/chosen.json`); a calibration file that names its model (`model_sha256`) must name the one scored, and when the decision says E5 was refit the primary's file must name it. D1 of the phase 4 review will add `model_sha256` to E5's file in general. The quantized model is of the primary.
- **One basis (phase 4 review D4):** every E9 image number is on cleaned rows for `field_test` (`config.basis` states "cleaned rows: N of M kept") and on all `lab_test` photos for `lab_test` (the cleaning covers real photos only). `image_report` and `report` start every numbered line with its basis, and the rag_test report says no photo cleaning applies to its questions.
- **Frozen records (closed gap):** `check_run`'s rag_test branch now checks that the frozen config's `dev_record` and `blind_read_record` exist, are E9 `rag_dev` records of the right stage, were finished before `frozen_at`, name the frozen prompt's hash, and that the blind read belongs to that dev run.
- **Two judges:** the local judge's per-claim verdicts come from E7's `run_judging`; Sonnet grades the same claims through E14's `grade` and `judge_claim`, with the same rubric, from the blind view (no verdict, judge or system name). The record stores them as shares of claims judged supported, named `claims_supported_by_local_judge` (upper figure) and `claims_supported_by_sonnet` (conservative figure). The word accuracy is in none of the record, report or sidecar, and `check_run` flags a rag_test record that uses it.
- **Wrong-disease check:** one Sonnet call per answer that has at least one claim (the same answers the faithfulness judges grade). It sees the question, the diagnosis and the answer, and no passages or history. Its reply is read as exactly FITS or WRONG, and anything else is counted as an unusable reply, not guessed. The diagnosis for a question without a photo is the text the writer was given: E7's simulated diagnosis (the plant's class that the question names, never the gold label), written by `answer.diagnosis_summary`; with none, "No photo diagnosis is available", plus the plant when the question states or names one (the question's `crop` field is the only plant field the question set has; there is no disease field). `rag_dev` and `rag_test` use the same function. Considered: showing passages too (turns it into a second faithfulness check), one call per claim (about five times the cost), history in the question (not asked; multi-turn questions are judged on their last turn, a limit to report).
- **Dev run, blind read, freeze:** `wrong-disease-dev` runs on the served answers in the final E7 judgments file (`E7-prompts-v2-1e90d37a`, hash-checked through E14's loader) and writes an E9 `rag_dev` record with the rate and a Wilson interval. `blind-sample` draws, with a seed, up to 25 answers from each of the grader's two calls and shuffles them; the file holds only question, diagnosis and answer. `unblind` records how often the reader disagrees after a WRONG call and after a FITS call, with intervals, and an estimate for all dev answers weighted by how many fell in each call. The reader is one reader, not truth. `freeze` writes `configs/e9_wrong_disease.json` with the prompt's sha256, the sha256 of the messages built for a fixed input (so a layout change shows too), and the prompt file's last commit; it refuses if the prompt changed since the dev run. Every rag_test stage refuses unless that file and the prompt are committed, the tree is clean, and both hashes still match. Considered: hashing only the prompt file (misses a change to how it is shown); keeping a key file for the blind sample (the sample is recomputed from the seed instead, so the grader's calls are never in a second file).
- **Sonnet calls:** two at a time, each saved as it arrives, a rerun asks only for what is missing. Claims and wrong-disease calls are counted together against `max_sonnet_calls` and the shared cap before the first call, so a run that would pass either makes none.
- **`check_run`:** E9 image records use the generic checks. E9 `rag_dev` stage records are recomputed from their rows with `sprout.eval.answer_checks` (and their question and answer text from the hash-checked source file); the blind read must be the seeded sample of the named dev record with the grader's own calls. The rag_test record's claims must be the answer text's own claims, each verdict must follow from its reply, the judged samples must equal the rows' own, `claude_calls` must equal the rows', and the prompt hash must equal the file at the record's commit and the frozen config's. Tamper tests cover each.
- **GPU:** the code never reserves the GPU. Retrieval, the writer and the local judge each record where they ran (the model must sit in GPU memory), and the record refuses to be written without it. The person runs those stages under `ctl gpu-acquire`.
- **Evidence:** `tests/test_e9.py` (the tamper tests for each check above), `tests/test_e14.py` unchanged and passing.
- **Would change if:** E12's `decision.json` uses other key names (one function, `load_decision`); the blind read shows the check is too unreliable to report, which is then reported rather than tuned; the person changes where the signature lives.

### T0.1 - Run-plan stop rules (2026-11-02 to 2026-11-03)
These are workflow changes with no plan task of their own, filed under T0.1.
- **GPU lacks memory:** first, stop the whole run (`d5c0030`). Replaced on 2026-11-03 (`3493a8c`): tasks always start, CPU work never waits, GPU jobs share the GPU while reserved memory fits, and latency runs and fine-tuning run alone. The person's rule; further reason not recorded.
- **Phase review defects:** at most two fix loops, then leftover defects go to the plan task that first needs the fix, under a `From phase review:` line, and the run continues (`19682fe`). Before that, the run stopped and reported. The run still stops if a defect could put a false number on record with no later task to own it. Example: the phase 3 review's gaps were carried into T4.1 (`ac07ecd`) and fixed there.
- **Blocked tasks:** a task blocks after 3 failed attempts; two blocked tasks in one run stop the whole run, since something shared is probably wrong.
- **Parallel tracks:** up to 3 tasks in git worktrees, landed one at a time (`58138a9`).
- **Evidence:** `.claude/skills/run-plan/SKILL.md`, `CLAUDE.md`, the commits above.

### T5.1 - Torch-free preprocessing and runtime (prepared 2026-11-03, before E12's model is chosen)
- **Chose:** `sprout.preprocess_np.preprocess_array` does the resize, divide by 255 and normalize in float32 with numpy; `sprout.preprocess.preprocess` wraps it with `torch.from_numpy`. `OnnxModel` and `quantize` in `sprout.serve.export` import no torch (a test runs both in a fresh process and checks `torch` is not loaded); only the export functions import it, inside the function.
- **Considered:** a second copy of the preprocessing for serving, which could drift from evaluation. Rejected: T5.2 must use the same preprocessing exactly.
- **Evidence:** `test_numpy_preprocess_is_identical_to_the_torch_one` (bit-identical, RGB, RGBA and grey inputs); `test_running_and_quantizing_a_model_loads_no_torch`; `tests/test_no_torch_imports.py`.
- **Would change if:** a measured difference between the two paths, which the identity test would catch.

### T5.1 - Class map and preprocessing travel inside the model file (prepared 2026-11-03)
- **Chose:** `export_onnx` writes the class map, preprocessing settings and model name into the ONNX metadata; `quantize` copies them to the quantized file; `OnnxModel` refuses a file without a class map and exposes `class_map` and `class_names`.
- **Considered:** a side file next to the model. Rejected: a model file and its class order can then be separated.
- **Evidence:** `test_class_map_travels_with_the_model` (plain, dynamic and static files).
- **Would change if:** the serving image needs a different container for the class order.

### T5.1 - Legacy TorchScript exporter (prepared 2026-11-03)
- **Chose:** `torch.onnx.export(..., dynamo=False)` at opset 17 with an open batch axis.
- **Considered:** torch's newer default exporter. Not used because it needs another package (`onnxscript`) and the legacy one exports the models here and matches torch in the tests. It prints a deprecation warning.
- **Would change if:** a future torch removes the legacy exporter, or a DINOv2 model will not export with it.

### T5.1 - Calibration photos come from the plan's `lab_dev`, never `field_dev` or a test split (prepared 2026-11-03)
- **Chose:** `configs/e6.json` sets calibration to 100 seeded `lab_dev` photos, as the plan says. The code accepts only `lab_dev`, `lab_train` or `field_train`; `field_dev` is refused because E6 scores it, and test splits because they stay unseen.
- **Open concern:** lab photos look unlike phone photos, so static quantization ranges set on them may fit real photos worse. `field_train` is the safer calibration set (real photos, never scored). Changing it is one config line and is the person's or coordinator's call: the plan names `lab_dev`.
- **Would change if:** static quantization on cleaned `field_dev` scores differently with `field_train` calibration than with `lab_dev`.

### T5.1 - Latency is its own stage, and the reference model runs on the CPU unless config says otherwise (prepared 2026-11-03)
- **Chose:** four stages, `export`, `probs`, `latency`, `record`. `latency` times one photo at a time on the CPU for PyTorch and each ONNX file, threads fixed by config, and saves the CPU name and thread count; it is meant to be run alone. `probs` runs ONNX on the CPU and PyTorch on `torch_device` (default `cpu`) and saves the device used. `export` and `probs` resume after a crash and discard outputs when the checkpoint's checksum changes.
- **Considered:** timing inside the scoring stage. Rejected: scoring may use the GPU or share the machine, and a deployment latency number must be measured where it will run, with nothing else running.
- **Would change if:** the person wants latency under load; that is E8.

### T5.1 - Crop condition read from config (prepared 2026-11-03)
- **Chose:** `crop_kept` in `configs/e6.json` is false because E4 kept no crop (T4.1, "Crop kept out of serving"). The record states that no segmentation model was exported or timed. If it is set true the stages refuse, since the combined export and timing path is not built.
- **Would change if:** a later task keeps a crop.

### T4.8 - Plant Pathology 2021 used from the third-party mirror, single-label scab, healthy and rust only (2026-11-03)
- **Chose:** fetch `timm/plant-pathology-2021` (16.1 GB, public, no login) and use only rows whose single label is `scab`, `healthy` or `rust`, mapped to `Apple___Apple_scab`, `Apple___healthy`, `Apple___Cedar_apple_rust`. Multi-label rows (1,355) and the single labels `complex`, `frog_eye_leaf_spot`, `powdery_mildew` (5,967) are excluded with reasons in `datasets/class_map_extra.csv`.
- **Considered:** using multi-label rows by their first label, which would file a photo with several diseases under one; mapping `complex` or `frog_eye_leaf_spot` to a lab class, which has none. Not using the source, as T4.5 first decided; the person approved the fetch on 2026-11-03 through the coordinator.
- **Evidence:** `docs/extra-sources.md` (T4.8 section), `docs/datasets.md`. The `rust` to cedar apple rust mapping is an inference from the name; 24 viewed samples looked like it (20 clear lesions, 4 only tiny spots). The mirror's `apache-2.0` is the uploader's tag; the Kaggle and paper terms are unconfirmed, so weights trained on it need a check before publishing.
- **Would change if:** the original terms forbid this use; or E12 or a closer look shows the rust class is another apple rust.

### T4.8 - Established sources first in duplicates and the shared cap of 200 (2026-11-03)
- **Chose:** rows from the T4.5 sources (PlantWild v1 and v2) win duplicate groups and come before any new-source row in the per-class cap; new rows fill only the room left, by sha1 order among themselves. The cap stays 200. Result: no T4.5 row moved (3,347 of 3,347 kept, same label and sha1), and the new source added 137 rows (scab 106, rust 31, healthy 0).
- **Considered:** pooling all rows and keeping the first 200 by sha1, which would cut PlantWild rows already chosen and, with about 7,440 usable Plant Pathology rows against 7,300 PlantWild rows after checks, would make one orchard most of the apple classes; raising the cap, which changes a number fixed before any E12 result and is the person's call.
- **Evidence:** `tests/test_extra_sources.py::test_a_new_source_fills_only_the_room_the_cap_leaves`; row-by-row comparison with the T4.5 manifest; the per-step difference table in `docs/extra-sources.md` closes exactly (11,310 - 321 - 3,530 - 3 - 16 - 7,303 = 137).
- **Would change if:** E12 shows the apple classes starved, or the person raises the cap (then the 7,303 cut rows are the pool).

### T4.8 - Copy checks unchanged for the new source, over-removal reported and not fixed (2026-11-03)
- **Chose:** the T4.5 correlation (0.75) and crop-aware rule applied as they are. The crop step dropped 3,530 of 10,989 new rows (32%), almost all by class-token cosine of at least 0.80 with an apple `field_dev` or `field_test` photo.
- **Considered:** loosening the rule for apple, which would be tuning on the new data after seeing it and would risk leaked evaluation photos; the task says to report this cost.
- **Evidence:** a seeded sample of 400 rows: 36.8% flagged, best inlier count 28 (the 30-inlier clause never fired), and the 12 highest-cosine flagged pairs viewed were all different leaves.
- **Would change if:** a pair-specific check (inliers, not class-token cosine) were wanted for apple; that would be a new task.

### T5.2 - The identifier reads E5's chosen.json, applies the gate before the abstention threshold (2026-11-04)
- **Chose:** `Identifier(model_path, calibration_path)` takes the file `results/e5/chosen.json` has (`temperature`, `abstain_threshold`, `gate.settings`), so no new file format and no number is written in code. Probabilities are calibrated as E5 does (log with a 1e-12 floor, then temperature), then `gate.accepts` gives `not_plant`, then `decide` gives `answered` or `unsure`. The model path is also an argument; both depend on E12 and T5.1 outputs.
- **Considered:** a separate calibration format (a second thing to keep in step with E5); abstaining before the gate (a non-plant photo would be told to "retake the leaf photo" instead of "this is not a plant").
- **Evidence:** `tests/test_identify.py` with a tiny exported ONNX model; `FLOOR` is asserted equal to E5's.
- **Would change if:** E5's selection puts the gate after abstention, or the chosen file's shape changes.

### T5.2 - Input limits and edge cases (2026-11-04)
- **Chose:** reject files over 20 MB and images over 60 megapixels from the header before `load()`; PIL's decompression-bomb warning and error also count as too large; any decode failure returns `invalid` with no exception. EXIF orientation is applied with `ImageOps.exif_transpose`. When `decide` returns no candidates (a flat spread), or answers with none above 10%, the reply is `unsure` with `top=None`. `unsure` carries fixed photo tips.
- **Considered:** a side limit instead of a pixel count (rejects odd panoramas that are small in pixels); `answered` with `top=None` (breaks the contract).
- **Evidence:** `test_oversize_image_is_rejected_before_decode_fully` (`load` is patched to fail), `test_flat_distribution_is_unsure_without_candidates`.
- **Would change if:** real phone photos in T7.x exceed 60 MP, or the app wants a different message per failure.
