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

### T0.1 - Run-plan stop rules (2026-11-02 to 2026-11-03)
These are workflow changes with no plan task of their own, filed under T0.1.
- **GPU lacks memory:** first, stop the whole run (`d5c0030`). Replaced on 2026-11-03 (`3493a8c`): tasks always start, CPU work never waits, GPU jobs share the GPU while reserved memory fits, and latency runs and fine-tuning run alone. The person's rule; further reason not recorded.
- **Phase review defects:** at most two fix loops, then leftover defects go to the plan task that first needs the fix, under a `From phase review:` line, and the run continues (`19682fe`). Before that, the run stopped and reported. The run still stops if a defect could put a false number on record with no later task to own it. Example: the phase 3 review's gaps were carried into T4.1 (`ac07ecd`) and fixed there.
- **Blocked tasks:** a task blocks after 3 failed attempts; two blocked tasks in one run stop the whole run, since something shared is probably wrong.
- **Parallel tracks:** up to 3 tasks in git worktrees, landed one at a time (`58138a9`).
- **Evidence:** `.claude/skills/run-plan/SKILL.md`, `CLAUDE.md`, the commits above.

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
