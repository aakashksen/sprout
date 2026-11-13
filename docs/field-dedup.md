# Field-set near-duplicate handling (T1.2)

Why this exists: PlantDoc and PlantSeg are scraped from the web, so one photo appears several times (resized, cropped, watermarked, mirrored, sometimes under different labels). Copies in two splits would leak.

## Rule

1. Group against lab and within field with `group_near_duplicates` (pHash, max distance 4; unchanged, so the lab manifest hash `7eda8dbf1b7dbbec8aea577ccd3e9f439817f5f734f62be823756c8c2aaa1c30` is unchanged).
2. Hashes only propose pairs (pHash or dHash within 12 bits over the 8 flips and rotations). A pair is merged only if `same_scene_score` confirms it. That score is the best correlation of two 64x64 gray images over the 8 symmetries and 85% and 70% crops, counting only pixels that are not white in both images (so blank backgrounds on isolated-leaf stock photos don't match). Merge if score >= 0.88, or score >= 0.80 and dHash <= 10.
3. Any group holding more than one lab label is removed whole and listed in `datasets/manifests/field_conflicts.csv`. No winner is chosen.

This replaces the earlier rule (single-link on dHash <= 6), which joined unrelated images through chains of loose hash matches. Under the new rule a hash match alone never merges anything.

## How the thresholds were chosen (by viewing candidate pairs)

Candidates: 9,181 hash-proposed pairs among the 4,365 images (kept plus removed). Viewed montages of sampled pairs by score band:

| Band (foreground score) | Pairs | Viewed | Same photo |
|---|---|---|---|
| >= 0.88 | 324 | 16 (0.88-0.94 band) | 15 |
| 0.80-0.88 and dHash <= 6 | 11 | 11 | 11 |
| 0.80-0.88 and dHash 7-10 | 17 | 17 | 11 |
| 0.80-0.88 and dHash > 10 | 48 | 16 (whole band, mixed) | about 7 |
| 0.70-0.80 | 39 | 16 | 0 |

Plain (unmasked) correlation was tried first; it gave false matches on white-background leaves at 0.88-0.93, which is why the foreground mask was added. The 0.80 / dHash 10 rule was chosen to favour fewer leaked copies: a wrong merge only forces two images into one split (or removes them if labels conflict), while a missed copy leaks.

Rough rates, from 71 viewed pairs and so only an estimate: about 3% of merged pairs (roughly 10 of about 350) are different photos; sampled bands >= 0.88 and the dHash <= 6 secondary band showed 1 wrong in 27. Leaks the rule misses are handled by the reviewed links below.

## Reviewed links and residual pairs

Six pairs that the rule missed were judged by eye to be the same photo (a tomato photo with and without an overlay, two crops of an alamy potato leaf, a shifted potato-row crop, a rusted leaf, and two stock-site copies of a leaf sprig and of a three-leaf cutting). They are in `datasets/manifests/field_reviewed_links.csv` with a reason each. `sprout.data.field` merges them on top of the rule, so they share a group and a split (or go through conflict removal if their labels differ).

`datasets/manifests/field_residual_pairs.csv` lists every pair left in different splits that is within 8 bits by pHash or dHash, with a `reviewed_same_photo` column. All 25 pairs now listed were viewed and are "no" (different images: isolated leaves from one stock series, apples, diagrams, a figure). The first pass found 32 hash-close pairs across splits; 6 were judged "yes" and linked, 1 pair appeared after the splits shifted (viewed, "no"), and 2 more stopped crossing splits.

Final numbers: 4,144 images; field_train 2,069, field_dev 1,040, field_test 1,035 (all 31 shared classes present); 219 images removed (215 in 96 groups as label conflicts by the rule, plus 4 by the same-name review below). Merging the six links reshuffled the group-to-split assignment: 339 of the 4,148 images at that step changed split (assignment stays deterministic for a given set of groups).

`tests/test_field_data.py` checks the real manifest, when raw images are present, for cross-split pairs within 8 bits and asserts all are in the residual list; it also asserts no listed pair is marked "yes" and that every reviewed link shares a group and split.

## Same-name review (last step)

Seven file names appear on more than one image. All seven were viewed (`datasets/manifests/field_same_name_review.csv`). Two pairs are the same photo under conflicting labels (a corn leaf at two crops, and one alamy tomato leaf filed under Potato and Tomato early blight); all four images are removed through `datasets/manifests/field_reviewed_removals.csv` and listed in `field_conflicts.csv`. The other five names (`0.jpg`, `0000.jpg`, `image.jpg`, `gray-leaf-spot.jpg`, `lateblightleaf.jpg`) are different photos and stay. Removal runs after split assignment, so no other image moved: compared with commit 08cb625, `field.csv` lost exactly those four rows and every other row is identical. A test fails if a descriptive (not digits-only, not in a short fixed word list) shared name has no recorded judgment.

The duplicate search on the field set was the last one; a few same-scene images could remain across splits, as a known limit of web-scraped data.

## No leaf-only subset (decided 2026-11-03)

After seeing `field_dev` mistakes, a hand sample of 50 (`results/error-analysis-e2.md`) found fruit photos and doubtful labels, so a leaf-only secondary subset defined by source class names was considered. It isn't declared, because that rule excludes nothing: none of the 48 shared rows in `datasets/class_map.csv` names fruit or a nutrient deficiency, and PlantSeg files fruit photos under disease names such as `apple scab`. PlantSeg's `Metadata.csv` has no plant-part field. This was checked from class names and that file's header only, with no `field_test` image or score read.

The manifests don't change, and the headline stays all of each split. In record `E2-20261103T092315605470Z`, recomputed from its saved predictions joined with `field.csv`'s `source` column, E1's `field_dev` accuracy is 142/619 = 0.2294 on PlantDoc and 91/421 = 0.2162 on PlantSeg. The two are close, so fruit photos don't appear to be driving the fall. This is a weak signal rather than a measurement of fruit photos, since fruit photos weren't counted across the split.
