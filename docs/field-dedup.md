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

Rough rates, from 71 viewed pairs and so only an estimate: about 3% of merged pairs (roughly 10 of about 350) are different photos; sampled bands >= 0.88 and the dHash <= 6 secondary band showed 1 wrong in 27.

## Residual pairs

`datasets/manifests/field_residual_pairs.csv` lists every pair in different splits that is within 8 bits by pHash or dHash but was not merged (32 pairs, with scores and labels). All 32 were viewed: about 6 are the same photo (top scores 0.75-0.79 with dHash 4-5, plus a few same-stock-photo variants); the other 26 are different images (isolated leaves, apples, diagrams). So the measured miss is about 6 same-photo pairs across 4,148 images, with about 13 of the 32 sharing a label and 19 not. Merging them needs a threshold (about 0.75 with dHash <= 5) that I did not adopt; it is left to the person to judge. A further 2 of 324 accepted pairs were found only beyond the 12-bit hash proposal radius, so pairs the hashes never propose are not counted here.

`tests/test_field_data.py` checks the real manifest, when raw images are present, for cross-split pairs within 8 bits and asserts all of them are in the residual list.
