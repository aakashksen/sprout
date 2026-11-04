# Extra real-photo sources for training (T4.5)

Rebuilt 2026-11-03 (fourth pass, after two failed verifications) by `tools/build_extra_sources.py`, which calls `sprout.data.extra_sources.main`. Output: `datasets/manifests/extra_train.csv` (columns `path, label, group, split, sha1, source`, `split` always `extra_train`, groups prefixed `x`) and `datasets/class_map_extra.csv`. The images stay in the git-ignored `datasets/raw/` and are never committed. `field.csv` is not touched. `field_dev` and `field_test` are used only to find and remove copies, and no score was computed on them.

**Device.** Both copy checks on the GPU ran with a share held from `taskctl gpu-acquire T4.5` and released right after (`device: cuda` in the build output). The build tool waits for a share rather than falling back to the CPU, because DINOv2 over about 7,300 images and their mirrors would take hours there. Hashing and the rest of the rule ran on the CPU (pure PIL and small numpy arrays). On the real data the full-frame correlation gave the same 2,890 drops on CPU and on the GPU (compared as sets). The crop-aware step was not compared across devices; it was checked on the CPU against 7 known pairs.

## Sources

| source | status |
|---|---|
| PlantWild v1 (`plantwild.zip`, 18,542 images, 89 classes) | used. CC BY-NC-ND 4.0; weights trained on it need the person's check before publishing (no-derivatives term). |
| PlantWild v2 (`plantwild_v2.zip`, 11,488 images, 115 disease classes) | used, but adds almost nothing (15 images): nearly every mapped v2 image is a copy of a PlantSeg image already in `field.csv`, or of v1. |
| Plant Pathology 2021 | not used. The Kaggle source needs login; the only public copy is a third-party 16 GB mirror covering three usable apple classes. The coordinator may overrule. |

Details and licenses: `docs/datasets.md`. Both zips were extracted into new folders with a check that no member path escapes them.

## Mapping

150 of 204 source classes are excluded with the reason "no lab class for this plant or disease" (19,834 of 30,030 images). 54 map to lab classes. Mapping and reasons per class are in `datasets/class_map_extra.csv`. One judgment call: PlantWild `bell pepper leaf spot` goes to `Pepper,_bell___Bacterial_spot`, as PlantDoc's `Bell_pepper leaf spot` already does. `grape leaf spot` and `grapevine leafroll disease` are excluded, since the match to `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` is not certain.

## Copy checks

Four layers, all of which only drop extras. Layers 1 and 2 run first, on the GPU.

1. **Full-frame correlation** (`src/sprout/data/copy_check.py`). 32x32 grayscale, mean-centred correlation, best over the 8 rotations and flips, against every `field_dev` and `field_test` photo whatever the class; drop at 0.75 or more. It uses no hashes and no mask. It removes exact and near copies, but a crop, zoom or caption shifts the pixel grid and scores low.
2. **Crop-aware check** (`src/sprout/data/crop_copies.py`). DINOv2 small (frozen, from the pinned torch hub cache, as in T4.1) gives each extra's class token and 16x16 patch tokens, for the image as it is and mirrored. For each extra the 3 nearest `field_dev` and 3 nearest `field_test` photos by class token are the candidates (each of the 16 known copies was the 1st or 2nd nearest of its split). OpenCV is not installed, so local matching uses the patch tokens: mutual nearest-neighbour patches with cosine at least 0.4, then a fixed-seed RANSAC over scale and shift counts the patches that agree on one placement (tolerance one patch). An extra is dropped if its best class-token cosine is at least 0.80, or its best inlier count is at least 30, or it has cosine at least 0.75 and at least 20 inliers. Mirrored copies are covered; 90-degree rotations are not (layer 1 covers those only at full frame).
3. **The field-dedup rule** (exact sha1, pHash <= 4 near copy, hash proposal confirmed by `same_scene_score`), against lab and every field split.
4. **The close-match guard** (hash-close within 8 bits and `same_scene_score` >= 0.75) against every field split. It removed nothing in the final build.

`field.csv` and the field rule in `docs/field-dedup.md` are unchanged. Layers 1 and 2 are deliberately over-inclusive because extras are plentiful and a leaked evaluation photo corrupts E12.

**How the crop rule was chosen.** A labelled set was built from the pairs seen in the failed verifications and from stratified samples of an earlier manifest that I viewed myself:
- 16 known copies: the 6 from the first verdict, the 7 from the second, and 3 more seen while checking (all named in `tests/test_extra_sources.py`);
- 10 of 10 sampled pairs with 60 or more inliers were copies, one more high-cosine pair was a copy, and 4 pairs I could not call are counted as "drop";
- 45 non-copies (stock leaves on white with shared footers, similar corn or tomato close-ups, and so on) from the bands 15 to 59 inliers and cosine 0.8 to 0.9.

That is 31 positives and 45 negatives, split at random with a fixed seed into a part for choosing the rule (16 positives, 23 negatives) and a held-back part for measuring (15 positives, 22 negatives). The rule with the fewest drops that caught all positives in the choosing part was cosine at least 0.90 or at least 40 inliers. On the held-back part it caught 13 of 15 and missed two: one pair I had marked "unsure", and the potato late blight zoom `potato late blight/google_0097.jpg` (cosine 0.68, 39 inliers, one short). So the estimated miss rate on copies like these is 2 of 15, with a 95% upper bound of about 40%. Fifteen positives is a small sample, and the positives are copies that were found, so the miss rate on copies nobody has found yet is probably higher.

Because unsure means drop, the shipped rule is looser than that optimum. After the first rebuild I viewed fresh random samples of pairs just under the rule, and three rounds each turned up one probable crop that the rule missed (cosine 0.79 with 24 inliers, 0.85 with 15, 0.83 with 15), about 1 in 20 to 1 in 30. I then lowered the cosine floor to 0.80 and added the combined clause. The shipped rule catches all 15 held-back positives, but that is no longer an unbiased measure: it was set after the held-back part and the audits were seen. It drops 13 of 22 held-back negatives and 18 of 23 choosing negatives, which is the price of over-inclusion (those negatives were sampled from near-threshold bands, so this is not the drop rate over all extras).

**How I searched for more of the same.** Every mapped extra was scored against every `field_dev` and `field_test` photo, not just the 13 named leaks. I then viewed:
- the top 30 pairs of the verifier's own crop-tolerant gray correlation (80% and 60% central crops either way, 8 symmetries) and the top 10 of its Sobel edge correlation on an intermediate manifest: no copies;
- the 24 extras that an earlier scoring with 5 candidates per split had flagged at 30 or more inliers only on the 4th or 5th candidate: all non-copies, which supports using 3 candidates;
- the fresh random samples above, three rounds with one probable copy each, which produced the looser rule;
- last, 20 pairs sampled from the 1,349 extras that sit just under the shipped rule (cosine 0.72 to 0.80 with fewer than 20 inliers, or 18 or more inliers with cosine under 0.75): no copies. That bounds the rate in that pool at about 14% at 95% confidence (0 of 20), so up to roughly 190 extras there could still be copies.

The real-manifest test recomputes the rule over all kept extras and requires that it flags none.

## What was removed (30,030 images in)

Each image is counted once, at the first step that removed it.

| step | removed |
|---|---|
| unmapped class | 19,834 |
| full-frame correlation >= 0.75 with `field_test` / `field_dev` | 1,374 / 1,516 |
| crop-aware check, copy of `field_test` / `field_dev` | 1,095 / 1,078 |
| exact copy (sha1) of `field_train` / lab | 994 / 2 |
| near copy (pHash) of `field_train` / lab | 50 / 23 |
| same scene (`same_scene_score`) as `field_train` | 16 |
| conflicting lab labels in one group | 218 |
| duplicate of another kept extra image (one per group stays) | 438 |
| over the per-class cap of 200 | 45 |
| **kept** | **3,347** (v1 3,332, v2 15) |

The full-frame correlation also removes exact copies of `field_dev` and `field_test` (their score is 1.0), and the crop-aware check removes many extras that are near-identical or merely similar scenes, so the two steps account for 5,063 images (2,890 + 2,173). The crop-aware step dropped many corn photos because DINOv2 scores close-ups of corn leaves as similar to each other, so the corn disease classes keep only 17 to 23 extras each. Earlier builds of this task kept 4,720, 4,351, 3,995 and 3,898 images, and each still contained copies of dev or test photos.

## Cap

200 per lab class, set before any E12 result. The largest `field_train` class has 136 photos and the median is about 60, so 200 lets thin classes keep everything while no class outweighs the real-photo sets. Kept rows are chosen by sha1 order (deterministic, unrelated to image content). After the copy checks only 45 images were cut, and one class ended at exactly 200 (Apple healthy).

## Groups

Extra group ids are prefixed `x`. In an earlier build 98 ids also appeared in `field.csv`. The ids come from a numbering (`g` plus a running number) that every manifest build restarts, so those were the same numbers reused for different images, not shared groups. The prefix keeps the namespaces apart so E12 cannot merge an extra group with a field or lab group. A test checks that no extra group id is in `field.csv` or `lab.csv`.

## Added per class (kept images, with `field_train` and `field_dev` for comparison)

| lab class | PlantWild v1 | v2 | total added | field_train | field_dev |
|---|--:|--:|--:|--:|--:|
| Apple___Apple_scab | 94 | 0 | 94 | 126 | 63 |
| Apple___Black_rot | 84 | 0 | 84 | 30 | 15 |
| Apple___Cedar_apple_rust | 169 | 0 | 169 | 90 | 46 |
| Apple___healthy | 200 | 0 | 200 | 44 | 22 |
| Blueberry___healthy | 158 | 0 | 158 | 57 | 29 |
| Cherry_(including_sour)___Powdery_mildew | 73 | 0 | 73 | 13 | 6 |
| Cherry_(including_sour)___healthy | 178 | 0 | 178 | 29 | 14 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 17 | 0 | 17 | 65 | 31 |
| Corn_(maize)___Common_rust_ | 23 | 0 | 23 | 115 | 57 |
| Corn_(maize)___Northern_Leaf_Blight | 20 | 0 | 20 | 132 | 67 |
| Corn_(maize)___healthy | 69 | 0 | 69 | 0 | 0 |
| Grape___Black_rot | 87 | 0 | 87 | 73 | 37 |
| Grape___healthy | 77 | 0 | 77 | 35 | 17 |
| Orange___Haunglongbing_(Citrus_greening) | 124 | 0 | 124 | 47 | 23 |
| Peach___healthy | 114 | 0 | 114 | 55 | 28 |
| Pepper,_bell___Bacterial_spot | 91 | 15 | 106 | 58 | 29 |
| Pepper,_bell___healthy | 151 | 0 | 151 | 31 | 15 |
| Potato___Early_blight | 77 | 0 | 77 | 67 | 34 |
| Potato___Late_blight | 98 | 0 | 98 | 81 | 41 |
| Potato___healthy | 178 | 0 | 178 | 0 | 0 |
| Raspberry___healthy | 59 | 0 | 59 | 59 | 30 |
| Soybean___healthy | 173 | 0 | 173 | 33 | 16 |
| Squash___Powdery_mildew | 87 | 0 | 87 | 136 | 68 |
| Strawberry___Leaf_scorch | 33 | 0 | 33 | 9 | 5 |
| Strawberry___healthy | 85 | 0 | 85 | 47 | 24 |
| Tomato___Bacterial_spot | 113 | 0 | 113 | 91 | 46 |
| Tomato___Early_blight | 116 | 0 | 116 | 98 | 50 |
| Tomato___Late_blight | 124 | 0 | 124 | 107 | 54 |
| Tomato___Leaf_Mold | 74 | 0 | 74 | 93 | 47 |
| Tomato___Septoria_leaf_spot | 70 | 0 | 70 | 109 | 55 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 77 | 0 | 77 | 63 | 32 |
| Tomato___Tomato_mosaic_virus | 98 | 0 | 98 | 45 | 23 |
| Tomato___healthy | 141 | 0 | 141 | 31 | 16 |

## Which thin classes it helped

- Two classes had no real photo in `field.csv`, so none in field training: `Corn_(maize)___healthy` now has 69 and `Potato___healthy` 178. They still have no `field_dev` or `field_test` photos, so a field score cannot show the effect.
- Classes where the added photos are at least 3 times `field_train`: Cherry Powdery_mildew (73 vs 13), Cherry healthy (178 vs 29), Apple healthy (200 vs 44), Bell pepper healthy (151 vs 31), Soybean healthy (173 vs 33), Tomato healthy (141 vs 31), Strawberry Leaf_scorch (33 vs 9), Apple Black_rot (84 vs 30; exactly 2.8 times, so just under).
- The three corn disease classes lost most of their extras to the copy checks (17 to 23 each, against 65 to 132 in `field_train`), so the corn classes gain little.
- Classes still without any real photo in training or evaluation: Grape Esca, Grape Leaf_blight, Peach Bacterial_spot, Tomato Spider_mites, Tomato Target_Spot. PlantWild has no matching class.
- Nothing here says the added photos improve a model. That is for E12 (T4.6).

## Limits

- **Noisy photos.** The verifier's sample of PlantWild photos showed that many are not clean leaf photos: the sampled `bell pepper leaf spot` images show whole plants and fruit, and the `corn leaf` (Corn healthy) images include corn cobs, dry yellow-brown leaves and whole fields. Labels are plausible for the crop but noisy for the disease or health state. Nothing was filtered by hand; E12 measures whether the extras help as they are.
- **Residual copies.** The copy checks are tuned by viewing pairs and are not exhaustive. The audit of the pool just under the rule found 0 copies in 20, which still allows up to about 14% of that pool of 1,349 (95% bound). 90-degree rotations of cropped copies are not covered. A copy whose true match is not among its 3 nearest dev and test photos by class token is not seen by the crop check (not observed in the 16 known copies).
- Per-image rights on the web-scraped photos were not checked; use stays local, non-commercial and unpublished.
- Healthy classes make up most of the added photos and come from one source whose style (stock-like leaf photos) differs from the diseased field photos, so they may push a model toward "healthy" for white-background leaves. Worth watching in E12.
- The copy search covered lab, every field split and the extras themselves, not photos outside the repo.
