# Extra real-photo sources for training (T4.5)

Rebuilt 2026-11-03 (second pass after a failed verification) by `tools/build_extra_sources.py`, which calls `sprout.data.extra_sources.main`. Output: `datasets/manifests/extra_train.csv` (columns `path, label, group, split, sha1, source`, `split` always `extra_train`, groups prefixed `x`) and `datasets/class_map_extra.csv`. The images stay in the git-ignored `datasets/raw/` and are never committed. `field.csv` is not touched. `field_dev` and `field_test` are used only to find and remove copies, and no score was computed on them.

**Device.** The copy-correlation step (below) ran on the GPU (`device: cuda` in the build output), holding a share from `taskctl gpu-acquire T4.5` just before and releasing it just after. Hashing and the rest of the rule ran on the CPU (pure PIL and small numpy arrays). On the real data the correlation gave the same 2,890 drops on CPU and on the GPU (compared as sets).

## Sources

| source | status |
|---|---|
| PlantWild v1 (`plantwild.zip`, 18,542 images, 89 classes) | used. CC BY-NC-ND 4.0; weights trained on it need the person's check before publishing (no-derivatives term). |
| PlantWild v2 (`plantwild_v2.zip`, 11,488 images, 115 disease classes) | used, but adds almost nothing (20 images): nearly every mapped v2 image is a copy of a PlantSeg image already in `field.csv`, or of v1. |
| Plant Pathology 2021 | not used. The Kaggle source needs login; the only public copy is a third-party 16 GB mirror covering three usable apple classes. The coordinator may overrule. |

Details and licenses: `docs/datasets.md`. Both zips were extracted into new folders with a check that no member path escapes them.

## Mapping

150 of 204 source classes are excluded with the reason "no lab class for this plant or disease" (19,834 of 30,030 images). 54 map to lab classes. Mapping and reasons per class are in `datasets/class_map_extra.csv`. One judgment call: PlantWild `bell pepper leaf spot` goes to `Pepper,_bell___Bacterial_spot`, as PlantDoc's `Bell_pepper leaf spot` already does. `grape leaf spot` and `grapevine leafroll disease` are excluded, since the match to `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` is not certain.

## Copy checks

Three layers, all of which only drop extras:

1. **Correlation check (new, runs first).** 32x32 grayscale, mean-centred correlation, best over the 8 rotations and flips, for every mapped extra against every `field_dev` and `field_test` photo, whatever the class. An extra at or above 0.75 is dropped (`COPY_SCORE` in `src/sprout/data/copy_check.py`). It uses no hashes and no foreground mask, so it does not share the field rule's blind spot (watermarks, crops and caption strips lower that rule's masked score). It is deliberately over-inclusive for extras, and the field rule in `docs/field-dedup.md` and `field.csv` are unchanged.
2. **The field-dedup rule** (exact sha1, pHash <= 4 near copy, hash proposal confirmed by `same_scene_score`), against lab and every field split.
3. **The close-match guard** (hash-close within 8 bits and `same_scene_score` >= 0.75) against every field split.

**How the threshold was chosen.** The first version used 0.80, which came from the verifier's count. At 0.80 the extras that remained were checked again: the 20 highest-scoring kept pairs (0.797 to 0.800) were viewed, and one was the same photo (an apple branch with a white side strip added, 0.798). So the threshold was lowered to 0.75, which removes 316 more extras from that build. The next 20 pairs just under 0.75 (0.747 to 0.750) were viewed, and none were copies. Pairs below 0.747 were not viewed, so a few copies may remain.

## What was removed (30,030 images in)

Each image is counted once, at the first step that removed it.

| step | removed |
|---|---|
| unmapped class | 19,834 |
| correlation >= 0.75 with `field_test` / `field_dev` | 1,374 / 1,516 |
| exact copy (sha1) of `field_train` / lab | 1,659 / 2 |
| near copy (pHash) of `field_test` / `field_train` / lab | 2 / 93 / 43 |
| same scene (`same_scene_score`) as `field_train` | 28 |
| close-match guard, `field_test` / `field_train` | 1 / 1 |
| conflicting lab labels in one group | 335 |
| duplicate of another kept extra image (one per group stays) | 669 |
| over the per-class cap of 200 | 122 |
| **kept** | **4,351** (v1 4,331, v2 20) |

The correlation step removes exact copies of `field_dev` and `field_test` too (their score is 1.0), so the exact-copy and near-copy rows for those splits are now zero or near zero, and the old counts (883 and 884 exact) sit inside the 1,374 and 1,516. In all, 2,892 images were copies of a `field_dev` or `field_test` photo by some step. This is the PlantSeg overlap the lookup expected: PlantWild's images largely contain PlantSeg's.

The previous build kept 4,720 and had at least 6 kept watermark or crop copies of `field_dev`/`field_test` photos that the field rule and the 0.75 guard had missed (all stock leaves on white). This build removes those: the six pairs named by the verifier are absent from the manifest, and the manifest test below checks it.

## Cap

200 per lab class, set before any E12 result. The largest `field_train` class has 136 photos and the median is about 60, so 200 lets thin classes keep everything while no class outweighs the real-photo sets. Kept rows are chosen by sha1 order (deterministic, unrelated to image content). 122 images were cut; three classes ended at exactly 200 (Apple, Cherry and Soybean healthy).

## Groups

Extra group ids are prefixed `x`. In the previous build 98 ids also appeared in `field.csv`. The ids come from a numbering (`g` plus a running number) that every manifest build restarts, so those were the same numbers reused for different images, not shared groups. The prefix keeps the namespaces apart so E12 cannot merge an extra group with a field or lab group. A test checks that no extra group id is in `field.csv` or `lab.csv`.

## Added per class (kept images, with `field_train` and `field_dev` for comparison)

| lab class | PlantWild v1 | v2 | total added | field_train | field_dev |
|---|--:|--:|--:|--:|--:|
| Apple___Apple_scab | 109 | 0 | 109 | 126 | 63 |
| Apple___Black_rot | 90 | 0 | 90 | 30 | 15 |
| Apple___Cedar_apple_rust | 185 | 0 | 185 | 90 | 46 |
| Apple___healthy | 200 | 0 | 200 | 44 | 22 |
| Blueberry___healthy | 188 | 0 | 188 | 57 | 29 |
| Cherry_(including_sour)___Powdery_mildew | 79 | 0 | 79 | 13 | 6 |
| Cherry_(including_sour)___healthy | 200 | 0 | 200 | 29 | 14 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 108 | 0 | 108 | 65 | 31 |
| Corn_(maize)___Common_rust_ | 96 | 0 | 96 | 115 | 57 |
| Corn_(maize)___Northern_Leaf_Blight | 82 | 0 | 82 | 132 | 67 |
| Corn_(maize)___healthy | 132 | 0 | 132 | 0 | 0 |
| Grape___Black_rot | 116 | 0 | 116 | 73 | 37 |
| Grape___healthy | 145 | 0 | 145 | 35 | 17 |
| Orange___Haunglongbing_(Citrus_greening) | 140 | 0 | 140 | 47 | 23 |
| Peach___healthy | 131 | 0 | 131 | 55 | 28 |
| Pepper,_bell___Bacterial_spot | 99 | 20 | 119 | 58 | 29 |
| Pepper,_bell___healthy | 188 | 0 | 188 | 31 | 15 |
| Potato___Early_blight | 108 | 0 | 108 | 67 | 34 |
| Potato___Late_blight | 125 | 0 | 125 | 81 | 41 |
| Potato___healthy | 197 | 0 | 197 | 0 | 0 |
| Raspberry___healthy | 118 | 0 | 118 | 59 | 30 |
| Soybean___healthy | 200 | 0 | 200 | 33 | 16 |
| Squash___Powdery_mildew | 122 | 0 | 122 | 136 | 68 |
| Strawberry___Leaf_scorch | 47 | 0 | 47 | 9 | 5 |
| Strawberry___healthy | 136 | 0 | 136 | 47 | 24 |
| Tomato___Bacterial_spot | 134 | 0 | 134 | 91 | 46 |
| Tomato___Early_blight | 139 | 0 | 139 | 98 | 50 |
| Tomato___Late_blight | 132 | 0 | 132 | 107 | 54 |
| Tomato___Leaf_Mold | 103 | 0 | 103 | 93 | 47 |
| Tomato___Septoria_leaf_spot | 84 | 0 | 84 | 109 | 55 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 93 | 0 | 93 | 63 | 32 |
| Tomato___Tomato_mosaic_virus | 117 | 0 | 117 | 45 | 23 |
| Tomato___healthy | 188 | 0 | 188 | 31 | 16 |

## Which thin classes it helped

- Two classes had no real photo in `field.csv`, so none in field training: `Corn_(maize)___healthy` now has 132 and `Potato___healthy` 197. They still have no `field_dev` or `field_test` photos, so a field score cannot show the effect.
- Classes where the added photos are at least 3 times `field_train`: Cherry Powdery_mildew (79 vs 13), Cherry healthy (200 vs 29), Apple healthy (200 vs 44), Bell pepper healthy (188 vs 31), Soybean healthy (200 vs 33), Grape healthy (145 vs 35), Tomato healthy (188 vs 31), Strawberry Leaf_scorch (47 vs 9), Apple Black_rot (90 vs 30), Blueberry healthy (188 vs 57).
- Classes still without any real photo in training or evaluation: Grape Esca, Grape Leaf_blight, Peach Bacterial_spot, Tomato Spider_mites, Tomato Target_Spot. PlantWild has no matching class.
- Nothing here says the added photos improve a model. That is for E12 (T4.6).

## Limits

- **Noisy photos.** The verifier's sample of PlantWild photos showed that many are not clean leaf photos: the sampled `bell pepper leaf spot` images show whole plants and fruit, and the `corn leaf` (Corn healthy) images include corn cobs, dry yellow-brown leaves and whole fields. Labels are plausible for the crop but noisy for the disease or health state. Nothing was filtered by hand; E12 measures whether the extras help as they are.
- Per-image rights on the web-scraped photos were not checked; use stays local, non-commercial and unpublished.
- Healthy classes make up most of the added photos and come from one source whose style (stock-like leaf photos) differs from the diseased field photos, so they may push a model toward "healthy" for white-background leaves. Worth watching in E12.
- The copy search covered lab, every field split and the extras themselves, not photos outside the repo. Copies of `field_dev` or `field_test` photos scoring below 0.747 on the correlation and missed by the field rule could remain.
