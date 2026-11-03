# Extra real-photo sources for training (T4.5)

Built 2026-11-03 by `python -m sprout.data.extra_sources` (also `tools/build_extra_sources.py`), on CPU only. Hashing is pure PIL and the scene check works on small numpy arrays, and the rule in `docs/field-dedup.md` was kept as is, so nothing was moved to the GPU and no GPU share was taken. Output: `datasets/manifests/extra_train.csv` (columns `path, label, group, split, sha1, source`, `split` always `extra_train`) and `datasets/class_map_extra.csv`. The images stay in the git-ignored `datasets/raw/` and are never committed. `field.csv` is not touched. `field_dev` and `field_test` are used only to find and remove copies, and no score was computed on them.

## Sources

| source | status |
|---|---|
| PlantWild v1 (`plantwild.zip`, 18,542 images, 89 classes) | used. CC BY-NC-ND 4.0; weights trained on it need the person's check before publishing (no-derivatives term). |
| PlantWild v2 (`plantwild_v2.zip`, 11,488 images, 115 disease classes) | used, but adds almost nothing (21 images): nearly every mapped v2 image is a copy of a PlantSeg image already in `field.csv`, or of v1. |
| Plant Pathology 2021 | not used. The Kaggle source needs login; the only public copy is a third-party 16 GB mirror covering three usable apple classes. The coordinator may overrule. |

Details and licenses: `docs/datasets.md`. Both zips were extracted into new folders with a check that no member path escapes them.

## Mapping

150 of 204 source classes are excluded with the reason "no lab class for this plant or disease" (19,834 of 30,030 images). 54 map to lab classes. Mapping and reasons per class are in `datasets/class_map_extra.csv`. One judgment call: PlantWild `bell pepper leaf spot` goes to `Pepper,_bell___Bacterial_spot`, as PlantDoc's `Bell_pepper leaf spot` already does. `grape leaf spot` and `grapevine leafroll disease` are excluded, since the match to `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` is not certain.

## What was removed (30,030 images in)

| step | removed |
|---|---|
| unmapped class | 19,834 |
| exact copy (sha1) of `field_test` / `field_dev` / `field_train` / lab | 883 / 884 / 1,788 / 2 |
| near copy (pHash <= 4 over flips and rotations) of `field_test` / `field_dev` / `field_train` / lab | 52 / 60 / 105 / 69 |
| same scene (hash proposal confirmed by `same_scene_score`) as `field_test` / `field_dev` / `field_train` | 15 / 26 / 43 |
| close-match guard (below), `field_test` / `field_dev` / `field_train` | 6 / 1 / 4 |
| conflicting lab labels in one group | 351 |
| duplicate of another kept extra image (one per group stays) | 724 |
| over the per-class cap of 200 | 463 |
| **kept** | **4,720** (v1 4,699, v2 21) |

Each image is counted once, at the first step that removed it. Images that matched a `field_test` photo: 883 exact, 52 near, 15 same scene, 6 guard, 956 in all. For `field_dev`: 884 + 60 + 26 + 1 = 971. This is the PlantSeg overlap the lookup expected: PlantWild's images largely contain PlantSeg's.

**Guard added after viewing results.** With the plain rule the first build kept 4,727. The 60 hash-close pairs (within 8 bits) between kept extras and `field_dev` or `field_test` photos were scored and the three highest viewed: two were the same photo at another size (scores 0.83 and 0.785) that the 0.88 rule had missed, and one was a different corn leaf. The pipeline now also removes any extra with a hash-close pair to a field photo scoring at least 0.75 (`GUARD_SCORE`), 11 more images. The threshold was chosen from those 3 viewed pairs. The guard is stricter than the field rule, so it can remove a few non-copies. Pairs below 0.75 were not viewed one by one, so a few copies of dev or test photos could remain, as the field doc notes for web-scraped data.

## Cap

200 per lab class, set before any E12 result. The largest `field_train` class has 136 photos and the median is about 60, so 200 lets thin classes keep everything while no class outweighs the real-photo sets. Kept rows are chosen by sha1 order (deterministic, unrelated to image content). 463 images were cut, in healthy-leaf classes and in classes with more than 200 mapped images.

## Added per class (kept images, with `field_train` and `field_dev` for comparison)

| lab class | PlantWild v1 | v2 | total added | field_train | field_dev |
|---|--:|--:|--:|--:|--:|
| Apple___Apple_scab | 115 | 0 | 115 | 126 | 63 |
| Apple___Black_rot | 95 | 0 | 95 | 30 | 15 |
| Apple___Cedar_apple_rust | 195 | 0 | 195 | 90 | 46 |
| Apple___healthy | 200 | 0 | 200 | 44 | 22 |
| Blueberry___healthy | 200 | 0 | 200 | 57 | 29 |
| Cherry_(including_sour)___Powdery_mildew | 80 | 0 | 80 | 13 | 6 |
| Cherry_(including_sour)___healthy | 200 | 0 | 200 | 29 | 14 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 112 | 0 | 112 | 65 | 31 |
| Corn_(maize)___Common_rust_ | 105 | 0 | 105 | 115 | 57 |
| Corn_(maize)___Northern_Leaf_Blight | 91 | 0 | 91 | 132 | 67 |
| Corn_(maize)___healthy | 151 | 0 | 151 | 0 | 0 |
| Grape___Black_rot | 127 | 0 | 127 | 73 | 37 |
| Grape___healthy | 194 | 0 | 194 | 35 | 17 |
| Orange___Haunglongbing_(Citrus_greening) | 143 | 0 | 143 | 47 | 23 |
| Peach___healthy | 150 | 0 | 150 | 55 | 28 |
| Pepper,_bell___Bacterial_spot | 103 | 21 | 124 | 58 | 29 |
| Pepper,_bell___healthy | 200 | 0 | 200 | 31 | 15 |
| Potato___Early_blight | 115 | 0 | 115 | 67 | 34 |
| Potato___Late_blight | 131 | 0 | 131 | 81 | 41 |
| Potato___healthy | 200 | 0 | 200 | 0 | 0 |
| Raspberry___healthy | 170 | 0 | 170 | 59 | 30 |
| Soybean___healthy | 200 | 0 | 200 | 33 | 16 |
| Squash___Powdery_mildew | 125 | 0 | 125 | 136 | 68 |
| Strawberry___Leaf_scorch | 50 | 0 | 50 | 9 | 5 |
| Strawberry___healthy | 186 | 0 | 186 | 47 | 24 |
| Tomato___Bacterial_spot | 144 | 0 | 144 | 91 | 46 |
| Tomato___Early_blight | 149 | 0 | 149 | 98 | 50 |
| Tomato___Late_blight | 139 | 0 | 139 | 107 | 54 |
| Tomato___Leaf_Mold | 110 | 0 | 110 | 93 | 47 |
| Tomato___Septoria_leaf_spot | 93 | 0 | 93 | 109 | 55 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 99 | 0 | 99 | 63 | 32 |
| Tomato___Tomato_mosaic_virus | 127 | 0 | 127 | 45 | 23 |
| Tomato___healthy | 200 | 0 | 200 | 31 | 16 |

## Which thin classes it helped

- Two classes had no real photo in `field.csv`, so none in field training: `Corn_(maize)___healthy` now has 151 and `Potato___healthy` 200. They still have no `field_dev` or `field_test` photos, so a field score cannot show the effect.
- Classes where the added photos are at least 3 times `field_train`: Cherry Powdery_mildew (80 vs 13), Cherry healthy (200 vs 29), Apple healthy (200 vs 44), Bell pepper healthy (200 vs 31), Soybean healthy (200 vs 33), Grape healthy (194 vs 35), Tomato healthy (200 vs 31), Strawberry Leaf_scorch (50 vs 9), Apple Black_rot (95 vs 30), Strawberry healthy (186 vs 47), Blueberry healthy (200 vs 57), Raspberry healthy (170 vs 59), Tomato mosaic virus (127 vs 45).
- Classes still without any real photo in training or evaluation: Grape Esca, Grape Leaf_blight, Peach Bacterial_spot, Tomato Spider_mites, Tomato Target_Spot. PlantWild has no matching class.
- Nothing here says the added photos improve a model. That is for E12 (T4.6).

## Limits

- Per-image rights on the web-scraped photos were not checked; use stays local, non-commercial and unpublished.
- Healthy classes are mostly at the cap and come from one source whose style (stock-like leaf photos) differs from the diseased field photos, so they may push a model toward "healthy" for white-background leaves. Worth watching in E12.
- The copy search covered lab, every field split and the extras themselves, not photos outside the repo.
