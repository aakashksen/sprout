# Lab data counts (PlantVillage official 38-class release)

Source: https://github.com/spMohanty/PlantVillage-Dataset at commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, `raw/color` (`datasets/raw/lab/`) and `raw/segmented` (`datasets/raw/lab_segmented/`), CC BY-SA 3.0. Manifest: `datasets/manifests/lab.csv`, dataset hash `091c28b4fdbffd6d82aa0a38abf0f180f06c2c567f39abef7e54e6b6a076f376`.

The 38 classes include healthy classes, so they are not "38 diseases". Splits are near-duplicate-grouped: a source ID is not proven to identify one physical leaf.

## Totals
- `raw/color` actually holds 54305 images in 38 classes. The official README says 54,306. The difference is 1 image, which is the segmented-only file described below.
- `raw/segmented` holds 54,306 files.

## What the filenames contain
Surveyed over all 54305 color files:
- Most names are `<uuid>___<code> <number>.<ext>`. No uuid is shared by two files, so the uuid does not identify a leaf.
- The code after `___` (for example `RS_HL 5672`) is reused across classes for unrelated photos (the same code appears in Apple___healthy and Soybean___healthy), so a source ID is the class folder plus the code.
- Copy suffixes present: ` copy` (719 files), ` copy 2` (536), ` copy 3` (95), and rarer forms (` copy 4` 9, ` copy 5` 2, ` copy3` 2, ` copy  3` 1, ` copy ` 2; 1,366 copy-named files in all); each has its own uuid. A copy's code matches an original in the same class for only some of them (234 of 1,366 by an earlier survey), and the grouping treats them all as one source ID per class and code. No `flip` or `deg` suffixes exist anywhere in the data.
- 1,192 files in `Corn_(maize)___Common_rust_` have no uuid (`RS_Rust 1563.JPG`); their segmented copies do have one.
- Extensions: .JPG 52803, .jpg 1500, .png 1, .jpeg 1.
- Segmented files add `_final_masked` and are `.jpg`.

## Groups
- Source-ID grouping alone gives 53506 groups; 628 of them hold more than one image (1427 images).
- After merging by perceptual hash (phash, distance <= 4, across flips and 90-degree rotations) there are 51211 groups. 2295 merges came from perceptual hashes beyond the source IDs.
- 1201 final groups hold more than one image (4295 images). 239 final groups contain images from more than one class (the same perceptual hash across class folders); each such group is kept whole in one split and counted under its majority class for stratification.

## Segmented versus color
- Names match exactly (stem plus `_final_masked`) for all but a few files; mismatch counts across all 38 classes:
  - Segmented files with no same-named color file: 1,197. Of these, 1,192 are the Corn common rust files whose color names lack the uuid (they pair by the code, `RS_Rust N`, one to one), 4 are Peach___Bacterial_spot / Strawberry___Leaf_scorch / Tomato___Spider_mites files whose segmented names drop the uuid and pair by code, and 1 has no color image.
  - Color files with no same-named segmented file: 1,196 (1,192 corn plus the 4 above).
  - After pairing by code, color images with no segmented copy: 0.
- The extra segmented file is `Grape___Esca_(Black_Measles)/7e1fd9b9-1fd9-4f98-93f5-8cf9ebc60dd9___FAM_B.Msls 4430_final_masked.jpg`. No color file anywhere has that code, so it is a segmented copy whose color image is missing from the release (not a duplicate). It is ignored: segmented images are only used by matching a color training image, and this one has none.

## Counts by class and split
| class | train | val | test | total |
|---|---|---|---|---|
| Apple___Apple_scab | 440 | 95 | 95 | 630 |
| Apple___Black_rot | 436 | 93 | 92 | 621 |
| Apple___Cedar_apple_rust | 193 | 41 | 41 | 275 |
| Apple___healthy | 1166 | 244 | 235 | 1645 |
| Blueberry___healthy | 1052 | 226 | 224 | 1502 |
| Cherry_(including_sour)___Powdery_mildew | 736 | 158 | 158 | 1052 |
| Cherry_(including_sour)___healthy | 612 | 123 | 119 | 854 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 359 | 78 | 76 | 513 |
| Corn_(maize)___Common_rust_ | 834 | 179 | 179 | 1192 |
| Corn_(maize)___Northern_Leaf_Blight | 689 | 148 | 148 | 985 |
| Corn_(maize)___healthy | 813 | 174 | 175 | 1162 |
| Grape___Black_rot | 826 | 177 | 177 | 1180 |
| Grape___Esca_(Black_Measles) | 970 | 206 | 207 | 1383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 753 | 162 | 161 | 1076 |
| Grape___healthy | 296 | 64 | 63 | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 3853 | 826 | 828 | 5507 |
| Peach___Bacterial_spot | 1607 | 346 | 344 | 2297 |
| Peach___healthy | 254 | 53 | 53 | 360 |
| Pepper,_bell___Bacterial_spot | 702 | 147 | 148 | 997 |
| Pepper,_bell___healthy | 1047 | 215 | 216 | 1478 |
| Potato___Early_blight | 700 | 150 | 150 | 1000 |
| Potato___Late_blight | 706 | 146 | 148 | 1000 |
| Potato___healthy | 111 | 19 | 22 | 152 |
| Raspberry___healthy | 282 | 45 | 44 | 371 |
| Soybean___healthy | 3473 | 807 | 810 | 5090 |
| Squash___Powdery_mildew | 1285 | 275 | 275 | 1835 |
| Strawberry___Leaf_scorch | 777 | 166 | 166 | 1109 |
| Strawberry___healthy | 319 | 70 | 67 | 456 |
| Tomato___Bacterial_spot | 1495 | 315 | 317 | 2127 |
| Tomato___Early_blight | 699 | 152 | 149 | 1000 |
| Tomato___Late_blight | 1337 | 285 | 287 | 1909 |
| Tomato___Leaf_Mold | 667 | 143 | 142 | 952 |
| Tomato___Septoria_leaf_spot | 1245 | 262 | 264 | 1771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1174 | 250 | 252 | 1676 |
| Tomato___Target_Spot | 983 | 211 | 210 | 1404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 3750 | 804 | 803 | 5357 |
| Tomato___Tomato_mosaic_virus | 260 | 58 | 55 | 373 |
| Tomato___healthy | 1114 | 239 | 238 | 1591 |
| TOTAL | 38015 | 8152 | 8138 | 54305 |
