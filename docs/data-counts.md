# Lab data counts (PlantVillage official 38-class release)

Source: https://github.com/spMohanty/PlantVillage-Dataset at commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, `raw/color` (`datasets/raw/lab/`) and `raw/segmented` (`datasets/raw/lab_segmented/`), CC BY-SA 3.0. Manifest: `datasets/manifests/lab.csv`, dataset hash `7eda8dbf1b7dbbec8aea577ccd3e9f439817f5f734f62be823756c8c2aaa1c30`.

The 38 classes include healthy classes, so they are not "38 diseases". Splits are near-duplicate-grouped: a source ID is not proven to identify one physical leaf.

## Totals
- `raw/color` actually holds 54305 images in 38 classes. The official README says 54,306. The difference is 1 image, which is the segmented-only file described below.
- `raw/segmented` holds 54,306 files.

## What the filenames contain
Surveyed over all 54305 color files:
- Most names are `<uuid>___<code> <number>.<ext>`. No uuid is shared by two files, so the uuid does not identify a leaf.
- The code after `___` (for example `RS_HL 5672`) is reused across classes for unrelated photos (the same code appears in Apple___healthy and Soybean___healthy), so a source ID is the class folder plus the code.
- Copy-named files (any filename containing "copy", case-insensitive): 1367. Forms by the text from "copy" onward: `copy` 720, `copy 2` 536, `copy 3` 95, `copy 4` 9, `copy 5` 2, `copy ` 2, `copy3` 2, `copy  3` 1. All of them are suffixes after a space or a digit; no code merely ends in the letters "copy". Each copy has its own uuid. A copy's code matches an original in the same class for only some of them (236 of 1367), and the grouping treats them all as one source ID per class and code. No `flip` or `deg` suffixes exist anywhere in the data.
- 1,192 files in `Corn_(maize)___Common_rust_` have no uuid (`RS_Rust 1563.JPG`); their segmented copies do have one.
- Extensions: .JPG 52803, .jpg 1500, .png 1, .jpeg 1.
- Segmented files add `_final_masked` and are `.jpg`.

## Groups
- Source-ID grouping alone gives 53505 groups; 628 of them hold more than one image (1428 images).
- After merging by perceptual hash (phash, distance <= 4, across flips and 90-degree rotations) there are 51210 groups. 2295 merges came from perceptual hashes beyond the source IDs.
- 1201 final groups hold more than one image (4296 images). 239 final groups contain images from more than one class (the same perceptual hash across class folders); each such group is kept whole in one split and counted under its majority class for stratification.

## Change from the copy-suffix fix
- Before the no-space ` copy` fix (`R.S_HL 0626copy.jpg`), the previous manifest (same paths) had 51211 groups; now 51210. 4 files are in a group whose membership changed.

## Large groups (chaining)
Largest groups by image count: g000682 (1290), g004509 (107), g002282 (19), g026372 (14), g023011 (13).
- g000682: 1290 images in 20 classes; top classes Soybean___healthy 986, Raspberry___healthy 73, Cherry_(including_sour)___healthy 41, Pepper,_bell___healthy 39, Apple___healthy 31, Potato___Late_blight 26; all in lab_train.
- g004509: 107 images in 7 classes; top classes Soybean___healthy 91, Orange___Haunglongbing_(Citrus_greening) 6, Tomato___Bacterial_spot 4, Pepper,_bell___healthy 3, Blueberry___healthy 1, Grape___Black_rot 1; all in lab_train.
- Chained multi-class groups: 239 groups holding 2074 images (3.8% of all images).
- Effect on class balance: a group is never split, so every image of these groups is in lab_train. Soybean___healthy: 986 of 5090 images (19%) sit in g000682; Raspberry___healthy: 73 of 371 images (20%) sit in g000682; Cherry_(including_sour)___healthy: 41 of 854 images (5%) sit in g000682; Soybean___healthy: 91 of 5090 images (2%) sit in g004509; Orange___Haunglongbing_(Citrus_greening): 6 of 5507 images (0%) sit in g004509; Tomato___Bacterial_spot: 4 of 2127 images (0%) sit in g004509. Those classes therefore get a smaller share in lab_dev and lab_test than the 15% targets.
- Random sample of 200 images from g000682 (seed 5): 0.33% of image pairs are within the 4-bit threshold, the median pair distance is 16 bits, and 46% of the sampled images have a neighbour within 4 inside the sample.
- What the sample looks like (12 images viewed): different single leaves (10 soybean, 2 bell pepper, per the labels) photographed alone on a plain grey background, with similar outline and little texture, so their 64-bit perceptual hashes are close. This is chaining (single-link merging of near-neighbours), not true duplicates; images in the sample are visibly different leaves. The grouping method and the distance of 4 were left as the plan sets them. Keeping the groups whole is safe against leakage, at the cost of the class imbalance above.

## Segmented versus color
- Names match exactly (stem plus `_final_masked`) for all but a few files; mismatch counts across all 38 classes:
  - Segmented files with no same-named color file: 1,197. Of these, 1,192 are the Corn common rust files whose color names lack the uuid (they pair by the code, `RS_Rust N`, one to one), 4 are Peach___Bacterial_spot / Strawberry___Leaf_scorch / Tomato___Spider_mites files whose segmented names drop the uuid and pair by code, and 1 has no color image.
  - Color files with no same-named segmented file: 1,196 (1,192 corn plus the 4 above).
  - After pairing by code, color images with no segmented copy: 0.
- The extra segmented file is `Grape___Esca_(Black_Measles)/7e1fd9b9-1fd9-4f98-93f5-8cf9ebc60dd9___FAM_B.Msls 4430_final_masked.jpg`. No color file anywhere has that code, so it is a segmented copy whose color image is missing from the release (not a duplicate). It is ignored: segmented images are only used by matching a color training image, and this one has none.

## Counts by class and split
| class | lab_train | lab_dev | lab_test | total |
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
| Corn_(maize)___healthy | 814 | 174 | 174 | 1162 |
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
| TOTAL | 38016 | 8152 | 8137 | 54305 |

## Old accuracy figures, reconciled with the E1 run
Source of every number from this run: `results/runs/E1-20261103T074728015380Z.json` (commit `592fa2c0fd4c2cd09d3343c7dddd0b8296363f49`, clean tree, `lab_dev`, n = 8152, dataset hash as above, 10,000-resample group bootstrap, seed 13). All E1 scores are top-1 over the 38 official classes on `lab_dev` only; `lab_test` was not read.

| System | accuracy (95% interval) | macro-F1 (95% interval) | ECE |
|---|---|---|---|
| E1: ConvNeXt-Tiny from ImageNet weights, fine-tuned on `lab_train` (10 epochs, lr 0.0001, batch 32) | 0.9979146221786065 (0.9968109702576189 to 0.9988966565624972) | 0.9966107943502964 (0.994622070375535 to 0.9982896717025654) | 0.0011059093317409797 |
| Legacy: `legacy, may have seen these images; 39-class checkpoint mapped onto 38 shared classes` | 1.0 (1.0 to 1.0) | 1.0 (1.0 to 1.0) | 5.531762164523446e-05 |

The legacy checkpoint's 39 outputs were mapped onto the 38 official classes by name (mapping saved in the record). Its `Background_without_leaves` output was dropped and probabilities were renormalised over the mapped outputs; the share of `lab_dev` images whose top legacy output was an unmapped one is 0.0. The class order from `legacy/PlantProject/application.py` is sorted and gives a score far above chance (1/38 = 0.0263), so the record marks it confirmed.

**Do not read the legacy 1.0 as performance.** What the legacy files show: `legacy/PlantProject/ConvNextAIPredictionModel/model.ipynb` loads one folder, `9layerplantdatacnn/data/noaug/noaug data` (55,448 images in 39 classes), with `ImageFolder` and splits it 80/20 with `random_split` over individual images. There is no grouping by near-duplicate, and no random seed appears in the notebook. The files do not say which dataset that folder holds, so they do not prove the old model trained on the same underlying PlantVillage images as `lab_train`. They do show that the class folder names are PlantVillage-style names (for example `Orange___Haunglongbing_(Citrus_greening)`), that image names are `image (N).JPG` rather than this release's names, and that the count (55,448) differs from this release's 54,305. If it did draw on the same photos, a per-image random split means near-duplicates of a photo could sit on both sides of it, and such a model could have seen `lab_dev` images; that is the reason for the label, but it is not shown. Either way, a perfect score on images the model may well have trained on says nothing about its real-world ability, and the unknown split history means it cannot be compared with E1. The record makes no comparison between them.

Lab scores, both rows, are scores on clean PlantVillage photos (plain background, one leaf). They are not real-world performance; real photos are scored separately on `field_*` splits.

### The three old figures
- **99.97%**: the best `val_acc` in `legacy/PlantProject/ConvNextAIPredictionModel/training_history.json`: 99.97006884166417 at epoch 11 of 16 (`best_epoch` 11, `epochs_ran` 16). The file records no split or dataset. The notebook in the same folder is the only training code found; it splits at random per image as described above, but its saved output is a 1-epoch run (`num_epochs = 1`), and no notebook output contains 99.97, so the files do not tie this history to that exact run. The validation split's membership is unknown.
- **97.5%**: the sentence "Accuracy: 97.5% on validation data" in `legacy/PlantProject/README.md`. The README gives no run, split or file for it, and it does not match the 99.97 in the training history. Its source is not established.
- **98.5%**: no source found. It is not in `legacy/`, the notebook, the training history, the README or the other docs; searching `legacy/`, `README.md` and `docs/` for the literal text finds it only in `docs/PLAN.md` (the task that asks for this reconciliation) and in this section.

None of the three is reproduced by this run, and none should be quoted as a result. The README also says 38 classes while the checkpoint has 39 outputs.
