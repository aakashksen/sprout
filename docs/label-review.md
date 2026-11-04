# Label review of the real photos

Agent review, not a human review: vision agents looked at each photo with its label and file name and nothing else (no split name, no score, no prediction). A removal counts only after a second look at larger size. The rules: not a plant photo, wrong plant, or a label clearly contradicted by the photo; no relabeling and no fine disease judgments. Removed photos stay in `field.csv`; `field_clean.csv` leaves them out. Exclusion version: 1.

Reviewed: 2075 photos of `field_dev` and `field_test`, shuffled together. Removed: 31. Uncertain (kept): 248. First-pass removals: 46. At the larger second look: 30 confirmed, 3 called ok (overturned), 13 called uncertain (kept, in `label_uncertain.csv`).

Limit, from a calibration of the first pass: of 4 known-bad photos placed on the sheets, 2 were `remove`, 1 was `uncertain` ("framed slide-like image") and 1 was `ok`. The sheets are conservative, so the removal rates below are a lower bound on bad photos.

How the removals began (all reviewed photos): began as `remove`: 30; began as `uncertain`: 8 (an uncertain photo went to the larger sheets when its note matched the configured pattern, and was removed only if the second look said remove).

## By split

| split | reviewed | removed | removed % | uncertain |
|---|---|---|---|---|
| field_dev | 1040 | 12 | 1.2 | 128 |
| field_test | 1035 | 19 | 1.8 | 120 |

## By source

| source | split | reviewed | removed | removed % | uncertain |
|---|---|---|---|---|---|
| plantdoc | field_dev | 619 | 9 | 1.5 | 107 |
| plantdoc | field_test | 606 | 13 | 2.1 | 100 |
| plantseg | field_dev | 421 | 3 | 0.7 | 21 |
| plantseg | field_test | 429 | 6 | 1.4 | 20 |

## By class

| label | reviewed | removed | removed % | uncertain |
|---|---|---|---|---|
| Apple___Apple_scab | 126 | 2 | 1.6 | 14 |
| Apple___Black_rot | 30 | 0 | 0.0 | 4 |
| Apple___Cedar_apple_rust | 91 | 0 | 0.0 | 6 |
| Apple___healthy | 44 | 0 | 0.0 | 26 |
| Blueberry___healthy | 57 | 0 | 0.0 | 19 |
| Cherry_(including_sour)___Powdery_mildew | 12 | 2 | 16.7 | 1 |
| Cherry_(including_sour)___healthy | 28 | 0 | 0.0 | 8 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 63 | 1 | 1.6 | 1 |
| Corn_(maize)___Common_rust_ | 114 | 0 | 0.0 | 5 |
| Corn_(maize)___Northern_Leaf_Blight | 134 | 1 | 0.7 | 5 |
| Grape___Black_rot | 73 | 0 | 0.0 | 4 |
| Grape___healthy | 34 | 0 | 0.0 | 9 |
| Orange___Haunglongbing_(Citrus_greening) | 46 | 1 | 2.2 | 5 |
| Peach___healthy | 56 | 1 | 1.8 | 31 |
| Pepper,_bell___Bacterial_spot | 58 | 0 | 0.0 | 4 |
| Pepper,_bell___healthy | 30 | 1 | 3.3 | 4 |
| Potato___Early_blight | 68 | 0 | 0.0 | 4 |
| Potato___Late_blight | 82 | 0 | 0.0 | 3 |
| Raspberry___healthy | 60 | 1 | 1.7 | 14 |
| Soybean___healthy | 32 | 3 | 9.4 | 13 |
| Squash___Powdery_mildew | 136 | 0 | 0.0 | 7 |
| Strawberry___Leaf_scorch | 10 | 0 | 0.0 | 2 |
| Strawberry___healthy | 48 | 0 | 0.0 | 22 |
| Tomato___Bacterial_spot | 92 | 6 | 6.5 | 5 |
| Tomato___Early_blight | 99 | 2 | 2.0 | 5 |
| Tomato___Late_blight | 108 | 0 | 0.0 | 3 |
| Tomato___Leaf_Mold | 93 | 0 | 0.0 | 3 |
| Tomato___Septoria_leaf_spot | 110 | 1 | 0.9 | 5 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 64 | 4 | 6.2 | 3 |
| Tomato___Tomato_mosaic_virus | 45 | 0 | 0.0 | 2 |
| Tomato___healthy | 32 | 5 | 15.6 | 11 |

## By reason

| reason | removed |
|---|---|
| not_plant_photo | 17 |
| wrong_plant | 11 |
| label_contradicted | 3 |

## field_train, estimated from a seeded sample

A random sample of 400 of the `field_train` photos (seed 13), reviewed the same way and kept apart from the numbers above: 7 removed, 1.8%, Wilson 95% interval [0.9%, 3.6%]. Uncertain: 35. The point estimate is not above 5%; the interval's lower end is not above it.

## Effect on the earlier models' dev scores

Cleaned `field_dev` rows (exclusion version 1) against all rows, from the E13 run records (saved predictions filtered, so each pair is on the same models and the same photos apart from the removed ones; no model was rerun, no GPU). Only `field_dev` is scored; `field_test` waits for T8.1. Accuracy and macro-F1, old -> new.

| view | system | rows | accuracy | macro-F1 | record |
|---|---|---|---|---|---|
| shared_classes | e1 | 1040 -> 1028 | 0.2240 -> 0.2257 | 0.1990 -> 0.1978 | `E13-20261103T184414193293Z` |
| shared_classes | frozen_convnext | 1040 -> 1028 | 0.2519 -> 0.2539 | 0.2127 -> 0.2134 | `E13-20261103T184414193293Z` |
| shared_classes | xgboost | 1040 -> 1028 | 0.0413 -> 0.0418 | 0.0295 -> 0.0297 | `E13-20261103T184414193293Z` |
| shared_classes | dinov2_small | 1040 -> 1028 | 0.4067 -> 0.4076 | 0.3819 -> 0.3835 | `E13-20261103T184414193293Z` |
| shared_classes | convnext_joint | 1040 -> 1028 | 0.7644 -> 0.7695 | 0.7653 -> 0.7712 | `E13-20261103T184414193293Z` |
| shared_classes | dinov2_head | 1040 -> 1028 | 0.7462 -> 0.7500 | 0.7469 -> 0.7498 | `E13-20261103T184414193293Z` |
| heldout_plantdoc | e1 | 619 -> 610 | 0.2294 -> 0.2311 | 0.2251 -> 0.2242 | `E13-20261103T184438106036Z` |
| heldout_plantdoc | convnext_joint | 619 -> 610 | 0.7447 -> 0.7525 | 0.7322 -> 0.7398 | `E13-20261103T184438106036Z` |
| heldout_plantdoc | heldout_plantdoc | 619 -> 610 | 0.4330 -> 0.4377 | 0.3778 -> 0.3789 | `E13-20261103T184438106036Z` |
| heldout_plantseg | e1 | 421 -> 418 | 0.2162 -> 0.2177 | 0.1982 -> 0.1998 | `E13-20261103T184452515675Z` |
| heldout_plantseg | convnext_joint | 421 -> 418 | 0.7933 -> 0.7943 | 0.7752 -> 0.7757 | `E13-20261103T184452515675Z` |
| heldout_plantseg | heldout_plantseg | 421 -> 418 | 0.5463 -> 0.5478 | 0.5185 -> 0.5182 | `E13-20261103T184452515675Z` |
