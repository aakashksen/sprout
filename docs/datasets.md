# Datasets

Checked 2026-11-02 by researcher lookups for T0.2. Standing rule (from `CLAUDE.md`): use a dataset only if a primary source records a license allowing non-commercial research use, with no signed agreement. Otherwise `Approved: no, needs the person`, and do not download. Nothing was downloaded or installed.

Note: read directly (raw HTTP, not summarized): the PlantDoc `LICENSE.txt` and GitHub repo API record, the Zenodo API records for PlantSeg, and PlantVillage `README_HF.md`. Everything else, including https://plantvillage.psu.edu/tos and the Open Images page, came through a summarizing fetch tool, so its quoted wording should be checked against the source page before H1.

## Decisions by the person (2026-11-02)

URL: not applicable
License: See each dataset's section.
Classes: not applicable
1. PlantVillage: official 38-class release, CC BY-SA 3.0.
2. `not_plant`: COCO 2017 validation, local non-commercial evaluation, potted-plant images excluded, images never committed or redistributed.
3. PlantDoc: local evaluation only, no redistribution. Updated 2026-11-02: T4.1 may also fine-tune on PlantDoc images for local use. The images are not republished, and the person checks before publishing any weights trained on them.
4. H1 is not yet signed. The person signs it at the end.
Approved: yes, these are the person's recorded decisions

## PlantVillage (official 38-class release, used)

URL: https://github.com/spMohanty/PlantVillage-Dataset (`raw/color`, pinned to commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`)
License: CC BY-SA 3.0, per the repo's `README_HF.md` YAML (`license: cc-by-sa-3.0`), read directly (https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/README_HF.md). The repo has no LICENSE file, and the GitHub API reports no license. PlantVillage's terms also state CC BY-SA 3.0 for its own content (https://plantvillage.psu.edu/tos, read through a summarizing fetch). ShareAlike applies to anything derived from the data.
Size: Official README: 54,306 images. T1.1 records the actual counts in `docs/data-counts.md`.
Classes: 38 folders in `raw/color`, read from the GitHub API, including healthy classes and with no background class.
Access: Fetch the repo at the pinned commit into the git-ignored `datasets/raw/_pv_upstream`. Original filenames carry a source ID, for example `00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG`. Checked on one class folder (`Apple___Apple_scab`, 630 files); T1.1 checks the rest.
Approved: yes, by the person on 2026-11-02, under CC BY-SA 3.0

## PlantVillage (local 39-class copy, not used)

URL: https://github.com/spMohanty/PlantVillage-Dataset (official repo, `raw/color`, `raw/grayscale`, `raw/segmented`). Local copy: `<local path> Project\pytorchConvNextMulticlassClassification\9layerplantdatacnn\data\noaug\noaug data` (origin not recorded). Re-host with matching counts (secondary, not a license source): https://huggingface.co/datasets/Project-AgML/plant_village_classification
License: Official repo: no LICENSE file in the repo root listing (https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/) and no license section in its README (https://github.com/spMohanty/PlantVillage-Dataset). The repo's own `README_HF.md` YAML states `license: cc-by-sa-3.0` (https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/README_HF.md). PlantVillage's terms (https://plantvillage.psu.edu/tos, Section 8) say content made available by PlantVillage on its websites, apps or services is "licensed under the Creative Commons Attribution-ShareAlike 3.0 License, unless marked otherwise." CC BY-SA 3.0 permits non-commercial use. For the local 39-class copy, I could not confirm a license: the Project-AgML card has no license field, and the 39th class (`Background_without_leaves`) is not in the official repo, so its origin is untraced. A search snippet claimed CC0 on a third-party page; I did not open that page and did not use it.
Size: Local `noaug`: 39 folders, 55,448 image files (52,803 `.JPG`, 2,643 `.jpg`, 1 `.jpeg`, 1 `.png`). Official README: 54,306 images (https://github.com/spMohanty/PlantVillage-Dataset). Local `aug` sibling (not used for training): 39 folders, 61,486 files, same folder names; its license and origin were not checked.
Classes: 39 folders, same names as the local copy: Apple___Apple_scab (630), Apple___Black_rot (621), Apple___Cedar_apple_rust (275), Apple___healthy (1645), Background_without_leaves (1143), Blueberry___healthy (1502), Cherry___Powdery_mildew (1052), Cherry___healthy (854), Corn___Cercospora_leaf_spot Gray_leaf_spot (513), Corn___Common_rust (1192), Corn___Northern_Leaf_Blight (985), Corn___healthy (1162), Grape___Black_rot (1180), Grape___Esca_(Black_Measles) (1383), Grape___Leaf_blight_(Isariopsis_Leaf_Spot) (1076), Grape___healthy (423), Orange___Haunglongbing_(Citrus_greening) (5507), Peach___Bacterial_spot (2297), Peach___healthy (360), Pepper,_bell___Bacterial_spot (997), Pepper,_bell___healthy (1478), Potato___Early_blight (1000), Potato___Late_blight (1000), Potato___healthy (152), Raspberry___healthy (371), Soybean___healthy (5090), Squash___Powdery_mildew (1835), Strawberry___Leaf_scorch (1109), Strawberry___healthy (456), Tomato___Bacterial_spot (2127), Tomato___Early_blight (1000), Tomato___Late_blight (1909), Tomato___Leaf_Mold (952), Tomato___Septoria_leaf_spot (1771), Tomato___Spider_mites Two-spotted_spider_mite (1676), Tomato___Target_Spot (1404), Tomato___Tomato_Yellow_Leaf_Curl_Virus (5357), Tomato___Tomato_mosaic_virus (373), Tomato___healthy (1591). Per-folder counts match the Project-AgML card exactly for all 39 classes (https://huggingface.co/datasets/Project-AgML/plant_village_classification). The official `raw/color` has 38 folders with different names (for example `Cherry_(including_sour)___Powdery_mildew`, `Corn_(maize)___...`) and no `Background_without_leaves` (https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color). So the local copy is a 39-class re-release, not the official 38-class release.
Access: Local copy is already on disk, read-only. Official: `git clone https://github.com/spMohanty/PlantVillage-Dataset` (not run). The 39th class's origin still needs tracing before approval.
Approved: no, needs the person. Not used: the person chose the official release on 2026-11-02

## PlantDoc

URL: https://github.com/pratikkayal/PlantDoc-Dataset
License: LICENSE.txt was read directly and begins "Attribution 4.0 International" (https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/LICENSE.txt), and the GitHub API reports `license.spdx_id` CC-BY-4.0 (https://api.github.com/repos/pratikkayal/PlantDoc-Dataset). The repo README also names CC BY 4.0. This is the authors' repo (owner `pratikkayal`; I did not check the paper's author list). The repo hosts Cropped-PlantDoc only; the object-detection version is linked separately, and its license was not checked. The images were scraped from the internet, so per-image rights were not checked; this records the license the dataset publishes. Whether the unchecked per-image rights of scraped images are acceptable is for the person to decide at H1.
Size: 2,598 data points in total (README: "Our dataset contains 2,598 data points in total across 13 plant species and up to 17 classes of diseases"). A Roboflow mirror lists 2,569 images and 30 classes (https://universe.roboflow.com/sherif/plantdoc-cbl3c-wvbns); that is secondary and conflicts, so it was not used.
Classes: "up to 17 classes of diseases across 13 plant species" (repo README). Per-class counts not given on the page read.
Access: Browse or clone https://github.com/pratikkayal/PlantDoc-Dataset (not run). Top-level folders: `train/` and `test/`.
Downloaded 2026-11-02 by `git clone --depth 1` into `datasets/raw/plantdoc_src`. 94 files have names too long for a Windows path, so they were written under a 12-character hash name; 2,572 images in 28 class folders (2,336 in train and 236 in test), staged flat per class in `datasets/raw/plantdoc`.
Approved: yes, by the person on 2026-11-02, for local evaluation and T4.1 fine-tuning, with no redistribution. The images are never committed, and the person checks before publishing any weights trained on them

## PlantSeg (second real-photo source, in-the-wild)

URL: https://github.com/tqwei05/PlantSeg ; Zenodo record https://zenodo.org/records/16665770 ; DOI https://doi.org/10.5281/zenodo.17719108 ; paper https://arxiv.org/abs/2409.04038
License: Use record 17719108, the newest version (the DOI above resolves to it; confirmed via https://zenodo.org/api/records/16665770/versions/latest, `metadata.license` = `cc-by-nc-4.0`). That is CC BY-NC 4.0 (non-commercial). The older record 16665770 (v6, July 31, 2025) showed CC BY 4.0. The non-commercial term covers this project's research use, but it would need rechecking for any commercial use. The record's creators are the paper authors (Tianqi Wei et al.), so this is the authors' own publication. The GitHub repo has no License section or LICENSE file (https://github.com/tqwei05/PlantSeg).
Size: One file, `plantseg.zip`, 1.7 GB (older record 16665770; size not rechecked on 17719108). Image count conflicts: GitHub README says "more than 7,774 images" (https://github.com/tqwei05/PlantSeg); the arXiv abstract says "11,400 images with disease segmentation masks and an additional 8,000 healthy plant images" (https://arxiv.org/abs/2409.04038). Not reconciled; counted after download in T1.2 (below).
Counted 2026-11-02 after download (md5 matched the Zenodo record; `plantseg.zip` is 1,057,281,724 bytes, not 1.7 GB): 7,774 images (all diseased, no healthy set in this file), 34 plants, 115 plant-disease classes, per-image licenses 7,741 CC-BY-NC and 33 CC0 in `Metadata.csv`. The 11,400 and 8,000-healthy figures in the arXiv abstract are not in this file. T1.2 maps 21 of the 115 classes to lab classes (`datasets/class_map.csv`).
Access: Download `plantseg.zip` from Zenodo record 17719108 (https://zenodo.org/records/17719108), then place it under the `data` folder as the repo README says (not done). Each image has a segmentation mask of the diseased part.
Approved: yes

## not_plant: COCO 2017 validation (used)

URL: https://cocodataset.org ; images http://images.cocodataset.org/zips/val2017.zip ; annotations http://images.cocodataset.org/annotations/annotations_trainval2017.zip
License: Couldn't confirm from a primary source: https://cocodataset.org/termsofuse returned HTTP 404 and the homepage has no license text. Used because the person approved it directly on 2026-11-02 for local non-commercial evaluation; this approval is the person's decision and not a license finding.
Size: Annotations zip is 252,907,541 bytes (HTTP Content-Length). Counted 2026-11-02: 5,000 validation images, 4,828 left after excluding every image with a `potted plant` annotation (172 excluded).
Classes: Not used as classes. All images are treated as `not_plant`, after excluding every image annotated with the `potted plant` category.
Access: Download the two zips above into the git-ignored `datasets/raw/`. Never commit or redistribute the images or anything derived from them.
Approved: yes, by the person on 2026-11-02, for local non-commercial evaluation only

## Open Images (not_plant candidate, not used)

URL: https://storage.googleapis.com/openimages/web/factsfigures_v7.html (Google's Open Images page; the CVDF repo at https://github.com/cvdfoundation/open-images-dataset was also read, but has no license text)
License: "The annotations are licensed by Google LLC under CC BY 4.0 license." "The images are listed as having a CC BY 2.0 license." Google adds: "we make no representations or warranties regarding the license status of each image" and "you should verify the license for each image yourself" (https://storage.googleapis.com/openimages/web/factsfigures_v7.html). Image rights are per image, so no single dataset license permits non-commercial research use for the images. Not approved.
Size: train 9,011,219 images; validation 41,620; test 125,436 (same page).
Classes: not checked. A usable `not_plant` subset would need label filtering that has not been designed.
Access: Not checked; not downloaded.
Approved: no, needs the person. Not used: the person chose COCO 2017 validation on 2026-11-02

## Segmented PlantVillage images

URL: https://github.com/spMohanty/PlantVillage-Dataset (folder `raw/segmented`; listing https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/segmented)
License: Same repo and approval as the official release above: CC BY-SA 3.0.
Size: Not downloaded.
Classes: The official `raw/segmented` has 38 class folders (the same 38 names as `raw/color`, with no `Background_without_leaves`).
Access: Browse or clone the official repo (not run).
Approved: yes, by the person on 2026-11-02, as part of the official release. Used for T4.1

Finding: No segmented PlantVillage images are on this machine. `<local path>` is an empty directory (zero entries). The local `noaug` and `aug` folders hold color photos: one sample, `Apple___Apple_scab\image (1).JPG`, shows a leaf on a plain gray background, not a black-background segmented image. That was a single-image check, not a full survey. The official repo does have a `raw/segmented` folder with 38 class folders, and on the one class checked, `Apple___Apple_scab`, its 630 files match `raw/color` by name with a `_final_masked.jpg` suffix.

## Local serving and fine-tuning model: Qwen3.5-4B

URL: https://huggingface.co/Qwen/Qwen3.5-4B
License: Apache-2.0 (Hugging Face API: `apache-2.0`, not gated).
Classes: none; it's a language and vision model.
Use: the local model through Ollama, as `qwen3.5:4b-q4_K_M`. See `docs/model-choice.md`.
Approved: yes. Reason: Apache-2.0, ungated, no signed agreement.
