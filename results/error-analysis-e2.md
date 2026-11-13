# E2 error analysis

50 wrong predictions on field_dev, seed 0, selected by `sprout.experiments.e2_review.select_mistakes` from the E2 field_dev record. Categories were assigned by viewing each photo; they describe what was seen in this sample and are not a proven cause of the errors. Where a category was a judgment call or the label looks wrong, the note says so; these notes are input for the H2 label review.

Scope: lab and real-photo scores are each computed on the classes the two datasets share and are not merged. field_dev is split from field_train by near-duplicate group, but a few same-scene images could remain across the field splits. The Grad-CAM overlays kept for field_dev are a different random sample from these mistakes, so no overlay was used for these categories.

`blur` here includes very small low-resolution images. `unsupported_condition` here covers fruit photos and a nutrient deficiency, which the leaf-trained classes do not cover.

Mask check: 16 of 40 sampled real-photo masks judged good (reliable needs all 40 judged and at least 80% good): NOT shown to be reliable, so the energy metric is lab only.

Categories: blur, background, similar_symptoms, wrong_label, unsupported_condition. Paths, categories and short notes only; no images are embedded.

| Category | Mistakes |
|---|---|
| blur | 5 |
| background | 12 |
| similar_symptoms | 15 |
| wrong_label | 9 |
| unsupported_condition | 9 |
| not viewed | 0 |
| total | 50 |

| Path | True class | Predicted | Confidence | Category | Note |
|---|---|---|---|---|---|
| datasets/raw/plantdoc/Apple Scab Leaf/applescab004.jpg | Apple___Apple_scab | Grape___Esca_(Black_Measles) | 0.1497 | blur | blurry low-res photo on grass; brown leaf edge only, scab label doubtful |
| datasets/raw/plantdoc/Apple rust leaf/601fbee78c62e32a76e768a92ee40193.jpg | Apple___Cedar_apple_rust | Apple___Black_rot | 0.5999 | similar_symptoms | purple-brown leaf spots read as black rot |
| datasets/raw/plantdoc/Bell_pepper leaf spot/H5w92.jpg | Pepper,_bell___Bacterial_spot | Pepper,_bell___healthy | 0.7712 | wrong_label | tan leaf-edge necrosis, no bacterial-spot lesions; label doubtful |
| datasets/raw/plantdoc/Bell_pepper leaf/uGz0O.jpg | Pepper,_bell___healthy | Grape___Black_rot | 0.3805 | background | whole plant on mulch, small leaves in cluttered scene; mostly healthy |
| datasets/raw/plantdoc/Blueberry leaf/ovalifolium1.jpg | Blueberry___healthy | Strawberry___Leaf_scorch | 0.7618 | similar_symptoms | small brown leaf spots on a blueberry branch read as scorch; judgment call |
| datasets/raw/plantdoc/Blueberry leaf/stock-photo-blueberry-leaves-branch-isolated-on-white-background-490111684.jpg | Blueberry___healthy | Strawberry___healthy | 0.8858 | background | stock photo on white with watermarks; healthy leaves taken for another species |
| datasets/raw/plantdoc/Blueberry leaf/vaccinium-myrtilloides-le-mlovit.jpg | Blueberry___healthy | Tomato___Late_blight | 0.5614 | similar_symptoms | red-tinted leaves read as late blight; judgment call |
| datasets/raw/plantdoc/Cherry leaf/100_0123_2.jpg | Cherry_(including_sour)___healthy | Pepper,_bell___healthy | 0.6123 | background | close-up branch on dark backdrop with watermark; healthy leaves; judgment call |
| datasets/raw/plantdoc/Corn leaf blight/Slide1.preview.jpg | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | 0.9397 | wrong_label | annotated slide about bacterial ooze and freckles, not northern leaf blight; label doubtful |
| datasets/raw/plantdoc/Corn leaf blight/anthracnoseblight.jpg | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.2989 | wrong_label | small lesions with halos and an anthracnose file name, not NLB cigar lesions; label doubtful |
| datasets/raw/plantdoc/Corn leaf blight/nclb3x1200.jpg | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | 0.4638 | background | NLB lesions on one leaf inside a dense corn field with ears and many leaves |
| datasets/raw/plantdoc/Potato leaf early blight/IMG_0014.jpg | Potato___Early_blight | Tomato___Late_blight | 0.8732 | wrong_label | blackened tissue looks like late blight rather than early blight; label doubtful, judgment call |
| datasets/raw/plantdoc/Potato leaf early blight/potato-blight-phytophthora-infestans-close-up-of-upper-surface-of-BMMRXC.jpg | Potato___Early_blight | Potato___Late_blight | 0.964 | wrong_label | file name says Phytophthora infestans which is late blight; model predicted potato late blight |
| datasets/raw/plantdoc/Raspberry leaf/IMG_0649.JPG.jpg | Raspberry___healthy | Strawberry___Leaf_scorch | 0.6257 | wrong_label | yellowing and browning senescent leaves labeled healthy |
| datasets/raw/plantdoc/Raspberry leaf/depositphotos_22512509-stock-photo-raspberry-leaf.jpg | Raspberry___healthy | Strawberry___healthy | 0.6171 | background | stock photo on white with watermark; healthy leaves |
| datasets/raw/plantdoc/Raspberry leaf/depositphotos_2795312-stock-photo-green-raspberry-leaf.jpg | Raspberry___healthy | Tomato___Late_blight | 0.5175 | background | stock photo on white with watermark; healthy leaf |
| datasets/raw/plantdoc/Soyabean leaf/kdeficiency4.jpg | Soybean___healthy | Strawberry___Leaf_scorch | 0.7724 | unsupported_condition | file name says potassium deficiency, labeled healthy; not a supported condition |
| datasets/raw/plantdoc/Strawberry leaf/7.jpg | Strawberry___healthy | Strawberry___Leaf_scorch | 0.7495 | background | healthy leaf among grass blades |
| datasets/raw/plantdoc/Strawberry leaf/img_20110302_170719.jpg | Strawberry___healthy | Strawberry___Leaf_scorch | 0.5225 | background | leaf held against a hand among litter; plant may not be strawberry |
| datasets/raw/plantdoc/Tomato Early blight leaf/796984c81925.jpg | Tomato___Early_blight | Tomato___Late_blight | 0.9693 | similar_symptoms | large brown lesion with yellow halo read as late blight |
| datasets/raw/plantdoc/Tomato Septoria leaf spot/5008922249_880e15c21f.jpg | Tomato___Septoria_leaf_spot | Grape___Black_rot | 0.9315 | wrong_label | glossy round-spotted leaf does not look like a tomato leaf; label doubtful |
| datasets/raw/plantdoc/Tomato leaf bacterial spot/3182.jpg | Tomato___Bacterial_spot | Strawberry___Leaf_scorch | 0.738 | similar_symptoms | small dark bacterial-spot lesions read as leaf scorch |
| datasets/raw/plantdoc/Tomato leaf bacterial spot/Bacterial+Spot+-+Tomatoes.jpg | Tomato___Bacterial_spot | Tomato___Leaf_Mold | 0.5845 | blur | very small low-res image about 320 px; mottled leaves, label doubtful |
| datasets/raw/plantdoc/Tomato leaf mosaic virus/DSC02246.JPG.jpg | Tomato___Tomato_mosaic_virus | Tomato___Late_blight | 0.9968 | similar_symptoms | mild leaf curl and mottling held in a hand read as late blight; judgment call |
| datasets/raw/plantdoc/Tomato leaf yellow virus/dscn1652cropped.jpg | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Strawberry___Leaf_scorch | 0.3171 | background | curled top leaves inside a whole greenhouse crop scene |
| datasets/raw/plantdoc/Tomato leaf/tomato-leaf-26180166.jpg | Tomato___healthy | Strawberry___healthy | 0.2664 | background | stock photo on white with watermark; healthy leaf |
| datasets/raw/plantdoc/Tomato mold leaf/leaf_mold2.jpg | Tomato___Leaf_Mold | Tomato___Late_blight | 0.9797 | similar_symptoms | yellow leaf-mold patches read as late blight |
| datasets/raw/plantdoc/grape leaf black rot/220px-Guignardia_bidwellii_08.jpg | Grape___Black_rot | Tomato___Late_blight | 0.9957 | blur | tiny 220 px image |
| datasets/raw/plantseg_field/apple black rot/apple_black_rot_google_0001.jpg | Apple___Black_rot | Strawberry___Leaf_scorch | 0.606 | similar_symptoms | brown leaf spots on cluttered tree read as leaf scorch |
| datasets/raw/plantseg_field/apple black rot/apple_black_rot_google_0080.jpg | Apple___Black_rot | Strawberry___Leaf_scorch | 0.7377 | unsupported_condition | fruit photo not a leaf; label doubtful, lesion looks like rust or similar |
| datasets/raw/plantseg_field/apple rust/apple_rust_151.jpg | Apple___Cedar_apple_rust | Squash___Powdery_mildew | 0.506 | similar_symptoms | orange rust spots on leaf that also has a powdery coating, read as powdery mildew |
| datasets/raw/plantseg_field/apple scab/apple_scab_15.jpg | Apple___Apple_scab | Strawberry___Leaf_scorch | 0.4931 | unsupported_condition | fruit photo, looks like pears; not a leaf; label doubtful |
| datasets/raw/plantseg_field/apple scab/apple_scab_82.jpg | Apple___Apple_scab | Apple___healthy | 0.2052 | unsupported_condition | fruit photo with Chinese caption; not a leaf |
| datasets/raw/plantseg_field/apple scab/apple_scab_google_0086.jpg | Apple___Apple_scab | Strawberry___Leaf_scorch | 0.3584 | unsupported_condition | fruit photo; not a leaf |
| datasets/raw/plantseg_field/apple scab/apple_scab_google_0270.jpg | Apple___Apple_scab | Peach___Bacterial_spot | 0.5912 | unsupported_condition | pear fruit photo; not an apple leaf; label doubtful |
| datasets/raw/plantseg_field/cherry powdery mildew/cherry_powdery_mildew_44.jpg | Cherry_(including_sour)___Powdery_mildew | Tomato___Early_blight | 0.5057 | wrong_label | leaflets look like rose rather than cherry though mildew is visible; label doubtful |
| datasets/raw/plantseg_field/citrus greening disease/citrus_greening_disease_11.jpg | Orange___Haunglongbing_(Citrus_greening) | Tomato___Late_blight | 0.7225 | background | whole young tree with soil and pipe; yellow mottling is typical of the label |
| datasets/raw/plantseg_field/citrus greening disease/citrus_greening_disease_93.jpg | Orange___Haunglongbing_(Citrus_greening) | Pepper,_bell___healthy | 0.4806 | unsupported_condition | fruit and leaves on tree with a hand, no clear leaf symptom; fruit photo |
| datasets/raw/plantseg_field/corn northern leaf blight/corn_northern_leaf_blight_54.jpg | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | 0.769 | similar_symptoms | dead tan streaks read as healthy or senescent tissue; judgment call |
| datasets/raw/plantseg_field/corn rust/corn_rust_google_0007.jpg | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | 0.3676 | blur | low-res image about 360 px with dense rust pustules |
| datasets/raw/plantseg_field/corn rust/corn_rust_google_0160.jpg | Corn_(maize)___Common_rust_ | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.5988 | similar_symptoms | macro slide with dark-rimmed pustules read as gray leaf spot; judgment call |
| datasets/raw/plantseg_field/grape black rot/grape_black_rot_23.jpg | Grape___Black_rot | Strawberry___Leaf_scorch | 0.8364 | unsupported_condition | grape berries not a leaf |
| datasets/raw/plantseg_field/potato late blight/potato_late_blight_13.jpg | Potato___Late_blight | Tomato___Early_blight | 0.5722 | similar_symptoms | late blight leaf with grey mould read as tomato early blight |
| datasets/raw/plantseg_field/tomato bacterial leaf spot/tomato_bacterial_leaf_spot_32.jpg | Tomato___Bacterial_spot | Tomato___Septoria_leaf_spot | 0.7387 | similar_symptoms | small dark spots read as Septoria leaf spot |
| datasets/raw/plantseg_field/tomato early blight/tomato_early_blight_6.jpg | Tomato___Early_blight | Tomato___Late_blight | 0.9996 | similar_symptoms | single tan lesion with chlorosis read as late blight |
| datasets/raw/plantseg_field/tomato late blight/tomato_late_blight_220.jpg | Tomato___Late_blight | Strawberry___Leaf_scorch | 0.6766 | unsupported_condition | fruit on the ground not a leaf |
| datasets/raw/plantseg_field/tomato leaf mold/tomato_leaf_mold_13.jpg | Tomato___Leaf_Mold | Tomato___Late_blight | 0.9867 | similar_symptoms | brown necrotic patches with watermark read as late blight; label doubtful |
| datasets/raw/plantseg_field/tomato mosaic virus/tomato_mosaic_virus_147.jpg | Tomato___Tomato_mosaic_virus | Pepper,_bell___healthy | 0.3267 | background | wide scene of stems and flowers with no clear mosaic; label doubtful |
| datasets/raw/plantseg_field/tomato mosaic virus/tomato_mosaic_virus_230.jpg | Tomato___Tomato_mosaic_virus | Tomato___Late_blight | 0.7737 | wrong_label | large leaves look like tobacco not tomato; label doubtful |
| datasets/raw/plantseg_field/tomato yellow leaf curl virus/tomato_yellow_leaf_curl_virus_google_0000.jpg | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Late_blight | 0.4129 | blur | small low-res image about 330 px of a whole plant |
