# Document coverage

For each class the classifier can output: does a document in `knowledge/manifest.csv`
describe its symptoms? Judged by reading the indexed passages for that crop and condition,
not from the manifest's disease list. The evidence is in `docs/document-coverage.csv`; this
page is rendered from it. A disease named only in a spray table or a passing comparison does
not count. A healthy class has no symptoms to describe and is marked not applicable.

Yes: 10. No: 16. Not applicable: 12. Crops with at least one document: apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, strawberry, tomato.

| label | crop | symptoms_described | document | page | evidence |
|---|---|---|---|---|---|
| Apple___Apple_scab | apple | yes | Apple.pdf | 19 | SYMPTOMS of scab: on leaves early infections appear velvety, olive-green and become puckered; spots first on the underside. |
| Apple___Black_rot | apple | yes | Apple.pdf | 26;28 | Frogeye leaf spot (same fungus as black rot): small purple specks 1-3 weeks after petal fall enlarge; p28 black rot fruit symptom: brown to black spot at blossom end. |
| Apple___Cedar_apple_rust | apple | yes | Apple.pdf | 25 | SYMPTOMS: small pale yellow spots on upper leaf surface enlarge to 1/4 inch and become bright yellow-orange; orange-yellow tubular fruiting structures on lower surface. |
| Apple___healthy | apple | not applicable |  |  | A healthy class has no symptoms to describe. |
| Blueberry___healthy | blueberry | not applicable |  |  | A healthy class has no symptoms to describe. |
| Cherry_(including_sour)___Powdery_mildew | cherry | no | Cherry-em8329.pdf; 13-cherry.pdf | 6;4;11;14 | Mentioned only in spray tables and a note ("can be a problem in some years in western Oregon", "include an appropriate fungicide"). No description of what the mildew looks like. |
| Cherry_(including_sour)___healthy | cherry | not applicable |  |  | A healthy class has no symptoms to describe. |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | corn | no | S-0003-45-Corn.pdf | 2 | Only a comparison that spots are "a regular shape, such as gray leaf spot". Corn1.pdf lists it in a management table. No symptom description. |
| Corn_(maize)___Common_rust_ | corn | no | S-0003-45-Corn.pdf; Corn1.pdf | 3;4 | Named in a severity-threshold sentence and a management table. No symptom description. |
| Corn_(maize)___Northern_Leaf_Blight | corn | no | S-0003-45-Corn.pdf; Corn1.pdf | 2;3;4 | Named in "irregular in shape" comparison, a yield-loss example and a management table. No symptom description. |
| Corn_(maize)___healthy | corn | not applicable |  |  | A healthy class has no symptoms to describe. |
| Grape___Black_rot | grape | yes | grapediseaseidguide.pdf; winegrapes8.pdf | 34;3 | Leaf lesions small (2 to 10 mm), tan and circular with a dark brown border, with tiny black pycnidia; winegrapes8 p3 repeats tan circular leaf spots. |
| Grape___Esca_(Black_Measles) | grape | yes | grapediseaseidguide.pdf; Grape-343478.pdf | 32;1 | Black Measles symptoms: leaves have small chlorotic areas between veins that enlarge and dry out; small dark spots with brownish-purple rings on berries. Grape-343478 p1 states esca is also known as black measles. |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | grape | no | grapediseaseidguide.pdf | 13;17;21 | "Leaf blight (Pseudocercospora vitis)" appears only in fungicide target lists. No symptom description. |
| Grape___healthy | grape | not applicable |  |  | A healthy class has no symptoms to describe. |
| Orange___Haunglongbing_(Citrus_greening) | orange | yes | CitrusOrange-Greening-Guide.pdf; OrangeFarming.pdf | 1;2;4;7 | Symptoms for detection: blotchy mottling and yellowing of leaves, bunched narrow leaves, reduced fruit size; OrangeFarming p4 asymmetrical blotchy leaf mottle, yellow shoots, dieback. |
| Peach___Bacterial_spot | peach | yes | BacterialSpotPeach.pdf | 1 | Distinguished by angular leaf lesions; fruit symptoms start as small circular brown spots; severe defoliation. |
| Peach___healthy | peach | not applicable |  |  | A healthy class has no symptoms to describe. |
| Pepper,_bell___Bacterial_spot | pepper | yes | PP362_PP362_ Bacterial Spot of Pepper.pdf | 1;2 | Symptoms: water-soaked spots on the lower leaf surface that expand to yellow and brown areas with a greasy look; fruit lesions raised warts or scabs. |
| Pepper,_bell___healthy | pepper | not applicable |  |  | A healthy class has no symptoms to describe. |
| Potato___Early_blight | potato | yes | Potato - early-blight-bulletin.pdf; potato.pdf | 1;2 | Small irregular to circular dark brown spots on lower leaves, angular, with concentric rings and a chlorotic halo. |
| Potato___Late_blight | potato | no | Potato - early-blight-bulletin.pdf; potato.pdf | 1;3;5 | Late blight appears only as a contrast ("unlike late blight lesions, they are angular") and as a fungicide note. No description of late blight itself. |
| Potato___healthy | potato | not applicable |  |  | A healthy class has no symptoms to describe. |
| Raspberry___healthy | none | not applicable |  |  | A healthy class has no symptoms to describe. |
| Soybean___healthy | none | not applicable |  |  | A healthy class has no symptoms to describe. |
| Squash___Powdery_mildew | none | no |  |  | No squash or cucurbit document in knowledge/manifest.csv. |
| Strawberry___Leaf_scorch | strawberry | yes | strawberry.pdf; strawberry2.pdf | 1;2 | Leaf spots: pinpoint or blotchy lesions 1/4 to 1/2 inch, reddish to purple, coalescing to a burnt look; petiole and fruit stalk lesions. |
| Strawberry___healthy | strawberry | not applicable |  |  | A healthy class has no symptoms to describe. |
| Tomato___Bacterial_spot | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named only in seed-treatment advice and a pest list; no leaf or fruit description. PP362 describes pepper, not tomato. |
| Tomato___Early_blight | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named in tomatodoc1 as a disease with resistant varieties and in tomatodoc2 spray tables. The early blight bulletins are for potato, not tomato. |
| Tomato___Late_blight | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named in tomatodoc1 resistant-variety list and tomatodoc2 tables. No symptom description. |
| Tomato___Leaf_Mold | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named in tomatodoc1 resistant-variety list and as "Cladisporium Leaf Mold" in tomatodoc2 ranking. No symptom description. |
| Tomato___Septoria_leaf_spot | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named only in tomatodoc2 spray and cultural-control tables. No symptom description. |
| Tomato___Spider_mites Two-spotted_spider_mite | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | tomatodoc1 p1 says two-spotted spider mite is the most common mite pest and how to control it; it does not describe damage on the plant. |
| Tomato___Target_Spot | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named only in a tomatodoc2 pest ranking list. No symptom description. |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | TYLCV named only as whitefly-transmitted in tomatodoc1. No symptom description. |
| Tomato___Tomato_mosaic_virus | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | tomatodoc2 gives one generic line that most virus diseases cause stunting, leaf distortion, mosaic discoloration; tomatodoc1 names TMV as aphid-transmitted. Nothing specific to tomato mosaic virus. |
| Tomato___healthy | tomato | not applicable |  |  | A healthy class has no symptoms to describe. |

