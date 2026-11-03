# Document coverage

For each class the classifier can output: does a document in `knowledge/manifest.csv`
describe its symptoms? Judged by reading the indexed passages for that crop and condition,
not from the manifest's disease list. The evidence is in `docs/document-coverage.csv`; this
page is rendered from it. A disease named only in a spray table or a passing comparison does
not count. A healthy class has no symptoms to describe and is marked not applicable.

Yes: 13. No: 13. Not applicable: 12. Crops with at least one document: apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, strawberry, tomato.

| label | crop | symptoms_described | document | page | evidence |
|---|---|---|---|---|---|
| Apple___Apple_scab | apple | yes | Apple.pdf | 19 | SYMPTOMS of scab: on leaves early infections appear velvety, olive-green and become puckered; spots first on the underside. |
| Apple___Black_rot | apple | yes | Apple.pdf | 26;28 | Frogeye leaf spot (same fungus as black rot): small purple specks 1-3 weeks after petal fall enlarge; p28 black rot fruit symptom: brown to black spot at blossom end. |
| Apple___Cedar_apple_rust | apple | yes | Apple.pdf | 25 | SYMPTOMS: small pale yellow spots on upper leaf surface enlarge to 1/4 inch and become bright yellow-orange; orange-yellow tubular fruiting structures on lower surface. |
| Apple___healthy | apple | not applicable |  |  | A healthy class has no symptoms to describe. |
| Blueberry___healthy | blueberry | not applicable |  |  | A healthy class has no symptoms to describe. |
| Cherry_(including_sour)___Powdery_mildew | cherry | no | Cherry-em8329.pdf; 13-cherry.pdf | 6;4;11;14 | Spray tables and notes only ("can be a problem in some years in western Oregon", "include an appropriate fungicide"). Queries (crops cherry and general): powdery mildew; podosphaera; sphaerotheca; mildew; white powdery growth. Apple documents describe apple powdery mildew, not cherry, so they do not count. |
| Cherry_(including_sour)___healthy | cherry | not applicable |  |  | A healthy class has no symptoms to describe. |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | corn | no | S-0003-45-Corn.pdf; Corn1.pdf | 2;4 | Only a comparison that spots have "a regular shape, such as gray leaf spot" and a management-table row. Queries (crops corn and general): gray leaf spot; cercospora; zeae; helminthosporium; leaf spot. No symptom description. |
| Corn_(maize)___Common_rust_ | corn | no | S-0003-45-Corn.pdf; Corn1.pdf | 3;4 | Named in a severity-threshold sentence and a management table. Queries (crops corn and general): common rust; puccinia; sorghi; pustule; rust; southern rust. No symptom description. |
| Corn_(maize)___Northern_Leaf_Blight | corn | no | S-0003-45-Corn.pdf; Corn1.pdf | 2;3;4 | Named in an "irregular in shape" comparison, a yield-loss example and a management table. Queries (crops corn and general): northern leaf blight; northern corn leaf blight; turcicum; exserohilum; setosphaeria; helminthosporium. No symptom description. |
| Corn_(maize)___healthy | corn | not applicable |  |  | A healthy class has no symptoms to describe. |
| Grape___Black_rot | grape | yes | grapediseaseidguide.pdf; winegrapes8.pdf | 34;3 | Leaf lesions small (2 to 10 mm), tan and circular with a dark brown border, with tiny black pycnidia; winegrapes8 p3 repeats tan circular leaf spots. |
| Grape___Esca_(Black_Measles) | grape | yes | grapediseaseidguide.pdf; Grape-343478.pdf | 32;1 | Black Measles symptoms: leaves have small chlorotic areas between veins that enlarge and dry out; small dark spots with brownish-purple rings on berries. Grape-343478 p1 states esca is also known as black measles. |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | grape | yes | winegrapes8.pdf | 12 | Section "Leaf Blight, Isariopsis Leaf Spot" (Pseudocercospora vitis, formerly Isariopsis clavispora): large, irregular shaped spots, initially dull red to brown, turning black and brittle with age. grapediseaseidguide.pdf mentions it only in fungicide lists. |
| Grape___healthy | grape | not applicable |  |  | A healthy class has no symptoms to describe. |
| Orange___Haunglongbing_(Citrus_greening) | orange | yes | CitrusOrange-Greening-Guide.pdf; OrangeFarming.pdf | 1;2;4;7 | Symptoms for detection: blotchy mottling and yellowing of leaves, bunched narrow leaves, reduced fruit size; OrangeFarming p4 asymmetrical blotchy leaf mottle, yellow shoots, dieback. |
| Peach___Bacterial_spot | peach | yes | BacterialSpotPeach.pdf | 1 | Distinguished by angular leaf lesions; fruit symptoms start as small circular brown spots; severe defoliation. |
| Peach___healthy | peach | not applicable |  |  | A healthy class has no symptoms to describe. |
| Pepper,_bell___Bacterial_spot | pepper | yes | PP362_PP362_ Bacterial Spot of Pepper.pdf | 1;2 | Symptoms: water-soaked spots on the lower leaf surface that expand to yellow and brown areas with a greasy look; fruit lesions raised warts or scabs. |
| Pepper,_bell___healthy | pepper | not applicable |  |  | A healthy class has no symptoms to describe. |
| Potato___Early_blight | potato | yes | Potato - early-blight-bulletin.pdf; potato.pdf | 1;2 | Small irregular to circular dark brown spots on lower leaves, angular, with concentric rings and a chlorotic halo. |
| Potato___Late_blight | potato | no | Potato - early-blight-bulletin.pdf; potato.pdf | 1;3;5 | Late blight appears only as a contrast ("unlike late blight lesions, they are angular") and as a fungicide note. Queries (crops potato, tomato and general): late blight; phytophthora infestans; infestans. No description of late blight itself. |
| Potato___healthy | potato | not applicable |  |  | A healthy class has no symptoms to describe. |
| Raspberry___healthy | none | not applicable |  |  | A healthy class has no symptoms to describe. |
| Soybean___healthy | none | not applicable |  |  | A healthy class has no symptoms to describe. |
| Squash___Powdery_mildew | none | no |  |  | No squash or cucurbit document in knowledge/manifest.csv. Queries (all crops): squash; cucurbit; zucchini; pumpkin; cucumber; podosphaera xanthii. Only hits are cucumber mosaic virus in the tomato documents. |
| Strawberry___Leaf_scorch | strawberry | yes | strawberry.pdf; strawberry2.pdf | 1;2 | Leaf spots: pinpoint or blotchy lesions 1/4 to 1/2 inch, reddish to purple, coalescing to a burnt look; petiole and fruit stalk lesions. |
| Strawberry___healthy | strawberry | not applicable |  |  | A healthy class has no symptoms to describe. |
| Tomato___Bacterial_spot | tomato | yes | tomatodoc2.md | 1 | Bacterial Spot section: leaf lesions begin as small water-soaked spots that become necrotic and brown in the center, sunken on the upper surface and raised on the lower; spots coalesce into chlorosis; fruit lesions small, round, dark brown to black. |
| Tomato___Early_blight | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named in tomatodoc1 as a disease with resistant varieties and in tomatodoc2 spray tables only. Queries (crops tomato and general): early blight; alternaria; concentric. The early blight bulletins are for potato and the crop filter keeps them off tomato. |
| Tomato___Late_blight | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named in tomatodoc1 resistant-variety list and a tomatodoc2 table. Queries (crops tomato and general): late blight; phytophthora infestans; infestans. No symptom description. |
| Tomato___Leaf_Mold | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Named in tomatodoc1 resistant-variety list and as "Cladisporium Leaf Mold" in the tomatodoc2 ranking. Queries (crops tomato and general): leaf mold; cladospor; fulvia; mold. The only mold passages are sooty mold from honeydew and white mold of southern stem rot. No symptom description. |
| Tomato___Septoria_leaf_spot | tomato | no | tomatodoc2.md | 1 | Named only in tomatodoc2 spray and cultural-control tables. Queries (crops tomato and general): septoria; leaf spot; cercospora. No symptom description. |
| Tomato___Spider_mites Two-spotted_spider_mite | tomato | yes | tomatodoc2.md | 1 | Spider mite profile: leaves lightly stippled with pale blotches, then light in color, drying and reddish-brown with webbing; speckled appearance; wilting, leaf deformity, desiccation; fruit bronzing or russeting. |
| Tomato___Target_Spot | tomato | no | tomatodoc2.md | 1 | Named only in the tomatodoc2 pest ranking list. Queries (crops tomato and general): target spot; corynespora; lesion; concentric. No symptom description. |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Ruling: no. tomatodoc1 only lists TYLCV as whitefly-transmitted; tomatodoc2 says whitefly-borne geminiviruses "severely stunt plant growth and affect fruit quality" and ranks "tomato curl viruses", but never describes leaf curling or yellowing. Queries (crops tomato and general): yellow leaf curl; TYLCV; geminivirus; begomovirus; curl; whitefly. A vector note is not a symptom description a grower could match to a leaf. |
| Tomato___Tomato_mosaic_virus | tomato | no | tomatodoc1.md; tomatodoc2.md | 1 | Ruling: no, borderline. tomatodoc2 lists Tobacco mosaic virus among the viruses it says it covers, but its virus section describes only TSWV and CMV in detail. The one sentence that most virus diseases cause stunting, leaf distortion, mosaic leaf discoloration and fruit spots is generic to all viruses; tomatodoc1 names TMV/ToMV only for transmission and as a cause of graywall. Queries (crops tomato and general): tomato mosaic; tobacco mosaic; TMV; ToMV; tobamovirus; mosaic. |
| Tomato___healthy | tomato | not applicable |  |  | A healthy class has no symptoms to describe. |

