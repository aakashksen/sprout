# Document sources

Where the documents added in T6.5 came from and how they were chosen. Raw files are in the git-ignored `knowledge/raw/` and are never committed or published; `knowledge/manifest.csv` holds each document's URL, publisher, title and date, and `docs/document-urls.csv` is the researcher's original candidate list (163 rows).

## How they were fetched

- One request at a time, 2 seconds apart, with the client name `sprout-research-fetch/1.0 (local non-commercial research)` and no personal contact. Each host's `robots.txt` was read first and respected: two `vtechworks.lib.vt.edu` URLs were skipped because it disallows them.
- Only public pages and PDFs were saved. Nothing was opened or run. Five pages were an Incapsula bot-protection block page instead of the document; they were deleted and not retried or worked around.
- Titles, publishers, crops and diseases in the manifest come from each saved document's own text and host, not from the search snippets in the candidate list. The crop is the crop the text mentions most, or `general` when it covers several without one clearly ahead; mixed documents (such as tomato and pepper bacterial spot) carry only their leading crop. A disease is listed only if its name appears at least twice in the text (or once, for two documents where the name is in the title).
- Where a PDF has no title of its own, the title was written from its first heading or opening line (files 010, 011, 013, 025, 036, 039, 046, 051, 057, 061, 064, 066, 067, 075, 089, 102, 106, 112, 119, 121, 122, 126, 131, 138, 139, 140, 147). Dates are only those the document states; the rest are blank.
- The two Penn State candidate pages listed for cherry powdery mildew and the pages listed for apple scab and apple rust on `extension.psu.edu` redirected to general pests-and-diseases index pages. Files 001 and 002 are those index pages and their manifest URL is the page actually saved; 012 and 017 were identical copies and were dropped.
- File names start with the row number in the candidate list, then the URL.

## Counts

Added: 134 documents (163 listed, 2 skipped by robots.txt, 27 dropped after download, listed below). Total in the manifest: 162. Below the 200 to 400 the plan asks for.

| crop | new | before | total |
|---|---|---|---|
| apple | 14 | 3 | 17 |
| blueberry | 3 | 3 | 6 |
| cherry | 1 | 2 | 3 |
| corn | 19 | 2 | 21 |
| general | 7 | 1 | 8 |
| grape | 11 | 3 | 14 |
| orange | 3 | 3 | 6 |
| peach | 4 | 3 | 7 |
| pepper | 9 | 1 | 10 |
| potato | 7 | 2 | 9 |
| raspberry | 6 | 0 | 6 |
| soybean | 3 | 0 | 3 |
| squash | 7 | 0 | 7 |
| strawberry | 6 | 3 | 9 |
| tomato | 34 | 2 | 36 |

Crop-specific documents whose manifest disease list names each class (not the same as describing its symptoms; see `docs/document-coverage.md` for that):

| class | new documents | all documents |
|---|---|---|
| Apple___Apple_scab | 8 | 11 |
| Apple___Black_rot | 4 | 6 |
| Apple___Cedar_apple_rust | 4 | 7 |
| Cherry_(including_sour)___Powdery_mildew | 0 | 1 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 7 | 8 |
| Corn___Common_rust_ | 8 | 10 |
| Corn___Northern_Leaf_Blight | 6 | 8 |
| Grape___Black_rot | 5 | 7 |
| Grape___Esca_(Black_Measles) | 3 | 5 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1 | 1 |
| Orange___Haunglongbing_(Citrus_greening) | 2 | 5 |
| Peach___Bacterial_spot | 4 | 6 |
| Pepper,_bell___Bacterial_spot | 5 | 6 |
| Potato___Early_blight | 3 | 5 |
| Potato___Late_blight | 4 | 4 |
| Squash___Powdery_mildew | 7 | 7 |
| Strawberry___Leaf_scorch | 5 | 7 |
| Tomato___Bacterial_spot | 6 | 7 |
| Tomato___Early_blight | 9 | 9 |
| Tomato___Late_blight | 6 | 6 |
| Tomato___Leaf_Mold | 6 | 6 |
| Tomato___Septoria_leaf_spot | 8 | 8 |
| Tomato___Spider_mites Two-spotted_spider_mite | 3 | 3 |
| Tomato___Target_Spot | 4 | 4 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 2 | 2 |
| Tomato___Tomato_mosaic_virus | 2 | 2 |

Cherry powdery mildew has no cherry document that describes it. The `general` documents that mention powdery mildew do so for other crops or in a list.

## Reuse restrictions

The person's later rule allows public documents for local, unpublished use without one-by-one approval, so every document above is in use. Twelve of them were also in `docs/document-candidates.md`, where each is marked `Approved: no, needs the person`. Each has stated restrictions on reuse, or no stated permission, which matters if the documents, their text or long quotations are ever published. Do not publish them without checking with the person.

- `141-homegarden-cahnr-uconn-edu-factsheets-powdery-mildew-of-cucurbits.html` (UConn Home and Garden Education Center, University of Connecticut), https://homegarden.cahnr.uconn.edu/factsheets/powdery-mildew-of-cucurbits/: The page states "(c) University of Connecticut" and "The information in this document is for educational purposes only." The UConn site statement (http://uconn.edu/disclaimers-privacy-copyright/) states: "Permission is required to copy, download, or use any text, photographs, or image files."
- `142-content-ces-ncsu-edu-cucurbit-powdery-mildew.html` (NC State Extension), https://content.ces.ncsu.edu/cucurbit-powdery-mildew: couldn't confirm. The page has no copyright or license statement, only a disclaimer that chemical recommendations are "a convenience to the reader" and that the publication "does not imply endorsement by NC State University or N.C. A&T State University."
- `078-umass-edu-agriculture-food-environment-home-lawn-garden-fact-sheets-tomato-problems-early-.html` (UMass Extension, University of Massachusetts Amherst), https://www.umass.edu/agriculture-food-environment/home-lawn-garden/fact-sheets/tomato-problems-early-blight: The page footer shows "(c)2025 University of Massachusetts Amherst." The UMass site policy (https://www.umass.edu/gateway/site-policies) states that UMass-owned content "may not be used or reproduced without permission" (quoted in a fetch summary).
- `079-extension-umd-edu-resource-early-blight-vegetables.html` (University of Maryland Extension), https://extension.umd.edu/resource/early-blight-vegetables: couldn't confirm. The page shows "(c) 2018-2026 College of Agriculture and Natural Resources" and nondiscrimination, accessibility, and privacy links, but no reuse terms were readable.
- `085-ipm-cahnr-uconn-edu-tomato-late-blight.html` (UConn Integrated Pest Management Program, University of Connecticut), https://ipm.cahnr.uconn.edu/tomato-late-blight/: The page states "(c) University of Connecticut" and "The information in this document is for educational purposes only." The UConn site statement (http://uconn.edu/disclaimers-privacy-copyright/) states: "Permission is required to copy, download, or use any text, photographs, or image files."
- `086-extension-usu-edu-vegetableguide-tomato-pepper-eggplant-late-blight.html` (Utah State University Extension), https://extension.usu.edu/vegetableguide/tomato-pepper-eggplant/late-blight: The page has only footer labels "Terms of Use," "Copyright," and "Privacy." The Utah State copyright page (https://www.usu.edu/copyright/) states: "This website and its content are copyright of Utah State University." and "Any redistribution or reproduction of part or all of the contents in any form is prohibited."
- `080-umass-edu-agriculture-food-environment-vegetable-fact-sheets-tomato-leaf-mold.html` (UMass Extension, University of Massachusetts Amherst), https://www.umass.edu/agriculture-food-environment/vegetable/fact-sheets/tomato-leaf-mold: The page footer shows "(c)2025 University of Massachusetts Amherst." The UMass site policy (https://www.umass.edu/gateway/site-policies) states that UMass-owned content "may not be used or reproduced without permission" (quoted in a fetch summary).
- `091-vegetables-cornell-edu-crops-tomatoes-tomato-leaf-mold.html` (Cornell Vegetables, Cornell University), https://vegetables.cornell.edu/crops/tomatoes/tomato-leaf-mold: couldn't confirm. The footer shows only "Cornell University(c)2026," with links to "University Privacy" and "Web Accessibility Assistance." No reuse terms were readable.
- `095-extension-illinois-edu-plant-problems-septoria-leaf-spot-tomato.html` (University of Illinois Extension), https://extension.illinois.edu/plant-problems/septoria-leaf-spot-tomato: couldn't confirm. The page shows no copyright or license text, only footer links "EEO," "myExtension," and "Staff Login."
- `098-yardandgarden-extension-iastate-edu-encyclopedia-septoria-leaf-spot.html` (Iowa State University Extension and Outreach), https://yardandgarden.extension.iastate.edu/encyclopedia/septoria-leaf-spot: The page states: "Links to this article are strongly encouraged, and this article may be republished without further permission" "if published as written and if credit is given to the author, Yard and Garden," "and Iowa State University Extension and Outreach." It also states: "If this article is to be used in any other manner, permission from the author is required." Footer: "(c) Iowa State University of Science and Technology."
- `105-content-ces-ncsu-edu-tomato-yellow-leaf-curl-virus.html` (NC State Extension), https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus: couldn't confirm. The page has no copyright, license, or terms-of-use text; the only related text is a nondiscrimination statement and links to a Privacy Statement.
- `107-ipm-cahnr-uconn-edu-mosaic-diseases-of-tomatoes.html` (UConn Integrated Pest Management Program, University of Connecticut), https://ipm.cahnr.uconn.edu/mosaic-diseases-of-tomatoes: The page states "(c) University of Connecticut" and "The information in this document is for educational purposes only." The UConn site statement (http://uconn.edu/disclaimers-privacy-copyright/) states: "Permission is required to copy, download, or use any text, photographs, or image files."

The other documents were not checked for licence text. Treat them the same way.

## Dropped after download

| file | URL | reason |
|---|---|---|
| 003 | https://www.canr.msu.edu/uploads/files/Research_Center/NW_Mich_Hort/Pests_IPM_Mgmt/PreHarvDiseaseControl2005.pdf | blocked by the host's bot protection (Incapsula page); not worked around |
| 004 | https://extension.umd.edu/resources/yard-garden/fruit/cherries | index page for stone fruits, under 600 characters |
| 006 | https://www.canr.msu.edu/cherries/horticulture/pruning-and-training | blocked by the host's bot protection (Incapsula page); not worked around |
| 007 | https://boone.ca.uky.edu/sites/boone.ca.uky.edu/files/midwest_home_fruit_production_guide_0.pdf | county office landing page, no plant-care content |
| 012 | https://extension.psu.edu/apple-scab-in-the-home-fruit-garden | same page as 001 |
| 017 | https://extension.psu.edu/rust-diseases-of-apple-in-the-home-fruit-planting | same page as 001 |
| 038 | https://extensionpubs.unl.edu/publication/g2270/northern-corn-leaf-blight-g2270 | landing page for a PDF, abstract only |
| 048 | https://extension.umaine.edu/publications/2427e/embed/ | landing page, abstract only |
| 053 | https://www.uidaho.edu/extension/publications/publication-detail?id=cis1000 | extension home page, no plant-care content |
| 054 | https://www.canr.msu.edu/uploads/files/growingPotatoes.pdf | blocked by the host's bot protection (Incapsula page); not worked around |
| 055 | https://pubs.ext.vt.edu/426/426-312/426-312.html | general vegetable-garden planning, nothing on a target class |
| 068 | https://extension.umd.edu/resources/yard-garden/fruit/peaches | same page as 004 |
| 069 | https://peaches.ces.ncsu.edu/peaches-educational-resources-for-home-gardeners/ | list of links only |
| 076 | https://yardandgarden.extension.iastate.edu/article/2013/04-12/strawberries.html | announcement of a pamphlet, no care content |
| 093 | https://extensionpubs.unl.edu/publication/ec1864/leaf-and-fruit-diseases-of-tomatoes | landing page for a PDF, abstract only |
| 103 | https://ipm.ifas.ufl.edu/pdfs/TYLCV.pdf | 'coming soon' placeholder page |
| 104 | https://ipm.ifas.ufl.edu/resources/success_stories/t&pguide/pdfs/Chapter5/TYLCV.pdf | 'coming soon' placeholder page |
| 120 | https://nwdistrict.ifas.ufl.edu/phag/?p=23616 | regional news listing, no plant-care content |
| 148 | https://extensionpubs.unl.edu/publication/9000016363769/tomatoes-in-the-home-garden | landing page for a PDF, abstract only |
| 149 | https://solutions.tennessee.edu/publication/SP291K | list of other publications, no care content |
| 150 | https://openprairie.sdstate.edu/extension_fact/94 | repository landing page, document not on the page |
| 155 | https://www.canr.msu.edu/ipm/uploads/files/StoneFruitIPM_BacterialSpot8.pdf | blocked by the host's bot protection (Incapsula page); not worked around |
| 156 | https://www.canr.msu.edu/ipm/agriculture/fruit/bacterial_spot_of_peach_and_plum | blocked by the host's bot protection (Incapsula page); not worked around |
| 158 | https://extension.umd.edu/resources/yard-garden/fruit/grapes | landing page, abstract only |
| 161 | https://njaes.rutgers.edu/pubs/publication.php?pid=FS112 | landing page for a PDF, abstract only |
| 162 | https://uidaho.edu/extension/publications/publication-detail?id=pnw0400 | extension home page, no plant-care content |
| 163 | https://publications.mgcafe.uky.edu/ppfs-fr-t-18 | landing page for a PDF, abstract only |

## Not done

- Several dropped pages (038, 048, 093, 148, 158, 161, 163) are landing pages that link to a PDF. The PDF links were not followed, since they are not in the candidate list.
- No cherry powdery mildew document was saved: the two Penn State pages redirected to index pages, and the Michigan State cherry pages were blocked. Another search is needed for that class.
