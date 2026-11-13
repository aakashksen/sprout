# Manifest source details: gaps and changes

Scope: `knowledge/manifest.csv` (28 rows). Proposed copy: `<temp>`. `file`, `crop`, `diseases`, and `title` are unchanged. Nothing in the repo was edited.

## Counts

| Field | Filled from blank | Corrected | Still blank |
|---|---|---|---|
| source_url | 10 | 0 | 18 |
| publisher | 6 | 5 | 2 |
| date | 10 | 1 | 4 |

Totals: 10 source_url values filled, 18 still blank. 11 publisher values filled or corrected, 2 still blank. 11 date values filled or corrected, 4 still blank.

## Table 1: documents with a blank or doubtful field (one row per document)

| File | Field | What I tried | Why it stays blank or doubtful |
|---|---|---|---|
| Apple - G-sible_240513.pdf | source_url | Fetched the printed URL content.ces.ncsu.edu/integrated-orchard-management-guide-for-commercial-apples-in-the-southeast. Fetched NC State's 2024 announcement (apples.ces.ncsu.edu/news/2024-apple-manual-available). | The printed URL now shows the 2026 edition (publication date Feb. 13, 2026), not this 2024 document (AG-572, "05/24"). The 2024 post gives no PDF link. |
| Apple-Pest-Management-for-Home-Gardeners-1.pdf | source_url | Searched; fetched learningstore.extension.wisc.edu product page p424 (HTTP 429 on one attempt). Read the extension hort page. | The store code is p424, not A2179. The hort page gives a last revision of March 17, 2011, which does not match this copy's 2006 copyright. No URL tied to this copy. |
| Apple-Pest-Management-for-Home-Gardeners-1.pdf | date (filled 2006) | Read p.12. | "Copyright 2006" is a copyright year, not a stated publication date. Kept as the only date in the text. |
| Apple.pdf | source_url | Searched; found entomology.ipm.prod-web.ca.uky.edu/files/ipm9hmap.pdf. Fetch refused connection twice. Checked ipm.mgcafe.uky.edu/manuals (no IPM-9 link listed). | Found URL not opened. Not confirmed. |
| BacterialSpotPeach.pdf | source_url | Searched; found plantclinic.cornell.edu/factsheets/BacterialSpotPeach.docx. Fetch returned 404. | Not confirmed. |
| BacterialSpotPeach.pdf | date (filled 2024-10) | Read p.2 "Updated ... October 2024". | This is an update stamp, not an original publication date. |
| CitrusOrange-Greening-Guide.pdf | source_url | Searched. Only county-site copies found (liberty.agrilife.org, hidalgo.agrilife.org). | Not on the main AgriLife site, and not confirmed as this printing. |
| CitrusOrange-Greening-Guide.pdf | date (filled 2010-10) | Read p.1 "Citrus Tip Sheet E-264 10/10". | Inferred from a printed "10/10" code. Not an explicit date. |
| Grape-343478.pdf | source_url | Searched for The Crush, January 2021, on cawg.org. Found a UGA viticulture post linking to nxtbook.com/cawg/cawg/Jan2021/index.php. | The nxtbook host is a third-party magazine platform and was not opened. Not confirmed. |
| grapediseaseidguide.pdf | source_url | Found syngenta-us.com/crops/grapes/pdfs/GrapeDiseaseIDguide.pdf in search. WebFetch hit the 10 MB limit. | Not confirmed as this document. |
| grapediseaseidguide.pdf | date (kept 2019-04) | Read p.1 and p.50. | Only a print stamp "4/11/19" and "©2019 Syngenta". No stated publication month. Kept as doubtful. |
| Growing Blueberries.pdf | source_url | Searched; found tnstate.edu/4rconservation/documents/growing-blueberries.pdf. Fetch returned 404. | Not confirmed. |
| Growing Blueberries.pdf | date | Read p.8. | Only "Publication # TSU-12-0013(A)" appears, which is not a date. No date found online. |
| growing_blueberries_in_your_home_garden_osu_2.pdf | source_url | Fetched catalog.extension.oregonstate.edu/ec1304 (redirects to extension.oregonstate.edu/catalog/pub/ec1304). | The page is EC 1304 "Revised May 2025". This document is the Revised March 2008 archival copy (p.8). Different edition, so not used. |
| Learning-from-Peach-Bacterial-Spot-Epidemics.pdf | source_url | Searched; found plantpathology.ces.ncsu.edu/wp-content/uploads/2013/06/Learning-from-Peach-Bacterial-Spot-Epidemics.pdf. Fetch returned 404 (Retry-After 3600). | Not confirmed. |
| Learning-from-Peach-Bacterial-Spot-Epidemics.pdf | publisher (kept, doubtful) | Read p.1. | This is an abstract for the 2011 Great Lakes EXPO. NC State appears only as the author's affiliation. No publisher is named. |
| Orangeax739e.pdf | source_url | Searched; www.fao.org/3/ax739e/ax739e.pdf redirects to openknowledge.fao.org/3/ax739e/ax739e.pdf. | The openknowledge fetch returned 403. Not confirmed. |
| Organic-Blueberry-Growing-Guide.pdf | source_url | Fetched melissaknorris.com/blueberries/ search result page. Searched. | The free guide is behind an email signup. No public copy confirmed. |
| Organic-Blueberry-Growing-Guide.pdf | date | Read the full text (pp.2-13). | No date in the text. |
| PP362_PP362_ Bacterial Spot of Pepper.pdf | source_url | Tried edis.ifas.ufl.edu (403), ask.ifas.ufl.edu (redirect loop), doi.org/10.32473/edis-pp362-2021 (redirects to journals.flvc.org/edis/article/view/127041, 403). | Not confirmed. |
| PP362_PP362_ Bacterial Spot of Pepper.pdf | date (filled 2021) | Read p.7 DOI suffix "edis-pp362-2021". Also printed "Reviewed At: April 22, 2024". | The year comes from the DOI suffix, not a printed publication line. |
| strawberry.pdf | publisher | Read the text. Searched for the leaf scorch factsheet by Heidenreich and Turechek. | No publisher is named in the text. An NC State reference lists "Cornell University Extension" for this title, but this copy was not found. |
| strawberry.pdf | source_url | Searched. | Not found. |
| strawberry.pdf | date | Read the text. | No date in the text, and none found online. |
| strawberry2.pdf | source_url | Printed www.fruit.cornell.edu/berry is the site root. Searched for the 2013 article. | Article not located. Only an excerpt in a Cornell Eastern NY newsletter citing the March 2013 Berry News. |
| tomatodoc1.md | publisher | Read p.1. Searched for the 2023 handbook. | Text names Clemson and Mississippi State authors and "2023 Vegetable Crop Handbook for Southeastern United States", but no publisher. vegcrophandbook.com fetch failed (ECONNRESET). |
| tomatodoc1.md | source_url | Searched; found vegetables.tennessee.edu 2023 handbook PDF. Fetch returned 404. | Not confirmed as this tomato section. |
| tomatodoc2.md | source_url | Searched. Found NC State page vegetables.ces.ncsu.edu/pmsptomato/, which lists a 2007 GA/SC PDF. | Not reached. The 2007 PDF was not opened. |
| winegrapes8.pdf | date | Fetched content.ces.ncsu.edu chapter 8 page (no chapter date). Read the guide-level page. | Guide published Jan. 21, 2015 (AG-535), reviewed Aug. 11, 2025. These are guide-level dates, not chapter-level. |
| potato.pdf | title (not changed) | Compared the manifest title with the document and uidaho.edu/extension/publications/cis-1196. | Manifest says "Early Blight of Potato (CIS1182)". Document says CIS 1196, "Early Blight Biology and Control in Potatoes". Flagged only. |
| Peach2.pdf | title (not changed) | Compared the manifest title with the fetched PDF. | Manifest says "Stone fruit home orchard pest and disease management". Document is "Backyard Peach & Stone Fruit Disease, Pest and Cultural Practices Calendar" (PPFS-FR-T-22). Flagged only. |
| Orangeax739e.pdf | title (not changed) | Read p.1. | Manifest says "Issue Brief 4". Document prints "Key facts #4". Flagged only. |
| tomatodoc1.md | title (not changed) | Read p.1. | Document is an excerpt from the 2023 Southeast U.S. Vegetable Crop Handbook. Flagged only. |

## Table 2: values changed (one row per document)

| File | Changed field | New value | Evidence |
|---|---|---|---|
| 13-cherry.pdf | publisher (was "New England tree fruit extension"); source_url | UMass Extension Fruit Program (Center for Agriculture, Food, and the Environment, UMass Amherst); https://www.umass.edu/agriculture-food-environment/sites/default/files/pdf-doc-ppt/13-cherry.pdf | Publisher: https://www.umass.edu/agriculture-food-environment/fruit/2013-new-england-tree-fruit-management-guide. URL: fetched PDF, Chapter 13 header and "2013 New England Tree Fruit Management Guide" footer. |
| 2022-Strawberry-IPM-Guide.pdf | publisher (was "Southeast regional extension"); source_url | University of Georgia Extension (UGA Cooperative Extension); https://fieldreport.caes.uga.edu/wp-content/uploads/2025/08/AP-119-3_1.pdf | Publisher: p.77 "Published by the University of Georgia..."; cover (p.1) "University of Georgia Extension". URL: fetched PDF cover shows the 2022 guide title. |
| Apple - G-sible_240513.pdf | publisher (was four university names) | NC State Extension (North Carolina State University) | p.106 "Published by NC State Extension"; p.1 lists the co-contributing extension services. |
| Apple-Pest-Management-for-Home-Gardeners-1.pdf | date (blank) | 2006 | p.12 "Copyright 2006 by the Board of Regents of the University of Wisconsin". Copyright year only. |
| BacterialSpotPeach.pdf | date (blank) | 2024-10 | p.2 "Updated by SLJ2 & LG658, October 2024". |
| Cherry-em8329.pdf | source_url (blank) | https://extension.oregonstate.edu/sites/default/files/documents/em8329.pdf | Fetched PDF (redirects to the extension.oregonstate.edu sites/extd8 copy): cover "EM 8329, Revised April 2023", "Oregon State University Extension Service". |
| CitrusOrange-Greening-Guide.pdf | publisher (blank); date (blank) | Texas A&M AgriLife Extension Service; 2010-10 | Publisher: p.2 "Visit the Texas AgriLife Extension Service at AgriLifeextension.tamu.edu". Date: p.1 "Citrus Tip Sheet E-264 10/10" (inferred). |
| Corn1.pdf | publisher (was "National Corn Handbook multi-university"); source_url (blank); date (blank) | Iowa State University Extension (National Corn Handbook series); https://corn.aae.wisc.edu/Management/pdfs/NCH04.pdf; 1993-04 | Fetched PDF page 1: "IOWA STATE UNIVERSITY University Extension", "NCH 4 Revised April 1993", "Corn Disease Management". The URL is hosted on the National Corn Handbook site at the University of Wisconsin. |
| Grape-343478.pdf | publisher (blank) | California Association of Winegrape Growers (The Crush) | p.12 "published for members of the California Association of Winegrape Growers"; p.3 CAWG.ORG. |
| Growing Blueberries.pdf | publisher (blank) | Tennessee State University | p.8 "Tennessee State University", "Publication # TSU-12-0013(A)". |
| Orangeax739e.pdf | publisher (blank); date (blank) | Food and Agriculture Organization of the United Nations (FAO), Subregional Office for the Caribbean; 2013-10 | Publisher: p.4 contact block "Food and Agriculture Organization of the United Nations, Subregional Office for the Caribbean". Date: p.1 "Key facts #4 October 2013". |
| OrangeFarming.pdf | publisher (was "University of Florida; The Organic Center"); source_url (blank); date (blank) | The Organic Center; https://organic-center.org/sites/default/files/project/2019/01/HLB_FarmerGuide.pdf; 2019-01 | Publisher: cover logo; p.2 "The Organic Center ... January 2019". URL: fetched PDF cover matches the title and authors. University of Florida is only a co-author affiliation. |
| Peach2.pdf | publisher (blank); source_url (blank); date (blank) | University of Kentucky (Extension); https://publications.mgcafe.uky.edu/sites/publications.ca.uky.edu/files/ppfs-fr-t-22.pdf; 2019-07 | URL: fetched PDF page 1 is "Backyard Peach & Stone Fruit Disease, Pest and Cultural Practices Calendar, PPFS-FR-T-22", UK logo, Extension authors. Matches this file's ID-21 and HO-6 references. Date: p.2 "Reviewed July 2019". |
| Potato - early-blight-bulletin.pdf | source_url (blank) | https://sanweb.lib.msu.edu/DMC/extension_publications/e2991/e2991.pdf | Fetched PDF page 1: "Extension Bulletin E-2991 May 2007", Wharton and Kirk, Michigan State University. Hosted on the MSU Libraries domain. |
| potato.pdf | publisher (blank); source_url (blank); date (blank) | University of Idaho Extension; https://www.uidaho.edu/extension/publications/cis-1196; 2013-07 | Publisher: p.5 "© 2013 by the University of Idaho". Date: p.5 "Published July 2013". URL: uidaho page shows CIS 1196 with Wharton and Wood. Title differs (see Table 1). |
| PP362_PP362_ Bacterial Spot of Pepper.pdf | date (blank) | 2021 | p.7 DOI "10.32473/edis-pp362-2021"; p.7 "Reviewed At: April 22, 2024". |
| S-0003-45-Corn.pdf | source_url (blank) | https://extension.sdstate.edu/sites/default/files/2019-09/S-0003-45-Corn.pdf | Fetched PDF page 1: iGrow Corn, Chapter 45 "Scouting for Corn Diseases", "© 2019, South Dakota Board of Regents". |
| tomatodoc1.md | date (blank) | 2023 | p.1 running head "2023 Vegetable Crop Handbook for Southeastern United States". |
| tomatodoc2.md | date (was 2007-01) | 2007-08 | p.1 "August 22, 2007" (document date). The January 5-6, 2007 workshop is the meeting, not the document date. |
| winegrapes8.pdf | source_url (blank) | http://content.ces.ncsu.edu/chapter-8-pest-management.pdf | Fetch returned an HTML page titled "Chapter 8. Pest Management" in the NC State Extension "North Carolina Winegrape Grower's Guide" collection. Not a PDF. |

## Notes

- Publisher and date values kept as they were: 2019-04 for grapediseaseidguide.pdf (print stamp only), and "Learning from Peach" publisher (see Table 1).
- Sources cited: the fetched pages above. For each source_url, I either fetched the document or read its page 1 (the PDF was saved and read). Where a fetch was redirected, the final URL is noted in the Table 2 text.

## Dates left blank because they were only inferred

- `CitrusOrange-Greening-Guide.pdf`: inferred only from the printed code E-264 10/10. The proposed value was not applied.
- `PP362_PP362_ Bacterial Spot of Pepper.pdf`: inferred only from the DOI suffix edis-pp362-2021. The proposed value was not applied.

## Values to check

- `Corn1.pdf`: the URL is on the National Corn Handbook site at the University of Wisconsin, while the document names Iowa State University Extension.
- `Potato - early-blight-bulletin.pdf`: the URL is on the MSU Libraries host.
- `winegrapes8.pdf`: the URL ends in `.pdf` but returns an HTML chapter page.
- `potato.pdf`: the manifest title says CIS1182, while the document and URL say CIS 1196. Titles were not changed.
