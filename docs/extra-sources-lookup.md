# T4.5 extra real-photo sources: lookup

Checked 2026-11-03 by lookups only. Nothing was downloaded, installed, or committed. Each fact cites the page it was read from. "Couldn't confirm" means the fact was not found in a fetched primary source.

Rule applied (from the person): use data only if it is publicly reachable without login, paywall, or getting around a block. A public mirror from the authors qualifies. A login-gated competition does not.

## Summary

| source | reachable without login | size | mapped lab classes |
|---|---|---|---|
| PlantWild v1 (`plantwild.zip`) | Yes, Hugging Face by the authors' listed repo, no gating. Also a password-gated UQRDM link (password printed on the authors' page; not used) and a Google Drive link (not tested beyond HEAD). | 2,708,493,416 bytes | Couldn't confirm the full class list. Paper text names potato early and late blight; mapping to `Potato___Early_blight` and `Potato___Late_blight` is probable but not confirmed. |
| PlantWild v2 (`plantwild_v2.zip`) | Yes, same HF repo, no gating | 1,640,342,849 bytes | Couldn't confirm; 115 disease-only classes, class list not read. |
| Plant Pathology 2021 (FGVC8) | No. Kaggle competition; anonymous requests return HTTP 404 and the rules page shows only its title. No authors' mirror found. | Not measured (Kaggle not reachable anonymously). Third-party mirror below is 16,094,440,857 bytes. | Apple only: `Apple___Apple_scab`, `Apple___Cedar_apple_rust`, `Apple___healthy` (by class name). Not usable under the rule. |

Recommendation for the person: PlantWild v1 and v2 are candidates. Two open points before use: the license is CC BY-NC-ND 4.0 (see below), and the class list and labels still need to be read from the zip. Plant Pathology 2021 does not qualify under the rule.

## 1. PlantWild

**Access, tested without login**

- Project page: `curl -sIL -m 30 https://tqwei05.github.io/PlantWild` returned HTTP 301 to `https://tqwei05.github.io/PlantWild/`, then HTTP 200, 39,179 bytes. Source: https://tqwei05.github.io/PlantWild
- Access page: `https://tqwei05.github.io/PlantWild/access`, HTTP 200, 33,077 bytes. It lists three download routes. Source: https://tqwei05.github.io/PlantWild/access (read via WebFetch).
  - Google Drive: https://drive.google.com/file/d/1s7FOoztTHvO03yVfw75pQY_kzZqvAckD/view?usp=drive_link (v1). The HEAD request returned HTTP 200 with an HTML page and no content length, so the file size could not be confirmed. Not downloaded.
  - UQRDM: https://cloud.rdm.uq.edu.au/index.php/s/5iTzoyby9Xobmq2 (v1). Password-protected; the password is printed on the access page. Anonymous `curl -sIL` returned HTTP 404 (16,085 bytes). Not used.
  - Hugging Face: the access page did not list this route. The HF repo is linked from the project page (https://tqwei05.github.io/PlantWild, which lists `https://huggingface.co/datasets/uqtwei2/PlantWild` under v1 and v2).
- HF repo `uqtwei2/PlantWild`: `curl -sIL -m 30 https://huggingface.co/datasets/uqtwei2/PlantWild` returned HTTP 200, 360,888 bytes. The HF API (https://huggingface.co/api/datasets/uqtwei2/PlantWild) reports `"private":false,"gated":false`, and siblings `plantwild.zip` and `plantwild_v2.zip`.
  - `curl -sIL -m 30 https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip` returned HTTP 302 to a CDN URL, then HTTP 200 with `content-length: 2708493416`. Final URL is a signed `us.aws.cdn.hf.co` link (not reproduced; it expires).
  - `curl -sIL -m 30 https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild_v2.zip` returned HTTP 302, then HTTP 200 with `content-length: 1640342849`.
- Login: the HF repo is public and not gated per the API. The authors' page does not state a login requirement for HF. The UQRDM route needs a password printed on the page.

**Size**

- `plantwild.zip`: 2,708,493,416 bytes (HF API tree, https://huggingface.co/api/datasets/uqtwei2/PlantWild/tree/main?recursive=true; HTTP content-length confirmed as above).
- `plantwild_v2.zip`: 1,640,342,849 bytes (same API tree).
- Total of both zips: 4,348,836,265 bytes (sum). The HF card states "Total file size: 4.35 GB" and notes "the information about the size of the dataset is not coherent" (https://huggingface.co/datasets/uqtwei2/PlantWild, read via WebFetch).

**Image count and classes**

- v1: 18,542 images, 89 classes, 56 diseased and 33 healthy. Source: arXiv HTML, https://arxiv.org/html/2408.03120v1 (read via WebFetch). The same paper gives the per-class range: largest 589, smallest 44. Per-class counts are in its Figure 3, which was not readable as text. Splits 70/10/20 (train/val/test), per the same paper.
- v2: 115 disease-only classes (authors' access page and HF card, both via WebFetch). Image count 11,488 comes from the Voxel51 mirror card only (https://huggingface.co/datasets/Voxel51/PlantWild), which says it is a repackaging. Treat 11,488 as secondary.
- Conflicts: the HF `uqtwei2` card shows "train · 30k rows" and "Number of rows: 30,035", which does not match 18,542 + 11,488 = 30,030. Not reconciled.
- Class list: couldn't confirm. Only potato early blight and potato late blight are named in the paper text, plus apple scab and bean rust as examples in its figures (arXiv HTML). The full list was not read from the zip or a machine-readable file.

**Labels and file layout**

- Couldn't confirm. The access page and the HF card do not state the file layout or how labels are given. The Voxel51 mirror says v1 has `split` and `ground_truth` fields and a `source_url` per image, but it is a conversion, not the authors' description (https://huggingface.co/datasets/Voxel51/PlantWild).
- Two annotation steps are stated in the paper: labels "checked by at least two annotators and verified by an expert" (arXiv HTML).

**License and terms**

- Authors' access page: "Our dataset follows the copyright Creative Commons BY-NC-ND 4.0 license," linking to the CC BY-NC-ND 4.0 deed (https://tqwei05.github.io/PlantWild/access). No other terms stated.
- HF repo metadata: `license:cc-by-nc-nd-4.0` (https://huggingface.co/api/datasets/uqtwei2/PlantWild).
- The paper's own statement: the dataset images were collected from Google Images, Ecosia, and Baidu Images (arXiv HTML). Per-image rights were not checked.
- Point for the person: NC is non-commercial, and ND is "no derivatives." Whether training a model counts as a derivative is a legal question I did not resolve. This is flagged, not decided.

## 2. Plant Pathology 2021 (FGVC8)

**Access, tested without login**

- Kaggle competition page, `curl -sIL -m 30 https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8`: HTTP 404. The `/data` and `/rules` sub-pages also returned HTTP 404, with and without a browser user-agent.
- WebFetch of `https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/rules` returned only the page title. Data files, license text, and sizes were not visible without login.
- Login: the competition data is behind Kaggle. I could not confirm the exact login requirement from fetched content, but an anonymous request does not reach the data. Under the person's rule, a login-gated Kaggle competition does not qualify.
- Authors' mirror: none found. The Cornell PDF "The Plant Pathology 2021 Challenge dataset to classify foliar disease of apples" (https://vision.cornell.edu/se3/wp-content/uploads/2021/09/029.pdf), which the search index shows, failed to load: a TLS hostname mismatch for `vision.cornell.edu` in WebFetch, and curl returned no headers. Couldn't confirm its content, so not used for any fact here.

**Third-party mirror (does not qualify, recorded for reference)**

- `https://huggingface.co/datasets/timm/plant-pathology-2021`: HF account `timm`, not the authors. HTTP 200. It re-hosts the Kaggle data as parquet.
- Size: 36 files, 16,094,440,857 bytes total (HF API tree, https://huggingface.co/api/datasets/timm/plant-pathology-2021/tree/main?recursive=true).
- Rows: 18,632, with 16.8k train and 1.86k validation. No test split (the card says "I do not have access to the test samples"). Source: https://huggingface.co/datasets/timm/plant-pathology-2021 (WebFetch).
- Classes (6): complex, frog_eye_leaf_spot, healthy, powdery_mildew, rust, scab (card `dataset_info`).
- Mapping if it were usable: `healthy` to `Apple___healthy`, `rust` to `Apple___Cedar_apple_rust`, `scab` to `Apple___Apple_scab`. `complex`, `frog_eye_leaf_spot`, and `powdery_mildew` have no lab class for apple.
- License on the mirror card: `apache-2.0`. This is the uploader's statement, not the authors' or Kaggle's. The card does not cite the competition's terms. Couldn't confirm the competition's own license.

## 3. Additional sources

None added. The plan allows up to two, only if clearly useful for thin classes. The seven classes with zero real photos (Corn healthy, Grape Esca, Grape Leaf blight, Peach Bacterial spot, Potato healthy, Tomato Spider mites, Tomato Target Spot, per `docs/class-report.md`) are the targets. I did not survey a second source in this pass, so I have no verified candidate to propose.

## Open items for the person

1. PlantWild license is CC BY-NC-ND 4.0. Confirm whether NoDerivatives blocks training a model on the images.
2. PlantWild class list and labels are unread. Read `plantwild.zip` (or its README) before any mapping, and only after the person approves the download.
3. The Kaggle route for Plant Pathology 2021 needs the person to decide only if they want an exception. Under the stated rule, it is excluded.
