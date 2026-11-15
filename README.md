<div align="center">

<img src="site/favicon.svg" width="72" alt="">

# SPROUT

**Smart Plant Recovery and Optimization Utility Tool**

Photograph a sick leaf. SPROUT names the disease, asks the few questions that settle it, and tells you what to do, citing the university extension guide the advice came from.

**Follow-up questions raise field accuracy from 81.3% to 95.2%** on 1,006 held-out field photos ([how it works](#question-engine)).

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-ConvNeXt_V2-EE4C2C?logo=pytorch&logoColor=white) ![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED?logo=onnx&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white) ![Tests](https://img.shields.io/badge/tests-901-2ea44f) ![License](https://img.shields.io/badge/license-MIT-blue)

### [See it running: sprout.aakashsenthilnathan.com](https://sprout.aakashsenthilnathan.com)

[Walkthrough](https://sprout.aakashsenthilnathan.com/tour.html) · [How it's built](https://sprout.aakashsenthilnathan.com/specs.html) · [Live demo](https://sprout.aakashsenthilnathan.com/live.html)

<img src="site/img/live-demo.gif" width="820" alt="A recorded run: a tomato leaf read as potato late blight, one question about the plant, then tomato late blight at 97% with cited care advice and a follow-up chat">

<sub>A real run on a tomato leaf the model never trained on.</sub>

</div>

---

## Contents

[What it does](#what-it-does) · [The stack](#the-stack) · [System overview](#system-overview) · [Life of a check](#life-of-a-check) · [Question engine](#question-engine) · [Guide search](#guide-search) · [Vision model](#vision-model) · [Live demo server](#live-demo-server) · [Website](#website) · [Data and experiments](#data-and-experiments) · [Quick start](#quick-start) · [Repository map](#repository-map)

## What it does

A photo alone often can't tell look-alike leaf diseases apart. SPROUT combines three things to get to one answer:

1. **A vision model**: ConvNeXt V2 Tiny, trained on lab and field photos together, giving calibrated odds over 29 conditions on 14 plants.
2. **A question engine** with 54 fixed questions aimed at the diseases the photo most often confuses, including things a photo can't show (the leaf underside, stem, fruit, how it spread, weather). Code picks the one that narrows the odds most and stops as soon as it's sure. On held-out field photos this lifts accuracy from 81.3% to 95.2% with 10% of the answers wrong.
3. **A search engine across 250+ university extension guides.** The questions are written from guide passages, the care advice and the chat replies are written from them too, and each one names its source.

The design rule behind all of it: **an LLM writes the words, plain code does the counting.** Every probability, the choice of the next question and the decision to stop are code, so the same answers always give the same result.

## The stack

Every layer, from the training GPU to the page in the browser.

```mermaid
flowchart LR
    L1["<b>Data</b><br/>DVC, MLflow,<br/>imagehash,<br/>DINOv2"]
    L2["<b>Training</b><br/>PyTorch, timm,<br/>CUDA AMP,<br/>RTX 4070"]
    L3["<b>Serving model</b><br/>ONNX Runtime,<br/>temperature<br/>scaling"]
    L4["<b>Knowledge</b><br/>pypdf, BM25,<br/>fastembed,<br/>MiniLM"]
    L5["<b>Server</b><br/>Flask, gunicorn,<br/>thread pool,<br/>LLM API"]
    L6["<b>Front end</b><br/>HTML, CSS, JS,<br/>Cloudflare<br/>Workers"]
    L1 --> L2 --> L3 --> L5
    L4 --> L5
    L5 --> L6
```

| Layer | Technology | Role |
|---|---|---|
| Vision model | PyTorch, timm, ConvNeXt V2 Tiny (ImageNet-22k pretrained) | Reads the leaf and gives odds over 29 conditions |
| Training | CUDA mixed precision (AMP, GradScaler), AdamW, cosine schedule, RTX 4070 | Fine-tunes on lab and field photos |
| Data hygiene | `imagehash` perceptual hashing, DINOv2 (transformers), group-aware splits | Keeps near-duplicate photos on one side of a split |
| Calibration | Temperature scaling, abstention bar, photo gate | Makes confidence mean something and rejects non-leaves |
| Model export | ONNX, ONNX Runtime (CPU), Pillow and NumPy preprocessing | PyTorch-free serving image |
| Search | pypdf, BM25, BGE-small embeddings via fastembed, MiniLM cross-encoder reranker | Finds the right guide passage for a plant and a disease |
| Question engine | Plain Python: a fixed bank of 54 guide-based questions, fitted answer strengths, expected information gain, confidence bar | Chooses questions and the diagnosis; no LLM involved |
| Server | Flask, gunicorn, `ThreadPoolExecutor`, signed session cookies | JSON API, background writing, spend caps, rate limits |
| Text | An LLM API for the live demo; a local LLM through Ollama for the full app | Words only, never numbers |
| Front end | Hand-written HTML, CSS and JavaScript, inline SVG, no framework | Walkthrough, architecture page, live demo |
| Hosting | Cloudflare Workers static assets (site), Docker image with gunicorn (full app) | sprout.aakashsenthilnathan.com |
| Tracking and quality | MLflow, DVC, JSON run records, pytest, ruff, GitHub Actions | Reproducible runs and checked code |

## System overview

Four parts: a static website, one Python server, background threads inside that server, and an external LLM API that only writes text. The model and the guide index are built offline and loaded when the server starts.

```mermaid
flowchart LR
    B["Browser"] <-->|"JSON"| S["Flask server"]
    S --> W["Background<br/>threads"]
    W <-->|"passages in,<br/>text out"| L["LLM API"]
    T["Offline training<br/>and indexing"] -.->|"ONNX model,<br/>guide index"| S
```

The browser holds nothing but the id of the current check; the server owns the odds. The same core (identifier, retriever, guide manifest) also backs a second app, `src/sprout/web`: a full Flask app with accounts, upload history and a local LLM through Ollama.

[**Full system overview**](docs/architecture/system.md)

## Life of a check

One photo, a few taps, a diagnosis, then a conversation. The slow work (writing the advice) happens in the background while you read, and the questions come from a fixed bank, so each tap only runs the odds engine.

```mermaid
sequenceDiagram
    actor G as Gardener
    participant B as Browser
    participant S as Server
    participant W as Background
    G->>B: photo
    B->>S: POST /api/check
    S->>W: start writing the advice
    S-->>B: odds + first question
    G->>B: taps an answer
    B->>S: POST /api/answer
    S-->>B: new odds + next question (milliseconds)
    S-->>B: diagnosis with cited advice
    G->>B: follow-up question
    B->>S: POST /api/chat
    S-->>B: reply with sources, odds may move
```

<table>
<tr>
<td width="50%"><img src="site/img/step-odds.jpg" alt="Odds after the photo and the plant question"></td>
<td width="50%"><img src="site/img/step-question.jpg" alt="A yes or no question with its guide source"></td>
</tr>
<tr>
<td><img src="site/img/step-final.jpg" alt="The diagnosis with cited care advice"></td>
<td><img src="site/img/step-chat.jpg" alt="The follow-up chat with numbered sources"></td>
</tr>
</table>

[**Every request, step by step**](docs/architecture/check-lifecycle.md)

## Question engine

Plain Python that turns uncertain odds into one answer. It asks which plant first when the suspects belong to different plants, then picks, from a fixed bank of 54 questions written from the guides, the one expected to leave the least uncertainty (expected information gain). Each answer updates the odds with strengths fitted on a development set, and it stops at 95% confidence (98.5% for healthy), after 8 questions, or when no question helps. An LLM never touches a number. Questions are asked only about the diseases the photo is still unsure between, and 28 of the 54 are about things a photo can't show.

**Results** on 1,006 held-out field photos (everything was fitted on a separate development set): photo only 81.3%, plus the plant question 84.0%, plus the follow-up questions 95.2%. A photo that gets questions gets about 2.8, and when it says it is 93% or more sure it is right about 97% of the time.

```mermaid
flowchart LR
    O["Odds"] --> S{"Sure?"}
    S -->|yes| D["Diagnosis"]
    S -->|no| P["Pick the most<br/>informative question"]
    P --> A["Answer"]
    A --> U["Update odds with<br/>fitted answer strengths"]
    U --> S
```

[**The question bank, odds updates, information gain, the stop rule and the E18 results**](docs/architecture/question-engine.md)

## Guide search

Everything SPROUT says starts from a passage in 250+ extension guides. Guides are extracted with pypdf, chunked, and tagged by crop and disease. A search filters to the plant and disease in question, merges BM25 keyword scores with BGE-small embedding similarity (fastembed), and lets a MiniLM cross-encoder reranker pick the best passages. The same retriever grounds questions, advice and chat.

```mermaid
flowchart LR
    G["250+ guides"] --> C["Chunk + tag<br/>by crop and disease"]
    C --> I["BM25 index +<br/>BGE-small vectors"]
    I --> F["Filtered<br/>hybrid search"]
    F --> R["MiniLM<br/>reranker"]
    R --> P["Passages with<br/>their source"]
```

[**How the index is built and searched**](docs/architecture/guide-search.md)

## Vision model

ConvNeXt V2 Tiny fine-tuned with PyTorch and timm on a CUDA GPU with mixed precision, on lab and field photos together so it holds up on real garden photos. Splits are grouped by near-duplicate photos (perceptual hashes, plus DINOv2 similarity for extra sources) so look-alikes stay on one side; 13 exact copies still cross the train and test line, and removing the closest look-alikes from the dev set lowers its accuracy by under a point. Temperature scaling calibrates the confidence, the model is exported to ONNX and served on CPU with ONNX Runtime, and a photo gate turns away images that aren't leaves.

```mermaid
flowchart LR
    D["Lab + field photos"] --> S["Near-duplicate-safe<br/>splits"]
    S --> T["Fine-tune on GPU"]
    T --> C["Calibrate"]
    C --> X["Export to ONNX"]
    X --> R["Serve on CPU"]
```

Reading one photo at serving time:

```mermaid
flowchart LR
    P["Photo bytes"] --> V["Validate, EXIF upright,<br/>224 px resize"]
    V --> M["ONNX Runtime<br/>ConvNeXt V2"]
    M --> K["Temperature-scaled<br/>probabilities"]
    K --> G{"Photo gate"}
    G -->|pass| O["Odds for<br/>29 conditions"]
    G -->|fail| N["Not a leaf"]
```

[**From photos to a served model, and how one photo is read**](docs/architecture/vision-model.md)

## Live demo server

A single Flask process (`tools/live_demo.py`) that serves the website and its JSON API. A request thread answers immediately with odds and the first question (the questions come from the fixed bank, no LLM call), while an 8-worker thread pool writes the advice through the LLM API. Per-visitor, per-IP and daily spend caps, per-IP photo rate limits, per-check locks and in-memory checks that expire after an hour keep it safe to leave running.

```mermaid
flowchart LR
    R["Request threads"] -->|"odds, first question"| C[("Checks in memory<br/>1 hour expiry")]
    R -->|"submit jobs"| P["Thread pool<br/>8 workers"]
    P -->|"question pool,<br/>advice"| C
    P --> S["Retriever"]
    P <--> L["LLM API"]
```

| Endpoint | Does |
|---|---|
| `POST /api/check` | Reads the photo, returns the odds and the first question |
| `POST /api/answer` | Applies an answer, returns the next question or the diagnosis |
| `POST /api/next` | Waits for the next question when one is still being written |
| `POST /api/advice` | Returns the diagnosis once its advice is written |
| `POST /api/chat` | Answers a follow-up from the guides |

[**Background work, limits and concurrency**](docs/architecture/live-demo-server.md)

## Website

Plain HTML, CSS and JavaScript in `site/`, no framework and no build step, published as static assets on Cloudflare Workers: a home page, an eight-step walkthrough, an architecture page and the live demo. The live demo client marks your tap instantly, animates the odds, and drops stale replies when you switch photos.

```mermaid
flowchart LR
    H["Home"] --> T["Walkthrough"]
    H --> A["How it's built"]
    H --> L["Live demo"]
    T --> L
    A --> L
```

[**Pages, the demo client and its states**](docs/architecture/website.md)

## Data and experiments

Training and evaluation are reproducible end to end. DVC versions the datasets, MLflow logs the training and evaluation runs, and each experiment under `src/sprout/experiments` takes its settings from `configs/` and writes a JSON run record under `results/` that the reported numbers are read from.

```mermaid
flowchart LR
    D[("Datasets<br/>DVC")] --> X["Experiments<br/>configs/ + seeds"]
    X --> M[("MLflow<br/>runs")]
    X --> R[("Run records<br/>results/")]
    R --> T["pytest + ruff<br/>GitHub Actions"]
    X --> O["ONNX model +<br/>calibration"]
```

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

Run the live demo, which serves the website and its API:

```bash
.venv/Scripts/python tools/live_demo.py --site site
# then open http://127.0.0.1:8765/live.html
```

The demo needs the exported model in `models/` and the guide index in `knowledge/` (built as described in the [vision model](docs/architecture/vision-model.md) and [guide search](docs/architecture/guide-search.md) docs), plus an LLM API key in `.env` (the variable name is in `tools/live_demo.py`). Without a key it still runs, using 52 questions pre-extracted from the guides.

## Repository map

```
src/sprout/
  serve/        photo reading, the photo gate, calibration, the ONNX classifier
  rag/          guide ingest, the retriever, grounded answers
  train/ eval/  training loops, metrics, run records
  experiments/  the experiments behind every reported number
  web/          a full Flask app with accounts and history
tools/live_demo.py   the live demo server behind the website
site/                the website
docs/architecture/   the deep dives linked above
tests/               the pytest suite, run on every push
configs/ results/    experiment settings and the run records every number comes from
deploy/              Dockerfile, gunicorn entry point for the full app
```

## Author

**Aakash Senthilnathan**, Statistics and Data Science at UC Santa Barbara.
[LinkedIn](https://www.linkedin.com/in/aakashsenthilnathan/) · [GitHub](https://github.com/aakashksen) · aakash.senthilnathan@gmail.com

## License

[MIT](LICENSE) for the code. Datasets, pretrained weights and extension guides keep their own licenses and are not included. Sample leaf photos on the website come from the PlantDoc (CC BY 4.0) and PlantSeg (CC BY-NC 4.0) datasets.
