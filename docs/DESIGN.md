# SPROUT V2 design

## Purpose
A person photographs a sick leaf and gets a likely condition, an honest sense of how sure the system is, and care advice that links to its source.

The original reached a very high score on lab-style PlantVillage photos, from a random split with no separate test set. That doesn't show how it does on a real garden photo. The rebuild's story, and the reason it fills the resume's ML gap, is:

> **The lab score was X. On real photos it fell to Y. Here's why, using Grad-CAM. Here's what raised it to Z, with statistics showing the gain is real.**

ContextLens covers agents, LLM fine-tuning, and C++. SPROUT covers **training rigor, computer vision, classic ML, statistics, experiment tracking, and model optimization**, so the two don't repeat each other.

## What the user sees
- Upload or take a photo, and get the top condition with up to 2 plausible alternatives, each with a calibrated confidence.
- **Unsure** is a real answer: "I'm not sure, try a closer photo in daylight", with specific guidance. It's better than a confident wrong answer.
- **Not a plant** is handled using model confidence and an out-of-distribution score, since the official data has no background class, with real rejection rates measured.
- Choosing a different condition is stored as the user's choice, separate from the model's prediction.
- The chatbot's advice has clickable sources (document and page). When no document supports an answer, it says so.

## Architecture

```
 Photo ─> validate and fix orientation ─> shared preprocessing ─> ONNX model (CPU) ─> calibration ─> answer or "unsure"
                                                                                           │
 Knowledge docs ─> extraction check ─> chunk with crop/disease/page metadata               ▼
                     ─> ChromaDB + BM25 ─> metadata filter ─> rerank ─> per-passage filter ─> LLM ─> cited answer
                                                                                           │
                                                                     Flask app, SQLite/uploads volume, Docker

 Training (separate): manifests + DVC ─> PyTorch ─> MLflow ─> evaluation ─> run records ─> claims
```

## Decisions

| Area | Decision | Reason |
|---|---|---|
| Data | Official 38-class PlantVillage release (CC BY-SA 3.0) for training; PlantDoc and PlantSeg for real-world testing; COCO 2017 validation, without potted-plant images, for `not_plant`. COCO and PlantDoc images stay local and are never redistributed | Measures the real gap with data whose source is known |
| Splits | Group by the source IDs in the original filenames and by perceptual hashing, including flips and rotations, then split by group; permanent manifests | A source ID isn't proven to identify one physical leaf, and same-leaf photos that look different can still slip through, so reports say "near-duplicate-grouped" |
| Primary model | ConvNeXt V2 Tiny, if it scores at least as well as E4 and E12 on the same 29 classes (T4.9); otherwise the best of those | The person's decision, 2026-11-04; a newer version of the original architecture |
| Comparisons | HOG and color histograms + XGBoost; frozen ConvNeXt head; DINOv2 | Classic ML and a newer model |
| Fixes for real photos | Train on real photos (PlantDoc and PlantSeg) and lab photos together from the start, with lab capped per class; stronger augmentation and background randomization; a separate arm cropping to the disease area with a segmentation model trained on PlantSeg's masks. All selected on `field_dev`, with `field_test` frozen. The person checks before publishing any weights trained on PlantDoc | Real photos are what the model fell on, and training on lab photos alone fell far short |
| Understanding | Grad-CAM | Shows what the model looks at |
| Statistics | Bootstrap confidence intervals, McNemar's test, ECE calibration; top-1, top-3, top-5, and plant-given top-1 and top-3, stating options per plant | Uses the stats major; plant-given top-3 is easy, since plants average 2.7 diseases, so it's never shown alone |
| Tracking | MLflow for runs, DVC for data, run records for claims | Reproducibility |
| Serving | ONNX with dynamic or static quantization, CPU | Cheap hosting and a real optimization result |
| RAG | ChromaDB + BM25, fastembed (ONNX) embeddings and cross-encoder rerank, metadata filters | The original claimed hybrid and rerank without having them; now it's built, and claimed only if measured to help |
| LLM | DeepSeek removed from the build. Ollama locally with `qwen3.5:4b-q4_K_M` (tested on this GPU, see `docs/model-choice.md`), and Claude through the `claude -p` CLI as a comparison, with no API key | Available now |
| App | Keep Flask and the existing templates, split into modules | Enough for the product; rewriting adds no value |
| Hosting | Google Cloud Run, only when the person does it | They hold the account |

**Hosted chatbot:** `claude -p` and local Ollama can't serve a public site. So the hosted demo needs the person's decision at H3: bring an API key for chat, or host with the identification model and leave chat local-only.

## Limits
- **Hardware:** RTX 4070 laptop GPU with 8 GB VRAM, shared with other projects. One GPU job at a time. Mixed precision is on. If something can't fit, block the task.
- **Claude usage:** comparisons use Sonnet, with a 5,000-call cap as a runaway guard, counted in `results/claude_calls.json`. `ClaudeCodeClient` refuses past the cap. Only the person can raise it.
- **Experiments:** a limited search. Every setting tried goes in the run record.
- **Data leaving the machine:** only benchmark questions and public document text go to a cloud model. Never user photos or history.

## Non-goals
Detecting pests or diseases outside the supported classes, plant species identification, mobile apps, payment, an admin console, multi-agent features (ContextLens has those), and a frontend redesign.

## Anti-goals: this project has failed if
- A number on the resume can't be traced to a run record, or a lab score is presented as real-world performance.
- It gives a confident diagnosis for a photo it should have called unsure, and that case wasn't measured.
- Treatment advice appears with no source and no label saying so.
- The person can't explain and defend any part in an interview.

## Left to the builder
Library versions, internal helpers, and hyperparameters within the stated search, as long as choices are recorded in the run records.
