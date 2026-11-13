# Pretrained weights and models

Looked up 2026-11-04; nothing was downloaded for this lookup.
Quotes and tags were read from each primary source named here. "Unconfirmed" means no primary source that could be reached states it.
- **ConvNeXt-Tiny, torchvision `IMAGENET1K_V1`** (E1, E3, E4, E12): no license for the weights is stated at https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html. Unconfirmed, and fine-tuned checkpoints inherit the question. torchvision's code is BSD-3-Clause and Meta's ConvNeXt code is MIT, which cover code only.
- **ConvNeXt V2 Tiny, `timm/convnextv2_tiny.fcmae_ft_in22k_in1k`** (the final model, T4.9): `cc-by-nc-4.0`, from its Hugging Face model card, as recorded in the T4.9 decision entry (reported by the coordinator; not re-read here). The ConvNeXt-V2 repository's LICENSE file has an MIT section and an Attribution-NonCommercial 4.0 section and does not say which covers the weights; both were read through a summarizing fetch, not verbatim. The weights are treated as non-commercial, and the fine-tuned final model inherits that. Publishing or commercial use of weights derived from them needs the person's check.
- **DINOv2 small, `dinov2_vits14`** at commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`: Apache-2.0. The README states "DINOv2 code and model weights are released under the Apache License 2.0." Its training data's terms were not checked.
- **`BAAI/bge-small-en-v1.5`:** MIT, from the card's metadata. Serving downloads the third-party ONNX copy `Qdrant/bge-small-en-v1.5-onnx-Q`, tagged MIT by its uploader.
- **`cross-encoder/ms-marco-MiniLM-L6-v2`** (GPU build): Apache-2.0, from the card's metadata.
- **`Xenova/ms-marco-MiniLM-L-6-v2`** (serving): no license in its metadata. Unconfirmed. That it holds the same weights as the model above rests on its card and tag; the weights were not compared.
- **Training data behind these:** ImageNet's access agreement was seen only through a search snippet, because the page refused connections, so its terms are unconfirmed here. MS MARCO's page says its datasets are for non-commercial research; whether that limits derived reranker weights is unconfirmed.

All of this is local, non-commercial and unpublished use today. The person checks the unconfirmed items before publishing any weights.
