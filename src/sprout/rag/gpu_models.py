"""Embedder and reranker that run on CUDA through torch. Torch is imported only when one is built,
so serving (which uses fastembed and ONNX) never loads it."""

from __future__ import annotations

import numpy as np

RERANKER_TORCH_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"  # the weights behind Xenova/ms-marco-MiniLM-L-6-v2
MAX_LENGTH = 512


def resolve_device(requested: str) -> str:
    """The device to use, or an error. A missing GPU is never replaced by the CPU."""
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; refusing to fall back to the CPU")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"unknown device {requested!r}")


def _device_info(model, requested: str) -> dict:
    import torch

    actual = next(model.parameters()).device.type
    if actual != requested:
        raise RuntimeError(f"model is on {actual}, not {requested}")
    info = {"device": actual, "torch": torch.__version__}
    if actual == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


class TorchEmbedder:
    """bge-small style CLS embeddings, L2-normalised; the same numbers as the saved index vectors."""

    cache_ok = True  # its vectors may be saved and loaded like the default embedder's

    def __init__(self, model: str, *, device: str = "cuda", batch: int = 64) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._device = resolve_device(device)
        self._batch = batch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).to(self._device).eval()
        self.info = _device_info(self._model, self._device)

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self._model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for lo in range(0, len(order), self._batch):
                ids = order[lo : lo + self._batch]
                enc = self._tokenizer([texts[i] for i in ids], padding=True, truncation=True,
                                      max_length=MAX_LENGTH, return_tensors="pt")
                cls = self._model(**{k: v.to(self._device) for k, v in enc.items()}).last_hidden_state[:, 0]
                out[ids] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class TorchReranker:
    """Cross-encoder scores squashed to 0..1, as the ONNX reranker does."""

    def __init__(self, model: str = RERANKER_TORCH_MODEL, *, device: str = "cuda", batch: int = 32) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = resolve_device(device)
        self._batch = batch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model).to(self._device).eval()
        self.info = _device_info(self._model, self._device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        out: list[float] = []
        with torch.inference_mode():
            for lo in range(0, len(texts), self._batch):
                part = texts[lo : lo + self._batch]
                enc = self._tokenizer([query] * len(part), part, padding=True, truncation=True,
                                      max_length=MAX_LENGTH, return_tensors="pt")
                logits = self._model(**{k: v.to(self._device) for k, v in enc.items()}).logits.reshape(-1)
                out += torch.sigmoid(logits).cpu().tolist()
        return out
