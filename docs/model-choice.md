# Local model choice

Tested 2026-11-02 on this RTX 4070 laptop GPU (8 GB) through Ollama, with thinking off: 6 tool choices, 10 consistency labels, and 6 cited answers including abstention and an instruction hidden in the transcript.

| Model | On GPU | Loaded | Speed | Avg answer | Tools | Labels | Cards |
|---|---|---|---|---|---|---|---|
| **qwen3.5:4b-q4_K_M** | **100%** | 3.3 GB | 63.5 tok/s | 0.6 s | 5/6 | 10/10 | 6/6 |
| gemma4:e4b-it-qat | 100% | 3.1 GB | 61.9 tok/s | 0.6 s | 5/6 | 9/10 | 6/6 |
| qwen3.5:9b-q4_K_M | 85% | 6.4 GB | 29.8 tok/s | 1.0 s | 6/6 | 9/10 | 5/6 |
| granite4.2:8b | 92% | 6.8 GB | 27.7 tok/s | 1.1 s | 6/6 | 10/10 | 5/6 |

**Chosen: `qwen3.5:4b-q4_K_M`.** It fits fully on the GPU, is about twice as fast as the larger models, and leaves room for other work. Apache-2.0, not gated (Hugging Face `Qwen/Qwen3.5-4B`), and it accepts images.

`gemma4:e4b-it-qat` is the second local model, used as the separate judge wherever the writer and judge must differ.

This is a small test for choosing a model, so these numbers must never be used as results or resume claims. The installed `gemma4:26b` (18.7 GB) doesn't fit this GPU and isn't used.
