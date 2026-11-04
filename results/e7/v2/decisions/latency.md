### T6.4 - Cost of reranking, for T7.4 (2026-11-03)
- **Chose:** nothing; this is a measurement. Milliseconds per search with 15 candidates reranked (k=5, pool 3):
  - cuda: reranker call p50 30.1 / p95 42.5 ms; whole search p50 57.7 / p95 79.5 ms
  - cpu_torch: reranker call p50 631.1 / p95 1063.4 ms; whole search p50 682.4 / p95 1085.9 ms
  - cpu_onnx: reranker call p50 604.2 / p95 975.6 ms; whole search p50 630.0 / p95 1078.5 ms
- **Do the paths agree?** recall@5 per path: rerank_cuda 0.833, rerank_cpu_torch 0.833, rerank_cpu_onnx 0.833
- **Machine:** {'cpu': 'AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD', 'cpu_count': 24, 'python': '3.12.8', 'gpu_memory_in_use_mib_when_stage_began': 2043, 'candidates_reranked_per_search': 15, 'searches_timed': 60, 'warmup_searches': 3, 'note': 'Embedding the query runs on CUDA in every path, so search_ms differs between paths by the reranker only.'}
- **Device:** {'device': 'cuda', 'torch': '2.11.0+cu128', 'gpu': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'cuda': '12.8', 'peak_gpu_mib': 479}
- **Evidence:** record `E7-20261103T235630916335Z`.
- **Would change if:** another machine or a busier GPU gives other numbers; this is one laptop.