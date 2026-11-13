### T6.4 - Retrieval, queries: dropped, base kept (2026-11-03)
- **Chose:** base = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Considered:** history = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'history', 'crops_from_title': False}; diseases = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'diseases', 'crops_from_title': False}; full = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'full', 'crops_from_title': False}
- **Evidence:** record `E7-20261103T211620506065Z`, n=60 answerable rag_dev questions, gold hit per `config.gold_matching`. Per system recall@1 / @3 / @5 / @10 / MRR: base: 0.417 / 0.617 / 0.717 / 0.817 / 0.543; history: 0.417 / 0.600 / 0.683 / 0.783 / 0.528; diseases: 0.367 / 0.550 / 0.700 / 0.833 / 0.507; full: 0.317 / 0.500 / 0.667 / 0.783 / 0.456; 
  - recall@5 history minus base: -0.033, 95% CI [-0.100, +0.033], McNemar p=0.625 (no measurable difference)
  - recall@5 diseases minus base: -0.017, 95% CI [-0.050, +0.000], McNemar p=1.000 (no measurable difference)
  - recall@5 full minus base: -0.050, 95% CI [-0.133, +0.017], McNemar p=0.375 (no measurable difference)
- **Device:** {'device': 'cuda', 'torch': '2.11.0+cu128', 'gpu': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'cuda': '12.8', 'peak_gpu_mib': 137}
- **Would change if:** a larger or reviewed question set shows a different recall@5 gain.