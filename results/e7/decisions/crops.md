### T6.4 - Retrieval, crops: dropped, base kept (2026-11-03)
- **Chose:** base = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Considered:** crops_from_title = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': True}
- **Evidence:** record `E7-20261103T211528498754Z`, n=60 answerable rag_dev questions, gold hit per `config.gold_matching`. Per system recall@1 / @3 / @5 / @10 / MRR: base: 0.417 / 0.617 / 0.717 / 0.817 / 0.543; crops_from_title: 0.400 / 0.600 / 0.717 / 0.817 / 0.533; 
  - recall@5 crops_from_title minus base: +0.000, 95% CI [+0.000, +0.000], McNemar p=1.000 (no measurable difference)
- **Device:** {'device': 'cuda', 'torch': '2.11.0+cu128', 'gpu': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'cuda': '12.8', 'peak_gpu_mib': 137}
- **Would change if:** a larger or reviewed question set shows a different recall@5 gain.