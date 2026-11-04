### T6.4 - Retrieval, chunking: dropped, base kept (2026-11-03)
- **Chose:** base = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Considered:** structured = {'index': 'index-structured', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Evidence:** record `E7-20261103T211358771284Z`, n=60 answerable rag_dev questions, gold hit per `config.gold_matching`. Per system recall@1 / @3 / @5 / @10 / MRR: base: 0.417 / 0.617 / 0.717 / 0.817 / 0.543; structured: 0.267 / 0.483 / 0.683 / 0.817 / 0.425; 
  - recall@5 structured minus base: -0.033, 95% CI [-0.150, +0.083], McNemar p=0.774 (no measurable difference)
- **Device:** {'device': 'cuda', 'torch': '2.11.0+cu128', 'gpu': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'cuda': '12.8', 'peak_gpu_mib': 137}
- **Would change if:** a larger or reviewed question set shows a different recall@5 gain.