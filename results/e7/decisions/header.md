### T6.4 - Retrieval, header: dropped, base kept (2026-11-03)
- **Chose:** base = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Considered:** header = {'index': 'index-header', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Evidence:** record `E7-20261103T211442616430Z`, n=60 answerable rag_dev questions, gold hit per `config.gold_matching`. Per system recall@1 / @3 / @5 / @10 / MRR: base: 0.417 / 0.617 / 0.717 / 0.817 / 0.543; header: 0.383 / 0.683 / 0.767 / 0.917 / 0.550; 
  - recall@5 header minus base: +0.050, 95% CI [-0.050, +0.150], McNemar p=0.549 (no measurable difference)
- **Device:** {'device': 'cuda', 'torch': '2.11.0+cu128', 'gpu': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'cuda': '12.8', 'peak_gpu_mib': 137}
- **Would change if:** a larger or reviewed question set shows a different recall@5 gain.