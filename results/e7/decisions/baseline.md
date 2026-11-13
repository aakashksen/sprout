### T6.4 - Retrieval, baseline: kept dense (2026-11-03)
- **Chose:** dense = {'index': 'index', 'mode': 'dense', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}
- **Considered:** hybrid = {'index': 'index', 'mode': 'hybrid', 'rerank': False, 'query_mode': 'question', 'crops_from_title': False}; hybrid_rerank = {'index': 'index', 'mode': 'hybrid', 'rerank': True, 'query_mode': 'question', 'crops_from_title': False}
- **Evidence:** record `E7-20261103T211314827721Z`, n=60 answerable rag_dev questions, gold hit per `config.gold_matching`. Per system recall@1 / @3 / @5 / @10 / MRR: dense: 0.417 / 0.617 / 0.717 / 0.817 / 0.543; hybrid: 0.367 / 0.583 / 0.717 / 0.783 / 0.507; hybrid_rerank: 0.600 / 0.800 / 0.833 / 0.933 / 0.712; 
  - recall@5 hybrid minus dense: +0.000, 95% CI [-0.150, +0.150], McNemar p=1.000 (no measurable difference)
  - recall@5 hybrid_rerank minus dense: +0.117, 95% CI [+0.000, +0.233], McNemar p=0.092 (no measurable difference)
- **Device:** {'device': 'cuda', 'torch': '2.11.0+cu128', 'gpu': 'NVIDIA GeForce RTX 4070 Laptop GPU', 'cuda': '12.8', 'peak_gpu_mib': 480}
- **Would change if:** a larger or reviewed question set shows a different recall@5 gain.