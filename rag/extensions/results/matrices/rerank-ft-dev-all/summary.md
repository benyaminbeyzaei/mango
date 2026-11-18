# Matrix rerank-ft-dev-all on dev-all

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261116-1720-baseline-dev-all | baseline | 0.382 | 0.479 [0.409, 0.558] | 0.583 | 0.727 | 0.461 | 0.402 [0.338, 0.470] | - | - | - | - | 163.4 | 0.000 | 0.000 | True |
| 20261116-1720-rerank-minilm-p20-dev-all | rerank-minilm-p20 | 0.381 | 0.495 [0.426, 0.569] | 0.583 | 0.727 | 0.442 | 0.393 [0.333, 0.460] | - | - | - | - | 1315.9 | 0.000 | 0.000 | True |
| 20261116-1723-rerank-minilm-ft-p20-dev-all | rerank-minilm-ft-p20 | 0.361 | 0.491 [0.425, 0.562] | 0.583 | 0.727 | 0.422 | 0.371 [0.315, 0.428] | - | - | - | - | 1393.8 | 0.000 | 0.000 | True |
