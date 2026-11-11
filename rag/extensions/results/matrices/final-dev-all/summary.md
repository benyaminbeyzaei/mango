# Matrix final-dev-all on dev-all

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261111-0115-baseline-dev-all | baseline | 0.382 | 0.479 [0.409, 0.558] | 0.583 | 0.727 | 0.461 | 0.402 [0.338, 0.470] | 0.687 | 2.568 | 2.792 | 2.880 | 87048.6 | 1.000 | 4004.9 | True |
| 20261111-0419-rerank-best-dev-all | rerank-best | 0.381 | 0.495 [0.426, 0.569] | 0.583 | 0.727 | 0.442 | 0.393 [0.333, 0.460] | 0.646 | 2.432 | 2.640 | 2.680 | 85282.7 | 1.000 | 4310.0 | True |
| 20261111-0719-raptor-tree-gpt5-evidence-dev-all | raptor-tree-gpt5-evidence | 0.382 | 0.477 [0.409, 0.556] | 0.579 | 0.727 | 0.460 | 0.400 [0.336, 0.470] | 0.690 | 2.568 | 2.824 | 2.888 | 90092.2 | 1.000 | 4085.9 | True |
