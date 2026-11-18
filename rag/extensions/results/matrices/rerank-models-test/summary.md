# Matrix rerank-models-test on test

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261116-2051-baseline-test | baseline | 0.382 | 0.479 [0.409, 0.558] | 0.583 | 0.727 | 0.461 | 0.402 [0.338, 0.470] | - | - | - | - | 117.1 | 0.000 | 0.000 | True |
| 20261116-2051-rerank-minilm-p20-test | rerank-minilm-p20 | 0.381 | 0.495 [0.426, 0.569] | 0.583 | 0.727 | 0.442 | 0.393 [0.333, 0.460] | - | - | - | - | 1571.2 | 0.000 | 0.000 | True |
| 20261116-2055-rerank-bge-m3-p20-test | rerank-bge-m3-p20 | 0.467 | 0.529 [0.465, 0.600] | 0.583 | 0.727 | 0.529 | 0.460 [0.399, 0.525] | - | - | - | - | 42116.7 | 0.000 | 0.000 | True |
| 20261116-2358-rerank-mxbai-v2-p20-test | rerank-mxbai-v2-p20 | 0.470 | 0.537 [0.472, 0.611] | 0.583 | 0.727 | 0.532 | 0.465 [0.402, 0.534] | - | - | - | - | 61896.6 | 0.000 | 0.000 | True |
