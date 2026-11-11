# Matrix rerank-smoke on practice-smoke20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261110-1906-baseline-practice-smoke20 | baseline | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.386 [0.244, 0.525] | - | - | - | - | 104.0 | 0.000 | 0.000 | True |
| 20261110-1907-rerank-minilm-p20-practice-smoke20 | rerank-minilm-p20 | 0.246 | 0.483 [0.317, 0.650] | 0.550 | 0.779 | 0.336 | 0.313 [0.188, 0.436] | - | - | - | - | 1086.5 | 0.000 | 0.000 | True |
| 20261110-1907-rerank-minilm-p50-practice-smoke20 | rerank-minilm-p50 | 0.221 | 0.412 [0.267, 0.562] | 0.575 | 0.779 | 0.322 | 0.280 [0.163, 0.405] | - | - | - | - | 2788.5 | 0.000 | 0.000 | True |
| 20261110-1908-rerank-minilm-p100-practice-smoke20 | rerank-minilm-p100 | 0.221 | 0.362 [0.217, 0.513] | 0.521 | 0.779 | 0.315 | 0.258 [0.148, 0.378] | - | - | - | - | 4602.4 | 0.000 | 0.000 | True |
| 20261110-1909-rerank-bge-p20-practice-smoke20 | rerank-bge-p20 | 0.338 | 0.438 [0.267, 0.600] | 0.550 | 0.779 | 0.350 | 0.321 [0.185, 0.454] | - | - | - | - | 9803.5 | 0.000 | 0.000 | True |
| 20261110-1913-rerank-bge-p50-practice-smoke20 | rerank-bge-p50 | 0.317 | 0.342 [0.163, 0.513] | 0.483 | 0.779 | 0.310 | 0.268 [0.127, 0.406] | - | - | - | - | 18179.5 | 0.000 | 0.000 | True |
| 20261110-1919-rerank-bge-p100-practice-smoke20 | rerank-bge-p100 | 0.292 | 0.325 [0.158, 0.492] | 0.492 | 0.779 | 0.281 | 0.248 [0.119, 0.381] | - | - | - | - | 43149.9 | 0.000 | 0.000 | True |
