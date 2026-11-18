# Matrix rerank-practice on practice

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261110-1934-baseline-practice | baseline | 0.428 | 0.485 [0.443, 0.526] | 0.582 | 0.768 | 0.493 | 0.425 [0.385, 0.462] | - | - | - | - | 114.2 | 0.000 | 0.000 | True |
| 20261110-1935-rerank-minilm-p20-practice | rerank-minilm-p20 | 0.418 | 0.509 [0.466, 0.549] | 0.582 | 0.768 | 0.470 | 0.418 [0.380, 0.454] | - | - | - | - | 1644.1 | 0.000 | 0.000 | True |
| 20261110-1944-rerank-minilm-p50-practice | rerank-minilm-p50 | 0.400 | 0.494 [0.454, 0.534] | 0.572 | 0.768 | 0.457 | 0.404 [0.368, 0.439] | - | - | - | - | 3047.9 | 0.000 | 0.000 | True |
| 20261110-2002-rerank-minilm-p100-practice | rerank-minilm-p100 | 0.391 | 0.486 [0.443, 0.526] | 0.562 | 0.768 | 0.455 | 0.401 [0.363, 0.436] | - | - | - | - | 4497.7 | 0.000 | 0.000 | True |
| 20261110-2028-rerank-bge-p20-practice | rerank-bge-p20 | 0.375 | 0.485 [0.442, 0.525] | 0.582 | 0.768 | 0.414 | 0.376 [0.340, 0.411] | - | - | - | - | 9699.5 | 0.000 | 0.000 | True |
| 20261110-2125-rerank-bge-p50-practice | rerank-bge-p50 | 0.344 | 0.436 [0.396, 0.478] | 0.544 | 0.768 | 0.385 | 0.341 [0.306, 0.376] | - | - | - | - | 18358.2 | 0.000 | 0.000 | True |
