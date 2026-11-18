# Matrix final-practice on practice

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261111-0105-baseline-practice | baseline | 0.428 | 0.485 [0.443, 0.526] | 0.582 | 0.768 | 0.493 | 0.425 [0.385, 0.462] | - | - | - | - | 143.3 | 0.000 | 0.000 | True |
| 20261111-0105-rerank-best-practice | rerank-best | 0.418 | 0.509 [0.466, 0.549] | 0.582 | 0.768 | 0.470 | 0.418 [0.380, 0.454] | - | - | - | - | 1285.6 | 0.000 | 0.000 | True |
| 20261111-0113-raptor-tree-gpt5-evidence-practice | raptor-tree-gpt5-evidence | 0.428 | 0.485 [0.443, 0.526] | 0.582 | 0.768 | 0.492 | 0.425 [0.385, 0.462] | - | - | - | - | 207.1 | 0.000 | 0.000 | True |
