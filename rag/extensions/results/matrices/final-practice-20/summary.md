# Matrix final-practice-20 on practice-20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261111-0017-baseline-practice-20 | baseline | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.386 [0.244, 0.525] | 0.592 | 2.100 | 2.450 | 2.550 | 141279.4 | 1.000 | 4069.2 | True |
| 20261110-2313-rerank-best-practice-20 | rerank-best | 0.246 | 0.483 [0.317, 0.650] | 0.550 | 0.779 | 0.336 | 0.313 [0.188, 0.436] | 0.729 | 2.700 | 3.000 | 3.050 | 181390.7 | 1.000 | 4050.9 | True |
| 20261110-2313-raptor-tree-gpt5-evidence-practice-20 | raptor-tree-gpt5-evidence | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.385 [0.244, 0.524] | 0.692 | 2.700 | 2.850 | 2.750 | 188601.5 | 1.000 | 3953.8 | True |
