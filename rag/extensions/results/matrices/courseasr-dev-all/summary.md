# Matrix courseasr-dev-all on dev-all

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261114-1448-baseline-courseasr-dev-all | baseline-courseasr | 0.379 | 0.475 [0.404, 0.552] | 0.580 | 0.724 | 0.460 | 0.398 [0.338, 0.468] | 0.689 | 2.584 | 2.776 | 2.904 | 72494.6 | 1.000 | 3983.8 | True |
| 20261114-1721-raptor-tree-gpt5-evidence-courseasr-dev-all | raptor-tree-gpt5-evidence-courseasr | 0.379 | 0.475 [0.404, 0.552] | 0.576 | 0.724 | 0.459 | 0.398 [0.338, 0.467] | 0.663 | 2.472 | 2.704 | 2.784 | 74914.0 | 1.000 | 3988.2 | True |
