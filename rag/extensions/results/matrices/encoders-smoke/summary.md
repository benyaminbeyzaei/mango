# Matrix encoders-smoke on practice-smoke20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261108-1949-baseline-practice-smoke20 | baseline | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | - | - | - | - | 108.2 | 0.000 | 0.000 | True |
| 20261108-2213-enc-b-asis-practice-smoke20 | enc-b-asis | 0.292 | 0.400 | 0.512 | 0.696 | 0.402 | 0.319 | - | - | - | - | 101.1 | 0.000 | 0.000 | True |
| 20261108-2213-enc-b-reencode-practice-smoke20 | enc-b-reencode | 0.350 | 0.438 | 0.546 | 0.783 | 0.496 | 0.375 | - | - | - | - | 96.228 | 0.000 | 0.000 | True |
| 20261108-2214-enc-b-reencode-prefix-practice-smoke20 | enc-b-reencode-prefix | 0.354 | 0.396 | 0.512 | 0.754 | 0.487 | 0.362 | - | - | - | - | 93.949 | 0.000 | 0.000 | True |
| 20261108-2214-enc-b-translate-lstsq-practice-smoke20 | enc-b-translate-lstsq | 0.417 | 0.458 | 0.479 | 0.771 | 0.507 | 0.396 | - | - | - | - | 104.3 | 0.000 | 0.000 | True |
| 20261108-2214-enc-b-translate-procrustes-practice-smoke20 | enc-b-translate-procrustes | 0.308 | 0.425 | 0.458 | 0.700 | 0.466 | 0.351 | - | - | - | - | 104.4 | 0.000 | 0.000 | True |
