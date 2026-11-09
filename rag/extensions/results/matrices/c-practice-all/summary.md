# Matrix c-practice-all on practice-all

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261109-1811-baseline-practice-all | baseline | 0.428 | 0.485 [0.443, 0.526] | 0.582 | 0.768 | 0.493 | 0.425 [0.385, 0.462] | - | - | - | - | 108.4 | 0.000 | 0.000 | True |
| 20261109-1815-enc-b-asis-practice-all | enc-b-asis | 0.363 | 0.465 [0.422, 0.508] | 0.554 | 0.711 | 0.401 | 0.361 [0.324, 0.399] | - | - | - | - | 113.3 | 0.000 | 0.000 | True |
| 20261109-1816-enc-b-reencode-practice-all | enc-b-reencode | 0.428 | 0.505 [0.462, 0.549] | 0.579 | 0.739 | 0.495 | 0.432 [0.393, 0.471] | - | - | - | - | 101.4 | 0.000 | 0.000 | True |
| 20261109-1817-enc-b-reencode-prefix-practice-all | enc-b-reencode-prefix | 0.441 | 0.516 [0.476, 0.560] | 0.589 | 0.759 | 0.492 | 0.437 [0.398, 0.476] | - | - | - | - | 104.2 | 0.000 | 0.000 | True |
| 20261109-1817-enc-b-translate-lstsq-practice-all | enc-b-translate-lstsq | 0.389 | 0.467 [0.423, 0.509] | 0.551 | 0.724 | 0.470 | 0.400 [0.361, 0.439] | - | - | - | - | 112.9 | 0.000 | 0.000 | True |
| 20261109-1818-enc-b-translate-procrustes-practice-all | enc-b-translate-procrustes | 0.342 | 0.422 [0.376, 0.463] | 0.496 | 0.669 | 0.419 | 0.353 [0.316, 0.392] | - | - | - | - | 112.7 | 0.000 | 0.000 | True |
