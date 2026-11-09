# Matrix c-dev-all on dev-all

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261109-1819-baseline-dev-all | baseline | 0.382 | 0.479 [0.409, 0.558] | 0.583 | 0.727 | 0.461 | 0.402 [0.338, 0.470] | - | - | - | - | 112.2 | 0.000 | 0.000 | True |
| 20261109-1820-enc-b-asis-dev-all | enc-b-asis | 0.342 | 0.437 [0.365, 0.514] | 0.480 | 0.708 | 0.398 | 0.349 [0.289, 0.416] | - | - | - | - | 114.8 | 0.000 | 0.000 | True |
| 20261109-1821-enc-b-reencode-dev-all | enc-b-reencode | 0.396 | 0.491 [0.419, 0.562] | 0.575 | 0.747 | 0.482 | 0.413 [0.352, 0.477] | - | - | - | - | 102.2 | 0.000 | 0.000 | True |
| 20261109-1821-enc-b-reencode-prefix-dev-all | enc-b-reencode-prefix | 0.418 | 0.526 [0.455, 0.599] | 0.590 | 0.726 | 0.490 | 0.434 [0.371, 0.502] | - | - | - | - | 105.5 | 0.000 | 0.000 | True |
| 20261109-1821-enc-b-translate-lstsq-dev-all | enc-b-translate-lstsq | 0.396 | 0.473 [0.405, 0.548] | 0.560 | 0.709 | 0.458 | 0.389 [0.332, 0.454] | - | - | - | - | 114.9 | 0.000 | 0.000 | True |
| 20261109-1821-enc-b-translate-procrustes-dev-all | enc-b-translate-procrustes | 0.355 | 0.425 [0.355, 0.497] | 0.503 | 0.670 | 0.434 | 0.363 [0.299, 0.431] | - | - | - | - | 117.3 | 0.000 | 0.000 | True |
