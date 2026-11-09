# Matrix a-smoke on practice-smoke20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261108-1949-baseline-practice-smoke20 | baseline | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | - | - | - | - | 108.2 | 0.000 | 0.000 | True |
| 20261108-1949-bm25-practice-smoke20 | bm25 | 0.167 | 0.183 | 0.212 | 0.442 | 0.246 | 0.160 | - | - | - | - | 1.791 | 0.000 | 0.000 | True |
| 20261109-1134-grep-static-practice-smoke20 | grep-static | 0.046 | 0.133 | 0.158 | 0.217 | 0.111 | 0.074 | - | - | - | - | 1404.8 | 1.000 | 146.1 | True |
| 20261109-1134-grep-agentic-practice-smoke20 | grep-agentic | 0.067 | 0.067 | 0.067 | 0.092 | 0.175 | 0.087 | - | - | - | - | 5571.8 | 3.900 | 2714.0 | True |
| 20261108-2202-topics-k256-m2-practice-smoke20 | topics-k256-m2 | 0.279 | 0.325 | 0.354 | 0.521 | 0.333 | 0.265 | - | - | - | - | 26.935 | 0.000 | 0.000 | True |
| 20261108-2202-topics-k256-m4-practice-smoke20 | topics-k256-m4 | 0.396 | 0.433 | 0.475 | 0.667 | 0.442 | 0.350 | - | - | - | - | 20.056 | 0.000 | 0.000 | True |
| 20261108-2202-topics-k256-m8-practice-smoke20 | topics-k256-m8 | 0.404 | 0.450 | 0.525 | 0.700 | 0.479 | 0.376 | - | - | - | - | 20.443 | 0.000 | 0.000 | True |
| 20261108-2202-toc-centroid-m4-practice-smoke20 | toc-centroid-m4 | 0.346 | 0.383 | 0.400 | 0.592 | 0.379 | 0.306 | - | - | - | - | 18.901 | 0.000 | 0.000 | True |
| 20261108-2202-toc-llm-m4-practice-smoke20 | toc-llm-m4 | 0.212 | 0.275 | 0.292 | 0.421 | 0.317 | 0.235 | - | - | - | - | 1124.9 | 2.000 | 1340.3 | True |
| 20261108-2202-raptor-lite-expand-practice-smoke20 | raptor-lite-expand | 0.404 | 0.450 | 0.550 | 0.779 | 0.479 | 0.376 | - | - | - | - | 104.1 | 0.000 | 0.000 | True |
| 20261108-2203-lazy-graph-rr-practice-smoke20 | lazy-graph-rr | 0.029 | 0.071 | 0.096 | 0.267 | 0.050 | 0.043 | - | - | - | - | 117.9 | 0.000 | 0.000 | True |
