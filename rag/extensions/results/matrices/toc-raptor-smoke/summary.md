# Matrix toc-raptor-smoke on practice-smoke20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261108-1949-baseline-practice-smoke20 | baseline | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | - | - | - | - | 108.2 | 0.000 | 0.000 | True |
| 20261108-2202-topics-k256-m4-practice-smoke20 | topics-k256-m4 | 0.396 | 0.433 | 0.475 | 0.667 | 0.442 | 0.350 | - | - | - | - | 20.056 | 0.000 | 0.000 | True |
| 20261108-2202-toc-centroid-m4-practice-smoke20 | toc-centroid-m4 | 0.346 | 0.383 | 0.400 | 0.592 | 0.379 | 0.306 | - | - | - | - | 18.901 | 0.000 | 0.000 | True |
| 20261108-2202-toc-llm-m4-practice-smoke20 | toc-llm-m4 | 0.212 | 0.275 | 0.292 | 0.421 | 0.317 | 0.235 | - | - | - | - | 1124.9 | 2.000 | 1340.3 | True |
| 20261109-1632-toc-llm-topics-m4-practice-smoke20 | toc-llm-topics-m4 | 0.212 | 0.275 | 0.292 | 0.471 | 0.312 | 0.233 | - | - | - | - | 1524.6 | 2.000 | 2603.8 | True |
| 20261109-1633-toc-llm-flat-m4-practice-smoke20 | toc-llm-flat-m4 | 0.263 | 0.271 | 0.287 | 0.404 | 0.317 | 0.240 | - | - | - | - | 2373.0 | 1.000 | 6199.5 | True |
| 20261109-1728-raptor-lite-expand-practice-smoke20 | raptor-lite-expand | 0.404 | 0.450 | 0.550 | 0.779 | 0.479 | 0.376 | - | - | - | - | 109.0 | 0.000 | 0.000 | True |
| 20261109-1728-raptor-tree-expand-practice-smoke20 | raptor-tree-expand | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | - | - | - | - | 102.0 | 0.000 | 0.000 | True |
| 20261109-1728-raptor-lite-evidence-practice-smoke20 | raptor-lite-evidence | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.385 | - | - | - | - | 101.8 | 0.000 | 0.000 | True |
| 20261109-1728-raptor-tree-evidence-practice-smoke20 | raptor-tree-evidence | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | - | - | - | - | 103.9 | 0.000 | 0.000 | True |
