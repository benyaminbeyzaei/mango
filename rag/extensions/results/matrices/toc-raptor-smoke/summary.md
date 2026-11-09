# Matrix toc-raptor-smoke on practice-smoke20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261108-1949-baseline-practice-smoke20 | baseline | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.386 [0.244, 0.525] | - | - | - | - | 108.2 | 0.000 | 0.000 | True |
| 20261108-2202-topics-k256-m4-practice-smoke20 | topics-k256-m4 | 0.396 | 0.433 [0.283, 0.600] | 0.475 | 0.667 | 0.442 | 0.350 [0.219, 0.478] | - | - | - | - | 20.056 | 0.000 | 0.000 | True |
| 20261108-2202-toc-centroid-m4-practice-smoke20 | toc-centroid-m4 | 0.346 | 0.383 [0.217, 0.550] | 0.400 | 0.592 | 0.379 | 0.306 [0.177, 0.441] | - | - | - | - | 18.901 | 0.000 | 0.000 | True |
| 20261108-2202-toc-llm-m4-practice-smoke20 | toc-llm-m4 | 0.212 | 0.275 [0.125, 0.450] | 0.292 | 0.421 | 0.317 | 0.235 [0.107, 0.381] | - | - | - | - | 1124.9 | 2.000 | 1340.3 | True |
| 20261109-1632-toc-llm-topics-m4-practice-smoke20 | toc-llm-topics-m4 | 0.212 | 0.275 [0.125, 0.450] | 0.292 | 0.471 | 0.312 | 0.233 [0.105, 0.380] | - | - | - | - | 1524.6 | 2.000 | 2603.8 | True |
| 20261109-1633-toc-llm-flat-m4-practice-smoke20 | toc-llm-flat-m4 | 0.263 | 0.271 [0.117, 0.433] | 0.287 | 0.404 | 0.317 | 0.240 [0.109, 0.381] | - | - | - | - | 2373.0 | 1.000 | 6199.5 | True |
| 20261109-1728-raptor-lite-expand-practice-smoke20 | raptor-lite-expand | 0.404 | 0.450 [0.292, 0.608] | 0.550 | 0.779 | 0.479 | 0.376 [0.239, 0.509] | - | - | - | - | 109.0 | 0.000 | 0.000 | True |
| 20261109-1728-raptor-tree-expand-practice-smoke20 | raptor-tree-expand | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.386 [0.244, 0.525] | - | - | - | - | 102.0 | 0.000 | 0.000 | True |
| 20261109-1728-raptor-lite-evidence-practice-smoke20 | raptor-lite-evidence | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.385 [0.244, 0.524] | - | - | - | - | 101.8 | 0.000 | 0.000 | True |
| 20261109-1728-raptor-tree-evidence-practice-smoke20 | raptor-tree-evidence | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.386 [0.244, 0.525] | - | - | - | - | 103.9 | 0.000 | 0.000 | True |
| 20261109-1953-toc-centroid-m4-gpt5labels-practice-smoke20 | toc-centroid-m4-gpt5labels | 0.346 | 0.383 [0.217, 0.550] | 0.400 | 0.592 | 0.379 | 0.306 [0.177, 0.441] | - | - | - | - | 872.0 | 0.000 | 0.000 | True |
| 20261109-1953-toc-llm-m4-gpt5labels-practice-smoke20 | toc-llm-m4-gpt5labels | 0.212 | 0.275 [0.125, 0.442] | 0.317 | 0.471 | 0.350 | 0.251 [0.112, 0.395] | - | - | - | - | 4009.9 | 2.000 | 3238.6 | True |
| 20261109-1954-toc-llm-topics-m4-gpt5labels-practice-smoke20 | toc-llm-topics-m4-gpt5labels | 0.287 | 0.333 [0.183, 0.492] | 0.392 | 0.579 | 0.392 | 0.288 [0.153, 0.420] | - | - | - | - | 4527.5 | 2.000 | 5433.3 | True |
| 20261109-1956-toc-llm-flat-m4-gpt5labels-practice-smoke20 | toc-llm-flat-m4-gpt5labels | 0.263 | 0.300 [0.150, 0.450] | 0.329 | 0.396 | 0.333 | 0.250 [0.130, 0.373] | - | - | - | - | 6543.1 | 1.000 | 13126.5 | True |
| 20261109-1958-raptor-tree-gpt5-expand-practice-smoke20 | raptor-tree-gpt5-expand | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.387 [0.244, 0.531] | - | - | - | - | 1036.4 | 0.000 | 0.000 | True |
| 20261109-1958-raptor-tree-gpt5-evidence-practice-smoke20 | raptor-tree-gpt5-evidence | 0.404 | 0.475 [0.316, 0.642] | 0.550 | 0.779 | 0.479 | 0.385 [0.244, 0.524] | - | - | - | - | 945.0 | 0.000 | 0.000 | True |
