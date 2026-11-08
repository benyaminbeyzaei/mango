# Matrix context-smoke on practice-smoke20

| run | config | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | NDCG@10 | answer_norm | correctness | grounding | completeness | latency_ms | calls/q | tokens/q | fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261108-1858-baseline-practice-smoke20 | baseline | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | 0.579 | 2.200 | 2.300 | 2.450 | 52231.2 | 1.000 | 4135.6 | True |
| 20261108-1916-context-budgeted-practice-smoke20 | context-budgeted | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | 0.650 | 2.400 | 2.650 | 2.750 | 47431.5 | 1.000 | 3371.6 | True |
| 20261108-1933-context-dedupe-practice-smoke20 | context-dedupe | 0.404 | 0.475 | 0.550 | 0.779 | 0.479 | 0.386 | 0.671 | 2.600 | 2.700 | 2.750 | 42978.8 | 1.000 | 3249.8 | True |
