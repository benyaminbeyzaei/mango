# Results

Every result of the project, in one place and one format. The numbers are copied from the committed
summaries in `rag/extensions/results/matrices/<name>/` (named under each table).

**How to read the tables**

- **Question sets.** `practice` holds 350 questions, used to try ideas and pick settings. `test` holds 125
  questions, used only to report; nothing is tuned on it. The course calls it the `dev` split. `practice-20` is a
  fixed 20-question sample of practice, used for quick checks (its intervals are wide).
- **Retrieval:** Recall@5, Recall@10, MRR@10 and NDCG@10 from the course's own metrics (0 to 1, higher is better).
- **Answers:** a Qwen3.5-397B answer per question, scored by the course judge (Mistral-Small-4) for correctness,
  grounding and completeness (0 to 4 each). The **answer score** is their sum divided by 12 (0 to 1); summaries call it
  `answer_norm`.
- **Intervals** `[lo, hi]` are 95 % bootstrap intervals over questions (1,000 resamples, seed 0). **vs baseline** is
  the paired difference per question, with its own interval; if the interval contains 0, the change is not
  significant.
- **Latency** is the mean time per question in milliseconds, from fresh runs only (no cache reads), on the laptop CPU.
  Rows marked "retrieval only" made no answer call.
- Unless a section says otherwise: whisper-1 transcripts, gte-small queries on the course's enc_a index, top 5
  records in the prompt, the course prompt.

## 1. Answers on test: does a change help the final answer?

`final-test`, `final-bge-test`, `final-mxbai-test` (125 questions each; paired against the same baseline run).

| Approach | Config | NDCG@10 | Answer score [95% CI] | vs baseline [95% CI] | Correctness | Grounding | Completeness |
|---|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.402 | 0.687 [0.622, 0.751] |  | 2.568 | 2.792 | 2.880 |
| Rerank, MiniLM-L6, pool 20 | `rerank-minilm-p20` | 0.393 | 0.646 [0.577, 0.714] | -0.041 [-0.095, 0.011] | 2.432 | 2.640 | 2.680 |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | `raptor-tree-gpt5-evidence` | 0.400 | 0.690 [0.625, 0.755] | +0.003 [-0.045, 0.051] | 2.568 | 2.824 | 2.888 |
| Rerank, bge-reranker-v2-m3, pool 20 | `rerank-bge-m3-p20` | 0.460 | 0.661 [0.598, 0.719] | -0.026 [-0.082, 0.033] | 2.464 | 2.672 | 2.792 |
| Rerank, mxbai-rerank-base-v2, pool 20 | `rerank-mxbai-v2-p20` | 0.465 | 0.678 [0.613, 0.741] | -0.009 [-0.063, 0.045] | 2.584 | 2.712 | 2.840 |

- **No change improves the answer score.** Every "vs baseline" interval contains 0.
- **The two strong rerankers do improve retrieval** (NDCG@10 +0.058 and +0.063, both significant; section 3), but the
  answers do not follow. See [Why](#7-why-the-answers-do-not-move).
- **RAPTOR summaries reach the prompt rarely.** A summary was among the top 5 for only 10 of the 125 questions, so
  115 prompts were identical to the baseline's.

Retrieval of the first three rows on practice (`final-practice`, retrieval only):

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] |
|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.428 | 0.485 | 0.493 | 0.425 [0.385, 0.462] |
| Rerank, MiniLM-L6, pool 20 | `rerank-minilm-p20` | 0.418 | 0.509 | 0.470 | 0.418 [0.380, 0.454] |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | `raptor-tree-gpt5-evidence` | 0.428 | 0.485 | 0.492 | 0.425 [0.385, 0.462] |

## 2. Retrieval methods without an LLM at query time

`retrievers-test` and `retrievers-practice` (retrieval only). How each method works: [retrievers](retrievers.md) and
[pipelines](pipelines.md).

Test:

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.382 | 0.479 | 0.461 | 0.402 [0.338, 0.470] | 113 |
| BM25 | `bm25` | 0.216 | 0.281 | 0.267 | 0.227 [0.171, 0.285] | 2 |
| Topic routing, 2 topics | `topics-k256-m2` | 0.302 | 0.340 | 0.342 | 0.289 [0.229, 0.354] | 29 |
| Topic routing, 4 topics | `topics-k256-m4` | 0.328 | 0.420 | 0.383 | 0.342 [0.281, 0.409] | 30 |
| Topic routing, 8 topics | `topics-k256-m8` | 0.363 | 0.458 | 0.432 | 0.381 [0.318, 0.453] | 31 |
| TOC, centroid route | `toc-centroid-m4` | 0.299 | 0.376 | 0.336 | 0.302 [0.237, 0.367] | 30 |
| RAPTOR-lite, expand | `raptor-lite-expand` | 0.382 | 0.469 | 0.464 | 0.399 [0.336, 0.466] | 115 |
| RAPTOR-lite, summaries as evidence | `raptor-lite-evidence` | 0.382 | 0.469 | 0.458 | 0.398 [0.334, 0.466] | 115 |
| RAPTOR-tree, expand | `raptor-tree-expand` | 0.386 | 0.461 | 0.459 | 0.396 [0.333, 0.464] | 116 |
| RAPTOR-tree, summaries as evidence | `raptor-tree-evidence` | 0.382 | 0.469 | 0.459 | 0.398 [0.334, 0.466] | 115 |
| RAPTOR-tree (gpt-5-mini), expand | `raptor-tree-gpt5-expand` | 0.382 | 0.477 | 0.461 | 0.401 [0.336, 0.470] | 114 |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | `raptor-tree-gpt5-evidence` | 0.382 | 0.477 | 0.460 | 0.400 [0.336, 0.470] | 115 |
| RAPTOR-full, expand | `raptor-full-expand` | 0.364 | 0.437 | 0.449 | 0.383 [0.317, 0.454] | 121 |
| RAPTOR-full, summaries as evidence | `raptor-full-evidence` | 0.344 | 0.433 | 0.397 | 0.348 [0.289, 0.413] | 121 |

Practice:

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.428 | 0.485 | 0.493 | 0.425 [0.385, 0.462] | 109 |
| BM25 | `bm25` | 0.255 | 0.297 | 0.282 | 0.246 [0.209, 0.280] | 2 |
| Topic routing, 2 topics | `topics-k256-m2` | 0.282 | 0.328 | 0.385 | 0.302 [0.263, 0.341] | 20 |
| Topic routing, 4 topics | `topics-k256-m4` | 0.327 | 0.382 | 0.416 | 0.342 [0.301, 0.382] | 23 |
| Topic routing, 8 topics | `topics-k256-m8` | 0.375 | 0.435 | 0.457 | 0.385 [0.345, 0.423] | 25 |
| TOC, centroid route | `toc-centroid-m4` | 0.300 | 0.355 | 0.381 | 0.313 [0.276, 0.353] | 32 |
| RAPTOR-lite, expand | `raptor-lite-expand` | 0.428 | 0.485 | 0.492 | 0.425 [0.385, 0.462] | 114 |
| RAPTOR-lite, summaries as evidence | `raptor-lite-evidence` | 0.428 | 0.484 | 0.491 | 0.423 [0.384, 0.460] | 113 |
| RAPTOR-tree, expand | `raptor-tree-expand` | 0.428 | 0.485 | 0.490 | 0.424 [0.384, 0.461] | 114 |
| RAPTOR-tree, summaries as evidence | `raptor-tree-evidence` | 0.428 | 0.485 | 0.490 | 0.423 [0.384, 0.460] | 113 |
| RAPTOR-tree (gpt-5-mini), expand | `raptor-tree-gpt5-expand` | 0.427 | 0.484 | 0.492 | 0.424 [0.385, 0.461] | 108 |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | `raptor-tree-gpt5-evidence` | 0.428 | 0.485 | 0.492 | 0.425 [0.385, 0.462] | 115 |
| RAPTOR-full, expand | `raptor-full-expand` | 0.392 | 0.461 | 0.467 | 0.399 [0.362, 0.437] | 121 |
| RAPTOR-full, summaries as evidence | `raptor-full-evidence` | 0.389 | 0.452 | 0.442 | 0.382 [0.346, 0.419] | 120 |

- Dense search is the best of these. Every RAPTOR variant except RAPTOR-full is within 0.006 NDCG@10 of it (0.002 on practice), because a
  summary hit is expanded into records that dense search had already ranked.
- RAPTOR-full (UMAP and Gaussian-mixture clustering, five summary layers) is the only RAPTOR variant below the baseline
  on every metric. Its summaries crowd the ranked list (about 23 summary hits per question), and 95 % of them stop at
  the 100-token cap mid-sentence.
- Topic routing gets better with more topics probed and costs a quarter of dense search's time. BM25 is far below
  everything else.

## 3. Rerankers

The dense top 20 (or 50, 100) re-scored by a local cross-encoder; the top 5 go into the prompt.

**Choosing a setting on practice** (`rerank-practice`, retrieval only; the rule was the highest NDCG@10):

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.428 | 0.485 | 0.493 | 0.425 [0.385, 0.462] | 114 |
| Rerank, MiniLM-L6, pool 20 | `rerank-minilm-p20` | 0.418 | 0.509 | 0.470 | 0.418 [0.380, 0.454] | 1,644 |
| Rerank, MiniLM-L6, pool 50 | `rerank-minilm-p50` | 0.400 | 0.494 | 0.457 | 0.404 [0.368, 0.439] | 3,048 |
| Rerank, MiniLM-L6, pool 100 | `rerank-minilm-p100` | 0.391 | 0.486 | 0.455 | 0.401 [0.363, 0.436] | 4,498 |
| Rerank, bge-reranker-base, pool 20 | `rerank-bge-p20` | 0.375 | 0.485 | 0.414 | 0.376 [0.340, 0.411] | 9,700 |
| Rerank, bge-reranker-base, pool 50 | `rerank-bge-p50` | 0.344 | 0.436 | 0.385 | 0.341 [0.306, 0.376] | 18,358 |

None of the first-generation rerankers beat the baseline; MiniLM with pool 20 was the best and went into the
answer comparison of section 1.

**Stronger models on test** (`rerank-models-test`, retrieval only). Latency here is end to end on all 14 laptop cores,
mostly the reranker:

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.382 | 0.479 | 0.461 | 0.402 [0.338, 0.470] | 117 |
| Rerank, MiniLM-L6, pool 20 | `rerank-minilm-p20` | 0.381 | 0.495 | 0.442 | 0.393 [0.333, 0.460] | 1,571 |
| Rerank, bge-reranker-v2-m3, pool 20 | `rerank-bge-m3-p20` | 0.467 | 0.529 | 0.529 | 0.460 [0.399, 0.525] | 42,117 |
| Rerank, mxbai-rerank-base-v2, pool 20 | `rerank-mxbai-v2-p20` | 0.470 | 0.537 | 0.532 | 0.465 [0.402, 0.534] | 61,897 |

| Approach | NDCG@10 vs baseline [95% CI] | Relevant records out / in of the top 5 |
|---|---|---|
| Rerank, MiniLM-L6, pool 20 | -0.008 [-0.044, 0.025] | 18 / 19 |
| Rerank, bge-reranker-v2-m3, pool 20 | +0.058 [0.023, 0.098] | 16 / 38 |
| Rerank, mxbai-rerank-base-v2, pool 20 | +0.063 [0.027, 0.101] | 10 / 34 |

bge-reranker-v2-m3 and mxbai-rerank-base-v2 beat the baseline on every metric and tie with each other. So reranking
works on this benchmark; MiniLM failed because of the model (small, trained only on MS MARCO web passages), not the
idea.

**MiniLM fine-tuned on practice** (`rerank-ft-test`, retrieval only; recipe in [retrievers](retrievers.md#fine-tuned-reranker)):

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.382 | 0.479 | 0.461 | 0.402 [0.338, 0.470] | 163 |
| Rerank, MiniLM-L6, pool 20 | `rerank-minilm-p20` | 0.381 | 0.495 | 0.442 | 0.393 [0.333, 0.460] | 1,316 |
| Rerank, MiniLM-L6 fine-tuned on practice, pool 20 | `rerank-minilm-ft-p20` | 0.361 | 0.491 | 0.422 | 0.371 [0.315, 0.428] | 1,394 |

Fine-tuning made MiniLM worse (NDCG@10 -0.031 [-0.072, 0.008] against the baseline). It overfits: NDCG@10 0.466 on
its own training questions, 0.407 on held-out practice folds, 0.371 on test. With 350 questions and no relevant record
shared between questions, it learns which records are labelled rather than what relevance looks like.

## 4. Encoder policies: querying the enc_b index

The course ships a second index, enc_b, built with bge-small. These rows query it with our gte-small encoder under
each policy (`encoders-test`, `encoders-practice`, retrieval only).

Test:

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.382 | 0.479 | 0.461 | 0.402 [0.338, 0.470] | 112 |
| enc_b index, gte-small query as is | `enc-b-asis` | 0.342 | 0.437 | 0.398 | 0.349 [0.289, 0.416] | 115 |
| enc_b index, query re-encoded with bge-small | `enc-b-reencode` | 0.396 | 0.491 | 0.482 | 0.413 [0.352, 0.477] | 102 |
| enc_b index, re-encoded with bge's query prefix | `enc-b-reencode-prefix` | 0.418 | 0.526 | 0.490 | 0.434 [0.371, 0.502] | 106 |
| enc_b index, query mapped (least squares) | `enc-b-translate-lstsq` | 0.396 | 0.473 | 0.458 | 0.389 [0.332, 0.454] | 115 |
| enc_b index, query mapped (Procrustes) | `enc-b-translate-procrustes` | 0.355 | 0.425 | 0.434 | 0.363 [0.299, 0.431] | 117 |

Practice:

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.428 | 0.485 | 0.493 | 0.425 [0.385, 0.462] | 108 |
| enc_b index, gte-small query as is | `enc-b-asis` | 0.363 | 0.465 | 0.401 | 0.361 [0.324, 0.399] | 113 |
| enc_b index, query re-encoded with bge-small | `enc-b-reencode` | 0.428 | 0.505 | 0.495 | 0.432 [0.393, 0.471] | 101 |
| enc_b index, re-encoded with bge's query prefix | `enc-b-reencode-prefix` | 0.441 | 0.516 | 0.492 | 0.437 [0.398, 0.476] | 104 |
| enc_b index, query mapped (least squares) | `enc-b-translate-lstsq` | 0.389 | 0.467 | 0.470 | 0.400 [0.361, 0.439] | 113 |
| enc_b index, query mapped (Procrustes) | `enc-b-translate-procrustes` | 0.342 | 0.422 | 0.419 | 0.353 [0.316, 0.392] | 113 |

Re-encoding the question with enc_b's own encoder plus bge's query prefix is the best policy on both sets, slightly
above the baseline (intervals overlap). Using the wrong encoder as is, or a Procrustes map, is clearly worse.

## 5. ASR: whisper-1 against our course ASR model

The same two configurations on test, with transcripts from whisper-1 or from the course's GLM-ASR model running on
our CUDA kernels (`courseasr-test`; `asr-compare.json` holds the paired analysis).

| Approach | ASR | NDCG@10 [95% CI] | Answer score [95% CI] |
|---|---|---|---|
| Dense search (baseline) | whisper-1 | 0.402 [0.338, 0.470] | 0.687 [0.622, 0.751] |
| Dense search (baseline) | course ASR | 0.398 [0.338, 0.468] | 0.689 [0.628, 0.749] |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | whisper-1 | 0.400 [0.336, 0.470] | 0.690 [0.625, 0.755] |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | course ASR | 0.398 [0.338, 0.467] | 0.663 [0.597, 0.727] |

The two systems give word-identical transcripts for 100 of 125 questions (2.8 % of words differ). Retrieval is the
same within noise (NDCG@10 -0.003 [-0.017, 0.008] for the baseline). The course ASR takes a median 1.45 s per question on
the GPU, after a one-time model load of 13 minutes.

## 6. Methods measured on practice-20 only

Methods that call an LLM at query time (grep, LLM-routed TOC) and the context builders were only run on the
20-question sample. Read their intervals before comparing.

Retrieval (`retrievers-practice-20`, retrieval only; the gpt-5-mini rows ran while another job loaded the CPU, so
their latency is not comparable):

| Approach | Config | Recall@5 | Recall@10 | MRR@10 | NDCG@10 [95% CI] | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.404 | 0.475 | 0.479 | 0.386 [0.244, 0.525] | 108 |
| BM25 | `bm25` | 0.167 | 0.183 | 0.246 | 0.160 [0.072, 0.254] | 2 |
| Grep, static | `grep-static` | 0.046 | 0.133 | 0.111 | 0.074 [0.028, 0.131] | 1,405 |
| Grep, agentic | `grep-agentic` | 0.067 | 0.067 | 0.175 | 0.087 [0.019, 0.179] | 5,572 |
| Grep, feedback | `grep-feedback` | 0.158 | 0.158 | 0.275 | 0.161 [0.061, 0.289] | 7,728 |
| Topic routing, 2 topics | `topics-k256-m2` | 0.279 | 0.325 | 0.333 | 0.265 [0.136, 0.403] | 27 |
| Topic routing, 4 topics | `topics-k256-m4` | 0.396 | 0.433 | 0.442 | 0.350 [0.219, 0.478] | 20 |
| Topic routing, 8 topics | `topics-k256-m8` | 0.404 | 0.450 | 0.479 | 0.376 [0.239, 0.509] | 20 |
| TOC, centroid route | `toc-centroid-m4` | 0.346 | 0.383 | 0.379 | 0.306 [0.177, 0.441] | 19 |
| TOC, LLM route | `toc-llm-m4` | 0.212 | 0.275 | 0.317 | 0.235 [0.107, 0.381] | 1,125 |
| TOC, LLM route with topic names | `toc-llm-topics-m4` | 0.212 | 0.275 | 0.312 | 0.233 [0.105, 0.380] | 1,525 |
| TOC, one LLM call over all topics | `toc-llm-flat-m4` | 0.263 | 0.271 | 0.317 | 0.240 [0.109, 0.381] | 2,373 |
| RAPTOR-lite, expand | `raptor-lite-expand` | 0.404 | 0.450 | 0.479 | 0.376 [0.239, 0.509] | 109 |
| RAPTOR-tree, expand | `raptor-tree-expand` | 0.404 | 0.475 | 0.479 | 0.386 [0.244, 0.525] | 102 |
| TOC (gpt-5-mini labels), centroid route | `toc-centroid-m4-gpt5labels` | 0.346 | 0.383 | 0.379 | 0.306 [0.177, 0.441] | 872 |
| TOC (gpt-5-mini labels), LLM route | `toc-llm-m4-gpt5labels` | 0.212 | 0.275 | 0.350 | 0.251 [0.112, 0.395] | 4,010 |
| TOC (gpt-5-mini labels), LLM route with topic names | `toc-llm-topics-m4-gpt5labels` | 0.287 | 0.333 | 0.392 | 0.288 [0.153, 0.420] | 4,528 |
| TOC (gpt-5-mini labels), one LLM call over all topics | `toc-llm-flat-m4-gpt5labels` | 0.263 | 0.300 | 0.333 | 0.250 [0.130, 0.373] | 6,543 |
| RAPTOR-tree (gpt-5-mini), expand | `raptor-tree-gpt5-expand` | 0.404 | 0.475 | 0.479 | 0.387 [0.244, 0.531] | 1,036 |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | `raptor-tree-gpt5-evidence` | 0.404 | 0.475 | 0.479 | 0.385 [0.244, 0.524] | 945 |
| RAPTOR-full, expand | `raptor-full-expand` | 0.375 | 0.458 | 0.467 | 0.379 [0.232, 0.517] | 112 |

Context builders (`context-practice-20`, answers judged). Retrieval is identical, so only the prompt changes:

| Approach | Config | Answer score | Correctness | Grounding | Completeness | Tokens per question |
|---|---|---|---|---|---|---|
| Dense search (baseline) | `baseline` | 0.579 | 2.200 | 2.300 | 2.450 | 4,136 |
| Context: 800-token budget | `context-budgeted` | 0.650 | 2.400 | 2.650 | 2.750 | 3,372 |
| Context: drop near-duplicates, then budget | `context-dedupe` | 0.671 | 2.600 | 2.700 | 2.750 | 3,250 |

On 20 questions these gains are within the judge noise described below.

## 7. Why the answers do not move

- **The judge scores answers against the gold records, not the retrieved ones.** The course judge
  (`rag/src/evaluation/answer_judge.py`) gets the question, the answer and the records the qrels mark as relevant.
  Putting a relevant record in the prompt barely changes the score: the baseline's answer score is 0.681 on the 52 test
  questions with no relevant record in its top 5, against 0.691 on the 73 with one. Why is not measured; a likely
  reason is that Qwen answers these general personal-finance questions well from its own knowledge.
- **Generating and judging are noisy.** The RAPTOR row put exactly the baseline's prompt in front of the model for 115
  test questions, yet the answer score changed on 58 of them (standard deviation of the difference 0.26). Scores are
  close to all or nothing: 13-19 % of answers score 0 on everything and 42-46 % score 4 on everything.
- **Where the top 5 changed, answers moved the expected way, weakly.** Questions whose Recall@5 rose gained +0.016
  (bge, 32 questions) and +0.038 (mxbai, 26), neither significant. The 5 questions where mxbai lost a relevant record
  fell by -0.300 [-0.501, -0.067].
- **MiniLM promoted generic records.** On practice it pushed 61 relevant records out of the top 5 and pulled 50 in.
  In 10 cases read by hand it favoured records that repeat the question's words or give a general answer, over the
  thread's own answer that the labels mark.

## 8. Cost

| Approach | One-time cost | Added time per question |
|---|---|---|
| Dense search (baseline) | none | 0.1 s retrieval |
| Rerank, MiniLM-L6 | 88 MB download | about 1.2-1.6 s on CPU |
| Rerank, bge-reranker-v2-m3 | 2.2 GB download | median 44 s on CPU |
| Rerank, mxbai-rerank-base-v2 | 0.96 GB download | median 26 s on CPU |
| RAPTOR-tree (gpt-5-mini) | US$1.46, 439 summary calls | none |
| RAPTOR-full | US$1.11, 5,243 summary calls, 4.1 h | none |
| Course ASR instead of whisper-1 | a GPU; 13 min model load | median 1.45 s (whisper-1: 1.1 s, US$0.006 per minute) |

Answer generation dominates everything: about 45-90 s per question on the free Qwen model, depending on ELM load.
Rerankers would run in well under a second on a GPU; these times are the laptop CPU.
