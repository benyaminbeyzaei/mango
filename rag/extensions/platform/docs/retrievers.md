# Retrievers, context builders, encoder policies, ASR comparison

How each method works. Their numbers are in [results.md](results.md).

## Context builders

All three write the course prompt (`rag/src/context_builder.py`) around the evidence they keep.

- `simple`: the full text of the top k records (the course baseline).
- `budgeted` (`params.budget_tokens`, 800): records in rank order until the evidence reaches the
  budget; the record that crosses it is cut at the last sentence end that fits (dropped if none
  fits). The budget counts evidence only, not the question or the instructions. Tokens are a
  word-based estimate (words × 1.3), enough to compare prompt sizes; the real prompt tokens are in
  the trace's gateway cost.
- `dedupe` (`params.dedupe_threshold`, 0.9, and `params.budget_tokens`): drops a record whose enc_a
  cosine similarity to an already kept record is above the threshold (vectors from `stages.enc_a()`,
  not re-embedded; records without a vector, such as raptor-lite and raptor-tree summaries (see [pipelines.md](pipelines.md)), are kept), then the budget.

Counters per query: `context.records_used`, `context.tokens`, `context.truncated` (0/1),
`context.dropped_duplicates` (dedupe only). `configs/context-budgeted.yaml` and
`configs/context-dedupe.yaml` are the baseline with each builder; `configs/matrices/matrix-context-practice-20.yaml`
compares the three on practice-20.

## Encoder policies (query encoder vs corpus index)

enc_a was built with `thenlper/gte-small` (`rag/models/enc_a`), enc_b with `BAAI/bge-small-en-v1.5`.
Both are 384-d and normalised, but they are different spaces: for the same record the two vectors
have cosine 0.82 on average. When `query_encoder` does not belong to `corpus_index`,
`mismatch_policy` decides what happens:

- `error` (default): the config does not validate.
- `as-is`: use the query vector anyway (the "wrong space" control).
- `re-encode`: ignore `query_encoder` and encode the query with the index's own encoder.
  `params.bge_prefix: true` prepends bge's query instruction
  (`Represent this sentence for searching relevant passages: `); it is meant for bge-small only.
- `translate`: encode with `query_encoder`, multiply by a fitted 384x384 map into the index space
  (`params.map: lstsq | procrustes`, default lstsq), re-normalise.

Configs: `configs/enc-b-{asis,reencode,reencode-prefix,translate-lstsq,translate-procrustes}.yaml`,
all gte-small queries against enc_b; `configs/matrices/matrix-encoders-practice-20.yaml` runs them with the baseline.

### Fitting the map

```bash
mlsrag fit-map --from enc_a --to enc_b --method lstsq --exclude-dev       # ~4 s
mlsrag fit-map --from enc_a --to enc_b --method procrustes --exclude-dev  # ~1 s
```

The 24,000 corpus records have a vector in both spaces (same row order), so they are training
pairs: lstsq finds any linear W with `enc_a @ W ~ enc_b`; Procrustes restricts W to a rotation.
`--exclude-dev` leaves the records relevant to any test question (the course's `qrels_dev.tsv`, 325 records) out
of the fit, so a test evaluation never sees a map trained on its own answers; those rows are the
held-out set on which the sidecar reports the mean cosine between mapped enc_a and true enc_b
(lstsq 0.983, Procrustes 0.968, unmapped 0.819). Maps go to `rag/.cache/maps/<src>-to-<dst>-<method>.npy`
with a `.json` sidecar (fit time, rows, cosines); they are not committed, refit them in seconds.

## Comparing two ASR systems

```bash
mlsrag asr-compare --a rag/.cache/transcripts/whisper1-practice.jsonl \
    --b rag/.cache/transcripts/course-practice.jsonl --subset practice-20 [--config baseline]
```

Both files are transcript caches (`{query_id, transcript}` per line). It reports WER of `b` with
`a` as the reference (`jiwer`, after lowercasing and dropping punctuation), the share of queries
with identical transcripts, and the retrieval metrics of the config (retrieval only, no generation)
run on each cache. Output: `rag/extensions/results/asr-compare/<a>-vs-<b>-<subset>/summary.json`
(plus the two run folders). **Limitation:** there are no reference transcripts, so WER measures
disagreement between the two systems, not the accuracy of either; the retrieval metrics are the
only quality signal.

### ASR latency

`mlsrag transcribe` writes `seconds` (wall time of the transcription call) and `audio_s` (the WAV
duration) into every new cache line, and the one-off model load time into `<out>.meta.json`. Old
lines without timings stay valid. `--fresh` makes whisper-1 skip the gateway's cache read (it still
writes), so the time is a real API call; it is paid, so use it on practice-20 only. When both
caches carry timings for every query, `asr-compare` adds per side the mean, median and p95 seconds
per query and the real-time factor (total seconds / total audio seconds), `null` otherwise.

whisper-1 on practice-20, from the laptop with network included: mean 1.35 s, median 1.13 s, p95 2.53 s, real-time
factor 0.28, $0.009 (`rag/extensions/results/asr-latency/whisper1-practice-20.json`).

#### The course ASR on the cluster

What it needs: the `mls-asr` conda environment and the 4.5 GB model from `hw1-asr/setup_env.sh`
(run once on the head node, which has internet), the RAG venv with this package installed, and the
course data installed under `rag/data/` (`rag/README.md`, Steps 1 and 2). `ASR_CUDA=none` runs the
supplied PyTorch reference and needs no finished kernels; `ASR_CUDA=all` runs our hw1 kernels, so
`make test` must pass first and `make -C hw1-asr lib` must be built on the GPU node in use. No ELM
key or calls are needed: the course ASR runs locally on the GPU.

```bash
# head node (mlp.inf.ed.ac.uk), once: pull the branch and install this package into the RAG venv
cd ~/edin-mls-26-autumn && git fetch origin && git checkout main && git pull
source ~/mls-rag-env/bin/activate && python -m pip install -e rag/extensions/platform

# GPU shell (one GPU, exit when done)
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:30:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source ~/mls-rag-env/bin/activate
cd ~/edin-mls-26-autumn && nvidia-smi -L && make -C hw1-asr lib
export ASR_CUDA=all     # our kernels; ASR_CUDA=none for the PyTorch reference (use another --out)
mlsrag transcribe --asr course --subset practice-20 \
    --out rag/.cache/transcripts/course-practice-20-timed.jsonl
exit
```

Note the GPU from `nvidia-smi -L` with the numbers (speeds differ a lot between GPU types). Copy
both files back into the main checkout (worktrees link `rag/.cache` to it); from outside the
university network add `-J YOUR_UUN@ssh.inf.ed.ac.uk`:

```bash
scp 'YOUR_UUN@mlp.inf.ed.ac.uk:edin-mls-26-autumn/rag/.cache/transcripts/course-practice-20-timed.jsonl*' \
    .execution/edin-mls-26-autumn/rag/.cache/transcripts/
```

Then compare locally (no GPU, no ELM calls):
`mlsrag asr-compare --a rag/.cache/transcripts/whisper1-practice-timed.jsonl --b rag/.cache/transcripts/course-practice-20-timed.jsonl`.

## Lexical retrieval (BM25, grep)

Configs `configs/bm25.yaml`, `configs/grep-static.yaml`, `configs/grep-agentic.yaml`,
`configs/grep-feedback.yaml`.

- **BM25** (`retriever: bm25`, standard flow): `bm25s` over the record texts, tokens = lowercase
  `\w+` words minus English stop words. The index is pickled to `rag/.cache/bm25/<corpus hash>.pkl`;
  the first query's trace has `bm25.build_s` (build time, or load time when the pickle was
  cached) and `bm25.index_mb` (pickle size). Only records with a positive score are
  returned; a query that is empty after tokenising returns nothing.
- **Grep static** (`retriever: grep`, `flow: grep-static`): one LLM call (`params.grep_model`,
  default Llama-3.3: Qwen's hidden reasoning took ~70-100 s per call) turns the
  transcript into 3 to 8 plain patterns, a case-insensitive `re` scan over all 24,000 texts
  ranks records by patterns matched, then total matches, then shorter text.
  Invalid patterns and patterns that match the empty string are dropped and counted.
  `retriever: grep` is only valid with the grep flows.
- **Grep agentic** (`flow: grep-agentic`): the static step, then up to 2 refinement calls (the
  model sees the patterns and the top 10 hits, 200 characters each, and stops or gives new
  patterns), then one clean-up call that picks at most `k` ids from the top 20 hits (300
  characters each). The context gets only the kept ids (the top k if the reply was not a list);
  the ranking for the metrics is the kept ids, then the rest of the scan order. Cap: 3 scans per
  query (`MAX_ITERATIONS`), so at most 4 LLM calls (3.65 on average on practice-20).
- **Grep feedback** (`flow: grep-feedback`): relevance feedback that never drops what
  static grep found. The static step, then a keep call (`KEEP`: the top 20 hits, 300 characters
  each; the model returns the helpful ids, or `[]`); if nothing is kept the ranking is the static
  one (2 calls). Otherwise one expand call (`EXPAND`) writes new patterns from the kept records'
  text (at most 5, 300 characters each), a second scan drops ids already in the static top 20, and
  one rerank call (the `CLEANUP` prompt) picks at most `k` from the pool (kept ids, then new hits,
  20 records). Context: the reranked ids (the top k if the reply was unusable). Ranking: reranked,
  rest of the pool, rest of the static hits, rest of the new hits, cut at 100. At most 4 calls;
  `grep.kept` counts the kept ids, `artifacts.kept_ids` and `reranked_ids` hold them. On
  practice-20 with Llama-3.3 (`results/20261109-1426-grep-feedback-practice-20`) it kept
  something in 19 of 20 queries (a relevant record in 7), the expansion found no relevant record
  that static grep missed, and the gain over static (MRR@10 0.111 → 0.275, Recall@10 0.133 →
  0.158, Recall@100 unchanged at 0.217) comes from the rerank; ~7.7 s and 3.9 calls per query.

How costs are counted: the `retrieve` span of the grep flows covers every LLM call and scan, so
`latency_ms.retrieve` is comparable with the dense retriever; `grep.scan` spans time the scans
alone. Per-query counters in `queries.jsonl`: `grep.llm_calls`, `grep.iterations`,
`grep.patterns`, `grep.matched_records`, `grep.invalid_patterns`, `grep.parse_errors`, `grep.kept`; the
patterns per round are in `artifacts.patterns`. Tokens come from the gateway's `cost`, which counts
only new calls (a rerun from the cache costs 0). The prompts are fixed (`VERSION = "grep_v1"`,
passed to the gateway as `tag` so it enters the cache key); do not tune them on the test set.

## Cross-encoder rerank

Configs `configs/rerank-<minilm|bge>-p<20|50|100>.yaml`, `rerank-bge-m3-p20.yaml` (BAAI/bge-reranker-v2-m3) and
`rerank-mxbai-v2-p20.yaml` (mixedbread-ai/mxbai-rerank-base-v2). `retriever: rerank` wraps another retriever (`params.base`, default `exact`): it asks it for the
top `params.pool` records, scores each (transcript, record text) pair with a local
`sentence_transformers.CrossEncoder` (`params.model`, pinned `params.revision`) on CPU, and returns the pool in
the new order followed by the rest of the base ranking unchanged, so Recall@100 equals the base retriever's.
The call goes through `gateway.rerank` (timed, cached on model + revision + query + text hashes, skipped
cache read with `--fresh`). Pairs longer than 512 tokens are truncated; the trace counts them
(`rerank.truncated` of `rerank.pool`). The `rerank` span sits inside `retrieve`; `latency_ms.rerank` reports it
separately (`latency_ms.retrieve` includes it). Models are downloaded once to `rag/.cache/hf/`
(ms-marco-MiniLM-L-6-v2: 88 MB, bge-reranker-base: 1.1 GB, bge-reranker-v2-m3: 2.2 GB, mxbai-rerank-base-v2:
0.96 GB). Every reranker is loaded in float32: mxbai ships bfloat16 weights, which ran about 6 times slower on the CPU.

## Fine-tuned reranker

The MiniLM reranker (trained on MS MARCO) lowered NDCG@10 and MRR@10. The likely cause was its idea of relevance
(web passages that repeat the query) against this benchmark's labels (the answers in the original forum thread).
`mlsrag/finetune.py` fine-tunes the same model on the practice set's public labels; the test set is never read.

- **Pairs** (`prepare`, laptop): per practice question, the whisper-1 transcript as the query; every labelled
  relevant record (label 1, 793 in all) and the non-relevant records in its dense top 20 (label 0, the hard
  negatives the reranker must push down). Dense ranking: gte-small on enc_a, as the baseline. The file
  `jobs/finetune-data.json` (not in git) holds the transcripts, rankings, labels and the 5,182 record texts.
- **Training** (`train`, GPU): a plain PyTorch loop over the CrossEncoder's model: binary cross-entropy on its
  single logit, AdamW (weight decay 0.01), linear warm-up over the first 10 % of steps then linear decay,
  batch 16, pairs truncated at 512 tokens as at test time, seed 0.
- **Settings** (`cv`, GPU): 5-fold cross-validation on practice (folds fixed by seed 0, 70 questions held out
  each). Learning rates 2e-5 and 5e-5, evaluated after epochs 1, 2 and 3 on the held-out fold by reranking the
  dense top 20 and computing the course metrics; the dense baseline and the untuned model are scored on the
  same folds. Rule: the highest mean NDCG@10 over the 5 folds. Output: `results/finetune/cv.json`. The rule chose
  learning rate 5e-5 and 1 epoch (mean NDCG@10 0.407; untuned 0.418, dense 0.425).
- **Final model**: trained on all 350 practice questions with the chosen setting, saved to
  `rag/.cache/models/rerank-minilm-ft/` (not in git; `finetune.json` beside the weights records the recipe).
  Config `configs/rerank-minilm-ft-p20.yaml` (`params.model` is that folder; the gateway loads a folder path
  relative to the checkout root). Test measurement: `configs/matrices/matrix-rerank-ft-test.yaml`.

Rebuild:

```bash
python -m mlsrag.finetune prepare jobs/finetune-data.json            # laptop, seconds
.ai/bin/cluster.sh sync <run> && .ai/bin/cluster.sh submit <run> rag/extensions/platform/jobs/finetune.sbatch   # cv
.ai/bin/cluster.sh submit <run> rag/extensions/platform/jobs/finetune.sbatch --export=ALL,MODE=train,LR=<lr>,EPOCHS=<n>
# copy outputs/rerank-minilm-ft from the cluster run folder to rag/.cache/models/rerank-minilm-ft, then:
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-rerank-ft-test.yaml --no-generate --fresh --allow-full --yes
```

No relevant record is shared between practice and test (0 of 325 test records), so the model cannot memorise test
answers, only the labelling style.

## Fine-tuned reranker (plan 035)

Plan 032's MiniLM reranker (MS MARCO) lowered NDCG@10 and MRR@10. The likely cause was its idea of relevance
(web passages that repeat the query) against this benchmark's labels (the answers in the original forum thread).
`mlsrag/finetune.py` fine-tunes the same model on the practice split's public labels; dev is never read.

- **Pairs** (`prepare`, laptop): per practice question, the whisper-1 transcript as the query; every labelled
  relevant record (label 1, 793 in all) and the non-relevant records in its dense top 20 (label 0, the hard
  negatives the reranker must push down). Dense ranking: gte-small on enc_a, as the baseline. The file
  `jobs/finetune-data.json` (not in git) holds the transcripts, rankings, labels and the 5,182 record texts.
- **Training** (`train`, GPU): a plain PyTorch loop over the CrossEncoder's model: binary cross-entropy on its
  single logit, AdamW (weight decay 0.01), linear warm-up over the first 10 % of steps then linear decay,
  batch 16, pairs truncated at 512 tokens as at test time, seed 0.
- **Settings** (`cv`, GPU): 5-fold cross-validation on practice (folds fixed by seed 0, 70 questions held out
  each). Learning rates 2e-5 and 5e-5, evaluated after epochs 1, 2 and 3 on the held-out fold by reranking the
  dense top 20 and computing the course metrics; the dense baseline and the untuned model are scored on the
  same folds. Rule: the highest mean NDCG@10 over the 5 folds. Output: `results/finetune/cv.json`.
- **Final model**: trained on all 350 practice questions with the chosen setting, saved to
  `rag/.cache/models/rerank-minilm-ft/` (not in git; `finetune.json` beside the weights records the recipe).
  Config `configs/rerank-minilm-ft-p20.yaml` (`params.model` is that folder; the gateway loads a folder path
  relative to the checkout root). Dev measurement: `configs/matrices/matrix-rerank-ft-dev-all.yaml`.

Rebuild:

```bash
python -m mlsrag.finetune prepare jobs/finetune-data.json            # laptop, seconds
.ai/bin/cluster.sh sync <run> && .ai/bin/cluster.sh submit <run> rag/extensions/platform/jobs/finetune.sbatch   # cv
.ai/bin/cluster.sh submit <run> rag/extensions/platform/jobs/finetune.sbatch --export=ALL,MODE=train,LR=<lr>,EPOCHS=<n>
# copy outputs/rerank-minilm-ft from the cluster run folder to rag/.cache/models/rerank-minilm-ft, then:
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-rerank-ft-dev-all.yaml --no-generate --fresh --allow-full --yes
```

No relevant record is shared between practice and dev (0 of 325 dev records), so the model cannot memorise dev
answers, only the labelling style.

Result (2026-11-16). Cross-validation on practice (job 3723683, one H200, 15 min; mean NDCG@10 over the 5
held-out folds): dense baseline 0.425, untuned MiniLM 0.418, fine-tuned 0.389-0.407 for every setting; the rule
chose lr 5e-5, 1 epoch (0.407). Final model: job 3723717 (H200, 74 s of training on 7,354 pairs). Dev-all,
retrieval only (`results/matrices/rerank-ft-dev-all/`, `analysis.json`; CPU, 10 cores, nice 19):

| Row | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | NDCG@10 minus baseline [95 % CI] | Relevant out / in of top 5 |
|---|---|---|---|---|---|---|
| baseline | 0.382 | 0.479 | 0.461 | 0.402 | | |
| MiniLM (MS MARCO) | 0.381 | 0.495 | 0.442 | 0.393 | -0.008 [-0.044, 0.025] | 18 / 19 |
| MiniLM fine-tuned on practice | 0.361 | 0.491 | 0.422 | 0.371 | -0.031 [-0.072, 0.008] | 25 / 19 |

Fine-tuning made the reranker worse, not better, so no answer run followed. Why: the model fits its own
training questions (NDCG@10 0.466 on practice-all, against 0.425 dense and 0.418 untuned) but not new ones
(0.407 held out, 0.371 on dev): it overfits. With 350 questions and no relevant record shared between questions
or splits, it learns which particular records are labelled rather than a transferable notion of relevance.
Label noise is not the obvious cause: no hard negative is a near-copy of a positive (0 of 6,561 above enc_a cosine
0.95; 8 % above 0.9). Pool and labels are also partly misaligned: 42 % of the practice positives lie outside
the dense top 20, so the reranker never sees them at test time.

## Stronger rerankers (plan 036)

Same setting as plan 032 (whisper-1 transcripts, gte-small on enc_a, dense top 20 re-scored, top 5 into the
prompt, laptop CPU), only the model changes: `BAAI/bge-reranker-v2-m3` (568M, XLM-RoBERTa large) and
`mixedbread-ai/mxbai-rerank-base-v2` (0.5B, Qwen2.5), configs `rerank-bge-m3-p20` and `rerank-mxbai-v2-p20`.
The gateway loads every reranker in float32: mxbai ships bfloat16 weights, which ran about 6 times slower on this CPU.

Retrieval on dev-all (`matrix-rerank-models-dev-all`, `--no-generate --fresh --allow-full`; paired 95 %
bootstrap against the baseline in `results/matrices/rerank-models-dev-all/analysis.json`):

| reranker | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | NDCG@10 minus baseline | relevant out / in of the top 5 |
|---|---|---|---|---|---|---|
| none (baseline) | 0.382 | 0.479 | 0.461 | 0.402 | | |
| MiniLM (MS MARCO) | 0.381 | 0.495 | 0.442 | 0.393 | -0.008 [-0.044, 0.025] | 18 / 19 |
| bge-reranker-v2-m3 | 0.467 | 0.529 | 0.529 | 0.460 | +0.058 [0.023, 0.098] | 16 / 38 |
| mxbai-rerank-base-v2 | 0.470 | 0.537 | 0.532 | 0.465 | +0.063 [0.027, 0.101] | 10 / 34 |

Both beat the baseline on every retrieval metric (all intervals above zero); they tie with each other. So
reranking can work on this benchmark: MiniLM's failure was the model (small, MS MARCO only), not the idea.
Cost on the laptop CPU (all 14 cores, not pinned as in plan 032): median rerank time per question
44 s (bge) and 26 s (mxbai) in the end-to-end runs, against 1.5 s (mean) for MiniLM. Answer generation
time varies with ELM load (median 74 s for the baseline on 11 Nov, 44 s on 17 Nov), so end-to-end latency is not
comparable across days.

End to end on dev-all (`matrix-final-{bge,mxbai}-dev-all`, `--fresh`, free Qwen3.5 answers and the same judge),
compared with the plan 032 baseline run; `python -m mlsrag.rerank_answers` writes
`results/matrices/final-rerank-dev-all/analysis.json`:

| reranker | answer_norm | minus baseline | correctness | grounding | completeness |
|---|---|---|---|---|---|
| none (baseline) | 0.687 | | 2.57 | 2.79 | 2.88 |
| bge-reranker-v2-m3 | 0.661 | -0.026 [-0.082, 0.033] | 2.46 | 2.67 | 2.79 |
| mxbai-rerank-base-v2 | 0.678 | -0.009 [-0.063, 0.045] | 2.58 | 2.71 | 2.84 |

Better retrieval does not show up in the answer scores. Two measured reasons:

- The answer score barely depends on retrieval. The judge grades the answer against the evidence it was given,
  so an answer that says "the evidence does not cover this" can score full marks. Baseline: answer_norm 0.681
  on the 52 questions with no relevant record in the top 5, against 0.691 on the 73 with one.
- The answer score is noisy. The plan 032 RAPTOR row put the same top 5 as the baseline into the prompt for
  115 questions; regenerating and re-judging changed the score on 58 of them (standard deviation of the per-question
  difference 0.26). That noise is of the same size as any retrieval effect here.

Where the top 5 did change, the answers moved the expected way, but weakly: questions whose Recall@5 rose gained
+0.016 (bge, 32 questions) and +0.038 (mxbai, 26 questions), neither significant; the 5 questions where mxbai lost
a relevant record fell by -0.300 [-0.501, -0.067].

## Adding a retriever or a flow

A new retriever is a class with `search(query_vec, k)` (with `uses_question = True` it also gets
`question=<transcript>`); add `name: factory(config, gateway, data)` to `RETRIEVERS` in `mlsrag/stages.py`
(context builders: `CONTEXT_BUILDERS`) and set `retriever: <name>` in a config. A new flow is a module in
`mlsrag/flows/` with a `run_query(query_id, audio_path, stages, config, trace)` that fills
`trace.artifacts` (at least `ranked_ids`) inside spans; register it in `FLOWS` and its extra ELM calls per query in `FLOW_CALLS_PER_QUERY` (`evaluate.py`). All model
calls go through the gateway so they are cached and costed; options go in `config.params`.
