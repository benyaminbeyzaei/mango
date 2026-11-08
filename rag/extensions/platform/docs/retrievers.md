# Retrievers, context builders, encoder policies, ASR comparison

Part of the [mlsrag README](../README.md).

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
  not re-embedded; records without a vector, such as raptor summaries, are kept), then the budget.

Counters per query: `context.records_used`, `context.tokens`, `context.truncated` (0/1),
`context.dropped_duplicates` (dedupe only). `configs/context-budgeted.yaml` and
`configs/context-dedupe.yaml` are the baseline with each builder; `configs/matrices/matrix-context-smoke.yaml`
compares the three on the smoke subset.

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
all gte-small queries against enc_b; `configs/matrices/matrix-encoders-smoke.yaml` runs them with the baseline.

### Fitting the map

```bash
mlsrag fit-map --from enc_a --to enc_b --method lstsq --exclude-dev       # ~4 s
mlsrag fit-map --from enc_a --to enc_b --method procrustes --exclude-dev  # ~1 s
```

The 24,000 corpus records have a vector in both spaces (same row order), so they are training
pairs: lstsq finds any linear W with `enc_a @ W ~ enc_b`; Procrustes restricts W to a rotation.
`--exclude-dev` leaves the records relevant to any dev query (`qrels_dev.tsv`, 325 records) out
of the fit, so a dev evaluation never sees a map trained on its own answers; those rows are the
held-out set on which the sidecar reports the mean cosine between mapped enc_a and true enc_b
(lstsq 0.983, Procrustes 0.968, unmapped 0.819). Maps go to `rag/.cache/maps/<src>-to-<dst>-<method>.npy`
with a `.json` sidecar (fit time, rows, cosines); they are not committed, refit them in seconds.

## Comparing two ASR systems

```bash
mlsrag asr-compare --a rag/.cache/transcripts/whisper1-practice.jsonl \
    --b rag/.cache/transcripts/course-practice.jsonl --subset practice-smoke20 [--config baseline]
```

Both files are transcript caches (`{query_id, transcript}` per line). It reports WER of `b` with
`a` as the reference (`jiwer`, after lowercasing and dropping punctuation), the share of queries
with identical transcripts, and the retrieval metrics of the config (retrieval only, no generation)
run on each cache. Output: `rag/extensions/results/asr-compare/<a>-vs-<b>-<subset>/summary.json`
(plus the two run folders). **Limitation:** there are no reference transcripts, so WER measures
disagreement between the two systems, not the accuracy of either; the retrieval metrics are the
only quality signal.

## Lexical retrieval (BM25, grep)

Configs `configs/bm25.yaml`, `configs/grep-static.yaml`, `configs/grep-agentic.yaml`.

- **BM25** (`retriever: bm25`, standard flow): `bm25s` over the record texts, tokens = lowercase
  `\w+` words minus English stop words. The index is pickled to `rag/.cache/bm25/<corpus hash>.pkl`;
  the first query's trace has `bm25.build_s` (build time, or load time when the pickle was
  cached) and `bm25.index_mb` (pickle size). Only records with a positive score are
  returned; a query that is empty after tokenising returns nothing.
- **Grep static** (`retriever: grep`, `flow: grep-static`): one LLM call (`params.grep_model`,
  default Qwen) turns the transcript into 3 to 8 plain patterns, a case-insensitive `re` scan over
  all 24,000 texts ranks records by patterns matched, then total matches, then shorter text.
  Invalid patterns and patterns that match the empty string are dropped and counted.
  `retriever: grep` is only valid with the two grep flows.
- **Grep agentic** (`flow: grep-agentic`): the static step, then up to 2 refinement calls (the
  model sees the patterns and the top 10 hits, 200 characters each, and stops or gives new
  patterns), then one clean-up call that picks at most `k` ids from the top 20 hits (300
  characters each). The context gets only the kept ids (the top k if the reply was not a list);
  the ranking for the metrics is the kept ids, then the rest of the scan order. Cap: 3 scans per
  query (`MAX_ITERATIONS`), so at most 4 LLM calls (the smoke run averaged 3.65).

How costs are counted: the `retrieve` span of the grep flows covers every LLM call and scan, so
`latency_ms.retrieve` is comparable with the dense retriever; `grep.scan` spans time the scans
alone. Per-query counters in `queries.jsonl`: `grep.llm_calls`, `grep.iterations`,
`grep.patterns`, `grep.matched_records`, `grep.invalid_patterns`, `grep.parse_errors`; the
patterns per round are in `artifacts.patterns`. Tokens come from the gateway's `cost`, which counts
only new calls (a rerun from the cache costs 0). The prompts are fixed (`VERSION = "grep_v1"`,
passed to the gateway as `tag` so it enters the cache key); do not tune them on the dev split.

## Adding a retriever or a flow

A new retriever is a class with `search(query_vec, k)` (with `uses_question = True` it also gets
`question=<transcript>`); add `name: factory(config, gateway, data)` to `RETRIEVERS` in `mlsrag/stages.py`
(context builders: `CONTEXT_BUILDERS`) and set `retriever: <name>` in a config. A new flow is a module in
`mlsrag/flows/` with a `run_query(query_id, audio_path, stages, config, trace)` that fills
`trace.artifacts` (at least `ranked_ids`) inside spans; register it in `FLOWS` and its extra ELM calls per query in `FLOW_CALLS_PER_QUERY` (`evaluate.py`). All model
calls go through the gateway so they are cached and costed; options go in `config.params`.
