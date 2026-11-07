# mlsrag: the experiment platform

Every RAG experiment runs through this package. It wraps the course code in `rag/src/` (never
modified), traces each query, sends every model call through one cached, logged, costed gateway.

Install (from the checkout root, inside a venv): `uv pip install -e "rag/extensions/platform[dev]"`.
Tests: `pytest rag/extensions/platform` (fast); `pytest -m slow rag/extensions/platform` (needs the data and the transcript cache).

## Layout

- `mlsrag/course.py`: puts `rag/` on `sys.path`, re-exports the course classes, and `paths`.
- `mlsrag/config.py`: `RunConfig`, `load_config(path or preset)`, preset `baseline`.
- `mlsrag/trace.py`: `Trace` (spans, counters, cost, resources, artifacts), one per query.
- `mlsrag/gateway.py`: `ModelGateway.chat / embed / transcribe`. Disk cache `rag/.cache/gateway/`, log `rag/.cache/gateway.log.jsonl`, at most 4 concurrent ELM requests, 3 retries on 429, 5xx and timeouts. The key is read from `rag/.env`.
- `mlsrag/stages.py`: the five interfaces and their adapters; `build_stages(config, gateway, data)`.
- `mlsrag/flows/`: `standard` = asr → encode → retrieve (100 ranked ids) → context (top k) → generate.
- `mlsrag/results.py`: the `metrics.json` schema (pydantic), `load_run`, `list_runs`, `compare`.
- `mlsrag/evaluate.py`: answer judging through the course evaluator, cost estimates.
- `mlsrag/matrix.py`: many configs on one subset, one summary table.
- `mlsrag/runner.py`, `mlsrag/cli.py`: the commands below. `subsets/`: fixed query-id lists (seed 0):
  `practice-smoke20`, `practice-20pct` (70), `dev-20pct` (25), `practice-all`, `dev-all`.

## The five interfaces

`ASR.transcribe(audio_path) -> str` (`CachedASR`, `WhisperASR`, `CourseASR`),
`Encoder.encode(text) -> vector` (`GatewayEncoder`: gte-small or bge-small),
`Retriever.search(vec, k) -> [(record_id, score)]` (`ExactRetriever`: enc_a or enc_b),
`ContextBuilder.build(question, record_ids) -> prompt` (`SimpleContext`),
`Generator.generate(prompt) -> answer` (`GatewayGenerator`, the course system prompt, temperature 0.1).

## RunConfig fields

`name`; `asr` (cached | whisper1 | course); `asr_cache` (JSONL, relative to the checkout root);
`query_encoder` (gte-small | bge-small); `corpus_index` (enc_a | enc_b); `mismatch_policy`
(error | as-is: gte-small belongs to enc_a, bge-small to enc_b); `retriever` (exact); `k` (5);
`context_builder` (simple); `generator_model`; `judge_model`; `flow` (standard); `params` (free dict).
A YAML/JSON file with these keys works anywhere a preset name does.

## Commands (run from the checkout root)

```bash
mlsrag make-subsets          # regenerate subsets/*.txt (identical output every time)
mlsrag transcribe --asr whisper1 --subset practice-smoke20 \
    --out rag/.cache/transcripts/whisper1-practice.jsonl     # one-off, ~1.6 audio min, resumable
mlsrag run --config baseline --subset practice-smoke20 --no-generate   # no generator; free with cached ASR
mlsrag run --config baseline --subset practice-smoke20                 # + 20 Qwen calls
```

## Results

`rag/extensions/results/<YYYYMMDD-HHMM>-<config name>-<subset>/`: `config.json`;
`queries.jsonl` (one trace per line: spans, counters, cost, resources, artifacts
`transcript, ranked_ids, scores, prompt, answer`, plus per-query `metrics`); `metrics.json`
(schema below); after `mlsrag judge`, also `answers.jsonl` (the course candidate format) and
`judge/` (the course evaluator's output: per-query scores, summary, judge cache).
The course `.gitignore` ignores `results/`: commit a run with `git add -f`.

### metrics.json

Checked by `mlsrag.results.Metrics` on every write and load (`tests/test_results.py` checks every saved run).

```
retrieval   {Recall@5, Recall@10, Recall@20, Recall@100, MRR@10, NDCG@10}      means over queries
answers     {correctness, grounding, completeness (0-4), normalized (sum/12),
             judged_queries, judge_model, repeats, judge_calls}                 null until judged
cost        {elm_calls, prompt_tokens, completion_tokens, audio_seconds, est_usd}  totals; cached calls are free
latency_ms  {asr, encode, retrieve, context, generate: mean ms per query (null if the stage did not run),
             end_to_end: {mean, p50, p95}}
resources   {cpu_s (summed over queries), peak_rss_mb (whole process)}
run         {run_id, config_name, subset, n_queries, started, finished, git_commit}
```

`est_usd` counts whisper-1 audio only ($0.006 per minute); the free ELM chat models are $0.
`judge_calls` counts new judge requests; verdicts already in `judge/judge_cache.jsonl` are free.
Judge calls are evaluation overhead, so they are not part of `cost` (which is what the system spends).

## Evaluation

```bash
mlsrag judge <run-dir>                       # one judge call per query (cached); --dry-run validates only
mlsrag estimate --config baseline --subset practice-20pct --judge   # expected calls before a run
mlsrag matrix --matrix rag/extensions/platform/configs/matrix-smoke.yaml --no-judge
mlsrag compare <run-dir> <run-dir> ...       # Markdown table, one column per run
```

- **Judge.** `mlsrag judge` writes `answers.jsonl` and runs the course's own
  `rag/scripts/evaluate_answers.py --allow-subset` as a subprocess with `ELM_JUDGE_MODEL` set from
  the config's `judge_model` (default Mistral-Small, so the judge is not the Qwen generator), so the
  scores are exactly what the markers' tool gives. Keep the judge fixed across a comparison.
  `--repeats n` judges each answer n times and averages (for variance and statistical tests later).
- **Estimate.** Upper bounds: one generation call per query, one judge call per query and repeat,
  the flow's extra calls (`FLOW_CALLS_PER_QUERY` in `evaluate.py`; add new flows there), and
  whisper calls for queries missing from the transcript cache.
- **Matrix.** A YAML with `name`, `subset` and `configs` (preset names, `configs/<name>.yaml`
  without the extension, or paths). It prints the estimate and asks for `y` (`--yes` skips),
  refuses `practice-all`/`dev-all` without `--allow-full`, reuses the newest finished run with the
  same config, subset and generation mode (`--force` reruns), judges unless `--no-judge` (again
  if the saved verdicts used another judge model; `--no-generate` runs retrieval only), and writes
  `rag/extensions/results/matrices/<name>/summary.json` and `summary.md`.

## Adding a retriever or a flow

A new retriever is a class with `search(query_vec, k)`; add a branch for its name in
`build_stages` and set `retriever: <name>` in a config. A new flow is a module in
`mlsrag/flows/` with a `run_query(query_id, audio_path, stages, config, trace)` that fills
`trace.artifacts` (at least `ranked_ids`) inside spans; register it in `FLOWS`. All model
calls go through the gateway so they are cached and costed; options go in `config.params`.
