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
(mean Recall@5/10/20/100, MRR@10, NDCG@10, wall seconds, mean ms per span name, ELM calls,
tokens, audio seconds, cache hits). The course `.gitignore` ignores `results/`: commit a run with `git add -f`.

## Adding a retriever or a flow

A new retriever is a class with `search(query_vec, k)`; add a branch for its name in
`build_stages` and set `retriever: <name>` in a config. A new flow is a module in
`mlsrag/flows/` with a `run_query(query_id, audio_path, stages, config, trace)` that fills
`trace.artifacts` (at least `ranked_ids`) inside spans; register it in `FLOWS`. All model
calls go through the gateway so they are cached and costed; options go in `config.params`.
