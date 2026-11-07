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
- `mlsrag/maps.py`: linear maps between the enc_a and enc_b spaces (`mlsrag fit-map`).
- `mlsrag/asr_compare.py`: two transcript caches compared (`mlsrag asr-compare`).
- `mlsrag/preprocess/`, `mlsrag/retrievers/topic.py`: the preprocessing pipelines and the topic retriever (see the end of this file).
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
(error | as-is | re-encode | translate: gte-small belongs to enc_a, bge-small to enc_b; see below); `retriever` (exact); `k` (5);
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
all gte-small queries against enc_b; `configs/matrix-encoders-smoke.yaml` runs them with the baseline.

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

## Adding a retriever or a flow

A new retriever is a class with `search(query_vec, k)`; add `name: factory(config, gateway, data)` to
`RETRIEVERS` in `mlsrag/stages.py` (context builders: `CONTEXT_BUILDERS`) and set `retriever: <name>` in a config. A new flow is a module in
`mlsrag/flows/` with a `run_query(query_id, audio_path, stages, config, trace)` that fills
`trace.artifacts` (at least `ranked_ids`) inside spans; register it in `FLOWS`. All model
calls go through the gateway so they are cached and costed; options go in `config.params`.

## Preprocessing pipelines

`mlsrag/preprocess/`: a pipeline is a list of `Stage`s run in order. Pipeline `topics`:
`select` (which records) → `cluster` (k-means on enc_a) → `label` (one LLM call per cluster) →
`topic_index` (`index.json` + `centroids.npy` for `retriever: topic`).

- **Hashes.** Each stage writes to `rag/.cache/preprocess/<pipeline>/<stage>/<hash>/`; the hash
  covers the upstream hashes, the stage's config section, its code `version` and `prompt_version`.
  Change one stage's config and only that stage and the ones below it re-run.
- **Manifest.** `manifest.json` (`status: ok`, inputs, config, wall seconds, counts, ELM cost,
  package versions) marks a stage done. A stage writes into `<hash>.partial/` and is renamed when
  it finishes, so a stopped run leaves a `partial` stage that the next run continues.
- **Ledger.** LLM stages append one line per item to `ledger.jsonl` (`item_id, input_hash, status,
  output, tokens, latency_ms, error`). Items with `ok` are skipped on resume; `failed` ones are retried.
- **Selection.** `select` is an ordinary stage, so every downstream hash depends on it.
- **Clustering.** `KMeans(k, random_state=0, n_init=1)` on the normalised vectors. Spherical k-means
  is not needed at this size; centroids are re-normalised for cosine routing.

```bash
mlsrag preprocess run    --pipeline topics --config rag/extensions/platform/configs/topics-k256.yaml
mlsrag preprocess status --pipeline topics --config ...   # per stage: hash, ok|partial|missing, time, ELM calls/tokens
mlsrag preprocess clean  --pipeline topics --config ... --stage label
# selection flags (override the config's select section): --all | --ids FILE | --sample N --seed S
#   | --split dev-relevant --distractors M --seed S
mlsrag run --config rag/extensions/platform/configs/topics-k256-m4.yaml --subset practice-smoke20 --no-generate
```

`retriever: topic` takes `params: {index: <topics pipeline config>, m: 4}`: it ranks the centroids,
takes the `m` nearest clusters and searches their members exactly (trace: span `topic.route`,
counters `topic.clusters_probed`, `topic.candidates_scanned`).

To add a stage: write `run(inputs, config, ctx)` in a module under `mlsrag/preprocess/`.
`inputs` maps each upstream stage name you declared to its output directory; `config` is your
section of the YAML; write your files into `ctx.out_dir`, call models through `ctx.gateway`
(costs land on `ctx.trace`), and return a small dict of counts for the manifest. For per-item LLM
work, keep a `Ledger(ctx.out_dir / "ledger.jsonl")` and skip `ledger.done_ids()`. Register it in
the pipeline's list in `mlsrag/preprocess/__init__.py` with `Stage(name, run, inputs=[...])`, and
bump `version` whenever you change what it writes.

## HTTP API (`mlsrag serve`)

```bash
mlsrag serve --host 127.0.0.1 --port 8000 [--web-dir web/dist]   # API under /api, the client (if given) at /
mlsrag openapi                                                    # rewrite openapi.json (a test checks it is current)
```

`openapi.json` (next to this README) is the contract the web client is built against. Every query
goes through the same `build_stages` and flow as `mlsrag run`, with one trace per request; stages are
built once per config (at most 8 kept) and queries run one at a time. Typed text replaces the ASR
stage's output, a given vector replaces the encoder's.

| Method and path | What it does |
|---|---|
| `GET /api/health` | `{status, git_commit}` |
| `GET /api/models` | ASR (`course` is `available: false` without a GPU), encoders and their space, indexes, the four free ELM models, retrievers and context builders (`stages.RETRIEVERS`, `stages.CONTEXT_BUILDERS`) and flows (`flows.FLOWS`) |
| `GET /api/configs` | presets and the RunConfig YAMLs in `configs/` (matrix and preprocessing files skipped), with all fields |
| `POST /api/query` | JSON `{config: <name or RunConfig>, input: {text} \| {vector, space, question?}}` → `{answer, transcript, retrieved: [{record_id, score, text[:500]}], prompt, trace}` |
| `POST /api/query/audio` | `multipart/form-data`: `config` (name or RunConfig JSON), `audio` (WAV or WebM/Opus, passed to whisper-1 unconverted). A `cached` ASR config uses whisper-1, since an upload is never in a transcript cache. Non-WAV audio is costed as 0 s |
| `GET /api/results` | committed runs `{run_id, config_name, subset, n_queries, finished}`, matrix summaries (`results/matrices/`) and ASR comparisons (`results/asr-compare/`) |
| `GET /api/results/{run_id}` | its `config.json` and `metrics.json` |
| `GET /api/results/{run_id}/queries?offset&limit` | a page of `queries.jsonl` traces (limit at most 100) |
| `GET /api/records/{record_id}` | the full record text |
| `POST /api/runs` (admin) | `{config, subset, generate, allow_full}`: starts `mlsrag run` in a background thread, returns a job id and the call estimate; full splits are refused unless `allow_full: true` |
| `GET /api/runs/{id}/status` (admin) | `running`, `done` (with `run_id`) or `failed` (with `error`) |

Errors: 400 for an unknown or invalid config, a vector of the wrong size, or a vector in the wrong
space whose `mismatch_policy` cannot apply (`as-is` and `translate` can; `re-encode` needs text);
502 when an ELM request fails.

**Admin.** Set `ADMIN_TOKEN` in the environment and send it as the `X-Admin-Token` header. Without
`ADMIN_TOKEN` the admin endpoints always return 403. Jobs live in memory only.

**Privacy and safety.** Nothing about users is stored: no accounts, no logs of who asked what
(uploads go to a temp folder and are deleted after transcription). Model calls are still cached and
logged by the gateway as for any run, under `rag/.cache/`. Model output is returned as text and
never executed.
