# mlsrag: the experiment platform

Every RAG experiment runs through this package. It wraps the course code in `rag/src/` (never
modified), traces each query, and sends every model call through one cached, logged, costed gateway.
The markers' yardstick stays the course scripts (`rag/README.md`); this platform sits on top.

Install (from the checkout root, inside a venv): `uv pip install -e "rag/extensions/platform[dev]"`.
Tests: `pytest rag/extensions/platform` (fast); `pytest -m slow rag/extensions/platform` (needs the data and the transcript cache).

Quickest way to the web client: [Run with Docker](#run-with-docker). More detail in `docs/`:
[retrievers](docs/retrievers.md) (context builders, encoder policies, ASR comparison, BM25, grep, adding a
retriever or flow), [pipelines](docs/pipelines.md), [graphs](docs/graphs.md), [HTTP API](docs/api.md).

## Map

```
rag/extensions/platform/
├── mlsrag/
│   ├── course.py       rag/ on sys.path, the course classes, `paths`
│   ├── config.py       RunConfig, load_config(path or preset)
│   ├── trace.py        Trace: spans, counters, cost, resources, artifacts (one per query)
│   ├── gateway.py      ModelGateway.chat / embed / transcribe: cache, log, retries, cost
│   ├── stages.py       the five stage interfaces, RETRIEVERS, CONTEXT_BUILDERS, build_stages
│   ├── context.py      budgeted and dedupe context builders
│   ├── flows/          standard, grep-static, grep-agentic: one query through the stages
│   ├── retrievers/     bm25, grep, topic, toc, raptor-lite, lazy-graph, kg
│   ├── prompts/        grep_v1
│   ├── preprocess/     pipelines of hashed, resumable stages (topics, toc, raptor, lazy-graph, kg)
│   ├── runner.py       run a config over a subset → a results folder
│   ├── results.py      metrics.json schema, load_run, list_runs, compare
│   ├── evaluate.py     judging through the course evaluator, call estimates
│   ├── matrix.py       many configs on one subset, one summary table
│   ├── maps.py, asr_compare.py, subsets.py   encoder maps, ASR comparison, subset files
│   ├── api/            FastAPI app (`mlsrag serve`) and its pydantic models
│   └── cli.py          the `mlsrag` command
├── configs/            RunConfig YAMLs; matrices/ and pipelines/ below it
├── subsets/            fixed query-id lists
├── docs/, tests/, openapi.json, Dockerfile
```

## Concepts

**RunConfig** (`config.py`). One experiment's settings, a pydantic model: `name`; `asr` (cached |
whisper1 | course) and `asr_cache`; `query_encoder` (gte-small | bge-small); `corpus_index` (enc_a |
enc_b) and `mismatch_policy` (error | as-is | re-encode | translate); `retriever` (exact | topic |
lazy-graph | kg | toc | raptor-lite | bm25 | grep); `k` (5); `context_builder` (simple | budgeted |
dedupe | with-summaries); `generator_model`; `judge_model`; `flow` (standard | grep-static |
grep-agentic); `params` (free dict for the retriever's options). A YAML file works like a preset name.

**Trace** (`trace.py`). One per query: timed `spans` (asr, encode, retrieve, context, generate and
any inner ones), `counters` (e.g. `grep.llm_calls`), `cost` per model, `resources` (CPU, RSS) and
`artifacts` (transcript, ranked ids, scores, prompt, answer). Saved as one line of `queries.jsonl`.

**Gateway** (`gateway.py`). The only way to call a model. Disk cache `rag/.cache/gateway/`, call log
`rag/.cache/gateway.log.jsonl`, at most 4 concurrent ELM requests, 3 retries on 429, 5xx and
timeouts, tokens and audio seconds added to the current trace. The key is read from `rag/.env`.
Embeddings run locally (sentence-transformers); chat and whisper go to ELM.

**Stages** (`stages.py`). Five interfaces: `ASR.transcribe(audio) -> str`, `Encoder.encode(text) ->
vector`, `Retriever.search(vec, k) -> [(record_id, score)]` (with `uses_question = True` it also gets
the transcript), `ContextBuilder.build(question, ids) -> prompt`, `Generator.generate(prompt) ->
answer`. `build_stages(config, gateway, data)` picks one implementation of each from the registries.

**Flows** (`flows/`). A flow runs one query through the stages and fills the trace. `standard`: asr →
encode → retrieve (100 ranked ids) → context (top k) → generate. `grep-static` and `grep-agentic`
replace encode and retrieve with LLM-written regex scans.

**Pipelines** (`preprocess/`). Offline work before a run (clusters, labels, summaries, graphs). Each
stage writes to a folder named by a hash of its inputs, config and code version, so only changed
stages rerun; LLM stages keep a per-item ledger and resume after a stop. See [docs/pipelines.md](docs/pipelines.md).

**Results** (`runner.py`, `results.py`). `mlsrag run` writes one folder per run (below); `matrix`,
`compare`, the API and the web client only read those folders.

## Commands (run from the checkout root)

| Command | What it does |
|---|---|
| `mlsrag run --config C --subset S [--no-generate] [--fresh]` | run a config over a subset |
| `mlsrag transcribe --asr whisper1 --subset S --out FILE` | fill a transcript cache (resumable) |
| `mlsrag judge RUN_DIR [--repeats n] [--dry-run]` | judge a run's answers with the course evaluator |
| `mlsrag estimate --config C --subset S [--judge] [--fresh]` | expected ELM calls and cost, no calls made |
| `mlsrag matrix --matrix FILE [--no-generate] [--no-judge] [--fresh] [--yes]` | several configs, one summary |
| `mlsrag compare RUN_DIR RUN_DIR ...` | Markdown table, one column per run |
| `mlsrag preprocess run\|status\|clean --pipeline P --config FILE` | preprocessing ([docs/pipelines.md](docs/pipelines.md)) |
| `mlsrag fit-map --from enc_a --to enc_b --method lstsq --exclude-dev` | map between encoder spaces |
| `mlsrag asr-compare --a FILE --b FILE --subset S` | WER and retrieval of two transcript caches |
| `mlsrag serve [--host --port --web-dir web/dist]` | HTTP API and web client ([docs/api.md](docs/api.md)) |
| `mlsrag openapi` | rewrite `openapi.json` (a test checks it is current) |
| `mlsrag make-subsets` | regenerate `subsets/*.txt` (identical every time) |

```bash
mlsrag run --config baseline --subset practice-smoke20 --no-generate   # retrieval only: no ELM call
mlsrag run --config baseline --subset practice-smoke20 --fresh         # + 20 Qwen calls, real latency
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-a-smoke.yaml --no-generate --fresh
```

**Subsets** (`subsets/`, seed 0): `practice-smoke20` (20), `practice-20pct` (70), `dev-20pct` (25),
`practice-all` (350), `dev-all` (125); `records-dev-relevant-d275.txt` is a record list (600) for
`params.record_subset` ([docs/graphs.md](docs/graphs.md)).

## Cost rules

- **`--fresh`** (on `run` and `matrix`; `fresh: true` on `POST /api/runs`): the gateway skips cache
  reads for chat and embed calls (it still writes them), so every system call is made and timed.
  Judge calls stay cached (the judge is the instrument, not the system); whisper and `asr: cached`
  stay as configured; preprocessing always keeps the cache. `metrics.json` records `run.fresh`.
  **Latency and cost are valid only for fresh runs:** `matrix`, `compare` and the web client show
  them as `–` for any other run; retrieval and answer scores are shown either way.
- **Full splits** (`practice-all`, `dev-all`) need `--allow-full` on `run`, `transcribe` and `matrix`
  (`allow_full: true` on `POST /api/runs`); `mlsrag preprocess run` asks `y/N` before LLM calls unless `--yes`.
- `matrix` prints its estimate and asks for `y` (`--yes` skips). Agents run only smoke matrices; the
  20 % and full-split runs need Benyamin's go-ahead.
- `est_usd` counts whisper-1 audio only ($0.006 per minute); the free ELM chat models are $0.

## Results

`rag/extensions/results/<YYYYMMDD-HHMM>-<config name>-<subset>/`: `config.json`; `queries.jsonl`
(one trace per line plus per-query `metrics`); `metrics.json` (below); after `mlsrag judge`, also
`answers.jsonl` (the course candidate format) and `judge/` (the course evaluator's output and its
cache). Matrix summaries: `results/matrices/<name>/summary.{json,md}`. The course `.gitignore`
ignores `results/`: commit a run with `git add -f`.

```
retrieval   {Recall@5, Recall@10, Recall@20, Recall@100, MRR@10, NDCG@10}      means over queries
answers     {correctness, grounding, completeness (0-4), normalized (sum/12),
             judged_queries, judge_model, repeats, judge_calls}                 null until judged
cost        {elm_calls, prompt_tokens, completion_tokens, audio_seconds, est_usd}  cached calls are free
latency_ms  {asr, encode, retrieve, context, generate (mean ms per query, null if not run),
             end_to_end: {mean, p50, p95}}
resources   {cpu_s (summed over queries), peak_rss_mb (whole process)}
run         {run_id, config_name, subset, n_queries, started, finished, git_commit, fresh}
```

`mlsrag.results.Metrics` checks this on every write and load. Judge calls are not part of `cost`.

## Evaluation and matrices

- **Judge.** `mlsrag judge` writes `answers.jsonl` and runs the course's own
  `rag/scripts/evaluate_answers.py --allow-subset` with `ELM_JUDGE_MODEL` set from the config's
  `judge_model` (Mistral-Small by default, so the judge is not the Qwen generator): the scores are
  exactly what the markers' tool gives. Keep the judge fixed across a comparison.
- **Estimate.** Upper bounds: one generation and one judge call per query (and repeat), the flow's
  extra calls (`FLOW_CALLS_PER_QUERY` in `evaluate.py`) and whisper calls for uncached transcripts.
- **Matrix.** A YAML with `name`, `subset`, `configs` (presets, `configs/<name>`, or paths) and an
  optional `override:` of top-level RunConfig fields applied to every config. Each overridden field
  except `asr_cache` is appended to the run name (`baseline+context_builder=budgeted`), so two
  matrices that differ only in an override write distinguishable runs. A matrix reuses the newest
  finished run with the same config, subset and generation mode (with `--fresh`, only a fresh one;
  `--force` reruns), judges unless `--no-judge` (`--no-generate` runs retrieval only).

## How to run the experiments

The primary question: retrieval strategy vs cost per correct answer. Controls, the same in every row
of matrices A and B and stated in each matrix file's header: the whisper-1 transcript cache,
gte-small on enc_a, k=5, generator `Qwen/Qwen3.5-397B-A17B-FP8` at temperature 0.1, judge
`mistralai/Mistral-Small-4-119B-2603` through the course evaluator, the course context prompt and
`grep_v1`. Always pass `--fresh`.

Wall time (queries one at a time, from the fresh smoke runs): Qwen generation ~52 s, grep static ~94 s,
grep agentic ~304 s per query, toc-llm ~1 s, all other retrievers < 0.2 s. Judge time is not measured.

| Matrix file (`configs/matrices/`) | Subset | Mode | ELM calls (fresh) | Wall time |
|---|---|---|---|---|
| A `matrix-a-smoke.yaml` (11 retrieval configs) | practice-smoke20 | `--no-generate` | 140 (100 grep, 40 toc-llm) | ~2.2 h |
| A `matrix-a-retrieval.yaml` | practice-20pct (70) | `--no-generate` | 490, after filling the transcript cache | ~7.8 h |
| A `matrix-a-retrieval-dev.yaml` | dev-20pct (25) | `--no-generate` | 175 | ~2.8 h |
| B `matrix-b-answers.yaml` (7 configs, simple context) | dev-20pct | generate + judge | 275 + 175 judge | ~4.6 h + judge |
| B `matrix-b-answers-budgeted.yaml` (same rows, budgeted) | dev-20pct | generate + judge | 275 + 175 judge | ~4.5 h + judge |
| C `matrix-c-secondary.yaml` (encoder policies) | dev-20pct | `--no-generate` | 0 | ~1 min |

Matrix C varies the encoder and index. The practice transcript cache covers only the smoke subset:
before matrix A on practice-20pct, fill it once (50 whisper-1 calls, 268 audio s, about $0.03) with
`mlsrag transcribe --asr whisper1 --subset practice-20pct --out rag/.cache/transcripts/whisper1-practice.jsonl`.
Matrix B's rows are one per family; after matrix A, replace each with its family's best setting.
A `dev-all` run (125 queries) costs five times the dev-20pct numbers.

## Run with Docker

One image (`rag/extensions/platform/Dockerfile`, about 2.1 GB, CPU only) holds the API, the CLI and
the built web client. The data, the model, `rag/.cache/`, the results and `rag/.env` stay on your
disk and are mounted (`compose.yaml`). From a fresh clone:

```bash
git clone <this repo> && cd <it> && git lfs install && git lfs pull --include="rag-assets.zip"
unzip rag-assets.zip -d ~/mls-rag-assets   # the course script needs numpy, so run it with uv:
uv run --no-project --with numpy python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
cp rag/.env.example rag/.env    # then put your ELM key in rag/.env
mkdir -p rag/.cache             # so the mount is owned by you, not root
docker compose up               # API + client at http://localhost:8000 (first run builds the image, ~2 min)
# transcript caches are not in git: fill one before a cached-ASR run (20 whisper-1 calls, ~$0.01)
docker compose run --rm cli mlsrag transcribe --asr whisper1 --subset practice-smoke20 --out rag/.cache/transcripts/whisper1-practice.jsonl
docker compose run --rm cli mlsrag run --config baseline --subset practice-smoke20 --no-generate
```

The port is bound to `127.0.0.1` only: the API has no login and spends ELM calls. `rag/.env` is
mounted read-only, never copied into the image. Containers run as uid/gid 1000; to change it, `export
UID GID` first (shells set `UID` but do not export it). Set `ADMIN_TOKEN` in your shell to enable the
admin endpoints. `.git` is not in the image, so `/api/health` and run metadata show `git_commit:
null`. The course ASR needs a GPU and is not in the image; `whisper1` and `cached` work.
