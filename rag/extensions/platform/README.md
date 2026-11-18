# mlsrag: the experiment platform

Every RAG experiment runs through this package. It wraps the course code in `rag/src/` (never modified), traces
each question, and sends every model call through one cached, logged and costed gateway. The course scripts
(`rag/README.md`) stay the yardstick; this platform sits on top of them.

- **Results:** [docs/results.md](docs/results.md), every number in one format.
- **Methods:** [retrievers and context builders](docs/retrievers.md), [preprocessing pipelines](docs/pipelines.md),
  [HTTP API](docs/api.md).

```bash
uv pip install -e "rag/extensions/platform[dev]"     # from the checkout root, inside a venv
pytest rag/extensions/platform                       # fast tests; -m slow needs the data and transcripts
docker compose up                                    # API + Mango, the website, at http://localhost:8000 (see below)
```

## Question sets

Fixed lists in `subsets/` (seed 0), so every run uses the same questions:

| Name | Questions | Use |
|---|---|---|
| `practice-20` | 20 from practice | quick checks while building |
| `practice-70` | 70 from practice (20 %) | development comparisons |
| `practice` | all 350 practice | choosing settings |
| `test-25` | 25 from test (20 %) | quick final comparisons |
| `test` | all 125 test (the course's `dev` split) | reporting only; nothing is tuned on it |

`practice` and `test` are full splits: `run`, `transcribe` and `matrix` refuse them without `--allow-full`.

## Map

```
rag/extensions/platform/
├── mlsrag/
│   ├── course.py       rag/ on sys.path, the course classes, `paths`
│   ├── config.py       RunConfig, load_config(path or preset)
│   ├── trace.py        Trace: spans, counters, cost, resources, artifacts (one per question)
│   ├── gateway.py      chat / embed / transcribe / rerank: cache, log, retries, cost
│   ├── stages.py       the five stage interfaces and their registries, build_stages
│   ├── context.py      budgeted and dedupe context builders
│   ├── flows/          standard, grep-static, grep-agentic, grep-feedback
│   ├── retrievers/     bm25, grep, topic, toc, raptor (lite, tree, full), rerank
│   ├── preprocess/     hashed, resumable offline stages (topics, toc, raptor, raptor-tree, raptor-full)
│   ├── runner.py       run a config over a question set → a results folder
│   ├── results.py      metrics.json schema, load_run, list_runs, compare, bootstrap intervals
│   ├── evaluate.py     judging through the course evaluator, call estimates
│   ├── matrix.py       many configs on one question set, one summary table
│   ├── final.py, rerank_answers.py, finetune.py   the analyses behind docs/results.md
│   ├── maps.py, asr_compare.py, subsets.py       encoder maps, ASR comparison, question sets
│   ├── api/            FastAPI app (`mlsrag serve`)
│   └── cli.py          the `mlsrag` command
├── configs/            one YAML per approach; matrices/ and pipelines/ below it
├── subsets/            the question lists
└── docs/, tests/, jobs/ (cluster scripts), openapi.json, Dockerfile
```

## Concepts

**RunConfig** (`config.py`). One experiment's settings, a pydantic model: `name`; `asr` (cached |
whisper1 | course) and `asr_cache`; `query_encoder` (gte-small | bge-small); `corpus_index` (enc_a |
enc_b) and `mismatch_policy` (error | as-is | re-encode | translate); `retriever` (exact | topic |
toc | raptor-lite | raptor-tree | raptor-full | bm25 | grep | rerank); `k` (5); `context_builder` (simple | budgeted |
dedupe | with-summaries); `generator_model`; `judge_model`; `flow` (standard | grep-static |
grep-agentic | grep-feedback); `params` (free dict for the retriever's options). A YAML file works like a preset name.

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
encode → retrieve (100 ranked ids) → context (top k) → generate. `grep-static`, `grep-agentic` and
`grep-feedback` replace encode and retrieve with LLM-written regex scans.

**Pipelines** (`preprocess/`). Offline work before a run (clusters, labels, summaries). Each
stage writes to a folder named by a hash of its inputs, config and code version, so only changed
stages rerun; LLM stages keep a per-item ledger and resume after a stop. See [docs/pipelines.md](docs/pipelines.md).

**Results** (`runner.py`, `results.py`). `mlsrag run` writes one folder per run (below); `matrix`,
`compare`, the API and Mango only read those folders.

## Commands (from the checkout root)

| Command | What it does |
|---|---|
| `mlsrag run --config C --subset S [--no-generate] [--fresh]` | run a config over a question set |
| `mlsrag matrix --matrix FILE [--no-generate] [--no-judge] [--fresh] [--yes]` | several configs, one summary table |
| `mlsrag judge RUN_DIR [--repeats n]` | judge a run's answers with the course evaluator |
| `mlsrag estimate --config C --subset S [--judge]` | expected ELM calls and cost, no calls made |
| `mlsrag compare RUN_DIR ...` | Markdown table, one column per run |
| `mlsrag transcribe --asr whisper1 --subset S --out FILE` | fill a transcript cache (resumable) |
| `mlsrag asr-compare --a FILE --b FILE --subset S` | word disagreement and retrieval of two transcript caches |
| `mlsrag preprocess run\|status\|clean --pipeline P --config FILE` | offline stages ([pipelines](docs/pipelines.md)) |
| `mlsrag fit-map --from enc_a --to enc_b --method lstsq --exclude-dev` | map between encoder spaces |
| `mlsrag serve [--host --port --web-dir web/dist]` | HTTP API and Mango, the website |
| `mlsrag openapi` / `mlsrag make-subsets` | rewrite `openapi.json` / the question lists |

```bash
mlsrag run --config baseline --subset practice-20 --no-generate    # retrieval only, no ELM call
mlsrag run --config baseline --subset practice-20 --fresh          # + 20 Qwen answers, real latency
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-final-test.yaml --fresh --allow-full
```

A matrix YAML has `name`, `subset`, `configs` and an optional `override:` applied to every config. It reuses the
newest finished run with the same config, question set and mode (`--force` reruns), judges unless `--no-judge`,
and writes `results/matrices/<name>/summary.{json,md}` with 95 % bootstrap intervals for Recall@10 and NDCG@10.

## Cost rules

- **`--fresh`** skips cache reads for chat, embed and rerank calls, so every call is made and timed. Latency and
  cost are valid only for fresh runs; tables show `–` otherwise. Judge calls stay cached (the judge is the
  instrument, not the system).
- **Full splits** need `--allow-full`; `matrix` prints its estimate and asks before running (`--yes` skips).
- The free ELM chat models cost $0; `est_usd` counts whisper-1 audio only ($0.006 per minute).
- Transcript caches are not in git. Fill them once (430 whisper-1 calls, about $0.23):
  `mlsrag transcribe --asr whisper1 --subset practice --allow-full --out rag/.cache/transcripts/whisper1-practice.jsonl`,
  and the same with `--subset test` and `whisper1-dev.jsonl` (cache files keep the course's split names).

## Results

`rag/extensions/results/<YYYYMMDD-HHMM>-<config name>-<question set>/`: `config.json`; `queries.jsonl`
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

## Run with Docker

One image (`rag/extensions/platform/Dockerfile`, about 2.1 GB, CPU only) holds the API, the CLI and
the built Mango website. The data, the model, `rag/.cache/`, the results and `rag/.env` stay on your
disk and are mounted (`compose.yaml`). From a fresh clone:

```bash
git clone <this repo> && cd <it> && git lfs install && git lfs pull --include="rag-assets.zip"
unzip rag-assets.zip -d ~/mls-rag-assets   # the course script needs numpy, so run it with uv:
uv run --no-project --with numpy python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
cp rag/.env.example rag/.env    # then put your ELM key in rag/.env
mkdir -p rag/.cache             # so the mount is owned by you, not root
docker compose up               # API + client at http://localhost:8000 (first run builds the image, ~2 min)
# transcript caches are not in git: fill one before a cached-ASR run (20 whisper-1 calls, ~$0.01)
docker compose run --rm cli mlsrag transcribe --asr whisper1 --subset practice-20 --out rag/.cache/transcripts/whisper1-practice.jsonl
docker compose run --rm cli mlsrag run --config baseline --subset practice-20 --no-generate
```

The port is bound to `127.0.0.1` only: the API has no login and spends ELM calls. `rag/.env` is
mounted read-only, never copied into the image. Containers run as uid/gid 1000; to change it, `export
UID GID` first (shells set `UID` but do not export it). Set `ADMIN_TOKEN` in your shell to enable the
admin endpoints. `.git` is not in the image, so `/api/health` and run metadata show `git_commit:
null`. The course ASR needs a GPU and is not in the image; `whisper1` and `cached` work.
