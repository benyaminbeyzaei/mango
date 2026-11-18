# mlsrag: the experiment platform

Every RAG experiment runs through this package. It wraps the course code in `rag/src/` (never
modified), traces each query, and sends every model call through one cached, logged, costed gateway.
The markers' yardstick stays the course scripts (`rag/README.md`); this platform sits on top.

Install (from the checkout root, inside a venv): `uv pip install -e "rag/extensions/platform[dev]"`.
Tests: `pytest rag/extensions/platform` (fast); `pytest -m slow rag/extensions/platform` (needs the data and the transcript cache).

Quickest way to the web client: [Run with Docker](#run-with-docker). More detail in `docs/`:
[retrievers](docs/retrievers.md) (context builders, encoder policies, ASR comparison, BM25, grep, adding a
retriever or flow), [pipelines](docs/pipelines.md), [HTTP API](docs/api.md).

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
│   ├── flows/          standard, grep-static, grep-agentic, grep-feedback: one query
│   ├── retrievers/     bm25, grep, topic, toc, raptor (raptor-lite, raptor-tree, raptor-full), rerank
│   ├── prompts/        grep_v1
│   ├── preprocess/     pipelines of hashed, resumable stages (topics, toc, raptor, raptor-tree, raptor-full)
│   ├── runner.py       run a config over a subset → a results folder
│   ├── results.py      metrics.json schema, load_run, list_runs, compare
│   ├── evaluate.py     judging through the course evaluator, call estimates
│   ├── matrix.py       many configs on one subset, one summary table
│   ├── final.py        plan 032: reranker choice and results/matrices/final/comparison.json
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
mlsrag run --config baseline --subset practice-20 --no-generate   # retrieval only: no ELM call
mlsrag run --config baseline --subset practice-20 --fresh         # + 20 Qwen calls, real latency
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-retrievers-practice-20.yaml --no-generate --fresh
```

**Subsets** (`subsets/`, seed 0): `practice-20` (20), `practice-70` (70), `test-25` (25),
`practice` (350), `test` (125).

## Cost rules

- **`--fresh`** (on `run` and `matrix`; `fresh: true` on `POST /api/runs`): the gateway skips cache
  reads for chat and embed calls (it still writes them), so every system call is made and timed.
  Judge calls stay cached (the judge is the instrument, not the system); whisper and `asr: cached`
  stay as configured; preprocessing always keeps the cache. `metrics.json` records `run.fresh`.
  **Latency and cost are valid only for fresh runs:** `matrix` and `compare` show them as `-`
  (the web client as `–`) for any other run; retrieval and answer scores are shown either way.
- **Full splits** (`practice`, `test`) need `--allow-full` on `run`, `transcribe` and `matrix`
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
  Each summary row carries `ci95`: a 95 % percentile bootstrap interval over the run's queries
  (1,000 resamples, seed 0) for Recall@10 and NDCG@10, shown in `summary.md` as `0.485 [0.443, 0.526]`.

## How to run the experiments

The primary question: retrieval strategy vs cost per correct answer. Controls, the same in every row
of matrices A and B and stated in each matrix file's header: the whisper-1 transcript cache,
gte-small on enc_a, k=5, generator `Qwen/Qwen3.5-397B-A17B-FP8` at temperature 0.1, judge
`mistralai/Mistral-Small-4-119B-2603` through the course evaluator, the course context prompt and
`grep_v1`. The grep flows make their pattern calls on `meta-llama/Llama-3.3-70B-Instruct` (plan 021:
Qwen's hidden reasoning took ~70-100 s per call); answers stay on Qwen. Always pass `--fresh`.

Wall time (queries one at a time, from the fresh smoke runs): Qwen generation ~52 s, grep static ~1.4 s,
grep agentic ~5.6 s, grep feedback ~7.7 s per query (Llama-3.3), toc-llm ~1.1 s, toc-llm-topics ~1.5 s,
toc-llm-flat ~2.4 s, all other retrievers < 0.2 s (the plan 030 `toc-llm*-gpt5labels` runs took 4.0 / 4.5 / 6.5 s
while another job loaded the CPU, so their latency is not comparable). Judge time is
not measured.

The fresh Qwen grep runs (`results/20261108-1949-grep-static-practice-20`,
`results/20261108-2021-grep-agentic-practice-20`) are kept for the helper-model comparison
with the Llama runs (`results/20261109-1134-grep-*`): grep static Recall@10 0.108 → 0.133,
MRR@10 0.172 → 0.111, ~94 s → ~1.4 s per query; grep agentic Recall@10 0.208 → 0.067,
MRR@10 0.350 → 0.175, ~304 s → ~5.6 s per query (Qwen → Llama, practice-20).

| Matrix file (`configs/matrices/`) | Subset | Mode | ELM calls (fresh) | Wall time |
|---|---|---|---|---|
| A `matrix-retrievers-practice-20.yaml` (21 retrieval configs) | practice-20 | `--no-generate` | 380 (180 grep, 200 toc-llm: 2 x (40 + 40 + 20)) | ~12 min |
| A `matrix-a-retrieval.yaml` | practice-70 (70) | `--no-generate` | 980, after filling the transcript cache | ~24 min |
| A `matrix-a-retrieval-dev.yaml` | test-25 (25) | `--no-generate` | 350 | ~9 min |
| B `matrix-b-answers.yaml` (7 configs, simple context) | test-25 | generate + judge | 375 + 175 judge | ~2.6 h + judge |
| B `matrix-b-answers-budgeted.yaml` (same rows, budgeted) | test-25 | generate + judge | 375 + 175 judge | ~2.6 h + judge |
| C `matrix-c-secondary.yaml` (encoder policies) | test-25 | `--no-generate` | 0 | ~1 min |
| A `matrix-retrievers-practice.yaml` (14 no-LLM retrievers) | practice (350) | `--no-generate --allow-full` | 0 | ~5 min |
| A `matrix-retrievers-test.yaml` (same rows) | test (125) | `--no-generate --allow-full` | 0 | ~2 min |
| C `matrix-encoders-practice.yaml` (baseline + 5 encoder policies) | practice | `--no-generate --allow-full` | 0 | ~3 min |
| C `matrix-encoders-test.yaml` (same rows) | test | `--no-generate --allow-full` | 0 | ~1 min |

Matrix C varies the encoder and index. Since plan 029 the local transcript caches cover all of
practice (350) and dev (125), so no matrix makes whisper-1 calls here. They are not in git: a fresh
clone fills them once (430 whisper-1 calls, 2,263 audio s, about $0.23; resumable) with
`mlsrag transcribe --asr whisper1 --subset practice --allow-full --out rag/.cache/transcripts/whisper1-practice.jsonl`
and the same with `test` and `whisper1-dev.jsonl`.
Matrix B's rows are one per family; after matrix A, replace each with its family's best setting.
A `test` run (125 queries) costs five times the test-25 numbers.

**Full-split results for the no-LLM methods (plans 029, 030).** The four `*-all` matrices above ran fresh
on 2026-11-09, after 430 whisper-1 calls filled both transcript caches (2,263 audio s, ~$0.23); the two A
matrices ran again for plan 030, with identical numbers for the earlier rows.
Their summaries give a 95 % percentile bootstrap interval over queries (1,000 resamples, seed 0) for
Recall@10 and NDCG@10. Only `results/matrices/<name>/summary.{md,json}` is committed; the run
folders they name stay local (~70 MB), so the web Compare page answers "no run" for them.
Recall@10 [95 % CI]:

| Config | practice | test |
|---|---|---|
| baseline (gte-small, enc_a, exact) | 0.485 [0.443, 0.526] | 0.479 [0.409, 0.558] |
| bm25 | 0.297 [0.256, 0.337] | 0.281 [0.215, 0.351] |
| topics-k256-m2 / m4 / m8 | 0.328 / 0.382 / 0.435 [0.392, 0.477] | 0.340 / 0.420 / 0.458 [0.390, 0.535] |
| toc-centroid-m4 | 0.355 [0.314, 0.396] | 0.376 [0.303, 0.449] |
| raptor-lite / raptor-tree, expand and evidence | 0.484-0.485 | 0.461-0.469 |
| raptor-tree with gpt-5-mini summaries (plan 030), expand / evidence | 0.484 / 0.485 | 0.477 / 0.477 [0.409, 0.556] |
| raptor-full expand / evidence (plan 031) | 0.461 [0.418, 0.502] / 0.452 [0.411, 0.493] | 0.437 [0.369, 0.512] / 0.433 [0.363, 0.510] |
| enc-b-asis | 0.465 [0.422, 0.508] | 0.437 [0.365, 0.514] |
| enc-b-reencode / reencode-prefix | 0.505 / 0.516 [0.476, 0.560] | 0.491 / 0.526 [0.455, 0.599] |
| enc-b-translate-lstsq / procrustes | 0.467 / 0.422 [0.376, 0.463] | 0.473 / 0.425 [0.355, 0.497] |

On practice, bm25, topics m=2 and m=4 and toc-centroid-m4 lie below the baseline's interval
(non-overlapping), and m=8 is above m=2; topics m=8, every RAPTOR variant and every encoder policy
overlap the baseline on Recall@10. On NDCG@10, enc-b-asis and enc-b-translate-procrustes lie below the
baseline. Overlapping intervals do not show two methods are equal, only that 350 queries cannot
separate them. The faithful RAPTOR (`raptor-full`, plan 031: UMAP + GMM soft clustering, five
summary layers, gpt-4o-mini, $1.11 to build) is the only RAPTOR variant below the baseline on every
metric: practice Recall@10 0.461 vs 0.485, NDCG@10 0.399 [0.362, 0.437] vs 0.425 [0.385, 0.462]
(intervals overlap; evidence mode is lower still); its ranked lists hold about 23 summary hits per
query, nearly all layer 1. Chosen on practice (dev only reports): topics m=8, raptor-lite-expand (the four RAPTOR
rows are within 0.002 of the baseline), encoder policy `enc-b-reencode-prefix`; the same order holds
on dev. Against the 20-question smoke runs: baseline best in matrix A, topics improving with m, toc
below topics, bm25 far below all held; the smoke encoder ranking (lstsq best, prefix worst
re-encode) did not: on both full splits prefix is best and lstsq is level with as-is. Full tables,
NDCG@10 and latency are in the summaries.

**Final three-way comparison (plan 032).** Three rows, everything else fixed (whisper-1 transcripts,
gte-small on enc_a, k=5, Qwen3.5-397B at 0.1 with the course prompt, Mistral-Small-4 judge): `baseline`;
`rerank-best` (dense top 20 re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU, chosen on
practice from five settings: highest NDCG@10, then the fastest within 0.005); and
`raptor-tree-gpt5-evidence` (records and gpt-5-mini summaries searched together, a summary hit goes into the
prompt). Matrices `matrix-final-{smoke,practice,test}.yaml`; every number, the paired differences and
evidence-backed explanations are in `results/matrices/final/comparison.json` (`python -m mlsrag.final`).
Intervals are 95 % bootstraps over questions.

| Row | practice NDCG@10 | test NDCG@10 | test answer_norm | paired vs baseline (dev) | rerank ms/q |
|---|---|---|---|---|---|
| baseline | 0.425 [0.385, 0.462] | 0.402 [0.338, 0.470] | 0.687 [0.622, 0.751] | - | - |
| rerank-best | 0.418 [0.380, 0.454] | 0.393 [0.333, 0.460] | 0.646 [0.577, 0.714] | -0.041 [-0.095, 0.011] | 1,250 |
| raptor-tree-gpt5-evidence | 0.425 [0.385, 0.462] | 0.400 [0.336, 0.470] | 0.690 [0.625, 0.755] | +0.003 [-0.045, 0.051] | - |

Reranking lowers the rank-sensitive metrics (it raises Recall@10 slightly), and larger pools or
bge-reranker-base lower them further. Summaries reach the prompt in only 10 of 125 dev questions. Neither
change measurably improves answers: with identical context, generation plus judging alone changed
answer_norm in 58 of 115 questions. Latency was measured on 6 cores (selection) or 10 cores (final runs) at
nice 19 while the laptop was in use, with generation sequential (about 85-90 s per answer, which dominates).

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
docker compose run --rm cli mlsrag transcribe --asr whisper1 --subset practice-20 --out rag/.cache/transcripts/whisper1-practice.jsonl
docker compose run --rm cli mlsrag run --config baseline --subset practice-20 --no-generate
```

The port is bound to `127.0.0.1` only: the API has no login and spends ELM calls. `rag/.env` is
mounted read-only, never copied into the image. Containers run as uid/gid 1000; to change it, `export
UID GID` first (shells set `UID` but do not export it). Set `ADMIN_TOKEN` in your shell to enable the
admin endpoints. `.git` is not in the image, so `/api/health` and run metadata show `git_commit:
null`. The course ASR needs a GPU and is not in the image; `whisper1` and `cached` work.
