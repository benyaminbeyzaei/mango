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
- `mlsrag/stages.py`: the five interfaces and their adapters; registries `RETRIEVERS`, `CONTEXT_BUILDERS`; `build_stages(config, gateway, data)`.
- `mlsrag/flows/`: `standard` = asr → encode → retrieve (100 ranked ids) → context (top k) → generate; `grep-static`, `grep-agentic` (see Lexical retrieval).
- `mlsrag/retrievers/`: `bm25.py` (`BM25Retriever`), `grep.py` (`GrepIndex`). `mlsrag/prompts/grep_v1.py`: the grep prompts.
- `mlsrag/results.py`: the `metrics.json` schema (pydantic), `load_run`, `list_runs`, `compare`.
- `mlsrag/evaluate.py`: answer judging through the course evaluator, cost estimates.
- `mlsrag/matrix.py`: many configs on one subset, one summary table.
- `mlsrag/maps.py`: linear maps between the enc_a and enc_b spaces (`mlsrag fit-map`).
- `mlsrag/asr_compare.py`: two transcript caches compared (`mlsrag asr-compare`).
- `mlsrag/preprocess/`, `mlsrag/retrievers/`: the preprocessing pipelines and the topic, toc and raptor-lite retrievers (see the end of this file).
- `mlsrag/retrievers/graph.py`: the lazy-graph and knowledge-graph retrievers (see Graph retrievers).
- `mlsrag/runner.py`, `mlsrag/cli.py`: the commands below. `subsets/`: fixed query-id lists (seed 0):
  `practice-smoke20`, `practice-20pct` (70), `dev-20pct` (25), `practice-all`, `dev-all`.

## The five interfaces

`ASR.transcribe(audio_path) -> str` (`CachedASR`, `WhisperASR`, `CourseASR`),
`Encoder.encode(text) -> vector` (`GatewayEncoder`: gte-small or bge-small),
`Retriever.search(vec, k) -> [(record_id, score)]` (`ExactRetriever`: enc_a or enc_b; `BM25Retriever` sets
`uses_question = True`, so its `search(query_vec, k, question)` ranks by the question text and ignores the
vector; the API refuses vector input without `input.question` for such retrievers),
`ContextBuilder.build(question, record_ids) -> prompt` (`SimpleContext`),
`Generator.generate(prompt) -> answer` (`GatewayGenerator`, the course system prompt, temperature 0.1).

## RunConfig fields

`name`; `asr` (cached | whisper1 | course); `asr_cache` (JSONL, relative to the checkout root);
`query_encoder` (gte-small | bge-small); `corpus_index` (enc_a | enc_b); `mismatch_policy`
(error | as-is | re-encode | translate: gte-small belongs to enc_a, bge-small to enc_b; see below); `retriever` (exact | topic | bm25 | grep); `k` (5);
`context_builder` (simple); `generator_model`; `judge_model`; `flow` (standard | grep-static | grep-agentic); `params` (free dict).
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

## Lexical retrieval (BM25, grep)

Configs `configs/bm25.yaml`, `configs/grep-static.yaml`, `configs/grep-agentic.yaml`.

- **BM25** (`retriever: bm25`, standard flow): `bm25s` over the record texts, tokens = lowercase
  `\w+` words minus English stop words. The index is pickled to `rag/.cache/bm25/<corpus hash>.pkl`;
  the first query's trace has `bm25.build_s` and `bm25.build_rss_mb` (growth of peak RSS while
  building; `load_s`/`load_rss_mb` when the cache was used) and `bm25.index_mb` (pickle size).
  Only records with a positive score are
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
  the ranking for the metrics is the kept ids, then the rest of the scan order. Caps: 3 scans
  and 6 LLM calls per query (it makes at most 4).

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

### Two-level table of contents (`toc`) and RAPTOR-lite (`raptor`)

Both pipelines share the `topics` stages and folder (same config file, `configs/topics-k256.yaml`),
so the clusters and labels are computed once.

```bash
mlsrag preprocess run --pipeline toc    --config rag/extensions/platform/configs/topics-k256.yaml  # 16 LLM calls
mlsrag preprocess run --pipeline raptor --config rag/extensions/platform/configs/topics-k256.yaml  # 256 LLM calls
mlsrag matrix --matrix rag/extensions/platform/configs/matrix-toc-raptor-smoke.yaml --no-generate
```

**TOC** (`retriever: toc`, `params: {index, sections: 2, m: 4, route: centroid | llm, route_model}`).
Stage `section` runs k-means (n=16) on the 256 topic centroids and names each section from its
topics' labels (one LLM call per section, ledger); `toc_index` writes `toc.json` (sections →
topics → record ids) and both centroid files. With `route: centroid` the query takes the
`sections` nearest section centroids, then the `m` nearest topic centroids among their topics,
then searches those topics' members exactly (no LLM). With `route: llm` one chat call picks up to
`sections` sections from the 16 titles and a second picks up to `m` topics from those sections'
labels (Llama 3.3 by default; an unusable reply falls back to the centroids and counts
`toc.llm_fallbacks`). The retriever sets `uses_question = True`, so the flow passes it the transcript. Trace:
span `toc.route`, counters `toc.sections_probed`, `toc.topics_probed`, `toc.llm_calls`,
`toc.candidates_scanned`; `mlsrag estimate` counts the 2 routing calls per query.

**RAPTOR-lite** (`retriever: raptor-lite`, `params: {index, summary_as_evidence: false, expand_top: 3}`).
One level of RAPTOR (Sarthi et al., ICLR 2024): stage `summarise` writes one summary (at most 120
words) per cluster from its 8 records nearest the centroid (first 400 characters each; ledger,
prompt `v1`), and `summary_index` embeds them with gte-small as passages (`summaries.npy`,
`summaries.jsonl`). A query searches the 24,000 records and 256 summaries together, exactly. A
summary that ranks is replaced by its cluster's `expand_top` members most similar to the query
(records already listed are skipped), or, with `summary_as_evidence: true`, kept as the id
`summary:<cluster>`, which needs `context_builder: with-summaries` to put its text in the prompt
(it never counts as relevant in the retrieval metrics). Trace: span `raptor.search`, counters
`raptor.summary_hits`, `raptor.candidates_scanned`. No LLM calls at query time.

## Graph retrievers (plan 008)

Two graph strategies, both ordering ties by enc_a cosine with the query (so `corpus_index: enc_a`).
A retriever whose class sets `uses_question = True` is also given `question=<transcript>` by the
standard flow; the others are called exactly as before.

- **Lazy graph** (`retriever: lazy-graph`, no LLM at index time, after LazyGraphRAG). Pipeline
  `lazy-graph`: `select` → `phrases` (spaCy `en_core_web_sm`: noun chunks and named entities per
  record, lowercased lemmas without stop words; numeric entity types dropped; resumable per record)
  → `lazy_graph` (nodes = phrases in ≥ `min_records` records, edges = co-occurrence in a record,
  Leiden communities; `graph.json`, `edges.npy`, `stats.json`) → `lazy_index` (`index.json`).
  Search: the question's normalised 1–5-word runs that are nodes; every record holding one scores
  the sum of 1/degree of its matched nodes; ties (`params.rerank_dense`, default true) by cosine.
  Counters `lazy.nodes_matched`, `lazy.candidates`, `lazy.community_hits` (communities are recorded,
  not used for ranking: they matter for corpus-wide questions, which we do not have).
- **LLM knowledge graph** (`retriever: kg`). Pipeline `kg`: `select` → `kg_extract` (one call per
  record, prompt `v1`: entities with types, relations as triples, JSON; through the ledger; at most
  600 attempts over all runs, failures included, and at most 600 selected records) → `kg_index`
  (entities resolved by the same lowercase + lemma normalisation only, so "Apple" and "Apple Inc."
  stay apart). Search: question entities by `params.question_entities: spacy` (same phrase matching,
  free) or `llm` (one call per query, `params.question_model`, default the generator model; counted
  by `mlsrag estimate`), then 1-hop neighbours; records ranked by direct matches, then 1-hop matches,
  then cosine. Counters `kg.entities_matched`, `kg.entities_1hop`, `kg.candidates`.

```bash
mlsrag preprocess run --pipeline lazy-graph --config rag/extensions/platform/configs/lazy-graph.yaml
mlsrag preprocess run --pipeline kg --config rag/extensions/platform/configs/kg.yaml   # ≤ 600 LLM calls
```

### Record subsets (`params.record_subset`)

The KG is built on 600 records only (`configs/kg.yaml`: every record in `qrels_dev.tsv` plus 275
random distractors, seed 0). To compare it fairly, every other retriever must see the same 600
records: `params.record_subset: <file with one record id per line>` makes the flow ask the
retriever for every candidate and keep only ids in the file (counters `subset.candidates_in`,
`subset.candidates_kept`), and the metrics count only relevant records inside it. It is refused on
practice subsets (the selection is built around the dev qrels). The file for the KG selection is
`subsets/records-dev-relevant-d275.txt`, written from the `select` stage's `records.jsonl`.
Restricted numbers are only comparable with other restricted numbers: with 600 instead of 24,000
records, flat dense Recall@10 on `dev-20pct` goes from 0.597 to 0.823.

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

## Run with Docker

One image (`rag/extensions/platform/Dockerfile`, about 2.1 GB, CPU only) holds the API, the CLI and the built web client. The data, the model, `rag/.cache/`, the results and `rag/.env` stay on your disk and are mounted (`compose.yaml`). From the checkout root:

```bash
git clone <this repo> && cd <it> && git lfs install && git lfs pull --include="rag-assets.zip"
unzip rag-assets.zip -d ~/mls-rag-assets && python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
cp rag/.env.example rag/.env    # then put your ELM key in rag/.env
mkdir -p rag/.cache             # so the mount is owned by you, not root
docker compose up               # API + client at http://localhost:8000 (first run builds the image)
docker compose run --rm cli mlsrag run --config baseline --subset practice-smoke20 --no-generate
```

`rag/.env` is mounted read-only, never copied into the image. Containers run as uid/gid 1000 (set `UID`/`GID` to change it) so written files stay yours. Set `ADMIN_TOKEN` in your shell to enable the admin endpoints. `.git` is not in the image, so `/api/health` and run metadata show `git_commit: null`. The course ASR needs a GPU and is not in the image; `whisper1` and `cached` work.
