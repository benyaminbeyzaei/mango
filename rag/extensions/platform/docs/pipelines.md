# Preprocessing pipelines (topics, toc, raptor)

Part of the [mlsrag README](../README.md).

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
mlsrag preprocess run    --pipeline topics --config rag/extensions/platform/configs/pipelines/topics-k256.yaml
mlsrag preprocess status --pipeline topics --config ...   # per stage: hash, ok|partial|missing, time, ELM calls/tokens
mlsrag preprocess clean  --pipeline topics --config ... --stage label
# selection flags (override the config's select section): --all | --ids FILE | --sample N --seed S
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

## Two-level table of contents (`toc`) and RAPTOR-lite (`raptor`)

Both pipelines share the `topics` stages and folder (same config file, `configs/pipelines/topics-k256.yaml`),
so the clusters and labels are computed once.

```bash
mlsrag preprocess run --pipeline toc    --config rag/extensions/platform/configs/pipelines/topics-k256.yaml  # 16 LLM calls
mlsrag preprocess run --pipeline raptor --config rag/extensions/platform/configs/pipelines/topics-k256.yaml  # 256 LLM calls
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-toc-raptor-smoke.yaml --no-generate
```

**TOC** (`retriever: toc`, `params: {index, sections: 2, m: 4, route: centroid | llm | llm-topics | llm-flat, route_model}`).
Stage `section` runs k-means (n=16) on the 256 topic centroids and names each section from its
topics' labels (one LLM call per section, ledger); `toc_index` writes `toc.json` (sections →
topics → record ids) and both centroid files. With `route: centroid` the query takes the
`sections` nearest section centroids, then the `m` nearest topic centroids among their topics,
then searches those topics' members exactly (no LLM). With `route: llm` one chat call picks up to
`sections` sections from the 16 titles and a second picks up to `m` topics from those sections'
labels (Llama 3.3 by default; an unusable reply falls back to the centroids and counts
`toc.llm_fallbacks`). `route: llm-topics` is the same two calls, but the section prompt lists the
names of each section's 16 topics under its title. `route: llm-flat` skips the sections: one call
shows all 256 topics (number, label, summary) and picks up to `m` of them; an unusable reply takes
the centroid route. The retriever sets `uses_question = True`, so the flow passes it the transcript. Trace:
span `toc.route`, counters `toc.sections_probed`, `toc.topics_probed`, `toc.llm_calls`,
`toc.candidates_scanned`; `mlsrag estimate` counts 2 routing calls per query (1 for `llm-flat`).

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
