# Preprocessing pipelines (topics, toc, raptor, raptor-tree, raptor-full)

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
mlsrag run --config rag/extensions/platform/configs/topics-k256-m4.yaml --subset practice-20 --no-generate
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

## Two-level table of contents (`toc`), RAPTOR-lite (`raptor`) and RAPTOR-tree (`raptor-tree`)

Both pipelines share the `topics` stages and folder (same config file, `configs/pipelines/topics-k256.yaml`),
so the clusters and labels are computed once.

```bash
mlsrag preprocess run --pipeline toc    --config rag/extensions/platform/configs/pipelines/topics-k256.yaml  # 16 LLM calls
mlsrag preprocess run --pipeline raptor --config rag/extensions/platform/configs/pipelines/topics-k256.yaml  # 256 LLM calls
mlsrag preprocess run --pipeline raptor-tree --config rag/extensions/platform/configs/pipelines/topics-k256.yaml  # 423 + 16 LLM calls
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-toc-raptor-practice-20.yaml --no-generate
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
names of each section's topics (4-36) under its title. `route: llm-flat` skips the sections: one call
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

**RAPTOR-tree** (`retriever: raptor-tree`, same `params`; plan 028). Pipeline `raptor-tree` reuses
the `topics` clusters and the `toc` sections and fixes RAPTOR-lite's three gaps at about 6 % of
full RAPTOR's cost. `soft_members` (no LLM): a record also joins its second-nearest centroid when
`cos2 >= cos1 - delta`, with `delta` the `share` (0.15) quantile of `cos1 - cos2`, written to the
manifest. `summarise_full` (Llama 3.3, prompt `v1` of `preprocess/tree.py`): one summary (at most
150 words) per cluster from the full text of all its members; a cluster over `max_input_tokens`
(24 k, estimated as characters / 4) is summarised in parts and one more call combines them; the
stage logs its call plan first and stops above `max_calls` (450). `section_summarise`: one
summary per section from its topics' summaries. `tree_index` embeds the 256 + 16 summaries
(rows `t<cluster>` level 1, `s<section>` level 2). Search is the collapsed tree: one exact search
over the 24,000 records and 272 summaries; a level-1 hit expands to its cluster's members
(primary and soft), a level-2 hit to all its section's members, `expand_top` most similar to the
query each; with `summary_as_evidence: true` the id is `summary:t<cluster>` or `summary:s<section>`.
Extra counters `raptor.level1_hits`, `raptor.level2_hits`. Configs `raptor-tree-expand.yaml`,
`raptor-tree-evidence.yaml`.

**gpt-5-mini versions (plan 030).** `configs/pipelines/topics-k256-gpt5.yaml` is a copy of
`topics-k256.yaml` that changes only `label`, `section`, `summarise_full` and `section_summarise`
(model `gpt-5-mini`, a paid model named by Benyamin for these builds only; `max_tokens: 4000` leaves
room for its hidden reasoning). `select`, `cluster` and `soft_members` keep their hashes and are
reused; the Llama builds stay where they are. `label: {prompt: v2}` (selected in the config, so
`prompt_version` and the old hash stay unchanged) shows the top words and the 8 records nearest the
centroid (300 characters each), then renames every topic whose name another topic shares (ignoring
case): one call per topic in a clash, seeing the clashing topics' records and the names already
taken, at most 3 rounds, and the stage fails if a clash remains. Sections are named from the new
topic names with the unchanged section prompt; the section membership (k-means on the centroids)
is the same as before. The gateway sends models whose name starts with `gpt-5` no `temperature`,
`max_completion_tokens` instead of `max_tokens`, and `reasoning_effort: low` (in their cache key),
and logs `reasoning_tokens`; requests and cache keys of every other model are unchanged.
Configs: `toc-{centroid,llm,llm-topics,llm-flat}-m4-gpt5labels.yaml` (the route model stays
Llama-3.3) and `raptor-tree-gpt5-{expand,evidence}.yaml`.
Built on 2026-11-09 at concurrency 1: labels and sections 272 calls, 171 k prompt and 43 k completion
tokens (26 k of them reasoning), 19 min, US$0.13, no name clash so no rename call; RAPTOR-tree 439 calls,
4.68 M prompt and 144 k completion tokens (55 k reasoning), 44 min, US$1.46 (list prices $0.25 / $2.00
per M tokens; `results.py` prices only whisper, so run metrics show these builds as $0).

## Faithful RAPTOR (`raptor-full`, plan 031)

```bash
mlsrag preprocess run --pipeline raptor-full --config rag/extensions/platform/configs/pipelines/raptor-full.yaml
```

**RAPTOR-full** (`retriever: raptor-full`, same `params` as RAPTOR-lite). Pipeline `raptor-full`
(its own folder, `rag/.cache/preprocess/raptor-full/`): `select` → `raptor_build`
(`preprocess/raptor_full.py`), a port of the reference implementation
(`github.com/parthsarthi03/raptor`, MIT, commit `7da1d48`) onto our stage and ledger code.
Layer 0 is the selected records. Each layer: UMAP to 10 dimensions (cosine, `n_neighbors =
int(sqrt(n - 1))`), a Gaussian mixture with the number of components chosen by BIC over 1..49
(`random_state` 224 for the BIC loop, 0 for the final fit), soft membership (a node joins every
cluster with probability > 0.1); inside each global cluster of more than 11 nodes the same again
with UMAP `n_neighbors = 10` (local clusters); a cluster of 2+ nodes over 3,500 cl100k_base tokens
is clustered again. Every cluster is summarised with the paper's prompt (Table 11: system "You
are a Summarizing Text Portal", user "Write a summary of the following, including as many key
details as possible: {context}:"; the children's texts each on one line followed by a blank line;
`max_tokens` 100, temperature 1 as the reference leaves it), the summaries are embedded with
gte-small (`role=passage`) and become the next layer's nodes, until 11 or fewer nodes are left or
5 layers are built. Each layer's clusters are saved before its calls (`layer<n>_clusters.json`),
so a stopped build resumes from the ledger without re-clustering; before a layer's calls the
stage stops if an upper bound of the total cost (ledger tokens at the output price, plus this
layer's input tokens and 100 output tokens per call at list prices in the config) is over
`max_usd` (20). Rows `L<layer>.<j>` in `summaries.jsonl` carry `children` (node ids) and
`members` (every leaf record under the node, through all layers), so an expanded hit at any
layer reaches records. Search is the collapsed tree, as for RAPTOR-tree; counters
`raptor.level<n>_hits` per layer. Configs `raptor-full-expand.yaml`, `raptor-full-evidence.yaml`.

Build (2026-11-09, 4.1 h, 33 min of it layer-1 clustering): 24,000 records → 4,547 → 567 → 107 →
18 → 4 summary nodes (5 layers, mean 6.1 / 8.5 / 5.3 / 5.9 / 4.5 children, no node left out of
every cluster), 5,243 gpt-4o-mini calls, 5.28 M input + 0.52 M output tokens, $1.11 at list prices,
0 failures. 95 % of summaries stop at the 100-token cap mid-sentence (gpt-4o-mini writes longer
than gpt-3.5-turbo; the paper reports 131 tokens on average, so its runs must have used a larger
cap than the code's default), and an upper-layer summary often describes only its first children.
Summary hits per query in the ranked lists, layers 1 / 2 / 3 / 4 / 5: practice expand 20.54 / 2.07 /
0.41 / 0.06 / 0.01, evidence 27.11 / 2.83 / 0.53 / 0.07 / 0.01; test expand 20.95 / 2.30 / 0.42 /
0.03 / 0, evidence 28.96 / 3.15 / 0.58 / 0.05 / 0 (counters `raptor.level<n>_hits`).

Deviations from the reference, on purpose: (1) leaves are the 24,000 records, not 100-token
chunks (qrels are per record; 56.8 % of records exceed 100 tokens); (2) enc_a / gte-small vectors
instead of OpenAI ada-002, so queries, records and summaries share one space; (3) `gpt-4o-mini`
instead of `gpt-3.5-turbo` (not offered as a chat model on ELM); (4) UMAP gets `random_state` 224
for repeatable clusters (which makes UMAP single-threaded); (5) nodes are tracked by row index
instead of by vector equality (same result except for identical vectors; the corpus has one
duplicate pair); (6) a too-large cluster of 11 nodes or fewer, or one equal to its whole input,
is kept instead of clustered again (the reference would crash in UMAP or recurse forever);
(7) an API error is retried, not stored as the summary. Where the paper and the code differ,
only the system message does: we use the paper's (the code has "You are a helpful assistant.").
