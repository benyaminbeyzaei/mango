# Graph retrievers (plan 008)

Part of the [mlsrag README](../README.md).

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
  Extraction model: `meta-llama/Llama-3.3-70B-Instruct` (plan 018: ~4.4 s and ~520 tokens per
  record; Qwen3.5 took ~156 s and ~7,000 tokens because of its hidden reasoning). The budget is
  spent: 1 Qwen + 599 Llama attempts, 598 records extracted, 3,919 entities, 3,727 edges. The
  configs `kg-spacy-dev-sub600.yaml` and `kg-llm-dev-sub600.yaml` (question model Llama) run it on
  `dev` restricted to those 600 records.

```bash
mlsrag preprocess run --pipeline lazy-graph --config rag/extensions/platform/configs/pipelines/lazy-graph.yaml
mlsrag preprocess run --pipeline kg --config rag/extensions/platform/configs/pipelines/kg.yaml   # ≤ 600 LLM calls
```

## Record subsets (`params.record_subset`)

The KG is built on 600 records only (`configs/pipelines/kg.yaml`: every record in `qrels_dev.tsv` plus 275
random distractors, seed 0). To compare it fairly, every other retriever must see the same 600
records: `params.record_subset: <file with one record id per line>` makes the flow ask the
retriever for every candidate and keep only ids in the file (counters `subset.candidates_in`,
`subset.candidates_kept`), and the metrics count only relevant records inside it. It is refused on
practice subsets (the selection is built around the dev qrels). The file for the KG selection is
`subsets/records-dev-relevant-d275.txt`, written from the `select` stage's `records.jsonl`.
Restricted numbers are only comparable with other restricted numbers: with 600 instead of 24,000
records, flat dense Recall@10 on `dev-20pct` goes from 0.597 to 0.823.
