"""Faithful RAPTOR: UMAP + Gaussian-mixture soft clustering, global then local, with
size re-clustering, summarised layer by layer with the paper's prompt.

Follows github.com/parthsarthi03/raptor (MIT licence, commit 7da1d48), raptor/cluster_utils.py,
cluster_tree_builder.py, tree_builder.py, utils.py, and Sarthi et al., ICLR 2024 (arXiv
2401.18059). Each constant below cites its source line. Deviations (docs/pipelines.md): the leaves
are the 24,000 records, vectors are enc_a/gte-small, the summariser is gpt-4o-mini, UMAP gets a
fixed random_state, nodes are tracked by index (the reference matches them by vector equality),
and a too-large cluster that cannot be split (11 nodes or fewer, or GMM gives it back whole) is
kept as it is, where the reference would crash or recurse forever.
"""
import json

import numpy as np

from .cluster import load_vectors
from .ledger import Ledger, run_items

RANDOM_SEED = 224  # cluster_utils.py:19, the BIC loop's random_state
DIM = 10  # cluster_utils.py:138 reduction_dimension
THRESHOLD = 0.1  # cluster_utils.py:139 soft-membership probability
MAX_CLUSTERS = 50  # cluster_utils.py:47; the BIC loop tries 1..49 (np.arange(1, 50))
LOCAL_NEIGHBORS = 10  # cluster_utils.py:38
MAX_LENGTH_IN_CLUSTER = 3500  # cluster_utils.py:136, in cl100k_base tokens (cluster_utils.py:137)
NUM_LAYERS = 5  # tree_builder.py:49
SUMMARIZATION_LENGTH = 100  # tree_builder.py:73, passed as max_tokens (cluster_tree_builder.py:71-74)
SYSTEM = "You are a Summarizing Text Portal"  # paper Table 11 (the code has "You are a helpful assistant.")
USER = "Write a summary of the following, including as many key details as possible: {context}:"  # paper Table 11
TEMPERATURE = 1.0  # the reference passes none, so the OpenAI default (1) applies


def reduce(x, n_neighbors, dim, seed):
    import umap  # slow to import (numba)

    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric="cosine",
                     random_state=seed).fit_transform(x)


def optimal_clusters(x) -> int:
    """get_optimal_clusters: the number of components, 1..min(50, n) - 1, with the lowest BIC."""
    from sklearn.mixture import GaussianMixture

    ks = np.arange(1, min(MAX_CLUSTERS, len(x)))
    bics = [GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit(x).bic(x) for k in ks]
    return int(ks[np.argmin(bics)])


def gmm_cluster(x, threshold=THRESHOLD):
    """GMM_cluster: ([labels of each row], n clusters); a row joins every cluster with probability > threshold."""
    from sklearn.mixture import GaussianMixture

    k = optimal_clusters(x)
    probs = GaussianMixture(n_components=k, random_state=0).fit(x).predict_proba(x)  # random_state=0 as in the reference
    return [np.flatnonzero(p > threshold) for p in probs], k


def perform_clustering(x, seed=RANDOM_SEED) -> list:
    """cluster_utils.perform_clustering: global UMAP (n_neighbors = sqrt(n - 1)) + GMM, then within each
    global cluster of more than DIM + 1 rows a local UMAP (n_neighbors = 10) + GMM. Returns the clusters
    as arrays of row indices, in label order."""
    n = len(x)
    global_labels, n_global = gmm_cluster(reduce(x, int((n - 1) ** 0.5), min(DIM, n - 2), seed))
    clusters = []
    for g in range(n_global):
        rows = np.array([i for i, labels in enumerate(global_labels) if g in labels], dtype=np.int64)
        if len(rows) == 0:
            continue
        if len(rows) <= DIM + 1:
            clusters.append(rows)
            continue
        local_labels, n_local = gmm_cluster(reduce(x[rows], LOCAL_NEIGHBORS, DIM, seed))
        for j in range(n_local):
            members = rows[[i for i, labels in enumerate(local_labels) if j in labels]]
            if len(members):
                clusters.append(members)
    return clusters


def raptor_clusters(x, tokens, seed=RANDOM_SEED) -> list:
    """RAPTOR_Clustering.perform_clustering: a cluster of 2+ nodes over MAX_LENGTH_IN_CLUSTER tokens is
    clustered again on its own nodes. `tokens[i]`: node i's text length."""
    tokens = np.asarray(tokens)
    out = []
    for c in perform_clustering(x, seed):
        small = len(c) == 1 or tokens[c].sum() <= MAX_LENGTH_IN_CLUSTER
        if small or len(c) <= DIM + 1 or len(c) == len(x):  # all of x: clustering again would repeat this call
            out.append(c)
        else:
            out.extend(c[s] for s in raptor_clusters(x[c], tokens[c], seed))
    return out


def get_text(texts) -> str:
    """utils.get_text: each text on one line, followed by a blank line."""
    return "".join(" ".join(t.splitlines()) + "\n\n" for t in texts)


def parse_reply(reply) -> str:
    if not reply or not reply.strip():
        raise ValueError("empty summary")
    return reply


def usd(prompt_tokens, completion_tokens, config) -> float:
    return (prompt_tokens * config["usd_per_m_input"] + completion_tokens * config["usd_per_m_output"]) / 1e6


def build_run(inputs, config, ctx):
    """Layer 0 = the selected records. Each layer: cluster the current nodes, summarise each cluster,
    embed the summaries (gte-small, passage), until DIM + 1 or fewer nodes are left or NUM_LAYERS layers.
    Writes summaries.jsonl ({id, level, summary, children, members}: members = the leaf records under it)
    and summaries.npy, the format SummaryRetriever loads. A layer's clusters are saved before its calls,
    so a resumed run reuses them; before a layer's calls the stage checks an upper bound of the total
    cost against `max_usd`."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    records = [json.loads(line) for line in (inputs["select"] / "records.jsonl").open(encoding="utf-8")]
    ids = [r["record_id"] for r in records]
    texts = [r["text"] for r in records]
    vectors = load_vectors(ids)
    lengths = [len(enc.encode(t)) for t in texts]
    node_ids = list(ids)  # every node, leaves first, then each layer's summaries
    members = [[rid] for rid in ids]  # leaf records under each node
    current = list(range(len(ids)))  # node indices of the layer being clustered
    rows, layers = [], [{"layer": 0, "nodes": len(ids)}]
    ledger = Ledger(ctx.out_dir / "ledger.jsonl")
    for layer in range(1, NUM_LAYERS + 1):
        if len(current) <= DIM + 1:  # cluster_tree_builder.py:95
            break
        path = ctx.out_dir / f"layer{layer}_clusters.json"
        if path.exists():
            clusters = [np.array(c, dtype=np.int64) for c in json.loads(path.read_text(encoding="utf-8"))]
        else:
            clusters = raptor_clusters(vectors[current], [lengths[i] for i in current], config.get("seed", RANDOM_SEED))
            path.write_text(json.dumps([c.tolist() for c in clusters]) + "\n", encoding="utf-8")
        items = {f"L{layer}.{j}": get_text(texts[current[i]] for i in c) for j, c in enumerate(clusters)}
        in_any = {int(i) for c in clusters for i in c}
        done = ledger.done_ids()
        todo = [k for k in items if k not in done]
        prompt_tokens = sum(len(enc.encode(SYSTEM + USER.format(context=items[k]))) + 10 for k in todo)
        spent = sum(r["tokens"] for r in ledger.rows()) * config["usd_per_m_output"] / 1e6  # upper bound
        bound = spent + usd(prompt_tokens, len(todo) * SUMMARIZATION_LENGTH, config)
        sizes = [len(c) for c in clusters]
        ctx.log(f"layer {layer}: {len(current)} nodes -> {len(clusters)} clusters (sizes {min(sizes)}-{max(sizes)}, "
                f"mean {np.mean(sizes):.1f}, {len(current) - len(in_any)} nodes in no cluster); {len(todo)} calls, "
                f"{prompt_tokens} input tokens; cost bound so far + this layer ${bound:.2f}")
        if layer == 1:  # estimate the upper layers: layer 1's node ratio and soft overlap, full-length summaries
            ratio, overlap = len(current) / len(clusters), sum(sizes) / len(current)
            n, rest_in, rest_calls = len(clusters), 0, 0
            for _ in range(NUM_LAYERS - 1):
                if n <= DIM + 1:
                    break
                rest_in += n * overlap * SUMMARIZATION_LENGTH
                n = max(1, round(n / ratio))
                rest_calls += n
            total = bound + usd(rest_in + 30 * rest_calls, rest_calls * SUMMARIZATION_LENGTH, config)
            ctx.log(f"upper layers (assuming {ratio:.1f} nodes per cluster, overlap {overlap:.2f}): about "
                    f"{rest_calls} calls, {rest_in:.0f} input tokens; whole tree about ${total:.2f}")
            if total > config["max_usd"]:
                raise RuntimeError(f"whole-tree estimate ${total:.2f} is over max_usd {config['max_usd']}")
        if bound > config["max_usd"]:
            raise RuntimeError(f"cost bound ${bound:.2f} is over max_usd {config['max_usd']}")
        summaries = run_items(items, ctx, config["model"], lambda context, attempt: USER.format(context=context),
                              parse_reply, max_tokens=SUMMARIZATION_LENGTH, system=SYSTEM, temperature=TEMPERATURE,
                              retry_tag=True)  # the prompt stays exactly the paper's on a retry
        new = [summaries[k] for k in items]
        new_vectors = ctx.gateway.embed(config.get("encoder", "gte-small"), new, role="passage").astype(np.float32)
        nxt = []
        for k, c, text in zip(items, clusters, new, strict=True):
            children = [current[i] for i in c]
            members.append(list(dict.fromkeys(r for ch in children for r in members[ch])))
            rows.append({"id": k, "level": layer, "summary": text, "children": [node_ids[ch] for ch in children],
                         "members": members[-1]})
            node_ids.append(k)
            texts.append(text)
            lengths.append(len(enc.encode(text)))
            nxt.append(len(texts) - 1)
        vectors = np.concatenate([vectors, new_vectors])
        layers.append({"layer": layer, "nodes": len(nxt), "orphans": len(current) - len(in_any),
                       "mean_children": float(np.mean(sizes))})
        current = nxt
    np.save(ctx.out_dir / "summaries.npy", vectors[len(ids):])
    with (ctx.out_dir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"records": len(ids), "summaries": len(rows), "layers": layers}
