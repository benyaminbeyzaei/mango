"""Lazy graph (no LLM at index time, after LazyGraphRAG).

`lazy_graph`: nodes are phrases found in at least `min_records` records (default 2), edges join
two phrases that occur in the same record (weight = number of such records), communities come
from Leiden (modularity, seed 0). Writes graph.json (nodes), edges.npy ([a, b, weight] rows)
and stats.json.
`lazy_index`: index.json that LazyGraphRetriever loads: per phrase its degree, its community and
the records it occurs in.
"""
import json
from collections import Counter

import numpy as np


def read_phrases(path) -> list:
    return [json.loads(line) for line in (path / "phrases.jsonl").open(encoding="utf-8")]


def run_graph(inputs, config, ctx):
    import igraph
    import leidenalg

    rows = read_phrases(inputs["phrases"])
    df = Counter(p for r in rows for p in r["phrases"])
    nodes = sorted(p for p, c in df.items() if c >= config.get("min_records", 2))
    node_id = {p: i for i, p in enumerate(nodes)}

    n = len(nodes)
    pairs = []  # each co-occurring pair a < b encoded as a * n + b
    for r in rows:
        ids = np.array(sorted(node_id[p] for p in r["phrases"] if p in node_id), dtype=np.int64)
        a, b = np.triu_indices(len(ids), k=1)
        pairs.append(ids[a] * n + ids[b])
    codes, weights = np.unique(np.concatenate(pairs), return_counts=True)
    edges = np.stack([codes // n, codes % n, weights], axis=1)
    np.save(ctx.out_dir / "edges.npy", edges)

    g = igraph.Graph(n=n, edges=edges[:, :2].tolist())
    g.es["weight"] = edges[:, 2].tolist()
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=0)
    degree = g.degree()
    graph = {"nodes": [{"phrase": p, "records": df[p], "degree": degree[i], "community": part.membership[i]}
                       for i, p in enumerate(nodes)]}
    (ctx.out_dir / "graph.json").write_text(json.dumps(graph) + "\n", encoding="utf-8")

    sizes = sorted(Counter(part.membership).values(), reverse=True)
    stats = {"records": len(rows), "phrases": len(df), "nodes": n, "edges": len(edges),
             "communities": len(sizes), "largest_communities": sizes[:10],
             "modularity": part.modularity, "mean_degree": float(np.mean(degree)) if n else 0.0}
    (ctx.out_dir / "stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return stats


def run_index(inputs, config, ctx):
    graph = json.loads((inputs["lazy_graph"] / "graph.json").read_text(encoding="utf-8"))
    nodes = {nd["phrase"]: {"degree": nd["degree"], "community": nd["community"], "records": []}
             for nd in graph["nodes"]}
    for r in read_phrases(inputs["phrases"]):
        for p in r["phrases"]:
            if p in nodes:
                nodes[p]["records"].append(r["record_id"])
    (ctx.out_dir / "index.json").write_text(json.dumps({"nodes": nodes}) + "\n", encoding="utf-8")
    return {"nodes": len(nodes), "postings": sum(len(v["records"]) for v in nodes.values())}
