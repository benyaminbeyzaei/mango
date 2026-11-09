"""RAPTOR-tree stages (plan 028): soft cluster membership, one summary per cluster from all its
members (level 1), one summary per TOC section from its topics' summaries (level 2), and one
index over both levels for SummaryRetriever. Uses the `topics` clusters and `toc` sections."""
import json

import numpy as np

from .cluster import load_vectors
from .ledger import run_items
from .summarise import parse_summary

PROMPT_VERSION = "v1"
RECORDS_V1 = (
    "Below are records from one cluster of a document collection.\n\n{records}\n\n"
    "Summarise in at most {words} words what these records are about: the main subjects, names, "
    "places, dates and facts. Reply with the summary only."
)
SUMMARIES_V1 = (
    "Below are summaries of {what} of a document collection.\n\n{summaries}\n\n"
    "Combine them into one summary of at most {words} words: the main subjects, names, places, "
    "dates and facts. Reply with the summary only."
)
RETRY = "\n\n(Retry {attempt}: the previous reply was not usable. Reply with the summary only.)"


def tokens(text) -> int:
    """Rough token count (about 4 characters per token for English text)."""
    return len(text) // 4 + 1


def records_prompt(texts, attempt=0, words=150) -> str:
    prompt = RECORDS_V1.format(records="\n".join(f"- {t}" for t in texts), words=words)
    return prompt + (RETRY.format(attempt=attempt) if attempt else "")


def summaries_prompt(summaries, what, attempt=0, words=150) -> str:
    prompt = SUMMARIES_V1.format(what=what, summaries="\n".join(f"- {s}" for s in summaries), words=words)
    return prompt + (RETRY.format(attempt=attempt) if attempt else "")


def split(texts, cap) -> list:
    """Consecutive parts of `texts`, each under `cap` tokens (a single longer text is its own part)."""
    parts, size = [[]], 0
    for t in texts:
        if parts[-1] and size + tokens(t) > cap:
            parts.append([])
            size = 0
        parts[-1].append(t)
        size += tokens(t)
    return parts


def load_texts(select_dir) -> dict:
    texts = {}
    for line in (select_dir / "records.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        texts[r["record_id"]] = r["text"]
    return texts


def soft_members_run(inputs, config, ctx):
    """Each record also joins its second-nearest centroid when cos2 >= cos1 - delta; delta is the
    `share` quantile of cos1 - cos2, so about that share of records gets a second cluster."""
    clusters = json.loads((inputs["cluster"] / "clusters.json").read_text(encoding="utf-8"))
    centroids = np.load(inputs["cluster"] / "centroids.npy")
    ids = [r for c in clusters for r in c["members"]]
    primary = np.array([c["cluster"] for c in clusters for _ in c["members"]])
    rows = np.arange(len(ids))
    sims = load_vectors(ids) @ centroids.T
    cos1 = sims[rows, primary].copy()
    sims[rows, primary] = -np.inf
    second = sims.argmax(axis=1)
    gap = cos1 - sims[rows, second]
    delta = float(np.quantile(gap, config["share"]))
    soft = {c["cluster"]: [] for c in clusters}
    for i in np.flatnonzero(gap <= delta):
        soft[int(second[i])].append(ids[i])
    out = [{"cluster": c["cluster"], "members": c["members"] + soft[c["cluster"]], "soft": len(soft[c["cluster"]])}
           for c in clusters]
    (ctx.out_dir / "members.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    sizes = [len(c["members"]) for c in out]
    return {"delta": delta, "records": len(ids), "records_with_second": int((gap <= delta).sum()),
            "min_members": min(sizes), "mean_members": sum(sizes) / len(sizes), "max_members": max(sizes)}


def summarise_full_run(inputs, config, ctx):
    """One summary per cluster from the full text of all its members; a cluster over
    `max_input_tokens` is summarised in parts, then one call combines the part summaries."""
    texts = load_texts(inputs["select"])
    clusters = json.loads((inputs["soft_members"] / "members.json").read_text(encoding="utf-8"))
    words, cap, model = config.get("words", 150), config.get("max_input_tokens", 24000), config["model"]
    max_tokens = config.get("max_tokens", 300)
    first, n_parts = {}, {}
    for c in clusters:
        parts = split([texts[r] for r in c["members"]], cap)
        n_parts[c["cluster"]] = len(parts)
        if len(parts) == 1:
            first[c["cluster"]] = parts[0]
        else:
            first.update({f"{c['cluster']}.{j}": p for j, p in enumerate(parts)})
    split_clusters = [c for c, n in n_parts.items() if n > 1]
    calls = len(first) + len(split_clusters)
    input_tokens = sum(tokens(t) for p in first.values() for t in p)
    ctx.log(f"call plan: {len(clusters) - len(split_clusters)} clusters under {cap} tokens, {len(split_clusters)} "
            f"over ({len(first) - len(clusters) + len(split_clusters)} parts + {len(split_clusters)} combine calls); "
            f"{calls} calls, about {input_tokens} input tokens")
    if calls > config.get("max_calls", 450):
        raise RuntimeError(f"{calls} calls is over max_calls {config.get('max_calls', 450)}")

    done = run_items(first, ctx, model, lambda t, a: records_prompt(t, a, words),
                     lambda reply: parse_summary(reply, words), max_tokens=max_tokens)
    combine = {c: [done[f"{c}.{j}"] for j in range(n_parts[c])] for c in split_clusters}
    combined = run_items(combine, ctx, model, lambda s, a: summaries_prompt(s, "parts of one cluster", a, words),
                         lambda reply: parse_summary(reply, words), max_tokens=max_tokens)
    with (ctx.out_dir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters:
            k = c["cluster"]
            summary = combined[k] if k in combined else done[k]
            f.write(json.dumps({"cluster": k, "summary": summary, "parts": n_parts[k]}, ensure_ascii=False) + "\n")
    return {"clusters": len(clusters), "split_clusters": len(split_clusters), "calls_planned": calls,
            "input_tokens_estimate": input_tokens}


def section_summarise_run(inputs, config, ctx):
    """One summary per TOC section from its topics' level-1 summaries."""
    sections = json.loads((inputs["section"] / "sections.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (inputs["summarise_full"] / "summaries.jsonl").open(encoding="utf-8")]
    summary = {r["cluster"]: r["summary"] for r in rows}
    words = config.get("words", 150)
    items = {s["section"]: [summary[t] for t in s["topics"]] for s in sections}
    done = run_items(items, ctx, config["model"],
                     lambda s, a: summaries_prompt(s, "the topics in one section", a, words),
                     lambda reply: parse_summary(reply, words), max_tokens=config.get("max_tokens", 300))
    with (ctx.out_dir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps({"section": s["section"], "summary": done[s["section"]]}, ensure_ascii=False) + "\n")
    return {"sections": len(sections)}


def tree_index_run(inputs, config, ctx):
    """summaries.npy + summaries.jsonl as summary_index writes them, level-1 rows first; each row
    has an `id` (t<cluster> or s<section>), its `level` and its members (a section: all its topics' members)."""
    members = {c["cluster"]: c["members"]
               for c in json.loads((inputs["soft_members"] / "members.json").read_text(encoding="utf-8"))}
    sections = json.loads((inputs["section"] / "sections.json").read_text(encoding="utf-8"))
    level1 = [json.loads(line) for line in (inputs["summarise_full"] / "summaries.jsonl").open(encoding="utf-8")]
    level2 = {json.loads(line)["section"]: json.loads(line)["summary"]
              for line in (inputs["section_summarise"] / "summaries.jsonl").open(encoding="utf-8")}
    rows = [{"id": f"t{r['cluster']}", "level": 1, "summary": r["summary"], "members": members[r["cluster"]]}
            for r in level1]
    for s in sections:
        union = list(dict.fromkeys(r for t in s["topics"] for r in members[t]))
        rows.append({"id": f"s{s['section']}", "level": 2, "summary": level2[s["section"]], "members": union})
    vectors = ctx.gateway.embed(config.get("encoder", "gte-small"), [r["summary"] for r in rows], role="passage")
    np.save(ctx.out_dir / "summaries.npy", vectors.astype(np.float32))
    with (ctx.out_dir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"summaries": len(rows), "level1": len(level1), "level2": len(sections), "dim": int(vectors.shape[1])}
