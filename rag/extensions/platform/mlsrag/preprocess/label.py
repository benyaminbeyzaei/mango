"""Label stage: one LLM call per cluster names its topic, through the gateway and a ledger.

The config's `prompt` picks the prompt (default v1). v2 (plan 030) shows the records nearest the
centroid, then renames every topic whose name another topic shares until all names are distinct."""
import json
import re

import numpy as np

from .cluster import load_vectors
from .ledger import Ledger, run_items

PROMPT_VERSION = "v1"
MAX_CLUSTERS = 512  # plan 002: never label more than this many clusters in one run
PROMPT_V1 = (
    "Below are the most frequent words and a few example records from one cluster of a "
    "document collection.\n\nFrequent words: {words}\n\nExample records:\n{examples}\n\n"
    "Name the cluster's topic in at most six words and describe it in one sentence. "
    'Reply with JSON only: {{"topic": "...", "summary": "..."}}'
)
PROMPT_V2 = (
    "Below are the most frequent words and the records nearest the centre of one cluster of a "
    "document collection.\n\nFrequent words: {words}\n\nRecords:\n{examples}\n\n"
    "Name the cluster's topic in at most six words, specific enough to tell it apart from the other "
    "topics of the collection, and describe it in one sentence. "
    'Reply with JSON only: {{"topic": "...", "summary": "..."}}'
)
RENAME_V2 = (
    'Several topics of a document collection were given the same name: "{name}". Below are '
    "records from this topic and from the other topics with that name.\n\nThis topic:\n{mine}\n\n"
    "{others}\n\nNames already used by other topics (do not reuse any of them): {taken}\n\n"
    "Give this topic a new name of at most six words that tells it apart from the other topics, "
    "and describe it in one sentence. "
    'Reply with JSON only: {{"topic": "...", "summary": "..."}}'
)
RETRY = "\n\n(Retry {attempt}: the previous reply was not usable. Reply with the JSON object only.)"
EXAMPLES_V2 = 8  # records nearest the centroid, 300 characters each
RENAME_ROUNDS = 3


def bullets(texts) -> str:
    return "\n".join(f"- {t}" for t in texts)


def build_prompt_v2(words, examples, attempt=0) -> str:
    prompt = PROMPT_V2.format(words=", ".join(words), examples=bullets(examples))
    return prompt + (RETRY.format(attempt=attempt) if attempt else "")


def rename_prompt(name, mine, others, taken, attempt=0) -> str:
    other_text = "\n\n".join(f"Another topic named \"{name}\":\n{bullets(o)}" for o in others)
    prompt = RENAME_V2.format(name=name, mine=bullets(mine), others=other_text, taken="; ".join(taken))
    return prompt + (RETRY.format(attempt=attempt) if attempt else "")


def clashes(labels) -> list:
    """Groups of cluster ids whose topic names are equal (ignoring case and outer spaces)."""
    by_name = {}
    for c, lab in labels.items():
        by_name.setdefault(lab["topic"].strip().lower(), []).append(c)
    return [cs for cs in by_name.values() if len(cs) > 1]


def label_v2(clusters, texts, centroids, ctx, model) -> tuple:
    """({cluster: label}, number of rename calls): one call per cluster from its nearest records,
    then up to RENAME_ROUNDS rounds with one call per topic in a clash."""
    examples = {}
    for c in clusters:
        sims = load_vectors(c["members"]) @ centroids[c["cluster"]]
        nearest = [c["members"][i] for i in np.argsort(-sims, kind="stable")[:EXAMPLES_V2]]
        examples[c["cluster"]] = [texts[r][:300] for r in nearest]
    words = {c["cluster"]: c["top_words"] for c in clusters}
    labels = run_items({c: c for c in words}, ctx, model,
                       lambda c, attempt: build_prompt_v2(words[c], examples[c], attempt), parse_label)
    renames = 0
    for n in range(1, RENAME_ROUNDS + 1):
        groups = clashes(labels)
        if not groups:
            break
        items = {}
        for cs in groups:
            taken = sorted({lab["topic"] for c, lab in labels.items() if c not in cs})
            for c in cs:
                items[f"{c}.rename{n}"] = (c, [o for o in cs if o != c], taken)
        new = run_items(items, ctx, model, lambda it, attempt: rename_prompt(
            labels[it[0]]["topic"], examples[it[0]], [examples[o] for o in it[1]], it[2], attempt), parse_label)
        for key, (c, _, _) in items.items():
            labels[c] = new[key]
        renames += len(items)
    if clashes(labels):
        raise RuntimeError(f"names still shared after {RENAME_ROUNDS} rename rounds: {clashes(labels)}")
    return labels, renames


def build_prompt(cluster, texts, attempt=0) -> str:
    shortest = sorted(cluster["members"], key=lambda r: (len(texts[r]), r))[:5]
    examples = "\n".join(f"- {texts[r][:300]}" for r in shortest)
    prompt = PROMPT_V1.format(words=", ".join(cluster["top_words"]), examples=examples)
    if attempt:  # a retry: a different prompt, so the gateway does not return the cached bad reply
        prompt += f"\n\n(Retry {attempt}: the previous reply was not usable. Reply with the JSON object only.)"
    return prompt


def parse_label(reply) -> dict:
    """The first {...} in the reply, with non-empty `topic` and `summary` strings."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    obj = json.loads(match.group(0))
    if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in ("topic", "summary")):
        raise ValueError("topic and summary must be non-empty strings")
    return {k: obj[k].strip() for k in ("topic", "summary")}


def run(inputs, config, ctx):
    clusters = json.loads((inputs["cluster"] / "clusters.json").read_text(encoding="utf-8"))
    if len(clusters) > MAX_CLUSTERS:
        raise ValueError(f"{len(clusters)} clusters > {MAX_CLUSTERS}; this plan does not label that many")
    texts = {}
    for line in (inputs["select"] / "records.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        texts[r["record_id"]] = r["text"]

    ledger = Ledger(ctx.out_dir / "ledger.jsonl")
    done = ledger.done_ids()
    todo = sum(c["cluster"] not in done for c in clusters)
    renames = 0
    if config.get("prompt", "v1") == "v2":
        centroids = np.load(inputs["cluster"] / "centroids.npy")
        labels, renames = label_v2(clusters, texts, centroids, ctx, config["model"])
    else:
        labels = run_items({c["cluster"]: c for c in clusters}, ctx, config["model"],
                           lambda c, attempt: build_prompt(c, texts, attempt), parse_label)
    with (ctx.out_dir / "labels.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps({"cluster": c["cluster"], **labels[c["cluster"]]}, ensure_ascii=False) + "\n")
    rows = ledger.rows()
    return {"clusters": len(clusters), "labelled_this_run": todo, "rename_calls": renames, "ledger_lines": len(rows),
            "failed_attempts": sum(r["status"] == "failed" for r in rows),
            "ledger_tokens": sum(r["tokens"] for r in ledger.ok_rows().values())}  # over all runs
