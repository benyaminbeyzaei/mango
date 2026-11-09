"""Label stage: one LLM call per cluster names its topic, through the gateway and a ledger."""
import json
import re

from .ledger import Ledger, run_items

PROMPT_VERSION = "v1"
MAX_CLUSTERS = 512  # plan 002: never label more than this many clusters in one run
PROMPT_V1 = (
    "Below are the most frequent words and a few example records from one cluster of a "
    "document collection.\n\nFrequent words: {words}\n\nExample records:\n{examples}\n\n"
    "Name the cluster's topic in at most six words and describe it in one sentence. "
    'Reply with JSON only: {{"topic": "...", "summary": "..."}}'
)


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
    todo = len(clusters) - len(ledger.done_ids())
    labels = run_items({c["cluster"]: c for c in clusters}, ctx, config["model"],
                       lambda c, attempt: build_prompt(c, texts, attempt), parse_label)
    with (ctx.out_dir / "labels.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps({"cluster": c["cluster"], **labels[c["cluster"]]}, ensure_ascii=False) + "\n")
    rows = ledger.rows()
    return {"clusters": len(clusters), "labelled_this_run": todo, "ledger_lines": len(rows),
            "failed_attempts": sum(r["status"] == "failed" for r in rows),
            "ledger_tokens": sum(r["tokens"] for r in ledger.ok_rows().values())}  # over all runs
