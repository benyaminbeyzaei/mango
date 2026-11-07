"""Label stage: one LLM call per cluster names its topic, through the gateway and a ledger."""
import hashlib
import json
import re
import time

from .ledger import Ledger

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
    done = ledger.done_ids()
    failures = {}
    for r in ledger.rows():
        if r["status"] == "failed":
            failures[r["item_id"]] = failures.get(r["item_id"], 0) + 1
    todo = [c for c in clusters if c["cluster"] not in done]
    ctx.log(f"label: {len(done)} clusters done, {len(todo)} to go")
    for i, c in enumerate(todo, start=1):
        prompt = build_prompt(c, texts, attempt=failures.get(c["cluster"], 0))
        input_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        before = sum(v["prompt_tokens"] + v["completion_tokens"] for v in ctx.trace.cost.values())
        start = time.perf_counter()
        try:
            reply = ctx.gateway.chat(config["model"], [{"role": "user", "content": prompt}], temperature=0.0)
            label = parse_label(reply)
        except Exception as err:  # recorded and retried on the next run
            ms = (time.perf_counter() - start) * 1000.0
            ledger.append(c["cluster"], input_hash, "failed", latency_ms=ms, error=f"{type(err).__name__}: {err}")
            ctx.log(f"[{i}/{len(todo)}] cluster {c['cluster']} FAILED: {err}")
            continue
        tokens = sum(v["prompt_tokens"] + v["completion_tokens"] for v in ctx.trace.cost.values()) - before
        ms = (time.perf_counter() - start) * 1000.0
        ledger.append(c["cluster"], input_hash, "ok", output=label, tokens=tokens, latency_ms=ms)
        ctx.log(f"[{i}/{len(todo)}] cluster {c['cluster']}: {label['topic']}")

    ok = ledger.ok_rows()
    failed = len(clusters) - len(ok)
    if failed:
        raise RuntimeError(f"{failed} clusters failed to label; re-run to retry them")
    with (ctx.out_dir / "labels.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps({"cluster": c["cluster"], **ok[c["cluster"]]["output"]}, ensure_ascii=False) + "\n")
    rows = ledger.rows()
    return {"clusters": len(clusters), "labelled_this_run": len(todo), "ledger_lines": len(rows),
            "failed_attempts": sum(r["status"] == "failed" for r in rows),
            "ledger_tokens": sum(r["tokens"] for r in ok.values())}  # over all runs, not just this one
