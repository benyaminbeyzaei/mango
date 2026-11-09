"""LLM knowledge graph on a small selection (after LightRAG/HippoRAG-style extraction).

`kg_extract`: one LLM call per selected record (entities with types, relations as triples),
through the ledger; never more than MAX_CALLS attempts in total, failures included.
`kg_index`: index.json for KGRetriever: per entity (lowercase + lemma, the same normalisation as
the phrases stage) its types, its records and its neighbours over the relations. Entity resolution
is that string match only: "Apple" and "Apple Inc." stay two entities.
"""
import json
import re

from .ledger import ItemsFailed, Ledger, run_items
from .phrases import load_nlp, normalise

PROMPT_VERSION = "v1"
MAX_CALLS = 600  # plan 008: the extraction budget, over all runs and all configs of this stage
PROMPT_V1 = (
    "Extract the entities and the relations between them from the text below.\n"
    "Entities: specific things a reader could look up (people, organisations, products, financial "
    "instruments, accounts, taxes, laws, places, finance concepts). Relations: facts the text states, "
    "as [subject, relation, object] where subject and object are entity names.\n"
    'Reply with JSON only: {{"entities": [{{"name": "...", "type": "..."}}], '
    '"relations": [["subject", "relation", "object"]]}}\n\nText:\n{text}'
)


def build_prompt(text, attempt=0) -> str:
    prompt = PROMPT_V1.format(text=text)
    if attempt:  # a retry: a different prompt, so the gateway does not return the cached bad reply
        prompt += f"\n\n(Retry {attempt}: the previous reply was not usable. Reply with the JSON object only.)"
    return prompt


def parse_graph(reply) -> dict:
    """The first {...} in the reply: entities with a string name, relations as 3 strings."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    obj = json.loads(match.group(0))
    if not (isinstance(obj.get("entities"), list) and isinstance(obj.get("relations"), list)):
        raise ValueError("reply needs `entities` and `relations` lists")
    entities = [{"name": e["name"].strip(), "type": str(e.get("type", "")).strip()}
                for e in obj.get("entities", []) if isinstance(e, dict) and isinstance(e.get("name"), str)]
    relations = [[s.strip() for s in t] for t in obj.get("relations", [])
                 if isinstance(t, list) and len(t) == 3 and all(isinstance(s, str) for s in t)]
    return {"entities": [e for e in entities if e["name"]], "relations": relations}


def run_extract(inputs, config, ctx):
    records = [json.loads(line) for line in (inputs["select"] / "records.jsonl").open(encoding="utf-8")]
    if len(records) > MAX_CALLS:
        raise ValueError(f"{len(records)} records selected > {MAX_CALLS}; this plan extracts at most {MAX_CALLS}")
    ledger = Ledger(ctx.out_dir / "ledger.jsonl")
    done = ledger.done_ids()
    todo = [r["record_id"] for r in records if r["record_id"] not in done]
    # every kg_extract output (other models or prompts too) shares the one budget
    spent = sum(len(Ledger(p).rows()) for p in ctx.out_dir.parent.glob("*/ledger.jsonl"))
    budget = max(MAX_CALLS - spent, 0)  # a negative slice would still run calls
    ctx.log(f"kg_extract: {budget} calls left")
    over_budget = set(todo[budget:])
    try:
        run_items({r["record_id"]: r["text"] for r in records if r["record_id"] not in over_budget}, ctx,
                  config["model"], build_prompt, parse_graph, usage=True)
    except ItemsFailed as err:  # failed records are retried on the next run while the budget lasts
        ctx.log(str(err))

    rows = ledger.rows()
    ok = ledger.ok_rows()
    fresh = [r for r in ok.values() if r["tokens"]]  # calls answered from the gateway cache cost 0 tokens
    n = max(len(fresh), 1)
    return {"records": len(records), "ok": len(ok), "failed_records": len(records) - len(ok),
            "ledger_lines": len(rows),
            "failed_attempt_tokens": sum(r["tokens"] for r in rows if r["status"] == "failed"),
            "prompt_tokens": sum(r["output"]["usage"]["prompt_tokens"] for r in ok.values()),
            "completion_tokens": sum(r["output"]["usage"]["completion_tokens"] for r in ok.values()),
            "mean_tokens_per_record": sum(r["tokens"] for r in fresh) / n,
            "mean_seconds_per_record": sum(r["latency_ms"] for r in fresh) / n / 1000.0}


def run_index(inputs, config, ctx):
    ok = Ledger(inputs["kg_extract"] / "ledger.jsonl").ok_rows()
    names = sorted({e["name"] for r in ok.values() for e in r["output"]["entities"]}
                   | {s for r in ok.values() for t in r["output"]["relations"] for s in (t[0], t[2])})
    norm = {name: normalise(doc) for name, doc in zip(names, load_nlp().pipe(names), strict=True)}

    entities = {}
    def entity(name):
        return entities.setdefault(norm[name], {"types": set(), "records": set(), "neighbours": set()})

    for rid, r in sorted(ok.items()):
        for e in r["output"]["entities"]:
            if norm[e["name"]]:
                entity(e["name"])["types"].add(e["type"])
                entity(e["name"])["records"].add(rid)
        for s, _, o in r["output"]["relations"]:
            if norm[s] and norm[o] and norm[s] != norm[o]:
                entity(s)["records"].add(rid)
                entity(o)["records"].add(rid)
                entity(s)["neighbours"].add(norm[o])
                entity(o)["neighbours"].add(norm[s])
    index = {"entities": {k: {f: sorted(v[f]) for f in v} for k, v in sorted(entities.items())}}
    (ctx.out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"records": len(ok), "entities": len(entities),
            "relations": sum(len(r["output"]["relations"]) for r in ok.values()),
            "edges": sum(len(v["neighbours"]) for v in entities.values()) // 2}
