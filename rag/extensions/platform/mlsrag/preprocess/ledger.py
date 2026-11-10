"""Per-item ledger for LLM stages: one JSONL line per attempt, so a stopped run resumes."""
import json
from pathlib import Path


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    def rows(self) -> list:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # empty, or cut off by an interrupt: that item is not done
                pass
        return rows

    def ok_rows(self) -> dict:
        """{item_id: row} for items whose latest attempt succeeded."""
        latest = {r["item_id"]: r for r in self.rows()}
        return {i: r for i, r in latest.items() if r["status"] == "ok"}

    def done_ids(self) -> set:
        return set(self.ok_rows())

    def append(self, item_id, input_hash, status, output=None, tokens=0, latency_ms=0.0, error=None):
        row = {
            "item_id": item_id, "input_hash": input_hash, "status": status, "output": output,
            "tokens": tokens, "latency_ms": latency_ms, "error": error,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        start = "\n" if text and not text.endswith("\n") else ""  # never append onto a cut-off line
        with self.path.open("a", encoding="utf-8") as f:  # one write per item, flushed on close
            f.write(start + json.dumps(row, ensure_ascii=False) + "\n")


def ask(ctx, ledger, items, calls):
    """Before an LLM stage's first call: log the items and calls, then let ctx.confirm stop the run."""
    if not calls:
        return
    rows = ledger.rows()
    fresh = [r["latency_ms"] for r in rows if r["tokens"]]  # calls answered from the cache cost 0 tokens
    pace = f", mean {sum(fresh) / len(fresh) / 1000:.1f}s per call so far" if fresh else ""
    message = (f"{items} items, {len(ledger.done_ids())} done after {len(rows)} calls; "
               f"{calls} ELM calls to make{pace}")
    ctx.log(message)
    if getattr(ctx, "confirm", None):
        ctx.confirm(message)


class ItemsFailed(RuntimeError):
    """Some items are still not `ok` after run_items; re-running retries them."""


def tokens_so_far(trace) -> dict:
    return {k: sum(v[k] for v in trace.cost.values()) for k in ("prompt_tokens", "completion_tokens")}


def run_items(items, ctx, model, build_prompt, parse, max_tokens=None, system=None, temperature=0.0,
              retry_tag=False) -> dict:
    """One chat call per item not yet `ok` in ctx.out_dir/ledger.jsonl; returns {item_id: output}.

    `items`: {item_id: data}. `build_prompt(data, attempt)` makes the prompt (attempt > 0 on a
    retry, so the gateway does not return the cached bad reply); `parse(reply)` raises on a bad one.
    A failure is recorded and the item retried on the next run; if any item is not `ok` at the end,
    ItemsFailed. `system`, if given, goes before the prompt as a system message. With `retry_tag`, a
    retry keeps the prompt and passes tag=retry<attempt> instead, so the gateway cache key still changes.
    """
    import hashlib
    import time

    ledger = Ledger(ctx.out_dir / "ledger.jsonl")
    done = ledger.done_ids()
    attempts = {}
    for r in ledger.rows():
        if r["status"] == "failed":
            attempts[r["item_id"]] = attempts.get(r["item_id"], 0) + 1
    todo = [i for i in items if i not in done]
    ctx.log(f"{len(done)} items done, {len(todo)} to go")
    ask(ctx, ledger, len(items), len(todo))
    for n, item_id in enumerate(todo, start=1):
        prompt = build_prompt(items[item_id], attempts.get(item_id, 0))
        input_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        before = tokens_so_far(ctx.trace)
        start = time.perf_counter()
        try:
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            tag = {"tag": f"retry{attempts.get(item_id, 0)}"} if retry_tag and attempts.get(item_id) else {}
            reply = ctx.gateway.chat(model, messages, temperature=temperature, max_tokens=max_tokens, **tag)
            output, status, error = parse(reply), "ok", None
        except Exception as err:  # recorded and retried on the next run
            output, status, error = None, "failed", f"{type(err).__name__}: {err}"
        ms = (time.perf_counter() - start) * 1000.0
        after = tokens_so_far(ctx.trace)
        spent = sum(after[k] - before[k] for k in after)
        ledger.append(item_id, input_hash, status, output=output, tokens=spent,
                      latency_ms=ms, error=error)
        ctx.log(f"[{n}/{len(todo)}] {item_id} {error or 'ok'}")
    ok = ledger.ok_rows()
    failed = [i for i in items if i not in ok]
    if failed:
        raise ItemsFailed(f"{len(failed)} items failed; re-run to retry them")
    return {i: ok[i]["output"] for i in items}
