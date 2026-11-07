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


def run_items(items, ctx, model, build_prompt, parse, max_tokens=None) -> dict:
    """One chat call per item not yet `ok` in ctx.out_dir/ledger.jsonl; returns {item_id: output}.

    `items`: {item_id: data}. `build_prompt(data, attempt)` makes the prompt (attempt > 0 on a
    retry, so the gateway does not return the cached bad reply); `parse(reply)` raises on a bad one.
    A failure is recorded and the item retried on the next run.
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
        before = sum(v["prompt_tokens"] + v["completion_tokens"] for v in ctx.trace.cost.values())
        start = time.perf_counter()
        try:
            reply = ctx.gateway.chat(model, [{"role": "user", "content": prompt}], temperature=0.0,
                                     max_tokens=max_tokens)
            output = parse(reply)
        except Exception as err:  # recorded and retried on the next run
            tokens = sum(v["prompt_tokens"] + v["completion_tokens"] for v in ctx.trace.cost.values()) - before
            ms = (time.perf_counter() - start) * 1000.0
            ledger.append(item_id, input_hash, "failed", tokens=tokens, latency_ms=ms,
                          error=f"{type(err).__name__}: {err}")
            ctx.log(f"[{n}/{len(todo)}] {item_id} FAILED: {err}")
            continue
        tokens = sum(v["prompt_tokens"] + v["completion_tokens"] for v in ctx.trace.cost.values()) - before
        ms = (time.perf_counter() - start) * 1000.0
        ledger.append(item_id, input_hash, "ok", output=output, tokens=tokens, latency_ms=ms)
        ctx.log(f"[{n}/{len(todo)}] {item_id} ok")
    ok = ledger.ok_rows()
    failed = [i for i in items if i not in ok]
    if failed:
        raise RuntimeError(f"{len(failed)} items failed; re-run to retry them")
    return {i: ok[i]["output"] for i in items}
