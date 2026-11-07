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
