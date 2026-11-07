"""Grep: case-insensitive regex scan over all record texts, held in memory."""
import re


class GrepIndex:
    def __init__(self, data):
        self.records = data.records
        self.ids = list(data.records)
        self.texts = [data.records[rid] for rid in self.ids]

    def scan(self, patterns, limit):
        """Rank records by patterns matched, then total matches, then shorter text first.

        Returns (hits, n_matched_records, n_invalid_patterns), hits = [(record_id, n_patterns, n_matches)].
        """
        compiled, invalid = [], 0
        for p in patterns:
            try:
                c = re.compile(p, re.IGNORECASE)
            except re.error:
                invalid += 1
                continue
            if c.search(""):  # matches the empty string, so it would match every record
                invalid += 1
            else:
                compiled.append(c)
        rows = []
        for i, text in enumerate(self.texts):
            counts = [len(c.findall(text)) for c in compiled]
            n_patterns = sum(1 for n in counts if n)
            if n_patterns:
                rows.append((-n_patterns, -sum(counts), len(text), i))
        rows.sort()
        hits = [(self.ids[i], -neg_p, -neg_m) for neg_p, neg_m, _, i in rows[:limit]]
        return hits, len(rows), invalid
