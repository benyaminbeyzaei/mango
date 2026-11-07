"""Prompts for the grep flows. Fixed: do not tune them on the dev split. VERSION enters the cache key."""

VERSION = "grep_v1"

PATTERNS = """You search a collection of short forum answers about personal finance with grep.
Write 3 to 8 search patterns for the question below. Each pattern is a word or a short phrase
that a relevant answer is likely to contain, matched case-insensitively. Use plain text; avoid
regex symbols. Reply with only a JSON list of strings.

Question: {question}"""

REFINE = """You search a collection of short forum answers about personal finance with grep.
Patterns are matched case-insensitively; records are ranked by how many patterns they match.

Question: {question}
Current patterns: {patterns}

Top hits (first 200 characters each):
{hits}

If these hits answer the question well, reply {{"stop": true}}.
Otherwise reply {{"patterns": [...]}} with 3 to 8 new patterns (plain words or short phrases).
Reply with only the JSON object."""

CLEANUP = """Question: {question}

Candidate forum answers (first 300 characters each):
{hits}

Return the ids of the candidates worth keeping to answer the question, most useful first,
at most {k}. Reply with only a JSON list of ids."""
