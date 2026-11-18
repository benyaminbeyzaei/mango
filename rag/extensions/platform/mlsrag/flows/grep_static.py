"""Grep static: one LLM call turns the question into patterns, a regex scan ranks the records."""
import json

from ..prompts import grep_v1
from .standard import RANKED_DEPTH, finish

HELPER_TIMEOUT = 900  # seconds; Qwen with reasoning can think for over 3 min on one grep call
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # Qwen reasoning made each pattern call ~70-100 s


def parse_json(text):
    """The first JSON list or object in a model reply, or None."""
    text = text.split("</think>")[-1]
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("]" if text[start] == "[" else "}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def clean_patterns(value):
    """At most 8 distinct non-empty strings, or [] if the reply was not a list of strings."""
    if not isinstance(value, list):
        return []
    out = []
    for p in value:
        if isinstance(p, str) and p.strip() and p.strip() not in out:
            out.append(p.strip())
    return out[:8]


def ask(stages, config, trace, prompt):
    """One model call through the gateway, counted as a grep LLM call; returns parsed JSON or None."""
    model = config.params.get("grep_model", DEFAULT_MODEL)
    trace.count("grep.llm_calls")
    reply = stages.gateway.chat(
        model, [{"role": "user", "content": prompt}], temperature=0.0, tag=grep_v1.VERSION, timeout=HELPER_TIMEOUT
    )
    value = parse_json(reply)
    if value is None:
        trace.count("grep.parse_errors")
    return value


def scan(stages, trace, patterns):
    with trace.span("grep.scan", n_patterns=len(patterns)):
        hits, n_matched, n_invalid = stages.retriever.scan(patterns, limit=RANKED_DEPTH)
    trace.count("grep.iterations")
    trace.count("grep.patterns", len(patterns))
    trace.count("grep.matched_records", n_matched)
    trace.count("grep.invalid_patterns", n_invalid)
    return hits


def run_query(query_id, audio_path, stages, config, trace) -> dict:
    a = trace.artifacts
    with trace.span("asr"):
        a["transcript"] = stages.asr.transcribe(audio_path)
    with trace.span("retrieve"):  # the LLM call and the scan: what this flow spends to retrieve
        patterns = clean_patterns(ask(stages, config, trace, grep_v1.PATTERNS.format(question=a["transcript"])))
        a["patterns"] = [patterns]
        hits = scan(stages, trace, patterns)
    return finish(stages, config, trace, [rid for rid, _, _ in hits], [float(n) for _, n, _ in hits])
