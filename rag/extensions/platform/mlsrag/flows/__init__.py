"""Flows: the order in which stages run for one query."""
from . import grep_agentic, grep_feedback, grep_static, standard

FLOWS = {
    "standard": standard.run_query,
    "grep-static": grep_static.run_query,
    "grep-agentic": grep_agentic.run_query,
    "grep-feedback": grep_feedback.run_query,
}


def get_flow(name):
    if name not in FLOWS:
        raise ValueError(f"unknown flow {name!r}; known: {sorted(FLOWS)}")
    return FLOWS[name]
