"""Flows: the order in which stages run for one query."""
from . import standard

FLOWS = {"standard": standard.run_query}


def get_flow(name):
    if name not in FLOWS:
        raise ValueError(f"unknown flow {name!r}; known: {sorted(FLOWS)}")
    return FLOWS[name]
