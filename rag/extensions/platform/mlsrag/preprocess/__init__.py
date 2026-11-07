"""Preprocessing pipelines. `topics`: select -> cluster -> label -> topic_index.
`lazy-graph`: select -> phrases -> lazy_graph -> lazy_index. `kg`: select -> kg_extract -> kg_index.
`toc` and `raptor` share the `topics` stages and folder, so they reuse them:
`toc`: select -> cluster -> label -> section -> toc_index; `raptor`: select -> cluster -> summarise -> summary_index."""
from pathlib import Path

import yaml

from . import cluster, kg, label, lazy_graph, phrases, section, select, summarise, summary_index, toc_index, topic_index
from .pipeline import Pipeline, Stage


def _topic_stages():
    return [
        Stage("select", select.run),
        Stage("cluster", cluster.run, inputs=["select"], version="2"),  # 2: deterministic top_words
        Stage("label", label.run, inputs=["select", "cluster"], prompt_version=label.PROMPT_VERSION),
    ]


def topics_pipeline(root=None) -> Pipeline:
    return Pipeline("topics", _topic_stages() + [
        Stage("topic_index", topic_index.run, inputs=["cluster", "label"]),
    ], root=root)


def lazy_graph_pipeline(root=None) -> Pipeline:
    return Pipeline("lazy-graph", [
        Stage("select", select.run),
        Stage("phrases", phrases.run, inputs=["select"]),
        Stage("lazy_graph", lazy_graph.run_graph, inputs=["phrases"]),
        Stage("lazy_index", lazy_graph.run_index, inputs=["phrases", "lazy_graph"]),
    ], root=root)


def kg_pipeline(root=None) -> Pipeline:
    return Pipeline("kg", [
        Stage("select", select.run),
        Stage("kg_extract", kg.run_extract, inputs=["select"], prompt_version=kg.PROMPT_VERSION),
        Stage("kg_index", kg.run_index, inputs=["kg_extract"]),
    ], root=root)


def toc_pipeline(root=None) -> Pipeline:
    return Pipeline("topics", _topic_stages() + [
        Stage("section", section.run, inputs=["cluster", "label"], prompt_version=section.PROMPT_VERSION),
        Stage("toc_index", toc_index.run, inputs=["cluster", "label", "section"]),
    ], root=root)


def raptor_pipeline(root=None) -> Pipeline:
    return Pipeline("topics", _topic_stages()[:2] + [  # no labels needed
        Stage("summarise", summarise.run, inputs=["select", "cluster"], prompt_version=summarise.PROMPT_VERSION),
        Stage("summary_index", summary_index.run, inputs=["cluster", "summarise"]),
    ], root=root)


PIPELINES = {"topics": topics_pipeline, "lazy-graph": lazy_graph_pipeline, "kg": kg_pipeline,
             "toc": toc_pipeline, "raptor": raptor_pipeline}


def load_pipeline_config(path) -> dict:
    """{stage name: stage config} from a YAML file."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
