"""Preprocessing pipelines. `topics`: select -> cluster -> label -> topic_index."""
from pathlib import Path

import yaml

from . import cluster, label, select, topic_index
from .pipeline import Pipeline, Stage


def topics_pipeline(root=None) -> Pipeline:
    return Pipeline("topics", [
        Stage("select", select.run),
        Stage("cluster", cluster.run, inputs=["select"], version="2"),  # 2: deterministic top_words
        Stage("label", label.run, inputs=["select", "cluster"], prompt_version=label.PROMPT_VERSION),
        Stage("topic_index", topic_index.run, inputs=["cluster", "label"]),
    ], root=root)


PIPELINES = {"topics": topics_pipeline}


def load_pipeline_config(path) -> dict:
    """{stage name: stage config} from a YAML file."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
