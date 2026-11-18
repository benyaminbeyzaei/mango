import json

import pytest

from mlsrag.config import baseline
from mlsrag.runner import METRIC_KEYS, run


@pytest.mark.slow
def test_two_runs_in_one_minute_get_separate_folders(tmp_path):
    first, _ = run(baseline(), "practice-20", generate=False, results_dir=tmp_path)
    second, metrics = run(baseline(), "practice-20", generate=False, results_dir=tmp_path)
    if first.name[:13] == second.name[:13]:  # same minute
        assert second.name == first.name + "~2"
    for out in (first, second):
        assert {p.name for p in out.iterdir()} == {"config.json", "queries.jsonl", "metrics.json"}
    assert set(METRIC_KEYS) == set(json.loads((second / "metrics.json").read_text())["retrieval"])
    assert metrics["cost"]["elm_calls"] == 0 and metrics["run"]["subset"] == "practice-20"
