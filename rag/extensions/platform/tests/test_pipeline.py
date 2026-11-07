import json

import pytest

from mlsrag.preprocess import topics_pipeline
from mlsrag.preprocess.pipeline import Pipeline, Stage

calls = []


def make(inputs, config, ctx):
    calls.append("make")
    (ctx.out_dir / "n.json").write_text(json.dumps(config["n"]))
    return {"n": config["n"]}


def double(inputs, config, ctx):
    calls.append("double")
    n = json.loads((inputs["make"] / "n.json").read_text())
    if config.get("crash"):
        raise KeyboardInterrupt
    (ctx.out_dir / "out.json").write_text(json.dumps(n * config["factor"]))


def toy(root):
    return Pipeline("toy", [Stage("make", make), Stage("double", double, inputs=["make"])], root=root)


def test_rerun_skips_and_config_change_reruns_only_that_stage(tmp_path):
    calls.clear()
    p = toy(tmp_path)
    dirs = p.run({"make": {"n": 3}, "double": {"factor": 2}}, log=lambda *_: None)
    assert json.loads((dirs["double"] / "out.json").read_text()) == 6
    assert json.loads((dirs["make"] / "manifest.json").read_text())["status"] == "ok"
    p.run({"make": {"n": 3}, "double": {"factor": 2}}, log=lambda *_: None)
    assert calls == ["make", "double"]
    p.run({"make": {"n": 3}, "double": {"factor": 5}}, log=lambda *_: None)
    assert calls == ["make", "double", "double"]
    assert [r["status"] for r in p.status({"make": {"n": 3}, "double": {"factor": 5}})] == ["ok", "ok"]


def test_interrupted_stage_is_partial_and_not_done(tmp_path):
    p = toy(tmp_path)
    config = {"make": {"n": 1}, "double": {"factor": 2, "crash": True}}
    with pytest.raises(KeyboardInterrupt):
        p.run(config, log=lambda *_: None)
    assert [r["status"] for r in p.status(config)] == ["ok", "partial"]
    assert p.clean(config, "double")
    assert [r["status"] for r in p.status(config)] == ["ok", "missing"]


def test_prompt_version_changes_only_label_and_index_hashes():
    config = {"select": {"mode": "all"}, "cluster": {"k": 256}, "label": {"model": "m"}, "topic_index": {}}
    p = topics_pipeline()
    before = p.hashes(config)
    p.stages[2].prompt_version = "v2"
    after = p.hashes(config)
    assert [s for s in before if before[s] != after[s]] == ["label", "topic_index"]
