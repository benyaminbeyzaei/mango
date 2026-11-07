import json

import pytest
from pydantic import ValidationError

from mlsrag.config import RunConfig, baseline, load_config


def test_baseline_validates():
    c = baseline()
    assert (c.query_encoder, c.corpus_index, c.retriever, c.k) == ("gte-small", "enc_a", "exact", 5)
    assert c.generator_model == "Qwen/Qwen3.5-397B-A17B-FP8"


def test_mismatch_raises_unless_allowed():
    with pytest.raises(ValidationError, match="does not match"):
        RunConfig(name="x", query_encoder="bge-small", corpus_index="enc_a")
    RunConfig(name="x", query_encoder="bge-small", corpus_index="enc_a", mismatch_policy="as-is")
    RunConfig(name="x", query_encoder="bge-small", corpus_index="enc_b")


def test_load_yaml_json_and_preset(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\nk: 10\n")
    (tmp_path / "b.json").write_text(json.dumps({"name": "b", "corpus_index": "enc_a"}))
    assert load_config(tmp_path / "a.yaml").k == 10
    assert load_config(tmp_path / "b.json").name == "b"
    assert load_config("baseline") == baseline()
    (tmp_path / "bad.yaml").write_text("name: a\ntypo_field: 1\n")
    with pytest.raises(ValidationError):
        load_config(tmp_path / "bad.yaml")
