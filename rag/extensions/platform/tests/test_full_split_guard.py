"""Every entry point refuses practice / test without the explicit flag (plan 015)."""
from types import SimpleNamespace

import pytest
from conftest import needs_data

from mlsrag import cli, evaluate, runner
from mlsrag.matrix import resolve_config
from mlsrag.preprocess.ledger import run_items
from mlsrag.preprocess.pipeline import Pipeline, Stage


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the gateway must not be built")
    monkeypatch.setattr(runner, "ModelGateway", refuse)


@needs_data  # the estimate in the message reads the audio manifest
def test_cli_run_refuses_full_split_and_writes_nothing(tmp_path, monkeypatch, no_network):
    monkeypatch.setattr(runner, "paths", SimpleNamespace(results_dir=tmp_path, data_dir=tmp_path))
    with pytest.raises(ValueError, match="practice is a full split") as err:
        cli.main(["run", "--config", "baseline", "--subset", "practice", "--no-generate"])
    assert "generation calls: 0" in str(err.value)
    assert list(tmp_path.iterdir()) == []


@needs_data
def test_refused_run_reports_the_estimate(no_network):
    config = resolve_config("grep-agentic")
    want = evaluate.estimate(config, "practice")
    assert want["calls"]["flow"] == 4 * 350
    with pytest.raises(ValueError) as err:
        runner.run(config, "practice")
    assert f"total ELM calls:  {want['elm_calls']}" in str(err.value)


def test_cli_transcribe_refuses_full_split(tmp_path, no_network):
    out = tmp_path / "x.jsonl"
    with pytest.raises(ValueError, match="test is a full split"):
        cli.main(["transcribe", "--asr", "whisper1", "--subset", "test", "--out", str(out)])
    assert not out.exists()


class CountingGateway:
    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        return "ok"


def llm_stage(inputs, config, ctx):
    run_items({i: i for i in range(3)}, ctx, "m", lambda data, attempt: f"item {data}", str)
    return {}


def test_preprocess_asks_before_llm_calls(tmp_path, monkeypatch, capsys):
    p = Pipeline("t", [Stage("llm", llm_stage, prompt_version="v1")], root=tmp_path)
    gw = CountingGateway()
    monkeypatch.setattr("builtins.input", lambda prompt: print(prompt) or "n")
    with pytest.raises(SystemExit, match="aborted"):
        p.run({}, gateway=gw, log=lambda *_: None, confirm=cli.confirm)
    assert gw.calls == 0
    assert "3 items, 0 done after 0 calls; 3 ELM calls to make" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    p.run({}, gateway=gw, log=lambda *_: None, confirm=cli.confirm)
    assert gw.calls == 3


def test_preprocess_cli_yes_skips_the_question(monkeypatch):
    seen = {}

    class FakePipeline:
        stages = [SimpleNamespace(name="last")]

        def run(self, config, confirm):
            seen["confirm"] = confirm
            return {"last": "out"}

    monkeypatch.setattr("mlsrag.preprocess.PIPELINES", {"fake": FakePipeline})
    monkeypatch.setattr("mlsrag.preprocess.load_pipeline_config", lambda path: {})
    cli.main(["preprocess", "run", "--pipeline", "fake", "--yes"])
    assert seen["confirm"] is None
    cli.main(["preprocess", "run", "--pipeline", "fake"])
    assert seen["confirm"] is cli.confirm


def test_prompt_shows_calls_left_and_pace(tmp_path):
    from mlsrag.preprocess.ledger import Ledger

    out = tmp_path / "h1"
    Ledger(out / "ledger.jsonl").append("r0", "h", "ok", output="t", tokens=10, latency_ms=2000.0)
    Ledger(out / "ledger.jsonl").append("r1", "h", "failed", tokens=10, latency_ms=4000.0, error="bad json")
    messages = []

    def stop(message):
        messages.append(message)
        raise SystemExit("aborted")

    logged = []
    ctx = SimpleNamespace(out_dir=out, gateway=CountingGateway(), trace=SimpleNamespace(cost={}),
                          log=logged.append, confirm=stop)
    with pytest.raises(SystemExit):
        run_items({f"r{i}": "t" for i in range(3)}, ctx, "m", lambda data, attempt: "p", str)
    assert ctx.gateway.calls == 0
    want = "3 items, 1 done after 2 calls; 2 ELM calls to make, mean 3.0s per call so far"
    assert messages == [want] and want in logged


def test_estimate_is_logged_even_without_confirm(tmp_path):
    logged = []
    ctx = SimpleNamespace(out_dir=tmp_path, gateway=CountingGateway(), trace=SimpleNamespace(cost={}),
                          log=logged.append, confirm=None)
    run_items({0: 0}, ctx, "m", lambda data, attempt: "p", str)
    assert "1 items, 0 done after 0 calls; 1 ELM calls to make" in logged
