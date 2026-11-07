"""Run several configs on one subset and write one summary table."""
import hashlib
import json
from pathlib import Path

import yaml

from . import evaluate, runner
from .config import PRESETS, RunConfig, load_config
from .course import paths
from .results import list_runs, load_run, markdown_table, summary_row

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
FULL_SUBSETS = {"practice-all", "dev-all"}


def resolve_config(name) -> RunConfig:
    """A preset name, a file in configs/ (without .yaml), or a path."""
    if name in PRESETS:
        return load_config(name)
    for path in (CONFIGS_DIR / f"{name}.yaml", Path(name)):
        if path.exists():
            return load_config(path)
    raise SystemExit(f"unknown config {name!r}: not a preset, configs/{name}.yaml or a file")


def config_hash(config) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:12]


def find_run(config, subset, generate):
    """The newest completed run with the same config, subset and generation mode."""
    want = config_hash(config)
    for run_dir in reversed(list_runs(paths.results_dir)):
        run = load_run(run_dir)
        if (run["metrics"]["run"]["subset"] == subset
                and config_hash(RunConfig(**run["config"])) == want
                and (run["metrics"]["latency_ms"]["generate"] is not None) == generate):
            return run_dir
    return None


def run_matrix(path, judge=True, generate=True, force=False, allow_full=False, yes=False):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    subset = spec["subset"]
    if subset in FULL_SUBSETS and not allow_full:
        raise SystemExit(f"{subset} is a full split; pass --allow-full if Benyamin approved it")
    configs = [resolve_config(name) for name in spec["configs"]]
    judge = judge and generate  # nothing to judge without answers

    estimates = [evaluate.estimate(c, subset, judge=judge) for c in configs]
    for e in estimates:
        if not generate:
            e["elm_calls"] -= e["calls"]["generation"]
            e["calls"]["generation"] = 0
        print(evaluate.format_estimate(e))
    print(f"matrix total: {sum(e['elm_calls'] for e in estimates)} ELM calls (upper bound)")
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")

    rows = []
    for config in configs:
        run_dir = None if force else find_run(config, subset, generate)
        if run_dir:
            print(f"{config.name}: reusing {run_dir.name}")
        else:
            run_dir, _ = runner.run(config, subset, generate=generate)
        answers = load_run(run_dir)["metrics"]["answers"]
        if judge and (answers is None or answers["judge_model"] != config.judge_model):
            evaluate.judge(run_dir)
        rows.append(summary_row(load_run(run_dir)))

    out = paths.results_dir / "matrices" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"name": spec["name"], "subset": subset, "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    header = list(rows[0])
    (out / "summary.md").write_text(
        f"# Matrix {spec['name']} on {subset}\n\n"
        + markdown_table(header, [[r[k] for k in header] for r in rows]),
        encoding="utf-8")
    return out
