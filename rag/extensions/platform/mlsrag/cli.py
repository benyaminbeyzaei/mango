"""mlsrag run | transcribe | make-subsets | judge | estimate | matrix | compare"""
import argparse
import json

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mlsrag")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a config over a subset")
    run.add_argument("--config", required=True, help="preset name (baseline) or .yaml/.json file")
    run.add_argument("--subset", default="practice-smoke20")
    run.add_argument("--no-generate", action="store_true", help="skip the generator (retrieval only; free with asr: cached)")

    tr = sub.add_parser("transcribe", help="fill a transcript cache (resumable)")
    tr.add_argument("--asr", choices=["whisper1", "course"], default="whisper1")
    tr.add_argument("--subset", default="practice-smoke20")
    tr.add_argument("--out", required=True, help="JSONL path, relative to the checkout root")

    sub.add_parser("make-subsets", help="regenerate the subset files (seed 0)")

    jd = sub.add_parser("judge", help="judge a run's answers with the course evaluator")
    jd.add_argument("run_dir")
    jd.add_argument("--model", help="judge model (default: the run config's judge_model)")
    jd.add_argument("--repeats", type=int, default=1)
    jd.add_argument("--dry-run", action="store_true", help="validate the candidates only, no API calls")

    es = sub.add_parser("estimate", help="print the expected ELM calls and cost of a run")
    es.add_argument("--config", required=True)
    es.add_argument("--subset", default="practice-smoke20")
    es.add_argument("--judge", action="store_true", help="include the judge calls")

    mx = sub.add_parser("matrix", help="run several configs on one subset and summarise them")
    mx.add_argument("--matrix", required=True, help="YAML with name, subset, configs")
    mx.add_argument("--no-judge", action="store_true")
    mx.add_argument("--no-generate", action="store_true", help="retrieval only (implies --no-judge)")
    mx.add_argument("--force", action="store_true", help="rerun configs that already have a run")
    mx.add_argument("--allow-full", action="store_true", help="allow practice-all / dev-all")
    mx.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    cp = sub.add_parser("compare", help="print a Markdown table of several runs side by side")
    cp.add_argument("run_dirs", nargs="+")

    args = ap.parse_args(argv)
    from . import evaluate, matrix, results, runner, subsets  # after parsing so --help stays fast

    if args.command == "run":
        out, metrics = runner.run(load_config(args.config), args.subset, generate=not args.no_generate)
        print(json.dumps(metrics, indent=2))
        print(f"results: {out}")
    elif args.command == "transcribe":
        print(json.dumps(runner.transcribe(args.asr, args.subset, args.out), indent=2))
    elif args.command == "make-subsets":
        subsets.make_subsets()
        print(f"wrote subsets to {subsets.SUBSETS_DIR}")
    elif args.command == "judge":
        print(json.dumps(evaluate.judge(args.run_dir, args.model, args.repeats, args.dry_run), indent=2))
    elif args.command == "estimate":
        config = matrix.resolve_config(args.config)
        print(evaluate.format_estimate(evaluate.estimate(config, args.subset, judge=args.judge)))
    elif args.command == "matrix":
        out = matrix.run_matrix(args.matrix, judge=not args.no_judge, generate=not args.no_generate,
                                force=args.force, allow_full=args.allow_full, yes=args.yes)
        print((out / "summary.md").read_text(encoding="utf-8"))
        print(f"summary: {out}")
    else:
        print(results.compare(args.run_dirs), end="")
