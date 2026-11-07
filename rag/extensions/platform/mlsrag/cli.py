"""mlsrag run | transcribe | make-subsets"""
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

    args = ap.parse_args(argv)
    from . import runner, subsets  # after parsing so --help stays fast

    if args.command == "run":
        out, metrics = runner.run(load_config(args.config), args.subset, generate=not args.no_generate)
        print(json.dumps(metrics, indent=2))
        print(f"results: {out}")
    elif args.command == "transcribe":
        print(json.dumps(runner.transcribe(args.asr, args.subset, args.out), indent=2))
    else:
        subsets.make_subsets()
        print(f"wrote subsets to {subsets.SUBSETS_DIR}")
