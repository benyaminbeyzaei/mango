"""mlsrag run | transcribe | make-subsets | judge | estimate | matrix | compare | fit-map | asr-compare | preprocess | serve | openapi"""
import argparse
import json
from pathlib import Path

from .config import load_config

OPENAPI_PATH = Path(__file__).resolve().parents[1] / "openapi.json"


def openapi_json(app) -> str:
    return json.dumps(app.openapi(), indent=2) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mlsrag")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a config over a subset")
    run.add_argument("--config", required=True, help="preset name (baseline) or .yaml/.json file")
    run.add_argument("--subset", default="practice-smoke20")
    run.add_argument("--no-generate", action="store_true", help="skip the generator (retrieval only; free with asr: cached)")
    run.add_argument("--allow-full", action="store_true", help="allow practice-all / dev-all")

    tr = sub.add_parser("transcribe", help="fill a transcript cache (resumable)")
    tr.add_argument("--asr", choices=["whisper1", "course"], default="whisper1")
    tr.add_argument("--subset", default="practice-smoke20")
    tr.add_argument("--out", required=True, help="JSONL path, relative to the checkout root")
    tr.add_argument("--allow-full", action="store_true", help="allow practice-all / dev-all")

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

    fm = sub.add_parser("fit-map", help="fit a linear map between two corpus encoder spaces")
    fm.add_argument("--from", dest="src", default="enc_a", choices=["enc_a", "enc_b"])
    fm.add_argument("--to", dest="dst", default="enc_b", choices=["enc_a", "enc_b"])
    fm.add_argument("--method", default="lstsq", choices=["lstsq", "procrustes"])
    fm.add_argument("--exclude-dev", action="store_true", help="leave the dev-relevant records out of the fit")

    ac = sub.add_parser("asr-compare", help="WER between two transcript caches and the retrieval of each")
    ac.add_argument("--a", required=True, help="reference transcript cache (JSONL)")
    ac.add_argument("--b", required=True, help="transcript cache to compare (JSONL)")
    ac.add_argument("--subset", default="practice-smoke20")
    ac.add_argument("--config", default="baseline")
    ac.add_argument("--name", help="output folder name (default: <a>-vs-<b>-<subset>)")

    pre = sub.add_parser("preprocess", help="run, inspect or clean a preprocessing pipeline")
    pre.add_argument("action", choices=["run", "status", "clean"])
    pre.add_argument("--pipeline", default="topics")
    pre.add_argument("--config", default="rag/extensions/platform/configs/pipelines/topics-k256.yaml")
    pre.add_argument("--stage", help="clean: the stage to delete")
    pre.add_argument("--yes", action="store_true", help="run: do not ask before the LLM calls")
    sel = pre.add_argument_group("record selection (overrides the config's select section)")
    sel.add_argument("--all", action="store_true", help="every corpus record")
    sel.add_argument("--ids", help="file with one record id per line")
    sel.add_argument("--sample", type=int, metavar="N", help="N random records")
    sel.add_argument("--split", choices=["dev-relevant"], help="records in qrels_dev.tsv + distractors")
    sel.add_argument("--distractors", type=int, default=0, metavar="M")
    sel.add_argument("--seed", type=int, default=0)

    sv = sub.add_parser("serve", help="run the HTTP API (routes under /api)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--web-dir", help="also serve this folder (the built web client) at /")

    oa = sub.add_parser("openapi", help="write the API's OpenAPI document")
    oa.add_argument("--out", default=str(OPENAPI_PATH))

    args = ap.parse_args(argv)
    if args.command == "preprocess":
        return preprocess(args)
    if args.command in ("serve", "openapi"):
        from .api.app import create_app

        if args.command == "serve":
            import uvicorn

            uvicorn.run(create_app(web_dir=args.web_dir), host=args.host, port=args.port)
        else:
            Path(args.out).write_text(openapi_json(create_app()), encoding="utf-8")
            print(f"wrote {args.out}")
        return
    from . import asr_compare, evaluate, maps, matrix, results, runner, subsets  # after parsing so --help stays fast

    if args.command == "run":
        out, metrics = runner.run(load_config(args.config), args.subset, generate=not args.no_generate,
                                  allow_full=args.allow_full)
        print(json.dumps(metrics, indent=2))
        print(f"results: {out}")
    elif args.command == "transcribe":
        print(json.dumps(runner.transcribe(args.asr, args.subset, args.out, args.allow_full), indent=2))
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
    elif args.command == "fit-map":
        out, info = maps.fit_map(args.src, args.dst, args.method, args.exclude_dev)
        print(json.dumps(info, indent=2))
        print(f"map: {out}")
    elif args.command == "asr-compare":
        out, summary = asr_compare.asr_compare(args.a, args.b, args.subset,
                                               matrix.resolve_config(args.config), args.name)
        print(asr_compare.format_summary(summary), end="")
        print(f"summary: {out / 'summary.json'}")
    else:
        print(results.compare(args.run_dirs), end="")


def selection(args):
    """The select stage config from the command line flags, or None to keep the config's."""
    if args.all:
        return {"mode": "all"}
    if args.ids:
        from pathlib import Path

        ids = [line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"mode": "ids", "ids": ids}
    if args.sample is not None:
        return {"mode": "sample", "n": args.sample, "seed": args.seed}
    if args.split:
        return {"mode": args.split, "distractors": args.distractors, "seed": args.seed}
    return None


def confirm(message):
    if input(f"{message}. Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")


def preprocess(args):
    from .preprocess import PIPELINES, load_pipeline_config
    from .stages import checkout_path

    pipeline = PIPELINES[args.pipeline]()
    config = load_pipeline_config(checkout_path(args.config))
    if (sel := selection(args)) is not None:
        config["select"] = sel
    if args.action == "run":
        dirs = pipeline.run(config, confirm=None if args.yes else confirm)
        print(f"output: {dirs[pipeline.stages[-1].name]}")
    elif args.action == "status":
        for row in pipeline.status(config):
            m = row["manifest"] or {}
            calls = sum(c["calls"] for c in m.get("cost", {}).values())
            tokens = sum(c["prompt_tokens"] + c["completion_tokens"] for c in m.get("cost", {}).values())
            wall = f"{m['wall_seconds']:.1f}s" if m else "-"
            print(f"{row['stage']:<12} {row['hash']} {row['status']:<8} {wall:>8} "
                  f"elm_calls={calls} tokens={tokens} {m.get('counts', '')}")
    else:
        if not args.stage:
            raise SystemExit("clean needs --stage")
        removed = pipeline.clean(config, args.stage)
        print("\n".join(f"removed {d}" for d in removed) or "nothing to remove")
