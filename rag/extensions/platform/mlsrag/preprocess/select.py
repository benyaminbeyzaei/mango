"""Selection stage: which corpus records the rest of the pipeline sees (records.jsonl).

Config `mode`: all | ids (`ids`: list) | sample (`n`, `seed`).
"""
import json
import random

from ..course import CourseData, paths


def select_ids(all_ids, config) -> list:
    mode = config.get("mode", "all")
    seed = config.get("seed", 0)
    if mode == "all":
        return list(all_ids)
    if mode == "ids":
        wanted = set(config["ids"])
        return [r for r in all_ids if r in wanted]
    if mode == "sample":
        chosen = set(random.Random(seed).sample(list(all_ids), config["n"]))
        return [r for r in all_ids if r in chosen]
    raise ValueError(f"unknown selection mode {mode!r}")


def run(inputs, config, ctx):
    data = CourseData(paths.data_dir)
    ids = select_ids(list(data.records), config)
    with (ctx.out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for rid in ids:
            f.write(json.dumps({"record_id": rid, "text": data.records[rid]}, ensure_ascii=False) + "\n")
    return {"records": len(ids)}
