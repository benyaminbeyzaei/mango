"""A resumable preprocessing pipeline: stages run in order, each output is cached by a hash.

A stage's output directory is rag/.cache/preprocess/<pipeline>/<stage>/<hash>/, where the hash
covers the upstream hashes, the stage config, the stage's code version and its prompt version.
A directory whose manifest.json says `status: ok` is reused. A stage writes into
<hash>.partial/ first, which is renamed into place only when the stage finishes, so an
interrupted stage (and its ledger) is picked up by the next run.
"""
import hashlib
import json
import platform
import shutil
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from ..course import paths
from ..gateway import ModelGateway
from ..trace import Trace

PACKAGES = ["numpy", "scikit-learn", "openai", "mlsrag"]


@dataclass
class Stage:
    name: str
    run: Callable  # run(inputs: {stage name: output dir}, config: dict, ctx) -> counts dict
    inputs: list = field(default_factory=list)  # names of the upstream stages it reads
    version: str = "1"  # bump when the code changes what the stage writes
    prompt_version: str | None = None


def _versions():
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


class Pipeline:
    def __init__(self, name, stages, root=None):
        self.name = name
        self.stages = stages
        self.root = Path(root or paths.cache_dir / "preprocess") / name

    def hashes(self, config) -> dict:
        """{stage name: hash} for this config, without running anything."""
        out = {}
        for s in self.stages:
            key = {
                "inputs": {i: out[i] for i in s.inputs},
                "config": config.get(s.name, {}),
                "version": s.version,
                "prompt_version": s.prompt_version,
            }
            out[s.name] = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        return out

    def stage_dir(self, stage_name, h) -> Path:
        return self.root / stage_name / h

    def manifest(self, stage_name, h):
        p = self.stage_dir(stage_name, h) / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def run(self, config, gateway=None, log=print) -> dict:
        """Run every stage that is not done yet. Returns {stage name: output dir}."""
        hashes = self.hashes(config)
        dirs = {}
        for s in self.stages:
            h = hashes[s.name]
            out = self.stage_dir(s.name, h)
            dirs[s.name] = out
            if (self.manifest(s.name, h) or {}).get("status") == "ok":
                log(f"{s.name:<12} {h} skipped (done)")
                continue
            partial = out.with_name(h + ".partial")
            partial.mkdir(parents=True, exist_ok=True)
            trace = Trace(f"preprocess-{self.name}", s.name, self.name)
            gw = gateway or ModelGateway()
            gw.trace = trace
            ctx = SimpleNamespace(out_dir=partial, gateway=gw, trace=trace, log=log)
            log(f"{s.name:<12} {h} running")
            start = time.perf_counter()
            counts = s.run({i: dirs[i] for i in s.inputs}, config.get(s.name, {}), ctx)
            trace.finish()
            manifest = {
                "pipeline": self.name,
                "stage": s.name,
                "hash": h,
                "status": "ok",
                "inputs": {i: hashes[i] for i in s.inputs},
                "config": config.get(s.name, {}),
                "version": s.version,
                "prompt_version": s.prompt_version,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "wall_seconds": time.perf_counter() - start,
                "counts": counts or {},
                "cost": trace.cost,
                "counters": trace.counters,
                "resources": trace.resources,
                "versions": _versions(),
            }
            (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            if out.exists():  # a leftover without an ok manifest
                shutil.rmtree(out)
            partial.rename(out)
            log(f"{s.name:<12} {h} ok in {manifest['wall_seconds']:.1f}s {counts or ''}")
        return dirs

    def status(self, config) -> list:
        """One row per stage: name, hash, ok | partial | missing, and the manifest if ok."""
        rows = []
        for s in self.stages:
            h = self.hashes(config)[s.name]
            m = self.manifest(s.name, h)
            if m and m.get("status") == "ok":
                state = "ok"
            elif self.stage_dir(s.name, h).with_name(h + ".partial").exists():
                state = "partial"
            else:
                state = "missing"
            rows.append({"stage": s.name, "hash": h, "status": state, "manifest": m})
        return rows

    def clean(self, config, stage_name):
        """Delete one stage's output (and partial output) for this config."""
        h = self.hashes(config)[stage_name]
        removed = []
        for d in (self.stage_dir(stage_name, h), self.stage_dir(stage_name, h).with_name(h + ".partial")):
            if d.exists():
                shutil.rmtree(d)
                removed.append(str(d))
        return removed
