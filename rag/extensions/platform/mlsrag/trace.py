"""Trace: what happened during one query and what it cost."""
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class Span:
    name: str
    start_ms: float  # since the trace started
    duration_ms: float = 0.0
    attrs: dict = field(default_factory=dict)


@dataclass
class Trace:
    run_id: str
    query_id: str
    config_name: str
    spans: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)  # model -> {calls, prompt_tokens, ...}
    resources: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    def __post_init__(self):
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()

    @contextmanager
    def span(self, name, **attrs):
        start = time.perf_counter()
        s = Span(name, (start - self._t0) * 1000.0, attrs=dict(attrs))
        self.spans.append(s)
        try:
            yield s
        finally:
            s.duration_ms = (time.perf_counter() - start) * 1000.0

    def count(self, key, n=1):
        self.counters[key] = self.counters.get(key, 0) + n

    def add_cost(self, calls=1, prompt_tokens=0, completion_tokens=0, audio_seconds=0.0, model=None):
        c = self.cost.setdefault(
            model or "unknown",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "audio_seconds": 0.0},
        )
        c["calls"] += calls
        c["prompt_tokens"] += prompt_tokens
        c["completion_tokens"] += completion_tokens
        c["audio_seconds"] += audio_seconds

    def finish(self):
        """Record wall time, CPU seconds and peak RSS (peak is for the whole process)."""
        self.resources = {
            "wall_ms": (time.perf_counter() - self._t0) * 1000.0,
            "cpu_seconds": time.process_time() - self._cpu0,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }

    def to_dict(self):
        return asdict(self)
