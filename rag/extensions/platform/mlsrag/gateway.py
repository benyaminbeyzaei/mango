"""ModelGateway: the one place every model call goes through.

Each call gets a span on the current trace, its cost (from the API usage), a line in
rag/.cache/gateway.log.jsonl and a disk cache entry, so an identical call is free next time.
"""
import hashlib
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import openai
from dotenv import dotenv_values

from .course import paths
from .trace import Trace

LOCAL_ENCODERS = {
    "gte-small": str(paths.models_dir / "enc_a"),
    "bge-small": "BAAI/bge-small-en-v1.5",
}
DEFAULT_BASE_URL = "https://elm.edina.ac.uk/api/v1"
RETRIES = 3

_remote_slots = threading.Semaphore(4)  # at most 4 concurrent ELM requests per process
_log_lock = threading.Lock()
_encoders = {}  # name -> loaded SentenceTransformer, shared by all gateways


def _should_retry(err):
    """429, 5xx, timeouts and dropped connections are worth another try; other errors are not."""
    if isinstance(err, openai.APIConnectionError):  # includes APITimeoutError
        return True
    return err.status_code == 429 or err.status_code >= 500


def cache_key(op, model, payload) -> str:
    return hashlib.sha256(json.dumps([op, model, payload], sort_keys=True).encode()).hexdigest()


def cache_get(cache_dir, key):
    """The cached result of an identical call, or None."""
    path = Path(cache_dir) / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cache_put(cache_dir, key, result):
    path = Path(cache_dir) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(path)  # atomic, so a reader never sees half a file


def with_retries(call, log_failure):
    """(call(), retries). Retryable errors are retried RETRIES times with backoff 1, 2, 4 s;
    on giving up, log_failure(error=<name>, retries=<n>) is called and the error re-raised."""
    retries = 0
    while True:
        try:
            return call(), retries
        except (openai.APIStatusError, openai.APIConnectionError) as err:
            if not _should_retry(err) or retries == RETRIES:
                log_failure(error=type(err).__name__, retries=retries)
                raise
            time.sleep(2**retries)
            retries += 1


class ModelGateway:
    def __init__(self, trace=None, cache_dir=None, log_path=None, env_path=None):
        self.trace = trace
        self.cache_dir = Path(cache_dir or paths.cache_dir / "gateway")
        self.log_path = Path(log_path or paths.cache_dir / "gateway.log.jsonl")
        self.env_path = Path(env_path or paths.rag_dir / ".env")
        self._client = None

    def client(self):
        if self._client is None:
            env = dotenv_values(self.env_path)
            if not env.get("ELM_API_KEY"):
                raise RuntimeError(f"ELM_API_KEY missing from {self.env_path}")
            self._client = openai.OpenAI(
                api_key=env["ELM_API_KEY"],
                base_url=env.get("ELM_BASE_URL") or DEFAULT_BASE_URL,
                timeout=180,
                max_retries=0,  # retries are done here, so they are visible and counted
            )
        return self._client

    def encoder(self, name):
        """Load a local sentence-transformers encoder once per process."""
        if name not in _encoders:
            from sentence_transformers import SentenceTransformer

            if name == "gte-small":
                _encoders[name] = SentenceTransformer(LOCAL_ENCODERS[name], local_files_only=True)
            else:
                _encoders[name] = SentenceTransformer(
                    LOCAL_ENCODERS[name], cache_folder=str(paths.cache_dir / "hf")
                )
        return _encoders[name]

    def chat(self, model, messages, temperature=0.0, max_tokens=None, tag=None) -> str:
        """`tag` (e.g. a prompt VERSION) enters the cache key when given."""
        def call():
            extra = {"max_tokens": max_tokens} if max_tokens else {}
            with _remote_slots:
                r = self.client().chat.completions.create(
                    model=model, messages=messages, temperature=temperature, **extra
                )
            return {
                "text": r.choices[0].message.content.strip(),
                "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                "completion_tokens": r.usage.completion_tokens if r.usage else 0,
            }

        sizes = {"input_chars": sum(len(m["content"]) for m in messages)}
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tag:
            payload["tag"] = tag  # only when given, so untagged cache keys stay as they were
        key = cache_key("chat", model, payload)
        with self._trace().span("gateway.chat", model=model, **sizes) as span:
            result, retries = cache_get(self.cache_dir, key), 0
            cached = result is not None
            if not cached:
                result, retries = with_retries(call, self._failure_log("chat", model, key, sizes))
                cache_put(self.cache_dir, key, result)
        self._record("chat", model, key, span, result, cached, retries, sizes, remote=True)
        return result["text"]

    def embed(self, encoder_name, texts, role="query") -> np.ndarray:
        """Normalised embeddings. `role` is recorded and keyed, not used yet (plan 005)."""

        def call():
            vecs = self.encoder(encoder_name).encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return {"vectors": np.asarray(vecs, dtype=np.float32).tolist()}

        sizes = {"n_texts": len(texts)}
        key = cache_key("embed", encoder_name, {"texts": list(texts), "role": role})
        with self._trace().span("gateway.embed", model=encoder_name, **sizes) as span:
            result, retries = cache_get(self.cache_dir, key), 0
            cached = result is not None
            if not cached:
                result, retries = with_retries(call, self._failure_log("embed", encoder_name, key, sizes))
                cache_put(self.cache_dir, key, result)
        self._record("embed", encoder_name, key, span, result, cached, retries, sizes, remote=False)
        return np.asarray(result["vectors"], dtype=np.float32)

    def transcribe(self, audio_path, model="whisper-1") -> str:
        audio_path = Path(audio_path)
        audio = audio_path.read_bytes()
        try:
            with wave.open(str(audio_path)) as w:
                seconds = w.getnframes() / w.getframerate()
        except (wave.Error, EOFError):  # not a WAV (e.g. WebM from a browser): duration unknown, cost counts 0 s
            seconds = 0.0

        def call():
            with _remote_slots:
                r = self.client().audio.transcriptions.create(
                    model=model, file=(audio_path.name, audio)
                )
            return {"text": r.text.strip(), "audio_seconds": seconds}

        sizes = {"audio_seconds": seconds}
        key = cache_key("transcribe", model, {"audio_sha256": hashlib.sha256(audio).hexdigest()})
        with self._trace().span("gateway.transcribe", model=model, **sizes) as span:
            result, retries = cache_get(self.cache_dir, key), 0
            cached = result is not None
            if not cached:
                result, retries = with_retries(call, self._failure_log("transcribe", model, key, sizes))
                cache_put(self.cache_dir, key, result)
        self._record("transcribe", model, key, span, result, cached, retries, sizes, remote=True)
        return result["text"]

    def _trace(self):
        return self.trace or Trace("-", "-", "-")  # throwaway when nobody is tracing

    def _failure_log(self, op, model, key, sizes):
        return lambda **fields: self._log(op=op, model=model, cached=False, cache_key=key, **sizes, **fields)

    def _record(self, op, model, key, span, result, cached, retries, sizes, remote):
        """After a call: span attributes, counters, cost (new remote calls only) and the log line."""
        span.attrs.update(cached=cached, retries=retries)
        trace = self._trace()
        trace.count(f"gateway.{op}.cache_hits" if cached else f"gateway.{op}.calls")
        if remote and not cached:
            trace.add_cost(
                calls=1,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                audio_seconds=result.get("audio_seconds", 0.0),
                model=model,
            )
        self._log(
            op=op, model=model, cached=cached, retries=retries,
            duration_ms=span.duration_ms, cache_key=key, **sizes,
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
        )

    def _log(self, **fields):
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
