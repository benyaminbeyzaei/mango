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

    def chat(self, model, messages, temperature=0.0, max_tokens=None) -> str:
        def call():
            extra = {"max_tokens": max_tokens} if max_tokens else {}
            r = self.client().chat.completions.create(
                model=model, messages=messages, temperature=temperature, **extra
            )
            return {
                "text": r.choices[0].message.content.strip(),
                "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                "completion_tokens": r.usage.completion_tokens if r.usage else 0,
            }

        chars = sum(len(m["content"]) for m in messages)
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        result = self._run("chat", model, payload, call, remote=True, input_chars=chars)
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

        payload = {"texts": list(texts), "role": role}
        result = self._run("embed", encoder_name, payload, call, remote=False, n_texts=len(texts))
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
            r = self.client().audio.transcriptions.create(
                model=model, file=(audio_path.name, audio)
            )
            return {"text": r.text.strip(), "audio_seconds": seconds}

        payload = {"audio_sha256": hashlib.sha256(audio).hexdigest()}
        result = self._run("transcribe", model, payload, call, remote=True, audio_seconds=seconds)
        return result["text"]

    def _run(self, op, model, payload, call, remote, **sizes):
        key = hashlib.sha256(json.dumps([op, model, payload], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        trace = self.trace or Trace("-", "-", "-")  # throwaway when nobody is tracing
        retries = 0
        with trace.span(f"gateway.{op}", model=model, **sizes) as span:
            cached = path.exists()
            if cached:
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                while True:
                    try:
                        if remote:
                            with _remote_slots:
                                result = call()
                        else:
                            result = call()
                        break
                    except (openai.APIStatusError, openai.APIConnectionError) as err:
                        if not _should_retry(err) or retries == RETRIES:
                            self._log(
                                op=op, model=model, cached=False, retries=retries,
                                cache_key=key, error=type(err).__name__, **sizes,
                            )
                            raise
                        time.sleep(2**retries)
                        retries += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(result), encoding="utf-8")
                tmp.replace(path)
            span.attrs.update(cached=cached, retries=retries)

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
        return result

    def _log(self, **fields):
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
