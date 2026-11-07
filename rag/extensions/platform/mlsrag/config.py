"""RunConfig: everything that describes one run."""
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

# Which corpus index each query encoder's vectors are comparable with.
ENCODER_SPACE = {"gte-small": "enc_a", "bge-small": "enc_b"}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    asr: Literal["cached", "whisper1", "course"] = "cached"
    asr_cache: str | None = None  # JSONL of {query_id, transcript}; relative to the checkout root
    query_encoder: Literal["gte-small", "bge-small"] = "gte-small"
    corpus_index: Literal["enc_a", "enc_b"] = "enc_a"
    # error: refuse a mismatch; as-is: use the vector anyway; re-encode: encode with the index's
    # own encoder (params.bge_prefix); translate: map into the index space (params.map)
    mismatch_policy: Literal["error", "as-is", "re-encode", "translate"] = "error"
    retriever: str = "exact"
    k: int = 5
    context_builder: str = "simple"
    generator_model: str = "Qwen/Qwen3.5-397B-A17B-FP8"
    judge_model: str = "mistralai/Mistral-Small-4-119B-2603"
    flow: str = "standard"
    params: dict = {}

    @model_validator(mode="after")
    def check_encoder_matches_index(self):
        if self.mismatch_policy == "error" and ENCODER_SPACE[self.query_encoder] != self.corpus_index:
            raise ValueError(
                f"query_encoder {self.query_encoder} does not match corpus_index "
                f"{self.corpus_index}; set mismatch_policy to allow it"
            )
        return self


def baseline() -> RunConfig:
    return RunConfig(
        name="baseline",
        asr_cache="rag/.cache/transcripts/whisper1-practice.jsonl",
        query_encoder="gte-small",
        corpus_index="enc_a",
        retriever="exact",
        k=5,
        context_builder="simple",
        generator_model="Qwen/Qwen3.5-397B-A17B-FP8",
    )


PRESETS = {"baseline": baseline}


def load_config(path) -> RunConfig:
    """Load a RunConfig from a .json or .yaml/.yml file, or by preset name."""
    if str(path) in PRESETS:
        return PRESETS[str(path)]()
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return RunConfig(**raw)
