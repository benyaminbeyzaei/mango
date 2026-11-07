import json

import pytest

from mlsrag.asr_compare import asr_compare, compare_transcripts, normalise
from mlsrag.config import baseline
from mlsrag.stages import checkout_path, load_transcripts

WHISPER = "rag/.cache/transcripts/whisper1-practice.jsonl"


def drop_every_5th_word(text):
    return " ".join(w for i, w in enumerate(text.split(), start=1) if i % 5)


def write_cache(path, transcripts):
    lines = [json.dumps({"query_id": q, "transcript": t}) for q, t in transcripts.items()]
    path.write_text("\n".join(lines) + "\n")


def test_normalise():
    assert normalise("Hello, World!  It's  me.") == "hello world it's me"


def test_compare_transcripts_identical_and_corrupted():
    a = {"q1": "one two three four five six", "q2": "a short one"}
    assert compare_transcripts(a, a, ["q1", "q2"]) == {"wer": 0.0, "agreement": 1.0}
    b = {q: drop_every_5th_word(t) for q, t in a.items()}
    r = compare_transcripts(a, b, ["q1", "q2"])
    assert r["wer"] == pytest.approx(1 / 9)  # one deletion in nine reference words
    assert r["agreement"] == 0.5


@pytest.mark.slow
def test_asr_compare_whisper_against_itself_and_a_corrupted_copy(tmp_path):
    corrupted = tmp_path / "corrupted.jsonl"
    whisper = load_transcripts(checkout_path(WHISPER))
    write_cache(corrupted, {q: drop_every_5th_word(t) for q, t in whisper.items()})

    _, same = asr_compare(WHISPER, WHISPER, "practice-smoke20", baseline(), "same", tmp_path)
    assert (same["wer"], same["agreement"]) == (0.0, 1.0)
    assert same["retrieval_a"] == same["retrieval_b"]

    out, worse = asr_compare(WHISPER, corrupted, "practice-smoke20", baseline(), "corrupted", tmp_path)
    assert (out / "summary.json").exists()
    assert worse["wer"] > 0.1 and worse["agreement"] < 1.0
    assert worse["retrieval_b"] != worse["retrieval_a"]
