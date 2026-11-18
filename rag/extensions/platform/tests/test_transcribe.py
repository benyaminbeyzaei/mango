"""mlsrag transcribe: timings in new cache lines, the meta file, old lines still readable."""
import json
import wave
from types import SimpleNamespace

from mlsrag import runner
from mlsrag.stages import load_transcripts


def write_wav(path, seconds):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))


def fake_data(tmp_path, monkeypatch):
    write_wav(tmp_path / "q1.wav", 1.0)
    write_wav(tmp_path / "q2.wav", 2.5)
    manifest = [{"query_id": q, "audio_file": f"{q}.wav"} for q in ("q1", "q2")]
    monkeypatch.setattr(runner, "paths", SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(runner, "CourseData", lambda d: SimpleNamespace(load_audio_manifest=lambda: manifest))
    monkeypatch.setattr(runner, "load_subset", lambda s: ["q1", "q2"])


def test_new_lines_carry_seconds_and_audio_s(tmp_path, monkeypatch):
    fake_data(tmp_path, monkeypatch)
    made = []

    class FakeWhisper:
        def __init__(self, gateway, fresh=False):
            made.append(fresh)

        def transcribe(self, audio_path):
            return f"text of {audio_path.stem}"

    monkeypatch.setattr(runner, "WhisperASR", FakeWhisper)
    out = tmp_path / "cache.jsonl"
    out.write_text(json.dumps({"query_id": "q1", "transcript": "old line"}) + "\n")  # no timings

    r = runner.transcribe("whisper1", "practice-20", out, fresh=True)
    assert (r["skipped"], r["transcribed"], made) == (1, 1, [True])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert "seconds" not in rows[0]
    assert rows[1]["query_id"] == "q2" and rows[1]["seconds"] >= 0 and rows[1]["audio_s"] == 2.5
    assert load_transcripts(out) == {"q1": "old line", "q2": "text of q2"}
    meta = json.loads((tmp_path / "cache.jsonl.meta.json").read_text())
    assert meta["asr"] == "whisper1" and meta["load_seconds"] >= 0 and meta["fresh"] is True


def test_course_model_load_is_timed_once_and_kept_out_of_query_seconds(tmp_path, monkeypatch):
    fake_data(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    prepared = []

    class FakeAdapter:
        def prepare(self):
            prepared.append(1)
            clock[0] += 30.0  # model load

    class FakeCourseASR:
        def __init__(self):
            self.adapter = FakeAdapter()

        def transcribe(self, audio_path):
            clock[0] += 2.0  # inference
            return "t"

    monkeypatch.setattr(runner, "CourseASR", FakeCourseASR)
    out = tmp_path / "course.jsonl"
    runner.transcribe("course", "practice-20", out)
    assert prepared == [1]
    assert json.loads((tmp_path / "course.jsonl.meta.json").read_text())["load_seconds"] == 30.0
    assert [json.loads(line)["seconds"] for line in out.read_text().splitlines()] == [2.0, 2.0]
