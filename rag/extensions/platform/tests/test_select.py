from mlsrag.preprocess import select

from conftest import needs_data

IDS = [f"r{i}" for i in range(100)]


def test_modes_are_deterministic_and_keep_corpus_order():
    a = select.select_ids(IDS, {"mode": "sample", "n": 10, "seed": 1})
    assert a == select.select_ids(IDS, {"mode": "sample", "n": 10, "seed": 1})
    assert len(a) == 10 and a == sorted(a, key=IDS.index)
    assert select.select_ids(IDS, {"mode": "ids", "ids": ["r5", "r2"]}) == ["r2", "r5"]
    assert select.select_ids(IDS, {"mode": "all"}) == IDS
    dev = select.select_ids(IDS, {"mode": "dev-relevant", "distractors": 5, "seed": 0}, {"r1", "r2"})
    assert len(dev) == 7 and {"r1", "r2"} <= set(dev)


@needs_data
def test_dev_relevant_with_500_distractors(tmp_path):
    from types import SimpleNamespace

    ctx = SimpleNamespace(out_dir=tmp_path)
    config = {"mode": "dev-relevant", "distractors": 500, "seed": 0}
    assert select.run({}, config, ctx) == {"records": 825}
    first = (tmp_path / "records.jsonl").read_text()
    select.run({}, config, ctx)
    assert (tmp_path / "records.jsonl").read_text() == first


def test_top_words_breaks_ties_alphabetically():
    from mlsrag.preprocess.cluster import top_words

    assert top_words(["zebra apple", "zebra mango", "kiwi"], n=3) == ["zebra", "apple", "kiwi"]
