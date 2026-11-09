from mlsrag.preprocess import select


IDS = [f"r{i}" for i in range(100)]


def test_modes_are_deterministic_and_keep_corpus_order():
    a = select.select_ids(IDS, {"mode": "sample", "n": 10, "seed": 1})
    assert a == select.select_ids(IDS, {"mode": "sample", "n": 10, "seed": 1})
    assert len(a) == 10 and a == sorted(a, key=IDS.index)
    assert select.select_ids(IDS, {"mode": "ids", "ids": ["r5", "r2"]}) == ["r2", "r5"]
    assert select.select_ids(IDS, {"mode": "all"}) == IDS


def test_top_words_breaks_ties_alphabetically():
    from mlsrag.preprocess.cluster import top_words

    assert top_words(["zebra apple", "zebra mango", "kiwi"], n=3) == ["zebra", "apple", "kiwi"]
