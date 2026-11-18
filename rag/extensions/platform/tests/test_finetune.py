from mlsrag.finetune import POOL, build_pairs, folds, rerank_order


def test_pairs_are_all_positives_and_the_hard_negatives_in_the_pool():
    ranked = [f"r{i}" for i in range(30)]
    rankings = {"q": ("a question", ranked)}
    pairs = build_pairs(["q"], rankings, {"q": {"r3", "r25"}})  # r25 is relevant but outside the pool
    positives = [r for _, r, y in pairs if y == 1.0]
    negatives = [r for _, r, y in pairs if y == 0.0]
    assert positives == ["r25", "r3"]
    assert negatives == [r for r in ranked[:POOL] if r != "r3"]


def test_rerank_reorders_only_the_pool_and_keeps_ties_in_dense_order():
    ranked = [f"r{i}" for i in range(25)]
    scores = [0.0] * POOL
    scores[5] = 2.0
    out = rerank_order(ranked, scores)
    assert out[0] == "r5" and out[1:POOL] == [r for r in ranked[:POOL] if r != "r5"] and out[POOL:] == ranked[POOL:]


def test_folds_cover_every_question_once_and_are_fixed():
    qids = [f"q{i}" for i in range(23)]
    a = folds(qids)
    assert sorted(q for f in a for q in f) == sorted(qids) and a == folds(list(reversed(qids)))
