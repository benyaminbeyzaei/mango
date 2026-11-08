import numpy as np

from mlsrag.context import BudgetedContext, DedupeContext, apply_budget, estimate_tokens, truncate
from mlsrag.trace import Trace

TEXTS = {
    "a": "One two three four five six seven eight nine ten.",  # 10 words = 13 tokens
    "b": "First sentence here. Second sentence follows now. Third one ends it.",  # 11 words
    "c": "Never reached.",
}


class Data:
    def record_text(self, rid):
        return TEXTS[rid]


def test_estimate_tokens_is_words_times_1_3():
    assert estimate_tokens(TEXTS["a"]) == 13


def test_truncate_keeps_whole_sentences():
    assert truncate(TEXTS["b"], 9) == "First sentence here. Second sentence follows now."
    assert truncate(TEXTS["b"], 3) == ""


def test_budget_is_respected_and_last_record_cut_at_sentence():
    texts, truncated = apply_budget([TEXTS["a"], TEXTS["b"], TEXTS["c"]], 22)
    assert texts == [TEXTS["a"], "First sentence here. Second sentence follows now."]
    assert truncated and sum(estimate_tokens(t) for t in texts) <= 22
    assert apply_budget([TEXTS["a"]], 100) == ([TEXTS["a"]], False)


def test_budgeted_prompt_and_counters():
    trace = Trace("r", "q", "c")
    prompt = BudgetedContext(Data(), budget_tokens=22).build("q?", ["a", "b", "c"], trace)
    assert "[Document 1]\n" + TEXTS["a"] in prompt
    assert "[Document 2]\nFirst sentence here. Second sentence follows now." in prompt
    assert "Third" not in prompt and "Never" not in prompt
    assert trace.counters == {"context.records_used": 2, "context.tokens": 22, "context.truncated": 1}


def test_dedupe_drops_near_duplicates_before_the_budget():
    vectors = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]], dtype=np.float32)  # a and b nearly equal
    TEXTS["s"] = "A summary without a vector."
    trace = Trace("r", "q", "c")
    ctx = DedupeContext(Data(), vectors, ["a", "b", "c"], threshold=0.9, budget_tokens=800)
    prompt = ctx.build("q?", ["a", "b", "c", "s"], trace)
    assert "Second sentence" not in prompt
    assert "[Document 2]\nNever reached." in prompt and "[Document 3]\nA summary" in prompt
    assert trace.counters["context.dropped_duplicates"] == 1
