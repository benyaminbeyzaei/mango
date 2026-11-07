"""Phrases stage: noun chunks and named entities of every selected record (spaCy, no LLM).

A phrase is normalised to its lowercase lemmas without stop words and punctuation, so
"the Roth IRAs" and "Roth IRA" are the same phrase ("roth ira"). Numeric entity types
(dates, money, percentages, ...) are dropped: they connect unrelated records.
"""
import json

SKIP_ENTITY_TYPES = {"CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"}
_nlp = None


def load_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def normalise(tokens) -> str:
    """Lowercase lemmas of the non-stop, non-punctuation tokens; "" if none has a letter."""
    words = [t.lemma_.lower() for t in tokens if not (t.is_stop or t.is_punct or t.is_space)]
    phrase = " ".join(w for w in words if w.strip())
    return phrase if any(c.isalpha() for c in phrase) else ""


def doc_phrases(doc) -> list:
    spans = list(doc.noun_chunks) + [e for e in doc.ents if e.label_ not in SKIP_ENTITY_TYPES]
    return sorted({p for p in (normalise(s) for s in spans) if p})


def question_ngrams(text) -> set:
    """Every run of consecutive normalised words of the question: a node matches the question
    when its phrase is one of these (the same normalisation as the records)."""
    words = normalise(load_nlp()(text)).split()
    return {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, len(words) + 1)}


def run(inputs, config, ctx):
    out = ctx.out_dir / "phrases.jsonl"
    rows = []  # resume: records already written are kept and skipped
    for line in out.read_text(encoding="utf-8").splitlines() if out.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # cut off by an interrupt: redo that record
            pass
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    done = {r["record_id"] for r in rows}
    records = [json.loads(line) for line in (inputs["select"] / "records.jsonl").open(encoding="utf-8")]
    todo = [r for r in records if r["record_id"] not in done]
    ctx.log(f"phrases: {len(done)} records done, {len(todo)} to go")
    nlp = load_nlp()
    with out.open("a", encoding="utf-8") as f:
        docs = nlp.pipe((r["text"] for r in todo), batch_size=128, n_process=config.get("processes", 1))
        for i, (r, doc) in enumerate(zip(todo, docs, strict=True), start=1):
            f.write(json.dumps({"record_id": r["record_id"], "phrases": doc_phrases(doc)}, ensure_ascii=False) + "\n")
            if i % 2000 == 0:
                f.flush()
                ctx.log(f"phrases: {i}/{len(todo)}")
    return {"records": len(records), "processed_this_run": len(todo)}
