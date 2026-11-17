"""Plan 035: fine-tune the MiniLM cross-encoder on the practice relevance labels.

    python -m mlsrag.finetune prepare data.json          # laptop: practice questions, dense rankings, labels, texts
    python -m mlsrag.finetune cv data.json out/          # GPU: 5-fold cross-validation -> out/cv.json
    python -m mlsrag.finetune train data.json out/ --lr 2e-5 --epochs 2   # GPU: all practice -> out/rerank-minilm-ft/
    python -m mlsrag.finetune analyse                    # after the dev-all matrix: paired intervals vs the baseline

Training pairs for a question: every labelled relevant record (label 1) and the non-relevant records in its
dense top 20 (label 0: the distractors the reranker has to push down). The query is the whisper-1 transcript,
as at test time. Loss: binary cross-entropy on the model's single logit. Dev is never read here.
"""
import argparse
import json
import random
import time

import numpy as np
import torch

from pathlib import Path

from .course import CourseData, paths
from .gateway import RERANK_MAX_LENGTH, ModelGateway
from .runner import retrieval_metrics
from .stages import ExactRetriever, load_transcripts

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"  # the revision plan 032 used
POOL = 20  # dense candidates re-scored, as in rerank-best
DEPTH = 100  # dense ranking kept for Recall@100
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare(split="practice") -> dict:
    """Everything training needs, so the GPU job reads one file: per question the whisper-1 transcript, the
    dense top-100 (gte-small on enc_a, as the baseline runs) and the labels; and the text of every record used."""
    data = CourseData(paths.data_dir)
    qrels = data.load_qrels(split)
    transcripts = load_transcripts(paths.cache_dir / "transcripts" / f"whisper1-{split}.jsonl")
    gateway, retriever = ModelGateway(), ExactRetriever("enc_a")
    rankings = {}
    for qid in qrels:
        vec = gateway.embed("gte-small", [transcripts[qid]], role="query")[0]
        rankings[qid] = (transcripts[qid], [rid for rid, _ in retriever.search(vec, DEPTH)])
    needed = {r for _, ranked in rankings.values() for r in ranked[:POOL]} | {r for q in qrels.values() for r in q}
    return {"split": split, "rankings": rankings, "qrels": qrels,
            "texts": {r: data.record_text(r) for r in sorted(needed)}}


def build_pairs(qids, rankings, qrels) -> list[tuple[str, str, float]]:
    """(transcript, record id, label): all relevant records, plus the non-relevant ones in the dense top POOL."""
    pairs = []
    for qid in qids:
        question, ranked = rankings[qid]
        pairs += [(question, rid, 1.0) for rid in sorted(qrels[qid])]
        pairs += [(question, rid, 0.0) for rid in ranked[:POOL] if rid not in qrels[qid]]
    return pairs


def rerank_order(ranked, scores) -> list[str]:
    """The top POOL re-ordered by score (ties keep the dense order), then the rest of the dense ranking."""
    head = ranked[:POOL]
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [head[i] for i in order] + ranked[POOL:]


def folds(qids, k=5, seed=SEED) -> list[list[str]]:
    qids = sorted(qids)
    random.Random(seed).shuffle(qids)
    return [qids[i::k] for i in range(k)]


def load_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(BASE_MODEL, revision=BASE_REVISION, max_length=RERANK_MAX_LENGTH, device=DEVICE)


class Texts:
    """record_text() over the texts saved by prepare()."""

    def __init__(self, texts):
        self.texts = texts

    def record_text(self, rid):
        return self.texts[rid]


def evaluate(model, qids, rankings, qrels, data) -> list[dict]:
    """Retrieval metrics per question; model=None scores the dense ranking itself."""
    rows = []
    for qid in qids:
        question, ranked = rankings[qid]
        if model is not None:
            scores = model.predict([(question, data.record_text(r)) for r in ranked[:POOL]], show_progress_bar=False)
            ranked = rerank_order(ranked, list(scores))
        rows.append({"query_id": qid, **retrieval_metrics(ranked, qrels[qid])})
    return rows


def train(model, pairs, data, lr, epochs, after_epoch=None):
    """Plain PyTorch loop: AdamW, linear warm-up over the first 10 % of steps then linear decay, seed 0."""
    torch.manual_seed(SEED)
    net, tok = model.model, model.tokenizer
    steps = epochs * ((len(pairs) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(SEED)
    for epoch in range(1, epochs + 1):
        net.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        total, start = 0.0, time.time()
        for b in range(0, len(order), BATCH):
            batch = [pairs[i] for i in order[b: b + BATCH]]
            enc = tok([q for q, _, _ in batch], [data.record_text(r) for _, r, _ in batch], truncation=True,
                      max_length=RERANK_MAX_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
            logits = net(**enc).logits.view(-1)
            loss = loss_fn(logits, torch.tensor([y for _, _, y in batch], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        net.eval()
        print(f"  epoch {epoch}: loss {total / len(pairs):.4f}, {time.time() - start:.0f} s", flush=True)
        if after_epoch:
            after_epoch(epoch)
    return model


def mean(rows, key="NDCG@10") -> float:
    return float(np.mean([r[key] for r in rows]))


def cv(prepared, lrs=(2e-5, 5e-5), epochs=3) -> dict:
    rankings, qrels, data = prepared["rankings"], prepared["qrels"], Texts(prepared["texts"])
    untuned = load_model()
    out = {"base_model": BASE_MODEL, "revision": BASE_REVISION, "pool": POOL, "batch": BATCH, "seed": SEED,
           "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", "folds": [], "settings": {}}
    for f, held in enumerate(folds(qrels)):
        train_ids = [q for q in qrels if q not in held]
        fold = {"held_out": len(held), "dense": mean(evaluate(None, held, rankings, qrels, data)),
                "untuned": mean(evaluate(untuned, held, rankings, qrels, data))}
        pairs = build_pairs(train_ids, rankings, qrels)
        print(f"fold {f}: {len(train_ids)} train questions, {len(pairs)} pairs; dense {fold['dense']:.3f}, "
              f"untuned {fold['untuned']:.3f}", flush=True)
        for lr in lrs:
            model = load_model()

            def score(epoch, model=model, lr=lr, fold=fold, held=held):
                key = f"lr={lr:g} epochs={epoch}"
                fold[key] = mean(evaluate(model, held, rankings, qrels, data))
                print(f"  {key}: NDCG@10 {fold[key]:.3f}", flush=True)

            train(model, pairs, data, lr, epochs, after_epoch=score)
        out["folds"].append(fold)
    keys = [k for k in out["folds"][0] if k not in ("held_out",)]
    out["settings"] = {k: {"mean_ndcg10": float(np.mean([f[k] for f in out["folds"]])),
                           "per_fold": [f[k] for f in out["folds"]]} for k in keys}
    tuned = {k: v for k, v in out["settings"].items() if k.startswith("lr=")}
    out["rule"] = "highest mean NDCG@10 over the 5 held-out practice folds"
    out["chosen"] = max(tuned, key=lambda k: tuned[k]["mean_ndcg10"])
    return out


def train_all(prepared, out_dir, lr, epochs):
    rankings, qrels, data = prepared["rankings"], prepared["qrels"], Texts(prepared["texts"])
    pairs = build_pairs(list(qrels), rankings, qrels)
    print(f"{len(qrels)} questions, {len(pairs)} pairs", flush=True)
    model = train(load_model(), pairs, data, lr, epochs)
    out = Path(out_dir) / "rerank-minilm-ft"
    model.save(str(out))
    meta = {"base_model": BASE_MODEL, "revision": BASE_REVISION, "lr": lr, "epochs": epochs, "batch": BATCH,
            "seed": SEED, "pool": POOL, "pairs": len(pairs), "positives": sum(y for _, _, y in pairs),
            "split": "practice (350 questions, whisper-1 transcripts)"}
    (out / "finetune.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


DEV_KEYS = ["Recall@5", "Recall@10", "MRR@10", "NDCG@10"]


def analyse(matrix="rerank-ft-dev-all") -> dict:
    """Each reranker minus the baseline per dev question (mean, 95 % bootstrap interval), and relevant records
    moved into / out of the top 5 (plan 032's top5_moves)."""
    from .final import load_queries, matrix_runs, mean_ci, top5_moves

    qrels = CourseData(paths.data_dir).load_qrels("dev")
    runs = matrix_runs(matrix)
    rows = {c: load_queries(r) for c, r in runs.items()}
    base = {r["query_id"]: r for r in rows["baseline"]}
    out = {"matrix": matrix, "runs": runs, "rows": {}}
    for config, qs in rows.items():
        out["rows"][config] = {k: mean_ci([q[k] for q in qs]) for k in DEV_KEYS}
        if config != "baseline":
            out["rows"][config]["minus_baseline"] = {
                k: mean_ci([q[k] - base[q["query_id"]][k] for q in qs]) for k in DEV_KEYS}
            out["rows"][config]["top5_moves"] = top5_moves(qs, rows["baseline"], qrels)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "cv", "train", "analyse"])
    ap.add_argument("data", nargs="?", help="the file prepare writes and cv / train read")
    ap.add_argument("out", nargs="?", default="outputs", help="folder for cv.json and the trained model")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--matrix", default="rerank-ft-dev-all", help="analyse: the dev-all matrix to compare")
    a = ap.parse_args()
    if a.command == "analyse":
        result = analyse(a.matrix)
        out = paths.results_dir / "matrices" / result["matrix"] / "analysis.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        raise SystemExit
    if a.command == "prepare":
        Path(a.data).write_text(json.dumps(prepare()), encoding="utf-8")
        print(f"wrote {a.data}")
        raise SystemExit
    prepared = json.loads(Path(a.data).read_text(encoding="utf-8"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.command == "cv":
        result = cv(prepared)
        (Path(a.out) / "cv.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"chosen {result['chosen']}; wrote {a.out}/cv.json")
    else:
        train_all(prepared, a.out, a.lr, a.epochs)
