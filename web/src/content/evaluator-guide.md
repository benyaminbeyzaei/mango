# Evaluator guide

These are the course's own commands (from `rag/README.md`) for evaluating the system on **full splits**. They run on the university teaching cluster, in a GPU session, not in this app. Their numbers are the markers' yardstick: the results shown in this client come from the platform's subsets and should agree with them.

Run everything from the course repository root, where `hw1-asr/` and `rag/` are both visible.

## 1. Create the RAG environment, once

```bash
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m venv "$HOME/mls-rag-env"
source "$HOME/mls-rag-env/bin/activate"
python -m pip install -r rag/requirements.txt
cp rag/.env.example rag/.env
```

Add your own ELM key to `rag/.env` as `ELM_API_KEY=...`.

## 2. Install the course data, once

```bash
git lfs install
git lfs pull --include="rag-assets.zip"
mkdir -p "$HOME/mls-rag-assets"
unzip rag-assets.zip -d "$HOME/mls-rag-assets"
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
```

## 3. Start a GPU session, each time

Get a GPU allocation as the ASR guide describes, then:

```bash
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd ~/edin-mls-26-autumn
make -C hw1-asr lib
python rag/scripts/check_setup.py
export ASR_CUDA=all
```

## 4. Evaluate retrieval on a full split

```bash
python rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl
```

Prints Recall@5/10/20/100, MRR@10 and NDCG@10. Use `--split dev` for the dev split.

## 5. Generate and judge answers on a full split

```bash
python rag/scripts/generate_answers.py --device cpu --split practice --output rag/results/practice_answers.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/practice_answers.jsonl --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/practice_answers.jsonl --out-dir rag/results/practice_judge
```

The dry run checks candidate IDs and coverage without calling ELM. The last command scores correctness, grounding and completeness (0–4 each; normalised score = sum / 12). Choose a new candidate file and output directory for each experiment. Set `ELM_JUDGE_MODEL` in `rag/.env` for a separate judge model, or pass `--repeats` for repeated judging.

> Generation and judging send one ELM request per query (and per repeat): try `--limit 5` with `--allow-subset` first.

## 6. Optional: run our experiment matrices

The platform (`rag/extensions/platform/`, see its `README.md`) runs on any machine with the course data; no GPU, since it uses cached or whisper-1 transcripts. Install it in the RAG environment with `pip install -e "rag/extensions/platform[dev]"`, or use Docker (`docker compose run --rm cli mlsrag ...`). The committed results are:

- smoke runs on `practice-smoke20` for every method;
- full-split retrieval summaries (`practice-all`, `dev-all`) for every retriever that makes no LLM call at query time, and for the encoder policies (plans 029 and 030);
- the reranker choice on `practice-all` and the final three-way comparison (baseline, reranker, RAPTOR tree), retrieval on `practice-all` and answers with judge scores on `dev-all` (plan 032);
- the same final comparison on `dev-all` with our own ASR's transcripts instead of whisper-1 (plan 033).

The LLM-routed TOC and grep methods have smoke results only. To run any other matrix on a full split, copy its file with `subset: dev-all` (or `practice-all`) and pass both flags:

```bash
mlsrag transcribe --asr whisper1 --subset dev-all --allow-full --out rag/.cache/transcripts/whisper1-dev.jsonl
mlsrag estimate --config baseline --subset dev-all --judge --fresh      # calls for one config, nothing sent
mlsrag matrix --matrix my-matrix-b-dev-all.yaml --fresh --allow-full --yes
```

Transcript caches are not in git, so a fresh clone first needs 125 whisper-1 calls for `dev-all` (about 690 audio s, $0.07) and 350 for `practice-all` (about 1,800 s, $0.18); the `transcribe` command above is resumable and skips queries already cached.

Expected cost per matrix once the cache is filled. All chat models are free, university-hosted ELM models. ELM calls include the grep and TOC helper calls; wall time is with one request at a time and does not include judging.

| Matrix file (`configs/matrices/`)                                                    | Subset                 | Mode             | ELM calls         | Wall time     |
| ------------------------------------------------------------------------------------ | ---------------------- | ---------------- | ----------------- | ------------- |
| `matrix-a-retrieval-dev.yaml`, copied to `dev-all` (14 retrievers)                   | dev-all (125)          | `--no-generate`  | 1,750             | ~45 min       |
| `matrix-a-retrieval.yaml`, copied to `practice-all`                                  | practice-all (350)     | `--no-generate`  | 4,900             | ~2 h          |
| `matrix-b-answers.yaml`, copied to `dev-all` (7 configs, simple context)             | dev-all                | generate + judge | 1,875 + 875 judge | ~13 h + judge |
| `matrix-b-answers-budgeted.yaml`, copied to `dev-all` (budgeted context)             | dev-all                | generate + judge | 1,875 + 875 judge | ~13 h + judge |
| `matrix-a-free-dev-all.yaml` / `matrix-a-free-practice-all.yaml` (no-LLM retrievers) | dev-all / practice-all | `--no-generate`  | 0                 | ~2 / ~5 min   |
| `matrix-c-dev-all.yaml` / `matrix-c-practice-all.yaml` (encoder policies)            | dev-all / practice-all | `--no-generate`  | 0                 | minutes       |
| `matrix-rerank-practice-all.yaml` (reranker settings)                                | practice-all           | `--no-generate`  | 0                 | minutes       |
| `matrix-final-dev-all.yaml` (baseline, reranker, RAPTOR tree)                        | dev-all                | generate + judge | 375 + 375 judge   | ~9 h + judge  |
| `matrix-courseasr-dev-all.yaml` (baseline and RAPTOR tree, our ASR)                  | dev-all                | generate + judge | 250 + 250 judge   | ~5 h + judge  |

The rows that copy a smoke-sized matrix are the README's dev-20pct (25 queries) and practice-20pct (70 queries) estimates scaled ×5; the last two rows come from the committed runs (about 87 s and 72 s per answer). The course-ASR row needs `rag/.cache/transcripts/course-dev.jsonl`, which only the cluster can make.

Without `--allow-full` every command refuses a full split; without `--yes` the matrix prints its estimate and waits for `y`.
