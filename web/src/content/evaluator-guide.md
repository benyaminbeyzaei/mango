# Evaluator guide

These are the course's own commands (from `rag/README.md`) for evaluating the system on **full splits**. They run on the university teaching cluster, in a GPU session, not in this app. Their numbers are the markers' yardstick; the platform's results should agree with them.

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

Prints Recall@5/10/20/100, MRR@10 and NDCG@10. Use `--split dev` for the 125 questions we call **test** (the course calls them dev).

## 5. Generate and judge answers on a full split

```bash
python rag/scripts/generate_answers.py --device cpu --split practice --output rag/results/practice_answers.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/practice_answers.jsonl --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/practice_answers.jsonl --out-dir rag/results/practice_judge
```

The dry run checks candidate IDs and coverage without calling ELM. The last command scores correctness, grounding and completeness (0–4 each; normalised score = sum / 12). Choose a new candidate file and output directory for each experiment. Set `ELM_JUDGE_MODEL` in `rag/.env` for a separate judge model, or pass `--repeats` for repeated judging.

> Generation and judging send one ELM request per query (and per repeat): try `--limit 5` with `--allow-subset` first.

## 6. Optional: rerun our comparisons

The platform (`rag/extensions/platform/`, see its `README.md`) runs on any machine with the course data and needs no GPU, since it uses cached or whisper-1 transcripts. Install it in the RAG environment with `pip install -e "rag/extensions/platform[dev]"`, or use Docker (`docker compose run --rm cli mlsrag ...`). Every committed result is summarised in `rag/extensions/platform/docs/results.md`.

Question sets: `practice` (350), `test` (125, the course's dev split), and `practice-20`, a fixed 20-question sample for quick checks. Methods that call an LLM at query time (grep, LLM-routed TOC) were only run on `practice-20`.

Transcript caches are not in git, so a fresh clone first fills them: 125 whisper-1 calls for `test` (about 690 audio s, $0.07) and 350 for `practice` (about 1,800 s, $0.18). The command is resumable.

```bash
mlsrag transcribe --asr whisper1 --subset test --allow-full --out rag/.cache/transcripts/whisper1-dev.jsonl
mlsrag estimate --config baseline --subset test --judge --fresh      # calls for one config, nothing sent
mlsrag matrix --matrix rag/extensions/platform/configs/matrices/matrix-final-test.yaml --fresh --allow-full --yes
```

Cost once the cache is filled. All chat models are free university-hosted ELM models; wall time is with one request at a time and leaves out judging.

| Matrix file (`configs/matrices/`) | Question set | Mode | ELM calls | Wall time |
|---|---|---|---|---|
| `matrix-retrievers-test.yaml` / `-practice.yaml` (retrievers without an LLM) | test / practice | `--no-generate` | 0 | ~2 / ~5 min |
| `matrix-encoders-test.yaml` / `-practice.yaml` (encoder policies) | test / practice | `--no-generate` | 0 | minutes |
| `matrix-rerank-practice.yaml` (first reranker settings) | practice | `--no-generate` | 0 | ~3.5 h on CPU |
| `matrix-rerank-models-test.yaml` (MiniLM, bge-reranker-v2-m3, mxbai) | test | `--no-generate` | 0 | ~4 h on CPU |
| `matrix-final-test.yaml` (baseline, MiniLM rerank, RAPTOR-tree summaries) | test | generate + judge | 375 + 375 judge | ~9 h + judge |
| `matrix-final-bge-test.yaml`, `matrix-final-mxbai-test.yaml` | test | generate + judge | 125 + 125 judge each | ~3-4 h each |
| `matrix-courseasr-test.yaml` (baseline and RAPTOR-tree, our ASR) | test | generate + judge | 250 + 250 judge | ~5 h + judge |

The course-ASR row needs `rag/.cache/transcripts/course-dev.jsonl`, which only the cluster can make. Without `--allow-full` every command refuses a full question set; without `--yes` a matrix prints its estimate and waits for `y`.
