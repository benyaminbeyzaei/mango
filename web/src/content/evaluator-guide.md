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
