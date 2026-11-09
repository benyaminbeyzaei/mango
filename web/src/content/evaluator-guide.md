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

The platform (`rag/extensions/platform/`, see its `README.md`) runs on any machine with the course data; no GPU, since it uses cached or whisper-1 transcripts. Install it in the RAG environment with `pip install -e "rag/extensions/platform[dev]"`, or use Docker (`docker compose run --rm cli mlsrag ...`). The committed results are smoke and 20 % subsets. A full split needs a copy of a matrix file with `subset: dev-all` (or `practice-all`) and both flags:

```bash
mlsrag transcribe --asr whisper1 --subset dev-all --allow-full --out rag/.cache/transcripts/whisper1-dev.jsonl
mlsrag estimate --config baseline --subset dev-all --judge --fresh      # calls for one config, nothing sent
mlsrag matrix --matrix my-matrix-b-dev-all.yaml --fresh --allow-full --yes
```

Estimated cost on `dev-all` (125 queries), after the transcript cache is filled. Transcript caches are not in git, so a fresh clone first needs 125 whisper-1 calls for `dev-all` (about 690 audio s, $0.07) and 350 for `practice-all` (about 1,800 s, $0.18); the `transcribe` command above is resumable and skips queries already cached. The chat models are free university-hosted ELM models; wall time is at one request at a time, judging not included:

| Matrix | Mode | ELM calls | Wall time |
|---|---|---|---|
| A, 12 retrieval configs (`matrix-a-retrieval-dev.yaml`) | `--no-generate` | 1,375 | ~33 min |
| A on `practice-all` (350 queries) | `--no-generate` | 3,850 | ~1.5 h |
| B, 8 configs, generation + judge (`matrix-b-answers.yaml`) | generate + judge | 2,000 + 1,000 judge | ~15 h + judge |
| B with the budgeted builder (`matrix-b-answers-budgeted.yaml`) | generate + judge | 2,000 + 1,000 judge | ~15 h + judge |
| C, encoder policies (`matrix-c-secondary.yaml`) | `--no-generate` | 0 | minutes |

Without `--allow-full` every command refuses a full split; without `--yes` the matrix prints its estimate and waits for `y`.
