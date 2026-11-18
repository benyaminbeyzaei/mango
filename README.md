# MLS 2026 coursework: CUDA ASR kernels and a RAG systems study

Our work for INFR11269 Machine Learning Systems (University of Edinburgh, autumn 2026), built on the course
repository [ed-aisys/edin-mls-26-autumn](https://github.com/ed-aisys/edin-mls-26-autumn). It has two parts:

- **Homework 1:** four CUDA kernels (RoPE, reductions with RMSNorm and softmax, tiled GEMM with fused GELU,
  FlashAttention-style attention) that run the GLM-ASR-Nano speech model.
- **RAG study:** spoken question → ASR transcript → dense retrieval over 24,000 records → answer from an ELM
  chat model, judged by the course evaluator. On top of it we compared retrievers, rerankers, summary trees,
  encoder policies and ASR systems for quality against cost.

| Folder | What it holds | Start at |
|---|---|---|
| `hw1-asr/` | the speech model and our kernels in `csrc/task*` | [hw1-asr/README.md](hw1-asr/README.md) |
| `rag/` | the course RAG pipeline, scripts and data installer (unchanged) | [rag/README.md](rag/README.md) |
| `rag/extensions/platform/` | `mlsrag`, our experiment platform: traced, cached, costed runs | [README](rag/extensions/platform/README.md) |
| `rag/extensions/results/` | every run and matrix summary we report | [docs/results.md](rag/extensions/platform/docs/results.md) |
| `web/` | web client: results, comparisons, questions, interactive queries | [web/README.md](web/README.md) |
| `report/` | LaTeX template and PDF format checker (6-page main body) | [report/README.md](report/README.md) |
| `cuda-tutorial/` | the lecture's CUDA examples | [cuda-tutorial/README.md](cuda-tutorial/README.md) |

## Results in one table

The 125 **test** questions (the course's `dev` split; nothing was tuned on it), whisper-1 transcripts, top 5 records
in the prompt, Qwen3.5-397B answers, Mistral-Small-4 judge. Answer score = (correctness + grounding +
completeness) / 12. Intervals are 95 % bootstraps over questions.

| Approach | NDCG@10 | Answer score | vs baseline (paired) |
|---|---|---|---|
| Dense search (baseline) | 0.402 | 0.687 [0.622, 0.751] | |
| Rerank, MiniLM-L6, pool 20 | 0.393 | 0.646 [0.577, 0.714] | -0.041 [-0.095, 0.011] |
| RAPTOR-tree (gpt-5-mini), summaries as evidence | 0.400 | 0.690 [0.625, 0.755] | +0.003 [-0.045, 0.051] |
| Rerank, bge-reranker-v2-m3, pool 20 | 0.460 | 0.661 [0.598, 0.719] | -0.026 [-0.082, 0.033] |
| Rerank, mxbai-rerank-base-v2, pool 20 | 0.465 | 0.678 [0.613, 0.741] | -0.009 [-0.063, 0.045] |

The two strong rerankers improve retrieval significantly; no change moves the answer score beyond the noise of
generating and judging. All tables, the reasons and the costs: [docs/results.md](rag/extensions/platform/docs/results.md).

## Setup

Everything with a GPU runs on the university cluster (log in with `ssh YOUR_UUN@ssh.inf.ed.ac.uk`, then
`ssh mlp.inf.ed.ac.uk`; the head node has no GPU). The platform and web client also run on a laptop without one.

```bash
git clone <this repo> && cd edin-mls-26-autumn
git lfs install && git lfs pull --include="rag-assets.zip"     # course data, stored with Git LFS

# ASR environment (head node, once): conda env mls-asr and the 4.5 GB model
cd hw1-asr && bash setup_env.sh && cd ..

# RAG environment (once)
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m venv ~/mls-rag-env && source ~/mls-rag-env/bin/activate
python -m pip install -r rag/requirements.txt
unzip rag-assets.zip -d ~/mls-rag-assets
python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
cp rag/.env.example rag/.env      # then add your own ELM_API_KEY; never commit this file
```

## Homework 1: kernels

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH && conda activate mls-asr && cd hw1-asr
make && make test            # all four kernel tests (make test1 ... test4 for one)
./run.sh --cuda none         # PyTorch reference transcription of test_audio.wav
./run.sh --cuda task1,task2  # with some of our kernels switched on
./run.sh --profile           # our kernels against PyTorch/cuBLAS
```

Only the five `csrc/task*` files are ours; the rest of `hw1-asr/` stays as the course gave it.

## RAG: course baseline and our platform

The course scripts are the reference for reported numbers (full guide: [rag/README.md](rag/README.md)):

```bash
python rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl
python rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output rag/results/sample_answers.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/sample_answers.jsonl --allow-subset --out-dir rag/results/sample_judge
```

Our experiments run through `mlsrag`, which wraps the same code:

```bash
uv pip install -e "rag/extensions/platform[dev]"
mlsrag run --config baseline --subset practice-20 --no-generate    # retrieval only, no ELM call
docker compose up                                                   # API + web client at http://localhost:8000
```

Question sets: `practice-20` (quick checks), `practice` (350, for choosing settings) and `test` (125, for reporting).
Generation and judging send one ELM request per question, so start with `practice-20`.

## Report

Use `report/latex_template/` unchanged in format, keep the main body to 6 pages, include the AI-use declaration,
and run `report/format_checker/check_report_format.py` on the PDF before submitting.

## License

CC0 1.0 Universal. See [LICENSE](LICENSE).
