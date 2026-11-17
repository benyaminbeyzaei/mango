#!/bin/bash
# Plan 036: run a retrieval-only matrix (no ELM calls) on one L40S. Submit with .ai/bin/cluster.sh from the checkout root:
#   cluster.sh submit <run> rag/extensions/platform/jobs/matrix.sbatch --export=ALL,MATRIX=matrix-rerank-models-dev-all
# The run folder needs the course assets and rag/.cache (transcripts, preprocess) copied in with rsync -L.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=matrix-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/../../../.."  # cluster.sh submits from jobs/; go to the checkout root
source ~/mls/envs/mlsrag/bin/activate
export PYTHONPATH="$PWD/rag/extensions/platform" HF_HOME=~/.cache/huggingface HF_HUB_OFFLINE=1
hostname; nvidia-smi -L
mlsrag matrix --matrix "rag/extensions/platform/configs/matrices/$MATRIX.yaml" --no-generate --fresh --allow-full --yes
