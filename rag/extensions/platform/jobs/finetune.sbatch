#!/bin/bash
# Fine-tune the MiniLM reranker on practice on one L40S. Submit with .ai/bin/cluster.sh from this folder:
#   cluster.sh submit <run> rag/extensions/platform/jobs/finetune.sbatch                # cross-validation
#   cluster.sh submit <run> rag/extensions/platform/jobs/finetune.sbatch --export=ALL,MODE=train,LR=2e-5,EPOCHS=2
# Needs jobs/finetune-data.json (python -m mlsrag.finetune prepare jobs/finetune-data.json, on the laptop).
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=finetune-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."  # rag/extensions/platform
source ~/mls/envs/rerank-ft/bin/activate
export HF_HOME=~/.cache/huggingface HF_HUB_OFFLINE=1
hostname; nvidia-smi -L
if [ "${MODE:-cv}" = cv ]; then
    python -m mlsrag.finetune cv jobs/finetune-data.json outputs
else
    python -m mlsrag.finetune train jobs/finetune-data.json outputs --lr "$LR" --epochs "$EPOCHS"
fi
