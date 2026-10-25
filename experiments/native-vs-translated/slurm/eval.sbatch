#!/bin/bash
# One array task per model in models.tsv. Edit the partition, account and
# environment lines for your cluster, then from experiments/native-vs-translated/:
#   mkdir -p logs && sbatch slurm/eval.sbatch
#SBATCH --job-name=lga-native-vs-translated
#SBATCH --array=1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
##SBATCH --partition=gpu
##SBATCH --account=your-account
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# Activate the environment from the README, e.g.:
# source .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
ONLY="$SLURM_ARRAY_TASK_ID" ./run.sh models.tsv
