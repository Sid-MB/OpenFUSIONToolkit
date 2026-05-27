#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=john
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/IQL.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the collected dataset root}"
RUN_ID="${RUN_ID:-$(basename "${DATASET_DIR}")_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-out/iql/${RUN_ID}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_STEPS="${NUM_STEPS:-100000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-iql-training}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
RESUME_FROM="${RESUME_FROM:-auto}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_STEPS=${NUM_STEPS}"
echo "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
echo "LOG_INTERVAL=${LOG_INTERVAL}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "RUN_NAME=${RUN_NAME}"
echo "RESUME_FROM=${RESUME_FROM}"

uv run python IQL.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_steps "${NUM_STEPS}" \
  --project "${WANDB_PROJECT}" \
  --run_name "${RUN_NAME}" \
  --resume_from "${RESUME_FROM}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}"
