#!/usr/bin/env bash
#SBATCH --account=nlp
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=john
#SBATCH --mail-user=siddharth@cs.stanford.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# collection_wandb.sh — Log one completed dataset collection to a separate W&B project.
#
# This is a lightweight telemetry-only job. It reads the completed dataset root,
# records a summary of the collection contract and replay-cache state, and exits.
# Use it when you want collection visibility in wandb.com without mixing dataset
# telemetry into the training/eval project.
#
# Required env vars:
#   DATASET_DIR         completed dataset root containing run_manifest.json
#   WANDB_PROJECT       collection telemetry project (default: iql-collection)
#
# Optional env vars:
#   WANDB_GROUP         group related collection runs together
#   WANDB_MODE          online, offline, or disabled
#
# Typical use:
#   SUBMIT_COLLECTION_WANDB=1 \
#   COLLECTION_WANDB_PROJECT=iql-collection \
#   ./run_scripts/submit_collect_trajectories_cpu_array.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/run_scripts/log_collection_wandb.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"
export UV_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER/uv_cache}"
mkdir -p "${UV_CACHE_DIR}"

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the completed dataset root}"
export OFT_DISABLE_JAX_COMPILE_CACHE="${OFT_DISABLE_JAX_COMPILE_CACHE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-iql-collection}"
RUN_NAME="${RUN_NAME:-$(basename "${DATASET_DIR%/}")}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${DATASET_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/collection_wandb-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/collection_wandb-${SLURM_JOB_ID:-$$}.err" >&2)

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "RUN_NAME=${RUN_NAME}"
echo "WANDB_GROUP=${WANDB_GROUP:-<unset>}"
echo "WANDB_MODE=${WANDB_MODE:-<unset>}"

args=(
  --dataset_dir "${DATASET_DIR}"
  --project "${WANDB_PROJECT}"
  --run_name "${RUN_NAME}"
)
if [ -n "${WANDB_GROUP:-}" ]; then
  args+=(--group "${WANDB_GROUP}")
fi
if [ -n "${WANDB_MODE:-}" ]; then
  args+=(--mode "${WANDB_MODE}")
fi

echo "log_collection_wandb.py args: ${args[*]}"
uv run python run_scripts/log_collection_wandb.py "${args[@]}"
