#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=jag-standard
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Optional setup
# sh ./setup-env.sh

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/collect_trajectories_delta.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"
OFT_ROOT="$(cd "${PROJECT_DIR}/../../../../../../" && pwd -P)"
source "${OFT_ROOT}/scripts/oft_arch/select_oft_install.sh"

N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_${OFT_SELECTED_FLAVOR}_${RUN_ID}}"

export PYTHONUNBUFFERED=1

echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "N_WORKERS=${N_WORKERS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

uv run --extra cuda13 python collect_trajectories_delta.py \
  --n_trajectories 1000 \
  --start_idx 600 \
  --n_workers "${N_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
