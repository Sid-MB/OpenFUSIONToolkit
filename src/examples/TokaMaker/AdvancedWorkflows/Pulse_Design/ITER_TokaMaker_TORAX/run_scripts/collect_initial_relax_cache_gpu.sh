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

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/collect_trajectories_delta.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"
OFT_ROOT="$(cd "${PROJECT_DIR}/../../../../../../" && pwd -P)"
source "${OFT_ROOT}/scripts/oft_arch/select_oft_install.sh"

N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
SEED="${SEED:-42}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_gpu_cache_cpu_array_${RUN_ID}}"
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-${OUTPUT_BASE_DIR}/initial_relax_state.json}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OFT_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Running on host: $(hostname)"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "SEED=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"

mkdir -p "${OUTPUT_BASE_DIR}"

uv run --extra cuda13 python collect_trajectories_delta.py \
  --n_trajectories "${N_TRAJECTORIES}" \
  --seed "${SEED}" \
  --start_idx "${START_IDX}" \
  --end_idx "${START_IDX}" \
  --n_workers 1 \
  --output_dir "${OUTPUT_BASE_DIR}" \
  --initial_relax_cache "${INITIAL_RELAX_CACHE}" \
  --build_initial_relax_cache_only \
  "$@"
