#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Purpose:
#   Build one shared initial TORAX relax cache on the john CPU partition and
#   exit. It runs collect_trajectories_delta.py with --build_initial_relax_cache_only.
#
# Should you call this directly?
#   Usually no. For the standard full run, call
#   ./run_scripts/submit_collect_trajectories_cpu_array.sh instead. That helper
#   calls this script automatically only when USE_INITIAL_RELAX_CACHE=1.
#
# Direct-use example, only when you intentionally want just the cache:
#   OUTPUT_BASE_DIR=./rl_dataset_delta_cache_cpu_$(date +%Y%m%d_%H%M%S) \
#     START_IDX=600 END_IDX=1000 sbatch run_scripts/collect_initial_relax_cache_cpu.sh

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

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-20}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-${TOTAL_CPUS}}"

N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
SEED="${SEED:-42}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_${RUN_ID}}"
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-${OUTPUT_BASE_DIR}/initial_relax_state.json}"

# Force the CPU path even if this script is run from a GPU-capable login node.
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OFT_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"

echo "Running on host: $(hostname)"
echo "TOTAL_CPUS=${TOTAL_CPUS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "SEED=${SEED}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"

mkdir -p "${OUTPUT_BASE_DIR}"

uv run python collect_trajectories_delta.py \
  --n_trajectories "${N_TRAJECTORIES}" \
  --seed "${SEED}" \
  --start_idx "${START_IDX}" \
  --end_idx "${START_IDX}" \
  --n_workers 1 \
  --output_dir "${OUTPUT_BASE_DIR}" \
  --max_loop "${MAX_LOOP}" \
  --grid_size "${GRID_SIZE}" \
  --initial_relax_cache "${INITIAL_RELAX_CACHE}" \
  --build_initial_relax_cache_only \
  "$@"
