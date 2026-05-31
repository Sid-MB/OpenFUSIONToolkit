#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=jag-standard
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Purpose:
#   Build one shared initial TORAX relax cache on a jag-standard GPU node and
#   exit. It uses `uv run --extra cuda13` and the Python entrypoint checks that
#   JAX can see the GPU.
#
# Should you call this directly?
#   Usually no. The recommended production path is the CPU array submit helper.
#   Call this only when you are deliberately benchmarking or producing a GPU-built
#   shared relax cache for a dependent CPU array.
#
# Direct-use example, only when you intentionally want just the GPU-built cache:
#   OUTPUT_BASE_DIR=./rl_dataset_delta_cache_gpu_$(date +%Y%m%d_%H%M%S) \
#     START_IDX=600 END_IDX=1000 sbatch run_scripts/collect_initial_relax_cache_gpu.sh

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
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_gpu_cache_cpu_array_${RUN_ID}}"
# Shared keyed initial-relax cache. INITIAL_RELAX_CACHE (explicit path) overrides;
# otherwise a keyed file in INITIAL_RELAX_CACHE_DIR is used (resolved below).
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-}"
INITIAL_RELAX_CACHE_DIR="${INITIAL_RELAX_CACHE_DIR:-}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_BASE_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/collect_initial_relax_cache_gpu-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/collect_initial_relax_cache_gpu-${SLURM_JOB_ID:-$$}.err" >&2)

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OFT_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [ -z "${INITIAL_RELAX_CACHE}" ]; then
  INITIAL_RELAX_CACHE="$(uv run --extra cuda13 python collect_trajectories_delta.py \
    --print_initial_relax_cache_path --grid_size "${GRID_SIZE}" \
    ${INITIAL_RELAX_CACHE_DIR:+--initial_relax_cache_dir "${INITIAL_RELAX_CACHE_DIR}"})"
fi

echo "Running on host: $(hostname)"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "SEED=${SEED}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "INITIAL_RELAX_CACHE_DIR=${INITIAL_RELAX_CACHE_DIR:-<default>}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"

mkdir -p "${OUTPUT_BASE_DIR}"

uv run --extra cuda13 python collect_trajectories_delta.py \
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
