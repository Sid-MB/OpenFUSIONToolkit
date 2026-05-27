#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=john
#SBATCH --array=0-399%16
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Purpose:
#   Slurm array worker for CPU trajectory generation on john. Each array task
#   maps its SLURM_ARRAY_TASK_ID to a trajectory chunk and runs
#   collect_trajectories_delta.py for that chunk.
#
# Should you call this directly?
#   Usually no. Call ./run_scripts/submit_collect_trajectories_cpu_array.sh for
#   the standard full run; it sets the output directory and submits this script
#   with the right array shape.
#
# Direct-use example, only when you intentionally want manual sbatch control:
#   OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_manual_$(date +%Y%m%d_%H%M%S) \
#     START_IDX=600 END_IDX=1000 USE_INITIAL_RELAX_CACHE=0 N_WORKERS=1 CHUNK_SIZE=1 \
#     sbatch --cpus-per-task=4 --mem=16G --array=0-399%16 \
#       run_scripts/collect_trajectories_cpu_array.sh
#
# Slurm array syntax:
#   --array=0-399%16 creates task IDs 0..399, with at most 16 tasks running.
#   With CHUNK_SIZE=1 and START_IDX=600, task 0 runs [600, 601), task 1 runs
#   [601, 602), and task 399 runs [999, 1000).
#
# Resource scaling note:
#   The current best-supported shape is one trajectory worker per Slurm task,
#   with about four CPUs allocated to that worker. The standard 64-total-CPU
#   run uses `%16` array concurrency. N_WORKERS>1 can increase RAM pressure and
#   makes it harder to tell which trajectory is slow.
#
# Shared initial relax cache is optional. Set USE_INITIAL_RELAX_CACHE=0 to run
# without it, or build it first and submit this script with --dependency=afterok.

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

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-4}"
N_WORKERS="${N_WORKERS:-1}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( TOTAL_CPUS / N_WORKERS ))}"
if [ "${THREADS_PER_WORKER}" -lt 1 ]; then
  THREADS_PER_WORKER=1
fi

N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
SEED="${SEED:-42}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-7200}"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
CHUNK_START=$(( START_IDX + ARRAY_TASK_ID * CHUNK_SIZE ))
CHUNK_END=$(( CHUNK_START + CHUNK_SIZE ))
if [ "${CHUNK_END}" -gt "${END_IDX}" ]; then
  CHUNK_END="${END_IDX}"
fi

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_${ARRAY_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}/chunk_${ARRAY_TASK_ID}_${CHUNK_START}_${CHUNK_END}}"
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-${OUTPUT_BASE_DIR}/initial_relax_state.json}"

# Force the CPU path even if this script is run from a GPU-capable login node.
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1

# Keep native math/OpenMP libraries from oversubscribing cores across workers.
export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OFT_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"

echo "Running on host: $(hostname)"
echo "ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "ARRAY_TASK_ID=${ARRAY_TASK_ID}"
echo "TOTAL_CPUS=${TOTAL_CPUS}"
echo "N_WORKERS=${N_WORKERS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "SEED=${SEED}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "TRAJECTORY_TIMEOUT_SECONDS=${TRAJECTORY_TIMEOUT_SECONDS}"
echo "CHUNK_START=${CHUNK_START}"
echo "CHUNK_END=${CHUNK_END}"
echo "OFT_NUM_THREADS=${OFT_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"

if [ "${CHUNK_START}" -ge "${CHUNK_END}" ]; then
  echo "Array task has no trajectories in requested range; exiting."
  exit 0
fi

mkdir -p "${OUTPUT_BASE_DIR}" "${OUTPUT_DIR}"

if [ "${USE_INITIAL_RELAX_CACHE:-1}" != "0" ]; then
  if [ ! -s "${INITIAL_RELAX_CACHE}" ]; then
    echo "ERROR: shared initial relax cache is missing: ${INITIAL_RELAX_CACHE}" >&2
    echo "Submit collect_initial_relax_cache_cpu.sh first and submit this array with --dependency=afterok:<cache_job_id>." >&2
    exit 2
  fi
  echo "Using shared initial relax cache: ${INITIAL_RELAX_CACHE}"
  CACHE_ARGS=(--initial_relax_cache "${INITIAL_RELAX_CACHE}")
else
  CACHE_ARGS=(--no_initial_relax_cache)
fi

uv run python collect_trajectories_delta.py \
  --n_trajectories "${N_TRAJECTORIES}" \
  --seed "${SEED}" \
  --start_idx "${CHUNK_START}" \
  --end_idx "${CHUNK_END}" \
  --n_workers "${N_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_loop "${MAX_LOOP}" \
  --grid_size "${GRID_SIZE}" \
  --trajectory_timeout_seconds "${TRAJECTORY_TIMEOUT_SECONDS}" \
  "${CACHE_ARGS[@]}" \
  "$@"
