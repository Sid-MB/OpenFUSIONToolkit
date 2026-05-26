#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --array=0-49%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Example dependency workflow:
#   export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)
#   cache_jid=$(START_IDX=600 END_IDX=1000 sbatch --parsable run_scripts/collect_initial_relax_cache_cpu.sh)
#   START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
#     sbatch --dependency=afterok:${cache_jid} --array=0-19%4 run_scripts/collect_trajectories_cpu_array.sh
#
# Slurm array syntax:
#   --array=0-19%4 creates task IDs 0..19, with at most 4 tasks running at once.
#   With CHUNK_SIZE=20 and START_IDX=600, task 0 runs [600, 620), task 1 runs
#   [620, 640), and task 19 runs [980, 1000).
#
# This script expects the shared initial relax cache to already exist. Build it
# first with collect_initial_relax_cache_cpu.sh and use --dependency=afterok.

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
N_WORKERS="${N_WORKERS:-${TOTAL_CPUS}}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( TOTAL_CPUS / N_WORKERS ))}"
if [ "${THREADS_PER_WORKER}" -lt 1 ]; then
  THREADS_PER_WORKER=1
fi

N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
SEED="${SEED:-42}"

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
echo "CHUNK_START=${CHUNK_START}"
echo "CHUNK_END=${CHUNK_END}"
echo "OFT_NUM_THREADS=${OFT_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
  "${CACHE_ARGS[@]}" \
  "$@"
