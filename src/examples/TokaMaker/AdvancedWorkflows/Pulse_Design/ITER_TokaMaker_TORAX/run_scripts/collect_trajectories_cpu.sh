#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Purpose:
#   Run trajectories in a single non-array Slurm job on the john CPU partition.
#   This is useful for diagnostics or small slices because all requested
#   trajectories run inside one Slurm allocation.
#
# Should you call this directly?
#   Not for the standard full run. Use
#   ./run_scripts/submit_collect_trajectories_cpu_array.sh for production, since
#   it splits trajectories across independent Slurm array tasks.
#
# Direct-use example, for a small diagnostic slice:
#   OUTPUT_DIR=./rl_dataset_delta_cpu_diag_$(date +%Y%m%d_%H%M%S) \
#     START_IDX=600 END_IDX=601 N_WORKERS=1 sbatch run_scripts/collect_trajectories_cpu.sh

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

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-20}"
TOTAL_CPUS="${CPUS_PER_TASK}"
N_WORKERS="${N_WORKERS:-1}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( TOTAL_CPUS / N_WORKERS ))}"
if [ "${THREADS_PER_WORKER}" -lt 1 ]; then
  THREADS_PER_WORKER=1
fi
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_${RUN_ID}}"
N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-600}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
SEED="${SEED:-42}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-7200}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/collect_trajectories_cpu-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/collect_trajectories_cpu-${SLURM_JOB_ID:-$$}.err" >&2)

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
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "TOTAL_CPUS=${TOTAL_CPUS}"
echo "N_WORKERS=${N_WORKERS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "OFT_NUM_THREADS=${OFT_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "SEED=${SEED}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "TRAJECTORY_TIMEOUT_SECONDS=${TRAJECTORY_TIMEOUT_SECONDS}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

uv run python collect_trajectories_delta.py \
  --n_trajectories "${N_TRAJECTORIES}" \
  --seed "${SEED}" \
  --start_idx "${START_IDX}" \
  --end_idx "${END_IDX}" \
  --n_workers "${N_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_loop "${MAX_LOOP}" \
  --grid_size "${GRID_SIZE}" \
  --trajectory_timeout_seconds "${TRAJECTORY_TIMEOUT_SECONDS}" \
  "$@"
