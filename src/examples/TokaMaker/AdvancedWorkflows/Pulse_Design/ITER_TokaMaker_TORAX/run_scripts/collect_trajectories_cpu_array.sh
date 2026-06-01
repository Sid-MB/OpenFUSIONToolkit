#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --mail-user=siddharth@cs.stanford.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-399%32
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Purpose:
#   Slurm array worker for CPU trajectory generation on john. Each array task
#   maps its SLURM_ARRAY_TASK_ID to a trajectory chunk and runs
#   collect_trajectories_delta.py for that chunk.
#
# Should you call this directly?
#   Usually no. Call ./run_scripts/submit_collect_trajectories_cpu_array.sh for
#   the standard full run; it initializes the shared dataset manifest/actions,
#   sets the output directory, and submits this script with the right array shape.
#
# Direct-use example, only when you intentionally want manual sbatch control:
#   export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_manual
#   export N_TRAJECTORIES=1000 SEED=42 MAX_LOOP=2 GRID_SIZE=51
#   uv run python collect_trajectories_delta.py \
#     --n_trajectories "${N_TRAJECTORIES}" --seed "${SEED}" \
#     --output_dir "${OUTPUT_BASE_DIR}" --max_loop "${MAX_LOOP}" \
#     --grid_size "${GRID_SIZE}" --init_dataset_only
#   START_IDX=600 END_IDX=1000 USE_INITIAL_RELAX_CACHE=0 N_WORKERS=1 CHUNK_SIZE=1 \
#     sbatch --cpus-per-task=4 --mem=128G --array=0-399%32 \
#       run_scripts/collect_trajectories_cpu_array.sh
#
# Slurm array syntax:
#   --array=0-399%32 creates task IDs 0..399, with at most 32 tasks running.
#   With CHUNK_SIZE=1 and START_IDX=600, task 0 runs [600, 601), task 1 runs
#   [601, 602), and task 399 runs [999, 1000).
#
# Resource scaling note:
#   The current best-supported shape is one trajectory worker per Slurm task,
#   with about four CPUs allocated to that worker. The standard 128-total-CPU
#   run uses `%32` array concurrency. N_WORKERS>1 can increase RAM pressure and
#   makes it harder to tell which trajectory is slow.
#
# Shared initial relax cache is controlled explicitly by USE_INITIAL_RELAX_CACHE.
# When it is 1, INITIAL_RELAX_CACHE must point to an existing cache and this
# script should be submitted with the appropriate afterok dependency.

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
source "${PROJECT_DIR}/run_scripts/lib/threading.sh"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "ERROR: ${name} must be set by the submit wrapper or caller." >&2
    exit 2
  fi
}

add_arg() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"
  if [ -n "${value}" ]; then
    COLLECT_ARGS+=("${flag}" "${value}")
  fi
}

add_bool_output_arg() {
  local env_name="$1"
  local true_flag="$2"
  local false_flag="$3"
  local value="${!env_name:-}"
  if [ -z "${value}" ]; then
    return
  fi
  if [ "${value}" = "0" ] || [ "${value}" = "false" ] || [ "${value}" = "False" ]; then
    COLLECT_ARGS+=("${false_flag}")
  else
    COLLECT_ARGS+=("${true_flag}")
  fi
}

echo_env_or_argparse_default() {
  local name="$1"
  if [ -n "${!name:-}" ]; then
    echo "${name}=${!name}"
  else
    echo "${name}=<argparse default>"
  fi
}

require_env SLURM_CPUS_PER_TASK
require_env SLURM_ARRAY_TASK_ID
require_env SLURM_ARRAY_JOB_ID
require_env N_TRAJECTORIES
require_env START_IDX
require_env END_IDX
require_env N_WORKERS
require_env CHUNK_SIZE
require_env OUTPUT_BASE_DIR
require_env USE_INITIAL_RELAX_CACHE

RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_BASE_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/collect_trajectories-${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/collect_trajectories-${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}.err" >&2)

TOTAL_CPUS="${SLURM_CPUS_PER_TASK}"
if [ -z "${THREADS_PER_WORKER:-}" ]; then
  THREADS_PER_WORKER="$(( TOTAL_CPUS / N_WORKERS ))"
fi
THREADS_PER_WORKER="$(oft_cap_thread_budget "${THREADS_PER_WORKER}" "trajectory array worker")"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
CHUNK_START=$(( START_IDX + ARRAY_TASK_ID * CHUNK_SIZE ))
CHUNK_END=$(( CHUNK_START + CHUNK_SIZE ))
if [ "${CHUNK_END}" -gt "${END_IDX}" ]; then
  CHUNK_END="${END_IDX}"
fi

if [ -z "${CHUNK_DIR:-}" ]; then
  CHUNK_DIR="${OUTPUT_BASE_DIR}/chunks/chunk_${ARRAY_TASK_ID}_${CHUNK_START}_${CHUNK_END}"
fi

# Force the CPU path even if this script is run from a GPU-capable login node.
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export OFT_DISABLE_JAX_COMPILE_CACHE="${OFT_DISABLE_JAX_COMPILE_CACHE:-1}"
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
echo "PHYSICAL_CORES=$(oft_detect_physical_cores || echo '<unknown>')"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo_env_or_argparse_default SEED
echo_env_or_argparse_default MAX_LOOP
echo_env_or_argparse_default GRID_SIZE
echo_env_or_argparse_default OBSERVATION_MODE
echo_env_or_argparse_default TRAJECTORY_TIMEOUT_SECONDS
echo_env_or_argparse_default SAVE_REPLAY_SHARD
echo_env_or_argparse_default SAVE_FULL_ZARR
echo_env_or_argparse_default SAVE_JSON
echo_env_or_argparse_default SAVE_STATS_FOR_REWARD_RECALC
echo "CHUNK_START=${CHUNK_START}"
echo "CHUNK_END=${CHUNK_END}"
echo "OFT_NUM_THREADS=${OFT_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "CHUNK_DIR=${CHUNK_DIR}"
echo_env_or_argparse_default INITIAL_RELAX_CACHE

if [ "${CHUNK_START}" -ge "${CHUNK_END}" ]; then
  echo "Array task has no trajectories in requested range; exiting."
  exit 0
fi

mkdir -p "${OUTPUT_BASE_DIR}" "${CHUNK_DIR}"

COLLECT_ARGS=(
  --n_trajectories "${N_TRAJECTORIES}"
  --start_idx "${CHUNK_START}"
  --end_idx "${CHUNK_END}"
  --n_workers "${N_WORKERS}"
  --output_dir "${OUTPUT_BASE_DIR}"
  --chunk_dir "${CHUNK_DIR}"
  --require_existing_dataset
)

add_arg SEED --seed
add_arg MAX_LOOP --max_loop
add_arg GRID_SIZE --grid_size
add_arg OBSERVATION_MODE --observation_mode
add_arg TRAJECTORY_TIMEOUT_SECONDS --trajectory_timeout_seconds
add_bool_output_arg SAVE_REPLAY_SHARD --save_replay_shard --no_save_replay_shard
add_bool_output_arg SAVE_FULL_ZARR --save_full_zarr --no_save_full_zarr
add_bool_output_arg SAVE_JSON --save_json --no_save_json
add_bool_output_arg SAVE_STATS_FOR_REWARD_RECALC --save_stats_for_reward_recalc --no_save_stats_for_reward_recalc

if [ "${USE_INITIAL_RELAX_CACHE}" != "0" ]; then
  # INITIAL_RELAX_CACHE is normally exported by the submit helper; resolve the
  # keyed path here too so direct invocations work.
  if [ -z "${INITIAL_RELAX_CACHE:-}" ]; then
    INITIAL_RELAX_CACHE="$(uv run python collect_trajectories_delta.py \
      --print_initial_relax_cache_path --grid_size "${GRID_SIZE:-51}" \
      ${INITIAL_RELAX_CACHE_DIR:+--initial_relax_cache_dir "${INITIAL_RELAX_CACHE_DIR}"})"
  fi
  if [ ! -s "${INITIAL_RELAX_CACHE}" ]; then
    echo "ERROR: shared initial relax cache is missing: ${INITIAL_RELAX_CACHE}" >&2
    echo "Submit collect_initial_relax_cache_cpu.sh first and submit this array with --dependency=afterok:<cache_job_id>." >&2
    exit 2
  fi
  echo "Using shared initial relax cache: ${INITIAL_RELAX_CACHE}"
  COLLECT_ARGS+=(--initial_relax_cache "${INITIAL_RELAX_CACHE}")
else
  COLLECT_ARGS+=(--no_initial_relax_cache)
fi

echo "collect_trajectories_delta.py args: ${COLLECT_ARGS[*]} $*"

uv run python collect_trajectories_delta.py \
  "${COLLECT_ARGS[@]}" \
  "$@"
