#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Recommended production entrypoint for CPU trajectory generation. This script
#   changes to the example directory, chooses a timestamped output folder,
#   initializes the shared dataset manifest/action table, optionally submits the
#   initial relax cache job, and submits the john Slurm array that runs
#   collect_trajectories_cpu_array.sh. By default, it also submits a dependent
#   john CPU job that materializes a compact replay cache for IQL.
#
# Should you call this directly?
#   Yes. For the standard full run, paste the example below into your shell.
#
# Standard full-run example (64 total allocated CPUs: 16 tasks x 4 CPUs):
#   START_IDX=600 END_IDX=1000 ARRAY_CONCURRENCY=16 CPUS_PER_TASK=4 \
#     N_WORKERS=1 CHUNK_SIZE=1 \
#     MAX_LOOP=2 GRID_SIZE=51 TRAJECTORY_TIMEOUT_SECONDS=7200 \
#     ./run_scripts/submit_collect_trajectories_cpu_array.sh
#
# The default skips the shared initial relax cache because the no-cache path has
# completed a diagnostic trajectory cleanly. Set USE_INITIAL_RELAX_CACHE=1 to
# submit a cache job and a dependent trajectory array.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_DIR}"

N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
N_WORKERS="${N_WORKERS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
SEED="${SEED:-42}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"
SLURM_MAX_ARRAY_SIZE="${SLURM_MAX_ARRAY_SIZE:-1001}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-4}}"
MEM_PER_NODE="${MEM_PER_NODE:-${SLURM_MEM_PER_NODE:-16G}}"
USE_INITIAL_RELAX_CACHE="${USE_INITIAL_RELAX_CACHE:-0}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-7200}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_REPLAY_CACHE="${SUBMIT_REPLAY_CACHE:-1}"
SUBMIT_IQL="${SUBMIT_IQL:-0}"
REPLAY_CACHE_MEM="${REPLAY_CACHE_MEM:-120G}"
REPLAY_CACHE_CPUS="${REPLAY_CACHE_CPUS:-8}"
REPLAY_CACHE_WORKERS="${REPLAY_CACHE_WORKERS:-${REPLAY_CACHE_CPUS}}"
REPLAY_CACHE_WORKER_BACKEND="${REPLAY_CACHE_WORKER_BACKEND:-process}"
OVERWRITE_REPLAY_CACHE="${OVERWRITE_REPLAY_CACHE:-0}"
REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR:-}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)}"
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-${OUTPUT_BASE_DIR}/initial_relax_state.json}"

if [ "${CHUNK_SIZE}" -lt 1 ]; then
  echo "ERROR: CHUNK_SIZE must be >= 1, got ${CHUNK_SIZE}" >&2
  exit 2
fi
if [ "${END_IDX}" -lt "${START_IDX}" ]; then
  echo "ERROR: END_IDX must be >= START_IDX, got START_IDX=${START_IDX}, END_IDX=${END_IDX}" >&2
  exit 2
fi
if [ "${END_IDX}" -gt "${N_TRAJECTORIES}" ]; then
  echo "ERROR: END_IDX must be <= N_TRAJECTORIES, got END_IDX=${END_IDX}, N_TRAJECTORIES=${N_TRAJECTORIES}" >&2
  exit 2
fi

REQUESTED_TRAJECTORIES=$(( END_IDX - START_IDX ))
ARRAY_TASK_COUNT=$(( (REQUESTED_TRAJECTORIES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
if [ "${ARRAY_TASK_COUNT}" -lt 1 ]; then
  ARRAY_TASK_COUNT=1
fi
if [ "${ARRAY_TASK_COUNT}" -gt "${SLURM_MAX_ARRAY_SIZE}" ]; then
  echo "ERROR: requested ${ARRAY_TASK_COUNT} Slurm array tasks, but MaxArraySize=${SLURM_MAX_ARRAY_SIZE}." >&2
  echo "Increase CHUNK_SIZE or split the run into multiple submissions." >&2
  exit 2
fi
if [ -z "${ARRAY_SPEC:-}" ]; then
  ARRAY_SPEC="0-$(( ARRAY_TASK_COUNT - 1 ))%${ARRAY_CONCURRENCY}"
fi

export N_TRAJECTORIES START_IDX END_IDX N_WORKERS CHUNK_SIZE SEED
export MAX_LOOP GRID_SIZE TRAJECTORY_TIMEOUT_SECONDS
export OUTPUT_BASE_DIR INITIAL_RELAX_CACHE USE_INITIAL_RELAX_CACHE

echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"
echo "N_WORKERS=${N_WORKERS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "SEED=${SEED}"
echo "REQUESTED_TRAJECTORIES=${REQUESTED_TRAJECTORIES}"
echo "ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT}"
echo "ARRAY_SPEC=${ARRAY_SPEC}"
echo "ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY}"
echo "SLURM_MAX_ARRAY_SIZE=${SLURM_MAX_ARRAY_SIZE}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MEM_PER_NODE=${MEM_PER_NODE}"
echo "USE_INITIAL_RELAX_CACHE=${USE_INITIAL_RELAX_CACHE}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "TRAJECTORY_TIMEOUT_SECONDS=${TRAJECTORY_TIMEOUT_SECONDS}"
echo "SUBMIT_REPLAY_CACHE=${SUBMIT_REPLAY_CACHE}"
echo "REPLAY_CACHE_MEM=${REPLAY_CACHE_MEM}"
echo "REPLAY_CACHE_CPUS=${REPLAY_CACHE_CPUS}"
echo "REPLAY_CACHE_WORKERS=${REPLAY_CACHE_WORKERS}"
echo "REPLAY_CACHE_WORKER_BACKEND=${REPLAY_CACHE_WORKER_BACKEND}"
echo "OVERWRITE_REPLAY_CACHE=${OVERWRITE_REPLAY_CACHE}"
echo "REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
echo "SUBMIT_IQL=${SUBMIT_IQL}"
echo "DRY_RUN=${DRY_RUN}"

if [ "${DRY_RUN}" != "0" ]; then
  echo "Dry run requested; not initializing dataset or submitting Slurm jobs."
  exit 0
fi

mkdir -p "${OUTPUT_BASE_DIR}"

echo "Initializing shared dataset manifest/action table..."
uv run python collect_trajectories_delta.py \
  --n_trajectories "${N_TRAJECTORIES}" \
  --seed "${SEED}" \
  --start_idx "${START_IDX}" \
  --end_idx "${END_IDX}" \
  --n_workers 1 \
  --output_dir "${OUTPUT_BASE_DIR}" \
  --max_loop "${MAX_LOOP}" \
  --grid_size "${GRID_SIZE}" \
  --no_initial_relax_cache \
  --init_dataset_only

dependency_args=()
if [ "${USE_INITIAL_RELAX_CACHE}" != "0" ]; then
  echo "Submitting initial relax cache job..."
  cache_jid="$(sbatch --parsable "${SCRIPT_DIR}/collect_initial_relax_cache_cpu.sh")"
  echo "cache_jid=${cache_jid}"
  dependency_args=(--dependency=afterok:${cache_jid})
else
  echo "Skipping shared initial relax cache; trajectories will run initial relax independently."
fi

echo "Submitting dependent trajectory array (${ARRAY_SPEC})..."
array_jid="$(
  sbatch --parsable \
    "${dependency_args[@]}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM_PER_NODE}" \
    --array="${ARRAY_SPEC}" \
    "${SCRIPT_DIR}/collect_trajectories_cpu_array.sh"
)"
echo "array_jid=${array_jid}"

replay_cache_jid=""
if [ "${SUBMIT_REPLAY_CACHE}" != "0" ]; then
  echo "Submitting dependent replay-cache materialization job..."
  replay_cache_jid="$(
    sbatch --parsable \
      --dependency=afterok:${array_jid} \
      --cpus-per-task="${REPLAY_CACHE_CPUS}" \
      --mem="${REPLAY_CACHE_MEM}" \
      --export=ALL,DATASET_DIR="${OUTPUT_BASE_DIR}",REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR}",OVERWRITE_REPLAY_CACHE="${OVERWRITE_REPLAY_CACHE}",REPLAY_CACHE_WORKERS="${REPLAY_CACHE_WORKERS}",REPLAY_CACHE_WORKER_BACKEND="${REPLAY_CACHE_WORKER_BACKEND}" \
      "${SCRIPT_DIR}/materialize_replay_cache.sh"
  )"
  echo "replay_cache_jid=${replay_cache_jid}"
else
  echo "Skipping replay-cache materialization submission."
fi

if [ "${SUBMIT_IQL}" != "0" ]; then
  iql_dependency="${array_jid}"
  if [ -n "${replay_cache_jid}" ]; then
    iql_dependency="${replay_cache_jid}"
  fi
  echo "Submitting dependent IQL training job..."
  iql_jid="$(
    sbatch --parsable \
      --dependency=afterok:${iql_dependency} \
      --export=ALL,DATASET_DIR="${OUTPUT_BASE_DIR}",REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR}" \
      "${SCRIPT_DIR}/train_iql.sh"
  )"
  echo "iql_jid=${iql_jid}"
fi
