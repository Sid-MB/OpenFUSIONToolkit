#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Recommended production entrypoint for CPU trajectory generation. This script
#   changes to the example directory, chooses a timestamped output folder,
#   initializes the shared dataset manifest/action table, optionally submits the
#   initial relax cache job, and submits the john Slurm array that runs
#   collect_trajectories_cpu_array.sh. It can also submit dependent replay-cache
#   materialization and IQL jobs when explicitly requested.
#
# Should you call this directly?
#   Yes. For the standard full run, paste the example below into your shell.
#
# Explicit full-run example (64 total allocated CPUs: 16 tasks x 4 CPUs):
#   N_TRAJECTORIES=1000 START_IDX=600 END_IDX=1000 \
#     N_WORKERS=1 CHUNK_SIZE=1 ARRAY_CONCURRENCY=16 \
#     SLURM_MAX_ARRAY_SIZE=1001 CPUS_PER_TASK=4 MEM_PER_NODE=128G \
#     USE_INITIAL_RELAX_CACHE=0 OUTPUT_BASE_DIR=./my_dataset \
#     DRY_RUN=0 SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=0 \
#     GRID_SEARCH_CPUS=1 GRID_SEARCH_MEM=128G \
#     REPLAY_CACHE_CPUS=8 REPLAY_CACHE_MEM=128G \
#     ./run_scripts/submit_collect_trajectories_cpu_array.sh
#
# Shell wrappers intentionally do not assign run defaults. Collection defaults
# live in collect_trajectories_delta.py argparse; set an env var here only when
# the shell needs it for Slurm shape/dependencies or when overriding argparse.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_DIR}"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "ERROR: ${name} must be set by the caller." >&2
    exit 2
  fi
}

add_arg() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"
  if [ -n "${value}" ]; then
    INIT_ARGS+=("${flag}" "${value}")
  fi
}

add_bool_arg() {
  local env_name="$1"
  local true_flag="$2"
  local false_flag="$3"
  local value="${!env_name:-}"
  if [ -z "${value}" ]; then
    return
  fi
  if [ "${value}" = "0" ] || [ "${value}" = "false" ] || [ "${value}" = "False" ]; then
    INIT_ARGS+=("${false_flag}")
  else
    INIT_ARGS+=("${true_flag}")
  fi
}

append_export() {
  local env_name="$1"
  local value="${!env_name:-}"
  if [ -n "${value}" ]; then
    EXPORT_NAMES+=("${env_name}")
  fi
}

echo_env_or_argparse_default() {
  local name="$1"
  local fallback_label="$2"
  if [ -n "${!name:-}" ]; then
    echo "${name}=${!name}"
  else
    echo "${name}=<${fallback_label}>"
  fi
}

require_env N_TRAJECTORIES
require_env START_IDX
require_env END_IDX
require_env N_WORKERS
require_env CHUNK_SIZE
require_env ARRAY_CONCURRENCY
require_env SLURM_MAX_ARRAY_SIZE
# CPU and memory requests have defaults that work well; override per job class
# if needed. Memory defaults to the 128G floor.
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM_PER_NODE="${MEM_PER_NODE:-128G}"
require_env USE_INITIAL_RELAX_CACHE
require_env OUTPUT_BASE_DIR

if [ -z "${RUN_LOG_DIR:-}" ]; then
  output_slug="$(basename "${OUTPUT_BASE_DIR%/}")"
  RUN_LOG_DIR="logs/${output_slug}"
fi
require_env RUN_LOG_DIR
require_env DRY_RUN
require_env SUBMIT_GRID_SEARCH
require_env SUBMIT_REPLAY_CACHE
require_env SUBMIT_IQL

# When the shared cache is enabled, either an explicit INITIAL_RELAX_CACHE path
# or a keyed INITIAL_RELAX_CACHE_DIR may be provided. Both are optional: child
# jobs resolve the keyed path themselves (and agree, since the key is derived
# from grid_size + initial profiles + equilibrium). We just forward whatever is
# set so every job uses the same cache file.
if [ "${SUBMIT_GRID_SEARCH}" != "0" ]; then
  GRID_SEARCH_CPUS="${GRID_SEARCH_CPUS:-1}"
  GRID_SEARCH_MEM="${GRID_SEARCH_MEM:-128G}"
fi
if [ "${SUBMIT_REPLAY_CACHE}" != "0" ]; then
  REPLAY_CACHE_CPUS="${REPLAY_CACHE_CPUS:-8}"
  REPLAY_CACHE_MEM="${REPLAY_CACHE_MEM:-128G}"
fi

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

EXPORT_NAMES=(
  N_TRAJECTORIES
  START_IDX
  END_IDX
  N_WORKERS
  CHUNK_SIZE
  OUTPUT_BASE_DIR
  USE_INITIAL_RELAX_CACHE
)
append_export SEED
append_export MAX_LOOP
append_export GRID_SIZE
append_export TRAJECTORY_TIMEOUT_SECONDS
append_export SAVE_REPLAY_SHARD
append_export SAVE_FULL_ZARR
append_export SAVE_JSON
append_export INITIAL_RELAX_CACHE
append_export INITIAL_RELAX_CACHE_DIR
export "${EXPORT_NAMES[@]}"

echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
echo_env_or_argparse_default INITIAL_RELAX_CACHE "argparse/default unused when cache disabled"
echo "N_WORKERS=${N_WORKERS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo_env_or_argparse_default SEED "argparse default"
echo "REQUESTED_TRAJECTORIES=${REQUESTED_TRAJECTORIES}"
echo "ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT}"
echo "ARRAY_SPEC=${ARRAY_SPEC}"
echo "ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY}"
echo "SLURM_MAX_ARRAY_SIZE=${SLURM_MAX_ARRAY_SIZE}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MEM_PER_NODE=${MEM_PER_NODE}"
echo "USE_INITIAL_RELAX_CACHE=${USE_INITIAL_RELAX_CACHE}"
echo_env_or_argparse_default MAX_LOOP "argparse default"
echo_env_or_argparse_default GRID_SIZE "argparse default"
echo_env_or_argparse_default TRAJECTORY_TIMEOUT_SECONDS "argparse default"
echo_env_or_argparse_default SAVE_REPLAY_SHARD "argparse default"
echo_env_or_argparse_default SAVE_FULL_ZARR "argparse default"
echo_env_or_argparse_default SAVE_JSON "argparse default"
echo "SUBMIT_GRID_SEARCH=${SUBMIT_GRID_SEARCH}"
echo_env_or_argparse_default GRID_SEARCH_MEM "unset"
echo_env_or_argparse_default GRID_SEARCH_CPUS "unset"
echo_env_or_argparse_default GRID_SEARCH_OUTPUT_DIR "grid search argparse default"
echo_env_or_argparse_default GRID_SEARCH_GAMMA "grid search argparse default"
echo_env_or_argparse_default GRID_SEARCH_TOP_K "grid search argparse default"
echo "SUBMIT_REPLAY_CACHE=${SUBMIT_REPLAY_CACHE}"
echo_env_or_argparse_default REPLAY_CACHE_MEM "unset"
echo_env_or_argparse_default REPLAY_CACHE_CPUS "unset"
echo_env_or_argparse_default REPLAY_CACHE_WORKERS "materialize argparse default"
echo_env_or_argparse_default REPLAY_CACHE_WORKER_BACKEND "materialize argparse default"
echo_env_or_argparse_default OVERWRITE_REPLAY_CACHE "materialize argparse default"
echo_env_or_argparse_default REPLAY_CACHE_DIR "materialize argparse default"
echo "SUBMIT_IQL=${SUBMIT_IQL}"
echo "DRY_RUN=${DRY_RUN}"

if [ "${DRY_RUN}" != "0" ]; then
  echo "Dry run requested; not initializing dataset or submitting Slurm jobs."
  exit 0
fi

mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${RUN_LOG_DIR}"

echo "Initializing shared dataset manifest/action table..."
INIT_ARGS=(
  --n_trajectories "${N_TRAJECTORIES}"
  --start_idx "${START_IDX}"
  --end_idx "${END_IDX}"
  --output_dir "${OUTPUT_BASE_DIR}"
  --init_dataset_only
)
add_arg SEED --seed
add_arg MAX_LOOP --max_loop
add_arg GRID_SIZE --grid_size
add_arg TRAJECTORY_TIMEOUT_SECONDS --trajectory_timeout_seconds
add_bool_arg SAVE_REPLAY_SHARD --save_replay_shard --no_save_replay_shard
add_bool_arg SAVE_FULL_ZARR --save_full_zarr --no_save_full_zarr
add_bool_arg SAVE_JSON --save_json --no_save_json
if [ "${USE_INITIAL_RELAX_CACHE}" = "0" ]; then
  INIT_ARGS+=(--no_initial_relax_cache)
elif [ -n "${INITIAL_RELAX_CACHE:-}" ]; then
  INIT_ARGS+=(--initial_relax_cache "${INITIAL_RELAX_CACHE}")
elif [ -n "${INITIAL_RELAX_CACHE_DIR:-}" ]; then
  INIT_ARGS+=(--initial_relax_cache_dir "${INITIAL_RELAX_CACHE_DIR}")
fi
echo "collect_trajectories_delta.py init args: ${INIT_ARGS[*]}"
uv run python collect_trajectories_delta.py \
  "${INIT_ARGS[@]}"

dependency_args=()
if [ "${USE_INITIAL_RELAX_CACHE}" != "0" ]; then
  echo "Submitting initial relax cache job..."
  cache_jid="$(
    sbatch --parsable \
      --output="${RUN_LOG_DIR}/collect_initial_relax_cache-%j.out" \
      --error="${RUN_LOG_DIR}/collect_initial_relax_cache-%j.err" \
      "${SCRIPT_DIR}/collect_initial_relax_cache_cpu.sh"
  )"
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
    --output="${RUN_LOG_DIR}/collect_trajectories-%A_%a.out" \
    --error="${RUN_LOG_DIR}/collect_trajectories-%A_%a.err" \
    "${SCRIPT_DIR}/collect_trajectories_cpu_array.sh"
)"
echo "array_jid=${array_jid}"

grid_search_jid=""
if [ "${SUBMIT_GRID_SEARCH}" != "0" ]; then
  echo "Submitting dependent grid-search baseline job..."
  grid_search_export="ALL,DATASET_DIR=${OUTPUT_BASE_DIR}"
  if [ -n "${GRID_SEARCH_OUTPUT_DIR:-}" ]; then
    grid_search_export+=",GRID_SEARCH_OUTPUT_DIR=${GRID_SEARCH_OUTPUT_DIR}"
  fi
  if [ -n "${GRID_SEARCH_GAMMA:-}" ]; then
    grid_search_export+=",GRID_SEARCH_GAMMA=${GRID_SEARCH_GAMMA}"
  fi
  if [ -n "${GRID_SEARCH_TOP_K:-}" ]; then
    grid_search_export+=",GRID_SEARCH_TOP_K=${GRID_SEARCH_TOP_K}"
  fi
  grid_search_jid="$(
    sbatch --parsable \
      --dependency=afterok:${array_jid} \
      --cpus-per-task="${GRID_SEARCH_CPUS}" \
      --mem="${GRID_SEARCH_MEM}" \
      --export="${grid_search_export}" \
      --output="${RUN_LOG_DIR}/grid_search_baseline-%j.out" \
      --error="${RUN_LOG_DIR}/grid_search_baseline-%j.err" \
      "${SCRIPT_DIR}/grid_search_baseline.sh"
  )"
  echo "grid_search_jid=${grid_search_jid}"
else
  echo "Skipping grid-search baseline submission."
fi

replay_cache_jid=""
if [ "${SUBMIT_REPLAY_CACHE}" != "0" ]; then
  echo "Submitting dependent replay-cache materialization job..."
  replay_export="ALL,DATASET_DIR=${OUTPUT_BASE_DIR}"
  if [ -n "${REPLAY_CACHE_DIR:-}" ]; then
    replay_export+=",REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
  fi
  if [ -n "${OVERWRITE_REPLAY_CACHE:-}" ]; then
    replay_export+=",OVERWRITE_REPLAY_CACHE=${OVERWRITE_REPLAY_CACHE}"
  fi
  if [ -n "${REPLAY_CACHE_WORKERS:-}" ]; then
    replay_export+=",REPLAY_CACHE_WORKERS=${REPLAY_CACHE_WORKERS}"
  fi
  if [ -n "${REPLAY_CACHE_WORKER_BACKEND:-}" ]; then
    replay_export+=",REPLAY_CACHE_WORKER_BACKEND=${REPLAY_CACHE_WORKER_BACKEND}"
  fi
  replay_cache_jid="$(
    sbatch --parsable \
      --dependency=afterok:${array_jid} \
      --cpus-per-task="${REPLAY_CACHE_CPUS}" \
      --mem="${REPLAY_CACHE_MEM}" \
      --export="${replay_export}" \
      --output="${RUN_LOG_DIR}/materialize_replay_cache-%j.out" \
      --error="${RUN_LOG_DIR}/materialize_replay_cache-%j.err" \
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
  iql_export="ALL,DATASET_DIR=${OUTPUT_BASE_DIR}"
  if [ -n "${REPLAY_CACHE_DIR:-}" ]; then
    iql_export+=",REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
  fi
  iql_jid="$(
    sbatch --parsable \
      --dependency=afterok:${iql_dependency} \
      --export="${iql_export}" \
      --output="${RUN_LOG_DIR}/train_iql-%j.out" \
      --error="${RUN_LOG_DIR}/train_iql-%j.err" \
      "${SCRIPT_DIR}/train_iql.sh"
  )"
  echo "iql_jid=${iql_jid}"
fi
