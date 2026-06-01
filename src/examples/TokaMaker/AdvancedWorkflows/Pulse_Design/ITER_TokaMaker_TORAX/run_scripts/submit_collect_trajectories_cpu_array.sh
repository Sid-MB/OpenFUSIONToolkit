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
# Explicit full-run example. See README docs for the current default values
# and the collection parameter table for the recommended run knobs:
#   N_TRAJECTORIES=1000 START_IDX=600 END_IDX=1000 \
#     N_WORKERS=1 CHUNK_SIZE=1 \
#     SLURM_MAX_ARRAY_SIZE=1001 \
#     USE_INITIAL_RELAX_CACHE=0 OUTPUT_BASE_DIR=./my_dataset \
#     SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=0 \
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

REUSE_EXISTING_DATASET="${REUSE_EXISTING_DATASET:-1}"
if [ "${REUSE_EXISTING_DATASET}" != "0" ]; then
  N_WORKERS="${N_WORKERS:-1}"
  CHUNK_SIZE="${CHUNK_SIZE:-1}"
  ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
  SLURM_MAX_ARRAY_SIZE="${SLURM_MAX_ARRAY_SIZE:-1}"
fi
SLURM_NICE="${SLURM_NICE:-10}"

validate_existing_dataset() {
  echo "Validating that the requested dataset already exists..."
  uv run python - "${OUTPUT_BASE_DIR}" "${N_TRAJECTORIES}" "${START_IDX}" "${END_IDX}" "${OBSERVATION_MODE}" <<'PY'
from pathlib import Path
import sys

import numpy as np

from dataloader import load_json

root = Path(sys.argv[1]).resolve()
n_trajectories = int(sys.argv[2])
start_idx = int(sys.argv[3])
end_idx = int(sys.argv[4])
expected_observation_mode = str(sys.argv[5])

manifest_path = root / 'run_manifest.json'
actions_path = root / 'all_actions.npy'
if not manifest_path.is_file():
    raise SystemExit(
        f"Missing dataset manifest: {manifest_path}\n"
        "Recollecting trajectories is expensive: it reruns full TORAX/TokaMaker "
        "closed-loop simulations for every requested trajectory. If you already "
        "have a complete dataset elsewhere, point OUTPUT_BASE_DIR at it and set "
        "REUSE_EXISTING_DATASET=1."
    )
if not actions_path.is_file():
    raise SystemExit(
        f"Missing action table: {actions_path}\n"
        "Recollecting trajectories is expensive: it reruns full TORAX/TokaMaker "
        "closed-loop simulations for every requested trajectory. If you already "
        "have a complete dataset elsewhere, point OUTPUT_BASE_DIR at it and set "
        "REUSE_EXISTING_DATASET=1."
    )

manifest = load_json(manifest_path)
requested_range = manifest.get('requested_range', {})
existing_observation_mode = str(manifest.get('observation_mode', 'legacy'))
if int(manifest.get('n_trajectories', -1)) != n_trajectories:
    raise SystemExit(
        f"Dataset n_trajectories mismatch: expected {n_trajectories}, "
        f"found {manifest.get('n_trajectories')}\n"
        "Recollecting trajectories to change the range is expensive because it "
        "repeats the full closed-loop simulation for each trajectory. If the "
        "existing dataset is the one you want, keep REUSE_EXISTING_DATASET=1 and "
        "adjust OUTPUT_BASE_DIR to the complete dataset."
    )
if int(requested_range.get('start_idx', -1)) != start_idx or int(requested_range.get('end_idx', -1)) != end_idx:
    raise SystemExit(
        'Dataset requested_range mismatch: '
        f"expected [{start_idx}, {end_idx}), found "
        f"[{requested_range.get('start_idx')}, {requested_range.get('end_idx')})\n"
        "Recollecting is expensive because it reruns the full trajectory set. "
        "If you intended to reuse an existing dataset, point OUTPUT_BASE_DIR at "
        "the complete dataset and keep REUSE_EXISTING_DATASET=1."
    )
if existing_observation_mode != expected_observation_mode:
    raise SystemExit(
        'Dataset observation_mode mismatch: '
        f'expected {expected_observation_mode!r}, found {existing_observation_mode!r}. '
        'Legacy and prev_action datasets are not interchangeable. Recollecting to '
        'change observation mode is expensive because it reruns the full closed-loop '
        'simulation set; if you already have the matching schema, point OUTPUT_BASE_DIR '
        'at it and keep REUSE_EXISTING_DATASET=1.'
    )

actions = np.load(actions_path)
if actions.shape[0] < end_idx:
    raise SystemExit(
        f'Action table is too short for requested range: shape={actions.shape}, end_idx={end_idx}'
    )

missing = []
for run_id in range(start_idx, end_idx):
    shard = root / 'replay_shards' / f'trajectory_{run_id:04d}.npz'
    if not shard.is_file() or shard.stat().st_size == 0:
        missing.append(str(shard))

if missing:
    raise SystemExit(
        'Dataset reuse requested, but replay shards are missing:\n'
        + '\n'.join(f'  {path}' for path in missing)
        + '\nRecollecting trajectories to regenerate missing shards is expensive: '
        'it repeats the full TORAX/TokaMaker closed-loop simulation for every '
        'trajectory in the requested range. If a complete dataset already exists, '
        'use that OUTPUT_BASE_DIR with REUSE_EXISTING_DATASET=1 instead.'
    )

print(f'Existing dataset validated: {root}')
PY
}

require_env N_TRAJECTORIES
require_env START_IDX
require_env END_IDX
N_WORKERS="${N_WORKERS:-1}"
if [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
  require_env CHUNK_SIZE
  ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-64}"
  require_env SLURM_MAX_ARRAY_SIZE
fi
# CPU and memory requests have defaults that work well; override per job class
# if needed. Memory defaults to the 128G floor.
CPUS_PER_TASK="${CPUS_PER_TASK:-$(( N_WORKERS * 4 ))}"
MEM_PER_NODE="${MEM_PER_NODE:-128G}"
require_env USE_INITIAL_RELAX_CACHE
require_env OUTPUT_BASE_DIR

if [ -z "${RUN_LOG_DIR:-}" ]; then
  RUN_LOG_DIR="${OUTPUT_BASE_DIR%/}/logs"
fi
require_env RUN_LOG_DIR
require_env DRY_RUN
require_env SUBMIT_GRID_SEARCH
require_env SUBMIT_REPLAY_CACHE
require_env SUBMIT_IQL

# Optional Slurm priority tweak for the trajectory array itself. Leave unset to
# use the cluster's normal scheduling priority.
if [[ -n "${SLURM_NICE:-}" && ! "${SLURM_NICE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SLURM_NICE must be a non-negative integer, got ${SLURM_NICE}" >&2
  exit 2
fi
SBATCH_NICE_ARGS=()
if [ -n "${SLURM_NICE}" ]; then
  SBATCH_NICE_ARGS+=(--nice="${SLURM_NICE}")
fi

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

if [ "${END_IDX}" -lt "${START_IDX}" ]; then
  echo "ERROR: END_IDX must be >= START_IDX, got START_IDX=${START_IDX}, END_IDX=${END_IDX}" >&2
  exit 2
fi
if [ "${END_IDX}" -gt "${N_TRAJECTORIES}" ]; then
  echo "ERROR: END_IDX must be <= N_TRAJECTORIES, got END_IDX=${END_IDX}, N_TRAJECTORIES=${N_TRAJECTORIES}" >&2
  exit 2
fi

REQUESTED_TRAJECTORIES=$(( END_IDX - START_IDX ))
if [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
  if [ "${CHUNK_SIZE}" -lt 1 ]; then
    echo "ERROR: CHUNK_SIZE must be >= 1, got ${CHUNK_SIZE}" >&2
    exit 2
  fi
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
else
  ARRAY_TASK_COUNT=0
  ARRAY_SPEC="<reuse-existing-dataset>"
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
append_export OBSERVATION_MODE
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
echo "REUSE_EXISTING_DATASET=${REUSE_EXISTING_DATASET}"
if [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
  echo "ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY}"
  echo "SLURM_MAX_ARRAY_SIZE=${SLURM_MAX_ARRAY_SIZE}"
  echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
  echo "MEM_PER_NODE=${MEM_PER_NODE}"
else
  echo "ARRAY_CONCURRENCY=<unused>"
  echo "SLURM_MAX_ARRAY_SIZE=<unused>"
  echo "CPUS_PER_TASK=<unused>"
  echo "MEM_PER_NODE=<unused>"
fi
echo "USE_INITIAL_RELAX_CACHE=${USE_INITIAL_RELAX_CACHE}"
echo_env_or_argparse_default MAX_LOOP "argparse default"
echo_env_or_argparse_default GRID_SIZE "argparse default"
echo_env_or_argparse_default TRAJECTORY_TIMEOUT_SECONDS "argparse default"
echo_env_or_argparse_default SAVE_REPLAY_SHARD "argparse default"
echo_env_or_argparse_default SAVE_FULL_ZARR "argparse default"
echo_env_or_argparse_default SAVE_JSON "argparse default"
echo_env_or_argparse_default OBSERVATION_MODE "argparse default"
echo "SUBMIT_GRID_SEARCH=${SUBMIT_GRID_SEARCH}"
echo_env_or_argparse_default GRID_SEARCH_MEM "unset"
echo_env_or_argparse_default GRID_SEARCH_CPUS "unset"
echo_env_or_argparse_default GRID_SEARCH_OUTPUT_DIR "grid search argparse default"
echo_env_or_argparse_default GRID_SEARCH_GAMMA "grid search argparse default"
echo_env_or_argparse_default GRID_SEARCH_TOP_K "grid search argparse default"
echo_env_or_argparse_default SLURM_NICE "unset"
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
if [ ! -e "${OUTPUT_BASE_DIR}/.gitignore" ]; then
  printf '*\n' > "${OUTPUT_BASE_DIR}/.gitignore"
fi

if [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
  if [ -s "${OUTPUT_BASE_DIR}/run_manifest.json" ] || [ -d "${OUTPUT_BASE_DIR}/replay_shards" ]; then
    echo "NOTE: ${OUTPUT_BASE_DIR} already looks like a dataset root."
    echo "      This run will collect trajectories again."
    echo "      If you meant to reuse the existing dataset, stop here and rerun with the default REUSE_EXISTING_DATASET=1."
  else
    echo "Starting a fresh trajectory collection."
    echo "If you already have a complete dataset in ${OUTPUT_BASE_DIR}, stop and rerun with REUSE_EXISTING_DATASET=1."
  fi
  echo "Fresh collection is explicit opt-out because the dataset schema is part of the contract."
  echo "Use OBSERVATION_MODE=prev_action for actor-conditioned datasets, OBSERVATION_MODE=plasma_only for plasma-only ablations, and legacy only for compatibility checks."
else
  echo "Reusing an existing dataset is explicit."
  echo "The launcher will validate run_manifest.json, all_actions.npy, and replay_shards/ before skipping collection."
fi

collection_jid=""
if [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
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
  add_arg OBSERVATION_MODE --observation_mode
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
  collection_jid="$(
      sbatch --parsable \
        --output="${RUN_LOG_DIR}/collect_initial_relax_cache-%j.out" \
        --error="${RUN_LOG_DIR}/collect_initial_relax_cache-%j.err" \
        "${SCRIPT_DIR}/collect_initial_relax_cache_cpu.sh"
    )"
    echo "collection_jid=${collection_jid}"
    dependency_args=(--dependency=afterok:${collection_jid})
  else
    echo "Skipping shared initial relax cache; trajectories will run initial relax independently."
  fi

  echo "Submitting dependent trajectory array (${ARRAY_SPEC})..."
  collection_jid="$(
    sbatch --parsable \
      "${dependency_args[@]}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM_PER_NODE}" \
      --array="${ARRAY_SPEC}" \
      "${SBATCH_NICE_ARGS[@]}" \
      --output="${RUN_LOG_DIR}/collect_trajectories-%A_%a.out" \
      --error="${RUN_LOG_DIR}/collect_trajectories-%A_%a.err" \
      "${SCRIPT_DIR}/collect_trajectories_cpu_array.sh"
  )"
  echo "collection_jid=${collection_jid}"
else
  echo "Reusing existing dataset; skipping trajectory collection and initial-relax cache submission."
  validate_existing_dataset
fi

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
  grid_search_args=()
  if [ -n "${collection_jid}" ] && [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
    grid_search_args+=(--dependency=afterok:${collection_jid})
  fi
  grid_search_jid="$(
    sbatch --parsable \
      "${grid_search_args[@]}" \
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
  replay_cache_args=()
  if [ -n "${collection_jid}" ] && [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
    replay_cache_args+=(--dependency=afterok:${collection_jid})
  fi
  replay_cache_jid="$(
    sbatch --parsable \
      "${replay_cache_args[@]}" \
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
  echo "Submitting dependent IQL training job..."
  iql_export="ALL,DATASET_DIR=${OUTPUT_BASE_DIR}"
  if [ -n "${REPLAY_CACHE_DIR:-}" ]; then
    iql_export+=",REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
  fi
  iql_args=()
  if [ -n "${replay_cache_jid}" ]; then
    iql_args+=(--dependency=afterok:${replay_cache_jid})
  elif [ -n "${collection_jid}" ] && [ "${REUSE_EXISTING_DATASET}" = "0" ]; then
    iql_args+=(--dependency=afterok:${collection_jid})
  fi
  iql_jid="$(
    sbatch --parsable \
      "${iql_args[@]}" \
      --export="${iql_export}" \
      --output="${RUN_LOG_DIR}/train_iql-%j.out" \
      --error="${RUN_LOG_DIR}/train_iql-%j.err" \
      "${SCRIPT_DIR}/train_iql.sh"
  )"
  echo "iql_jid=${iql_jid}"
fi
