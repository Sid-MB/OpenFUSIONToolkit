#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Recommended production entrypoint for CPU trajectory generation. This script
#   changes to the example directory, chooses a timestamped output folder,
#   optionally submits the initial relax cache job, and submits the john Slurm
#   array that runs collect_trajectories_cpu_array.sh.
#
# Should you call this directly?
#   Yes. For the standard full run, paste the example below into your shell.
#
# Standard full-run example (64 total allocated CPUs: 16 tasks x 4 CPUs):
#   START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-399%16 CPUS_PER_TASK=4 \
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
ARRAY_SPEC="${ARRAY_SPEC:-0-399%16}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-4}}"
MEM_PER_NODE="${MEM_PER_NODE:-${SLURM_MEM_PER_NODE:-16G}}"
USE_INITIAL_RELAX_CACHE="${USE_INITIAL_RELAX_CACHE:-0}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-7200}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)}"
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-${OUTPUT_BASE_DIR}/initial_relax_state.json}"

export N_TRAJECTORIES START_IDX END_IDX N_WORKERS CHUNK_SIZE
export MAX_LOOP GRID_SIZE TRAJECTORY_TIMEOUT_SECONDS
export OUTPUT_BASE_DIR INITIAL_RELAX_CACHE USE_INITIAL_RELAX_CACHE

echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"
echo "N_WORKERS=${N_WORKERS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "ARRAY_SPEC=${ARRAY_SPEC}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MEM_PER_NODE=${MEM_PER_NODE}"
echo "USE_INITIAL_RELAX_CACHE=${USE_INITIAL_RELAX_CACHE}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "TRAJECTORY_TIMEOUT_SECONDS=${TRAJECTORY_TIMEOUT_SECONDS}"

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
