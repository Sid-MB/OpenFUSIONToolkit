#!/usr/bin/env bash

set -euo pipefail

# Example:
#   START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-19%4 ./run_scripts/submit_collect_trajectories_cpu_array.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_DIR}"

N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
ARRAY_SPEC="${ARRAY_SPEC:-0-49%4}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)}"
INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE:-${OUTPUT_BASE_DIR}/initial_relax_state.json}"

export N_TRAJECTORIES START_IDX END_IDX CHUNK_SIZE OUTPUT_BASE_DIR INITIAL_RELAX_CACHE

echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "INITIAL_RELAX_CACHE=${INITIAL_RELAX_CACHE}"
echo "Submitting initial relax cache job..."
cache_jid="$(sbatch --parsable "${SCRIPT_DIR}/collect_initial_relax_cache_cpu.sh")"
echo "cache_jid=${cache_jid}"

echo "Submitting dependent trajectory array (${ARRAY_SPEC})..."
array_jid="$(sbatch --parsable --dependency=afterok:${cache_jid} --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/collect_trajectories_cpu_array.sh")"
echo "array_jid=${array_jid}"
