#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=jag-standard
#SBATCH --mail-user=siddharth@cs.stanford.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Purpose:
#   Run trajectories in a single GPU Slurm job on jag-standard. It uses
#   `uv run --extra cuda13`, and collect_trajectories_delta.py exits early if a
#   GPU is visible but JAX cannot initialize a GPU backend.
#
# Should you call this directly?
#   Use this for GPU diagnostics or a small GPU comparison run. The current
#   recommended full production run is the CPU array submit helper, because the
#   bottleneck has been TORAX/TokaMaker simulation time rather than RAM.
#
# Direct-use example, for a small GPU diagnostic slice:
#   OUTPUT_DIR=./rl_dataset_delta_gpu_diag_$(date +%Y%m%d_%H%M%S) \
#     START_IDX=600 END_IDX=601 N_WORKERS=1 sbatch run_scripts/collect_trajectories_gpu.sh

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

add_bool_arg() {
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

N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
N_TRAJECTORIES="${N_TRAJECTORIES:-1000}"
START_IDX="${START_IDX:-600}"
END_IDX="${END_IDX:-${N_TRAJECTORIES}}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-7200}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_${OFT_SELECTED_FLAVOR}_${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/collect_trajectories_gpu-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/collect_trajectories_gpu-${SLURM_JOB_ID:-$$}.err" >&2)

export PYTHONUNBUFFERED=1

echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "N_WORKERS=${N_WORKERS}"
echo "N_TRAJECTORIES=${N_TRAJECTORIES}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "TRAJECTORY_TIMEOUT_SECONDS=${TRAJECTORY_TIMEOUT_SECONDS}"
echo "SAVE_STATS_FOR_REWARD_RECALC=${SAVE_STATS_FOR_REWARD_RECALC:-<argparse default>}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

COLLECT_ARGS=()
add_bool_arg SAVE_STATS_FOR_REWARD_RECALC --save_stats_for_reward_recalc --no_save_stats_for_reward_recalc

uv run --extra cuda13 python collect_trajectories_delta.py \
  --n_trajectories "${N_TRAJECTORIES}" \
  --start_idx "${START_IDX}" \
  --end_idx "${END_IDX}" \
  --n_workers "${N_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_loop "${MAX_LOOP}" \
  --grid_size "${GRID_SIZE}" \
  --trajectory_timeout_seconds "${TRAJECTORY_TIMEOUT_SECONDS}" \
  "${COLLECT_ARGS[@]}" \
  "$@"
