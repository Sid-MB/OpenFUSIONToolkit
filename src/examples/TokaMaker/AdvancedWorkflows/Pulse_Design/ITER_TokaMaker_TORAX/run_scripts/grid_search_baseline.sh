#!/usr/bin/env bash

# Purpose:
#   Rank already-generated TORAX trajectories and write a baseline summary.
#   This is an analysis step, not a simulation step.
#
# When to use:
#   Use after trajectory collection when you want a quick ranking of the
#   dataset's best-observed returns.
#
# Example:
#   DATASET_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_preprocessed \
#     sbatch run_scripts/grid_search_baseline.sh

#SBATCH --account=nlp
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/grid_search_baseline.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"

: "${DATASET_DIR:?Set DATASET_DIR to the collected dataset root}"

export PYTHONUNBUFFERED=1

args=(--dataset-dir "${DATASET_DIR}")

add_arg() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"
  if [ -n "${value}" ]; then
    args+=("${flag}" "${value}")
  fi
}

add_arg GRID_SEARCH_OUTPUT_DIR --output-dir
add_arg GRID_SEARCH_GAMMA --gamma
add_arg GRID_SEARCH_TOP_K --top-k

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "grid_search_baseline.py args: ${args[*]}"

uv run python grid_search_baseline.py "${args[@]}"
