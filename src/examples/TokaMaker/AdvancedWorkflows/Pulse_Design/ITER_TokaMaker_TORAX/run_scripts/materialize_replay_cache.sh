#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/materialize_replay_cache.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the collected dataset root}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK must be set by Slurm or sbatch --cpus-per-task}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
for name in REPLAY_CACHE_DIR OVERWRITE_REPLAY_CACHE REPLAY_CACHE_PROGRESS REPLAY_CACHE_WORKERS REPLAY_CACHE_WORKER_BACKEND; do
  if [ -n "${!name:-}" ]; then
    echo "${name}=${!name}"
  else
    echo "${name}=<argparse default>"
  fi
done

args=("${DATASET_DIR}")
if [ -n "${REPLAY_CACHE_DIR:-}" ]; then
  args+=(--cache_dir "${REPLAY_CACHE_DIR}")
fi
if [ -n "${OVERWRITE_REPLAY_CACHE:-}" ] && [ "${OVERWRITE_REPLAY_CACHE}" != "0" ]; then
  args+=(--overwrite)
fi
if [ -n "${REPLAY_CACHE_PROGRESS:-}" ] && [ "${REPLAY_CACHE_PROGRESS}" = "0" ]; then
  args+=(--no_progress)
fi
if [ -n "${REPLAY_CACHE_WORKERS:-}" ]; then
  args+=(--max_workers "${REPLAY_CACHE_WORKERS}")
fi
if [ -n "${REPLAY_CACHE_WORKER_BACKEND:-}" ]; then
  args+=(--worker_backend "${REPLAY_CACHE_WORKER_BACKEND}")
fi

echo "materialize_replay_cache.py args: ${args[*]}"

uv run python materialize_replay_cache.py "${args[@]}"
