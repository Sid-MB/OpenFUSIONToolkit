#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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
REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR:-}"
OVERWRITE_REPLAY_CACHE="${OVERWRITE_REPLAY_CACHE:-0}"
REPLAY_CACHE_PROGRESS="${REPLAY_CACHE_PROGRESS:-1}"
REPLAY_CACHE_WORKERS="${REPLAY_CACHE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
REPLAY_CACHE_WORKER_BACKEND="${REPLAY_CACHE_WORKER_BACKEND:-process}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
echo "OVERWRITE_REPLAY_CACHE=${OVERWRITE_REPLAY_CACHE}"
echo "REPLAY_CACHE_PROGRESS=${REPLAY_CACHE_PROGRESS}"
echo "REPLAY_CACHE_WORKERS=${REPLAY_CACHE_WORKERS}"
echo "REPLAY_CACHE_WORKER_BACKEND=${REPLAY_CACHE_WORKER_BACKEND}"

args=("${DATASET_DIR}")
if [ -n "${REPLAY_CACHE_DIR}" ]; then
  args+=(--cache_dir "${REPLAY_CACHE_DIR}")
fi
if [ "${OVERWRITE_REPLAY_CACHE}" != "0" ]; then
  args+=(--overwrite)
fi
if [ "${REPLAY_CACHE_PROGRESS}" = "0" ]; then
  args+=(--no_progress)
fi
args+=(--max_workers "${REPLAY_CACHE_WORKERS}")
args+=(--worker_backend "${REPLAY_CACHE_WORKER_BACKEND}")

uv run python materialize_replay_cache.py "${args[@]}"
