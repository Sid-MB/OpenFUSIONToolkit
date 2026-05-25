#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=john
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Optional setup
# sh ./setup-env.sh

PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(pwd -P)}" && pwd -P)"
OFT_PYTHONPATH="$(cd "${PROJECT_DIR}/../../../../../../" && pwd -P)/install_release/python"

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-20}"
N_WORKERS="${N_WORKERS:-${TOTAL_CPUS}}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( TOTAL_CPUS / N_WORKERS ))}"
if [ "${THREADS_PER_WORKER}" -lt 1 ]; then
  THREADS_PER_WORKER=1
fi
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_${RUN_ID}}"

# Force the CPU path even if this script is run from a GPU-capable login node.
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${OFT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep native math/OpenMP libraries from oversubscribing cores across workers.
export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OFT_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"

echo "Running on host: $(hostname)"
echo "TOTAL_CPUS=${TOTAL_CPUS}"
echo "N_WORKERS=${N_WORKERS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "OFT_NUM_THREADS=${OFT_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
echo "PYTHONPATH includes: ${OFT_PYTHONPATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
test -d "${OFT_PYTHONPATH}/OpenFUSIONToolkit"

uv run python collect_trajectories_delta.py \
  --n_trajectories 1000 \
  --start_idx 600 \
  --n_workers "${N_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
