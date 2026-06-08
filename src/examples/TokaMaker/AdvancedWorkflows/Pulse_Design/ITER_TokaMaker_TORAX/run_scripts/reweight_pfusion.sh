#!/usr/bin/env bash
#SBATCH --account=nlp
#SBATCH --mem=32G
#SBATCH --partition=john,sc-loprio
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_DIR}"
OFT_ROOT="$(cd "${PROJECT_DIR}/../../../../../../" && pwd -P)"
source "${OFT_ROOT}/scripts/oft_arch/select_oft_install.sh"

mkdir -p logs

exec > >(tee -a "logs/reweight_pfusion-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "logs/reweight_pfusion-${SLURM_JOB_ID:-$$}.err" >&2)

: "${DATASET_DIR:?Set DATASET_DIR}"

echo "Reweighting dataset: ${DATASET_DIR}"
echo "RL_REWARD_MODE=pfusion"

RL_REWARD_MODE=pfusion uv run python update_trajectories.py "${DATASET_DIR}"
