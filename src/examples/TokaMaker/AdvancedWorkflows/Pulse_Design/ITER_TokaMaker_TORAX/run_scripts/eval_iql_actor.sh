#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --partition=jag-standard
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/IQL.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"
OFT_ROOT="$(cd "${PROJECT_DIR}/../../../../../../" && pwd -P)"
source "${OFT_ROOT}/scripts/oft_arch/select_oft_install.sh"

ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:?Set ACTOR_CHECKPOINT to iql_weights.pt or checkpoint_step_*.pt}"
DATASET_DIR="${DATASET_DIR:-}"
RUN_ID="${RUN_ID:-$(basename "${ACTOR_CHECKPOINT%.pt}")_eval_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-out/iql_eval/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/eval_iql_actor-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/eval_iql_actor-${SLURM_JOB_ID:-$$}.err" >&2)
WANDB_PROJECT="${WANDB_PROJECT:-iql-training}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-${SLURM_JOB_ID:-${RUN_ID}}}"
INITIAL_RELAX_STATE="${INITIAL_RELAX_STATE:-}"
INITIAL_RELAX_CACHE_DIR="${INITIAL_RELAX_CACHE_DIR:-}"
MAX_LOOP="${MAX_LOOP:-2}"
GRID_SIZE="${GRID_SIZE:-51}"
IQL_EVAL_DEVICE="${IQL_EVAL_DEVICE:-}"
REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR:-}"
USE_REPLAY_CACHE="${USE_REPLAY_CACHE:-1}"
ALLOW_CPU_JAX_ON_GPU="${ALLOW_CPU_JAX_ON_GPU:-0}"
RL_SEGMENT_TIMEOUT_SECONDS="${RL_SEGMENT_TIMEOUT_SECONDS:-1800}"
RL_MAX_ACTION_POWER_W="${RL_MAX_ACTION_POWER_W:-150000000}"

# Cache selection is delegated to rl.eval: with no explicit INITIAL_RELAX_STATE it
# reuses a dataset's legacy initial_relax_state.json if present, otherwise a keyed
# file in INITIAL_RELAX_CACHE_DIR (built on demand).

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Running on host: $(hostname)"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "ACTOR_CHECKPOINT=${ACTOR_CHECKPOINT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "RUN_NAME=${RUN_NAME}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "INITIAL_RELAX_STATE=${INITIAL_RELAX_STATE}"
echo "INITIAL_RELAX_CACHE_DIR=${INITIAL_RELAX_CACHE_DIR:-<default>}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "IQL_EVAL_DEVICE=${IQL_EVAL_DEVICE}"
echo "REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
echo "USE_REPLAY_CACHE=${USE_REPLAY_CACHE}"
echo "ALLOW_CPU_JAX_ON_GPU=${ALLOW_CPU_JAX_ON_GPU}"
echo "RL_SEGMENT_TIMEOUT_SECONDS=${RL_SEGMENT_TIMEOUT_SECONDS}"
echo "RL_MAX_ACTION_POWER_W=${RL_MAX_ACTION_POWER_W}"
nvidia-smi || true

ARGS=(
  --actor_checkpoint "${ACTOR_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --project "${WANDB_PROJECT}"
  --run_name "${RUN_NAME}"
  --wandb_group "${WANDB_GROUP}"
  --max_loop "${MAX_LOOP}"
  --grid_size "${GRID_SIZE}"
  --rl_segment_timeout_seconds "${RL_SEGMENT_TIMEOUT_SECONDS}"
  --rl_max_action_power_w "${RL_MAX_ACTION_POWER_W}"
)
if [ -n "${DATASET_DIR}" ]; then
  ARGS+=(--dataset_dir "${DATASET_DIR}")
fi
if [ -n "${INITIAL_RELAX_STATE}" ]; then
  ARGS+=(--initial_relax_state "${INITIAL_RELAX_STATE}")
fi
if [ -n "${INITIAL_RELAX_CACHE_DIR}" ]; then
  ARGS+=(--initial_relax_cache_dir "${INITIAL_RELAX_CACHE_DIR}")
fi
if [ -n "${IQL_EVAL_DEVICE}" ]; then
  ARGS+=(--device "${IQL_EVAL_DEVICE}")
fi
if [ -n "${REPLAY_CACHE_DIR}" ]; then
  ARGS+=(--replay_cache_dir "${REPLAY_CACHE_DIR}")
fi
if [ "${USE_REPLAY_CACHE}" = "0" ]; then
  ARGS+=(--no_replay_cache)
fi
if [ "${ALLOW_CPU_JAX_ON_GPU}" != "0" ]; then
  ARGS+=(--allow_cpu_jax_on_gpu)
fi

uv run python -m rl.eval "${ARGS[@]}"
