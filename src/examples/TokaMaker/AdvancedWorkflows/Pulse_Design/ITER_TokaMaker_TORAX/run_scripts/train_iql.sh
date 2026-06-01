#!/usr/bin/env bash

# Purpose:
#   Train IQL on the collected TORAX dataset and optionally run a closed-loop
#   actor eval after training.
#
# When to use:
#   Use for offline RL training. Keep RUN_ACTOR_EVAL=1 for final runs and
#   disable it for quick parameter sweeps.
#
# Example:
#   DATASET_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_preprocessed \
#     ACTION_MODE=residual_prev_action OBSERVATION_MODE=prev_action \
#     ACTION_RATE_PENALTY=0.01 CHECKPOINT_INTERVAL=1000 \
#     sbatch run_scripts/train_iql.sh
#
# If you want checkpoint-by-checkpoint closed-loop plots without blocking
# training, run the fanout helper after training:
#   DATASET_DIR=./run_<date> \
#     TRAIN_OUTPUT_DIR=./out/iql/<run>/<wandb_run_id> \
#     ./run_scripts/fanout_checkpoint_evals.sh

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

: "${DATASET_DIR:?Set DATASET_DIR to the collected dataset root}"

RUN_LOG_DIR="${RUN_LOG_DIR:-${DATASET_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/train_iql-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/train_iql-${SLURM_JOB_ID:-$$}.err" >&2)

export PYTHONUNBUFFERED=1
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

args=(--dataset_dir "${DATASET_DIR}")

add_arg() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"
  if [ -n "${value}" ]; then
    args+=("${flag}" "${value}")
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
    args+=("${false_flag}")
  else
    args+=("${true_flag}")
  fi
}

add_arg OUTPUT_DIR --output_dir
add_arg BATCH_SIZE --batch_size
add_arg NUM_STEPS --num_steps
add_arg WANDB_PROJECT --project
add_arg RUN_NAME --run_name
add_arg WANDB_GROUP --wandb_group
add_arg RESUME_FROM --resume_from
add_arg WANDB_MODE --wandb_mode
add_arg CHECKPOINT_INTERVAL --checkpoint_interval
add_arg LOG_INTERVAL --log_interval
add_arg TAU --tau
add_arg BETA --beta
add_arg GAMMA --gamma
add_arg LR --lr
add_arg HIDDEN_DIM --hidden_dim
add_bool_arg USE_WANDB_RUN_SUBDIR --use_wandb_run_subdir --no-use_wandb_run_subdir
add_arg EVAL_INTERVAL --eval_interval
add_arg EVAL_BATCH_SIZE --eval_batch_size
add_arg EVAL_HISTOGRAM_INTERVAL --eval_histogram_interval
add_arg EVAL_SEED --eval_seed
add_arg IQL_DEVICE --device
add_arg REPLAY_CACHE_DIR --replay_cache_dir
add_bool_arg USE_REPLAY_CACHE --use_replay_cache --no-use_replay_cache
add_bool_arg RUN_ACTOR_EVAL --run_actor_eval --no-run_actor_eval
add_arg ACTOR_EVAL_OUTPUT_DIR --actor_eval_output_dir
add_arg ACTOR_EVAL_PROJECT --actor_eval_project
add_arg ACTOR_EVAL_RUN_NAME --actor_eval_run_name
add_arg ACTOR_EVAL_WANDB_MODE --actor_eval_wandb_mode
add_arg ACTOR_EVAL_WANDB_GROUP --actor_eval_wandb_group
add_arg ACTOR_EVAL_INITIAL_RELAX_STATE --actor_eval_initial_relax_state
add_arg ACTOR_EVAL_INITIAL_RELAX_CACHE_DIR --actor_eval_initial_relax_cache_dir
add_arg ACTOR_EVAL_MAX_LOOP --actor_eval_max_loop
add_arg ACTOR_EVAL_GRID_SIZE --actor_eval_grid_size
add_arg ACTOR_EVAL_DEVICE --actor_eval_device
add_arg ACTION_MODE --action_mode
add_arg ACTION_RATE_PENALTY --action_rate_penalty
add_arg CHECKPOINT_EVAL_INTERVAL --checkpoint_eval_interval
add_arg CHECKPOINT_EVAL_METRIC --checkpoint_eval_metric
add_arg OBSERVATION_MODE --observation_mode

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
echo "IQL args: ${args[*]}"
nvidia-smi || true

uv run python IQL.py "${args[@]}"
