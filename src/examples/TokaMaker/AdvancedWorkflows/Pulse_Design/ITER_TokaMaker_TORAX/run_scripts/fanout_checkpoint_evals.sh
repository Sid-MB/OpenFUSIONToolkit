#!/usr/bin/env bash
# fanout_checkpoint_evals.sh — Submit one full CPU eval job per saved training checkpoint.
#
# This is a post-training analysis helper, not part of the training loop.
# It scans a training output directory for checkpoint_step_*.pt files and
# submits a separate eval_iql_actor_cpu.sh job for each checkpoint so the
# closed-loop TORAX plots/movie are produced concurrently on Slurm.
#
# Use this when you want checkpoint-by-checkpoint visibility without blocking
# the training job itself.
#
# Default behavior:
#   Fanout evals disable the persistent JAX compile cache unless you override
#   OFT_DISABLE_JAX_COMPILE_CACHE=0. This keeps the rerun path focused on the
#   native TORAX/JAX behavior instead of reusing a possibly incompatible cache.
#
# Required env vars:
#   DATASET_DIR        dataset root used for the training run
#   TRAIN_OUTPUT_DIR    directory containing checkpoint_step_*.pt (for example, <dataset>/iql/<run>)
#
# Optional env vars:
#   OUTPUT_ROOT        where per-checkpoint eval outputs go (default: <TRAIN_OUTPUT_DIR>/checkpoint_evals)
#   MAX_LOOP           TORAX loop count for evals (default: 1)
#   GRID_SIZE          TORAX radial grid size for evals (default: 51)
#   EVAL_WANDB_MODE    W&B mode for eval jobs (default: inherited / unset)
#   EVAL_WANDB_GROUP   W&B group for eval jobs (default: <TRAIN_OUTPUT_DIR basename>)
#   EVAL_WANDB_PROJECT W&B project for eval jobs (default: iql-training)
#   DRY_RUN            1 to print the jobs that would be submitted
#
# Example:
#   DATASET_DIR=./run_prev_action_full_YYYYMMDD_HHMMSS \
#   TRAIN_OUTPUT_DIR=./out/iql/run_prev_action_full_YYYYMMDD_HHMMSS/<run_id> \
#   ./run_scripts/fanout_checkpoint_evals.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/IQL.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd "${PROJECT_DIR}"

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the dataset root used for training}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:?Set TRAIN_OUTPUT_DIR to the directory containing checkpoint_step_*.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT_DIR%/}/checkpoint_evals}"
MAX_LOOP="${MAX_LOOP:-1}"
GRID_SIZE="${GRID_SIZE:-51}"
EVAL_WANDB_MODE="${EVAL_WANDB_MODE:-${WANDB_MODE:-}}"
EVAL_WANDB_GROUP="${EVAL_WANDB_GROUP:-$(basename "${TRAIN_OUTPUT_DIR%/}")}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-iql-training}"
OFT_DISABLE_JAX_COMPILE_CACHE="${OFT_DISABLE_JAX_COMPILE_CACHE:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_ROOT%/}/logs}"

mkdir -p "${RUN_LOG_DIR}" "${OUTPUT_ROOT}"
exec > >(tee -a "${RUN_LOG_DIR}/fanout_checkpoint_evals-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/fanout_checkpoint_evals-${SLURM_JOB_ID:-$$}.err" >&2)

echo "Running on host: $(hostname)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
echo "EVAL_WANDB_GROUP=${EVAL_WANDB_GROUP}"
echo "EVAL_WANDB_MODE=${EVAL_WANDB_MODE:-<unset>}"
echo "OFT_DISABLE_JAX_COMPILE_CACHE=${OFT_DISABLE_JAX_COMPILE_CACHE}"
echo "DRY_RUN=${DRY_RUN}"

shopt -s nullglob
checkpoints=("${TRAIN_OUTPUT_DIR}"/checkpoint_step_*.pt)
shopt -u nullglob

if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "ERROR: no checkpoint_step_*.pt files found in ${TRAIN_OUTPUT_DIR}" >&2
  exit 2
fi

replay_cache_dir=""
if [ -d "${DATASET_DIR%/}/replay_cache" ]; then
  replay_cache_dir="${DATASET_DIR%/}/replay_cache"
fi

for checkpoint in "${checkpoints[@]}"; do
  step="$(basename "${checkpoint}" .pt | sed 's/^checkpoint_step_//')"
  eval_output_dir="${OUTPUT_ROOT%/}/step_${step}"
  if [ -e "${eval_output_dir}/actor_eval_summary.json" ] || [ -e "${eval_output_dir}/actor_eval_bundle.pkl" ]; then
    echo "Skipping existing eval output: ${eval_output_dir}"
    continue
  fi

  export_args=(
    ALL
    DATASET_DIR="${DATASET_DIR}"
    ACTOR_CHECKPOINT="${checkpoint}"
    OUTPUT_DIR="${eval_output_dir}"
    RUN_ID="checkpoint_step_${step}"
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}"
    WANDB_GROUP="${EVAL_WANDB_GROUP}"
    MAX_LOOP="${MAX_LOOP}"
    GRID_SIZE="${GRID_SIZE}"
  )
  if [ -n "${EVAL_WANDB_MODE}" ]; then
    export_args+=(WANDB_MODE="${EVAL_WANDB_MODE}")
  fi
  if [ "${OFT_DISABLE_JAX_COMPILE_CACHE}" = "1" ]; then
    export_args+=(OFT_DISABLE_JAX_COMPILE_CACHE=1)
  fi
  if [ -n "${replay_cache_dir}" ]; then
    export_args+=(REPLAY_CACHE_DIR="${replay_cache_dir}")
  fi

  sbatch_args=(
    sbatch
    --parsable
    --account=nlp
    --partition=john
    --mem=128G
    --export="$(IFS=, ; echo "${export_args[*]}")"
    --output="${RUN_LOG_DIR}/checkpoint_eval-step_${step}-%j.out"
    --error="${RUN_LOG_DIR}/checkpoint_eval-step_${step}-%j.err"
  )
  if [ -n "${replay_cache_dir}" ]; then
    # The replay cache is already on disk if it exists in the dataset root; no
    # dependency is needed. Keep this comment to make the intent explicit.
    :
  fi

  if [ "${DRY_RUN}" != "0" ]; then
    printf '%q ' "${sbatch_args[@]}" "${PROJECT_DIR}/run_scripts/eval_iql_actor_cpu.sh"
    printf '\n'
    continue
  fi

  jid="$("${sbatch_args[@]}" "${PROJECT_DIR}/run_scripts/eval_iql_actor_cpu.sh")"
  echo "Submitted eval for step ${step}: ${jid}"
done
