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
#   TRAIN_OUTPUT_DIR    IQL run dir; checkpoints are read from its checkpoints/ subfolder
#                       (for example, <dataset>/iql/<run>), with a fallback to the run dir itself
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
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:?Set TRAIN_OUTPUT_DIR to the IQL run dir (checkpoints read from its checkpoints/ subfolder)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT_DIR%/}/checkpoint_evals}"
MAX_LOOP="${MAX_LOOP:-1}"
GRID_SIZE="${GRID_SIZE:-51}"
EVAL_ARRAY_CONCURRENCY="${EVAL_ARRAY_CONCURRENCY:-16}"
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
# Checkpoints now live in the checkpoints/ subfolder; fall back to the run-dir
# top level for older runs that saved them directly under TRAIN_OUTPUT_DIR.
checkpoints=("${TRAIN_OUTPUT_DIR}"/checkpoints/checkpoint_step_*.pt)
if [ "${#checkpoints[@]}" -eq 0 ]; then
  checkpoints=("${TRAIN_OUTPUT_DIR}"/checkpoint_step_*.pt)
fi
shopt -u nullglob

if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "ERROR: no checkpoint_step_*.pt files found in ${TRAIN_OUTPUT_DIR}/checkpoints/ or ${TRAIN_OUTPUT_DIR}" >&2
  exit 2
fi

replay_cache_dir=""
if [ -d "${DATASET_DIR%/}/replay_cache" ]; then
  replay_cache_dir="${DATASET_DIR%/}/replay_cache"
fi

# Build a manifest of checkpoints that still need an eval (skip completed ones),
# then submit ONE Slurm array — one task per checkpoint. Each array task maps
# SLURM_ARRAY_TASK_ID to its manifest line inside eval_iql_actor_cpu.sh.
manifest="${OUTPUT_ROOT%/}/checkpoint_manifest.txt"
: > "${manifest}"
n=0
for checkpoint in "${checkpoints[@]}"; do
  step="$(basename "${checkpoint}" .pt | sed 's/^checkpoint_step_//')"
  eval_output_dir="${OUTPUT_ROOT%/}/step_${step}"
  if [ -e "${eval_output_dir}/actor_eval_summary.json" ] || [ -e "${eval_output_dir}/actor_eval_bundle.pkl" ]; then
    echo "Skipping existing eval output: ${eval_output_dir}"
    continue
  fi
  echo "${checkpoint}" >> "${manifest}"
  n=$((n + 1))
done

if [ "${n}" -eq 0 ]; then
  echo "Nothing to evaluate: every checkpoint already has eval output under ${OUTPUT_ROOT}."
  exit 0
fi

# Shared env for every array task; the per-checkpoint ACTOR_CHECKPOINT/OUTPUT_DIR
# are derived from CHECKPOINT_MANIFEST + SLURM_ARRAY_TASK_ID inside the eval script.
export_args=(
  ALL
  DATASET_DIR="${DATASET_DIR}"
  OUTPUT_ROOT="${OUTPUT_ROOT}"
  CHECKPOINT_MANIFEST="${manifest}"
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

array_spec="0-$((n - 1))%${EVAL_ARRAY_CONCURRENCY}"
# Resources (account/partition/mem/cpus) come from eval_iql_actor_cpu.sh's #SBATCH headers.
sbatch_args=(
  sbatch
  --parsable
  --array="${array_spec}"
  --export="$(IFS=, ; echo "${export_args[*]}")"
  --output="${RUN_LOG_DIR}/checkpoint_eval-%A_%a.out"
  --error="${RUN_LOG_DIR}/checkpoint_eval-%A_%a.err"
)

if [ "${DRY_RUN}" != "0" ]; then
  echo "Would submit a checkpoint-eval array over ${n} checkpoints (${array_spec}):"
  printf '%q ' "${sbatch_args[@]}" "${PROJECT_DIR}/run_scripts/eval_iql_actor_cpu.sh"
  printf '\n'
  echo "Manifest ${manifest}:"
  cat "${manifest}"
  exit 0
fi

jid="$("${sbatch_args[@]}" "${PROJECT_DIR}/run_scripts/eval_iql_actor_cpu.sh")"
echo "Submitted checkpoint-eval array over ${n} checkpoints: ${jid} (${array_spec})"
