#!/usr/bin/env bash
# eval_iql_actor_cpu.sh — Evaluate one IQL actor checkpoint on the CPU (john partition).
#
# Runs rl/eval.py with TokaMaker_TORAX in RL closed-loop mode, forcing JAX/TORAX
# onto CPU. This is the canonical single-checkpoint path for notebook-style
# outputs: it writes the summary bundle and then renders the plots/movie from
# the live tmtx object before the process exits. The IQL actor (PyTorch) already
# runs on CPU regardless.
#
# Why CPU instead of GPU?
#   The RL eval loop is latency-bound, not compute-bound: each of the ~22 cold-start
#   TORAX segments per loop is a short 1-D solve on a 51-point grid. The dominant
#   cost was XLA recompilation (~30 s per segment on GPU). Two changes fix this:
#     1. compile-once: heating-schedule arrays are constant length across segments
#        so JAX's jitted step_fn compiles once and every segment reuses it.
#     2. persistent cache: the single compile is written to .jax_cache/ and loaded
#        from disk by later runs/workers, so only the very first eval on a clean
#        cache pays any compilation cost at all.
#   On CPU, XLA compilation is also cheaper, and there is no GPU memory preallocation
#   overhead. For this workload CPU (john) is comparable to or faster than GPU.
#
# This script is the CPU counterpart of eval_iql_actor.sh (GPU, jag-standard).
# For evaluating multiple checkpoints in parallel use eval_iql_actor_cpu_batch.sh.
#
# Required env var:
#   ACTOR_CHECKPOINT  path to iql_weights.pt or checkpoint_step_*.pt
#
# Example — submit as a Slurm job:
#   ACTOR_CHECKPOINT=out/iql/<run>/iql_weights.pt \
#     DATASET_DIR=./rl_dataset_eval_smoke_1_20260528_130200 \
#     sbatch run_scripts/eval_iql_actor_cpu.sh
#
# Example — run synchronously (blocks until done, useful for testing/timing):
#   ACTOR_CHECKPOINT=out/iql/<run>/iql_weights.pt \
#     DATASET_DIR=./rl_dataset_eval_smoke_1_20260528_130200 \
#     WANDB_MODE=offline \
#     env -i PATH="$PATH" HOME="$HOME" TERM="$TERM" \
#     srun --account=nlp --partition=john \
#     /bin/bash run_scripts/eval_iql_actor_cpu.sh
#
# Key optional env vars (all have defaults):
#   OUTPUT_DIR               where results are written (default: <DATASET_DIR>/eval/<RUN_ID>)
#   MAX_LOOP                 MHD coupling loops (default: 2)
#   GRID_SIZE                TORAX radial grid points (default: 51)
#   WANDB_PROJECT            wandb project name (default: iql-training)
#   INITIAL_RELAX_CACHE_DIR  shared initial-relax cache dir (default: ./initial_relax_cache)
#   REPLAY_CACHE_DIR         preprocessed replay cache dir (optional)
#   JAX_COMPILATION_CACHE_DIR persistent XLA cache root (runtime namespaces by build fingerprint; default: ./.jax_cache)
#   RL_SEGMENT_TIMEOUT_SECONDS per-segment wall-clock timeout (default: 1800)
#   OFT_DISABLE_JAX_COMPILE_CACHE set to 1 to disable the persistent cache when debugging a cache/runtime mismatch

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=john
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
RUN_ID="${RUN_ID:-$(basename "${ACTOR_CHECKPOINT%.pt}")_eval_cpu_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR%/}/eval/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/eval_iql_actor_cpu-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/eval_iql_actor_cpu-${SLURM_JOB_ID:-$$}.err" >&2)
WANDB_PROJECT="${WANDB_PROJECT:-iql-training}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-${SLURM_JOB_ID:-${RUN_ID}}}"
INITIAL_RELAX_STATE="${INITIAL_RELAX_STATE:-}"
INITIAL_RELAX_CACHE_DIR="${INITIAL_RELAX_CACHE_DIR:-}"
MAX_LOOP="${MAX_LOOP:-1}"
GRID_SIZE="${GRID_SIZE:-51}"
REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR:-}"
USE_REPLAY_CACHE="${USE_REPLAY_CACHE:-1}"
RL_SEGMENT_TIMEOUT_SECONDS="${RL_SEGMENT_TIMEOUT_SECONDS:-1800}"
RL_MAX_ACTION_POWER_W="${RL_MAX_ACTION_POWER_W:-150000000}"

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-20}"
TOTAL_CPUS="${CPUS_PER_TASK}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-${TOTAL_CPUS}}"
if [ "${THREADS_PER_WORKER}" -lt 1 ]; then
  THREADS_PER_WORKER=1
fi

# Force the CPU path even if this script is run from a GPU-capable login node.
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1

# Persistent XLA compilation cache: the single TORAX compile is reused across
# processes/runs (e.g. a later batch worker loads it from disk instead of recompiling).
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_DIR}/.jax_cache}"
if [ "${OFT_DISABLE_JAX_COMPILE_CACHE:-0}" = "1" ]; then
  export OFT_DISABLE_JAX_COMPILE_CACHE=1
fi

# Keep native math/OpenMP libraries from oversubscribing cores.
export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OFT_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"

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
echo "TOTAL_CPUS=${TOTAL_CPUS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "JAX_COMPILATION_CACHE_DIR(base)=${JAX_COMPILATION_CACHE_DIR}"
echo "REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
echo "USE_REPLAY_CACHE=${USE_REPLAY_CACHE}"
echo "RL_SEGMENT_TIMEOUT_SECONDS=${RL_SEGMENT_TIMEOUT_SECONDS}"
echo "RL_MAX_ACTION_POWER_W=${RL_MAX_ACTION_POWER_W}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "TOTAL_CPUS=${TOTAL_CPUS}"

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
if [ -n "${REPLAY_CACHE_DIR}" ]; then
  ARGS+=(--replay_cache_dir "${REPLAY_CACHE_DIR}")
fi
if [ "${USE_REPLAY_CACHE}" = "0" ]; then
  ARGS+=(--no_replay_cache)
fi

uv run python -m rl.eval "${ARGS[@]}"
# Example:
#   ACTOR_CHECKPOINT=out/iql/<run>/iql_weights.pt \
#     DATASET_DIR=./rl_dataset_eval_smoke_1_20260528_130200 \
#     sbatch run_scripts/eval_iql_actor_cpu.sh
#
