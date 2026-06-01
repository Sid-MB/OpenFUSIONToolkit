#!/usr/bin/env bash
# eval_iql_actor_cpu_batch.sh — Evaluate multiple IQL checkpoints in parallel (CPU).
#
# Wraps rl/eval_batch.py to run N_WORKERS evals simultaneously, each in its own
# process, each calling the full TokaMaker_TORAX RL closed-loop eval. Within a
# single eval the work is serial (the RL decision chain is a sequential dependency);
# parallelism comes from evaluating different checkpoints or seeds concurrently.
#
# Thread budget: TOTAL_CPUS is divided evenly across N_WORKERS so no worker
# oversubscribes. Both Python (OMP/MKL/OpenBLAS) and XLA (intra_op) threads are
# capped per worker.
#
# Shared compilation cache (.jax_cache/): the first worker that runs compiles the
# TORAX XLA executable and writes it to the cache. Every subsequent worker in the
# same job (and future runs on the same machine) loads it from disk instead of
# recompiling. With compile-once schedules (constant-length heating arrays) there
# is only a single distinct executable to cache.
#
# Requires at least one of:
#   ACTOR_CHECKPOINTS  space-separated list of checkpoint paths
#   CHECKPOINTS_FILE   path to a file with one checkpoint path per line (# = comment)
#
# Example — 4 checkpoints in parallel, 8 CPUs each (32 total):
#   ACTOR_CHECKPOINTS="out/iql/run_a/iql_weights.pt out/iql/run_b/iql_weights.pt \
#     out/iql/run_c/checkpoint_step_50000.pt out/iql/run_d/checkpoint_step_50000.pt" \
#     N_WORKERS=4 \
#     sbatch run_scripts/eval_iql_actor_cpu_batch.sh
#
# Example — checkpoints from a file, synchronous (blocks until done):
#   CHECKPOINTS_FILE=my_checkpoints.txt \
#     N_WORKERS=4 \
#     env -i PATH="$PATH" HOME="$HOME" TERM="$TERM" \
#     srun --account=nlp --partition=john \
#     /bin/bash run_scripts/eval_iql_actor_cpu_batch.sh
#
# Key optional env vars (all have defaults):
#   N_WORKERS          parallel eval processes (default: 4)
#   OUTPUT_ROOT        root dir for all per-eval output dirs (default: <DATASET_DIR>/eval_batch/<id>)
#   DATASET_DIR        dataset for normalizer reconstruction (optional if ckpt has normalizers)
#   MAX_LOOP           MHD coupling loops per eval (default: 2)
#   GRID_SIZE          TORAX radial grid points (default: 51)
#   WANDB_PROJECT      wandb project name (default: iql-training)
#   INITIAL_RELAX_CACHE_DIR shared initial-relax cache dir
#   JAX_COMPILATION_CACHE_DIR persistent XLA cache root (runtime namespaces by build fingerprint; default: ./.jax_cache)
#   MP_CONTEXT         multiprocessing start method: fork (default) or spawn
#   DISABLE_JAX_COMPILE_CACHE set to 1 to disable the persistent cache
#   ALLOW_MISMATCHED_REWARDS set to 1 only when you intentionally want to evaluate checkpoints against a different reward config

#SBATCH --account=nlp
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --partition=john
#SBATCH --mail-user=siddharth@cs.stanford.edu
#SBATCH --mail-type=FAIL
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
source "${PROJECT_DIR}/run_scripts/lib/threading.sh"

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-20}"
TOTAL_CPUS="${CPUS_PER_TASK}"
N_WORKERS="${N_WORKERS:-4}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( TOTAL_CPUS / N_WORKERS ))}"
THREADS_PER_WORKER="$(oft_cap_thread_budget "${THREADS_PER_WORKER}" "batch eval worker")"

ACTOR_CHECKPOINTS="${ACTOR_CHECKPOINTS:-}"
CHECKPOINTS_FILE="${CHECKPOINTS_FILE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_DIR%/}/eval_batch/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_ROOT%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/eval_iql_actor_cpu_batch-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/eval_iql_actor_cpu_batch-${SLURM_JOB_ID:-$$}.err" >&2)
WANDB_GROUP="${WANDB_GROUP:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
DATASET_DIR="${DATASET_DIR:-}"
WANDB_PROJECT="${WANDB_PROJECT:-iql-training}"
MAX_LOOP="${MAX_LOOP:-1}"
GRID_SIZE="${GRID_SIZE:-51}"
INITIAL_RELAX_CACHE_DIR="${INITIAL_RELAX_CACHE_DIR:-}"
REPLAY_CACHE_DIR="${REPLAY_CACHE_DIR:-}"
USE_REPLAY_CACHE="${USE_REPLAY_CACHE:-1}"
RL_SEGMENT_TIMEOUT_SECONDS="${RL_SEGMENT_TIMEOUT_SECONDS:-1800}"
RL_MAX_ACTION_POWER_W="${RL_MAX_ACTION_POWER_W:-150000000}"

# Force the CPU path even if launched from a GPU-capable login node.
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export OFT_DISABLE_JAX_COMPILE_CACHE="${OFT_DISABLE_JAX_COMPILE_CACHE:-1}"
export PYTHONUNBUFFERED=1
export MP_CONTEXT="${MP_CONTEXT:-fork}"

# Shared persistent XLA compilation cache: first worker compiles, the rest load it.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_DIR}/.jax_cache}"

# Split native math/OpenMP threads across workers to avoid oversubscription.
export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OFT_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"
# Cap XLA CPU intra-op threads per worker too.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS_PER_WORKER}"

echo "Running on host: $(hostname)"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "TOTAL_CPUS=${TOTAL_CPUS}"
echo "N_WORKERS=${N_WORKERS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "PHYSICAL_CORES=$(oft_detect_physical_cores || echo '<unknown>')"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "MAX_LOOP=${MAX_LOOP}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "JAX_COMPILATION_CACHE_DIR(base)=${JAX_COMPILATION_CACHE_DIR}"
echo "MP_CONTEXT=${MP_CONTEXT}"
echo "ALLOW_MISMATCHED_REWARDS=${ALLOW_MISMATCHED_REWARDS:-0}"

ARGS=(
  --output_root "${OUTPUT_ROOT}"
  --project "${WANDB_PROJECT}"
  --wandb_group "${WANDB_GROUP}"
  --max_loop "${MAX_LOOP}"
  --grid_size "${GRID_SIZE}"
  --n_workers "${N_WORKERS}"
  --rl_segment_timeout_seconds "${RL_SEGMENT_TIMEOUT_SECONDS}"
  --rl_max_action_power_w "${RL_MAX_ACTION_POWER_W}"
)
# ACTOR_CHECKPOINTS is a whitespace-separated list.
for ckpt in ${ACTOR_CHECKPOINTS}; do
  ARGS+=(--actor_checkpoint "${ckpt}")
done
if [ -n "${CHECKPOINTS_FILE}" ]; then
  ARGS+=(--checkpoints_file "${CHECKPOINTS_FILE}")
fi
if [ -n "${DATASET_DIR}" ]; then
  ARGS+=(--dataset_dir "${DATASET_DIR}")
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

uv run python -m rl.eval_batch "${ARGS[@]}"
