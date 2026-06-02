#!/usr/bin/env bash
# eval_baseline_cpu.sh — Evaluate the TORAX baseline heating schedule on CPU.
#
# This is the apples-to-apples comparison for a trained IQL actor. It runs the
# same closed-loop TORAX workflow as eval_iql_actor_cpu.sh, but it does not load
# a checkpoint. TORAX therefore uses its built-in baseline fallback at each
# decision knot.
#
# Use this when you want:
#   - a reference run on the same dataset / max_loop / grid_size settings
#   - a direct comparison against a trained actor
#   - notebook-style plots/movie for the baseline schedule
#
# Required env vars:
#   DATASET_DIR   dataset root used to rebuild normalizers / initial relax state
#
# Key optional env vars (all have defaults):
#   OUTPUT_DIR               where results are written (default: <DATASET_DIR>/eval/<RUN_ID>)
#   MAX_LOOP                 MHD coupling loops (default: 1)
#   GRID_SIZE                TORAX radial grid points (default: 51)
#   WANDB_PROJECT            wandb project name (default: iql-training)
#   INITIAL_RELAX_CACHE_DIR  shared initial-relax cache dir (default: ./initial_relax_cache)
#   REPLAY_CACHE_DIR         preprocessed replay cache dir (optional)
#   RL_SEGMENT_TIMEOUT_SECONDS per-segment wall-clock timeout (default: 1800)
#   OFT_DISABLE_JAX_COMPILE_CACHE set to 1 to disable the persistent cache when debugging cache/runtime mismatches
#
# Example:
#   DATASET_DIR=./run_smoke_end2end_20260531_153343 \
#     RUN_ID=smoke_baseline_eval \
#     OFT_DISABLE_JAX_COMPILE_CACHE=1 \
#     sbatch run_scripts/eval_baseline_cpu.sh

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=john,sc-loprio
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
export UV_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER/uv_cache}"
mkdir -p "${UV_CACHE_DIR}"

DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the collected dataset root}"
RUN_ID="${RUN_ID:-baseline_eval_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR%/}/eval/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_DIR%/}/logs}"
mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/eval_baseline_cpu-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${RUN_LOG_DIR}/eval_baseline_cpu-${SLURM_JOB_ID:-$$}.err" >&2)
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

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
TOTAL_CPUS="${CPUS_PER_TASK}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-${TOTAL_CPUS}}"
THREADS_PER_WORKER="$(oft_cap_thread_budget "${THREADS_PER_WORKER}" "baseline eval")"

export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export OFT_DISABLE_JAX_COMPILE_CACHE="${OFT_DISABLE_JAX_COMPILE_CACHE:-1}"
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_DIR}/.jax_cache}"

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OFT_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"

echo "Running on host: $(hostname)"
echo "OFT_SELECTED_FLAVOR=${OFT_SELECTED_FLAVOR}"
echo "OFT_SELECTED_INSTALL=${OFT_SELECTED_INSTALL}"
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
echo "PHYSICAL_CORES=$(oft_detect_physical_cores || echo '<unknown>')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "JAX_COMPILATION_CACHE_DIR(base)=${JAX_COMPILATION_CACHE_DIR}"
echo "REPLAY_CACHE_DIR=${REPLAY_CACHE_DIR}"
echo "USE_REPLAY_CACHE=${USE_REPLAY_CACHE}"
echo "RL_SEGMENT_TIMEOUT_SECONDS=${RL_SEGMENT_TIMEOUT_SECONDS}"
echo "RL_MAX_ACTION_POWER_W=${RL_MAX_ACTION_POWER_W}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "TOTAL_CPUS=${TOTAL_CPUS}"

uv run python - <<'PY'
import os
from pathlib import Path

from collect_trajectories_delta import (
    default_relax_geometry,
    patch_initial_relax_cache_loader,
    preflight_required_inputs,
    resolve_seed_eqdsk_paths,
    setup_tokamaker,
    configure_tmtx,
)
from rl.eval_postprocess import postprocess_actor_eval
from rl.eval_sim import _plain, _bundle_safe, _action_history_to_array, _default_action_row


cwd = Path.cwd()
dataset_dir = os.environ["DATASET_DIR"]
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
log_dir = output_dir / "tokamaker_torax_logs"
log_dir.mkdir(parents=True, exist_ok=True)

grid_size = int(os.environ.get("GRID_SIZE", "51"))
max_loop = int(os.environ.get("MAX_LOOP", "1"))
rl_segment_timeout_seconds = float(os.environ.get("RL_SEGMENT_TIMEOUT_SECONDS", "1800"))
rl_max_action_power_w = float(os.environ.get("RL_MAX_ACTION_POWER_W", "150000000"))

eqdsk_list = resolve_seed_eqdsk_paths(str(cwd))
geom = default_relax_geometry()
preflight_required_inputs(
    str(cwd),
    eqdsk_list,
    initial_relax_cache=None,
    require_initial_relax_cache=False,
)
mygs, _, _, _ = setup_tokamaker(str(cwd))
tmtx = configure_tmtx(
    mygs,
    _default_action_row(),
    eqdsk_list,
    geom["eqtimes"],
    geom["coil_bounds"],
    geom["x_points"],
    geom["Ip_targets"],
    geom["ne_init"],
    geom["Te_init"],
    geom["psi_sample"],
    grid_size=grid_size,
)
patch_initial_relax_cache_loader(tmtx)

result = {}
started = os.popen("date -Iseconds").read().strip()
try:
    tmtx.fly(
        output_mode=False,
        max_loop=max_loop,
        run_name=os.environ.get("RUN_NAME", "baseline_eval"),
        t_ave_toggle="flattop",
        t_ave_window=25,
        relax=True,
        relax_duration=5,
        initial_relax_state=os.environ.get("INITIAL_RELAX_STATE") or None,
        log_dir=str(log_dir),
        use_rl_actor=True,
        actor_checkpoint=None,
        rl_segment_timeout_seconds=rl_segment_timeout_seconds,
        rl_max_action_power_w=rl_max_action_power_w,
    )
    summary = tmtx.summary()
    rewards = tmtx.compute_rewards()
    actions, action_records = _action_history_to_array(getattr(tmtx, "_rl_actions_history", []))
    metrics = {
        "actor_eval/reward_total": float(sum(rewards)),
        "actor_eval/reward_mean": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "actor_eval/n_decisions": int(len(actions)),
    }
    for key, val in summary.items():
        if val is not None:
            metrics[f"actor_eval/{key}"] = float(val)
    result = {
        "status": "success",
        "started_at": started,
        "finished_at": os.popen("date -Iseconds").read().strip(),
        "summary": _plain(summary),
        "rewards": _plain(rewards),
        "actions": _plain(actions),
        "action_records": action_records,
        "metrics": _plain(metrics),
        "output_dir": str(output_dir),
        "actor_checkpoint": None,
        "initial_relax_state": _bundle_safe(os.environ.get("INITIAL_RELAX_STATE") or None),
    }
    import json, pickle
    with (output_dir / "actor_eval_summary.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (output_dir / "actor_eval_actions.json").open("w") as f:
        json.dump(
            {"actions": _plain(actions), "action_records": action_records, "metrics": _plain(metrics)},
            f,
            indent=2,
        )
    with (output_dir / "actor_eval_bundle.pkl").open("wb") as f:
        pickle.dump(
            {
                "state": _bundle_safe(tmtx._state),
                "tm_times": list(getattr(tmtx, "_tm_times", [])),
                "current_loop": getattr(tmtx, "_current_loop", None),
                "flattop": _bundle_safe(getattr(tmtx, "_flattop", None)),
                "coil_bounds": _bundle_safe(getattr(tmtx, "_coil_bounds", None)),
                "results": _bundle_safe(getattr(tmtx, "_results", None)),
                "output_mode": getattr(tmtx, "_output_mode", None),
                "actor_checkpoint": None,
                "initial_relax_state": os.environ.get("INITIAL_RELAX_STATE") or None,
                "grid_size": grid_size,
                "max_loop": max_loop,
            },
            f,
        )
finally:
    pass

postprocess_actor_eval(
    result={"bundle_path": str(output_dir / "actor_eval_bundle.pkl")},
    output_dir=output_dir,
    render_plots=True,
    render_movie=True,
    render_summary=True,
    tmtx=tmtx,
)
PY
