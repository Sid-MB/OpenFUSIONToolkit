"""Evaluate a trained IQL actor with TokaMaker_TORAX in RL closed-loop mode.

Single-checkpoint entry point. For evaluating multiple checkpoints in parallel
use rl/eval_batch.py (driven by run_scripts/eval_iql_actor_cpu_batch.sh).

The eval reproduces the exact trajectory-collection convention: TORAX is cold-
started from t=0 for every RL decision segment so that each observation is
consistent with the heating schedule accumulated so far. See
TokaMaker_TORAX._run_tx_rl_segmented and _full_agent_knots_defaults for the
compile-once schedule fix that eliminates per-segment XLA recompilation.

Performance notes:
  - JAX compilation cache is enabled by default (see block below). The first
    eval compiles once and writes to .jax_cache/; subsequent evals (including
    batch workers) load from cache without recompiling.
  - CPU is the recommended backend for this workload (see eval_iql_actor_cpu.sh).
    The RL loop is latency-bound on a 51-point 1-D grid; GPU util spikes briefly
    then sits idle, while CPU avoids GPU memory preallocation and kernel-launch
    overhead.

Main entry points:
  run_actor_eval_from_config(...)   programmatic use
  main()                            CLI (python -m rl.eval --actor_checkpoint ...)
"""

import argparse
import io
import json
import os
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

# Set JAX compilation cache env vars before JAX is imported. pulse_design.fly()
# also enables the cache at the jax.config level, but setting these env vars
# ensures the cache dir is configured even if JAX initializes before fly() runs.
# An explicit JAX_COMPILATION_CACHE_DIR from the environment (e.g. set by the
# run script) takes precedence over the default here.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".jax_cache"),
)
# Cache every compile regardless of how fast or how small — the RL segments
# reuse a single executable so any cache hit is worth it.
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import torch
import wandb

from IQL import ReplayBuffer, normalize_buffer
from dataloader import describe_dataset_with_replay_cache, load_d4rl_dataset
from log import get_logger

logger = get_logger(__name__)


def _plain(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _default_action_row():
    from collect_trajectories_delta import DECISION_TIMES
    from OpenFUSIONToolkit.TokaMaker.pulse_design import TokaMaker_TORAX

    actions = []
    for decision_t in DECISION_TIMES:
        knot_t = decision_t + 20
        ecrh_w, nbi_w = TokaMaker_TORAX._rl_default_action_w_at_time(knot_t)
        actions.append([ecrh_w, nbi_w])
    return np.asarray(actions, dtype=np.float64)


def _normalizers_from_dataset(dataset_dir, replay_cache_dir=None, prefer_replay_cache=True):
    specs = describe_dataset_with_replay_cache(
        dataset_dir,
        cache_dir=replay_cache_dir,
        prefer_cache=prefer_replay_cache,
    )
    buffer = ReplayBuffer(
        specs["state_dim"],
        specs["action_dim"],
        specs["num_transitions"],
    )
    load_d4rl_dataset(
        str(dataset_dir),
        buffer,
        specs["state_keys"],
        cache_dir=replay_cache_dir,
        prefer_cache=prefer_replay_cache,
    )
    normalizers = normalize_buffer(buffer)
    return specs, normalizers


def prepare_actor_checkpoint(
    actor_checkpoint,
    output_dir,
    dataset_dir=None,
    replay_cache_dir=None,
    prefer_replay_cache=True,
):
    actor_checkpoint = Path(actor_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    ckpt = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    has_normalizers = "state_mean" in ckpt and "state_std" in ckpt
    if has_normalizers:
        ckpt["_checkpoint_had_normalizers"] = True
        return actor_checkpoint, ckpt, None
    if dataset_dir is None:
        raise ValueError(
            "Actor checkpoint has no state_mean/state_std. Pass --dataset_dir so eval can "
            "reconstruct the training normalizers."
        )

    specs, normalizers = _normalizers_from_dataset(
        Path(dataset_dir).resolve(),
        replay_cache_dir=replay_cache_dir,
        prefer_replay_cache=prefer_replay_cache,
    )
    patched = dict(ckpt)
    patched["state_mean"] = torch.as_tensor(normalizers["state_mean"])
    patched["state_std"] = torch.as_tensor(normalizers["state_std"])
    patched.setdefault("action_max", torch.as_tensor(normalizers["action_max"]))
    patched.setdefault("state_keys", specs["state_keys"])
    patched.setdefault("state_dim", specs["state_dim"])
    patched.setdefault("action_dim", specs["action_dim"])
    patched["_checkpoint_had_normalizers"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    patched_path = output_dir / f"{actor_checkpoint.stem}_with_eval_normalizers.pt"
    torch.save(patched, patched_path)
    logger.info("Wrote eval-ready actor checkpoint to %s", patched_path)
    return patched_path, patched, specs


def run_actor_eval_from_config(
    actor_checkpoint,
    output_dir,
    dataset_dir=None,
    project="iql-training",
    run_name=None,
    wandb_mode=None,
    initial_relax_state=None,
    initial_relax_cache_dir=None,
    max_loop=2,
    grid_size=51,
    device=None,
    replay_cache_dir=None,
    prefer_replay_cache=True,
    allow_cpu_jax_on_gpu=False,
    rl_segment_timeout_seconds=1800,
    rl_max_action_power_w=150.0e6,
):
    from collect_trajectories_delta import (
        build_initial_relax_cache,
        configure_tmtx,
        default_initial_relax_cache_dir,
        default_relax_geometry,
        patch_initial_relax_cache_loader,
        preflight_required_inputs,
        resolve_initial_relax_cache_path,
        resolve_seed_eqdsk_paths,
        setup_tokamaker,
        validate_jax_backend,
    )
    from OpenFUSIONToolkit.TokaMaker.pulse_design import RL_DECISION_TIMES
    n_decisions = len(RL_DECISION_TIMES)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_path, ckpt, dataset_specs = prepare_actor_checkpoint(
        actor_checkpoint,
        output_dir,
        dataset_dir=dataset_dir,
        replay_cache_dir=replay_cache_dir,
        prefer_replay_cache=prefer_replay_cache,
    )

    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    cwd = Path.cwd()
    eqdsk_list = resolve_seed_eqdsk_paths(str(cwd))

    action_row = _default_action_row()
    geom = default_relax_geometry()
    coil_bounds = geom["coil_bounds"]
    eqtimes = geom["eqtimes"]
    x_points = geom["x_points"]
    psi_sample = geom["psi_sample"]
    ip_targets = geom["Ip_targets"]
    ne_init = geom["ne_init"]
    te_init = geom["Te_init"]

    # Resolve the shared initial-relax cache. Explicit path wins; otherwise reuse
    # a dataset's legacy cache if present; otherwise use a keyed file in the
    # shared cache dir (built on demand below).
    relax_cache_params = None
    if initial_relax_state is None:
        legacy = None if dataset_dir is None else Path(dataset_dir) / "initial_relax_state.json"
        if legacy is not None and legacy.is_file():
            initial_relax_state = str(legacy)
        else:
            cache_dir = initial_relax_cache_dir or default_initial_relax_cache_dir()
            initial_relax_state, relax_key, relax_cache_params = resolve_initial_relax_cache_path(
                cache_dir, grid_size, ne_init, te_init, psi_sample,
                eqtimes, ip_targets, coil_bounds, x_points, eqdsk_list,
            )
            logger.info("Keyed initial relax cache: %s (key=%s)", initial_relax_state, relax_key)

    relax_cache_exists = initial_relax_state is not None and os.path.exists(initial_relax_state)
    preflight_required_inputs(
        str(cwd),
        eqdsk_list,
        initial_relax_cache=initial_relax_state,
        require_initial_relax_cache=relax_cache_exists,
    )
    validate_jax_backend(require_cuda_on_gpu=not allow_cpu_jax_on_gpu)

    config = {
        "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
        "eval_actor_checkpoint": str(actor_path),
        "dataset_dir": None if dataset_dir is None else str(Path(dataset_dir).resolve()),
        "output_dir": str(output_dir),
        "initial_relax_state": initial_relax_state,
        "max_loop": max_loop,
        "grid_size": grid_size,
        "checkpoint_has_normalizers": bool(ckpt.get("_checkpoint_had_normalizers", True)),
        "state_dim": int(ckpt.get("state_dim", 0) or len(ckpt["state_mean"])),
        "action_dim": int(ckpt.get("action_dim", 2)),
        "state_keys": ckpt.get("state_keys"),
        "dataset_specs": dataset_specs,
        "rl_segment_timeout_seconds": rl_segment_timeout_seconds,
        "rl_max_action_power_w": rl_max_action_power_w,
    }

    wandb_kwargs = {"project": project, "config": _plain(config), "reinit": True}
    if run_name:
        wandb_kwargs["name"] = run_name
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode
    run = wandb.init(**wandb_kwargs)
    # Tell wandb to use decision_index as the x-axis for all live per-decision
    # charts, so the plots show "decision 0 … 20" rather than an internal counter.
    wandb.define_metric("actor_eval_live/*", step_metric="actor_eval_live/decision_index")

    log_dir = output_dir / "tokamaker_torax_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Populated just before tmtx.fly() so elapsed_s in live logs covers the
    # full fly() duration, including the initial [0→80] cold-start.
    _fly_t0 = [0.0]

    def log_rl_event(event):
        """Stream one RL decision to wandb as it happens (live progress)."""
        if event.get("event") != "decision":
            return
        try:
            idx = int(event["decision_index"])
            ecrh_mw = float(event["ecrh_MW"])
            nbi_mw = float(event["nbi_MW"])
            wandb.log({
                # x-axis for these charts (see define_metric above)
                "actor_eval_live/decision_index": idx,
                # physical time this decision corresponds to
                "actor_eval_live/decision_t_s": float(event["decision_t"]),
                # actions chosen by the actor
                "actor_eval_live/ecrh_MW": ecrh_mw,
                "actor_eval_live/nbi_MW": nbi_mw,
                "actor_eval_live/total_heating_MW": ecrh_mw + nbi_mw,
                # wall-clock elapsed since fly() started (includes initial cold-start)
                "actor_eval_live/elapsed_s": time.time() - _fly_t0[0],
                # fraction of decisions completed (useful progress bar proxy)
                "actor_eval_live/progress": (idx + 1) / n_decisions,
            })
        except Exception as exc:
            logger.warning("Could not stream RL event to wandb: %s", exc)

    started = datetime.now().isoformat()
    try:
        mygs, _, _, _ = setup_tokamaker(str(cwd))
        if not os.path.exists(initial_relax_state):
            build_initial_relax_cache(
                mygs,
                action_row,
                initial_relax_state,
                eqdsk_list,
                eqtimes,
                coil_bounds,
                x_points,
                ip_targets,
                ne_init,
                te_init,
                psi_sample,
                log_dir=str(log_dir),
                grid_size=grid_size,
                params=relax_cache_params,
            )

        tmtx = configure_tmtx(
            mygs,
            action_row,
            eqdsk_list,
            eqtimes,
            coil_bounds,
            x_points,
            ip_targets,
            ne_init,
            te_init,
            psi_sample,
            grid_size=grid_size,
        )
        patch_initial_relax_cache_loader(tmtx)
        _fly_t0[0] = time.time()
        tmtx.fly(
            output_mode=False,
            max_loop=max_loop,
            run_name=run_name or "iql_actor_eval",
            t_ave_toggle="flattop",
            t_ave_window=25,
            relax=True,
            relax_duration=5,
            initial_relax_state=initial_relax_state,
            log_dir=str(log_dir),
            use_rl_actor=True,
            actor_checkpoint=str(actor_path),
            rl_event_callback=log_rl_event,
            rl_segment_timeout_seconds=rl_segment_timeout_seconds,
            rl_max_action_power_w=rl_max_action_power_w,
        )

        with redirect_stdout(io.StringIO()):
            summary = tmtx.summary()
        rewards = tmtx.compute_rewards()
        actions = getattr(tmtx, "_rl_actions_history", [])

        metrics = {
            "actor_eval/reward_total": float(np.sum(rewards)),
            "actor_eval/reward_mean": float(np.mean(rewards)),
            "actor_eval/reward_min": float(np.min(rewards)),
            "actor_eval/reward_max": float(np.max(rewards)),
            "actor_eval/n_decisions": len(actions),
        }
        # All physics figures of merit returned by summary() — Q, flux, stability,
        # confinement, temperatures, density, heating. Log all non-None values so
        # wandb captures the full physics outcome without a manual keep-list.
        for key, val in summary.items():
            if val is not None:
                metrics[f"actor_eval/{key}"] = float(val)

        # Action table: one row per RL decision (human-readable MW columns only).
        action_columns = ["decision_t", "knot_t", "ecrh_MW", "nbi_MW"]
        action_table = wandb.Table(
            columns=action_columns,
            data=[[row[col] for col in action_columns] for row in actions],
        )
        # Reward table: one row per decision with physical time for context.
        reward_table = wandb.Table(
            columns=["decision_index", "decision_t_s", "reward"],
            data=[
                [idx, float(RL_DECISION_TIMES[idx]), float(r)]
                for idx, r in enumerate(rewards)
            ],
        )
        wandb.log({
            **metrics,
            "actor_eval/actions": action_table,
            "actor_eval/rewards_per_decision": reward_table,
        })
        run.summary.update(metrics)

        result = {
            "status": "success",
            "started_at": started,
            "finished_at": datetime.now().isoformat(),
            "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
            "eval_actor_checkpoint": str(actor_path),
            "initial_relax_state": initial_relax_state,
            "summary": _plain(summary),
            "rewards": _plain(rewards),
            "actions": _plain(actions),
            "metrics": _plain(metrics),
        }
        result_path = output_dir / "actor_eval_summary.json"
        with result_path.open("w") as f:
            json.dump(result, f, indent=2)
        wandb.save(str(result_path))
        logger.info("Actor eval summary saved to %s", result_path)
        return result
    finally:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an IQL actor with TokaMaker_TORAX.fly(use_rl_actor=True)."
    )
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "iql-training"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--initial_relax_state", default=None,
                        help="Explicit initial-relax cache path (legacy/override). When "
                             "omitted, a keyed file in --initial_relax_cache_dir is used.")
    parser.add_argument("--initial_relax_cache_dir", default=None,
                        help="Shared directory for initial-relax caches keyed by "
                             "(grid_size, initial profiles, equilibrium). Defaults to the "
                             "INITIAL_RELAX_CACHE_DIR env var or <repo>/initial_relax_cache.")
    parser.add_argument("--max_loop", type=int, default=2)
    parser.add_argument("--grid_size", type=int, default=51)
    parser.add_argument("--device", default=None)
    parser.add_argument("--replay_cache_dir", default=None)
    parser.add_argument("--no_replay_cache", action="store_true")
    parser.add_argument("--allow_cpu_jax_on_gpu", action="store_true")
    parser.add_argument(
        "--rl_segment_timeout_seconds",
        type=float,
        default=float(os.environ.get("RL_SEGMENT_TIMEOUT_SECONDS", "1800")),
        help="Wall-clock timeout per RL TORAX segment; <=0 disables.",
    )
    parser.add_argument(
        "--rl_max_action_power_w",
        type=float,
        default=float(os.environ.get("RL_MAX_ACTION_POWER_W", "150000000")),
        help="Per-actuator RL action cap in watts; <=0 disables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        checkpoint = Path(args.actor_checkpoint).resolve()
        output_dir = checkpoint.parent / "actor_eval"
    run_actor_eval_from_config(
        actor_checkpoint=args.actor_checkpoint,
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        project=args.project,
        run_name=args.run_name,
        wandb_mode=args.wandb_mode,
        initial_relax_state=args.initial_relax_state,
        initial_relax_cache_dir=args.initial_relax_cache_dir,
        max_loop=args.max_loop,
        grid_size=args.grid_size,
        device=args.device,
        replay_cache_dir=args.replay_cache_dir,
        prefer_replay_cache=not args.no_replay_cache,
        allow_cpu_jax_on_gpu=args.allow_cpu_jax_on_gpu,
        rl_segment_timeout_seconds=args.rl_segment_timeout_seconds,
        rl_max_action_power_w=args.rl_max_action_power_w,
    )


if __name__ == "__main__":
    main()
