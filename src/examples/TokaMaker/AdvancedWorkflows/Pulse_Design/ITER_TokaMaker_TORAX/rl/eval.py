"""Top-level CLI for IQL actor evaluation.

This module stays intentionally thin. The actual simulation runs in
``rl.eval_sim`` and all plotting / movie generation / summary packaging lives in
``rl.eval_postprocess``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from log import get_logger
from rl.eval_postprocess import postprocess_actor_eval
from rl.eval_sim import run_actor_eval_simulation

logger = get_logger(__name__)


def run_actor_eval_from_config(
    actor_checkpoint,
    output_dir=None,
    dataset_dir=None,
    project=None,
    run_name=None,
    wandb_group=None,
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
    render_plots=True,
    render_movie=True,
    render_summary=True,
):
    if output_dir:
        output_dir = Path(output_dir)
    elif dataset_dir:
        run_id = run_name or Path(actor_checkpoint).resolve().stem
        output_dir = Path(dataset_dir).resolve() / "eval" / run_id
    else:
        output_dir = Path(actor_checkpoint).resolve().parent / "actor_eval"
    result = run_actor_eval_simulation(
        actor_checkpoint=actor_checkpoint,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        project=project or os.environ.get("WANDB_PROJECT", "iql-training"),
        run_name=run_name,
        wandb_group=wandb_group,
        wandb_mode=wandb_mode,
        initial_relax_state=initial_relax_state,
        initial_relax_cache_dir=initial_relax_cache_dir,
        max_loop=max_loop,
        grid_size=grid_size,
        device=device,
        replay_cache_dir=replay_cache_dir,
        prefer_replay_cache=prefer_replay_cache,
        allow_cpu_jax_on_gpu=allow_cpu_jax_on_gpu,
        rl_segment_timeout_seconds=rl_segment_timeout_seconds,
        rl_max_action_power_w=rl_max_action_power_w,
    )
    if render_plots or render_movie or render_summary:
        postprocess_actor_eval(
            result=result,
            output_dir=output_dir,
            render_plots=render_plots,
            render_movie=render_movie,
            render_summary=render_summary,
            tmtx=result.get("tmtx"),
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an IQL actor with TokaMaker_TORAX.fly(use_rl_actor=True)."
    )
    parser.add_argument("--actor_checkpoint", required=True, help="Path to the trained IQL checkpoint to evaluate.")
    parser.add_argument("--dataset_dir", default=None, help="Optional dataset root used to rebuild missing normalizers.")
    parser.add_argument("--output_dir", default=None, help="Directory for eval summaries, TORAX logs, plots, and movie artifacts.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "iql-training"), help="W&B project for the eval run.")
    parser.add_argument("--run_name", default=None, help="Optional W&B run name for the eval.")
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP"), help="W&B group to associate the eval with the training run.")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE"), help="W&B mode: online, offline, or disabled.")
    parser.add_argument("--initial_relax_state", default=None, help="Explicit initial-relax cache path. Use to bypass cache resolution.")
    parser.add_argument("--initial_relax_cache_dir", default=None, help="Directory that stores keyed initial-relax cache files.")
    parser.add_argument("--max_loop", type=int, default=1, help="Number of TORAX coupling loops to run in closed-loop eval. Use 1 for the standard fast path; use 2 only when you want the extra convergence check.")
    parser.add_argument("--grid_size", type=int, default=51, help="TORAX radial grid size used in the eval.")
    parser.add_argument("--device", default=None, help="Optional device override for the eval wrapper.")
    parser.add_argument("--replay_cache_dir", default=None, help="Optional compact replay-cache directory used to rebuild normalizers.")
    parser.add_argument("--no_replay_cache", action="store_true", help="Disable use of the replay cache when rebuilding normalizers.")
    parser.add_argument("--allow_cpu_jax_on_gpu", action="store_true", help="Allow CPU-backed JAX even if a GPU is visible.")
    parser.add_argument(
        "--rl_segment_timeout_seconds",
        type=float,
        default=float(os.environ.get("RL_SEGMENT_TIMEOUT_SECONDS", "1800")),
    )
    parser.add_argument(
        "--rl_max_action_power_w",
        type=float,
        default=float(os.environ.get("RL_MAX_ACTION_POWER_W", "150000000")),
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip scalar/profile/LCFS plots in the offline postprocess step.")
    parser.add_argument("--no_movie", action="store_true", help="Skip movie generation in the offline postprocess step.")
    parser.add_argument("--no_summary_artifacts", action="store_true", help="Skip summary re-rendering in the offline postprocess step.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.dataset_dir:
        run_id = args.run_name or Path(args.actor_checkpoint).resolve().stem
        output_dir = Path(args.dataset_dir).resolve() / "eval" / run_id
    else:
        output_dir = Path(args.actor_checkpoint).resolve().parent / "actor_eval"
    run_actor_eval_from_config(
        actor_checkpoint=args.actor_checkpoint,
        output_dir=output_dir,
        dataset_dir=args.dataset_dir,
        project=args.project,
        run_name=args.run_name,
        wandb_group=args.wandb_group,
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
        render_plots=not args.no_plots,
        render_movie=not args.no_movie,
        render_summary=not args.no_summary_artifacts,
    )


if __name__ == "__main__":
    main()
