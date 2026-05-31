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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an IQL actor with TokaMaker_TORAX.fly(use_rl_actor=True)."
    )
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "iql-training"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--initial_relax_state", default=None)
    parser.add_argument("--initial_relax_cache_dir", default=None)
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
    )
    parser.add_argument(
        "--rl_max_action_power_w",
        type=float,
        default=float(os.environ.get("RL_MAX_ACTION_POWER_W", "150000000")),
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_movie", action="store_true")
    parser.add_argument("--no_summary_artifacts", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.actor_checkpoint).resolve().parent / "actor_eval"
    result = run_actor_eval_simulation(
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
    )
    if not (args.no_plots and args.no_movie and args.no_summary_artifacts):
        postprocess_actor_eval(
            result=result,
            output_dir=output_dir,
            render_plots=not args.no_plots,
            render_movie=not args.no_movie,
            render_summary=not args.no_summary_artifacts,
            tmtx=result.get("tmtx"),
        )


if __name__ == "__main__":
    main()
