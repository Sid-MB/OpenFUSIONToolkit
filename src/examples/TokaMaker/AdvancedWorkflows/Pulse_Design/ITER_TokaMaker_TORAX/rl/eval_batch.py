"""Batch IQL actor evaluation: run several checkpoints in parallel.

Each individual evaluation is inherently serial: the RL decision chain is a
sequential data dependency (observation at t requires the [0→t] cold-start run to
finish), and TokaMaker's equilibria solves are coupled in time via eddy-current
continuity. Throughput therefore comes from running multiple checkpoints or seeds
concurrently in separate processes.

Architecture:
  - One process per eval (maxtasksperchild=1 for clean OFT/JAX isolation).
  - Each worker calls run_actor_eval_from_config() for one job, which builds its
    own TokaMaker object and TokaMaker_TORAX, runs fly(), and writes results.
  - Workers share a single JAX XLA compilation cache (.jax_cache/) so only the
    first worker to run pays compilation cost; the rest load from disk.
  - Progress and per-eval metrics are streamed to wandb (one run per eval).
  - A batch_eval_summary.json is written to OUTPUT_ROOT on completion.

Multiprocessing pattern is the same as collect_trajectories_delta.py (Pool +
imap_unordered + os._exit). fork is the default MP context; spawn is available via
MP_CONTEXT=spawn if needed (e.g. when mixing GPU and CPU backends).

Intended for the CPU (john) partition. See run_scripts/eval_iql_actor_cpu_batch.sh
for the Slurm wrapper with thread-budget splitting across workers.

Main entry points:
  main()   CLI — python -m rl.eval_batch --actor_checkpoint ... [--actor_checkpoint ...]
           or    python -m rl.eval_batch --checkpoints_file checkpoints.txt
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path

from rl.eval import run_actor_eval_from_config


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _run_one(job, *, dataset_dir, project, max_loop, grid_size,
             initial_relax_cache_dir, replay_cache_dir, prefer_replay_cache,
             allow_cpu_jax_on_gpu, allow_mismatched_rewards, rl_segment_timeout_seconds, rl_max_action_power_w,
             wandb_group, render_plots, render_movie, render_summary_artifacts):
    """Run one eval in this worker process and return a picklable status dict.

    Called by Pool.imap_unordered; all shared config is bound via functools.partial
    so only the per-job dict varies across pool tasks. Exceptions are caught and
    returned as failed status (rather than propagating) so one bad checkpoint does
    not abort the whole batch.
    """
    t0 = time.time()
    checkpoint = job["actor_checkpoint"]
    output_dir = job["output_dir"]
    run_name = job["run_name"]
    try:
        result = run_actor_eval_from_config(
            actor_checkpoint=checkpoint,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            project=project,
            run_name=run_name,
            initial_relax_cache_dir=initial_relax_cache_dir,
            max_loop=max_loop,
            grid_size=grid_size,
            replay_cache_dir=replay_cache_dir,
            prefer_replay_cache=prefer_replay_cache,
            allow_cpu_jax_on_gpu=allow_cpu_jax_on_gpu,
            allow_mismatched_rewards=allow_mismatched_rewards,
            rl_segment_timeout_seconds=rl_segment_timeout_seconds,
            rl_max_action_power_w=rl_max_action_power_w,
            wandb_group=wandb_group,
            render_plots=render_plots,
            render_movie=render_movie,
            render_summary=render_summary_artifacts,
        )
        return {
            "checkpoint": str(checkpoint),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "status": "ok",
            "elapsed_s": time.time() - t0,
            "metrics": result.get("metrics", {}),
        }
    except Exception as exc:  # keep the batch going if one eval fails
        return {
            "checkpoint": str(checkpoint),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "status": "failed",
            "elapsed_s": time.time() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _resolve_checkpoints(args):
    """Merge --actor_checkpoint repeats and --checkpoints_file into a deduplicated list."""
    checkpoints = list(args.actor_checkpoint or [])
    if args.checkpoints_file:
        with open(args.checkpoints_file) as f:
            checkpoints.extend(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )
    if not checkpoints:
        raise SystemExit("No checkpoints given. Use --actor_checkpoint or --checkpoints_file.")
    # Preserve order, drop duplicates (resolved to absolute paths for comparison).
    seen, unique = set(), []
    for c in checkpoints:
        cr = str(Path(c).resolve())
        if cr not in seen:
            seen.add(cr)
            unique.append(cr)
    return unique


def _build_jobs(checkpoints, output_root):
    """Assign a unique run_name and output_dir to each checkpoint path."""
    jobs = []
    used_names = {}
    for ckpt in checkpoints:
        stem = Path(ckpt).stem
        # Disambiguate checkpoints that share a filename stem (e.g. iql_weights.pt
        # from two different training runs) by appending a counter suffix.
        n = used_names.get(stem, 0)
        used_names[stem] = n + 1
        run_name = stem if n == 0 else f"{stem}_{n}"
        jobs.append({
            "actor_checkpoint": ckpt,
            "run_name": run_name,
            "output_dir": str(Path(output_root) / run_name),
        })
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run several IQL actor evaluations in parallel (one process per eval)."
    )
    parser.add_argument("--actor_checkpoint", action="append", help="Checkpoint path; repeatable. Combine with --checkpoints_file.")
    parser.add_argument("--checkpoints_file", help="File with one checkpoint path per line (# comments allowed).")
    parser.add_argument("--output_root", default=None, help="Directory under which each eval writes <run_name>/. Defaults to out/iql_eval_batch/<timestamp>.")
    parser.add_argument("--dataset_dir", default=None, help="Dataset root used to rebuild missing normalizers for eval.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "iql-training"), help="W&B project for each eval run.")
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP"), help="W&B group for all eval runs in the batch.")
    parser.add_argument("--max_loop", type=int, default=1, help="Number of TORAX coupling loops used for each eval. Use 1 for routine evaluation; use 2 only when you want the extra convergence check.")
    parser.add_argument("--grid_size", type=int, default=51, help="TORAX radial grid size used for each eval.")
    parser.add_argument("--n_workers", type=int,
                        default=int(os.environ.get("N_WORKERS", "1")))
    parser.add_argument("--initial_relax_cache_dir", default=None, help="Directory containing keyed initial-relax caches.")
    parser.add_argument("--replay_cache_dir", default=None, help="Optional compact replay-cache directory for normalizer reconstruction.")
    parser.add_argument("--no_replay_cache", action="store_true", help="Disable replay-cache use when reconstructing normalizers.")
    parser.add_argument("--allow_cpu_jax_on_gpu", action="store_true", help="Allow CPU-backed JAX even if a GPU is visible.")
    parser.add_argument(
        "--allow_mismatched_rewards",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ALLOW_MISMATCHED_REWARDS", False),
        help=(
            "Allow a batch eval to proceed even when a checkpoint's recorded training reward config differs from the current eval runtime reward config. "
            "Leave this off for normal runs so reward drift fails fast; turn it on only for deliberate legacy comparisons or reward-change ablations."
        ),
    )
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
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation in the postprocess step.")
    parser.add_argument("--no_movie", action="store_true", help="Skip movie generation in the postprocess step.")
    parser.add_argument("--no_summary_artifacts", action="store_true", help="Skip summary re-rendering in the postprocess step.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = _resolve_checkpoints(args)

    output_root = args.output_root or os.path.join(
        "out", "iql_eval_batch", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(checkpoints, output_root)

    n_workers = max(1, args.n_workers)
    print(f"Batch eval: {len(jobs)} checkpoint(s), n_workers={n_workers}")
    print(f"Output root: {os.path.abspath(output_root)}")
    for job in jobs:
        print(f"  - {job['run_name']}: {job['actor_checkpoint']}")

    worker = partial(
        _run_one,
        dataset_dir=args.dataset_dir,
        project=args.project,
        max_loop=args.max_loop,
        grid_size=args.grid_size,
        initial_relax_cache_dir=args.initial_relax_cache_dir,
        replay_cache_dir=args.replay_cache_dir,
        prefer_replay_cache=not args.no_replay_cache,
        allow_cpu_jax_on_gpu=args.allow_cpu_jax_on_gpu,
        allow_mismatched_rewards=args.allow_mismatched_rewards,
        rl_segment_timeout_seconds=args.rl_segment_timeout_seconds,
        rl_max_action_power_w=args.rl_max_action_power_w,
        wandb_group=args.wandb_group,
        render_plots=not args.no_plots,
        render_movie=not args.no_movie,
        render_summary_artifacts=not args.no_summary_artifacts,
    )

    t_start = time.time()
    results = []
    def _report(res):
        status_line = (
            f"  [{res['status']}] {res['run_name']} ({res['elapsed_s']/60:.1f} min)"
        )
        if res["status"] == "failed":
            status_line += f" — {res['error']}"
            print(status_line)
            # Print the full traceback so failures are debuggable without digging
            # into batch_eval_summary.json.
            print(res.get("traceback", "  (no traceback)"))
        else:
            print(status_line)

    if n_workers == 1:
        # Run serially in the main process — useful for debugging and timing without
        # multiprocessing overhead.
        for job in jobs:
            res = worker(job)
            results.append(res)
            _report(res)
    else:
        mp_context = os.environ.get("MP_CONTEXT", "fork")
        ctx = mp.get_context(mp_context)
        # maxtasksperchild=1: each worker process runs exactly one eval then exits.
        # This ensures each worker gets a fresh Python interpreter with no leftover
        # OFT/JAX state from a previous eval (same pattern as collect_trajectories_delta.py).
        with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(worker, jobs):
                results.append(res)
                done = len(results)
                elapsed = time.time() - t_start
                # Simple ETA: assume remaining jobs take the average of completed ones.
                eta = (elapsed / done) * (len(jobs) - done) / 60 if done else 0.0
                _report(res)
                print(f"    {done}/{len(jobs)} done | ETA {eta:.1f} min")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_failed = len(results) - n_ok
    summary = {
        "output_root": os.path.abspath(output_root),
        "n_checkpoints": len(jobs),
        "n_workers": n_workers,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "elapsed_s": time.time() - t_start,
        "results": results,
    }
    summary_path = Path(output_root) / "batch_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Batch done: {n_ok} ok, {n_failed} failed in "
          f"{summary['elapsed_s']/60:.1f} min. Summary: {summary_path}")

    # Hard-exit skips Python's atexit handlers (JAX/XLA cleanup, resource_tracker).
    # Those handlers can emit spurious semaphore-leak warnings when fork-based
    # workers are used, which would obscure a clean exit status. This pattern
    # matches collect_trajectories_delta.py.
    os._exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
