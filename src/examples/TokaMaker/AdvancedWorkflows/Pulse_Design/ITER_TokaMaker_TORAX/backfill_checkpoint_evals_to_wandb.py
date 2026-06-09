"""Backfill checkpoint-eval metrics into existing W&B training runs.

For each training run directory under --out_root that has
checkpoint_evals/step_<N>/actor_eval_summary.json files, this script
re-opens the corresponding W&B run (by run ID) and logs the eval metrics
at the correct training step — producing a proper curve instead of isolated
single-point runs.

Usage:
    uv run python backfill_checkpoint_evals_to_wandb.py \\
        --project iql-training \\
        --out_root out/iql \\
        [--delete_wandb_runs] \\
        [--eval_project iql-eval] \\
        [--dry_run]

The script is idempotent: W&B deduplicates points at the same step, so
re-running it for the same run is safe.

With --delete_wandb_runs the isolated per-checkpoint runs in --eval_project
are deleted after their metrics are successfully logged to the training run.
Matching is done by step number and the training run's group. Use --dry_run
first to preview which runs will be deleted.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def find_checkpoint_eval_dirs(out_root: Path):
    """Yield (run_id, step, summary_path) for every checkpoint eval found."""
    for summary_path in sorted(out_root.rglob("actor_eval_summary.json")):
        # Expected layout: <out_root>/.../<run_id>/checkpoint_evals/step_<N>/actor_eval_summary.json
        parts = summary_path.parts
        try:
            ce_idx = next(i for i, p in enumerate(parts) if p == "checkpoint_evals")
        except StopIteration:
            continue
        run_id = parts[ce_idx - 1]
        step_dir = parts[ce_idx + 1]
        m = re.fullmatch(r"step_(\d+)", step_dir)
        if not m:
            continue
        step = int(m.group(1))
        yield run_id, step, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--project",
        default="iql-training",
        help="W&B project that the training runs belong to. This must match the project used during training. Default: 'iql-training'.",
    )
    parser.add_argument(
        "--out_root",
        default="out/iql",
        help="Root directory to search for training run subdirectories. The script recurses into this tree looking for checkpoint_evals/step_*/actor_eval_summary.json files. Default: 'out/iql'.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (team or username). Leave unset to use the default entity associated with your W&B API key.",
    )
    parser.add_argument(
        "--delete_wandb_runs",
        action="store_true",
        help="After logging each step's metrics to the training run, delete the corresponding isolated run from --eval_project. Runs are matched by step number (name=checkpoint_step_<N>) and the training run's group. No-op under --dry_run.",
    )
    parser.add_argument(
        "--eval_project",
        default="iql-eval",
        help="W&B project containing the isolated per-checkpoint eval runs to delete. Only used with --delete_wandb_runs. Default: 'iql-eval'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be logged and deleted without actually calling wandb.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not out_root.exists():
        print(f"ERROR: --out_root {out_root} does not exist", file=sys.stderr)
        sys.exit(1)

    # Group by run_id so we open each W&B run only once.
    from collections import defaultdict
    by_run: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for run_id, step, summary_path in find_checkpoint_eval_dirs(out_root):
        by_run[run_id].append((step, summary_path))

    if not by_run:
        print("No checkpoint eval summaries found under", out_root)
        return

    print(f"Found {sum(len(v) for v in by_run.values())} checkpoint evals across {len(by_run)} run(s).")

    import wandb
    api = wandb.Api()

    # Pre-fetch eval runs to delete: build a lookup (training_run_id, step) -> Run.
    # Match by extracting the training run ID from the eval run's actor_checkpoint config field.
    eval_runs_by_train_id_step: dict[tuple[str, int], object] = {}
    if args.delete_wandb_runs:
        entity_prefix = f"{args.entity}/" if args.entity else ""
        print(f"Fetching runs from {entity_prefix}{args.eval_project} ...")
        eval_runs = api.runs(f"{entity_prefix}{args.eval_project}")
        for er in eval_runs:
            m_step = re.fullmatch(r"checkpoint_step_(\d+)", er.name)
            if not m_step:
                continue
            step = int(m_step.group(1))
            ckpt_path = er.config.get("actor_checkpoint", "")
            # Extract the training run ID from the checkpoint path.
            # Path pattern: .../out/iql/<reward_hash>/<run_id>/checkpoints/checkpoint_step_<N>.pt
            m_rid = re.search(r"/checkpoints/checkpoint_step_\d+\.pt$", ckpt_path)
            if m_rid:
                train_run_id = Path(ckpt_path).parent.parent.name
                eval_runs_by_train_id_step[(train_run_id, step)] = er
        print(f"  Found {len(eval_runs_by_train_id_step)} matching checkpoint_step_* runs to consider for deletion.")

    for run_id, entries in sorted(by_run.items()):
        entries.sort(key=lambda x: x[0])

        print(f"\nRun {run_id}: {len(entries)} checkpoint evals")

        if not args.dry_run:
            init_kwargs = dict(id=run_id, project=args.project, resume="allow", reinit=True)
            if args.entity:
                init_kwargs["entity"] = args.entity
            run = wandb.init(**init_kwargs)

        for step, summary_path in entries:
            try:
                summary = json.loads(summary_path.read_text())
            except Exception as exc:
                print(f"  step {step}: failed to read {summary_path}: {exc}", file=sys.stderr)
                continue

            metrics = summary.get("metrics", {})
            if not metrics:
                print(f"  step {step}: no metrics found, skipping")
                continue

            # Flatten to checkpoint_eval/<short_key> matching what the new online
            # code logs (strip the leading "actor_eval/" prefix).
            log_dict = {}
            for k, v in metrics.items():
                if not isinstance(v, (int, float)):
                    continue
                short = k.split("/")[-1]
                log_dict[f"checkpoint_eval/{short}"] = v

            # Find matching eval run for deletion.
            eval_run_to_delete = eval_runs_by_train_id_step.get((run_id, step))

            delete_note = ""
            if args.delete_wandb_runs:
                if eval_run_to_delete:
                    delete_note = f" (would delete {args.eval_project}/{eval_run_to_delete.id})" if args.dry_run else ""
                else:
                    delete_note = " (no matching eval run found)" if args.dry_run else ""

            print(f"  step {step}: {'would log' if args.dry_run else 'logging'} {len(log_dict)} metrics{delete_note}")

            if not args.dry_run:
                wandb.log(log_dict, step=step)
                if args.delete_wandb_runs and eval_run_to_delete:
                    eval_run_to_delete.delete()
                    print(f"  step {step}: deleted {args.eval_project}/{eval_run_to_delete.id} ({eval_run_to_delete.name})")

        if not args.dry_run:
            wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
