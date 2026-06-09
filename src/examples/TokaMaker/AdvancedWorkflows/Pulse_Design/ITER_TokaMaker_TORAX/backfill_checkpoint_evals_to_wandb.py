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
        [--delete_local] \\
        [--dry_run]

The script is idempotent: W&B deduplicates points at the same step, so
re-running it for the same run is safe.

With --delete_local each step_<N>/ directory is deleted after its metrics
are successfully logged to W&B. The parent checkpoint_evals/ directory is
also removed if it becomes empty. Use --dry_run first to preview what will
be deleted.
"""

import argparse
import json
import re
import shutil
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
        "--delete_local",
        action="store_true",
        help="Delete each step_<N>/ directory after its metrics are successfully logged to W&B. The parent checkpoint_evals/ directory is also removed if it becomes empty afterwards. No-op under --dry_run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be logged (and deleted with --delete_local) without actually calling wandb or touching the filesystem.",
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

    if not args.dry_run:
        import wandb

    for run_id, entries in sorted(by_run.items()):
        entries.sort(key=lambda x: x[0])
        print(f"\nRun {run_id}: {len(entries)} checkpoint evals")

        if not args.dry_run:
            init_kwargs = dict(
                id=run_id,
                project=args.project,
                resume="allow",
                reinit=True,
            )
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

            step_dir = summary_path.parent
            action = "would log" if args.dry_run else "logging"
            delete_action = f" (would delete {step_dir})" if (args.dry_run and args.delete_local) else ""
            print(f"  step {step}: {action} {len(log_dict)} metrics{delete_action}")
            if args.dry_run:
                for k, v in sorted(log_dict.items()):
                    print(f"    {k} = {v:.4g}")
            else:
                wandb.log(log_dict, step=step)
                if args.delete_local:
                    shutil.rmtree(step_dir)
                    print(f"  step {step}: deleted {step_dir}")
                    # Remove the parent checkpoint_evals/ dir if now empty.
                    ce_dir = step_dir.parent
                    remaining = [p for p in ce_dir.iterdir() if p.name != "checkpoint_manifest.txt"]
                    if not remaining:
                        ce_dir.rmdir()
                        print(f"  removed empty {ce_dir}")

        if not args.dry_run:
            wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
