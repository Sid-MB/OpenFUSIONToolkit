#!/usr/bin/env python3
"""Log a completed dataset collection summary to a separate W&B project.

This is a lightweight telemetry helper, not a per-trajectory logger. It reads
the completed dataset root, extracts the manifest and replay-cache summary, and
writes a single run with collection metadata so you can track dataset builds in
wandb.com without mixing them into the training/eval project.

Typical usage:
    python run_scripts/log_collection_wandb.py \
      --dataset_dir ./run_prev_action_full_YYYYMMDD_HHMMSS \
      --project iql-collection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help=(
            "Completed dataset root to summarize. Use the dataset directory that "
            "contains run_manifest.json and replay_shards/ after collection has finished."
        ),
    )
    parser.add_argument(
        "--project",
        default="iql-collection",
        help=(
            "Separate W&B project for collection telemetry. Use this to keep dataset "
            "builds out of the training/eval project."
        ),
    )
    parser.add_argument(
        "--group",
        default=None,
        help=(
            "Optional W&B group name. Set this when you want multiple collection runs "
            "grouped under one dataset family or experiment label."
        ),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help=(
            "Optional W&B run name. Defaults to the dataset directory name. Set this "
            "when you want a human-readable label in the collection project."
        ),
    )
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "W&B mode: online, offline, or disabled. Leave unset to use the environment "
            "default; set offline when you want local logging without uploading."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_dir.resolve()
    manifest = load_json(root / "run_manifest.json")
    replay_manifest_path = root / "replay_cache" / "replay_manifest.json"
    replay_manifest = load_json(replay_manifest_path) if replay_manifest_path.is_file() else {}

    run_name = args.run_name or root.name
    summary = {
        "dataset_dir": str(root),
        "output_root": str(root),
        "observation_mode": manifest.get("observation_mode", "legacy"),
        "n_trajectories": manifest.get("n_trajectories"),
        "requested_start_idx": manifest.get("requested_range", {}).get("start_idx"),
        "requested_end_idx": manifest.get("requested_range", {}).get("end_idx"),
        "grid_size": manifest.get("grid_size"),
        "max_loop": manifest.get("max_loop"),
        "seed": manifest.get("seed"),
        "seed_eqdsk_count": manifest.get("seed_eqdsk_count"),
        "trajectory_timeout_seconds": manifest.get("trajectory_timeout_seconds"),
        "save_replay_shard": manifest.get("save_replay_shard"),
        "save_full_zarr": manifest.get("save_full_zarr"),
        "save_json": manifest.get("save_json"),
        "has_replay_cache": replay_manifest != {},
        "replay_cache_dir": replay_manifest.get("cache_dir"),
        "replay_cache_examples": replay_manifest.get("num_examples"),
        "replay_cache_states_shape": replay_manifest.get("states_shape"),
        "replay_cache_actions_shape": replay_manifest.get("actions_shape"),
        "replay_cache_rewards_shape": replay_manifest.get("rewards_shape"),
        "replay_cache_dones_shape": replay_manifest.get("dones_shape"),
    }

    wandb_kwargs = {
        "project": args.project,
        "job_type": "collection",
        "name": run_name,
        "config": summary,
    }
    if args.group:
        wandb_kwargs["group"] = args.group
    if args.mode:
        wandb_kwargs["mode"] = args.mode

    run = wandb.init(**wandb_kwargs)
    try:
        run.summary.update(summary)
        run.log({"collection/summary": json.dumps(summary, sort_keys=True)})
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
