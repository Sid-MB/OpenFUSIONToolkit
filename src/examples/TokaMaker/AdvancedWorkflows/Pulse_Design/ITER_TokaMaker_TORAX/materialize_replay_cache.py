#!/usr/bin/env python3
"""Build a compact IQL replay cache from replay shards, Zarr, or JSON outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataloader import materialize_replay_cache, replay_cache_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Collected trajectory dataset root")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Output cache directory. Defaults to <dataset_dir>/replay_cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing replay cache.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress output.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help=(
            "Number of parallel Zarr readers. Replay shards are collated serially. "
            "Defaults to REPLAY_CACHE_WORKERS, "
            "then SLURM_CPUS_PER_TASK, then os.cpu_count(). Use 1 for serial."
        ),
    )
    parser.add_argument(
        "--worker_backend",
        choices=("process", "thread"),
        default=None,
        help="Parallel worker backend. Defaults to REPLAY_CACHE_WORKER_BACKEND or process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = replay_cache_path(args.dataset_dir, cache_dir=args.cache_dir)
    start = time.perf_counter()
    manifest = materialize_replay_cache(
        args.dataset_dir,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        max_workers=args.max_workers,
        worker_backend=args.worker_backend,
    )
    elapsed = time.perf_counter() - start
    payload = {
        "cache_dir": str(cache_dir),
        "elapsed_seconds": elapsed,
        "manifest": manifest,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
