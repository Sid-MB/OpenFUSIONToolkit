# Script Roles

For **data collection** use `submit_collect_trajectories_cpu_array.sh`.
For **evaluation** (single checkpoint) use `eval_iql_actor_cpu.sh`.
For **evaluation** (baseline fallback, no checkpoint) use `eval_baseline_cpu.sh`.
For **evaluation** (many checkpoints in parallel) use `eval_iql_actor_cpu_batch.sh`.
For **post-training checkpoint fanout** use `fanout_checkpoint_evals.sh`.

See also `docs/eval_performance.md` for the compile-once / persistent-cache
optimizations that make all eval scripts fast.

## Evaluation Scripts

- `eval_iql_actor_cpu.sh` _(direct use: `sbatch run_scripts/eval_iql_actor_cpu.sh`)_
  - Evaluates **one** IQL actor checkpoint in RL closed-loop mode on `john` (CPU).
  - Forces JAX/TORAX onto CPU; no GPU required.
  - Configures the persistent XLA compilation cache so only the first segment
    pays compilation cost; later segments (and later runs) load from `.jax_cache/`.
  - Required env var: `ACTOR_CHECKPOINT=<path/to/iql_weights.pt>`
  - GPU counterpart: `eval_iql_actor.sh` (requests `jag-standard` partition).

- `eval_baseline_cpu.sh` _(direct use: `sbatch run_scripts/eval_baseline_cpu.sh`)_
  - Evaluates the built-in TORAX baseline fallback on `john` (CPU) with no checkpoint.
  - Uses the same live postprocess path as the trained-actor eval so you get the same plots/movie.
  - Required env var: `DATASET_DIR=<dataset root>`

- `eval_iql_actor_cpu_batch.sh` _(direct use: `sbatch run_scripts/eval_iql_actor_cpu_batch.sh`)_
  - Evaluates **multiple** checkpoints in parallel on `john` (CPU).
  - Wraps `rl/eval_batch.py`; each checkpoint runs in its own worker process.
  - Workers share a single XLA cache directory: first worker compiles, the rest
    load from disk.
  - Requires `ACTOR_CHECKPOINTS="a.pt b.pt ..."` or `CHECKPOINTS_FILE=ckpts.txt`.
  - Thread budget is automatically divided across `N_WORKERS` (default: 4).

- `eval_iql_actor.sh` _(direct use: `sbatch run_scripts/eval_iql_actor.sh`)_
  - Evaluates **one** IQL actor checkpoint on `jag-standard` (GPU).
  - Use when GPU is intentionally needed; for normal eval prefer the CPU scripts.

- `train_iql_low_smoothness.sh` _(direct use: `sbatch run_scripts/train_iql_low_smoothness.sh`)_
  - Ablation wrapper for training with less smoothing bias.
  - Sets `OBSERVATION_MODE=plasma_only`; IQL auto-selects `ACTION_MODE=absolute`
    (the action-rate penalty is inactive in absolute mode).
  - Use this when testing whether the default actor setup is over-regularized.

- `fanout_checkpoint_evals.sh` _(direct use: run after training, or from a
  dependency)_
  - Scans a training output directory for `checkpoint_step_*.pt` files and
    submits one CPU eval job per checkpoint.
  - Use this when you want checkpoint-by-checkpoint closed-loop plots and movies
    without blocking training.

## Trajectory Collection Entrypoint

- `submit_collect_trajectories_cpu_array.sh`
  - Recommended launcher for data collection.
  - Creates a timestamped output directory unless `OUTPUT_BASE_DIR` is set.
  - Optionally submits the cache builder when `USE_INITIAL_RELAX_CACHE=1`.
  - Submits `collect_trajectories_cpu_array.sh`.

## Collection Slurm Workers

- `collect_trajectories_cpu_array.sh`
  - CPU array worker on `john`.
  - Maps each array task to a trajectory chunk.
  - Usually called by `submit_collect_trajectories_cpu_array.sh`, not directly.

- `collect_trajectories_cpu.sh`
  - Single CPU Slurm job on `john`.
  - Useful for small diagnostics, not full production.

- `collect_trajectories_gpu.sh`
  - Single GPU Slurm job on `jag-standard`.
  - Useful for GPU diagnostics or comparisons.

## Cache Builders

- `collect_initial_relax_cache_cpu.sh`
  - Builds only the keyed initial-relax cache (`initial_relax_<key>.json`) on `john`.
  - Resolves the path from `INITIAL_RELAX_CACHE_DIR` (or `INITIAL_RELAX_CACHE` override).
  - Usually called through the submit helper with `USE_INITIAL_RELAX_CACHE=1`.

- `collect_initial_relax_cache_gpu.sh`
  - Builds only the keyed initial-relax cache (`initial_relax_<key>.json`) on `jag-standard`.
  - Use only when intentionally benchmarking a GPU-built cache.

## Setup

- `../setup-env.sh`
  - Local Python environment setup helper.
