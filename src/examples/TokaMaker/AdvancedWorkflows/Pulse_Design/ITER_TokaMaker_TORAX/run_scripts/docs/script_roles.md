# Script Roles

Use `submit_collect_trajectories_cpu_array.sh` for normal production runs.

## Entrypoint

- `submit_collect_trajectories_cpu_array.sh`
  - Recommended launcher.
  - Creates a timestamped output directory unless `OUTPUT_BASE_DIR` is set.
  - Optionally submits the cache builder when `USE_INITIAL_RELAX_CACHE=1`.
  - Submits `collect_trajectories_cpu_array.sh`.

## Slurm Workers

- `collect_trajectories_cpu_array.sh`
  - CPU array worker on `john`.
  - Maps each array task to a trajectory chunk.
  - Usually called by the submit helper, not directly.

- `collect_trajectories_cpu.sh`
  - Single CPU Slurm job on `john`.
  - Useful for small diagnostics, not full production.

- `collect_trajectories_gpu.sh`
  - Single GPU Slurm job on `jag-standard`.
  - Useful for GPU diagnostics or comparisons.

## Cache Builders

- `collect_initial_relax_cache_cpu.sh`
  - Builds only `initial_relax_state.json` on `john`.
  - Usually called through the submit helper with `USE_INITIAL_RELAX_CACHE=1`.

- `collect_initial_relax_cache_gpu.sh`
  - Builds only `initial_relax_state.json` on `jag-standard`.
  - Use only when intentionally benchmarking a GPU-built cache.

## Setup

- `../setup-env.sh`
  - Local Python environment setup helper.
