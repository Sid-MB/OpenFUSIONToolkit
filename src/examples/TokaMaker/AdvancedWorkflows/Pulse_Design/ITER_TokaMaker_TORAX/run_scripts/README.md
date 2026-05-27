# TokaMaker/TORAX Run Scripts

Quick reference for launching trajectory collection from the
`ITER_TokaMaker_TORAX` example directory.

## Standard Run

Use the CPU array submit helper for production runs:

```bash
START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-399%16 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

This runs trajectories `600..999` on `john` with:

- `1` trajectory worker per Slurm task
- `4` CPUs and `16G` RAM per task
- `MAX_LOOP=2`, `GRID_SIZE=51`
- no shared initial relax cache by default
- at most `16` tasks running at once, for `64` total allocated CPUs
- one shared `run_manifest.json` and `all_actions.npy` for the whole dataset

For future full production runs targeting 64 total CPUs, keep the same
`ARRAY_SPEC=0-399%16` and `CPUS_PER_TASK=4` shape:

```bash
START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-399%16 CPUS_PER_TASK=4 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

Keep `N_WORKERS=1`; scale with Slurm array concurrency instead.

## Useful Overrides

```bash
MAX_LOOP=3                         # extra TORAX/TokaMaker coupling pass
TRAJECTORY_TIMEOUT_SECONDS=14400    # per-trajectory wall-time limit
OUTPUT_BASE_DIR=./my_output_dir     # explicit result directory
USE_INITIAL_RELAX_CACHE=1           # opt into cache build + dependency
```

Example small diagnostic:

```bash
START_IDX=600 END_IDX=601 ARRAY_SPEC=0-0%1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

## Outputs And Logs

The submit helper prints `OUTPUT_BASE_DIR`. Chunk outputs land under:

```text
<OUTPUT_BASE_DIR>/trajectories/trajectory_<run_id>.json
```

The dataset root also contains:

```text
<OUTPUT_BASE_DIR>/run_manifest.json
<OUTPUT_BASE_DIR>/all_actions.npy
<OUTPUT_BASE_DIR>/failures/failed_run_<run_id>.json
<OUTPUT_BASE_DIR>/chunks/chunk_<task>_<start>_<end>/task_status.json
<OUTPUT_BASE_DIR>/chunks/chunk_<task>_<start>_<end>/tokamaker_torax_logs/
```

Array workers validate `run_manifest.json` and `all_actions.npy` before doing
simulation work. If a worker sees a seed, trajectory count, sampler, grid, or
`MAX_LOOP` mismatch, it exits nonzero immediately.

Slurm logs land under:

```text
logs/collect_trajectories_cpu_array.sh-<array_job>_<task>.out
logs/collect_trajectories_cpu_array.sh-<array_job>_<task>.err
```

## More Details

- [Script roles](docs/script_roles.md)
- [Optional shared relax cache and GPU wrappers](docs/cache_and_gpu.md)
- [Architecture-specific OFT builds](docs/oft_builds.md)
