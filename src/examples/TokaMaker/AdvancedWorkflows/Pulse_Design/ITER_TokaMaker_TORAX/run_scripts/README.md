# TokaMaker/TORAX Run Scripts

Quick reference for launching trajectory collection from the
`ITER_TokaMaker_TORAX` example directory.

## Standard Run

Use the CPU array submit helper for production runs:

```bash
START_IDX=600 END_IDX=1000 ARRAY_CONCURRENCY=16 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

This runs trajectories `600..999` on `john` with:

- `1` trajectory worker per Slurm task
- `4` CPUs and `16G` RAM per task
- `MAX_LOOP=2`, `GRID_SIZE=51`
- no shared initial relax cache by default
- at most `16` tasks running at once, for `64` total allocated CPUs
- one shared `run_manifest.json` and `all_actions.npy` for the whole dataset
- a dependent `john` job that builds `<OUTPUT_BASE_DIR>/replay_cache/` for IQL

For future full production runs targeting 64 total CPUs, keep the same
`ARRAY_CONCURRENCY=16` and `CPUS_PER_TASK=4` shape. The submit helper derives
the array range from `START_IDX`, `END_IDX`, and `CHUNK_SIZE`.

```bash
START_IDX=600 END_IDX=1000 ARRAY_CONCURRENCY=16 CPUS_PER_TASK=4 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

Keep `N_WORKERS=1`; scale with Slurm array concurrency instead.
The cluster reports `MaxArraySize=1001`, so valid array task IDs are `0..1000`.
If a run would need more than 1001 array tasks, increase `CHUNK_SIZE` or split
the range into multiple submissions.

## Useful Overrides

```bash
MAX_LOOP=3                         # extra TORAX/TokaMaker coupling pass
TRAJECTORY_TIMEOUT_SECONDS=14400    # per-trajectory wall-time limit
OUTPUT_BASE_DIR=./my_output_dir     # explicit result directory
USE_INITIAL_RELAX_CACHE=1           # opt into cache build + dependency
SUBMIT_REPLAY_CACHE=0               # skip compact IQL replay-cache build
REPLAY_CACHE_MEM=120G               # RAM for replay-cache materialization
REPLAY_CACHE_CPUS=8                 # parallel Zarr readers for replay cache
REPLAY_CACHE_WORKERS=8              # override reader count independently
REPLAY_CACHE_WORKER_BACKEND=process # process or thread workers
REPLAY_CACHE_PROGRESS=0             # disable tqdm progress in replay-cache logs
SUBMIT_IQL=1                        # train IQL after replay cache is ready
DRY_RUN=1                           # print derived array shape without submitting
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
<OUTPUT_BASE_DIR>/replay_cache/{states,actions,next_states,rewards,dones}.npy
<OUTPUT_BASE_DIR>/replay_cache/replay_manifest.json
```

Array workers validate `run_manifest.json` and `all_actions.npy` before doing
simulation work. If a worker sees a seed, trajectory count, sampler, grid, or
`MAX_LOOP` mismatch, it exits nonzero immediately.

Slurm logs land under:

```text
logs/collect_trajectories_cpu_array.sh-<array_job>_<task>.out
logs/collect_trajectories_cpu_array.sh-<array_job>_<task>.err
logs/materialize_replay_cache.sh-<job>.out
logs/train_iql.sh-<job>.out
```

## Replay Cache For IQL

IQL prefers a compact replay cache when `<dataset>/replay_cache/` exists. Build
or rebuild it manually with:

```bash
uv run python materialize_replay_cache.py \
  rl_dataset_delta_sampling_maxloop=2_grid_51_full_zarr_2000_20260527_123853 \
  --overwrite \
  --max_workers 8 \
  --worker_backend process
```

For Slurm, use:

```bash
DATASET_DIR=rl_dataset_delta_sampling_maxloop=2_grid_51_full_zarr_2000_20260527_123853 \
  OVERWRITE_REPLAY_CACHE=1 \
  sbatch run_scripts/materialize_replay_cache.sh
```

The cache is a derived training artifact. Keep the per-trajectory Zarr outputs
as the rich archival/debug format; the replay cache is only the reduced
transition table used by offline RL.

## More Details

- [Script roles](docs/script_roles.md)
- [Optional shared relax cache and GPU wrappers](docs/cache_and_gpu.md)
- [Architecture-specific OFT builds](docs/oft_builds.md)
