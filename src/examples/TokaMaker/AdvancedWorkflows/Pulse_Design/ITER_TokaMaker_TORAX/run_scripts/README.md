# TokaMaker/TORAX Run Scripts

Quick reference for launching trajectory collection from the
`ITER_TokaMaker_TORAX` example directory.

## Standard Run

Use the CPU array submit helper for production runs:

```bash
N_TRAJECTORIES=1000 START_IDX=600 END_IDX=1000 \
N_WORKERS=1 CHUNK_SIZE=1 ARRAY_CONCURRENCY=16 SLURM_MAX_ARRAY_SIZE=1001 \
CPUS_PER_TASK=4 MEM_PER_NODE=16G \
USE_INITIAL_RELAX_CACHE=0 OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_run \
DRY_RUN=0 SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=0 \
GRID_SEARCH_CPUS=1 GRID_SEARCH_MEM=8G \
REPLAY_CACHE_CPUS=8 REPLAY_CACHE_MEM=120G \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

Shell wrappers intentionally do not assign run defaults. Collection defaults
live in `collect_trajectories_delta.py` argparse; set an env var only when the
wrapper needs it for Slurm shape/dependencies or when overriding argparse.

The command above runs trajectories `600..999` on `john` with:

- `1` trajectory worker per Slurm task
- `4` CPUs and `16G` RAM per task
- argparse defaults for omitted collection knobs like `MAX_LOOP`, `GRID_SIZE`,
  seed, timeout, and save formats
- no shared initial relax cache because `USE_INITIAL_RELAX_CACHE=0` is explicit
- at most `16` tasks running at once, for `64` total allocated CPUs
- one shared `run_manifest.json` and `all_actions.npy` for the whole dataset
- compact per-trajectory `replay_shards/trajectory_<run_id>.npz` outputs from
  the argparse save-format default
- a dependent `john` job that writes the best-observed grid-search baseline
  under `<OUTPUT_BASE_DIR>/grid_search/`
- a dependent `john` job that builds `<OUTPUT_BASE_DIR>/replay_cache/` for IQL

For future full production runs targeting 64 total CPUs, keep the same
`ARRAY_CONCURRENCY=16` and `CPUS_PER_TASK=4` shape. The submit helper derives
the array range from `START_IDX`, `END_IDX`, and `CHUNK_SIZE`.

Keep `N_WORKERS=1`; scale with Slurm array concurrency instead.
The cluster reports `MaxArraySize=1001`, so valid array task IDs are `0..1000`.
If a run would need more than 1001 array tasks, increase `CHUNK_SIZE` or split
the range into multiple submissions.

## Useful Overrides

```bash
MAX_LOOP=3                         # extra TORAX/TokaMaker coupling pass
TRAJECTORY_TIMEOUT_SECONDS=14400    # per-trajectory wall-time limit
USE_INITIAL_RELAX_CACHE=1           # opt into cache build + dependency
SUBMIT_GRID_SEARCH=0                # skip best-observed baseline ranking
GRID_SEARCH_MEM=8G                  # RAM for grid-search baseline job
GRID_SEARCH_CPUS=1                  # CPUs for grid-search baseline job
GRID_SEARCH_OUTPUT_DIR=./my_grid    # override baseline output directory
GRID_SEARCH_TOP_K=20                # number of top trajectories in summary
RUN_LOG_DIR=logs/my_dataset_run     # override grouped Slurm log directory
SUBMIT_REPLAY_CACHE=0               # skip compact IQL replay-cache build
SAVE_REPLAY_SHARD=1                 # write compact per-trajectory .npz shards
SAVE_FULL_ZARR=1                    # also write rich full TORAX Zarr traces
SAVE_JSON=1                         # also write legacy compact JSON files
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
N_TRAJECTORIES=5 START_IDX=0 END_IDX=5 \
N_WORKERS=1 CHUNK_SIZE=1 ARRAY_CONCURRENCY=5 SLURM_MAX_ARRAY_SIZE=1001 \
CPUS_PER_TASK=4 MEM_PER_NODE=16G \
USE_INITIAL_RELAX_CACHE=0 OUTPUT_BASE_DIR=./rl_dataset_smoke_5 \
DRY_RUN=0 SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
GRID_SEARCH_CPUS=1 GRID_SEARCH_MEM=8G \
REPLAY_CACHE_CPUS=4 REPLAY_CACHE_MEM=16G \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

## Outputs And Logs

The submit helper prints `OUTPUT_BASE_DIR`. Chunk outputs land under:

```text
<OUTPUT_BASE_DIR>/replay_shards/trajectory_<run_id>.npz
```

The dataset root also contains:

```text
<OUTPUT_BASE_DIR>/run_manifest.json
<OUTPUT_BASE_DIR>/all_actions.npy
<OUTPUT_BASE_DIR>/replay_shards/trajectory_<run_id>.npz
<OUTPUT_BASE_DIR>/full_trajectories/trajectory_<run_id>.zarr   # only with SAVE_FULL_ZARR=1
<OUTPUT_BASE_DIR>/trajectories/trajectory_<run_id>.json        # only with SAVE_JSON=1
<OUTPUT_BASE_DIR>/failures/failed_run_<run_id>.json
<OUTPUT_BASE_DIR>/chunks/chunk_<task>_<start>_<end>/task_status.json
<OUTPUT_BASE_DIR>/chunks/chunk_<task>_<start>_<end>/tokamaker_torax_logs/
<OUTPUT_BASE_DIR>/grid_search/grid_search_leaderboard.csv
<OUTPUT_BASE_DIR>/grid_search/best_trajectory.json
<OUTPUT_BASE_DIR>/grid_search/grid_search_summary.json
<OUTPUT_BASE_DIR>/replay_cache/{states,actions,next_states,rewards,dones}.npy
<OUTPUT_BASE_DIR>/replay_cache/replay_manifest.json
```

Array workers validate `run_manifest.json` and `all_actions.npy` before doing
simulation work. If a worker sees a seed, trajectory count, sampler, grid, or
`MAX_LOOP` mismatch, it exits nonzero immediately.

Slurm logs for a submit-helper run land together under `RUN_LOG_DIR`, which
defaults to `logs/<basename-of-OUTPUT_BASE_DIR>/`:

```text
logs/<run>/collect_trajectories-<array_job>_<task>.out
logs/<run>/collect_trajectories-<array_job>_<task>.err
logs/<run>/grid_search_baseline-<job>.out
logs/<run>/materialize_replay_cache-<job>.out
logs/<run>/train_iql-<job>.out
```

## Grid Search Baseline

The submit helper can launch a dependent `john` job after collection with
`SUBMIT_GRID_SEARCH=1`. The baseline ranks completed trajectories by
`return_sum` and reads the current compact replay shards, plus legacy JSON or
full Zarr datasets.

Run it manually with:

```bash
DATASET_DIR=rl_dataset_delta_sampling_run \
  sbatch run_scripts/grid_search_baseline.sh
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

The cache is a derived training artifact. The preferred source is the compact
per-trajectory replay shards, which avoids reopening full Zarr stores for IQL.
Use `SAVE_FULL_ZARR=1` on selected runs when you also need the rich archival or
debug traces.

## More Details

- [Script roles](docs/script_roles.md)
- [Optional shared relax cache and GPU wrappers](docs/cache_and_gpu.md)
- [Architecture-specific OFT builds](docs/oft_builds.md)
