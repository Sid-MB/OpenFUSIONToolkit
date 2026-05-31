# TokaMaker/TORAX RL Pipeline

Pipeline for training an offline RL (IQL) policy to control plasma heating in
a TORAX/TokaMaker coupled simulation.

```
collect trajectories  →  materialize replay cache  →  train IQL  →  evaluate
```

All scripts are run from the project root (`ITER_TokaMaker_TORAX/`).

---

## Scripts and Python Files

| File | Purpose |
|---|---|
| `submit_collect_trajectories_cpu_array.sh` | **Main launcher.** Submits collection + optional dependent jobs |
| `collect_trajectories_cpu_array.sh` | Slurm array worker: runs trajectory chunks on `john` (CPU) |
| `collect_trajectories_cpu.sh` | Single CPU collection job (diagnostics only) |
| `collect_trajectories_gpu.sh` | Single GPU collection job (`jag-standard`, benchmarking only) |
| `collect_initial_relax_cache_cpu.sh` | Builds the shared TORAX initial-relax cache on `john` |
| `collect_initial_relax_cache_gpu.sh` | Builds the shared TORAX initial-relax cache on GPU |
| `grid_search_baseline.sh` | Ranks completed trajectories by return; writes leaderboard |
| `materialize_replay_cache.sh` | Aggregates trajectory shards into a flat IQL replay cache |
| `train_iql.sh` | Trains an IQL actor on the replay cache |
| `eval_iql_actor_cpu.sh` | Evaluates **one** IQL checkpoint in closed-loop on `john` (CPU) |
| `eval_baseline_cpu.sh` | Evaluates the TORAX baseline fallback on `john` (CPU), no checkpoint |
| `eval_iql_actor_cpu_batch.sh` | Evaluates **multiple** checkpoints in parallel on `john` (CPU) |
| `eval_iql_actor.sh` | Evaluates one IQL checkpoint on `jag-standard` (GPU; use CPU scripts instead) |

Python entry points: `collect_trajectories_delta.py`, `IQL.py`,
`materialize_replay_cache.py`, `rl/eval.py`, `rl/eval_batch.py`.

---

## Run the Full Pipeline

```bash
# 1. Collect trajectories + grid-search baseline + replay cache + IQL training
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
OUTPUT_BASE_DIR=./run_$(date +%Y%m%d) \
SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh

# 2. Evaluate the trained actor (after train_iql.sh finishes)
DATASET_DIR=./run_<date> \
ACTOR_CHECKPOINT=./run_<date>/iql/iql_weights.pt \
  sbatch run_scripts/eval_iql_actor_cpu.sh
```

If you already have a complete dataset in `OUTPUT_BASE_DIR` and want to reuse
it instead of recollecting trajectories, set `REUSE_EXISTING_DATASET=1`.
The default is to collect fresh trajectories and print a reminder at startup:

```bash
REUSE_EXISTING_DATASET=1 \
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
OUTPUT_BASE_DIR=./run_20260530 \
SUBMIT_GRID_SEARCH=0 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

---

## Configuration Parameters

These are the main knobs you usually change for a run.

### Collection

| Variable | Default | When to change |
|---|---|---|
| `OUTPUT_BASE_DIR` | auto-timestamped | Set this to choose the dataset/run root explicitly. |
| `N_TRAJECTORIES` | required | Total trajectories to collect. |
| `START_IDX` / `END_IDX` | required | Inclusive/exclusive trajectory range. |
| `CPUS_PER_TASK` | `4` | CPUs per Slurm task for the collection array worker. |
| `MEM_PER_NODE` | `128G` | RAM per Slurm task for the collection array worker. |
| `ARRAY_CONCURRENCY` | `64` | Increase or decrease collection fan-out; lower it for smaller jobs or tighter clusters. |
| `CHUNK_SIZE` | `1` | Increase only when you need fewer array tasks. |
| `SLURM_MAX_ARRAY_SIZE` | `1001` | Cluster cap for the total array size; valid task IDs are `0..1000` on the default setup. |
| `MAX_LOOP` | `2` | Lower for smoke tests, raise for slower but more accurate coupling. |
| `GRID_SIZE` | `51` | Lower for smoke tests, use `51` for production runs. |
| `OBSERVATION_MODE` | `legacy` | Use `prev_action` for normal new datasets; use `plasma_only` only for ablations or when you intentionally want no action history. |
| `SUBMIT_GRID_SEARCH` | `0` | Set to `1` if you want the baseline leaderboard job. |
| `SUBMIT_REPLAY_CACHE` | `0` | Set to `1` if you want the flat replay cache materialized automatically. |
| `SUBMIT_IQL` | `0` | Set to `1` if you want training launched automatically. |
| `USE_INITIAL_RELAX_CACHE` | `0` | Set to `1` when you want the shared initial-relax cache. |
| `REUSE_EXISTING_DATASET` | `0` | Set to `1` when `OUTPUT_BASE_DIR` already contains a complete dataset. |
| `SLURM_NICE` | unset | Lower priority when you want other jobs to run first. |

### Evaluation

| Variable | Default | When to change |
|---|---|---|
| `ACTOR_CHECKPOINT` | required for actor evals | Point at the checkpoint you want to evaluate. |
| `ACTOR_CHECKPOINTS` / `CHECKPOINTS_FILE` | required for batch evals | Choose the checkpoints to fan out over workers. |
| `RUN_ID` | derived from the checkpoint | Set this to control the eval folder name. |
| `DATASET_DIR` | required for evals and training | Point this at the dataset root you want to reuse. |
| `N_WORKERS` | `4` for batch eval | Raise or lower parallel eval fan-out. |
| `MAX_LOOP` | `1` for baseline, `2` for actor evals | Match the collection settings you want to compare against. |
| `GRID_SIZE` | `51` | Match the dataset or run a smaller smoke test. |

Standalone CPU jobs on `john` default to `20` CPUs per task. The collection
array worker still defaults to `4` CPUs per task so collection can scale to the
128-CPU throughput shape when needed. `SLURM_MAX_ARRAY_SIZE=1001` means the
largest allowed array index is `1000`; it limits the size of a single array
submission, not the number of jobs running at once.

---

## Output Locations

### Collection

- `<OUTPUT_BASE_DIR>/run_manifest.json`
- `<OUTPUT_BASE_DIR>/replay_shards/`
- `<OUTPUT_BASE_DIR>/grid_search/` when `SUBMIT_GRID_SEARCH=1`
- `<OUTPUT_BASE_DIR>/replay_cache/` when `SUBMIT_REPLAY_CACHE=1`
- `<OUTPUT_BASE_DIR>/failures/`
- `<OUTPUT_BASE_DIR>/chunks/`
- `<OUTPUT_BASE_DIR>/logs/`
- `<OUTPUT_BASE_DIR>/.gitignore` with `*`

### Training

- `<DATASET_DIR>/iql/`
- `<DATASET_DIR>/logs/`

### Evaluation

- Single-checkpoint eval: `<DATASET_DIR>/eval/<RUN_ID>/`
- Batch eval: `<DATASET_DIR>/eval_batch/<JOB_ID>/`
- Eval logs: `.../logs/` inside the eval output root

---

## Run Just Parts

### Collect only (no IQL)

```bash
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
OUTPUT_BASE_DIR=./my_dataset \
SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=0 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```
Artifacts are written under `./my_dataset/` (trajectory shards, optional `grid_search/`, optional `replay_cache/`, and per-run logs).

### Train IQL only (replay cache already exists)

```bash
DATASET_DIR=./my_dataset \
  sbatch run_scripts/train_iql.sh
```
Artifacts are written under `./my_dataset/iql/` (final weights, checkpoints) and W&B. Logs are written under `./my_dataset/logs/`.

### Evaluate only (checkpoint already exists)

Single checkpoint:
```bash
DATASET_DIR=./my_dataset \
ACTOR_CHECKPOINT=./my_run/iql/iql_weights.pt \
  sbatch run_scripts/eval_iql_actor_cpu.sh
```
By default this writes to `./my_dataset/eval/<RUN_ID>/` and logs land in `./my_dataset/eval/<RUN_ID>/logs/`.

Baseline fallback (compares to using the TORAX fallback action at each decision point):
```bash
DATASET_DIR=./my_dataset \
  sbatch run_scripts/eval_baseline_cpu.sh
```

Multiple checkpoints in parallel (one process per checkpoint, `john` CPU):
```bash
DATASET_DIR=./my_dataset \
ACTOR_CHECKPOINTS="run_a/iql_weights.pt run_b/checkpoint_step_50000.pt" \
  sbatch run_scripts/eval_iql_actor_cpu_batch.sh
```
By default this writes to `./my_dataset/eval_batch/<JOB_ID>/` and logs land in `./my_dataset/eval_batch/<JOB_ID>/logs/`.

Or from a file (`#` lines are comments):
```bash
DATASET_DIR=./my_dataset \
CHECKPOINTS_FILE=my_checkpoints.txt \
  sbatch run_scripts/eval_iql_actor_cpu_batch.sh
```
Artifacts are written under the same batch output root as above.

### Changed the reward metric — what to re-run

Rewards are computed by `compute_reward()` in `collect_trajectories_delta.py`
and **baked into `replay_shards/*.npz` at collection time**. The replay cache
materializer only aggregates those saved values; it does not recompute them.

**You must recollect trajectories** to apply a new reward function. The normal
path is still the compact replay shards (`SAVE_FULL_ZARR=0`); do not enable
full Zarr unless you explicitly need deeper forensic traces or want to
recompute rewards from richer simulator state.

```bash
# 1. Recollect with the new reward logic (compact shards by default)
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
OUTPUT_BASE_DIR=./my_dataset_new_reward \
SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh

# 2. (If replay cache was not auto-submitted) Materialize it:
DATASET_DIR=./my_dataset_new_reward OVERWRITE_REPLAY_CACHE=1 \
  sbatch run_scripts/materialize_replay_cache.sh

# 3. Retrain IQL
DATASET_DIR=./my_dataset_new_reward \
  sbatch run_scripts/train_iql.sh

# 4. Re-evaluate
ACTOR_CHECKPOINT=./my_dataset_new_reward/iql/iql_weights.pt \
  sbatch run_scripts/eval_iql_actor_cpu.sh
```

### Smoke test (5 trajectories, quick end-to-end check)

```bash
N_TRAJECTORIES=5 START_IDX=0 END_IDX=5 \
OUTPUT_BASE_DIR=./smoke_test \
GRID_SIZE=11 MAX_LOOP=1 \
SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

---

## Outputs

### Collection (`submit_collect_trajectories_cpu_array.sh`)

```
<OUTPUT_BASE_DIR>/
  run_manifest.json              # dataset metadata (seed, grid, MAX_LOOP)
  replay_shards/*.npz            # per-trajectory data (states, actions, rewards)
  grid_search/                   # best-observed baseline (SUBMIT_GRID_SEARCH=1)
    grid_search_leaderboard.csv
    best_trajectory.json
  replay_cache/                  # flat IQL training data (SUBMIT_REPLAY_CACHE=1)
    states/actions/rewards/...npy
  failures/failed_run_*.json     # error details for any failed trajectories
  full_trajectories/*.zarr       # rich TORAX traces (SAVE_FULL_ZARR=1 only)
```

### IQL Training (`train_iql.sh`)

```
<DATASET_DIR>/iql/
  iql_weights.pt                 # final trained actor
  checkpoint_step_*.pt           # periodic checkpoints
```

Also logged to wandb (`iql-training` project by default).

### Evaluation (`eval_iql_actor_cpu.sh` / `eval_iql_actor_cpu_batch.sh`)

The single-checkpoint CPU eval is the canonical way to get the full notebook-
style outputs in one pass: it runs TORAX, writes the summary bundle, and then
renders plots/movie from the live `tmtx` object before the process exits.

```
<OUTPUT_DIR>/
  actor_eval_summary.json        # metrics + action history + TORAX metadata
  actor_eval_actions.json        # compact per-decision action trace
  actor_eval_bundle.pkl          # serialized TORAX state snapshot for diagnostics
  artifacts/
    plot_scalars.png
    plot_lcfs_evolution.png
    plot_profile_evolution_*.png
    movie.mp4
  tokamaker_torax_logs/          # TORAX solver output
```

Default single-checkpoint evals write under `<DATASET_DIR>/eval/<RUN_ID>/`.
Default batch evals write under `<DATASET_DIR>/eval_batch/<JOB_ID>/`.

Batch eval also writes `<OUTPUT_ROOT>/batch_eval_summary.json` with status and
elapsed time for every checkpoint.

Results are logged to wandb alongside collection and training.

The persistent JAX cache is automatic by default: the wrappers point JAX at a
project-local cache root and the TORAX runtime namespaces it by build/runtime
fingerprint so different installs do not reuse incompatible executables.

If you need to debug cache portability or force a clean compile path, set
`OFT_DISABLE_JAX_COMPILE_CACHE=1` when invoking the eval wrappers.

---

## More Details

- [Script roles and when to use each](docs/script_roles.md)
- [Collection options and full output tree](docs/collection_options.md)
- [Eval performance: compile-once, cache, CPU, batch](docs/eval_performance.md)
- [Shared relax cache and GPU wrappers](docs/cache_and_gpu.md)
- [Architecture-specific OFT builds](docs/oft_builds.md)
