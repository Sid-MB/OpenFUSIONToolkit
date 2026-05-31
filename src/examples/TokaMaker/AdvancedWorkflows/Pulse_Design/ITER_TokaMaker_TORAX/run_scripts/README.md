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
| `eval_iql_actor_cpu_batch.sh` | Evaluates **multiple** checkpoints in parallel on `john` (CPU) |
| `eval_iql_actor.sh` | Evaluates one IQL checkpoint on `jag-standard` (GPU; use CPU scripts instead) |

Python entry points: `collect_trajectories_delta.py`, `IQL.py`,
`materialize_replay_cache.py`, `rl/eval.py`, `rl/eval_batch.py`.

---

## Run the Full Pipeline

```bash
# 1. Collect trajectories + grid-search baseline + replay cache + IQL training
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
CPUS_PER_TASK=4 MEM_PER_NODE=128G ARRAY_CONCURRENCY=32 \
OUTPUT_BASE_DIR=./run_$(date +%Y%m%d) \
OBSERVATION_MODE=prev_action \
DRY_RUN=0 SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh

# 2. Evaluate the trained actor (after train_iql.sh finishes)
ACTOR_CHECKPOINT=./run_<date>/iql/iql_weights.pt \
  sbatch run_scripts/eval_iql_actor_cpu.sh
```

If you already have a complete dataset in `OUTPUT_BASE_DIR` and want to reuse
it instead of recollecting trajectories, set `REUSE_EXISTING_DATASET=1`.
The default is to collect fresh trajectories and print a reminder at startup:

```bash
REUSE_EXISTING_DATASET=1 \
OUTPUT_BASE_DIR=./run_20260530 \
SUBMIT_GRID_SEARCH=0 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

Common parameters to tune:
- `MAX_LOOP=2` — TORAX/TokaMaker coupling passes per trajectory (higher = slower, more accurate)
- `GRID_SIZE=51` — radial grid resolution (11 for smoke tests, 51 for production)
- `ARRAY_CONCURRENCY=32` — max parallel Slurm tasks (× `CPUS_PER_TASK` = total CPUs used)
- `N_TRAJECTORIES` / `START_IDX` / `END_IDX` — trajectory index range to collect
- `SLURM_NICE=10000` — lower the priority of the collection array so other pending jobs on `john` can be scheduled first

Standalone CPU jobs on `john` default to `20` CPUs per task. The array worker
still defaults to `4` CPUs per task so collection can scale to the 128-CPU
throughput shape when needed.

---

## Run Just Parts

### Collect only (no IQL)

```bash
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
ARRAY_CONCURRENCY=32 \
OUTPUT_BASE_DIR=./my_dataset \
DRY_RUN=0 SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=0 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

### Train IQL only (replay cache already exists)

```bash
DATASET_DIR=./my_dataset \
  sbatch run_scripts/train_iql.sh
```

### Evaluate only (checkpoint already exists)

Single checkpoint:
```bash
ACTOR_CHECKPOINT=./my_run/iql/iql_weights.pt \
DATASET_DIR=./my_dataset \
  sbatch run_scripts/eval_iql_actor_cpu.sh
```

Multiple checkpoints in parallel (one process per checkpoint, `john` CPU):
```bash
ACTOR_CHECKPOINTS="run_a/iql_weights.pt run_b/checkpoint_step_50000.pt" \
N_WORKERS=2 \
  sbatch run_scripts/eval_iql_actor_cpu_batch.sh
```

Or from a file (`#` lines are comments):
```bash
CHECKPOINTS_FILE=my_checkpoints.txt N_WORKERS=4 \
  sbatch run_scripts/eval_iql_actor_cpu_batch.sh
```

### Changed the reward metric — what to re-run

Rewards are computed by `compute_reward()` in `collect_trajectories_delta.py`
and **baked into `replay_shards/*.npz` at collection time**. The replay cache
materializer only aggregates those saved values; it does not recompute them.

**You must recollect trajectories** to apply a new reward function.
If you saved full Zarr traces (`SAVE_FULL_ZARR=1`), you can recompute rewards
from those without re-running the simulator — but that requires a custom
extraction step (not yet scripted).

```bash
# 1. Recollect (or recompute from Zarr) with the new reward logic
N_TRAJECTORIES=1000 START_IDX=0 END_IDX=1000 \
ARRAY_CONCURRENCY=32 \
OUTPUT_BASE_DIR=./my_dataset_new_reward \
DRY_RUN=0 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
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
CPUS_PER_TASK=4 MEM_PER_NODE=128G ARRAY_CONCURRENCY=5 \
GRID_SIZE=11 MAX_LOOP=1 \
OUTPUT_BASE_DIR=./smoke_test \
DRY_RUN=0 SUBMIT_GRID_SEARCH=1 SUBMIT_REPLAY_CACHE=1 SUBMIT_IQL=1 \
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

```
<OUTPUT_DIR>/
  eval_results.json              # per-loop metrics (Ip, Q_fusion, beta_N, ...)
  actions_history.json           # RL action sequence chosen by the actor
  tokamaker_torax_logs/          # TORAX solver output
```

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
