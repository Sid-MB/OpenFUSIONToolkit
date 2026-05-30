# Trajectory Collection: Options and Outputs

All options are passed as environment variables to the submit helper
`submit_collect_trajectories_cpu_array.sh`, which forwards them as needed to
Slurm and to `collect_trajectories_delta.py` (argparse).

## Key Environment Variables

### Trajectory Range and Scale
| Variable | Default | Description |
|---|---|---|
| `START_IDX` | — | First trajectory index (inclusive) |
| `END_IDX` | — | Last trajectory index (exclusive) |
| `N_TRAJECTORIES` | — | Total expected trajectories (used for seed-file generation) |
| `CHUNK_SIZE` | `1` | Trajectories per array task (increase if array size > 1001) |
| `ARRAY_CONCURRENCY` | — | Max simultaneously running array tasks |
| `SLURM_MAX_ARRAY_SIZE` | `1001` | Cluster limit; `sbatch --array` IDs must be < this |

### Resources (per array task)
| Variable | Default | Description |
|---|---|---|
| `CPUS_PER_TASK` | `4` | CPUs per Slurm task |
| `MEM_PER_NODE` | `128G` | RAM per Slurm task |
| `N_WORKERS` | `1` | Trajectory workers per task (keep at 1; scale via array concurrency) |

### Simulation
| Variable | Default (argparse) | Description |
|---|---|---|
| `MAX_LOOP` | `2` | TORAX/TokaMaker coupling loops per trajectory |
| `GRID_SIZE` | `51` | TORAX radial grid points |
| `TRAJECTORY_TIMEOUT_SECONDS` | (argparse) | Per-trajectory wall-time limit |

### Cache and Relax
| Variable | Default | Description |
|---|---|---|
| `USE_INITIAL_RELAX_CACHE` | `0` | `1` to build+use a shared initial-relax cache |
| `INITIAL_RELAX_CACHE_DIR` | `./initial_relax_cache` | Shared cache directory |

### Save Formats
| Variable | Default | Description |
|---|---|---|
| `SAVE_REPLAY_SHARD` | `1` (argparse) | Write compact `replay_shards/*.npz` (preferred) |
| `SAVE_FULL_ZARR` | `0` | Also write full rich TORAX Zarr traces |
| `SAVE_JSON` | `0` | Also write legacy compact JSON files |

### Dependent Jobs
| Variable | Default | Description |
|---|---|---|
| `SUBMIT_GRID_SEARCH` | `0` | Submit grid-search baseline after collection |
| `GRID_SEARCH_CPUS` | `1` | CPUs for the grid-search job |
| `GRID_SEARCH_MEM` | `128G` | RAM for the grid-search job |
| `GRID_SEARCH_TOP_K` | (argparse) | Top-K trajectories in the leaderboard |
| `GRID_SEARCH_OUTPUT_DIR` | `<dataset>/grid_search` | Override baseline output dir |
| `SUBMIT_REPLAY_CACHE` | `0` | Submit replay-cache materializer after collection |
| `REPLAY_CACHE_CPUS` | `8` | CPUs for the replay-cache job |
| `REPLAY_CACHE_MEM` | `128G` | RAM for the replay-cache job |
| `REPLAY_CACHE_WORKERS` | (= CPUS) | Parallel Zarr readers |
| `REPLAY_CACHE_WORKER_BACKEND` | `process` | `process` or `thread` |
| `SUBMIT_IQL` | `0` | Submit IQL training after replay cache is ready |

### Misc
| Variable | Default | Description |
|---|---|---|
| `OUTPUT_BASE_DIR` | auto-timestamped | Root directory for all outputs |
| `RUN_LOG_DIR` | `logs/<basename>` | Grouped Slurm log directory |
| `DRY_RUN` | `0` | `1` to print derived shape without submitting |

## Full Output Tree

```
<OUTPUT_BASE_DIR>/
  run_manifest.json                         # seed file, grid, MAX_LOOP, sampler
  all_actions.npy                           # action space array
  replay_shards/
    trajectory_<run_id>.npz                 # compact per-trajectory replay shard
  full_trajectories/
    trajectory_<run_id>.zarr               # rich TORAX trace (SAVE_FULL_ZARR=1 only)
  trajectories/
    trajectory_<run_id>.json               # legacy format (SAVE_JSON=1 only)
  failures/
    failed_run_<run_id>.json               # error info for failed trajectories
  chunks/
    chunk_<task>_<start>_<end>/
      task_status.json
      tokamaker_torax_logs/               # per-trajectory TORAX solver logs
  grid_search/                            # present when SUBMIT_GRID_SEARCH=1
    grid_search_leaderboard.csv
    best_trajectory.json
    grid_search_summary.json
  replay_cache/                           # present when SUBMIT_REPLAY_CACHE=1
    states.npy
    actions.npy
    next_states.npy
    rewards.npy
    dones.npy
    replay_manifest.json
```

## Slurm Log Locations

```
logs/<run>/collect_trajectories-<array_job>_<task>.{out,err}
logs/<run>/grid_search_baseline-<job>.out
logs/<run>/materialize_replay_cache-<job>.out
logs/<run>/train_iql-<job>.out
```

## Notes

- Array workers validate `run_manifest.json` on startup and exit nonzero if
  they detect a seed, grid, `MAX_LOOP`, or sampler mismatch — preventing
  mixed datasets.
- `MaxArraySize=1001` on this cluster means array task IDs `0..1000`; increase
  `CHUNK_SIZE` or split the range when you need more tasks.
- Scale throughput with `ARRAY_CONCURRENCY`, not `N_WORKERS`. Each worker in a
  task is a separate Python process sharing the same Slurm resource allocation.
- The relax cache (`USE_INITIAL_RELAX_CACHE=1`) is keyed by hash of
  `grid_size`, initial profiles, and equilibrium inputs; identical runs reuse
  the same file. See `docs/cache_and_gpu.md` for details.
