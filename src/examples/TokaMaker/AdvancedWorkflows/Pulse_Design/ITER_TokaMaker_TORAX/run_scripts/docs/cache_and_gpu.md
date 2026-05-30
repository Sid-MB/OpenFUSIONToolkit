# Optional Cache And GPU Runs

The standard CPU array disables the shared initial relax cache. Use the cache
only when a small benchmark shows it is stable and faster for the current code.

## Keyed Cache Files

The initial TORAX relax happens at `t=0`, before any heating/RL decision, so it
depends only on the fixed inputs — `grid_size`, the initial `n_e`/`T_e`
profiles, and the equilibrium (coil bounds, Ip targets, x-points, eqdsk). Cache
files are therefore **keyed by a hash of those inputs** and stored together in a
shared directory:

```
<INITIAL_RELAX_CACHE_DIR>/initial_relax_<key>.json        # the cached state
<INITIAL_RELAX_CACHE_DIR>/initial_relax_<key>.params.json # human-readable key inputs
```

Every distinct combination gets its own file, and identical combinations
transparently reuse the same file across collection, eval, and training-triggered
eval. As long as you do not change `grid_size` or the initial profiles, all runs
hit the same cache.

- `INITIAL_RELAX_CACHE_DIR` (env / `--initial_relax_cache_dir`): the shared
  directory. Defaults to the `INITIAL_RELAX_CACHE_DIR` env var, otherwise
  `<example_dir>/initial_relax_cache`.
- `INITIAL_RELAX_CACHE` (env / `--initial_relax_cache`): an explicit cache file
  path that overrides keying (legacy / manual reuse).
- The key logic lives once in `collect_trajectories_delta.py`
  (`default_relax_geometry`, `initial_relax_cache_key`,
  `resolve_initial_relax_cache_path`) and is imported by `rl/eval.py`, so
  collection and eval always compute the same key.

Resolve the keyed path for a given grid size without side effects:

```bash
uv run python collect_trajectories_delta.py \
  --print_initial_relax_cache_path --grid_size 51
```

## Shared Relax Cache

Submit a CPU cache job and a dependent CPU trajectory array:

```bash
START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-399%16 USE_INITIAL_RELAX_CACHE=1 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

Manual equivalent:

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)
cache_jid=$(START_IDX=600 END_IDX=1000 sbatch --parsable run_scripts/collect_initial_relax_cache_cpu.sh)
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
  USE_INITIAL_RELAX_CACHE=1 N_WORKERS=1 CHUNK_SIZE=1 \
  sbatch --dependency=afterok:${cache_jid} --cpus-per-task=4 --mem=128G --array=0-399%16 \
    run_scripts/collect_trajectories_cpu_array.sh
```

Reuse caches via a shared keyed directory (preferred): point every run at the
same `INITIAL_RELAX_CACHE_DIR`. The cache build and the array will resolve the
same `initial_relax_<key>.json`, building it once and reusing it thereafter.

```bash
export INITIAL_RELAX_CACHE_DIR=/path/to/shared/initial_relax_cache
START_IDX=600 END_IDX=1000 USE_INITIAL_RELAX_CACHE=1 \
  INITIAL_RELAX_CACHE_DIR="${INITIAL_RELAX_CACHE_DIR}" \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

Reuse a specific cache file (legacy / explicit override):

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_reuse_cache_$(date +%Y%m%d_%H%M%S)
export INITIAL_RELAX_CACHE=/path/to/initial_relax_state.json
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
  INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE}" USE_INITIAL_RELAX_CACHE=1 \
  sbatch --cpus-per-task=4 --mem=128G --array=0-399%16 \
    run_scripts/collect_trajectories_cpu_array.sh
```

## GPU Diagnostics

`collect_trajectories_gpu.sh` runs one GPU Slurm job on `jag-standard` with
`uv run --extra cuda13`. The Python entrypoint fails fast if a GPU is visible
but JAX cannot initialize a GPU backend.

Small GPU diagnostic:

```bash
OUTPUT_DIR=./rl_dataset_delta_gpu_$(date +%Y%m%d_%H%M%S) \
  START_IDX=600 END_IDX=601 N_WORKERS=1 \
  sbatch run_scripts/collect_trajectories_gpu.sh
```

Build only a GPU initial relax cache:

```bash
OUTPUT_BASE_DIR=./rl_dataset_delta_cache_gpu_$(date +%Y%m%d_%H%M%S) \
  START_IDX=600 END_IDX=1000 \
  sbatch run_scripts/collect_initial_relax_cache_gpu.sh
```
