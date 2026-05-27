# Optional Cache And GPU Runs

The standard CPU array disables the shared initial relax cache. Use the cache
only when a small benchmark shows it is stable and faster for the current code.

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
  sbatch --dependency=afterok:${cache_jid} --cpus-per-task=4 --mem=16G --array=0-399%16 \
    run_scripts/collect_trajectories_cpu_array.sh
```

Reuse an existing cache:

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_reuse_cache_$(date +%Y%m%d_%H%M%S)
export INITIAL_RELAX_CACHE=/path/to/initial_relax_state.json
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
  INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE}" USE_INITIAL_RELAX_CACHE=1 \
  sbatch --cpus-per-task=4 --mem=16G --array=0-399%16 \
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
