# TokaMaker/TORAX Run Scripts

Submit these commands from the `ITER_TokaMaker_TORAX` example directory unless
using the submit helper, which changes to the example directory automatically.

## Architecture-Specific OFT Builds

The run wrappers source:

```bash
../../../../../../scripts/oft_arch/select_oft_install.sh
```

That selector chooses the OFT install from the Slurm partition or hostname:

- `john` partition/hosts -> `install_release_john`
- `jag-standard` / `jag*` hosts -> `install_release_jag` if it exists, otherwise `install_release`

You can override the selection with:

```bash
OFT_INSTALL_FLAVOR=john   # or jag
OFT_INSTALL_DIR=/path/to/install_release_custom
```

Build or rebuild the flavor on the matching machine type before submitting jobs.
For a fresh build, run:

```bash
OFT_BUILD_FLAVOR=john OFT_MAKE_JOBS=8 bash ../../../../../../scripts/oft_arch/our_setup_arch.sh
```

For a GPU/Jaguar-side build:

```bash
OFT_BUILD_FLAVOR=jag OFT_MAKE_JOBS=8 bash ../../../../../../scripts/oft_arch/our_setup_arch.sh
```

If the flavor-specific third-party libraries already exist and only CMake/OFT
needs to be regenerated:

```bash
OFT_BUILD_FLAVOR=john bash ../../../../../../scripts/oft_arch/configure_cmake_arch.sh
OFT_BUILD_FLAVOR=john bash ../../../../../../scripts/oft_arch/rebuild_arch.sh
```

Use `OFT_BUILD_FLAVOR=jag` for the corresponding GPU-node install. The Slurm
wrappers will pick up the correct install automatically when they run on their
target partition.

## CPU Array With Shared Relax Cache

Use this for the large CPU run on `john`. It submits one cache-building job, then
submits the trajectory array with a Slurm `afterok` dependency so no array task
holds a node while waiting for the cache.

```bash
START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-199%8 N_WORKERS=2 CHUNK_SIZE=2 \
  ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

To build the shared initial relax cache on a GPU node and then run the dependent
trajectory array on `john`, submit the jobs manually:

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_gpu_cache_cpu_array_$(date +%Y%m%d_%H%M%S)
cache_jid=$(START_IDX=600 END_IDX=1000 sbatch --parsable run_scripts/collect_initial_relax_cache_gpu.sh)
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" N_WORKERS=2 CHUNK_SIZE=2 \
  sbatch --dependency=afterok:${cache_jid} --cpus-per-task=2 --mem=128G --array=0-199%8 \
    run_scripts/collect_trajectories_cpu_array.sh
```

This workflow assumes the `john` OFT install exists. If it does not, build it
first with:

```bash
OFT_BUILD_FLAVOR=john bash ../../../../../../scripts/oft_arch/our_setup_arch.sh
```

The example above runs trajectories `[600, 1000)` in 200 chunks. With
`CHUNK_SIZE=2`, task 0 runs `[600, 602)`, task 1 runs `[602, 604)`, and task 199
runs `[998, 1000)`. The `%8` limits Slurm to eight array tasks running at once.

RAM can be the limiter as `N_WORKERS` increases. On the ITER/TORAX sweep we saw
active trajectory workers use roughly 60 GiB RSS each, so a 128 GiB Slurm task
only has room for about two active workers. Prefer `N_WORKERS=2` and more Slurm
array tasks over `N_WORKERS=20` inside one task.

Outputs go under:

```text
rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_<timestamp>/
```

The shared initial relax cache is:

```text
initial_relax_state.json
```

inside that output base directory.

To reuse an existing shared initial relax cache, skip the cache-building job and
submit only the CPU array with `INITIAL_RELAX_CACHE` set. For example, this uses
the cache built by GPU job `15555869`:

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_reuse_cache_$(date +%Y%m%d_%H%M%S)
export INITIAL_RELAX_CACHE=./rl_dataset_delta_sampling_maxloop=2_grid_51_gpu_cache_cpu_array_20260525_181913/initial_relax_state.json
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" INITIAL_RELAX_CACHE="${INITIAL_RELAX_CACHE}" \
  N_WORKERS=2 CHUNK_SIZE=2 \
  sbatch --cpus-per-task=2 --mem=128G --array=0-199%8 run_scripts/collect_trajectories_cpu_array.sh
```

## Manual Dependency Workflow

If you want direct control over job IDs:

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)
cache_jid=$(START_IDX=600 END_IDX=1000 sbatch --parsable run_scripts/collect_initial_relax_cache_cpu.sh)
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" N_WORKERS=2 CHUNK_SIZE=2 \
  sbatch --dependency=afterok:${cache_jid} --cpus-per-task=2 --mem=128G --array=0-199%8 \
    run_scripts/collect_trajectories_cpu_array.sh
```

## Other Wrappers

- `collect_trajectories_cpu.sh`: single CPU Slurm job on `john`.
- `collect_trajectories_gpu.sh`: single GPU Slurm job on `jag-standard`.
- `collect_initial_relax_cache_gpu.sh`: GPU Slurm job that builds only the shared initial relax cache.
- `../setup-env.sh`: local environment setup helper.
