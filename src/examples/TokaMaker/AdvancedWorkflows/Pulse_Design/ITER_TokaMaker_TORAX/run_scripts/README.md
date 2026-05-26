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
START_IDX=600 END_IDX=1000 ARRAY_SPEC=0-19%4 ./run_scripts/submit_collect_trajectories_cpu_array.sh
```

This workflow assumes the `john` OFT install exists. If it does not, build it
first with:

```bash
OFT_BUILD_FLAVOR=john bash ../../../../../../scripts/oft_arch/our_setup_arch.sh
```

The example above runs trajectories `[600, 1000)` in 20 chunks. With
`CHUNK_SIZE=20`, task 0 runs `[600, 620)`, task 1 runs `[620, 640)`, and task 19
runs `[980, 1000)`. The `%4` limits Slurm to four array tasks running at once.

Outputs go under:

```text
rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_<timestamp>/
```

The shared initial relax cache is:

```text
initial_relax_state.json
```

inside that output base directory.

## Manual Dependency Workflow

If you want direct control over job IDs:

```bash
export OUTPUT_BASE_DIR=./rl_dataset_delta_sampling_maxloop=2_grid_51_cpu_array_$(date +%Y%m%d_%H%M%S)
cache_jid=$(START_IDX=600 END_IDX=1000 sbatch --parsable run_scripts/collect_initial_relax_cache_cpu.sh)
START_IDX=600 END_IDX=1000 OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
  sbatch --dependency=afterok:${cache_jid} --array=0-19%4 run_scripts/collect_trajectories_cpu_array.sh
```

## Other Wrappers

- `collect_trajectories_cpu.sh`: single CPU Slurm job on `john`.
- `collect_trajectories_gpu.sh`: single GPU Slurm job on `jag-standard`.
- `../setup-env.sh`: local environment setup helper.
