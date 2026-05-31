# Eval Performance: Compile-Once, Persistent Cache, CPU, and Batch

This document explains the performance changes made to the IQL actor evaluation
pipeline and how to use them.  See also `script_roles.md` for a map of which
script to run.

## Background: Why Eval Was Slow

The RL decision loop in `TokaMaker_TORAX._run_tx_rl_segmented` (in
`pulse_design.py`) runs a separate TORAX simulation for each RL decision:

```
loop:
  cold-start [0 → t_dec]:   re-run TORAX from t=0 with all actions so far
  observe state at t_dec
  actor chooses action
```

With 22 decisions per `max_loop=2` run this means 22+ TORAX calls per eval.
Each TORAX call runs `run_simulation`, which JAX JIT-compiles a step function.
Two problems made this very slow:

1. **Per-segment recompilation (~30 s each).**  Each cold-start segment had a
   slightly longer heating-schedule array than the previous one (one new action
   knot was added per decision).  JAX treats array shape as part of the
   JIT signature, so every segment triggered a fresh XLA trace and compile.
   With 22 segments × 30 s = **~11 min in compilation alone** (on CPU).

2. **Persistent compilation cache was disabled.**  Even if two segments happened
   to share a shape, the cache was hard-disabled in `pulse_design.py`
   (`jax.config.update('jax_enable_compilation_cache', False)`) to avoid
   semaphore leaks in prior multiprocessing setups.

## Fix 1: Compile-Once Heating Schedules

`TokaMaker_TORAX._full_agent_knots_defaults()` pre-seeds the `agent_knots`
dict with default actions at **every** future decision time before the loop
starts.  This means the merged heating-schedule arrays passed to TORAX have
the same length in every segment—JAX sees a constant JIT signature and
compiles exactly once.

**Correctness guarantee:** a future knot at time `t_future > t_seg_end` is
never interpolated within a segment that only integrates up to `t_seg_end`.
TORAX only reads knot values inside the integration window, so future knots
don't affect results.  As the actor makes real decisions, those entries are
overwritten in place (same shape, new values).

Implemented in `_run_tx_rl_segmented`:
```python
agent_knots = self._full_agent_knots_defaults()   # pre-seeded at all knots
# ...
for t_dec in RL_DECISION_TIMES:
    # ...action chosen by actor...
    agent_knots[knot_t] = action   # overwrite, does not change dict length
```

## Fix 2: Persistent XLA Compilation Cache

`pulse_design.fly()` now re-enables JAX's persistent compilation cache and
sets the minimum compile-time / entry-size thresholds to zero so every
compile is cached regardless of how fast it is:

```python
jax.config.update('jax_enable_compilation_cache', True)
jax.config.update('jax_compilation_cache_dir', cache_dir)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
```

`rl/eval.py` also sets the corresponding environment variables at import time
so the cache root is configured before JAX initializes. The TORAX runtime then
adds a namespace based on the installed OFT/JAX/Python build fingerprint:

```python
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "<project>/.jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
```

**Result:** the very first eval on a clean cache compiles once (~30 s on CPU).
Every subsequent eval with the same build fingerprint — including other
processes in a batch job — loads the compiled executable from disk and pays
**0 s compilation cost**.

### Cache location

Default: `.jax_cache/` inside the project directory (set by `rl/eval.py`),
with the runtime appending a build-specific namespace under that root.
Override with:

```bash
export JAX_COMPILATION_CACHE_DIR=/path/to/shared/cache
```

A shared cache directory works across multiple batch workers and across
separate sbatch jobs submitted to the same host.

### Disabling the cache

If you observe semaphore-leak warnings that are not suppressed by `os._exit`
(e.g. when running interactively), disable the cache with:

```bash
export OFT_DISABLE_JAX_COMPILE_CACHE=1
```

You can also request the same mode through the eval wrappers:

```bash
OFT_DISABLE_JAX_COMPILE_CACHE=1 sbatch run_scripts/eval_iql_actor_cpu.sh
OFT_DISABLE_JAX_COMPILE_CACHE=1 sbatch run_scripts/eval_iql_actor_cpu_batch.sh
```

## Fix 3: CPU Execution

The TORAX RL eval loop is latency-bound, not throughput-bound:

- Each segment is a short 1-D solve on a 51-point radial grid.
- GPU compute utilization spikes to ~5% then drops back to zero between
  segments; GPU memory stays allocated at ~75% throughout (XLA heap).
- On CPU, there is no GPU memory preallocation and XLA compilation is also
  cheaper.

**Recommendation:** run evals on the `john` (CPU) partition using
`eval_iql_actor_cpu.sh` or `eval_iql_actor_cpu_batch.sh`.  Reserve GPU
nodes for training (`train_iql.sh`) and collection with GPU-specific
benchmarking (`collect_trajectories_gpu.sh`).

## Fix 4: Batch Parallelism

Individual evals are serial (each RL decision depends on the previous one),
but multiple checkpoints/seeds are fully independent.  `rl/eval_batch.py`
runs `N_WORKERS` evals concurrently using `multiprocessing.Pool`:

- Each worker process runs one eval at a time (`maxtasksperchild=1` for
  clean OFT/JAX process isolation — same pattern as
  `collect_trajectories_delta.py`).
- Workers share the same `.jax_cache/` directory; the first to arrive
  compiles and writes, the rest load from disk.
- CPU thread budget is split evenly: `THREADS_PER_WORKER = TOTAL_CPUS / N_WORKERS`.

## Quick Reference

### Single checkpoint, CPU

This is the canonical path when you want the notebook-style outputs in one
pass. It runs the closed-loop TORAX eval, writes the actor summary bundle, and
renders plots/movie from the live `tmtx` object before exiting. The saved
`actor_eval_bundle.pkl` is useful for diagnostics, but it is not the preferred
source for rendering the full movie later.

```bash
# Submit to john partition:
ACTOR_CHECKPOINT=out/iql/<run>/iql_weights.pt \
  DATASET_DIR=./rl_dataset_eval_smoke_1_20260528_130200 \
  sbatch run_scripts/eval_iql_actor_cpu.sh

# Run synchronously (blocks until done, useful for timing):
ACTOR_CHECKPOINT=out/iql/<run>/iql_weights.pt \
  DATASET_DIR=./rl_dataset_eval_smoke_1_20260528_130200 \
  WANDB_MODE=offline \
  srun --account=nlp --mem=128G --cpus-per-task=20 --partition=john \
  bash run_scripts/eval_iql_actor_cpu.sh
```

### Multiple checkpoints in parallel, CPU

Batch mode is for throughput and checkpoint sweeps. Each worker still runs the
live eval path, but if you only need one full notebook-style run, prefer the
single-checkpoint wrapper above.

```bash
# 4 checkpoints, 4 workers × 5 CPUs = 20 total:
ACTOR_CHECKPOINTS="out/iql/run_a/iql_weights.pt out/iql/run_b/iql_weights.pt \
  out/iql/run_c/checkpoint_step_50000.pt out/iql/run_d/checkpoint_step_50000.pt" \
  N_WORKERS=4 \
  sbatch run_scripts/eval_iql_actor_cpu_batch.sh

# From a file (one path per line, # = comment):
CHECKPOINTS_FILE=my_checkpoints.txt \
  N_WORKERS=4 \
  sbatch run_scripts/eval_iql_actor_cpu_batch.sh
```

### Programmatic (Python)

```python
from rl.eval import run_actor_eval_from_config
result = run_actor_eval_from_config(
    actor_checkpoint="out/iql/<run>/iql_weights.pt",
    output_dir="out/iql_eval/my_run",
    dataset_dir="./rl_dataset_eval_smoke_1_20260528_130200",
    max_loop=2,
    grid_size=51,
)
```

### Bundle-only diagnostics

If you already have `actor_eval_summary.json` and `actor_eval_bundle.pkl`, you
can call `rl.eval_postprocess.postprocess_actor_eval(...)` directly. That is a
diagnostic path only. It may not have enough runtime state to reconstruct the
complete notebook movie, so do not use it as the primary eval workflow.

## Performance Findings

Measurements taken on CPU (`john` / `jagupard` partition), `grid_size=51`,
`max_loop=1` unless noted. "Segment" = one cold-start TORAX call covering
`[0, t_decision]`.

### Per-Segment Timings

| Condition | First segment | Later segments | Source |
|---|---|---|---|
| Before (growing schedule, cache disabled) | ~30 s (compile + run) | ~30 s (recompile every time) | empirical |
| After (fixed schedule, cache enabled) | ~30 s (compile + run) | ~0 s compile + run only | empirical |
| Cache warm (second+ run, same host) | ~0 s compile + run only | ~0 s compile + run only | empirical |

### End-to-End Wall-Clock

| Scenario | Wall-clock | Notes |
|---|---|---|
| Before: `max_loop=2`, 22 segments, recompiling | > 30 min (est.) | ~11 min compilation alone (22 × 30 s) |
| After: `max_loop=1`, cold cache | ~20 min | Empirical; first compile ~30 s, rest ~0 s |
| After: `max_loop=1`, warm cache | < 20 min | All compilation skipped |

### GPU vs CPU

| Backend | GPU util | GPU mem | Compile cost | Verdict |
|---|---|---|---|---|
| GPU (`jag-standard`) | ~5% brief spikes | ~75% allocated (XLA heap) | same or higher | not recommended |
| CPU (`john`) | — | — | lower | **recommended** |

The eval loop is latency-bound (short 1-D solves on a 51-point grid), not
compute-bound. GPU memory preallocation and kernel-launch overhead outweigh
any parallel compute benefit at this problem size.

### Batch Parallelism Scaling

With `N_WORKERS` independent evals on a single node and a warm shared cache:

| N_WORKERS | CPUs used | Wall-clock (est., `max_loop=1`) | Notes |
|---|---|---|---|
| 1 | 8 | ~20 min | baseline |
| 4 | 32 | ~20 min | 4× throughput, same wall-clock |
| 8 | 64 | ~20 min | 8× throughput, same wall-clock |

Wall-clock stays approximately constant as long as each worker gets enough
CPUs (≥ 4 recommended). Throughput scales linearly with workers up to the
node's core count.

### Warm-Start Fidelity Smoke Test

A quick `grid_size=11` comparison between cold-start and warm-start approaches
at one decision boundary showed exact numerical agreement:

| Metric | Cold `[0→100]` | Warm `[0→80]` + tail `[80→100]` | Max abs diff |
|---|---|---|---|
| `psi` | — | — | 0.0 |
| `n_e` | — | — | 0.0 |
| `T_e` | — | — | 0.0 |
| `T_i` | — | — | 0.0 |
| `Ip`, `Q_fusion`, `q95`, `beta_N`, `v_loop_lcfs` | — | — | 0.0 |

Timing at `grid_size=11` (before compile-once fix, so includes compile cost):

| Segment | Time |
|---|---|
| Cold `[0→100]` | 56.6 s |
| Warm prefix `[0→80]` | 41.7 s |
| Warm tail `[80→100]` | 50.3 s |
| Tail-only speedup vs cold | 1.13× |

The 1.13× speedup is misleading: each shape still paid its own compile cost.
After compile-once + persistent cache the warm-start advantage would be larger,
but the current cold-start approach already achieves ~0 s per-segment
recompilation so the additional complexity of warm-start is not yet warranted.
