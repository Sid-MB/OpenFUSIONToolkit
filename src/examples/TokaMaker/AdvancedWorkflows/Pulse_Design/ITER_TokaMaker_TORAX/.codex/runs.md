# Active Runs Memory

This file tracks the currently active or queued Slurm jobs in this workspace,
why they exist, and what to do with the outputs once they finish.

Current state was last refreshed from `squeue` in the project root.

## 1. `15710794` and `15710797` - full `prev_action` collection

- `15710794`: shared initial-relax cache for the full `prev_action` dataset.
- `15710797`: trajectory collection array for the full `prev_action` dataset.
- Purpose: produce a fresh production dataset with `OBSERVATION_MODE=prev_action` and reward-recalc stats enabled by default.
- Bigger context: this is the production schema the later IQL training should be judged against.
- Current state: the cache job is pending and the array is still waiting on it.
- When finished:
  - validate that `run_prev_action_full_recollect_20260531_1825/` contains the manifest, replay shards, replay cache, and training outputs.
  - prefer this dataset for future retraining when the goal is actor-conditioned control.
  - use it as the main reference for comparing against the smoke and plasma-only runs.
- Analysis:
  - inspect collection completeness first.
  - then compare replay-cache statistics and training / eval curves against the smoke baseline.
  - if the actor looks saturated or unstable, use this run to study whether the full dataset changed that behavior.

## 2. `15710798` and `15710801` - full `prev_action` downstream pipeline

- `15710798`: replay-cache materialization for the full `prev_action` dataset.
- `15710801`: IQL training job chained after the full collection.
- Purpose: turn the fresh full dataset into a trainable replay cache and train the actor.
- Bigger context: this is the first full production IQL run on the new schema.
- When finished:
  - capture the final checkpoint and the automatically triggered actor eval outputs.
  - compare its eval artifacts to the smoke run and the legacy-compatible baseline.
- Analysis:
  - use the actor eval outputs as the main signal for controller quality.
  - check action saturation, smoothness, and closed-loop convergence.
  - if the post-training eval diverges from the notebook-style baseline, inspect the closed-loop trajectory, not just the scalar loss.

## 3. `15710795` and `15710796` - full `plasma_only` collection

- `15710795`: shared initial-relax cache for the full `plasma_only` dataset.
- `15710796`: trajectory collection array for the full `plasma_only` dataset.
- Purpose: produce an ablation dataset with no action history in the observation.
- Bigger context: this is the cleanest test of whether previous-action feedback is part of the policy's smoothness / saturation behavior.
- Current state: the cache job is pending and the array is still waiting on it.
- When finished:
  - validate the dataset root `run_plasma_only_full_recollect_20260531_1825/`.
  - train and evaluate against it if the downstream jobs finish.
- Analysis:
  - compare against the `prev_action` full run.
  - if this produces less self-reinforcing behavior, that supports removing or weakening action-history leakage in future runs.

## 4. `15710799` and `15710800` - full `plasma_only` downstream pipeline

- `15710799`: replay-cache materialization for the `plasma_only` dataset.
- `15710800`: IQL training job chained after the plasma-only collection.
- Purpose: train the ablation model on the no-action-history schema.
- Bigger context: this is the ablation counterpart to the full `prev_action` production run.
- When finished:
  - compare its checkpoint and eval outputs against the `prev_action` run.
  - decide whether the action-history feature is helping or just smoothing the policy in a self-reinforcing way.
- Analysis:
  - this is the most informative run for testing whether `prev_action` should stay in the observation.

## 5. `15691772` - smoke-based full training run

- `15691772`: IQL training job reusing the completed smoke dataset.
- Purpose: get a full training pass without recollecting data.
- Bigger context: this is the low-risk, reuse-first check that the training/eval plumbing still works.
- When finished:
  - inspect the checkpoint and the automatically triggered eval outputs in the smoke dataset root.
  - use it as a fast sanity check for the training/eval contract.
- Analysis:
  - this is a good place to compare training dynamics against the fresh full runs.
  - if this behaves well but the fresh full run does not, the difference is likely in the dataset schema or coverage, not the basic trainer.

## 6. `15710559` and `15691315` - CPU eval comparisons on `john`

- `15710559`: cache-disabled actor-side eval retry on `john`.
- `15691315`: another actor eval pending on resources.
- Purpose: compare baseline notebook-style behavior against the trained actor on the same seed EQDSKs and shortened window, and isolate whether the persistent JAX cache is contributing to the native crash.
- Bigger context: this is the most direct check for whether the actor or the eval path is what causes convergence problems.
- When finished:
  - compare the first divergence point between baseline and actor.
  - treat it as a debugging artifact, not as the final benchmark.
- Analysis:
  - use this to isolate solver/runtime issues from policy behavior.
  - if both baseline and actor fail on the same backend path, the problem is lower-level than the policy.

## 7. `15691331` and `15691452` - dependency wait jobs

- These are queued wait-for jobs tied to earlier dependency chains.
- Purpose: keep the main pipeline flow moving when upstream jobs complete.
- Bigger context: they are bookkeeping, not model signal.
- When finished:
  - only inspect them if a downstream job appears stuck or never starts.
- Analysis:
  - these do not provide model insight on their own.

## 8. `15710802` and `15710803` - collection telemetry sidecars

- `15710802`: collection W&B telemetry for the `plasma_only` dataset.
- `15710803`: collection W&B telemetry for the `prev_action` dataset.
- Purpose: log dataset-level collection metadata to the separate `iql-collection` project without mixing it into training/eval.
- Bigger context: useful for tracking dataset builds, but not part of model training itself.
- When finished:
  - inspect only if you need the dataset provenance dashboard.

## 9. `15688937` - interactive shell session

- Purpose: live human-driven debugging session in the `jag` allocation.
- Bigger context: useful for ad hoc inspection, but not a pipeline artifact.
- When finished:
  - no model analysis needed; it is just the shell session ending.

## General analysis rules

- Prefer the full `prev_action` and `plasma_only` production runs for substantive comparisons.
- Use the smoke-backed training run as the reuse baseline.
- Use the CPU baseline-vs-actor compare only to diagnose convergence/runtime issues, not to judge final model quality.
- For any completed training run, the important outputs are:
  - the final checkpoint
  - the automatically triggered eval
  - the eval summary JSON
  - the action trace JSON
  - the plot/movie artifacts
- For any completed collection run, the important outputs are:
  - `run_manifest.json`
  - `all_actions.npy`
  - `replay_shards/`
  - `replay_cache/` if it was materialized
  - `logs/`
