# Active Runs Memory

This file tracks the current Slurm jobs and the small set of completed
reference runs that matter for ongoing analysis in this workspace.

Last refreshed from `squeue` and the current output tree on 2026-06-01.

## Active Jobs

### 1. `15712247` and `15712238` - full `prev_action` collection pipeline

- `15712247`: shared initial-relax cache for the fresh `prev_action` rerun.
- `15712087`: trajectory collection array for the fresh `prev_action` rerun.
- Purpose: produce a fresh production dataset with `OBSERVATION_MODE=prev_action` and the current no-cache CPU defaults.
- Current state: cache job is running; array is pending on it.
- When finished:
  - validate `run_prev_action_full_recollect_20260531_1825/`
  - prefer this dataset for future retraining when actor-conditioned control is the goal

### 2. `15712237` and `15712087` - full `prev_action` downstream pipeline

- `15712237`: replay-cache materialization for the full `prev_action` dataset.
- `15712238`: IQL training job chained after the full collection.
- Purpose: turn the fresh dataset into a trainable replay cache and train the actor.
- Current state: both waiting on dependencies.
- When finished:
  - capture the final checkpoint and the automatically triggered actor eval outputs
  - compare against the smoke run and the full final checkpoint eval

### 3. `15712248` and `15712102` - full `plasma_only` collection pipeline

- `15712248`: shared initial-relax cache for the fresh `plasma_only` rerun.
- `15712102`: trajectory collection array for the fresh `plasma_only` rerun.
- Purpose: ablation dataset with no action history in the observation, using the current no-cache CPU defaults.
- Current state: cache job is running; array is pending on it.
- When finished:
  - validate `run_plasma_only_full_recollect_20260531_1825/`
  - compare against the `prev_action` full run

### 4. `15712249` and `15712104` - full `plasma_only` downstream pipeline

- `15712249`: replay-cache materialization for the `plasma_only` dataset.
- `15712104`: IQL training job chained after the plasma-only collection.
- Purpose: train the ablation model on the no-action-history schema.
- Current state: waiting on dependencies.
- When finished:
  - compare its checkpoint and eval outputs against the `prev_action` run

### 5. `15712250` - dependency helper

- `15712250`: wait-for helper job chained off the fresh collection reruns.
- Purpose: keep the refresh chain visible while the cache jobs settle.
- Current state: pending on dependencies.

### 6. `15712038` - CPU eval rerun on `john` with current reward config

- `15712038`: actor-side eval rerun on `john` against
  `run_smb_prev_action_smoke_20260530_222522/` using the current
  `RLRewardConfig` defaults.
- Purpose: regenerate the eval artifact bundle so reward provenance is explicit.
- Current state: running on `john11`.
- When finished:
  - inspect `actor_eval_summary.json`, `actor_eval_actions.json`, and the plots/movie

### 7. `15712236` - separate `sc-loprio` parallelism test

- `15712236`: independent `collect_trajectories_cpu_array.sh` run on `sc-loprio`.
- Purpose: test a different scheduler / allocation shape, not part of the main RL pipelines.
- Current state: pending or running separately from the main collection chains.

### 8. `15712434` - low-smoothness smoke eval on `john`

- `15712434`: CPU eval of the low-smoothness smoke checkpoint
  `out/iql/train_low_smoothness_smoke_20260531/f329um07/iql_weights.pt`
  using the standard notebook-style eval wrapper on `john`.
- Purpose: compare the reduced-smoothing ablation against the reference smoke
  and full checkpoints once the rollout completes.
- Current state: pending on cluster resources.
- When finished:
  - inspect `actor_eval_summary.json`, `actor_eval_actions.json`, and the
    generated plots/movie in `out/iql_eval/low_smoothness_smoke_20260531/`

## Completed Reference Runs

### Smoke live-render eval

- Job: `15688944`
- Output: `out/iql_eval/smoke_live_render_docfix_20260531/`
- Status: completed successfully
- Notes:
  - canonical notebook-style eval path
  - earlier local attempt on `jagupard32` was interrupted when the host was killed

### Final full-checkpoint eval

- Job: `15691315`
- Output: `out/iql_eval/full_final_checkpoint_20260531/`
- Status: completed successfully
- Evaluated checkpoint:
  - `out/iql/rl_dataset_delta_sampling_maxloop=2_grid_51_full_zarr_2000_20260527_123853_gpu_iql/8l1ywupe/iql_weights.pt`

## Notes

- For notebook-style plots/movie, use `run_scripts/eval_iql_actor_cpu.sh`.
- For training ablations with less smoothing, use
  `run_scripts/train_iql_low_smoothness.sh`.
- The finished full checkpoint eval is the main reference point for comparison
  against the grid-search baseline.
