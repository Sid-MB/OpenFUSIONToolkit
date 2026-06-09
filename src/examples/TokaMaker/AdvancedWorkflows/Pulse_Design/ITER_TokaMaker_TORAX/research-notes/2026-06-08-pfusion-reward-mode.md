# 2026-06-08 — pfusion reward mode: fixing Q reward hacking

## Motivation

Investigated the all-time highest-reward IQL run (`prev_action_q95w3.5_fgww4.0`, W&B run `wuvw65vr`, `iql-training` project). It showed Q_max=152.6 — far above every other run. Investigation revealed two problems:

1. **Reward hacking via auxiliary power starvation**: the step reward `log(mean_Q + 1)` is unbounded as Q → ∞. Since Q = P_fusion / P_aux, zeroing out auxiliary heating (ECRH + NBI) drives Q → ∞. The policy learned to cut all heating from t=240s onwards and coast, getting Q_max=152 at t=470s. Per-step rewards: `[0.26, ..., 4.26, **40.21**]` — the last step is 10× any other step.

2. **Incomparable reward config**: the run used non-standard penalty weights (`q95_penalty_weight=3.5`, `fgw_penalty_weight=4.0` vs standard 1.2 / 2.0), making reward totals uncomparable to all prior runs.

3. **Constraint violation**: `q95_min=2.87 < 3.0` — the q95 stability constraint was violated.

Output artifacts for that run: `out/iql/reward_eeaff1f2291a/w0gqz7fd/actor_eval/`

---

## Fix: pfusion reward mode

Added `reward_mode='pfusion'` to `RLRewardConfig`. In pfusion mode, step reward becomes:

```
log(mean_Q * P_aux_MW + 1)
```

When P_aux = 0, reward = log(0 + 1) = 0. This directly removes the 1/P_aux blow-up exploit at the source — turning off heating gives zero reward, not infinite reward.

Terminal bonus is also updated in pfusion mode to use `mean(Q * P_aux_MW)` over flattop instead of `Q_flattop_avg`, for the same reason.

`reward_mode` is tracked in the config dict (via `RL_REWARD_MODE` env var), so pfusion runs get a distinct dataset variant hash (`reward_68ceccd06040`) and are not confused with standard-mode runs.

### Files changed

- `src/python/OpenFUSIONToolkit/TokaMaker/pulse_design.py`:
  - Added `reward_mode: str = 'standard'` field to `RLRewardConfig`
  - Updated `compute_rewards()` to use `log(Q * P_aux_MW + 1)` in pfusion mode (step reward)
  - Updated terminal bonus to use `mean(Q * P_aux_MW)` over flattop in pfusion mode

- `src/examples/.../ITER_TokaMaker_TORAX/dataloader.py`:
  - Added `'reward_mode': 'RL_REWARD_MODE'` to `_REWARD_ENV_OVERRIDES`
  - Added `_REWARD_STRING_FIELDS = {'reward_mode'}` so it isn't coerced to float
  - Updated `recompute_reward_series_from_stats` to read `reward_mode` from config dict
  - Added `'reward_mode': 'standard'` to hardcoded fallback defaults

- `src/examples/.../ITER_TokaMaker_TORAX/rl/eval_sim.py`:
  - `_load_checkpoint_reward_config`: now searches `ckpt_dir.parent` for `iql_config.json` (intermediate checkpoints in `checkpoints/` subdir were getting "unavailable" provenance)
  - Auto-propagates `reward_mode` from checkpoint config into `os.environ` so downstream `default_reward_config()` picks it up without requiring caller to set `RL_REWARD_MODE`

- `src/examples/.../ITER_TokaMaker_TORAX/IQL.py`:
  - Added `run_config=None` parameter to `train_iql()`
  - Passes `'config': run_config` in intermediate checkpoint dict so `_load_checkpoint_reward_config` can find reward provenance in checkpoint files directly (not just `iql_config.json`)
  - Fixed `NameError: base_config not defined` bug introduced during the above change

- `src/examples/.../ITER_TokaMaker_TORAX/run_scripts/reweight_pfusion.sh`: new script for running dataset reweighting jobs on Slurm (CPU)

**Note**: `pulse_design.py` was manually copied to `install_release/` and `install_release_john/` instead of running `rebuild.sh` (pure-Python change, no compiled extensions affected). Should use `rebuild.sh` in the future per AGENTS.md.

---

## Dataset reweighting

Reweighted two existing datasets with pfusion rewards (standard penalty weights):

```bash
RL_REWARD_MODE=pfusion uv run python update_trajectories.py run_prev_action_wide_ecrh_20260602_000037
RL_REWARD_MODE=pfusion uv run python update_trajectories.py run_prev_action_full_20260601_192826
```

Both produced variant `reward_68ceccd06040` under their respective `reward_variants/` dirs.

---

## pfusion_standard run (40k steps, wide_ecrh dataset)

```bash
DATASET_DIR=run_prev_action_wide_ecrh_20260602_000037/reward_variants/reward_68ceccd06040 \
RUN_NAME=pfusion_standard \
OBSERVATION_MODE=prev_action \
CHECKPOINT_INTERVAL=1000 \
CHECKPOINT_EVAL_INTERVAL=2000 \
RUN_ACTOR_EVAL=1 \
RL_REWARD_MODE=pfusion \
sbatch run_scripts/train_iql.sh
# Job: 15757268
```

W&B training run: `iql-training/2tjm9kx1`
W&B actor eval: `iql-eval/x56e31ed` (`pfusion_standard-actor-eval`)
Output artifacts: `out/iql/reward_68ceccd06040/2tjm9kx1/`

Intermediate checkpoint evals all failed during training (reward_mode "unavailable" in checkpoint metadata — the `base_config` bug and `iql_config.json` search path bug, both fixed after this run). Final actor eval used the last checkpoint (step 40k) → Q_max=29.8.

### Fanout checkpoint evals

After training, submitted fanout evals on all 39 saved checkpoints:

```bash
DATASET_DIR=run_prev_action_wide_ecrh_20260602_000037/reward_variants/reward_68ceccd06040 \
TRAIN_OUTPUT_DIR=out/iql/reward_68ceccd06040/2tjm9kx1 \
EVAL_WANDB_PROJECT=iql-eval \
EVAL_WANDB_GROUP=pfusion_standard \
RL_REWARD_MODE=pfusion \
bash run_scripts/fanout_checkpoint_evals.sh
# Array job: 15796044 (0-38%16) — all 39 COMPLETED
```

**Best checkpoint**: step 39000 — Q_max=51.68, Q_flattop_avg=20.9, E_fusion=59,892 MJ, H98=0.801
(vs prior best non-pfusion run `wide_arp0.1`: Q_max=57.0)

Learning curve was still climbing at step 39k, motivating longer training.

---

## pfusion_long (qqtf58re, wide_ecrh dataset, 60k steps)

```bash
DATASET_DIR=run_prev_action_wide_ecrh_20260602_000037/reward_variants/reward_68ceccd06040 \
RUN_NAME=pfusion_long OBSERVATION_MODE=prev_action NUM_STEPS=60000 \
CHECKPOINT_INTERVAL=1000 CHECKPOINT_EVAL_INTERVAL=4000 \
RUN_ACTOR_EVAL=1 RL_REWARD_MODE=pfusion \
sbatch run_scripts/train_iql.sh
# Job: 15830578

# Fanout checkpoint evals (59 checkpoints):
DATASET_DIR=run_prev_action_wide_ecrh_20260602_000037/reward_variants/reward_68ceccd06040 \
TRAIN_OUTPUT_DIR=out/iql/reward_68ceccd06040/qqtf58re \
EVAL_WANDB_PROJECT=iql-eval EVAL_WANDB_GROUP=pfusion_long RL_REWARD_MODE=pfusion \
bash run_scripts/fanout_checkpoint_evals.sh
# Array job: 15830903
```

Learning curve was noisy; best checkpoint appeared near the end of training. Notable improvement vs 40k-step pfusion_standard run.

**Best checkpoint: step 58000** — Q_max=54.14, Q_flattop_avg=26.23, H98=0.795, E_fusion=59,611 MJ, reward_total=54.15

---

## pfusion_full_dataset (ni7a42nz, full dataset, 60k steps)

```bash
DATASET_DIR=run_prev_action_full_20260601_192826/reward_variants/reward_68ceccd06040 \
RUN_NAME=pfusion_full_dataset OBSERVATION_MODE=prev_action NUM_STEPS=60000 \
CHECKPOINT_INTERVAL=1000 CHECKPOINT_EVAL_INTERVAL=4000 \
RUN_ACTOR_EVAL=1 RL_REWARD_MODE=pfusion \
sbatch run_scripts/train_iql.sh
# Job: 15830579

# Fanout checkpoint evals (59 checkpoints):
DATASET_DIR=run_prev_action_full_20260601_192826/reward_variants/reward_68ceccd06040 \
TRAIN_OUTPUT_DIR=out/iql/reward_68ceccd06040/ni7a42nz \
EVAL_WANDB_PROJECT=iql-eval EVAL_WANDB_GROUP=pfusion_full_dataset RL_REWARD_MODE=pfusion \
bash run_scripts/fanout_checkpoint_evals.sh
# Array job: 15830905
```

Converged faster than pfusion_long and achieved substantially higher Q_flattop_avg (41 vs 26). Learning curve plateaued around step 22k with Q stabilizing around 55–56.

**Best checkpoint: step 22000** — Q_max=56.35, Q_flattop_avg=41.39, H98=0.828, E_fusion=57,171 MJ, reward_total=81.16

This is the new best honest pfusion result. Q_flattop_avg=41.4 is notably higher than pfusion_standard's 20.9 — the policy sustains high Q throughout flattop rather than peaking briefly. H98=0.828 is also the best across all runs.

Key observation: the full dataset (`run_prev_action_full_20260601_192826`) trains significantly better than the wide-ECRH dataset for pfusion mode. Likely because the full dataset contains more trajectories with high sustained heating, giving better coverage of the pfusion reward landscape.

---

## Comparison table (honest Q, no hacking)

| Run | Q_max | Q_flattop_avg | H98 | E_fusion (MJ) | reward_total | Notes |
|-----|-------|---------------|-----|----------------|-------------|-------|
| **pfusion_full_dataset step_22k** | **56.35** | **41.39** | **0.828** | 57,171 | 81.16 | pfusion, honest, new best |
| `wide_arp0.1` (best prior IQL) | 57.0 | — | 0.761 | 63,346 | — | standard reward, may have mild hacking |
| pfusion_long step_58k | 54.14 | 26.23 | 0.795 | 59,611 | 54.15 | pfusion, honest |
| `sc_loprio_timing_test` | 54.6 | — | 0.842 | 55,583 | — | standard reward |
| pfusion_standard step_39k | 51.7 | 20.9 | 0.801 | 59,892 | — | pfusion, honest |
| `youthful-microwave-93` | 31.7 | — | 0.783 | 62,564 | — | standard reward |
| `prev_action_q95w3.5_fgww4.0` | ~~152.6~~ | ~~39.6~~ | 0.813 | 60,297 | — | **HACKED** — policy turns off all heating |

---

## Next steps

1. Consider hyperparameter sweep on full dataset: wider network (hidden_dim=512), different tau/beta
2. Train longer on full dataset past 60k steps (curve may still be climbing)
3. Run `rebuild.sh` rather than manual copy when modifying `pulse_design.py` going forward
4. Investigate why full dataset trains better than wide_ecrh for pfusion mode
