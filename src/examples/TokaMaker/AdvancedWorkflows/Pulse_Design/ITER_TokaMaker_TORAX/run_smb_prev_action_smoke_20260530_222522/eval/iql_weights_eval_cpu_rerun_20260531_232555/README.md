# Eval Summary

This directory contains a CPU actor-evaluation rerun of the checkpoint in
`run_smb_prev_action_smoke_20260530_222522/vwgpce31/iql_weights.pt` against the
current source reward defaults in `pulse_design.py`.

Why this run exists:
- The earlier eval artifact for this checkpoint was produced before the reward
  defaults were updated.
- We reran the eval to make the reward provenance explicit and to confirm that
  the new default `RLRewardConfig` is what this checkpoint is being judged
  against.
- The run was allowed to proceed with `ALLOW_MISMATCHED_REWARDS=1` because the
  checkpoint itself does not record reward provenance.

## Top-line results

- Job: `15712038`
- Status: `completed successfully`
- Total reward: `15.437109047908603`
- Reward config used for eval:
  - `q95_min = 3.0`
  - `beta_n_max = 2.8`
  - `fgw_max = 0.85`
  - `step_reward_weight = 1.0`
  - `q95_penalty_weight = 1.2`
  - `beta_n_penalty_weight = 1.0`
  - `fgw_penalty_weight = 2.0`
  - `q_flattop_weight = 1.0`
  - `flux_weight = 0.012`
- Reward provenance:
  - `reward_config_match = false`
  - `allow_mismatched_rewards = true`
  - `train_reward_config = unavailable for this older checkpoint`

## Closed-loop performance

The checkpoint still produces a strong closed-loop trajectory under the new
reward config:

- `Q_max = 5.0992`
- `Q_flattop_avg = 4.8856`
- `E_fusion_total_MJ = 65379.5`
- `H98_flattop_avg = 0.7107`
- `beta_N_max = 1.1212`
- `f_GW_max = 0.9446`
- `q95_min = 2.8558`
- `flux_consumed_Wb = 160.39`

Interpretation:
- The actor is not saturating either heating channel in this run.
- ECRH and NBI stay well below their maximum bounds in the action trace.
- The rollout reaches a useful high-performance regime, but it does not satisfy
  the q95 safety threshold in the current reward config (`q95_min = 3.0`), so
  the reward is being pulled down by that mismatch.
- The large final-step reward suggests the model reaches a strong final-state
  configuration by the end of the rollout even though the trajectory is not
  uniformly optimal under the new reward weights.

## Action behavior

The policy is smooth rather than bang-bang:

- `actor_eval/action_saturation_rate = 0.0`
- `actor_eval/nbi_saturation_rate = 0.0`
- `actor_eval/ecrh_saturation_rate = 0.0`
- `actor_eval/action_delta_abs_mean = 636401.05`

The actions drift gradually through the pulse rather than slamming against the
bounds. That is consistent with the residual-action parameterization and the
previous-action observation mode.

## What to inspect

- `actor_eval_summary.json` for the structured scalar summary.
- `actor_eval_reward_config.json` for the exact reward provenance.
- `actor_eval_actions.json` for the per-decision action trace.
- `artifacts/plot_scalars.png` for the high-level scalar curves.
- `artifacts/plot_profile_evolution_*.png` for the plasma profile evolution.
- `artifacts/plot_lcfs_evolution_*.png` for LCFS evolution across the pulse.
- `artifacts/movie.mp4` for the full rollout animation.

## Files in this directory

- `actor_eval_summary.json`
- `actor_eval_reward_config.json`
- `actor_eval_actions.json`
- `actor_eval_bundle.pkl`
- `artifacts/`
- `logs/`
- `tokamaker_torax_logs/`

## Note on provenance

This is a mismatched-reward rerun of an older checkpoint. It is useful as a
post-hoc benchmark, but it is not a clean apples-to-apples score against a
checkpoint trained under the current `RLRewardConfig`.
