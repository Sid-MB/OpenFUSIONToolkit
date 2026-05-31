# Results Analysis: Smoke Live-Render IQL Eval

This note compares the smoke IQL actor evaluation against the closest available
grid-search baseline for the same dataset family.

## Eval Run

- Output dir: `out/iql_eval/smoke_live_render_docfix_20260531/`
- Actor checkpoint:
  `out/iql/run_prev_action_smoke_20260530_222521/vwgpce31/iql_weights.pt`
- Eval status: `success`

## Actor Eval Summary

Key metrics from `actor_eval_summary.json`:

- `reward_total`: `15.4371`
- `reward_mean`: `0.7351`
- `n_decisions`: `21`
- `action_saturation_rate`: `0.0`
- `nbi_saturation_rate`: `0.0`
- `ecrh_saturation_rate`: `0.0`
- `Q_max`: `5.0992`
- `Q_flattop_avg`: `4.8856`
- `E_fusion_total_MJ`: `65379.5`
- `beta_N_max`: `1.1212`
- `H98_max`: `0.7452`
- `H98_flattop_avg`: `0.7107`
- `P_fusion_max_MW`: `167.87`

Action trace summary:

- NBI stays high throughout the rollout.
- ECRH ramps down steadily and reaches zero in later intervals.
- Actions are smooth; the mean absolute step change is modest relative to the
  actuator ranges.
- There is no hard saturation at the actuator bounds.

## Grid-Search Baseline

Closest comparable baseline found:

- Dataset: `out/grid_search/rl_dataset_delta_sampling_maxloop=2_grid_51_raw_comparable_20260528`
- Best trajectory: `rl_dataset_delta_sampling_maxloop=2_grid_51/trajectory_0148.json`
- `return_sum`: `10.1360`
- `Q_max`: `5.3443`
- `Q_flattop_avg`: `4.5904`
- `E_fusion_total_MJ`: `72456.8`
- `beta_N_max`: `1.1671`
- `H98_max`: `0.7599`
- `P_fusion_max_MW`: `194.63`

## Comparison

The actor is stable and produces a clean closed-loop rollout, but it does not
beat the best grid-search baseline on the main plasma-performance metrics:

- Lower `Q_max`
- Lower `E_fusion_total_MJ`
- Lower `P_fusion_max_MW`
- Slightly lower `beta_N_max` and `H98_max`

What the actor does well:

- No actuator saturation
- Smooth action evolution
- Reasonable closed-loop performance with a simple, repeatable policy

What the actor does not yet do:

- It does not reach the strongest baseline trajectory on peak performance.
- It appears to favor a conservative control strategy, with heavy NBI usage
  and a gradual reduction of ECRH.

## Interpretation

The model is usable, but it is likely optimizing for a safe, smooth regime
rather than the most aggressive high-performance regime.

Likely next improvements:

1. Compare against the top few grid-search baselines, not just the best one.
2. Use closed-loop checkpoint selection during training so rollout quality is
   part of model selection.
3. If higher peak performance matters, revisit the action parameterization or
   reward shaping so the actor is not biased toward a conservative NBI-heavy
   policy.
4. Run an ablation without previous-action feedback to test whether the smooth
   policy is learned control or self-reinforcing behavior.

