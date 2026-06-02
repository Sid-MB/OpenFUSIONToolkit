# Results summary — `run_smb_prev_action_full_20260601_192827`

IQL actor trained on the ITER TokaMaker+TORAX closed-loop dataset (`prev_action`
observation mode), with a closed-loop TORAX evaluation of the final actor and a
per-checkpoint eval fanout. Generated 2026-06-01.

## Run identity
- **W&B / run id:** `8vdqyziz`
- **Observation mode:** `prev_action` (actor-conditioned; 49-dim observation)
- **Action mode:** `residual_prev_action` (auto-derived from observation mode)
- **Final weights:** `8vdqyziz/iql_weights.pt`

## Dataset (collection)
- **Source:** `run_prev_action_full_20260601_192826` (on `/juice2/scr2`)
- **Trajectories:** 1000 · **seed** 2455387185 · **grid_size** 51 · **max_loop** 2
- **Observation mode:** `prev_action` (matches training)
- **Reward config** (baked at collection, recomputable from saved stats):
  `beta_n_max=2.8 (w=1.0)`, `fgw_max=0.85 (w=2.0)`, `q95_min=3.0 (w=1.2)`,
  `flux_weight=0.012`, `q_flattop_weight=1.0`, `step_reward_weight=1.0`

## Training (IQL)
| param | value | param | value |
|---|---|---|---|
| num_steps | 40000 | batch_size | 128 |
| lr | 1e-4 | hidden_dim | 256 |
| tau | 0.7 | beta | 3 |
| gamma | 0.99 | action_rate_penalty | 0.01 |
| train_seed | 2771703067 | checkpoint_interval | 1000 |

39 checkpoints saved (`checkpoint_step_1000.pt` … `checkpoint_step_39000.pt`).

## Final actor closed-loop eval
Full closed-loop TORAX rollout of `iql_weights.pt` (21 decisions).
**Reward config match: ✅ True** (eval reward config == training/collection config).

| metric | value |
|---|---|
| **Q_flattop_avg** | **9.17** |
| Q_max | 10.00 (at t≈263 s) |
| P_fusion_max | 154.0 MW |
| E_fusion_total | 60.6 GJ |
| reward_mean | 1.47 |
| reward_max | 11.04 |
| β_N max | 1.10 |
| q95_min | 2.79 |
| f_GW_max | 0.939 |
| H98_flattop_avg | 0.766 |
| nbi / ecrh saturation | 9.5% / 0% |
| action saturation rate | 4.8% |

Artifacts: `8vdqyziz/actor_eval/` (`actor_eval_summary.json`,
`actor_eval_actions.json`, `actor_eval_reward_config.json`, plots/movie, TORAX logs).

## Per-checkpoint eval fanout (in progress)
A closed-loop eval was fanned out across all 39 checkpoints (Slurm jobs
`15736166`–`15736204`, `john` CPU). Per-step results land in:
```
8vdqyziz/checkpoint_evals/step_<N>/actor_eval_summary.json
```
**Status at generation time: 0/39 complete (jobs queued).** Once they finish,
this section can be regenerated into a Q_flattop_avg / reward-vs-step table to
show how closed-loop performance evolves over training.

## Notes
- This is the `prev_action` pipeline; the companion ablation is
  `run_plasma_only_full_20260601_193047` (`plasma_only`, 47-dim, `action_mode=absolute`),
  whose final eval gave `Q_flattop_avg ≈ 4.72` — i.e. the action-conditioned
  `prev_action` actor reaches markedly higher closed-loop Q here (9.17 vs 4.72).
- Closed-loop eval requires the observation-mode-aware RL inference in
  `pulse_design.py` (reads `observation_mode` from the checkpoint); both the
  `jag` and `john` OFT builds carry this fix.
