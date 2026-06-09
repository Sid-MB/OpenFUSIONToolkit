# ITER TokaMaker-TORAX RL — Progress

## Status: Active (2026-06-08)

Offline RL pipeline for learning plasma control policies on the ITER scenario using IQL trained on TORAX simulation trajectories.

---

## Current best honest result (pfusion mode, no reward hacking)

**pfusion_standard step_39k**: Q_max=51.7, Q_flattop_avg=20.9, H98=0.801, E_fusion=59,892 MJ
W&B: `iql-eval` group `pfusion_standard`
Artifacts: `out/iql/reward_68ceccd06040/2tjm9kx1/checkpoints/checkpoint_step_39000.pt`

Two 60k-step runs currently training (jobs 15830578, 15830579) — expected to push Q_max above 51.7.

---

## Milestones

| Date | Event |
|------|-------|
| 2026-05-25 | Initial dataset collection (grid_51, maxloop=2) |
| 2026-05-27–28 | Large CPU array collection (1000 trajectories) |
| 2026-05-30–31 | IQL training on prev_action datasets; first reward_config sweeps |
| 2026-06-01–02 | Algorithm comparison (IQL vs BC vs CQL vs TD3+BC); BC competitive with IQL median |
| 2026-06-02 | Best Q_max=57 (`wide_arp0.1`); discovered reward hacking in `prev_action_q95w3.5_fgww4.0` run (Q_max=152, fake) |
| 2026-06-08 | Implemented pfusion reward mode to fix Q hacking; pfusion_standard achieves honest Q_max=51.7 |

---

## Key findings

- **Q reward hacking**: `log(Q+1)` reward is unbounded — policy learns to cut all heating → P_aux→0 → Q→∞. Fixed with `reward_mode='pfusion'`: `log(Q*P_aux_MW+1)` gives zero reward when heating is off.
- **BC competitive**: BC (pure imitation) matches IQL median (reward_total ~37) and produces physically sensible action trajectories.
- **IQL upside**: best IQL seed (43.95) beats BC (37.18) but variance is high (22–44 across seeds).
- **CQL/TD3+BC degenerate**: both collapse ECRH to zero mid-pulse (aux starvation hacking even with CQL conservatism).

---

## Active datasets

| Dataset | Trajectories | Notes |
|---------|-------------|-------|
| `run_prev_action_full_20260601_192826` | 1000 | Standard ECRH range, best H98 runs |
| `run_prev_action_wide_ecrh_20260602_000037` | 1000 | Wide ECRH sampling, used for most sweeps |
| Both above → `reward_variants/reward_68ceccd06040` | — | pfusion reweight of both |

---

## Research notes

- [2026-06-08 pfusion reward mode](research-notes/2026-06-08-pfusion-reward-mode.md)
