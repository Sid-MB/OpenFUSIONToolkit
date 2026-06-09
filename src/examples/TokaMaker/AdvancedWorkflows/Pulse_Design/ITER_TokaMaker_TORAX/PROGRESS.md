# ITER TokaMaker-TORAX RL — Progress

## Status: Active (2026-06-08)

Offline RL pipeline for learning plasma control policies on the ITER scenario using IQL trained on TORAX simulation trajectories.

---

## Current best honest result (pfusion mode, 2D heating-only)

**pfusion_standard step_39k**: Q_max=51.7, Q_flattop_avg=20.9, H98=0.801, E_fusion=59,892 MJ
W&B: `iql-eval` group `pfusion_standard`
Artifacts: `out/iql/reward_68ceccd06040/2tjm9kx1/checkpoints/checkpoint_step_39000.pt`

Two additional 60k-step 2D runs (jobs 15830578, 15830579) training on `reward_68ceccd06040` variant of `run_prev_action_full_20260601_192826` and `run_prev_action_wide_ecrh_20260602_000037` — expected to push Q_max above 51.7.

---

## Pellet fueling extension (3D action space) — in progress as of 2026-06-08

Action space extended from 2D `[ecrh_W, nbi_W]` to 3D `[ecrh_W, nbi_W, pellet_S_total]`. The agent now controls pellet fueling rate (particles/s) at each of the 21 decision times (t=80–480s, every 20s), applied with a 20s knot offset. Pellet is fixed off before t=90s, ramped to a baseline 5e21/s at t=90, agent-controlled from t=100–500, then shut off at t=520.

**Key implementation notes:**
- `collect_trajectories_delta.py`: LHS sampling in 3D (`PELLET_MIN=0`, `PELLET_MAX=1e22`, `pellet_delta_max=2e21`); `build_pellet_schedule()` translates action rows to per-timestep schedules.
- `pulse_design.py` (`src/python/OpenFUSIONToolkit/TokaMaker/`): `_load_rl_actor` accepts `action_dim ∈ {2, 3}`; `_merge_rl_pellet_schedule` blends agent knots with fixed baseline schedule; `residual_prev_action` mode blocked for 3D (use `absolute`). Requires `rebuild.sh` after edits.
- `RL_STATE_KEYS` unchanged — density signals (`fgw_n_e_line_avg`, `n_e_volume_avg`, `n_e_peaking`) already in state; pellet not added to observation.

**Active pellet runs:**

| Job | Dataset | Reward | W&B run | Status |
|-----|---------|--------|---------|--------|
| 15830880 (eval) | `run_prev_action_pellet_full_20260603_0956` | default (fgw=2.0, q95=1.2) | x0snq72a `pellet_absolute_default_reward` | Eval running (step_19000 ckpt, max_loop=2) |
| 15830879 (train) | `run_prev_action_pellet_recollect_20260608_1413/reward_variants/reward_50f8007cd7fe` | strong (fgw=4.0, q95=3.5) | `pellet_absolute_strong_reward` | Queued |

`pellet_absolute_default_reward` (x0snq72a): trained 20k steps on `run_prev_action_pellet_full_20260603_0956`. Training completed; actor eval in progress on step_19000 checkpoint.

`pellet_absolute_strong_reward`: trains on `reward_50f8007cd7fe` variant of `run_prev_action_pellet_recollect_20260608_1413` (1000 traj, `SAVE_STATS_FOR_REWARD_RECALC=1`). Strong reward matches best prior 2D run (`w0gqz7fd`): fgw_penalty=4.0, q95_penalty=3.5.

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
| 2026-06-08 | Extended action space to 3D (ECRH + NBI + pellet fueling); collected two pellet datasets; two IQL pellet runs in progress |

---

## Key findings

- **Q reward hacking**: `log(Q+1)` reward is unbounded — policy learns to cut all heating → P_aux→0 → Q→∞. Fixed with `reward_mode='pfusion'`: `log(Q*P_aux_MW+1)` gives zero reward when heating is off.
- **BC competitive**: BC (pure imitation) matches IQL median (reward_total ~37) and produces physically sensible action trajectories.
- **IQL upside**: best IQL seed (43.95) beats BC (37.18) but variance is high (22–44 across seeds).
- **CQL/TD3+BC degenerate**: both collapse ECRH to zero mid-pulse (aux starvation hacking even with CQL conservatism).
- **Pellet action mode**: `residual_prev_action` not supported for 3D actors; must use `absolute` mode. The prior best 2D run used `residual_prev_action`, so pellet runs are not directly comparable on action-mode axis.
- **rebuild.sh required**: any edits to `src/python/OpenFUSIONToolkit/TokaMaker/pulse_design.py` must be followed by `cd /juice2/scr2/siddharth/OpenFUSIONToolkit && bash rebuild.sh` before Slurm jobs will pick them up (jobs load from `install_release/`, not the source tree).

---

## Active datasets

| Dataset | Trajectories | Notes |
|---------|-------------|-------|
| `run_prev_action_full_20260601_192826` | 1000 | Standard ECRH range, best H98 runs |
| `run_prev_action_wide_ecrh_20260602_000037` | 1000 | Wide ECRH sampling, used for most sweeps |
| Both above → `reward_variants/reward_68ceccd06040` | — | pfusion reweight |
| Both above → `reward_variants/reward_eeaff1f2291a` | — | strong penalty reweight (fgw=4.0, q95=3.5) |
| `run_prev_action_pellet_full_20260603_0956` | 1000 | 3D action (pellet); no reward_recalc_stats |
| `run_prev_action_pellet_recollect_20260608_1413` | 1000 | 3D action (pellet); has reward_recalc_stats |
| Above → `reward_variants/reward_50f8007cd7fe` | — | strong penalty reweight of pellet recollect |

---

## Research notes

- [2026-06-08 pfusion reward mode](research-notes/2026-06-08-pfusion-reward-mode.md)
