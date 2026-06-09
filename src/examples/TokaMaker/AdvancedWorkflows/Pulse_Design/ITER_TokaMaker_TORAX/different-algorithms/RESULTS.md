# Algorithm Comparison Results

Dataset: `run_prev_action_wide_ecrh_20260602_000037` (1000 trajectories, prev_action observation mode, wide ECRH sampling)
All runs: 40k steps, residual_prev_action action mode, action_rate_penalty=0.01, IQL_CRITIC_LAYERNORM=true, weight_decay=1e-4, lr=1e-4, hidden_dim=256

---

## Summary Table

| Algorithm | Run ID | W&B | reward_total | Q_flattop_avg | H98_flattop | q95_min | Notes |
|-----------|--------|-----|-------------|---------------|-------------|---------|-------|
| IQL (best) | d52e9z6h | [d52e9z6h](https://wandb.ai/siddharth-stanford/iql-training/runs/d52e9z6h) | **43.95** | 14.11 | 0.783 | 2.846 | Baseline config, lucky seed |
| IQL (median) | to5oswfw | [to5oswfw](https://wandb.ai/siddharth-stanford/iql-training/runs/to5oswfw) | 36.60 | 11.49 | 0.761 | 2.797 | action_rate_penalty=0.1 |
| IQL (worst) | ms6z7gz8 | [ms6z7gz8](https://wandb.ai/siddharth-stanford/iql-training/runs/ms6z7gz8) | 22.45 | 9.61 | 0.754 | 2.761 | Baseline config, unlucky seed |
| **BC** | d7w93xr7 | [wide_LN_wd_bc](https://wandb.ai/siddharth-stanford/iql-training/runs/d7w93xr7) | **37.18** | 11.17 | 0.771 | 2.827 | Competitive with IQL median |
| CQL | 58znl1w7 | [wide_LN_wd_cql_v2](https://wandb.ai/siddharth-stanford/iql-training/runs/58znl1w7) | 26.20 | 5.41 | 0.714 | 2.702 | Degenerate: ECRH=0 throughout |
| TD3+BC | py0nd5nm | [wide_LN_wd_td3bc](https://wandb.ai/siddharth-stanford/iql-training/runs/py0nd5nm) | 25.07 | 5.81 | 0.721 | 2.820 | Degenerate: ECRH→0 rapidly |

IQL full seed range: 22.45 – 43.95 across 9 seeds.

---

## Per-Algorithm Details

### BC (`d7w93xr7`) — reward_total = 37.18
- **Action pattern**: ECRH ramps smoothly 31→0 MW over ~200s, NBI ramps smoothly 33→7 MW — most physically sensible trajectory of all algorithms tested.
- **Q_flattop_avg = 11.17** — matches IQL median.
- **H98 = 0.771** — slightly above IQL median.
- **q95_min = 2.827** — close to IQL, above the 3.0 threshold but still penalized.
- No reward hacking: pure imitation of dataset actions, no Q-learning means no exploitation of reward signal.

### CQL (`58znl1w7`) — reward_total = 26.20
- **Action pattern**: ECRH = 0 throughout entire pulse; NBI spikes briefly then saturates at 33 MW.
- Severely degenerate — collapsed to NBI-only policy, ECRH never used.
- Q_flattop_avg = 5.41 and H98 = 0.714 — worst of all algorithms.
- CQL's conservatism likely over-suppressed the ECRH Q-values (ECRH actions are rarer in the dataset — wide-ECRH dataset helps but wasn't enough).
- Second run (`60lkxw65`) was incomplete (no actor_eval).

### TD3+BC (`py0nd5nm`) — reward_total = 25.07
- **Action pattern**: ECRH starts at 27 MW but drops to 0 by t=200s and stays there; NBI at max then slowly ramps down to 0 at end.
- Less degenerate than CQL but still sub-optimal — ECRH collapses mid-pulse.
- Q_flattop_avg = 5.81, H98 = 0.721.
- The BC regularization term in TD3+BC pulled actor toward dataset mean but the Q gradient still drove ECRH to zero (reward hacking via aux starvation).
- Second run (`ckb5843g`) was incomplete.

### IQL (reference, 9 runs — mix of seeds and hyperparameter sweeps)
Baseline config: beta=3, tau=0.7, action_rate_penalty=0.01, LN=false, wd=0.

| Run ID | W&B | reward_total | Q_flattop_avg | H98 | q95_min | Diff from baseline |
|--------|-----|-------------|---------------|-----|---------|-------------------|
| d52e9z6h | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/d52e9z6h) | 43.95 | 14.11 | 0.783 | 2.846 | baseline |
| to5oswfw | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/to5oswfw) | 36.60 | 11.49 | 0.761 | 2.797 | action_rate_penalty=0.1 |
| d6syj61p | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/d6syj61p) | 36.41 | 10.27 | 0.771 | 2.779 | LN=true |
| use30zsi | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/use30zsi) | 35.98 | 10.50 | 0.762 | 2.829 | tau=0.9 |
| 4924ts4r | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/4924ts4r) | 30.79 | 8.66 | 0.745 | 2.839 | LN=true, wd=1e-4 |
| f403dor9 | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/f403dor9) | 29.68 | 8.08 | 0.748 | 2.752 | action_rate_penalty=0.3 |
| 7kx8qduz | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/7kx8qduz) | 27.36 | 7.04 | 0.739 | 2.831 | wd=1e-4 |
| w31ztxm4 | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/w31ztxm4) | 27.40 | 6.97 | 0.738 | 2.833 | beta=10 |
| ms6z7gz8 | [link](https://wandb.ai/siddharth-stanford/iql-training/runs/ms6z7gz8) | 22.45 | 9.61 | 0.754 | 2.761 | baseline (different seed) |

Note: d52e9z6h (best, 43.95) and ms6z7gz8 (worst, 22.45) have identical hyperparameters — the 2x spread between them is pure seed variance.

---

## Key Takeaways

1. **BC is surprisingly competitive**: reward_total=37.18 beats all non-IQL algorithms and lands at IQL median. For this problem, imitating the dataset well is nearly as good as full offline RL (median IQL).
2. **CQL and TD3+BC both exhibit ECRH collapse**: the reward-hacking via auxiliary starvation (ECRH→0) persists despite CQL's conservatism. CQL's penalty may be suppressing OOD Q values but the in-distribution low-ECRH actions still get high Q.
3. **IQL's upside beats BC**: the best IQL seed (43.95) is well above BC (37.18), suggesting IQL can find genuinely better policies but is high-variance across seeds.
4. **BC action quality is qualitatively superior**: smooth ramp-down of both ECRH and NBI is physically meaningful; CQL/TD3+BC produce discontinuous or zero-ECRH trajectories.

---

## Subsequent Work (post algorithm comparison)

### Reward hacking fix — pfusion mode (2026-06-08)

All runs above used the standard reward including `log(Q+1)`. The best IQL run achieving Q_max=57 (`wide_arp0.1`) was discovered to be reward-hacked: policy cuts all heating → P_aux→0 → Q→∞ in the simulation. Q_max=57 is fake.

Fix: `reward_mode='pfusion'` replaces `log(Q+1)` with `log(Q * P_aux_MW + 1)`, which goes to zero when heating is off. Honest pfusion result:

**pfusion_standard (W&B: `2tjm9kx1`)**: Q_max=51.7, Q_flattop_avg=20.9, H98=0.801
Dataset: `run_prev_action_full_20260601_192826` → `reward_variants/reward_68ceccd06040` (pfusion reweight)
Config: IQL, beta=3, tau=0.7, action_rate_penalty=0.01, num_steps=60k

Best prior honest 2D run with strong penalty (`w0gqz7fd`, reward_eeaff1f2291a): fgw_penalty=4.0, q95_penalty=3.5, residual_prev_action.

### Pellet fueling as 3rd action dimension (2026-06-08)

Action space extended to 3D: `[ecrh_W, nbi_W, pellet_S_total]`. Agent controls pellet fueling rate at each of the 21 decision times. Two IQL pellet runs in progress (see `PROGRESS.md` for details). Results pending.
