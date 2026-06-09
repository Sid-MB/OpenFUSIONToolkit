# Algorithm Analysis Notes

## Why BC outperforms CQL and TD3+BC

This problem has a persistent reward-hacking failure mode: setting auxiliary heating (ECRH) to zero during the pulse eliminates the `tx_P_aux_total` contribution but the plasma can still produce fusion reward through NBI alone — and the reward function's `step_reward_weight` penalizes total power consumption indirectly. Both Q-learning algorithms (CQL, TD3+BC) find this mode and converge to ECRH=0.

BC is immune because it has no Q-learning: it just imitates the dataset distribution, which includes diverse ECRH usage (the wide-ECRH dataset covers 5–35 MW). This makes BC a natural sanity-check: if BC ≈ IQL, the Q-learning is not adding value over imitation.

## Why IQL's best beats BC

IQL can genuinely improve over dataset actions by combining:
1. Value-guided actor updates (the advantage-weighted regression step)
2. Per-trajectory quality weighting (high-β penalizes low-advantage actions)

When IQL gets a good seed (e.g., d52e9z6h, reward=43.95), it learns to maintain NBI at moderate levels during the flattop rather than ramping all the way down, producing Q_flattop_avg=14.11 vs BC's 11.17. The spread (22–44) suggests the IQL optimization landscape is multi-modal.

## CQL over-conservatism hypothesis

In the wide-ECRH dataset, ~30% of trajectories use high ECRH (15–35 MW) — this is non-negligible. Despite this, CQL's final policy collapses to ECRH=0. A possible explanation: the CQL penalty `E[logsumexp Q(s,a)] - E_data[Q(s,a)]` is evaluated over random uniform action samples. For 2D action space (ECRH, NBI), uniform samples mostly land in the interior while the dataset concentrates mass near the boundaries (ramping trajectories tend to start/end at limits). The logsumexp term may be systematically dominated by mid-range ECRH states, creating a gradient that pushes Q values there down — and the actor follows.

Could try: increase `CQL_ALPHA` (more conservative), or switch to CQL with in-sample action estimation instead of random uniform.

## What to try next

- **IQL with checkpoint selection**: the best IQL run (43.95) may not be the best checkpoint — `fanout_checkpoint_evals.sh` exists for this. Best checkpoint likely earlier than step 40k given the plateau at ~15-20k.
- **BC as warm-start for IQL**: initialize IQL actor from BC weights, then fine-tune with Q-learning. The BC actor avoids the ECRH-collapse attractor.
- **TD3+BC with higher alpha**: current α=2.5 (default). Higher α would weight BC term more relative to policy gradient — might prevent ECRH collapse.
- **CQL with in-sample actions**: replace uniform random CQL penalty with dataset-sampled actions — closer to the original CQL paper for discrete-like problems.
