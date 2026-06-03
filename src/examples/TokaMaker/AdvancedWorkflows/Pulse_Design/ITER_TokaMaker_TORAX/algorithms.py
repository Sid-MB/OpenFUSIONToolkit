"""
Offline RL algorithms sharing the IQL interface (.update(batch), .actor, etc.).

All algorithms expose the same contract as IQL so train_iql() / train_from_config()
can swap them in by setting ALGORITHM=<name>:
  - iql         original IQL (default, handled in IQL.py)
  - td3bc        TD3+BC: TD3 actor with BC regularisation
  - cql          CQL: conservative Q-learning (SAC-style actor + CQL penalty)
  - bc           Behavioural Cloning: no Q-learning, pure imitation

Each class:
  - __init__  same keyword signature subset as IQL (action_max, state_dim, action_dim,
              hidden_dim, lr, gamma, device, action_mode, action_context_indices,
              action_rate_penalty, weight_decay)
  - update(batch) -> dict of loggable scalars
  - set_normalizers(state_mean, state_std)
  - .actor, .state_dim, .action_dim, .device  attributes expected by train loop
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from IQL import Actor, QNetwork, _build_mlp, IQL_CRITIC_LAYERNORM, IQL_WEIGHT_DECAY


# ───────────────────────────── TD3+BC ─────────────────────────────────────────

class TD3BC:
    """
    TD3 + Behavioural Cloning (Fujimoto & Gu, 2021).
    Actor loss = -λ·Q(s,π) + MSE(π(s), a_data)
    λ = α / mean|Q(s,a_data)|  (auto-normalises the scale)
    Conservative by design: the BC term keeps the actor close to the dataset.
    Hyper:
      TD3BC_ALPHA  (env, default 2.5)  — BC weight relative to policy gradient
    """

    def __init__(
        self,
        action_max,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device=None,
        action_mode: str = "absolute",
        action_context_indices=None,
        action_rate_penalty: float = 0.0,
        weight_decay: float = IQL_WEIGHT_DECAY,
        **_,
    ):
        self.device = torch.device(device or "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.action_rate_penalty = float(action_rate_penalty)

        try:
            self.alpha = float(os.environ.get("TD3BC_ALPHA", "") or 2.5)
        except ValueError:
            self.alpha = 2.5

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor = Actor(
            action_max, state_dim, action_dim, hidden_dim,
            action_mode=action_mode,
            action_context_indices=action_context_indices,
        ).to(self.device)

        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr, weight_decay=weight_decay)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr, weight_decay=weight_decay)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self._state_mean_tensor = self._state_std_tensor = None

    def set_normalizers(self, state_mean, state_std):
        self._state_mean_tensor = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self._state_std_tensor = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)

    def _soft_update(self, polyak=0.995):
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(polyak * tp.data + (1 - polyak) * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(polyak * tp.data + (1 - polyak) * p.data)

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch
        states = states.float().to(self.device, non_blocking=True)
        actions = actions.float().to(self.device, non_blocking=True)
        next_states = next_states.float().to(self.device, non_blocking=True)
        rewards = rewards.float().to(self.device, non_blocking=True)
        dones = dones.float().to(self.device, non_blocking=True)

        # ── Q update (TD3 style: target uses next greedy action + clipped double Q) ──
        with torch.no_grad():
            next_a = self.actor(next_states)
            noise = (torch.randn_like(next_a) * 0.2).clamp(-0.5, 0.5)
            next_a = (next_a + noise).clamp(
                torch.zeros_like(next_a),
                torch.ones_like(next_a),
            )
            q_next = torch.min(
                self.q1_target(next_states, next_a),
                self.q2_target(next_states, next_a),
            )
            target_q = rewards + self.gamma * (1 - dones) * q_next

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_opt.zero_grad(); q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # ── Actor update: BC + policy gradient ──
        action_pred = self.actor(states)

        action_target = actions
        if self.actor.action_mode == "residual_prev_action" and self.actor.action_context_indices:
            assert self._state_mean_tensor is not None and self._state_std_tensor is not None
            prev_action = torch.stack(
                [(states[:, idx] * self._state_std_tensor[idx] + self._state_mean_tensor[idx])
                 / self.actor.action_max[i]
                 for i, idx in enumerate(self.actor.action_context_indices)],
                dim=-1,
            )
            action_target = actions - prev_action

        bc_loss = F.mse_loss(action_pred, action_target)

        q_val = self.q1(states, action_pred)
        lam = self.alpha / (q_val.abs().mean().detach() + 1e-8)
        actor_loss = -lam * q_val.mean() + bc_loss

        if self.action_rate_penalty > 0 and self.actor.action_mode == "residual_prev_action":
            actor_loss = actor_loss + self.action_rate_penalty * (action_pred ** 2).mean()

        self.actor_opt.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self._soft_update()

        return {
            "train/q1_loss": q1_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/bc_loss": bc_loss.item(),
            "train/lam": lam.item(),
        }


# ───────────────────────────── CQL ────────────────────────────────────────────

class CQL:
    """
    Conservative Q-Learning (Kumar et al., 2020), SAC-style actor.
    Q loss adds CQL penalty: E[log-sum-exp Q(s,a)] − E_data[Q(s,a_data)]
    Forces Q to be pessimistic on OOD actions → directly inhibits the
    aux-starvation reward-hack (aux→0 is OOD; its Q is suppressed).
    Hyper:
      CQL_ALPHA   (env, default 1.0)  — CQL penalty weight
      CQL_TEMP    (env, default 1.0)  — temperature for logsumexp sampling
    """

    def __init__(
        self,
        action_max,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device=None,
        action_mode: str = "absolute",
        action_context_indices=None,
        action_rate_penalty: float = 0.0,
        weight_decay: float = IQL_WEIGHT_DECAY,
        tau: float = 0.7,   # not used by CQL, accepted for compat
        **_,
    ):
        self.device = torch.device(device or "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.action_rate_penalty = float(action_rate_penalty)
        self.action_max_np = np.asarray(action_max, dtype=np.float64)

        try:
            self.cql_alpha = float(os.environ.get("CQL_ALPHA", "") or 1.0)
            self.cql_temp = float(os.environ.get("CQL_TEMP", "") or 1.0)
        except ValueError:
            self.cql_alpha, self.cql_temp = 1.0, 1.0

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # SAC actor: use standard Actor (2-output mu net, compatible with pulse_design eval)
        # plus a separate _log_std_net for stochastic sampling during training.
        # actor.net (2-output) is trained as the mean; _log_std_net (2-output) produces log_std.
        self._log_std_min, self._log_std_max = -5.0, 2.0
        self.actor = Actor(
            action_max, state_dim, action_dim, hidden_dim,
            action_mode=action_mode,
            action_context_indices=action_context_indices,
        ).to(self.device)
        self._log_std_net = _build_mlp(state_dim, hidden_dim, action_dim, IQL_CRITIC_LAYERNORM).to(self.device)
        # log-alpha (auto-tuned entropy)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.target_entropy = -float(action_dim)

        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr, weight_decay=weight_decay)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr, weight_decay=weight_decay)
        self.actor_opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self._log_std_net.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self._state_mean_tensor = self._state_std_tensor = None

    def set_normalizers(self, state_mean, state_std):
        self._state_mean_tensor = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self._state_std_tensor = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)

    def _sample_action(self, states):
        """Reparametrised sample from the SAC Gaussian actor."""
        mu = self.actor.net(states)   # standard 2-output mean net
        log_std = self._log_std_net(states).clamp(self._log_std_min, self._log_std_max)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        raw = mu + eps * std
        action = torch.sigmoid(raw)
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(raw)
            - torch.log(action * (1 - action) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob

    def _soft_update(self, polyak=0.995):
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(polyak * tp.data + (1 - polyak) * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(polyak * tp.data + (1 - polyak) * p.data)

    def _cql_penalty(self, q_net, states, batch_size):
        """Estimate E[log-sum-exp Q(s,a)] via uniform random sampling."""
        n_samples = 10
        rand_actions = torch.rand(batch_size, n_samples, self.action_dim, device=self.device)
        s_expand = states.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.state_dim)
        a_flat = rand_actions.reshape(-1, self.action_dim)
        q_rand = q_net(s_expand, a_flat).reshape(batch_size, n_samples, 1)
        logsumexp_q = torch.logsumexp(q_rand / self.cql_temp, dim=1) * self.cql_temp - np.log(n_samples)
        return logsumexp_q

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch
        states = states.float().to(self.device, non_blocking=True)
        actions = actions.float().to(self.device, non_blocking=True)
        next_states = next_states.float().to(self.device, non_blocking=True)
        rewards = rewards.float().to(self.device, non_blocking=True)
        dones = dones.float().to(self.device, non_blocking=True)
        batch_size = states.shape[0]

        with torch.no_grad():
            next_a, next_log_prob = self._sample_action(next_states)
            alpha = self.log_alpha.exp().detach()
            q_next = torch.min(
                self.q1_target(next_states, next_a),
                self.q2_target(next_states, next_a),
            ) - alpha * next_log_prob
            target_q = rewards + self.gamma * (1 - dones) * q_next

        # ── Q update with CQL penalty ──
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        cql1 = (self._cql_penalty(self.q1, states, batch_size) - q1).mean()
        cql2 = (self._cql_penalty(self.q2, states, batch_size) - q2).mean()
        cql_loss = self.cql_alpha * (cql1 + cql2)

        q_loss = bellman_loss + cql_loss
        self.q1_opt.zero_grad(); self.q2_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q1_opt.step(); self.q2_opt.step()

        # ── Actor update ──
        new_a, log_prob = self._sample_action(states)
        alpha_detached = self.log_alpha.exp().detach()
        q_pi = torch.min(self.q1(states, new_a), self.q2(states, new_a))
        actor_loss = (alpha_detached * log_prob - q_pi).mean()
        if self.action_rate_penalty > 0:
            actor_loss = actor_loss + self.action_rate_penalty * (new_a ** 2).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self._log_std_net.parameters()), 1.0)
        self.actor_opt.step()

        # ── Alpha update: fresh sample, no graph conflict ──
        with torch.no_grad():
            _, log_prob_fresh = self._sample_action(states)
        alpha = self.log_alpha.exp()
        alpha_loss = -(alpha * (log_prob_fresh + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        self._soft_update()

        return {
            "train/bellman_loss": bellman_loss.item(),
            "train/cql_loss": cql_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/alpha": alpha.item(),
        }


# ───────────────────────────── BC ─────────────────────────────────────────────

class BC:
    """
    Behavioural Cloning — pure supervised learning on actions.
    No Q-learning, no reward. Immune to reward-hacking by construction.
    Useful as a lower-bound baseline: if BC matches IQL on real metrics,
    the reward signal is useless and the problem is purely data coverage.
    """

    def __init__(
        self,
        action_max,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        device=None,
        action_mode: str = "absolute",
        action_context_indices=None,
        action_rate_penalty: float = 0.0,
        weight_decay: float = IQL_WEIGHT_DECAY,
        **_,
    ):
        self.device = torch.device(device or "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_rate_penalty = float(action_rate_penalty)

        self.actor = Actor(
            action_max, state_dim, action_dim, hidden_dim,
            action_mode=action_mode,
            action_context_indices=action_context_indices,
        ).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self._state_mean_tensor = self._state_std_tensor = None

    def set_normalizers(self, state_mean, state_std):
        self._state_mean_tensor = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self._state_std_tensor = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch
        states = states.float().to(self.device, non_blocking=True)
        actions = actions.float().to(self.device, non_blocking=True)

        action_pred = self.actor(states)
        action_target = actions
        if self.actor.action_mode == "residual_prev_action" and self.actor.action_context_indices:
            assert self._state_mean_tensor is not None
            prev_action = torch.stack(
                [(states[:, idx] * self._state_std_tensor[idx] + self._state_mean_tensor[idx])
                 / self.actor.action_max[i]
                 for i, idx in enumerate(self.actor.action_context_indices)],
                dim=-1,
            )
            action_target = actions - prev_action

        bc_loss = F.mse_loss(action_pred, action_target)

        if self.action_rate_penalty > 0 and self.actor.action_mode == "residual_prev_action":
            bc_loss = bc_loss + self.action_rate_penalty * (action_pred ** 2).mean()

        self.actor_opt.zero_grad(); bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {"train/bc_loss": bc_loss.item()}


# ───────────────────────────── registry ───────────────────────────────────────

ALGORITHMS = {
    "td3bc": TD3BC,
    "cql": CQL,
    "bc": BC,
}


def make_algorithm(name: str, **kwargs):
    """Instantiate an algorithm by name. Raises KeyError for unknown names."""
    name = name.lower()
    if name not in ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm {name!r}. Available: {list(ALGORITHMS)}"
        )
    return ALGORITHMS[name](**kwargs)
