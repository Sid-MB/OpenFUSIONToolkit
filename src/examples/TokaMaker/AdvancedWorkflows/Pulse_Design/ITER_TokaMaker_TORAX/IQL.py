import itertools
import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import cast

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

from dataloader import describe_dataset_with_replay_cache, load_d4rl_dataset, reward_config_to_dict
from log import get_logger

# Things to also look at: src/python/OpenFUSIONToolkit/TokaMaker/pulse_design.py [path is from repo root]. Make sure if you edit pulse_design.py that you run rebuild.sh to update the Python package.

app = modal.App("iql-training")
logger = get_logger(__name__)


def _random_seed32() -> int:
    return int.from_bytes(os.urandom(4), "little")

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}

image = modal.Image.debian_slim().pip_install(
    "modal", "torch", "numpy", "wandb"
)

class ReplayBuffer(Dataset):
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_states = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.next_states[idx], 
                self.rewards[idx], self.dones[idx])

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class Actor(nn.Module):
    def __init__(
        self,
        action_max,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_mode: str = "absolute",
        action_context_indices=None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.register_buffer("action_max", torch.as_tensor(action_max, dtype=torch.float32))
        self.action_mode = action_mode
        self.action_context_indices = tuple(action_context_indices or [])

    def forward(self, state):
        raw = self.net(state)
        if self.action_mode == "residual_prev_action":
            return torch.tanh(raw)
        return torch.sigmoid(raw)

    def act(self, state, prev_action=None):
        raw = self.net(state)
        if self.action_mode == "residual_prev_action":
            action_max = cast(torch.Tensor, self.action_max)
            delta = torch.tanh(raw) * action_max
            if prev_action is None:
                prev_action = torch.zeros_like(delta)
            return torch.maximum(torch.zeros_like(delta), torch.minimum(prev_action + delta, action_max))
        return torch.sigmoid(raw) * cast(torch.Tensor, self.action_max)

class IQL:
    def __init__(self, action_max, state_dim: int, action_dim: int, 
                 tau: float = 0.7, beta: float = 3.0, 
                 gamma: float = 0.99, lr: float = 1e-4,
                 hidden_dim: int = 256,
                 device=None,
                 action_mode: str = "absolute",
                 action_context_indices=None,
                 action_rate_penalty: float = 0.0):
        self.device = torch.device(device or "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.v = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.actor = Actor(
            action_max,
            state_dim,
            action_dim,
            hidden_dim,
            action_mode=action_mode,
            action_context_indices=action_context_indices,
        ).to(self.device)
        
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.action_rate_penalty = float(action_rate_penalty)
        self._state_mean_tensor = None
        self._state_std_tensor = None

    def set_normalizers(self, state_mean, state_std):
        self._state_mean_tensor = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self._state_std_tensor = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)

    def soft_update_targets(self):
        """Polyak averaging for target networks"""
        polyak = 0.995
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)

    def expectile_loss(self, diff, expectile=0.9):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch
        states = states.float().to(self.device, non_blocking=True)
        actions = actions.float().to(self.device, non_blocking=True)
        next_states = next_states.float().to(self.device, non_blocking=True)
        rewards = rewards.float().to(self.device, non_blocking=True)
        dones = dones.float().to(self.device, non_blocking=True)
        
        # Update V
        with torch.no_grad():
            q = torch.min(
                self.q1_target(states, actions),  # USE TARGET
                self.q2_target(states, actions)   # USE TARGET
            )
            
        v = self.v(states)
        
        v_loss = self.expectile_loss(q - v, self.tau).mean()
        
        self.v_opt.zero_grad()
        v_loss.backward()
        v_grad_norm = torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)
        self.v_opt.step()

        # Update Q
        with torch.no_grad():
            next_v = self.v(next_states)
            target_q = rewards + self.gamma * (1 - dones) * next_v
        
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        self.q1_opt.zero_grad()
        q1_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        q2_loss.backward()
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # Update Actor
        with torch.no_grad():
            q = torch.min(self.q1(states, actions), self.q2(states, actions))
            v = self.v(states)
            adv = q - v
            exp_adv = torch.exp(self.beta * adv).clamp(max=100.0)
        
        action_target = actions
        action_pred = self.actor(states)
        if self.actor.action_mode == "residual_prev_action" and self.actor.action_context_indices:
            assert self._state_mean_tensor is not None
            assert self._state_std_tensor is not None
            prev_action = torch.stack(
                [
                    (states[:, idx] * self._state_std_tensor[idx] + self._state_mean_tensor[idx]) / self.actor.action_max[i]
                    for i, idx in enumerate(self.actor.action_context_indices)
                ],
                dim=-1,
            )
            action_target = actions - prev_action
        bc_loss = F.mse_loss(action_pred, action_target, reduction='none').sum(-1, keepdim=True)
        rate_loss = torch.zeros_like(bc_loss)
        if self.action_rate_penalty > 0 and self.actor.action_mode == "residual_prev_action":
            rate_loss = (action_pred ** 2).sum(dim=-1, keepdim=True)
        actor_loss = (exp_adv * (bc_loss + self.action_rate_penalty * rate_loss)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Update targets - ADD THIS WHOLE SECTION
        self.soft_update_targets()

        return {
            "loss/q_total": (q1_loss + q2_loss).item(),
            "loss/q1": q1_loss.item(),
            "loss/q2": q2_loss.item(),
            "loss/v": v_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/bc_unweighted": bc_loss.mean().item(),
            "loss/action_rate": rate_loss.mean().item(),
            "values/q_target_mean": target_q.mean().item(),
            "values/q1_mean": q1.mean().item(),
            "values/q2_mean": q2.mean().item(),
            "values/v_mean": v.mean().item(),
            "advantage/mean": adv.mean().item(),
            "advantage/std": adv.std(unbiased=False).item(),
            "advantage/max": adv.max().item(),
            "advantage/exp_mean": exp_adv.mean().item(),
            "advantage/exp_max": exp_adv.max().item(),
            "batch/reward_mean": rewards.mean().item(),
            "batch/reward_std": rewards.std(unbiased=False).item(),
            "batch/done_mean": dones.mean().item(),
            "batch/action_abs_mean": actions.abs().mean().item(),
            "batch/pred_action_abs_mean": action_pred.abs().mean().item(),
            "grad_norm/q1": float(q1_grad_norm),
            "grad_norm/q2": float(q2_grad_norm),
            "grad_norm/v": float(v_grad_norm),
            "grad_norm/actor": float(actor_grad_norm),
        }

    def select_action(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.actor.act(state).cpu().numpy()[0]
        
# Option 1: Load from individual trajectories
def load_trajectories_to_buffer(buffer, trajectories):
    """
    trajectories: list of dicts with keys ['states', 'actions', 'rewards', 'dones']
    Each trajectory is shape (T, dim)
    """
    for traj in trajectories:
        datapoint = {}

        states = traj['states']
        actions = traj['actions']
        rewards = traj['rewards']
        dones = traj['dones']
        
        for t in range(len(states) - 1):
            buffer.add(
                state=states[t],
                action=actions[t],
                next_state=states[t+1],
                reward=rewards[t],
                done=dones[t]
            )

def normalize_buffer(buffer):
    # Calculate mean and std across all states
    state_mean = buffer.states[:buffer.size].mean(axis=0)  # Mean of each feature
    state_std = buffer.states[:buffer.size].std(axis=0)    # Std of each feature
    state_std[state_std < 1e-8] = 1.0
    
    # Transform: (original - mean) / std
    buffer.states[:buffer.size] = (buffer.states[:buffer.size] - state_mean) / state_std
    buffer.next_states[:buffer.size] = (buffer.next_states[:buffer.size] - state_mean) / state_std
        # Add action normalization

    action_max = np.abs(buffer.actions[:buffer.size]).max(axis=0)  # Shape: (2,)
    action_max[action_max < 1e-8] = 1.0
    buffer.actions[:buffer.size] = buffer.actions[:buffer.size] / action_max  # Divide each dimension
    
    # Same for rewards
    reward_std = buffer.rewards[:buffer.size].std()
    if reward_std < 1e-8:
        reward_std = 1.0
    buffer.rewards[:buffer.size] = (buffer.rewards[:buffer.size] - buffer.rewards[:buffer.size].mean()) / reward_std

    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "action_max": action_max.astype(np.float32),
    }

def buffer_stats(buffer):
    states = buffer.states[:buffer.size]
    actions = buffer.actions[:buffer.size]
    rewards = buffer.rewards[:buffer.size]
    dones = buffer.dones[:buffer.size]
    return {
        "dataset/state_mean_abs": float(np.abs(states).mean()),
        "dataset/state_std_mean": float(states.std(axis=0).mean()),
        "dataset/action_abs_mean": float(np.abs(actions).mean()),
        "dataset/action_abs_max": float(np.abs(actions).max()),
        "dataset/reward_mean": float(rewards.mean()),
        "dataset/reward_std": float(rewards.std()),
        "dataset/reward_min": float(rewards.min()),
        "dataset/reward_max": float(rewards.max()),
        "dataset/done_fraction": float(dones.mean()),
    }

def make_eval_batch(buffer, eval_batch_size, seed, device):
    if eval_batch_size <= 0 or buffer.size == 0:
        return None
    rng = np.random.default_rng(seed)
    size = min(eval_batch_size, buffer.size)
    indices = rng.choice(buffer.size, size=size, replace=False)
    target_device = torch.device(device or "cpu")
    return tuple(
        torch.as_tensor(array[indices], dtype=torch.float32, device=target_device)
        for array in (buffer.states, buffer.actions, buffer.next_states, buffer.rewards, buffer.dones)
    )

def tensor_stats(prefix, value):
    flat = value.detach().cpu().flatten()
    return {
        f"{prefix}/mean": flat.mean().item(),
        f"{prefix}/std": flat.std(unbiased=False).item(),
        f"{prefix}/min": flat.min().item(),
        f"{prefix}/max": flat.max().item(),
    }

def evaluate_iql(iql, batch, include_histograms):
    if batch is None:
        return {}
    states, actions, next_states, rewards, dones = (
        tensor.to(iql.device, non_blocking=True) for tensor in batch
    )
    with torch.no_grad():
        q1_data = iql.q1(states, actions)
        q2_data = iql.q2(states, actions)
        q_data = torch.min(q1_data, q2_data)
        v_data = iql.v(states)
        policy_actions = iql.actor(states)
        if iql.actor.action_mode == "residual_prev_action" and iql.actor.action_context_indices and iql._state_mean_tensor is not None:
            prev_action = torch.stack(
                [
                    (states[:, idx] * iql._state_std_tensor[idx] + iql._state_mean_tensor[idx]) / iql.actor.action_max[i]
                    for i, idx in enumerate(iql.actor.action_context_indices)
                ],
                dim=-1,
            )
            policy_actions = prev_action + policy_actions
        q1_policy = iql.q1(states, policy_actions)
        q2_policy = iql.q2(states, policy_actions)
        q_policy = torch.min(q1_policy, q2_policy)
        v_next = iql.v(next_states)
        td_target = rewards + iql.gamma * (1 - dones) * v_next
        advantage_data = q_data - v_data
        advantage_policy = q_policy - v_data
        action_error = policy_actions - actions
        action_mse = (action_error ** 2).mean(dim=-1, keepdim=True)

    metrics = {
        **tensor_stats("eval/q_data", q_data),
        **tensor_stats("eval/q_policy", q_policy),
        **tensor_stats("eval/v", v_data),
        **tensor_stats("eval/td_target", td_target),
        **tensor_stats("eval/advantage_data", advantage_data),
        **tensor_stats("eval/advantage_policy", advantage_policy),
        **tensor_stats("eval/action_mse", action_mse),
        "eval/q_policy_minus_q_data_mean": (q_policy - q_data).mean().item(),
        "eval/action_abs_mean": policy_actions.abs().mean().item(),
        "eval/action_error_abs_mean": action_error.abs().mean().item(),
    }
    if include_histograms:
        metrics.update({
            "eval_hist/q_data": wandb.Histogram(q_data.detach().cpu().numpy()),
            "eval_hist/q_policy": wandb.Histogram(q_policy.detach().cpu().numpy()),
            "eval_hist/v": wandb.Histogram(v_data.detach().cpu().numpy()),
            "eval_hist/advantage_data": wandb.Histogram(advantage_data.detach().cpu().numpy()),
            "eval_hist/action_mse": wandb.Histogram(action_mse.detach().cpu().numpy()),
        })
    return metrics

def train_iql(
    iql,
    buffer,
    batch_size,
    num_steps,
    checkpoint_dir,
    resume_from,
    checkpoint_interval,
    log_interval,
    eval_batch,
    eval_interval,
    eval_histogram_interval,
    normalizers,
    checkpoint_eval_interval,
    checkpoint_eval_metric,
    checkpoint_eval_kwargs,
):
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    
    start_step = 0
    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, weights_only=False)
        try:
            iql.actor.load_state_dict(checkpoint['actor'])
            iql.q1.load_state_dict(checkpoint['q1'])
            iql.q2.load_state_dict(checkpoint['q2'])
            iql.v.load_state_dict(checkpoint['v'])
            iql.q1_target.load_state_dict(checkpoint['q1_target'])
            iql.q2_target.load_state_dict(checkpoint['q2_target'])
            start_step = checkpoint['step']
            logger.info("Resumed from step %s", start_step)
        except RuntimeError as exc:
            logger.warning("Skipping incompatible checkpoint %s: %s", resume_from, exc)
    
    batches = itertools.cycle(dataloader)
    best_eval_score = None
    best_eval_checkpoint = None
    for step in range(start_step, num_steps):
        batch = next(batches)
        metrics = iql.update(batch)
        
        if log_interval > 0 and step % log_interval == 0:
            wandb.log(metrics, step=step)

        if eval_interval > 0 and step % eval_interval == 0:
            eval_metrics = evaluate_iql(
                iql,
                eval_batch,
                include_histograms=eval_histogram_interval > 0 and step % eval_histogram_interval == 0,
            )
            if eval_metrics:
                wandb.log(eval_metrics, step=step)
        
        if checkpoint_interval > 0 and step % checkpoint_interval == 0 and step > 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_step_{step}.pt"
            torch.save({
                'actor': iql.actor.state_dict(),
                'q1': iql.q1.state_dict(),
                'q2': iql.q2.state_dict(),
                'v': iql.v.state_dict(),
                'q1_target': iql.q1_target.state_dict(),
                'q2_target': iql.q2_target.state_dict(),
                'step': step,
                'action_max': iql.actor.action_max.cpu(),
                'state_dim': iql.state_dim,
                'action_dim': iql.action_dim,
                'action_mode': getattr(iql.actor, "action_mode", "absolute"),
                'action_rate_penalty': getattr(iql, "action_rate_penalty", 0.0),
                **(normalizers or {}),
            }, checkpoint_path)
            logger.info("Saved checkpoint at step %s", step)
            if checkpoint_eval_interval > 0 and step % checkpoint_eval_interval == 0:
                try:
                    from rl.eval import run_actor_eval_from_config

                    eval_result = run_actor_eval_from_config(
                        actor_checkpoint=str(checkpoint_path),
                        output_dir=str(Path(checkpoint_dir) / f"checkpoint_eval_step_{step}"),
                        render_plots=False,
                        render_movie=False,
                        render_summary=False,
                        **(checkpoint_eval_kwargs or {}),
                    )
                    score = float(eval_result.get("metrics", {}).get(checkpoint_eval_metric, float("-inf")))
                    if best_eval_score is None or score > best_eval_score:
                        best_eval_score = score
                        best_eval_checkpoint = str(checkpoint_path)
                        with (Path(checkpoint_dir) / "best_closed_loop_checkpoint.json").open("w") as f:
                            json.dump(
                                {
                                    "best_checkpoint": best_eval_checkpoint,
                                    "best_score": best_eval_score,
                                    "metric": checkpoint_eval_metric,
                                    "step": step,
                                },
                                f,
                                indent=2,
                            )
                except Exception as exc:
                    logger.warning("Closed-loop checkpoint eval failed at step %s: %s", step, exc)

def latest_checkpoint(checkpoint_dir):
    checkpoints = list(Path(checkpoint_dir).glob('checkpoint_step_*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))

def resolve_action_mode(observation_mode, action_mode):
    """Pick the actor's action_mode, defaulting from observation_mode when unset.

    residual_prev_action needs the previous action inside the observation, which
    only exists for observation_mode=prev_action. Leaving action_mode unset lets
    callers (and the Slurm wrappers) choose observation_mode alone and get a
    compatible action_mode automatically: prev_action -> residual_prev_action,
    everything else -> absolute.
    """
    if action_mode is None:
        return "residual_prev_action" if observation_mode == "prev_action" else "absolute"
    if action_mode == "residual_prev_action" and observation_mode != "prev_action":
        raise ValueError(
            "action_mode=residual_prev_action requires observation_mode=prev_action; "
            f"got observation_mode={observation_mode!r}. Use action_mode=absolute, or "
            "leave action_mode unset to auto-select."
        )
    return action_mode

def train_from_config(
    dataset_dir,
    output_dir,
    batch_size,
    num_steps,
    project,
    run_name,
    resume_from,
    wandb_mode,
    checkpoint_interval,
    log_interval,
    train_seed,
    tau,
    beta,
    gamma,
    lr,
    hidden_dim,
    use_wandb_run_subdir,
    eval_interval,
    eval_batch_size,
    eval_histogram_interval,
    eval_seed,
    device,
    replay_cache_dir,
    prefer_replay_cache,
    run_actor_eval,
    actor_eval_output_dir,
    actor_eval_project,
    actor_eval_run_name,
    actor_eval_wandb_mode,
    actor_eval_initial_relax_state,
    actor_eval_initial_relax_cache_dir,
    actor_eval_max_loop,
    actor_eval_grid_size,
    actor_eval_device,
    actor_eval_wandb_group,
    observation_mode,
    action_mode,
    action_rate_penalty,
    checkpoint_eval_interval,
    checkpoint_eval_metric,
    allow_mismatched_rewards=False,
    wandb_group=None,
):
    action_mode = resolve_action_mode(observation_mode, action_mode)
    base_config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "use_wandb_run_subdir": use_wandb_run_subdir,
        "train_seed": train_seed,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "checkpoint_interval": checkpoint_interval,
        "log_interval": log_interval,
        "resume_from": resume_from,
        "tau": tau,
        "beta": beta,
        "gamma": gamma,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "eval_interval": eval_interval,
        "eval_batch_size": eval_batch_size,
        "eval_histogram_interval": eval_histogram_interval,
        "eval_seed": eval_seed,
        "device": device,
        "replay_cache_dir": None if replay_cache_dir is None else str(replay_cache_dir),
        "prefer_replay_cache": prefer_replay_cache,
        "run_actor_eval": run_actor_eval,
        "actor_eval_output_dir": actor_eval_output_dir,
        "actor_eval_project": actor_eval_project,
        "actor_eval_run_name": actor_eval_run_name,
        "actor_eval_wandb_mode": actor_eval_wandb_mode,
        "actor_eval_initial_relax_state": actor_eval_initial_relax_state,
        "actor_eval_initial_relax_cache_dir": actor_eval_initial_relax_cache_dir,
        "actor_eval_max_loop": actor_eval_max_loop,
        "actor_eval_grid_size": actor_eval_grid_size,
        "actor_eval_device": actor_eval_device,
        "actor_eval_wandb_group": actor_eval_wandb_group,
        "observation_mode": observation_mode,
        "action_mode": action_mode,
        "action_rate_penalty": action_rate_penalty,
        "checkpoint_eval_interval": checkpoint_eval_interval,
        "checkpoint_eval_metric": checkpoint_eval_metric,
        "allow_mismatched_rewards": allow_mismatched_rewards,
    }
    wandb_init_kwargs = {"project": project, "config": base_config, "job_type": "train"}
    if run_name:
        wandb_init_kwargs["name"] = run_name
    if wandb_mode:
        wandb_init_kwargs["mode"] = wandb_mode
    if wandb_group:
        wandb_init_kwargs["group"] = wandb_group

    run = wandb.init(**wandb_init_kwargs)
    config = dict(run.config)
    if config.get("train_seed") is None:
        train_seed = _random_seed32()
        run.config.update({"train_seed": int(train_seed)}, allow_val_change=True)
        config = dict(run.config)
        logger.info("IQL generated train_seed=%s", train_seed)
    else:
        train_seed = int(config["train_seed"])
        logger.info("IQL train_seed=%s", train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    dataset_dir = Path(config["dataset_dir"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    if config.get("use_wandb_run_subdir"):
        output_dir = output_dir / run.id
        run.config.update({"output_dir": str(output_dir)}, allow_val_change=True)
        config = dict(run.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("IQL dataset_dir=%s", dataset_dir)
    logger.info("IQL output_dir=%s", output_dir)

    specs = describe_dataset_with_replay_cache(
        dataset_dir,
        cache_dir=config.get("replay_cache_dir"),
        prefer_cache=bool(config.get("prefer_replay_cache", True)),
    )
    logger.info(
        "IQL dataset selected_format=%s replay_cache_used=%s replay_cache_dir=%s zarr_store_count=%s json_file_count=%s num_trajectories=%s num_transitions=%s state_dim=%s action_dim=%s explicit_next_state=%s zarr_explicit_next_state_store_count=%s",
        specs["selected_format"],
        specs.get("replay_cache_used", False),
        specs.get("replay_cache_dir"),
        specs["zarr_store_count"],
        specs["json_file_count"],
        specs["num_trajectories"],
        specs["num_transitions"],
        specs["state_dim"],
        specs["action_dim"],
        specs["dataset_has_explicit_next_state"],
        specs["zarr_explicit_next_state_store_count"],
    )
    if specs["zarr_takes_precedence"]:
        logger.warning("IQL dataset warning: both Zarr and JSON were found; using Zarr.")
    dataset_config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "num_trajectories": specs["num_trajectories"],
        "num_transitions": specs["num_transitions"],
        "state_dim": specs["state_dim"],
        "action_dim": specs["action_dim"],
        "dataset_format": specs.get("format", "unknown"),
        "dataset_selected_format": specs["selected_format"],
        "dataset_zarr_store_count": specs["zarr_store_count"],
        "dataset_json_file_count": specs["json_file_count"],
        "dataset_zarr_takes_precedence": specs["zarr_takes_precedence"],
        "dataset_has_explicit_next_state": specs["dataset_has_explicit_next_state"],
        "dataset_zarr_explicit_next_state_store_count": specs["zarr_explicit_next_state_store_count"],
        "dataset_replay_cache_used": specs.get("replay_cache_used", False),
        "dataset_replay_cache_dir": specs.get("replay_cache_dir"),
        "state_keys": specs["state_keys"],
        "dataset_reward_config": reward_config_to_dict(specs.get("reward_config")),
    }
    if config.get("eval_seed") is None:
        eval_seed = _random_seed32()
        run.config.update({"eval_seed": int(eval_seed)}, allow_val_change=True)
        config = dict(run.config)
        logger.info("IQL generated eval_seed=%s", eval_seed)
    else:
        eval_seed = int(config["eval_seed"])
        logger.info("IQL eval_seed=%s", eval_seed)
    run.config.update(dataset_config, allow_val_change=True)
    config = dict(run.config)
    with (output_dir / "iql_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    state_dim = specs["state_dim"]
    action_dim = specs["action_dim"]
    dataset_size = specs["num_transitions"]
    buffer = ReplayBuffer(state_dim, action_dim, dataset_size)
    load_d4rl_dataset(
        str(dataset_dir),
        buffer,
        specs["state_keys"],
        cache_dir=config.get("replay_cache_dir"),
        prefer_cache=bool(config.get("prefer_replay_cache", True)),
    )
    logger.info("IQL replay transitions_loaded=%s", buffer.size)
    run.config.update({"transitions_loaded": buffer.size}, allow_val_change=True)
    raw_stats = {f"raw_{key}": value for key, value in buffer_stats(buffer).items()}

    normalizers = normalize_buffer(buffer)
    action_max = normalizers["action_max"]
    normalized_stats = buffer_stats(buffer)
    run.log({**raw_stats, **normalized_stats}, step=0)
    run.summary.update({**raw_stats, **normalized_stats})
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        train_device = torch.device(requested_device)
    if train_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {train_device}, but torch.cuda.is_available() is False."
        )
    logger.info("IQL train_device=%s", train_device)
    run.config.update({
        "train_device": str(train_device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_device_name": (
            torch.cuda.get_device_name(train_device)
            if train_device.type == "cuda"
            else None
        ),
    }, allow_val_change=True)

    eval_batch = make_eval_batch(
        buffer,
        eval_batch_size=int(config["eval_batch_size"]),
        seed=eval_seed,
        device=train_device,
    )
    state_keys = specs["state_keys"]
    action_context_indices = []
    if config.get("observation_mode") == "prev_action":
        for key in ("prev_ecrh", "prev_nbi"):
            if key not in state_keys:
                raise ValueError(
                    f"observation_mode=prev_action requires {key!r} in dataset state_keys, got {state_keys}"
                )
            action_context_indices.append(state_keys.index(key))
    elif config.get("observation_mode") == "legacy":
        action_context_indices = []
    iql_agent = IQL(
        action_max,
        state_dim,
        action_dim,
        tau=float(config["tau"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        lr=float(config["lr"]),
        hidden_dim=int(config["hidden_dim"]),
        device=train_device,
        action_mode=str(config.get("action_mode", "absolute")),
        action_context_indices=action_context_indices,
        action_rate_penalty=float(config.get("action_rate_penalty", 0.0)),
    )
    iql_agent.set_normalizers(normalizers["state_mean"], normalizers["state_std"])

    checkpoint_path = None
    if config["resume_from"] == 'auto':
        checkpoint_path = latest_checkpoint(output_dir)
    elif config["resume_from"]:
        checkpoint_path = Path(config["resume_from"])

    train_iql(
        iql_agent,
        buffer,
        batch_size=int(config["batch_size"]),
        num_steps=int(config["num_steps"]),
        checkpoint_dir=str(output_dir),
        resume_from=checkpoint_path,
        checkpoint_interval=int(config["checkpoint_interval"]),
        log_interval=int(config["log_interval"]),
        eval_batch=eval_batch,
        eval_interval=int(config["eval_interval"]),
        eval_histogram_interval=int(config["eval_histogram_interval"]),
        normalizers={
            "state_mean": torch.as_tensor(normalizers["state_mean"]),
            "state_std": torch.as_tensor(normalizers["state_std"]),
        },
        checkpoint_eval_interval=int(config.get("checkpoint_eval_interval", 0)),
        checkpoint_eval_metric=str(config.get("checkpoint_eval_metric", "actor_eval/reward_total")),
        checkpoint_eval_kwargs={
            "dataset_dir": str(dataset_dir),
            "project": project,
            "run_name": f"{run.name or run.id}-checkpoint-eval",
            "wandb_mode": "disabled",
            "wandb_group": actor_eval_wandb_group or wandb_group,
            "initial_relax_state": config.get("actor_eval_initial_relax_state"),
            "initial_relax_cache_dir": config.get("actor_eval_initial_relax_cache_dir"),
            "max_loop": int(config.get("actor_eval_max_loop", 0)),
            "grid_size": int(config.get("actor_eval_grid_size", 51)),
            "device": config.get("actor_eval_device"),
            "allow_mismatched_rewards": bool(config.get("allow_mismatched_rewards", False)),
        },
    )

    weights_path = output_dir / 'iql_weights.pt'
    torch.save({
        'actor': iql_agent.actor.state_dict(),
        'q1': iql_agent.q1.state_dict(),
        'q2': iql_agent.q2.state_dict(),
        'v': iql_agent.v.state_dict(),
        'q1_target': iql_agent.q1_target.state_dict(),
        'q2_target': iql_agent.q2_target.state_dict(),
        'action_max': torch.as_tensor(action_max),
        'state_mean': torch.as_tensor(normalizers["state_mean"]),
        'state_std': torch.as_tensor(normalizers["state_std"]),
        'state_keys': specs["state_keys"],
        'state_dim': state_dim,
        'action_dim': action_dim,
        'action_mode': str(config.get("action_mode", "absolute")),
        'action_rate_penalty': float(config.get("action_rate_penalty", 0.0)),
        'observation_mode': str(config.get("observation_mode", "legacy")),
        'checkpoint_eval_interval': int(config.get("checkpoint_eval_interval", 0)),
        'checkpoint_eval_metric': str(config.get("checkpoint_eval_metric", "actor_eval/reward_total")),
        'config': config,
    }, weights_path)
    logger.info("Saved final weights to %s", weights_path)
    if config.get("run_actor_eval"):
        wandb.finish()
        from rl.eval import run_actor_eval_from_config

        eval_output_dir = config.get("actor_eval_output_dir") or str(output_dir / "actor_eval")
        eval_project = config.get("actor_eval_project") or project
        eval_run_name = config.get("actor_eval_run_name") or f"{run.name or run.id}-actor-eval"
        run_actor_eval_from_config(
            actor_checkpoint=weights_path,
            dataset_dir=dataset_dir,
            output_dir=eval_output_dir,
            project=eval_project,
            run_name=eval_run_name,
            wandb_mode=config.get("actor_eval_wandb_mode") or wandb_mode,
            wandb_group=(
                config.get("actor_eval_wandb_group")
                or config.get("wandb_group")
                or wandb_group
            ),
            initial_relax_state=config.get("actor_eval_initial_relax_state"),
            initial_relax_cache_dir=config.get("actor_eval_initial_relax_cache_dir"),
            max_loop=int(config.get("actor_eval_max_loop", 0)),
            grid_size=int(config.get("actor_eval_grid_size", 51)),
            device=config.get("actor_eval_device"),
            allow_mismatched_rewards=bool(config.get("allow_mismatched_rewards", False)),
        )
    else:
        wandb.finish()

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": modal.Volume.from_name("rl-dataset", create_if_missing=True)},
    timeout=86400
)
def train_modal():
    args = parse_args([
        "--dataset_dir",
        "/data/rl_dataset_delta_sampling_maxloop=2_grid_51_preprocessed",
        "--output_dir",
        "/data",
        "--no-use_wandb_run_subdir",
    ])
    train_from_config(**train_kwargs_from_args(args))

@app.local_entrypoint()
def modal_main():
    train_modal.remote()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train IQL on a collected TORAX trajectory dataset.")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root of the collected offline dataset. Must contain the replay shards / trajectory files used for training.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for checkpoints, config, and final weights. Defaults to out/iql/<dataset_name>.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Minibatch size for offline IQL updates. Larger values are steadier but use more memory.",
    )
    parser.add_argument(
        "--train_seed",
        type=int,
        default=None,
        help="Seed for training-time randomness. Leave unset to generate one in Python and persist it in the training config for reproducibility.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=40000,
        help="Number of gradient steps. Increase for longer final runs; reduce for debugging / quick sweeps.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "iql-training"),
        help="Weights & Biases project name.",
    )
    parser.add_argument("--run_name", default=None, help="Optional W&B run name.")
    parser.add_argument(
        "--wandb_group",
        default=os.environ.get("WANDB_GROUP"),
        help="W&B group for related training/eval runs. Useful for tying checkpoints, evals, and ablations together.",
    )
    parser.add_argument(
        "--resume_from",
        default="auto",
        help="Checkpoint path to resume from, 'auto' to use the latest checkpoint in output_dir, or empty string to disable resume.",
    )
    parser.add_argument(
        "--wandb_mode",
        default=os.environ.get("WANDB_MODE"),
        help="W&B mode: online, offline, or disabled. Use offline for local debugging.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help=(
            "Save a training checkpoint every N steps. Use 1000 for the normal default when you want "
            "frequent recovery points and side-by-side analysis; increase it only when checkpoint I/O "
            "becomes a problem on very long runs."
        ),
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log training metrics every N steps. Lower values give finer visibility at higher overhead.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help="Expectile for the value update. Higher values push V toward upper Q values more aggressively.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Advantage weight temperature for actor updates. Higher values make the policy more selective.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used in the offline TD target.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for all IQL optimizers.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden width for the Q, V, and actor MLPs.",
    )
    parser.add_argument(
        "--use_wandb_run_subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a W&B-run-specific subdirectory to avoid collisions between runs.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Run offline batch eval every N steps on held-out data.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2048,
        help="Number of transitions used in each offline eval batch.",
    )
    parser.add_argument(
        "--eval_histogram_interval",
        type=int,
        default=5000,
        help="How often to attach expensive W&B histograms during offline eval.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Seed used when sampling the offline eval batch. Leave unset to generate one in Python and record it in the training config.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("IQL_DEVICE", "auto"),
        help="Training device: auto, cpu, cuda, cuda:0, etc. Use auto unless you have a reason to pin it.",
    )
    parser.add_argument(
        "--replay_cache_dir",
        default=None,
        help="Optional compact replay cache directory. Use when dataset loading should be faster than reading the raw shards.",
    )
    parser.add_argument(
        "--use_replay_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the compact replay cache when available. Disable only if you are debugging the raw dataset path.",
    )
    parser.add_argument(
        "--run_actor_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a full closed-loop TORAX eval after training. Leave on for final runs; disable for fast training-only sweeps.",
    )
    parser.add_argument(
        "--actor_eval_output_dir",
        default=None,
        help="Directory for the post-training closed-loop eval outputs.",
    )
    parser.add_argument(
        "--actor_eval_project",
        default=None,
        help="W&B project for the post-training closed-loop eval. Defaults to the training project.",
    )
    parser.add_argument(
        "--actor_eval_run_name",
        default=None,
        help="W&B run name for the post-training closed-loop eval.",
    )
    parser.add_argument(
        "--actor_eval_wandb_mode",
        default=os.environ.get("ACTOR_EVAL_WANDB_MODE"),
        help="W&B mode for the post-training eval. Set offline/disabled if you do not want a second logged run.",
    )
    parser.add_argument(
        "--actor_eval_wandb_group",
        default=os.environ.get("ACTOR_EVAL_WANDB_GROUP"),
        help="W&B group for the post-training eval. Useful for pairing train and eval runs.",
    )
    parser.add_argument(
        "--actor_eval_initial_relax_state",
        default=None,
        help="Explicit initial-relax cache path for the actor eval. Use when you want to bypass cache resolution.",
    )
    parser.add_argument(
        "--actor_eval_initial_relax_cache_dir",
        default=None,
        help="Directory containing keyed initial-relax caches used by the closed-loop eval.",
    )
    parser.add_argument(
        "--actor_eval_max_loop",
        type=int,
        default=1,
        help="Number of TORAX coupling loops used in the eval. Use 1 for the fast default path; use 2 only when you need the extra convergence check / benchmark parity.",
    )
    parser.add_argument(
        "--actor_eval_grid_size",
        type=int,
        default=51,
        help="TORAX radial grid size used for eval. Keep this aligned with the benchmark unless you are testing sensitivity.",
    )
    parser.add_argument(
        "--actor_eval_device",
        default=None,
        help="Optional device override for actor eval. Usually leave unset so the eval wrapper can force CPU.",
    )
    parser.add_argument(
        "--action_mode",
        choices=["absolute", "residual_prev_action"],
        default=None,
        help="How the actor parameterizes heating commands. Leave unset to auto-select from observation_mode (prev_action -> residual_prev_action for smoother control; otherwise absolute). residual_prev_action is only valid with observation_mode=prev_action.",
    )
    parser.add_argument(
        "--action_rate_penalty",
        type=float,
        default=0.01,
        help="Penalty weight on action changes when using residual_prev_action. Set to 0 to disable, or increase slightly if the policy is still too jumpy.",
    )
    parser.add_argument(
        "--allow_mismatched_rewards",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ALLOW_MISMATCHED_REWARDS", False),
        help=(
            "Allow eval to continue even when the checkpoint's recorded training reward config differs from the current eval runtime reward config. "
            "Leave this off for normal runs so reward drift fails fast; turn it on only for deliberate legacy comparisons or reward-change ablations."
        ),
    )
    parser.add_argument(
        "--checkpoint_eval_interval",
        type=int,
        default=0,
        help=(
            "If >0, run closed-loop eval on saved checkpoints every N steps and keep the best score. "
            "Use 0 for fast exploratory training, a coarse value like 5000-10000 for serious runs, "
            "and a smaller value only if you want tighter checkpoint selection and can afford the extra eval cost."
        ),
    )
    parser.add_argument(
        "--checkpoint_eval_metric",
        default="actor_eval/reward_total",
        help="Metric name used to rank checkpoint evals. Use a closed-loop metric here, not a training loss, if you want the checkpoint selector to reflect deployment quality.",
    )
    parser.add_argument(
        "--observation_mode",
        choices=["legacy", "prev_action", "plasma_only"],
        default="prev_action",
        help="How trajectory observations are constructed from TORAX states and actions. Use prev_action for normal actor-conditioned datasets; use legacy only for older compatibility datasets; plasma_only removes action history entirely.",
    )
    return parser.parse_args(argv)

def train_kwargs_from_args(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("out") / "iql" / dataset_dir.name
    return {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "train_seed": args.train_seed,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "project": args.project,
        "run_name": args.run_name,
        "resume_from": args.resume_from,
        "wandb_mode": args.wandb_mode,
        "checkpoint_interval": args.checkpoint_interval,
        "log_interval": args.log_interval,
        "tau": args.tau,
        "beta": args.beta,
        "gamma": args.gamma,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "use_wandb_run_subdir": args.use_wandb_run_subdir,
        "eval_interval": args.eval_interval,
        "eval_batch_size": args.eval_batch_size,
        "eval_histogram_interval": args.eval_histogram_interval,
        "eval_seed": args.eval_seed,
        "device": args.device,
        "replay_cache_dir": args.replay_cache_dir,
        "prefer_replay_cache": args.use_replay_cache,
        "run_actor_eval": args.run_actor_eval,
        "actor_eval_output_dir": args.actor_eval_output_dir,
        "actor_eval_project": args.actor_eval_project,
        "actor_eval_run_name": args.actor_eval_run_name,
        "actor_eval_wandb_mode": args.actor_eval_wandb_mode,
        "actor_eval_wandb_group": args.actor_eval_wandb_group,
        "actor_eval_initial_relax_state": args.actor_eval_initial_relax_state,
        "actor_eval_initial_relax_cache_dir": args.actor_eval_initial_relax_cache_dir,
        "actor_eval_max_loop": args.actor_eval_max_loop,
        "actor_eval_grid_size": args.actor_eval_grid_size,
        "actor_eval_device": args.actor_eval_device,
        "observation_mode": args.observation_mode,
        "action_mode": args.action_mode,
        "action_rate_penalty": args.action_rate_penalty,
        "allow_mismatched_rewards": args.allow_mismatched_rewards,
        "checkpoint_eval_interval": args.checkpoint_eval_interval,
        "checkpoint_eval_metric": args.checkpoint_eval_metric,
        "wandb_group": args.wandb_group,
    }

def main():
    args = parse_args(sys.argv[1:])
    train_from_config(**train_kwargs_from_args(args))

if __name__ == "__main__":
    main()
