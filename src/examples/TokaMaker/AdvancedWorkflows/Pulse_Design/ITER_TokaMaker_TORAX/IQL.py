import itertools
import argparse
import json
import os
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

from dataloader import infer_dataset_specs, load_d4rl_dataset

app = modal.App("iql-training")

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
    def __init__(self, action_max, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.register_buffer("action_max", torch.as_tensor(action_max, dtype=torch.float32))

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        return self.net(state) * self.action_max

class IQL:
    def __init__(self, action_max, state_dim: int, action_dim: int, 
                 tau: float = 0.7, beta: float = 3.0, 
                 gamma: float = 0.99, lr: float = 1e-4,
                 hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.v = ValueNetwork(state_dim, hidden_dim)
        self.actor = Actor(action_max, state_dim, action_dim, hidden_dim)
        
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.tau = tau
        self.beta = beta
        self.gamma = gamma

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
        states = states.float()
        actions = actions.float()
        next_states = next_states.float()
        rewards = rewards.float()
        dones = dones.float()
        
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
        
        action_pred = self.actor(states)
        bc_loss = F.mse_loss(action_pred, actions, reduction='none').sum(-1, keepdim=True)
        actor_loss = (exp_adv * bc_loss).mean()
        
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
            state = torch.FloatTensor(state).unsqueeze(0)
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

    return action_max

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

def train_iql(
    iql,
    buffer,
    batch_size=128,
    num_steps=1000000,
    checkpoint_dir='/data',
    resume_from=None,
    checkpoint_interval=5000,
    log_interval=100,
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
            print(f"Resumed from step {start_step}")
        except RuntimeError as exc:
            print(f"Skipping incompatible checkpoint {resume_from}: {exc}")
    
    batches = itertools.cycle(dataloader)
    for step in range(start_step, num_steps):
        batch = next(batches)
        metrics = iql.update(batch)
        
        if log_interval > 0 and step % log_interval == 0:
            wandb.log(metrics, step=step)
        
        if checkpoint_interval > 0 and step % checkpoint_interval == 0 and step > 0:
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
            }, f'{checkpoint_dir}/checkpoint_step_{step}.pt')
            print(f"Saved checkpoint at step {step}")

def latest_checkpoint(checkpoint_dir):
    checkpoints = list(Path(checkpoint_dir).glob('checkpoint_step_*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))

def train_from_config(
    dataset_dir,
    output_dir,
    batch_size=128,
    num_steps=100000,
    project='iql-training',
    run_name=None,
    resume_from='auto',
    wandb_mode=None,
    checkpoint_interval=5000,
    log_interval=100,
    tau=0.7,
    beta=3.0,
    gamma=0.99,
    lr=1e-4,
    hidden_dim=256,
    use_wandb_run_subdir=False,
):
    base_config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "use_wandb_run_subdir": use_wandb_run_subdir,
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
    }
    wandb_init_kwargs = {"project": project, "config": base_config}
    if run_name:
        wandb_init_kwargs["name"] = run_name
    if wandb_mode:
        wandb_init_kwargs["mode"] = wandb_mode

    run = wandb.init(**wandb_init_kwargs)
    config = dict(run.config)

    dataset_dir = Path(config["dataset_dir"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    if config.get("use_wandb_run_subdir"):
        output_dir = output_dir / run.id
        run.config.update({"output_dir": str(output_dir)}, allow_val_change=True)
        config = dict(run.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"IQL dataset_dir={dataset_dir}", flush=True)
    print(f"IQL output_dir={output_dir}", flush=True)

    specs = infer_dataset_specs(dataset_dir)
    dataset_config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "num_trajectories": specs["num_trajectories"],
        "num_transitions": specs["num_transitions"],
        "state_dim": specs["state_dim"],
        "action_dim": specs["action_dim"],
        "dataset_format": specs.get("format", "unknown"),
        "state_keys": specs["state_keys"],
    }
    run.config.update(dataset_config, allow_val_change=True)
    config = dict(run.config)
    with (output_dir / "iql_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    state_dim = specs["state_dim"]
    action_dim = specs["action_dim"]
    dataset_size = specs["num_transitions"]
    buffer = ReplayBuffer(state_dim, action_dim, dataset_size)
    load_d4rl_dataset(str(dataset_dir), buffer, specs["state_keys"])
    raw_stats = {f"raw_{key}": value for key, value in buffer_stats(buffer).items()}

    action_max = normalize_buffer(buffer)
    normalized_stats = buffer_stats(buffer)
    run.log({**raw_stats, **normalized_stats}, step=0)
    run.summary.update({**raw_stats, **normalized_stats})
    iql_agent = IQL(
        action_max,
        state_dim,
        action_dim,
        tau=float(config["tau"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        lr=float(config["lr"]),
        hidden_dim=int(config["hidden_dim"]),
    )

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
        'state_keys': specs["state_keys"],
        'state_dim': state_dim,
        'action_dim': action_dim,
        'config': config,
    }, weights_path)
    print(f"Saved final weights to {weights_path}", flush=True)
    wandb.finish()

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": modal.Volume.from_name("rl-dataset", create_if_missing=True)},
    timeout=86400
)
def train_modal():
    train_from_config(
        dataset_dir='/data/rl_dataset_delta_sampling_maxloop=2_grid_51_preprocessed',
        output_dir='/data',
        batch_size=128,
        num_steps=100000,
    )

@app.local_entrypoint()
def modal_main():
    train_modal.remote()

def parse_args():
    parser = argparse.ArgumentParser(description="Train IQL on a collected TORAX trajectory dataset.")
    parser.add_argument("--dataset_dir", required=True, help="Collected dataset root containing trajectories/")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints, config, and final weights")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "iql-training"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from", default="auto", help="Checkpoint path, 'auto', or empty string to disable resume")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE"), help="Set to offline or disabled for debugging")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--use_wandb_run_subdir", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    train_from_config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        project=args.project,
        run_name=args.run_name,
        resume_from=args.resume_from,
        wandb_mode=args.wandb_mode,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        tau=args.tau,
        beta=args.beta,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        use_wandb_run_subdir=args.use_wandb_run_subdir,
    )

if __name__ == "__main__":
    main()
