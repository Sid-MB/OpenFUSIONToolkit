import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import json
import wandb

import modal

app = modal.App("iql-training")

image = modal.Image.debian_slim().pip_install(
    "torch", "numpy", "wandb"
)

class ReplayBuffer(Dataset):
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 600*12):
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
        self.action_max = action_max

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        return self.net(state) * self.action_max

class IQL:
    def __init__(self, action_max, state_dim: int, action_dim: int, 
                 tau: float = 0.7, beta: float = 3.0, 
                 gamma: float = 0.99, lr: float = 1e-4):
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.v = ValueNetwork(state_dim)
        self.actor = Actor(action_max, state_dim, action_dim)
        
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
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)
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
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)  # ADD HERE
        self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)  # ADD HERE
        self.q2_opt.step()

        # Update Actor
        with torch.no_grad():
            q = torch.min(self.q1(states, actions), self.q2(states, actions))
            v = self.v(states)
            adv = q - v
            exp_adv = torch.exp(self.beta * adv).clamp(max=100.0)
        
        action_pred = self.actor(states)
        actor_loss = -(exp_adv * F.mse_loss(action_pred, actions, reduction='none').sum(-1, keepdim=True)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # ADD HERE
        self.actor_opt.step()

        # Update targets - ADD THIS WHOLE SECTION
        self.soft_update_targets()

        return {"q_loss": (q1_loss + q2_loss).item(), "v_loss": v_loss.item(), "actor_loss": actor_loss.item()}

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return self.actor(state).cpu().numpy()[0]
        
import numpy as np
import pickle
from pathlib import Path

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
  
def load_state(state):
  arr = []
  for key in state:
      arr.append(state[key])
  new_row = np.array(arr)
  return new_row

# Option 2: Load from D4RL-style dataset
def load_d4rl_dataset(directory, buffer):
    trajectories = []

    for filepath in Path(directory).glob('trajectory_*.json'):
        with open(filepath, 'r') as f:
            traj = json.load(f)
            for i in range(len(traj['transitions'])):
                datapoint = {}
                datapoint["s"] = load_state(traj['transitions'][i]['s'])
                datapoint["a"] = traj['transitions'][i]['a']
                if i < len(traj['transitions']) - 1:
                  datapoint["s_next"] = load_state(traj['transitions'][i+1]['s'])
                else:
                    datapoint["s_next"] = np.zeros(len(traj['transitions'][i]['s']))

                datapoint["r"] = traj['transitions'][i]['r']
                datapoint["done"] = int(i == len(traj['transitions']) - 1)

                buffer.add(datapoint["s"], datapoint["a"], datapoint["s_next"], datapoint["r"], datapoint["done"])

# Option 3: Training loop
def train_iql(iql, buffer, batch_size=128, num_steps=1000000):
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    
    for step in range(num_steps):
        for batch in dataloader:
            metrics = iql.update(batch)
            
            if step % 100 == 0:
                #print(f"Step {step}: {metrics}")
                wandb.log(metrics, step=step)

def normalize_buffer(buffer):
    # Calculate mean and std across all states
    state_mean = buffer.states[:buffer.size].mean(axis=0)  # Mean of each feature
    state_std = buffer.states[:buffer.size].std(axis=0)    # Std of each feature
    
    # Transform: (original - mean) / std
    buffer.states[:buffer.size] = (buffer.states[:buffer.size] - state_mean) / state_std
    buffer.next_states[:buffer.size] = (buffer.next_states[:buffer.size] - state_mean) / state_std
        # Add action normalization

    action_max = np.abs(buffer.actions[:buffer.size]).max(axis=0)  # Shape: (2,)
    buffer.actions[:buffer.size] = buffer.actions[:buffer.size] / action_max  # Divide each dimension
    
    # Same for rewards
    buffer.rewards[:buffer.size] = (buffer.rewards[:buffer.size] - buffer.rewards[:buffer.size].mean()) / buffer.rewards[:buffer.size].std()

    return action_max


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": modal.Volume.from_name("rl-dataset", create_if_missing=True)},
    timeout=86400
)

def train_modal():
    wandb.init(project="iql-training", config={
        "state_dim": 34,
        "action_dim": 2,
        "batch_size": 128,
        "num_steps": 100000
    })
    
    state_dim = 34
    action_dim = 2
    dataset_size = 600*12
    buffer = ReplayBuffer(state_dim, action_dim, dataset_size)
    load_d4rl_dataset('/data/rl_dataset_test', buffer)

    action_max = normalize_buffer(buffer)
    IQL_agent = IQL(action_max, state_dim, action_dim)

    train_iql(IQL_agent, buffer, batch_size=128, num_steps=10000)

    # In train_modal() - line 314
    torch.save({
        'actor': IQL_agent.actor.state_dict(),
        'q1': IQL_agent.q1.state_dict(),
        'q2': IQL_agent.q2.state_dict(),
        'v': IQL_agent.v.state_dict(),
        'action_max': action_max,  # Add this
    }, '/data/iql_weights.pt')
    
    wandb.finish()

@app.local_entrypoint()
def main():
    train_modal.spawn()  # Changed from .remote()
