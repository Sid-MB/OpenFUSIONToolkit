import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import json
import wandb

class ReplayBuffer(Dataset):
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 300000):
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
                 tau: float = 0.8, beta: float = 3.0, 
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
        polyak = 0.995
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)

    def expectile_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch
        states = states.float()
        actions = actions.float()
        next_states = next_states.float()
        rewards = rewards.float()
        dones = dones.float()
        
        with torch.no_grad():
            q = torch.min(
                self.q1_target(states, actions),
                self.q2_target(states, actions)
            )
            
        v = self.v(states)
        v_loss = self.expectile_loss(q - v, self.tau).mean()
        
        self.v_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)
        self.v_opt.step()

        with torch.no_grad():
            next_v = self.v(next_states)
            target_q = rewards + self.gamma * (1 - dones) * next_v
        
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        with torch.no_grad():
            q = torch.min(self.q1(states, actions), self.q2(states, actions))
            v = self.v(states)
            adv = q - v
            exp_adv = torch.exp(self.beta * adv).clamp(max=100.0)
        
        action_pred = self.actor(states)
        actor_loss = -(exp_adv * F.mse_loss(action_pred, actions, reduction='none').sum(-1, keepdim=True)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self.soft_update_targets()

        return {"q_loss": (q1_loss + q2_loss).item(), "v_loss": v_loss.item(), "actor_loss": actor_loss.item()}

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return self.actor(state).cpu().numpy()[0]

def load_state(state):
    arr = []
    for key in state:
        arr.append(state[key])
    new_row = np.array(arr)
    return new_row

def load_d4rl_dataset(directory, buffer):
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

def normalize_buffer(buffer):
    state_mean = buffer.states[:buffer.size].mean(axis=0)
    state_std = buffer.states[:buffer.size].std(axis=0)
    
    buffer.states[:buffer.size] = (buffer.states[:buffer.size] - state_mean) / state_std
    buffer.next_states[:buffer.size] = (buffer.next_states[:buffer.size] - state_mean) / state_std

    action_max = np.abs(buffer.actions[:buffer.size]).max(axis=0)
    buffer.actions[:buffer.size] = buffer.actions[:buffer.size] / action_max
    
    buffer.rewards[:buffer.size] = (buffer.rewards[:buffer.size] - buffer.rewards[:buffer.size].mean()) / buffer.rewards[:buffer.size].std()

    return state_mean, state_std, action_max

def train_iql(iql, buffer, batch_size=128, num_steps=1000000, checkpoint_dir='./checkpoints', resume_from=None):
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    
    start_step = 0
    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, weights_only=False)
        iql.actor.load_state_dict(checkpoint['actor'])
        iql.q1.load_state_dict(checkpoint['q1'])
        iql.q2.load_state_dict(checkpoint['q2'])
        iql.v.load_state_dict(checkpoint['v'])
        iql.q1_target.load_state_dict(checkpoint['q1_target'])
        iql.q2_target.load_state_dict(checkpoint['q2_target'])
        start_step = checkpoint['step']
        print(f"Resumed from step {start_step}")
    
    step = start_step
    while step < num_steps:
        for batch in dataloader:
            metrics = iql.update(batch)
            
            if step % 100 == 0:
                wandb.log(metrics, step=step)
            
            if step % 5000 == 0 and step > 0:
                torch.save({
                    'actor': iql.actor.state_dict(),
                    'q1': iql.q1.state_dict(),
                    'q2': iql.q2.state_dict(),
                    'v': iql.v.state_dict(),
                    'q1_target': iql.q1_target.state_dict(),
                    'q2_target': iql.q2_target.state_dict(),
                    'step': step,
                    'action_max': iql.actor.action_max,
                }, f'{checkpoint_dir}/checkpoint_step_{step}.pt')
                print(f"Saved checkpoint at step {step}")
            
            step += 1
            if step >= num_steps:
                break

if __name__ == "__main__":
    wandb.init(project="iql-training", config={
        "state_dim": 34,
        "action_dim": 2,
        "batch_size": 128,
        "num_steps": 100000
    })
    
    state_dim = 34
    action_dim = 2
    dataset_size = 300000
    buffer = ReplayBuffer(state_dim, action_dim, dataset_size)
    load_d4rl_dataset('./rl_dataset', buffer)

    state_mean, state_std, action_max = normalize_buffer(buffer)
    print(f"Loaded {buffer.size} transitions")
    print(f"State mean: {state_mean}")
    print(f"State std: {state_std}")
    print(f"Action max: {action_max}")
    
    IQL_agent = IQL(action_max, state_dim, action_dim)

    checkpoint_path = None
    checkpoints = list(Path('./checkpoints').glob('checkpoint_step_*.pt'))
    if checkpoints:
        checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    train_iql(IQL_agent, buffer, batch_size=128, num_steps=100000, 
              checkpoint_dir='./checkpoints', resume_from=checkpoint_path)

    torch.save({
        'actor': IQL_agent.actor.state_dict(),
        'q1': IQL_agent.q1.state_dict(),
        'q2': IQL_agent.q2.state_dict(),
        'v': IQL_agent.v.state_dict(),
        'action_max': action_max,
        'state_mean': state_mean,
        'state_std': state_std,
    }, './checkpoints/iql_weights.pt')
    
    wandb.finish()