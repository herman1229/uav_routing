"""
DQN 基线（并发多流版）
与A3C的核心区别：
- 单线程，无法并行探索
- ε-greedy探索，早期随机性高
- Replay Buffer打破样本相关性（但无法利用多线程多样性）
- 固定目标网络（每N步同步一次）
- 无Actor-Critic结构，只有Q网络
- 不感知其他GBS位置（标准DQN状态设计）
"""
import numpy as np
import random
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.98,
                 buffer_size: int = 10000, batch_size: int = 64,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 500, target_update: int = 20):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

        self.q_net = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_net = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps / self.eps_decay)

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.choice(valid_actions)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state))
            valid_q = q[valid_actions]
            idx = valid_q.argmax().item()
        return valid_actions[idx]

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a).unsqueeze(1)
        r  = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d).unsqueeze(1)

        q_val = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
