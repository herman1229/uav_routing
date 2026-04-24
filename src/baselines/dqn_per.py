"""
DQN with Prioritized Experience Replay (PER)
解决5-UAV场景DQN不稳定的问题

标准DQN失败原因：
  Replay Buffer均匀采样，后期次优经验覆盖早期好经验
  → 策略在ep=800收敛后又退化

PER解决方案：
  TD误差大的经验（重要经验）被更频繁采样
  → 好的路径经验（T_up小，奖励高）TD误差大，被优先学习
  → 防止次优经验稀释好经验
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import deque


class SumTree:
    """优先经验回放的核心数据结构"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority ** self.alpha,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size: int, beta: float = 0.4):
        batch, indices, weights = [], [], []
        segment = self.tree.total() / batch_size
        min_prob = (self.tree.tree[self.tree.capacity - 1] / self.tree.total() + 1e-8)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                continue
            prob = (p / self.tree.total()) + 1e-8
            weight = (prob * self.tree.n_entries) ** (-beta)
            weight /= (min_prob * self.tree.n_entries) ** (-beta)
            batch.append(data)
            indices.append(idx)
            weights.append(weight)

        if not batch:
            return None
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a), np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32),
                indices, np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, (p + 1e-6) ** self.alpha)

    def __len__(self):
        return self.tree.n_entries


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgentPER:
    """DQN with Prioritized Experience Replay"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 lr: float = 5e-4, gamma: float = 0.98,
                 buffer_size: int = 20000, batch_size: int = 128,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 1000, target_update: int = 10,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0):

        self.action_dim   = action_dim
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update= target_update
        self.eps_start    = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.steps        = 0
        self.updates      = 0

        self.q_net     = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_net= QNetwork(state_dim, hidden_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = PrioritizedReplayBuffer(buffer_size, alpha)

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps / self.eps_decay)

    def beta(self, total_steps: int = None):
        """beta 从 beta_start 线性增加到 beta_end"""
        if total_steps is None:
            return self.beta_start
        frac = min(self.steps / total_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.choice(valid_actions)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state))
            return valid_actions[q[valid_actions].argmax().item()]

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, total_steps: int = None):
        if len(self.buffer) < self.batch_size:
            return

        beta = self.beta(total_steps)
        result = self.buffer.sample(self.batch_size, beta)
        if result is None:
            return

        s, a, r, ns, d, indices, weights = result
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a).unsqueeze(1)
        r  = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d).unsqueeze(1)
        w  = torch.FloatTensor(weights).unsqueeze(1)

        q_val = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)

        # 加权TD误差（PER的关键）
        td_errors = (q_val - target.detach()).abs().detach().numpy().squeeze()
        loss = (w * F.mse_loss(q_val, target.detach(), reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        # 更新优先级
        self.buffer.update_priorities(indices, td_errors + 1e-6)

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
