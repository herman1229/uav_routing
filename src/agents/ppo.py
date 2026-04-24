"""
PPO (Proximal Policy Optimization) - 单进程版
解决A3C不稳定收敛的根本方案

A3C不稳定的根本原因：
  每步梯度更新无约束，一次坏的episode可能让策略大幅偏移，
  导致陷入局部最优（Diversity=2）后很难逃出。

PPO的解决机制：
  clip ratio = 0.2 → 每次更新策略变化幅度被硬性限制在 [0.8, 1.2] 范围内
  → 即使遇到坏的episode，策略也不会崩溃
  → 多个epoch充分利用每个episode的数据，梯度更稳定
  → GAE优势估计比n-step TD方差更小

与改进A3C的关系：
  保留所有创新点（竞争感知状态、复合奖励、并发建模）
  只替换训练算法：policy gradient → PPO clip objective
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class PPOActor(nn.Module):
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

    def get_action(self, state: torch.Tensor, valid_actions: List[int]):
        logits = self.forward(state)
        valid_logits = logits[valid_actions]
        dist = torch.distributions.Categorical(logits=valid_logits)
        idx = dist.sample()
        return valid_actions[idx.item()], dist.log_prob(idx), dist.entropy()

    def evaluate_actions(self, states: torch.Tensor,
                         action_indices: torch.Tensor,
                         valid_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量计算 log_prob 和 entropy（PPO更新时使用）
        valid_masks: [B, action_dim]，有效动作位置为1
        action_indices: [B]，在 valid_actions 中的局部下标
        """
        logits = self.forward(states)
        # 将无效动作的 logit 设为极小值
        masked_logits = logits + (1.0 - valid_masks) * (-1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(action_indices)
        entropy   = dist.entropy()
        return log_probs, entropy


class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOAgent:
    """
    单进程 PPO 智能体
    保留改进A3C的所有创新点，仅替换训练目标函数
    """

    def __init__(self,
                 state_dim:    int,
                 hidden_dim:   int,
                 action_dim:   int,
                 actor_lr:     float = 3e-4,
                 critic_lr:    float = 1e-3,
                 gamma:        float = 0.98,
                 gae_lambda:   float = 0.95,
                 clip_ratio:   float = 0.2,
                 ppo_epochs:   int   = 4,
                 entropy_coef: float = 0.01,
                 max_grad_norm:float = 0.5,
                 ent_anneal:   bool  = True):

        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_ratio   = clip_ratio
        self.ppo_epochs   = ppo_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm= max_grad_norm
        self.ent_anneal   = ent_anneal
        self.action_dim   = action_dim

        self.actor  = PPOActor(state_dim, hidden_dim, action_dim)
        self.critic = PPOCritic(state_dim, hidden_dim)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray, valid_actions: List[int]):
        with torch.no_grad():
            s = torch.FloatTensor(state)
            action, log_prob, _ = self.actor.get_action(s, valid_actions)
        return action, log_prob.item()

    def update(self, rollout: Dict, ent_coef: float = None) -> Dict[str, float]:
        """
        PPO 更新
        rollout 字段：
          states       : [T, state_dim]
          actions_idx  : [T]  在 valid_actions 中的局部下标
          old_log_probs: [T]
          rewards      : [T]
          dones        : [T]
          valid_masks  : [T, action_dim]
          next_states  : [T, state_dim]  （用于 bootstrap）
        """
        if ent_coef is None:
            ent_coef = self.entropy_coef

        states       = torch.FloatTensor(np.array(rollout['states']))
        next_states  = torch.FloatTensor(np.array(rollout['next_states']))
        actions_idx  = torch.LongTensor(rollout['actions_idx'])
        old_log_probs= torch.FloatTensor(rollout['old_log_probs'])
        rewards      = rollout['rewards']
        dones        = rollout['dones']
        valid_masks  = torch.FloatTensor(np.array(rollout['valid_masks']))

        # ① GAE 优势估计（比 n-step TD 方差更小）
        with torch.no_grad():
            values      = self.critic(states).squeeze(-1).numpy()
            next_values = self.critic(next_states).squeeze(-1).numpy()

        returns, advantages = self._gae(rewards, values, next_values, dones)
        returns    = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        # 归一化优势（稳定训练）
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ② 多 epoch PPO 更新
        actor_losses, critic_losses, entropies, clip_fracs = [], [], [], []
        for _ in range(self.ppo_epochs):
            # Actor
            new_log_probs, entropy = self.actor.evaluate_actions(
                states, actions_idx, valid_masks)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # PPO clip 目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                        1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() \
                         - ent_coef * entropy.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            # Critic
            new_values  = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(new_values, returns.detach())

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

            # 记录 clip 比例（监控策略更新幅度）
            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > self.clip_ratio).float().mean()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.mean().item())
            clip_fracs.append(clip_frac.item())

        return {
            'actor_loss':  float(np.mean(actor_losses)),
            'critic_loss': float(np.mean(critic_losses)),
            'entropy':     float(np.mean(entropies)),
            'clip_frac':   float(np.mean(clip_fracs)),
        }

    def _gae(self, rewards, values, next_values, dones):
        T = len(rewards)
        advantages = np.zeros(T)
        returns    = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            nv = next_values[t] * (1 - dones[t])
            delta = rewards[t] + self.gamma * nv - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]
        return returns, advantages
