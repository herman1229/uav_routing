"""
MAPPO (Multi-Agent PPO) with CTDE
集中训练分布执行（Centralized Training, Decentralized Execution）

核心设计：
- 每个GBS有独立的Actor（参数共享，但输入是各自的局部观测）
- 集中式Critic：输入全局状态（所有GBS观测拼接），输出全局价值
- PPO clip目标函数，比A3C更稳定
- 共享全局奖励（T_up相关）+ 个人即时奖励

与改进A3C的核心区别：
- A3C：单智能体轮流决策，状态包含所有GBS位置
- MAPPO：N个智能体同时决策，Critic集中保证协调
- MAPPO的Actor只用局部观测，更符合分布式执行的实际场景
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class MAPPOActor(nn.Module):
    """
    局部Actor：只用当前GBS的局部观测做决策
    参数在所有GBS间共享（同质智能体）
    """
    def __init__(self, local_obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        return self.net(local_obs)

    def get_action(self, local_obs: torch.Tensor, valid_actions: List[int]):
        logits = self.forward(local_obs)
        valid_logits = logits[valid_actions]
        dist = torch.distributions.Categorical(logits=valid_logits)
        idx = dist.sample()
        action = valid_actions[idx.item()]
        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, local_obs: torch.Tensor,
                 actions_idx: torch.Tensor,
                 valid_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算给定动作的log_prob和entropy（用于PPO更新）
        valid_masks: [batch, action_dim]，1表示有效动作
        actions_idx: [batch]，在valid_actions中的索引
        """
        logits = self.forward(local_obs)
        # mask无效动作（设为极小值）
        masked_logits = logits + (1 - valid_masks) * (-1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions_idx)
        entropy = dist.entropy()
        return log_probs, entropy


class MAPPOCritic(nn.Module):
    """
    集中式Critic：输入全局状态（所有GBS局部观测拼接）
    训练时可见全局信息，执行时不需要
    """
    def __init__(self, global_state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


class MAPPOAgent:
    """
    MAPPO智能体（参数共享版本）
    所有GBS共享同一个Actor网络，但各自输入局部观测
    集中式Critic输入全局状态
    """
    def __init__(self,
                 local_obs_dim: int,
                 global_state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.98,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 ppo_epochs: int = 4,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim

        self.actor  = MAPPOActor(local_obs_dim, hidden_dim, action_dim)
        self.critic = MAPPOCritic(global_state_dim, hidden_dim)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, local_obs: np.ndarray, valid_actions: List[int]):
        """执行时只用局部观测"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(local_obs)
            action, log_prob, _ = self.actor.get_action(obs_t, valid_actions)
        return action, log_prob.item()

    def update(self, rollout: Dict) -> Dict[str, float]:
        """
        PPO更新
        rollout包含：
          local_obs: [T, local_obs_dim]
          global_states: [T, global_state_dim]
          actions_idx: [T]  (在valid_actions中的局部索引)
          old_log_probs: [T]
          rewards: [T]
          dones: [T]
          valid_masks: [T, action_dim]
        """
        local_obs    = torch.FloatTensor(np.array(rollout['local_obs']))
        global_states= torch.FloatTensor(np.array(rollout['global_states']))
        actions_idx  = torch.LongTensor(rollout['actions_idx'])
        old_log_probs= torch.FloatTensor(rollout['old_log_probs'])
        rewards      = rollout['rewards']
        dones        = rollout['dones']
        valid_masks  = torch.FloatTensor(np.array(rollout['valid_masks']))

        # GAE计算
        with torch.no_grad():
            values = self.critic(global_states).squeeze(-1)
        returns, advantages = self._compute_gae(rewards, values.numpy(), dones)
        returns    = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO多轮更新
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.ppo_epochs):
            # Actor
            new_log_probs, entropy = self.actor.evaluate(
                local_obs, actions_idx, valid_masks)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            # Critic
            new_values = self.critic(global_states).squeeze(-1)
            critic_loss = F.mse_loss(new_values, returns)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.mean().item())

        return {
            'actor_loss':  np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy':     np.mean(entropies),
        }

    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""
        T = len(rewards)
        returns    = np.zeros(T)
        advantages = np.zeros(T)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae   = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]
            next_value    = values[t]

        return returns, advantages
