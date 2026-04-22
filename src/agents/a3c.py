"""
A3C 智能体
- 动作mask在采样和log_prob中保持一致
- Worker在内部独立创建env实例
- 支持时延指标记录
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from typing import List

from ..envs.fl_routing_env import FLRoutingEnv
from ..envs.topology import TopologyConfig
from ..envs.delay_model import DelayConfig


# ------------------------------------------------------------------
# 网络定义
# ------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

class Worker(mp.Process):
    def __init__(self, worker_id: int, global_actor: PolicyNet, global_critic: ValueNet,
                 actor_opt: SharedAdam, critic_opt: SharedAdam,
                 global_ep: mp.Value, res_queue: mp.Queue, log_queue: mp.Queue,
                 max_episodes: int, gamma: float,
                 state_dim: int, hidden_dim: int, action_dim: int,
                 env_kwargs: dict):
        super().__init__()
        self.id = worker_id
        self.name = f"w{worker_id:02d}"
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.global_ep = global_ep
        self.res_queue = res_queue
        self.log_queue = log_queue
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.env_kwargs = env_kwargs

        self.local_actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.local_critic = ValueNet(state_dim, hidden_dim)

    def _sync(self):
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def _take_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """动作mask保持一致：采样和log_prob均基于有效动作子集"""
        s = torch.FloatTensor(state)
        logits = self.local_actor(s)
        valid_logits = logits[valid_actions]
        dist = torch.distributions.Categorical(logits=valid_logits)
        idx = dist.sample().item()
        return valid_actions[idx]

    def _update(self, transitions: dict):
        states = torch.FloatTensor(np.array(transitions['states']))
        actions = transitions['actions']       # List[int] 原始节点编号
        rewards = torch.FloatTensor(transitions['rewards'])
        valid_sets = transitions['valid_sets'] # List[List[int]]

        # 折扣回报
        R = 0.0
        returns = []
        for r in reversed(rewards.tolist()):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).view(-1, 1)

        values = self.local_critic(states)
        advantage = (returns - values).detach()

        # Critic loss
        critic_loss = F.mse_loss(values, returns.detach())

        # Actor loss（mask一致）
        actor_losses = []
        entropy_list = []
        logits_all = self.local_actor(states)
        for i, (valid, action) in enumerate(zip(valid_sets, actions)):
            if not valid:
                continue
            vl = logits_all[i][valid]
            dist = torch.distributions.Categorical(logits=vl)
            local_idx = valid.index(action)
            log_p = dist.log_prob(torch.tensor(local_idx))
            actor_losses.append(-log_p * advantage[i])
            entropy_list.append(dist.entropy())

        if not actor_losses:
            return

        actor_loss = torch.stack(actor_losses).mean()
        entropy = torch.stack(entropy_list).mean()
        actor_loss = actor_loss - 0.001 * entropy

        # 更新全局网络
        self.critic_opt.zero_grad()
        critic_loss.backward()
        for lp, gp in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            if lp.grad is not None:
                gp._grad = lp.grad
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        for lp, gp in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            if lp.grad is not None:
                gp._grad = lp.grad
        self.actor_opt.step()

    def run(self):
        env = FLRoutingEnv(**self.env_kwargs)

        while self.global_ep.value < self.max_episodes:
            self._sync()
            state = env.reset(seed=self.id)
            transitions = {'states': [], 'actions': [], 'rewards': [], 'valid_sets': []}
            done = False

            while not done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                action = self._take_action(state, valid)
                next_state, reward, done, _, info = env.step(action)

                transitions['states'].append(state)
                transitions['actions'].append(action)
                transitions['rewards'].append(reward)
                transitions['valid_sets'].append(valid)
                state = next_state

            result = env.get_episode_result()
            self._update(transitions)

            with self.global_ep.get_lock():
                self.global_ep.value += 1

            self.res_queue.put({
                "reward": result["total_reward"],
                "success": result["success_count"],
                "T_up": result["T_up"],
                "T_round": result["T_round"],
                "slot": result["current_slot"],
                "worker": self.name,
                "ep": self.global_ep.value,
            })

            print(f"{self.name} | Ep:{self.global_ep.value:4d} | "
                  f"R:{result['total_reward']:6.2f} | "
                  f"Succ:{result['success_count']}/{env.num_gbs} | "
                  f"T_up:{result['T_up']:5.2f}s | "
                  f"Slot:{result['current_slot']}")

        self.res_queue.put(None)
        print(f"{self.name} finished.")


# ------------------------------------------------------------------
# A3C Agent（协调器）
# ------------------------------------------------------------------

class A3CAgent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 actor_lr: float, critic_lr: float,
                 num_workers: int, max_episodes: int, gamma: float,
                 env_kwargs: dict):
        self.global_actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.global_critic = ValueNet(state_dim, hidden_dim)
        self.global_actor.share_memory()
        self.global_critic.share_memory()
        self.actor_opt = SharedAdam(self.global_actor.parameters(), lr=actor_lr)
        self.critic_opt = SharedAdam(self.global_critic.parameters(), lr=critic_lr)

        self.global_ep = mp.Value('i', 0)
        self.res_queue = mp.Queue()
        self.log_queue = mp.Queue()

        self.workers = [
            Worker(i, self.global_actor, self.global_critic,
                   self.actor_opt, self.critic_opt,
                   self.global_ep, self.res_queue, self.log_queue,
                   max_episodes, gamma,
                   state_dim, hidden_dim, action_dim, env_kwargs)
            for i in range(num_workers)
        ]

    def train(self) -> List[dict]:
        [w.start() for w in self.workers]
        results = []
        finished = 0
        while finished < len(self.workers):
            item = self.res_queue.get()
            if item is None:
                finished += 1
            else:
                results.append(item)
        [w.join() for w in self.workers]
        return results
