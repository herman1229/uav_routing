"""
单进程顺序训练（快速验证用）
macOS 上 PyTorch 多进程 share_memory 容易死锁，用此脚本做快速迭代验证
用法: python train_single.py [--episodes 300]
"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from src.envs.fl_routing_env import FLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet, ValueNet

# ======================================================
# 参数
# ======================================================
CFG = {
    "node_capacity": 50, "link_capacity": 100.0,
    "model_size": 10.0, "t_agg": 0.5,
    "delta_t": 5.0, "num_slots": 100,
    "g_hop": -1.0, "alpha_1": 0.4, "alpha_2": 0.1,
    "w_delay": 0.2, "r_success": 20.0, "r_fail": -5.0,
    "max_steps_per_gbs": 50,
    "hidden_dim": 256, "actor_lr": 1e-3, "critic_lr": 1e-3,
    "gamma": 0.98, "output_dir": "outputs",
}


def make_env():
    return FLRoutingEnv(
        topo_cfg=TopologyConfig(node_capacity=CFG["node_capacity"], link_capacity=CFG["link_capacity"]),
        delay_cfg=DelayConfig(model_size=CFG["model_size"], t_agg=CFG["t_agg"]),
        delta_t=CFG["delta_t"], num_slots=CFG["num_slots"],
        g_hop=CFG["g_hop"], alpha_1=CFG["alpha_1"], alpha_2=CFG["alpha_2"],
        w_delay=CFG["w_delay"], r_success=CFG["r_success"], r_fail=CFG["r_fail"],
        max_steps_per_gbs=CFG["max_steps_per_gbs"],
    )


def take_action(actor, state, valid_actions):
    s = torch.FloatTensor(state)
    logits = actor(s)
    valid_logits = logits[valid_actions]
    dist = torch.distributions.Categorical(logits=valid_logits)
    idx = dist.sample().item()
    return valid_actions[idx], dist.log_prob(torch.tensor(idx)), dist.entropy()


def update(actor, critic, actor_opt, critic_opt, transitions, gamma):
    if not transitions['states']:
        return
    states = torch.FloatTensor(np.array(transitions['states']))
    rewards = torch.FloatTensor(transitions['rewards'])
    valid_sets = transitions['valid_sets']
    actions = transitions['actions']

    # 折扣回报
    R, returns = 0.0, []
    for r in reversed(rewards.tolist()):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns).view(-1, 1)

    values = critic(states)
    advantage = (returns - values).detach()
    critic_loss = F.mse_loss(values, returns.detach())

    actor_losses, entropies = [], []
    logits_all = actor(states)
    for i, (valid, action) in enumerate(zip(valid_sets, actions)):
        if not valid:
            continue
        vl = logits_all[i][valid]
        dist = torch.distributions.Categorical(logits=vl)
        local_idx = valid.index(action)
        log_p = dist.log_prob(torch.tensor(local_idx))
        actor_losses.append(-log_p * advantage[i])
        entropies.append(dist.entropy())

    if not actor_losses:
        return

    actor_loss = torch.stack(actor_losses).mean() - 0.001 * torch.stack(entropies).mean()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()


def train(n_episodes: int = 300, log_every: int = 20):
    os.makedirs(f"{CFG['output_dir']}/logs", exist_ok=True)
    os.makedirs(f"{CFG['output_dir']}/figures", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = make_env()
    actor = PolicyNet(env.obs_dim, CFG["hidden_dim"], env.action_space_n)
    critic = ValueNet(env.obs_dim, CFG["hidden_dim"])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=CFG["actor_lr"])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CFG["critic_lr"])

    print(f"单进程训练 | obs_dim={env.obs_dim} | Episodes={n_episodes}")
    print("=" * 60)

    all_results = []
    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=ep)
        transitions = {'states': [], 'actions': [], 'rewards': [], 'valid_sets': []}
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action, _, _ = take_action(actor, state, valid)
            next_state, reward, done, _, _ = env.step(action)
            transitions['states'].append(state)
            transitions['actions'].append(action)
            transitions['rewards'].append(reward)
            transitions['valid_sets'].append(valid)
            state = next_state

        update(actor, critic, actor_opt, critic_opt, transitions, CFG["gamma"])
        result = env.get_episode_result()
        t_up = result["T_up"] if result["T_up"] != float('inf') else -1.0
        all_results.append({
            "ep": ep, "reward": result["total_reward"],
            "success": result["success_count"],
            "T_up": t_up, "T_round": result["T_round"] if result["T_round"] != float('inf') else -1.0,
            "slot": result["current_slot"],
        })

        if ep % log_every == 0:
            recent = all_results[-log_every:]
            avg_r = np.mean([r["reward"] for r in recent])
            avg_succ = np.mean([r["success"] / 3 for r in recent])
            valid_tup = [r["T_up"] for r in recent if r["T_up"] > 0]
            tup_str = f"{np.mean(valid_tup):.3f}s" if valid_tup else "N/A"
            print(f"Ep {ep:4d} | AvgR={avg_r:6.2f} | Succ={avg_succ:.2f} | T_up={tup_str} | Slot={result['current_slot']}")

    # 保存
    out_json = f"{CFG['output_dir']}/logs/a3c_single_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    rewards = [r["reward"] for r in all_results]
    t_ups = np.array([r["T_up"] for r in all_results])
    np.save(f"{CFG['output_dir']}/logs/a3c_rewards_{timestamp}.npy", np.array(rewards))
    np.save(f"{CFG['output_dir']}/logs/a3c_tup_{timestamp}.npy", t_ups)

    valid_tup = t_ups[t_ups > 0]
    print("\n" + "=" * 60)
    print(f"训练完成 | Episodes={n_episodes}")
    print(f"前50平均奖励: {np.mean(rewards[:50]):.3f}")
    print(f"后50平均奖励: {np.mean(rewards[-50:]):.3f}")
    print(f"后50成功率:   {np.mean([r['success']/3 for r in all_results[-50:]]):.3f}")
    if len(valid_tup) > 0:
        print(f"平均T_up:     {np.mean(valid_tup):.3f}s")
    print(f"结果: {out_json}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()
    train(args.episodes, args.log_every)
