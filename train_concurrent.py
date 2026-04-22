"""
并发多流路由 A3C 训练脚本
用法: python train_concurrent.py [--episodes 1000]
"""
import os, sys, json, argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet, ValueNet

CFG = {
    "gbs_to_router_capacity": 20.0,
    "router_to_router_capacity": 40.0,
    "router_to_server_capacity": 80.0,
    "node_capacity": 50,
    "model_size": 10.0, "t_agg": 0.5,
    "delta_t": 5.0, "num_slots": 100,
    "g_hop": -1.0, "alpha_1": 0.4, "alpha_2": 0.1,
    "w_delay": 0.5, "r_success": 20.0, "r_fail": -5.0,
    "r_loop": -2.0, "r_compete": -1.5, "r_diversity": 0.3, "beta_tup": 3.0,
    "max_steps_per_gbs": 50,
    "hidden_dim": 256, "actor_lr": 1e-3, "critic_lr": 1e-3,
    "gamma": 0.98, "output_dir": "outputs",
}


def make_env():
    return ConcurrentFLRoutingEnv(
        topo_cfg=TopologyConfig(
            node_capacity=CFG["node_capacity"],
            gbs_to_router_capacity=CFG["gbs_to_router_capacity"],
            router_to_router_capacity=CFG["router_to_router_capacity"],
            router_to_server_capacity=CFG["router_to_server_capacity"],
        ),
        delay_cfg=DelayConfig(model_size=CFG["model_size"], t_agg=CFG["t_agg"]),
        delta_t=CFG["delta_t"], num_slots=CFG["num_slots"],
        g_hop=CFG["g_hop"], alpha_1=CFG["alpha_1"], alpha_2=CFG["alpha_2"],
        w_delay=CFG["w_delay"], r_success=CFG["r_success"], r_fail=CFG["r_fail"],
        r_loop=CFG["r_loop"], r_compete=CFG["r_compete"],
        r_diversity=CFG["r_diversity"], beta_tup=CFG["beta_tup"],
        max_steps_per_gbs=CFG["max_steps_per_gbs"],
    )


def take_action(actor, state, valid_actions):
    s = torch.FloatTensor(state)
    logits = actor(s)
    valid_logits = logits[valid_actions]
    dist = torch.distributions.Categorical(logits=valid_logits)
    idx = dist.sample().item()
    return valid_actions[idx]


def update(actor, critic, actor_opt, critic_opt, transitions, gamma):
    if not transitions['states']:
        return
    states = torch.FloatTensor(np.array(transitions['states']))
    rewards = torch.FloatTensor(transitions['rewards'])
    valid_sets = transitions['valid_sets']
    actions = transitions['actions']

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


def train(n_episodes: int = 1000, log_every: int = 50):
    os.makedirs(f"{CFG['output_dir']}/logs", exist_ok=True)
    os.makedirs(f"{CFG['output_dir']}/figures", exist_ok=True)
    os.makedirs(f"{CFG['output_dir']}/models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = make_env()
    actor = PolicyNet(env.obs_dim, CFG["hidden_dim"], env.action_space_n)
    critic = ValueNet(env.obs_dim, CFG["hidden_dim"])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=CFG["actor_lr"])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CFG["critic_lr"])

    print(f"并发多流训练 | obs_dim={env.obs_dim} | Episodes={n_episodes}")
    print(f"链路容量: GBS->R={CFG['gbs_to_router_capacity']}Mbps, "
          f"R->R={CFG['router_to_router_capacity']}Mbps, "
          f"R->S={CFG['router_to_server_capacity']}Mbps")
    print("=" * 65)

    all_results = []
    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=ep)
        transitions = {'states': [], 'actions': [], 'rewards': [], 'valid_sets': []}
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = take_action(actor, state, valid)
            next_state, reward, done, _, info = env.step(action)
            transitions['states'].append(state)
            transitions['actions'].append(action)
            transitions['rewards'].append(reward)
            transitions['valid_sets'].append(valid)
            state = next_state

        update(actor, critic, actor_opt, critic_opt, transitions, CFG["gamma"])
        result = env.get_episode_result()
        t_up = result["T_up"] if result["T_up"] != float('inf') else -1.0
        all_results.append({
            "ep": ep,
            "reward": result["total_reward"],
            "success": result["success_count"],
            "T_up": t_up,
            "diversity": result["path_diversity"],
            "slot": result["current_slot"],
        })

        if ep % log_every == 0:
            recent = all_results[-log_every:]
            avg_r = np.mean([r["reward"] for r in recent])
            avg_s = np.mean([r["success"] / 3 for r in recent])
            valid_t = [r["T_up"] for r in recent if r["T_up"] > 0]
            avg_div = np.mean([r["diversity"] for r in recent])
            tup_str = f"{np.mean(valid_t):.2f}s" if valid_t else "N/A"
            print(f"Ep {ep:4d} | R={avg_r:6.2f} | Succ={avg_s:.2f} | "
                  f"T_up={tup_str} | Diversity={avg_div:.2f}")

    # 保存
    out_json = f"{CFG['output_dir']}/logs/a3c_concurrent_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    rewards = np.array([r["reward"] for r in all_results])
    t_ups = np.array([r["T_up"] for r in all_results])
    np.save(f"{CFG['output_dir']}/logs/concurrent_rewards_{timestamp}.npy", rewards)
    np.save(f"{CFG['output_dir']}/logs/concurrent_tup_{timestamp}.npy", t_ups)

    model_path = f"{CFG['output_dir']}/models/concurrent_actor_{timestamp}.pth"
    torch.save(actor.state_dict(), model_path)

    valid_t = t_ups[t_ups > 0]
    print("\n" + "=" * 65)
    print(f"训练完成 | Episodes={n_episodes}")
    print(f"前50奖励: {np.mean(rewards[:50]):.3f} → 后50奖励: {np.mean(rewards[-50:]):.3f}")
    print(f"后50成功率: {np.mean([r['success']/3 for r in all_results[-50:]]):.3f}")
    if len(valid_t) > 0:
        print(f"平均T_up: {np.mean(valid_t):.3f}s | 后50 T_up: {np.mean(valid_t[-50:]):.3f}s")
    print(f"模型: {model_path}")
    print("=" * 65)

    return all_results, actor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    train(args.episodes, args.log_every)
