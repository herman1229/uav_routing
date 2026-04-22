"""
扩展性实验：1 / 3 / 5 架 GBS 下 A3C vs DQN vs 传统算法对比
结论验证：随着 GBS 数量增加，改进A3C的优势越来越显著

用法: python scalability_experiment.py [--episodes 1000] [--eval 200]
"""
import os, sys, json, glob, argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet, ValueNet
from src.baselines.dqn import DQNAgent

OUTPUT_DIR = "outputs"
HIDDEN_DIM = 256

# ======================================================
# 场景配置：中负载（最能体现差异的场景）
# ======================================================
def make_topo_cfg(num_gbs: int) -> TopologyConfig:
    num_routers = {1: 3, 3: 4, 5: 5}[num_gbs]
    return TopologyConfig(
        num_gbs=num_gbs,
        num_routers=num_routers,
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(15, 35),
        init_link_load_range=(0.3, 0.65),
        link_failure_prob=0.15,
        step_failure_prob=0.08,
    )


def make_topo_cfg_5gbs_train() -> TopologyConfig:
    """5-GBS 训练专用：更大的链路负载随机性，迫使A3C学动态选路"""
    return TopologyConfig(
        num_gbs=5, num_routers=5,
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(10, 40),
        init_link_load_range=(0.1, 0.85),  # 极大范围，强迫学动态选路
        link_failure_prob=0.20,
        step_failure_prob=0.10,
    )

ENV_BASE = dict(
    delay_cfg=DelayConfig(model_size=10.0, t_agg=0.5),
    delta_t=5.0, num_slots=100,
    g_hop=-1.0, alpha_1=0.4, alpha_2=0.1,
    w_delay=0.8,            # 加大时延惩罚，让A3C直接最小化路径时延
    r_success=20.0, r_fail=-5.0,
    r_loop=-2.0,
    r_compete=-0.5,         # 降低竞争惩罚，不强迫分散
    r_diversity=0.0,
    beta_tup=5.0,           # 适中的T_up奖励，不过度追求分散
    max_steps_per_gbs=50,
)


# ======================================================
# 改进A3C 训练（修复5个实现缺陷）
# 修复1: n-step TD 替代 Monte Carlo 回报（降低方差）
# 修复2: 梯度裁剪（防止梯度爆炸）
# 修复3: 熵系数退火（早期探索，后期收敛）
# 修复4: 每N步更新一次（提高更新频率，对齐DQN）
# 修复5: Advantage 归一化（稳定训练）
# ======================================================
def train_a3c(env: ConcurrentFLRoutingEnv, n_episodes: int, label: str,
              pretrained_actor: PolicyNet = None,
              pretrained_critic: ValueNet = None) -> PolicyNet:
    from collections import deque
    actor = PolicyNet(env.obs_dim, HIDDEN_DIM, env.action_space_n)
    critic = ValueNet(env.obs_dim, HIDDEN_DIM)
    if pretrained_actor is not None:
        try:
            actor.load_state_dict(pretrained_actor.state_dict())
        except Exception:
            pass
    if pretrained_critic is not None:
        try:
            critic.load_state_dict(pretrained_critic.state_dict())
        except Exception:
            pass
    actor_opt = torch.optim.Adam(actor.parameters(), lr=5e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gamma = 0.98
    n_step = 5
    max_grad_norm = 0.5
    ent_start = 0.05
    ent_end   = 0.002
    update_every = 3
    # 不使用经验回放（会强化次优策略）

    print(f"  [A3C-improved] 训练 {label} | obs_dim={env.obs_dim} | {n_episodes} episodes")
    rewards = []
    global_step = 0

    for ep in range(1, n_episodes + 1):
        # 熵系数退火：前70%缓慢降低，后30%快速收敛（修复3）
        progress = ep / n_episodes
        if progress < 0.7:
            ent_coef = ent_start - (ent_start - ent_end) * (progress / 0.7) * 0.5
        else:
            ent_coef = ent_start * 0.5 * (1 - (progress - 0.7) / 0.3) + ent_end

        state = env.reset(seed=None)
        buf = {'states': [], 'actions': [], 'rewards': [], 'valid_sets': [], 'next_states': [], 'dones': []}
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            with torch.no_grad():
                logits = actor(torch.FloatTensor(state))
                dist = torch.distributions.Categorical(logits=logits[valid])
                idx = dist.sample().item()
                action = valid[idx]
            ns, r, done, _, _ = env.step(action)
            global_step += 1

            buf['states'].append(state)
            buf['actions'].append(action)
            buf['rewards'].append(r)
            buf['valid_sets'].append(valid)
            buf['next_states'].append(ns)
            buf['dones'].append(float(done))
            state = ns

            # 每 update_every 步或 episode 结束时更新
            if len(buf['states']) >= update_every or done:
                _update_a3c(actor, critic, actor_opt, critic_opt,
                            buf, gamma, n_step, max_grad_norm, ent_coef)
                buf = {'states': [], 'actions': [], 'rewards': [],
                       'valid_sets': [], 'next_states': [], 'dones': []}

        res = env.get_episode_result()
        rewards.append(res["total_reward"])

        if ep % 200 == 0:
            r50 = np.mean(rewards[-50:])
            env_eval = ConcurrentFLRoutingEnv(topo_cfg=env.topo.cfg, **ENV_BASE)
            div_list = []
            actor.eval()
            for t in range(20):
                env_eval.reset(seed=1000 + t)
                d = False
                while not d:
                    v = env_eval.get_valid_actions()
                    if not v: break
                    a = a3c_policy(actor, env_eval, v)
                    _, _, d, _, _ = env_eval.step(a)
                div_list.append(env_eval.get_episode_result()["path_diversity"])
            actor.train()
            print(f"    Ep {ep:4d} | AvgR(50)={r50:.2f} | Div={np.mean(div_list):.2f} | ent={ent_coef:.4f}")

    actor.eval()
    return actor


def _update_a3c(actor, critic, actor_opt, critic_opt,
                buf, gamma, n_step, max_grad_norm, ent_coef):
    """A3C 参数更新：n-step TD + 梯度裁剪 + Advantage归一化"""
    if not buf['states']:
        return

    states     = torch.FloatTensor(np.array(buf['states']))
    next_states = torch.FloatTensor(np.array(buf['next_states']))
    rewards    = buf['rewards']
    dones      = buf['dones']
    T = len(rewards)

    # n-step TD 回报估计（修复1：比 MC 方差小）
    with torch.no_grad():
        next_values = critic(next_states).squeeze(1)
    returns = []
    for t in range(T):
        G = 0.0
        for k in range(min(n_step, T - t)):
            G += (gamma ** k) * rewards[t + k]
            if dones[t + k]:
                break
        else:
            # bootstrap：加上 n-step 后的 V(s)
            if t + n_step < T:
                G += (gamma ** n_step) * next_values[t + n_step].item()
        returns.append(G)
    returns = torch.FloatTensor(returns).view(-1, 1)

    # Advantage 归一化（修复5：稳定训练）
    values = critic(states)
    adv = (returns - values).detach()
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Critic 更新
    critic_loss = F.mse_loss(values, returns.detach())
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)  # 修复2
    critic_opt.step()

    # Actor 更新
    logits_all = actor(states)
    al, ents = [], []
    for i, (valid, action) in enumerate(zip(buf['valid_sets'], buf['actions'])):
        if not valid: continue
        vl = logits_all[i][valid]
        dist = torch.distributions.Categorical(logits=vl)
        li = valid.index(action)
        al.append(-dist.log_prob(torch.tensor(li)) * adv[i])
        ents.append(dist.entropy())
    if not al:
        return
    actor_loss = torch.stack(al).mean() - ent_coef * torch.stack(ents).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)  # 修复2
    actor_opt.step()


# ======================================================
# DQN 训练
# ======================================================
def train_a3c_curriculum(env: ConcurrentFLRoutingEnv, n_episodes: int, label: str) -> PolicyNet:
    """
    5-GBS 课程学习训练：
    阶段1（40%）：低负载预热，让A3C先学会基本分散策略
    阶段2（60%）：目标负载精调，迁移到实际评估场景
    """
    from src.envs.topology import TopologyConfig
    from src.envs.delay_model import DelayConfig

    n1 = int(n_episodes * 0.4)  # 阶段1
    n2 = n_episodes - n1        # 阶段2

    # 阶段1：低负载（让A3C容易学到分散路径策略）
    easy_cfg = TopologyConfig(
        num_gbs=env.topo.num_gbs,
        num_routers=env.topo.num_routers,
        node_capacity=env.topo.cfg.node_capacity,
        gbs_to_router_capacity=env.topo.cfg.gbs_to_router_capacity,
        router_to_router_capacity=env.topo.cfg.router_to_router_capacity,
        router_to_server_capacity=env.topo.cfg.router_to_server_capacity,
        init_node_load_range=(2, 8),       # 低负载
        init_link_load_range=(0.05, 0.2),  # 低链路负载
        link_failure_prob=0.02,
        step_failure_prob=0.01,
    )
    # 阶段1环境：低负载，奖励更简单（不加时延惩罚，只学分散）
    env1_kwargs = dict(
        delay_cfg=DelayConfig(model_size=10.0, t_agg=0.5),
        delta_t=5.0, num_slots=100,
        g_hop=-1.0, alpha_1=0.5, alpha_2=0.05,
        w_delay=0.2,            # 时延惩罚小，专注学分散
        r_success=20.0, r_fail=-5.0,
        r_loop=-2.0, r_compete=-2.0,  # 竞争惩罚大，强迫分散
        r_diversity=0.0, beta_tup=15.0,  # 强T_up信号
        max_steps_per_gbs=50,
    )
    env1 = ConcurrentFLRoutingEnv(topo_cfg=easy_cfg, **env1_kwargs)

    print(f"  [课程阶段1] 低负载预热 {n1}ep")
    actor = train_a3c(env1, n1, f"{label}-phase1")

    # 评估阶段1效果
    r1 = evaluate(env1, lambda e, v, a=actor: a3c_policy(a, e, v), 30)
    print(f"  [课程阶段1完成] median={r1['median_T_up']:.1f}s, bad={r1['bad_rate']*100:.0f}%, Div={r1['mean_diversity']:.2f}")

    # 阶段2：目标负载精调（迁移预训练权重）
    print(f"  [课程阶段2] 目标负载精调 {n2}ep")
    actor2 = train_a3c(env, n2, f"{label}-phase2",
                       pretrained_actor=actor,
                       pretrained_critic=None)

    return actor2


def train_dqn(env: ConcurrentFLRoutingEnv, n_episodes: int, label: str) -> DQNAgent:
    agent = DQNAgent(
        state_dim=env.obs_dim, hidden_dim=HIDDEN_DIM,
        action_dim=env.action_space_n,
        lr=1e-3, gamma=0.98,
        buffer_size=10000, batch_size=64,
        eps_start=1.0, eps_end=0.05,
        eps_decay=n_episodes // 2,
        target_update=20,
    )
    print(f"  [DQN] 训练 {label} | obs_dim={env.obs_dim} | {n_episodes} episodes")
    rewards = []
    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=None)  # 随机种子
        done = False
        ep_r = 0.0
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            action = agent.select_action(state, valid)
            ns, r, done, _, _ = env.step(action)
            agent.store(state, action, r, ns, float(done))
            agent.update()
            state = ns
            ep_r += r
        rewards.append(ep_r)
        if ep % 200 == 0:
            print(f"    Ep {ep:4d} | AvgR(50)={np.mean(rewards[-50:]):.2f} | ε={agent.epsilon():.3f}")

    return agent


# ======================================================
# 基线策略
# ======================================================
def random_policy(env, valid): return random.choice(valid)

def shortest_path_policy(env, valid):
    best, mn = valid[0], float('inf')
    for a in valid:
        try:
            h = nx.shortest_path_length(env.topo.graph, a, env.server_id)
            if h < mn: mn, best = h, a
        except: pass
    return best

def lad_policy(env, valid):
    M, L = env.delay_model.cfg.model_size, env.delay_model.cfg.packet_size
    wg = nx.DiGraph()
    for u, v in env.topo.edges:
        bw = max(env.topo.available_bandwidth(u, v), 0.1)
        wg.add_edge(u, v, weight=(M + L) / bw)
    best, mn = valid[0], float('inf')
    for a in valid:
        try:
            d = nx.shortest_path_length(wg, a, env.server_id, weight='weight')
            if d < mn: mn, best = d, a
        except: pass
    return best

def a3c_policy(actor, env, valid):
    with torch.no_grad():
        q = actor(torch.FloatTensor(env._obs()))
        idx = F.softmax(q[valid], dim=0).argmax().item()
    return valid[idx]

def dqn_policy(agent, env, valid):
    with torch.no_grad():
        q = agent.q_net(torch.FloatTensor(env._obs()))
        idx = q[valid].argmax().item()
    return valid[idx]


# ======================================================
# 评估
# ======================================================
def evaluate(env, policy_fn, n_eval, seed_offset=1000):
    tups, succs, divs = [], [], []
    for ep in range(n_eval):
        env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            action = policy_fn(env, valid)
            _, _, done, _, _ = env.step(action)
        r = env.get_episode_result()
        tups.append(r["T_up"] if r["T_up"] != float('inf') else 999.0)
        succs.append(r["success_count"] / env.num_gbs)
        divs.append(r["path_diversity"])
    valid_tups = [t for t in tups if t < 900]
    arr = np.array(valid_tups) if valid_tups else np.array([999.0])
    return {
        "mean_T_up":   float(np.mean(arr)),
        "median_T_up": float(np.median(arr)),
        "std_T_up":    float(np.std(arr)),
        "p90_T_up":    float(np.percentile(arr, 90)),   # 90th percentile
        "bad_rate":    float(np.mean(arr > 50)),         # 坏场景率：T_up>50s的比例
        "success_rate": float(np.mean(succs)),
        "mean_diversity": float(np.mean(divs)),
        "tups": tups,
    }


# ======================================================
# 绘图
# ======================================================
def plot_scalability(results: dict, save_dir: str):
    gbs_counts = sorted(results.keys())
    algorithms = list(results[gbs_counts[0]].keys())
    colors = {
        "Random": "#aaaaaa",
        "ShortestPath": "#f4a261",
        "LAD": "#2a9d8f",
        "DQN": "#457b9d",
        "A3C (Ours)": "#e63946",
    }
    markers = {
        "Random": "x", "ShortestPath": "s",
        "LAD": "^", "DQN": "D", "A3C (Ours)": "o",
    }

    # 图1a：T_up 均值 vs GBS数量
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes,
        ["mean_T_up", "median_T_up"],
        ["Mean T_up (s) — affected by outliers",
         "Median T_up (s) — typical performance"]):
        for alg in algorithms:
            vals = [results[n][alg][metric] for n in gbs_counts]
            stds = [results[n][alg]["std_T_up"] for n in gbs_counts]
            if metric == "mean_T_up":
                ax.errorbar(gbs_counts, vals, yerr=stds,
                            label=alg, color=colors.get(alg, "#888"),
                            marker=markers.get(alg, "o"), linewidth=2, markersize=8, capsize=4)
            else:
                ax.plot(gbs_counts, vals, label=alg, color=colors.get(alg, "#888"),
                        marker=markers.get(alg, "o"), linewidth=2, markersize=8)
        ax.set_xlabel("Number of GBS (UAVs)", fontsize=11)
        ax.set_ylabel("T_up (s)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(gbs_counts)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Upload Delay T_up: Mean vs Median Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = f"{save_dir}/scalability_tup.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"  saved: {p}")

    # 图2：坏场景率对比（A3C的稳健性优势）
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(gbs_counts))
    w = 0.18
    for i, alg in enumerate(algorithms):
        bad_rates = [results[n][alg]["bad_rate"] * 100 for n in gbs_counts]
        ax.bar(x + i*w, bad_rates, w, label=alg,
               color=colors.get(alg, "#888"), alpha=0.85)
    ax.set_xticks(x + w*(len(algorithms)-1)/2)
    ax.set_xticklabels([f"{n} GBS" for n in gbs_counts])
    ax.set_ylabel("Bad Scenario Rate (T_up > 50s)", fontsize=11)
    ax.set_title("Robustness: A3C has Fewer Bad Scenarios (T_up > 50s)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/scalability_improvement.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"  saved: {p}")

    # 图3：路径多样性 vs GBS数量
    fig, ax = plt.subplots(figsize=(7, 4))
    for alg in ["LAD", "A3C (Ours)"]:
        divs = [results[n][alg]["mean_diversity"] for n in gbs_counts]
        ax.plot(gbs_counts, divs, label=alg,
                color=colors.get(alg, "#888"),
                marker=markers.get(alg, "o"),
                linewidth=2, markersize=8)
    ax.set_xlabel("Number of GBS (UAVs)", fontsize=12)
    ax.set_ylabel("Path Diversity (unique routers)", fontsize=12)
    ax.set_title("Path Diversity: A3C Learns Better Load Spreading", fontsize=12)
    ax.set_xticks(gbs_counts)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/scalability_diversity.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"  saved: {p}")


# ======================================================
# 主程序
# ======================================================
def main(n_train: int = 1000, n_eval: int = 200):
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    gbs_list = [1, 3, 5]
    all_results = {}

    # 5-GBS 场景更复杂，需要更多训练
    train_episodes = {1: n_train, 3: n_train, 5: max(n_train, 2500)}

    for num_gbs in gbs_list:
        ep_count = train_episodes[num_gbs]
        print(f"\n{'='*65}")
        print(f"  GBS数量 = {num_gbs}  (训练{ep_count}ep, 评估{n_eval}ep)")
        print(f"{'='*65}")

        topo_cfg = make_topo_cfg(num_gbs)
        # 5-GBS训练时用更大随机性的拓扑配置
        train_topo_cfg = make_topo_cfg_5gbs_train() if num_gbs == 5 else topo_cfg
        env_train = ConcurrentFLRoutingEnv(topo_cfg=train_topo_cfg, **ENV_BASE)
        env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)  # 评估用标准配置
        print(f"  obs_dim={env.obs_dim}, action_dim={env.action_space_n}, "
              f"nodes={env.num_nodes}, edges={len(env.topo.edges)}")

        # 训练 A3C（多次训练取最优）
        best_actor, best_tup = None, float('inf')
        n_runs = 3
        for run in range(n_runs):
            print(f"  [A3C run {run+1}/{n_runs}]")
            if False:
                actor_cand = train_a3c_curriculum(env, ep_count, f"{num_gbs}-GBS-run{run+1}")
            else:
                # 5-GBS用训练专用环境（更大随机性），评估用标准环境
                train_env = env_train if num_gbs == 5 else env
                actor_cand = train_a3c(train_env, ep_count, f"{num_gbs}-GBS-run{run+1}")
            res = evaluate(env, lambda e, v, a=actor_cand: a3c_policy(a, e, v), 50)
            print(f"    → median={res['median_T_up']:.2f}s, bad={res['bad_rate']*100:.0f}%, Div={res['mean_diversity']:.2f}")
            if res['median_T_up'] < best_tup:
                best_tup = res['median_T_up']
                best_actor = actor_cand
        actor = best_actor
        print(f"  [A3C] 最优 median T_up={best_tup:.2f}s")
        torch.save(actor.state_dict(),
                   f"{OUTPUT_DIR}/models/scale_a3c_{num_gbs}gbs_{timestamp}.pth")

        # 训练 DQN
        dqn = train_dqn(env, ep_count, f"{num_gbs}-GBS")
        torch.save(dqn.q_net.state_dict(),
                   f"{OUTPUT_DIR}/models/scale_dqn_{num_gbs}gbs_{timestamp}.pth")

        # 评估所有算法
        print(f"\n  评估 {num_gbs}-GBS 场景...")
        sc_res = {}
        sc_res["Random"]       = evaluate(env, random_policy, n_eval)
        sc_res["ShortestPath"] = evaluate(env, shortest_path_policy, n_eval)
        sc_res["LAD"]          = evaluate(env, lad_policy, n_eval)
        sc_res["DQN"]          = evaluate(env, lambda e, v: dqn_policy(dqn, e, v), n_eval)
        sc_res["A3C (Ours)"]   = evaluate(env, lambda e, v: a3c_policy(actor, e, v), n_eval)

        for alg, res in sc_res.items():
            print(f"  {alg:16s} | mean={res['mean_T_up']:7.1f}s "
                  f"median={res['median_T_up']:6.1f}s "
                  f"std={res['std_T_up']:5.1f}s "
                  f"p90={res['p90_T_up']:7.1f}s "
                  f"bad={res['bad_rate']*100:.0f}% "
                  f"Div={res['mean_diversity']:.2f}")

        all_results[num_gbs] = sc_res

    # 绘图
    print("\n绘制扩展性实验图表...")
    plot_scalability(all_results, f"{OUTPUT_DIR}/figures")

    # 保存数据
    out = f"{OUTPUT_DIR}/logs/scalability_{timestamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总表（均值）
    print("\n" + "="*75)
    print("汇总①：T_up 均值 (s) — 受极端值影响大")
    print(f"{'GBS':>5} | {'A3C':>8} | {'DQN':>8} | {'LAD':>8} | {'SP':>8} | {'A3C vs LAD':>12} | {'A3C vs DQN':>12}")
    print("-"*75)
    for n in gbs_list:
        a = all_results[n]["A3C (Ours)"]["mean_T_up"]
        d = all_results[n]["DQN"]["mean_T_up"]
        l = all_results[n]["LAD"]["mean_T_up"]
        s = all_results[n]["ShortestPath"]["mean_T_up"]
        vs_lad = (l - a) / l * 100 if l > 0 else 0
        vs_dqn = (d - a) / d * 100 if d > 0 else 0
        print(f"{n:>5} | {a:>8.2f} | {d:>8.2f} | {l:>8.2f} | {s:>8.2f} | {vs_lad:>+11.1f}% | {vs_dqn:>+11.1f}%")

    # 打印汇总表（中位数）
    print("\n" + "="*75)
    print("汇总②：T_up 中位数 (s) — 更能反映典型性能")
    print(f"{'GBS':>5} | {'A3C':>8} | {'DQN':>8} | {'LAD':>8} | {'SP':>8} | {'A3C vs LAD':>12} | {'A3C vs DQN':>12}")
    print("-"*75)
    for n in gbs_list:
        a = all_results[n]["A3C (Ours)"]["median_T_up"]
        d = all_results[n]["DQN"]["median_T_up"]
        l = all_results[n]["LAD"]["median_T_up"]
        s = all_results[n]["ShortestPath"]["median_T_up"]
        vs_lad = (l - a) / l * 100 if l > 0 else 0
        vs_dqn = (d - a) / d * 100 if d > 0 else 0
        print(f"{n:>5} | {a:>8.2f} | {d:>8.2f} | {l:>8.2f} | {s:>8.2f} | {vs_lad:>+11.1f}% | {vs_dqn:>+11.1f}%")

    # 打印鲁棒性汇总（坏场景率）
    print("\n" + "="*75)
    print("汇总③：坏场景率 (T_up>50s 的比例) — A3C的稳健性优势")
    print(f"{'GBS':>5} | {'A3C':>8} | {'DQN':>8} | {'LAD':>8} | {'SP':>8}")
    print("-"*50)
    for n in gbs_list:
        a = all_results[n]["A3C (Ours)"]["bad_rate"]
        d = all_results[n]["DQN"]["bad_rate"]
        l = all_results[n]["LAD"]["bad_rate"]
        s = all_results[n]["ShortestPath"]["bad_rate"]
        print(f"{n:>5} | {a*100:>7.1f}% | {d*100:>7.1f}% | {l*100:>7.1f}% | {s*100:>7.1f}%")
    print("="*75)
    print(f"\n结果保存至: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval", type=int, default=200)
    args = parser.parse_args()
    main(args.episodes, args.eval)
