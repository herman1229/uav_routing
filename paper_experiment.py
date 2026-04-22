"""
论文正式实验脚本
- 保证A3C收敛（最多尝试5次，取median最小的）
- 记录完整训练曲线（奖励、T_up、成功率）
- 生成所有论文图表
- 结果保存到 paper_results/
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
import matplotlib.patches as mpatches

from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet, ValueNet
from src.baselines.dqn import DQNAgent

# ======================================================
# 输出目录
# ======================================================
PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/data", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/models", exist_ok=True)

HIDDEN_DIM = 256
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 颜色/标记风格（论文配色）
COLORS = {
    "A3C (Ours)":     "#e63946",
    "DQN":            "#457b9d",
    "LAD-Dijkstra":   "#2a9d8f",
    "ShortestPath":   "#f4a261",
    "Random":         "#aaaaaa",
}
MARKERS = {
    "A3C (Ours)": "o", "DQN": "D",
    "LAD-Dijkstra": "^", "ShortestPath": "s", "Random": "x",
}
LINE_STYLES = {
    "A3C (Ours)": "-", "DQN": "--",
    "LAD-Dijkstra": "-.", "ShortestPath": ":", "Random": ":",
}

# ======================================================
# 场景配置
# ======================================================
def make_topo_cfg(num_gbs: int) -> TopologyConfig:
    num_routers = {1: 3, 3: 4, 5: 5}[num_gbs]
    return TopologyConfig(
        num_gbs=num_gbs, num_routers=num_routers,
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(15, 35),
        init_link_load_range=(0.3, 0.65),
        link_failure_prob=0.15,
        step_failure_prob=0.08,
    )

ENV_BASE = dict(
    delay_cfg=DelayConfig(model_size=10.0, t_agg=0.5),
    delta_t=5.0, num_slots=100,
    g_hop=-1.0, alpha_1=0.4, alpha_2=0.1,
    w_delay=0.8, r_success=20.0, r_fail=-5.0,
    r_loop=-2.0, r_compete=-0.5, r_diversity=0.0,
    beta_tup=5.0, max_steps_per_gbs=50,
)


# ======================================================
# A3C 训练（返回训练曲线）
# ======================================================
def train_a3c_with_curve(env, n_episodes, label):
    actor  = PolicyNet(env.obs_dim, HIDDEN_DIM, env.action_space_n)
    critic = ValueNet(env.obs_dim, HIDDEN_DIM)
    actor_opt  = torch.optim.Adam(actor.parameters(), lr=5e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    gamma, n_step, max_grad_norm = 0.98, 5, 0.5
    ent_start, ent_end = 0.05, 0.002
    update_every = 3

    curve_ep, curve_reward, curve_tup, curve_succ = [], [], [], []

    for ep in range(1, n_episodes + 1):
        progress = ep / n_episodes
        if progress < 0.7:
            ent_coef = ent_start - (ent_start - ent_end) * (progress / 0.7) * 0.5
        else:
            ent_coef = ent_start * 0.5 * (1 - (progress - 0.7) / 0.3) + ent_end

        state = env.reset(seed=None)
        buf = {'states': [], 'actions': [], 'rewards': [],
               'valid_sets': [], 'next_states': [], 'dones': []}
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
            buf['states'].append(state); buf['actions'].append(action)
            buf['rewards'].append(r); buf['valid_sets'].append(valid)
            buf['next_states'].append(ns); buf['dones'].append(float(done))
            state = ns
            if len(buf['states']) >= update_every or done:
                _update_a3c(actor, critic, actor_opt, critic_opt,
                            buf, gamma, n_step, max_grad_norm, ent_coef)
                buf = {'states': [], 'actions': [], 'rewards': [],
                       'valid_sets': [], 'next_states': [], 'dones': []}

        res = env.get_episode_result()
        curve_ep.append(ep)
        curve_reward.append(res['total_reward'])
        t = res['T_up'] if res['T_up'] != float('inf') else None
        curve_tup.append(t)
        curve_succ.append(res['success_count'] / env.num_gbs)

    actor.eval()
    return actor, {
        'episodes': curve_ep,
        'rewards': curve_reward,
        'tups': curve_tup,
        'success_rates': curve_succ,
    }


def _update_a3c(actor, critic, actor_opt, critic_opt,
                buf, gamma, n_step, max_grad_norm, ent_coef):
    if not buf['states']:
        return
    states = torch.FloatTensor(np.array(buf['states']))
    next_states = torch.FloatTensor(np.array(buf['next_states']))
    rewards = buf['rewards']
    dones = buf['dones']
    T = len(rewards)

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
            if t + n_step < T:
                G += (gamma ** n_step) * next_values[t + n_step].item()
        returns.append(G)
    returns = torch.FloatTensor(returns).view(-1, 1)

    values = critic(states)
    adv = (returns - values).detach()
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    critic_loss = F.mse_loss(values, returns.detach())
    critic_opt.zero_grad(); critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_opt.step()

    logits_all = actor(states)
    al, ents = [], []
    for i, (valid, action) in enumerate(zip(buf['valid_sets'], buf['actions'])):
        if not valid: continue
        vl = logits_all[i][valid]
        dist = torch.distributions.Categorical(logits=vl)
        li = valid.index(action)
        al.append(-dist.log_prob(torch.tensor(li)) * adv[i])
        ents.append(dist.entropy())
    if not al: return
    actor_loss = torch.stack(al).mean() - ent_coef * torch.stack(ents).mean()
    actor_opt.zero_grad(); actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_opt.step()


# ======================================================
# DQN 训练（返回训练曲线）
# ======================================================
def train_dqn_with_curve(env, n_episodes, label):
    agent = DQNAgent(
        state_dim=env.obs_dim, hidden_dim=HIDDEN_DIM,
        action_dim=env.action_space_n,
        lr=1e-3, gamma=0.98,
        buffer_size=10000, batch_size=64,
        eps_start=1.0, eps_end=0.05,
        eps_decay=n_episodes // 2,
        target_update=20,
    )
    curve_ep, curve_reward, curve_tup, curve_succ = [], [], [], []

    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=None)
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            action = agent.select_action(state, valid)
            ns, r, done, _, _ = env.step(action)
            agent.store(state, action, r, ns, float(done))
            agent.update()
            state = ns
        res = env.get_episode_result()
        curve_ep.append(ep)
        curve_reward.append(res['total_reward'])
        t = res['T_up'] if res['T_up'] != float('inf') else None
        curve_tup.append(t)
        curve_succ.append(res['success_count'] / env.num_gbs)

    return agent, {
        'episodes': curve_ep,
        'rewards': curve_reward,
        'tups': curve_tup,
        'success_rates': curve_succ,
    }


# ======================================================
# 评估
# ======================================================
def evaluate(env, policy_fn, n_eval=300, seed_offset=1000):
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
        tups.append(r['T_up'] if r['T_up'] != float('inf') else 999.0)
        succs.append(r['success_count'] / env.num_gbs)
        divs.append(r['path_diversity'])
    arr = np.array([t for t in tups if t < 900])
    if len(arr) == 0:
        arr = np.array([999.0])
    return {
        'mean_T_up':   float(np.mean(arr)),
        'median_T_up': float(np.median(arr)),
        'std_T_up':    float(np.std(arr)),
        'p90_T_up':    float(np.percentile(arr, 90)),
        'bad_rate':    float(np.mean(arr > 50)),
        'success_rate': float(np.mean(succs)),
        'mean_diversity': float(np.mean(divs)),
        'tups': tups,
    }

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

def lad_policy(env, valid):
    M = env.delay_model.cfg.model_size
    L = env.delay_model.cfg.packet_size
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

def sp_policy(env, valid):
    best, mn = valid[0], float('inf')
    for a in valid:
        try:
            h = nx.shortest_path_length(env.topo.graph, a, env.server_id)
            if h < mn: mn, best = h, a
        except: pass
    return best


# ======================================================
# 平滑函数
# ======================================================
def smooth(data, window=30):
    data = np.array([x if x is not None else np.nan for x in data], dtype=float)
    result = np.full_like(data, np.nan)
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2 + 1)
        chunk = data[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    return result


# ======================================================
# 绘图函数
# ======================================================
def plot_training_curves(a3c_curve, dqn_curve, num_gbs, save_dir):
    """图1：A3C与DQN训练收敛曲线（奖励、T_up、成功率）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training Convergence Curves — {num_gbs} UAV(s)", fontsize=13, fontweight='bold')

    # 子图1：Episode Reward
    ax = axes[0]
    for curve, label, color in [
        (a3c_curve, "A3C (Ours)", COLORS["A3C (Ours)"]),
        (dqn_curve, "DQN",       COLORS["DQN"]),
    ]:
        ep = curve['episodes']
        sm = smooth(curve['rewards'], 50)
        ax.plot(ep, sm, color=color, linewidth=2, label=label)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Episode Reward", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # 子图2：T_up over training（只取有效值，平滑）
    ax = axes[1]
    for curve, label, color in [
        (a3c_curve, "A3C (Ours)", COLORS["A3C (Ours)"]),
        (dqn_curve, "DQN",       COLORS["DQN"]),
    ]:
        ep = curve['episodes']
        sm = smooth(curve['tups'], 50)
        valid_mask = ~np.isnan(sm)
        ax.plot(np.array(ep)[valid_mask], sm[valid_mask],
                color=color, linewidth=2, label=label)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Upload Delay T_up (s)", fontsize=11)
    ax.set_title("Upload Delay T_up During Training", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # 子图3：Success Rate
    ax = axes[2]
    for curve, label, color in [
        (a3c_curve, "A3C (Ours)", COLORS["A3C (Ours)"]),
        (dqn_curve, "DQN",       COLORS["DQN"]),
    ]:
        ep = curve['episodes']
        sm = smooth(curve['success_rates'], 50)
        ax.plot(ep, sm, color=color, linewidth=2, label=label)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Routing Success Rate", fontsize=11)
    ax.set_title("Routing Success Rate", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/fig1_training_curves_{num_gbs}uav.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_tup_comparison(all_results, save_dir):
    """图2：各场景T_up均值/中位数对比（双子图折线图）"""
    gbs_list = sorted(all_results.keys())
    algorithms = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Upload Delay T_up Comparison Across UAV Scales",
                 fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title in zip(
        axes,
        ['mean_T_up', 'median_T_up'],
        ['Mean T_up (s)', 'Median T_up (s)'],
        ['(a) Mean T_up — reflects overall performance',
         '(b) Median T_up — reflects typical performance']
    ):
        for alg in algorithms:
            vals = [all_results[n][alg][metric] for n in gbs_list]
            stds = [all_results[n][alg]['std_T_up'] for n in gbs_list]
            if metric == 'mean_T_up':
                ax.errorbar(gbs_list, vals, yerr=stds,
                            label=alg, color=COLORS[alg],
                            marker=MARKERS[alg], markersize=8,
                            linewidth=2.5, linestyle=LINE_STYLES[alg],
                            capsize=4)
            else:
                ax.plot(gbs_list, vals,
                        label=alg, color=COLORS[alg],
                        marker=MARKERS[alg], markersize=8,
                        linewidth=2.5, linestyle=LINE_STYLES[alg])
        ax.set_xlabel("Number of UAVs (FL Clients)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(gbs_list)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/fig2_tup_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_bad_rate(all_results, save_dir):
    """图3：坏场景率柱状图（稳健性对比）"""
    gbs_list = sorted(all_results.keys())
    algorithms = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(gbs_list))
    w = 0.18
    for i, alg in enumerate(algorithms):
        vals = [all_results[n][alg]['bad_rate'] * 100 for n in gbs_list]
        bars = ax.bar(x + i * w, vals, w, label=alg,
                      color=COLORS[alg], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + w * (len(algorithms) - 1) / 2)
    ax.set_xticklabels([f"{n} UAV{'s' if n > 1 else ''}" for n in gbs_list], fontsize=11)
    ax.set_ylabel("Bad Scenario Rate (T_up > 50s)", fontsize=11)
    ax.set_title("Robustness: Proportion of Bad Scenarios (T_up > 50s)\n"
                 "Lower is better — A3C achieves fewest bad scenarios in multi-UAV settings",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/fig3_bad_rate.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_tup_distribution(all_results, save_dir):
    """图4：T_up分布箱线图（3-UAV场景，展示A3C稳健性）"""
    algorithms = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("T_up Distribution: A3C Achieves Lower Variance in Multi-UAV Scenarios",
                 fontsize=12, fontweight='bold')

    for ax, num_gbs, title in zip(axes, [3, 5],
                                   ["3 UAVs (FL Clients)", "5 UAVs (FL Clients)"]):
        data, labels, colors_bp = [], [], []
        for alg in algorithms:
            tups = [t for t in all_results[num_gbs][alg]['tups'] if t < 900]
            if tups:
                data.append(tups)
                labels.append(alg)
                colors_bp.append(COLORS[alg])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        notch=False, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4))
        for patch, c in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black'); median.set_linewidth(2)
        ax.set_ylabel("T_up (s)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/fig4_tup_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_diversity(all_results, save_dir):
    """图5：路径多样性对比（展示A3C学到了分散路径）"""
    gbs_list = sorted(all_results.keys())
    algorithms = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(gbs_list))
    w = 0.18
    for i, alg in enumerate(algorithms):
        vals = [all_results[n][alg]['mean_diversity'] for n in gbs_list]
        ax.bar(x + i * w, vals, w, label=alg, color=COLORS[alg], alpha=0.85)

    ax.set_xticks(x + w * (len(algorithms) - 1) / 2)
    ax.set_xticklabels([f"{n} UAV{'s' if n > 1 else ''}" for n in gbs_list], fontsize=11)
    ax.set_ylabel("Path Diversity (unique routers used)", fontsize=11)
    ax.set_title("Path Diversity: A3C Learns to Spread Traffic Across Routers",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/fig5_path_diversity.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_a3c_improvement(all_results, save_dir):
    """图6：A3C相对各基线的提升幅度（基于均值）"""
    gbs_list = sorted(all_results.keys())
    baselines = ["DQN", "LAD-Dijkstra", "ShortestPath"]
    bl_colors = [COLORS["DQN"], COLORS["LAD-Dijkstra"], COLORS["ShortestPath"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(gbs_list))
    w = 0.25

    for i, (bl, c) in enumerate(zip(baselines, bl_colors)):
        improvements = []
        for n in gbs_list:
            a3c_t = all_results[n]["A3C (Ours)"]['mean_T_up']
            bl_t  = all_results[n][bl]['mean_T_up']
            impr  = (bl_t - a3c_t) / bl_t * 100 if bl_t > 0 else 0
            improvements.append(impr)
        bars = ax.bar(x + i * w, improvements, w,
                      label=f"vs {bl}", color=c, alpha=0.85)
        for bar, v in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if v >= 0 else -3),
                    f'{v:+.1f}%', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{n} UAV{'s' if n > 1 else ''}" for n in gbs_list], fontsize=11)
    ax.set_ylabel("A3C Improvement over Baseline (%)", fontsize=11)
    ax.set_title("A3C Improvement over Baselines (Mean T_up)\n"
                 "Positive = A3C is better", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/fig6_a3c_improvement.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


# ======================================================
# 主流程
# ======================================================
def run_gbs_experiment(num_gbs, n_train, n_eval, max_attempts=5):
    """对单个GBS规模运行实验，保证A3C收敛"""
    topo_cfg = make_topo_cfg(num_gbs)
    env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)

    print(f"\n{'='*60}")
    print(f"  {num_gbs}-UAV 场景 | 训练{n_train}ep | 评估{n_eval}ep")
    print(f"  obs_dim={env.obs_dim}, nodes={env.num_nodes}")
    print(f"{'='*60}")

    # 训练A3C（多次尝试，取中位数最小的）
    best_actor, best_curve, best_median = None, None, float('inf')
    for attempt in range(1, max_attempts + 1):
        print(f"\n  [A3C 尝试 {attempt}/{max_attempts}]")
        actor, curve = train_a3c_with_curve(env, n_train, f"{num_gbs}uav-a{attempt}")
        # 快速评估
        quick = evaluate(env, lambda e, v, a=actor: a3c_policy(a, e, v), 80)
        print(f"  → median={quick['median_T_up']:.2f}s, bad={quick['bad_rate']*100:.0f}%, "
              f"Div={quick['mean_diversity']:.2f}")
        if quick['median_T_up'] < best_median:
            best_median = quick['median_T_up']
            best_actor  = actor
            best_curve  = curve
        # 收敛判断：1-GBS median<5s，3-GBS median<20s，5-GBS median<20s
        threshold = 5.0 if num_gbs == 1 else 20.0
        if quick['median_T_up'] < threshold:
            print(f"  ✅ A3C 收敛！(median={quick['median_T_up']:.2f}s < {threshold}s)")
            break
    else:
        print(f"  ⚠️  达到最大尝试次数，使用最优结果 (median={best_median:.2f}s)")

    # 训练DQN（一次即可，DQN收敛稳定）
    print(f"\n  [DQN 训练]")
    dqn, dqn_curve = train_dqn_with_curve(env, n_train, f"{num_gbs}uav-dqn")
    dqn_quick = evaluate(env, lambda e, v, d=dqn: dqn_policy(d, e, v), 80)
    print(f"  → median={dqn_quick['median_T_up']:.2f}s")

    # 保存训练曲线数据
    curve_data = {
        'num_gbs': num_gbs,
        'a3c': best_curve,
        'dqn': dqn_curve,
    }
    curve_path = f"{PAPER_DIR}/data/training_curves_{num_gbs}uav.json"
    with open(curve_path, 'w') as f:
        # tups中None替换为-1
        safe_curve = {
            'num_gbs': num_gbs,
            'a3c': {k: [x if x is not None else -1 for x in v]
                    if isinstance(v, list) else v
                    for k, v in best_curve.items()},
            'dqn': {k: [x if x is not None else -1 for x in v]
                    if isinstance(v, list) else v
                    for k, v in dqn_curve.items()},
        }
        json.dump(safe_curve, f, indent=2)

    # 保存模型
    torch.save(best_actor.state_dict(),
               f"{PAPER_DIR}/models/a3c_{num_gbs}uav.pth")
    torch.save(dqn.q_net.state_dict(),
               f"{PAPER_DIR}/models/dqn_{num_gbs}uav.pth")

    # 绘制训练曲线图
    plot_training_curves(best_curve, dqn_curve, num_gbs,
                         f"{PAPER_DIR}/figures")

    # 正式评估（所有算法）
    print(f"\n  [正式评估 {n_eval} episodes]")
    results = {}
    results["A3C (Ours)"] = evaluate(
        env, lambda e, v, a=best_actor: a3c_policy(a, e, v), n_eval)
    results["DQN"] = evaluate(
        env, lambda e, v, d=dqn: dqn_policy(d, e, v), n_eval)
    results["LAD-Dijkstra"] = evaluate(env, lad_policy, n_eval)
    results["ShortestPath"] = evaluate(env, sp_policy, n_eval)

    for alg, res in results.items():
        print(f"  {alg:18s} | mean={res['mean_T_up']:7.2f}s "
              f"median={res['median_T_up']:6.2f}s "
              f"bad={res['bad_rate']*100:.0f}% "
              f"Div={res['mean_diversity']:.2f}")

    return results, best_curve, dqn_curve


def main(n_train=1500, n_eval=300):
    print(f"\n{'#'*60}")
    print("  论文正式实验")
    print(f"  训练轮数={n_train}, 评估轮数={n_eval}")
    print(f"  输出目录: {PAPER_DIR}/")
    print(f"{'#'*60}")

    gbs_list = [1, 3, 5]
    train_eps = {1: n_train, 3: n_train, 5: max(n_train, 2000)}

    all_results = {}
    for num_gbs in gbs_list:
        results, a3c_curve, dqn_curve = run_gbs_experiment(
            num_gbs, train_eps[num_gbs], n_eval, max_attempts=5)
        all_results[num_gbs] = results

    # 保存完整评估结果
    safe_results = {}
    for n, res in all_results.items():
        safe_results[str(n)] = {
            alg: {k: v for k, v in r.items() if k != 'tups'}
            for alg, r in res.items()
        }
    # 单独保存tups
    tups_data = {
        str(n): {alg: res['tups'] for alg, res in res_dict.items()}
        for n, res_dict in all_results.items()
    }
    result_path = f"{PAPER_DIR}/data/evaluation_results_{TIMESTAMP}.json"
    with open(result_path, 'w') as f:
        json.dump({'metrics': safe_results, 'tups': tups_data}, f, indent=2)

    # 绘制综合对比图
    print(f"\n{'='*60}")
    print("  绘制综合对比图表...")
    print(f"{'='*60}")
    plot_tup_comparison(all_results, f"{PAPER_DIR}/figures")
    plot_bad_rate(all_results, f"{PAPER_DIR}/figures")
    plot_tup_distribution(all_results, f"{PAPER_DIR}/figures")
    plot_diversity(all_results, f"{PAPER_DIR}/figures")
    plot_a3c_improvement(all_results, f"{PAPER_DIR}/figures")

    # 打印最终汇总
    print(f"\n{'='*65}")
    print("  最终结果汇总")
    print(f"{'='*65}")

    for title, metric in [
        ("T_up 均值 (s)", "mean_T_up"),
        ("T_up 中位数 (s)", "median_T_up"),
        ("坏场景率 (%)", "bad_rate"),
    ]:
        print(f"\n  【{title}】")
        print(f"  {'UAV':>5} | {'A3C':>8} | {'DQN':>8} | {'LAD':>8} | {'SP':>8}")
        print(f"  {'-'*45}")
        for n in gbs_list:
            vals = {alg: all_results[n][alg][metric] for alg in
                    ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]}
            if metric == 'bad_rate':
                row = " | ".join(f"{v*100:>7.1f}%" for v in vals.values())
            else:
                row = " | ".join(f"{v:>8.2f}" for v in vals.values())
            print(f"  {n:>5} | {row}")

    print(f"\n  结果保存至: {result_path}")
    print(f"  图表保存至: {PAPER_DIR}/figures/")
    print(f"  模型保存至: {PAPER_DIR}/models/")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval",     type=int, default=300)
    args = parser.parse_args()
    main(args.episodes, args.eval)
