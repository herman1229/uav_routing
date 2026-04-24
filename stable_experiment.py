"""
稳定收敛实验：PPO vs 改进A3C vs DQN
核心目标：验证PPO能否在3-UAV和5-UAV场景下稳定收敛

实验设计：
  每个算法独立训练 N_RUNS 次（不同随机种子）
  统计：收敛率、收敛时T_up中位数、坏场景率
  "收敛"定义：评估时 median T_up < 阈值（3-UAV<20s，5-UAV<20s）

用法: python stable_experiment.py [--episodes 1200] [--eval 300] [--runs 5]
"""
import os, sys, json, argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.ppo import PPOAgent, PPOActor, PPOCritic
from src.agents.a3c import PolicyNet, ValueNet
from src.baselines.dqn import DQNAgent
from src.baselines.dqn_per import DQNAgentPER

PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/data", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/models", exist_ok=True)
HIDDEN_DIM = 256
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

CONV_THRESHOLD = {1: 5.0, 3: 20.0, 5: 20.0}   # median T_up 收敛阈值

# ======================================================
# 场景配置
# ======================================================
def make_topo_cfg(num_gbs):
    num_routers = {1: 3, 3: 4, 5: 5}[num_gbs]
    return TopologyConfig(
        num_gbs=num_gbs, num_routers=num_routers, node_capacity=50,
        gbs_to_router_capacity=20.0, router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(15, 35), init_link_load_range=(0.3, 0.65),
        link_failure_prob=0.15, step_failure_prob=0.08,
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
# PPO 训练（单次运行，返回曲线）
# ======================================================
def train_ppo_once(env: ConcurrentFLRoutingEnv, n_episodes: int,
                   seed: int = None) -> tuple:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    agent = PPOAgent(
        state_dim    = env.obs_dim,
        hidden_dim   = HIDDEN_DIM,
        action_dim   = env.action_space_n,
        actor_lr     = 5e-4,   # 与A3C对齐
        critic_lr    = 1e-3,
        gamma        = 0.98,
        gae_lambda   = 0.95,
        clip_ratio   = 0.2,
        ppo_epochs   = 8,      # 更多epoch充分利用每个episode的数据
        entropy_coef = 0.02,   # 更大熵系数鼓励早期探索
        max_grad_norm= 0.5,
    )

    ent_start, ent_end = 0.05, 0.002
    curve_r, curve_t, curve_clip = [], [], []

    for ep in range(1, n_episodes + 1):
        # 熵系数退火（与A3C完全相同的退火策略）
        p = ep / n_episodes
        ent_coef = (ent_start - (ent_start - ent_end) * (p / 0.7) * 0.5
                    if p < 0.7 else
                    ent_start * 0.5 * (1 - (p - 0.7) / 0.3) + ent_end)
        # 同步更新agent的entropy_coef（用于update调用）
        agent.entropy_coef = ent_coef

        state = env.reset(seed=None)
        rollout = {'states': [], 'actions_idx': [], 'old_log_probs': [],
                   'rewards': [], 'dones': [], 'valid_masks': [],
                   'next_states': []}
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid: break

            action, log_prob = agent.select_action(state, valid)
            local_idx = valid.index(action)

            # 构造 valid_mask
            mask = np.zeros(env.action_space_n, dtype=np.float32)
            for a in valid:
                mask[a] = 1.0

            ns, r, done, _, _ = env.step(action)

            rollout['states'].append(state)
            rollout['actions_idx'].append(local_idx)
            rollout['old_log_probs'].append(log_prob)
            rollout['rewards'].append(r)
            rollout['dones'].append(float(done))
            rollout['valid_masks'].append(mask)
            rollout['next_states'].append(ns)
            state = ns

        if rollout['states']:
            stats = agent.update(rollout, ent_coef=ent_coef)
            curve_clip.append(stats['clip_frac'])

        res = env.get_episode_result()
        curve_r.append(res['total_reward'])
        curve_t.append(res['T_up'] if res['T_up'] != float('inf') else None)

    agent.actor.eval()
    return agent, {'rewards': curve_r, 'tups': curve_t, 'clip_fracs': curve_clip}


# ======================================================
# 改进A3C 训练（单次）
# ======================================================
def _update_a3c(actor, critic, ao, co, buf, gamma=0.98, n_step=5, mg=0.5, ec=0.01):
    if not buf['states']: return
    states = torch.FloatTensor(np.array(buf['states']))
    ns_t   = torch.FloatTensor(np.array(buf['next_states']))
    T = len(buf['rewards'])
    with torch.no_grad():
        nv = critic(ns_t).squeeze(1)
    returns = []
    for t in range(T):
        G, steps = 0.0, min(n_step, T - t)
        for k in range(steps):
            G += (gamma**k) * buf['rewards'][t+k]
            if buf['dones'][t+k]: break
        else:
            if t + steps < T:
                G += (gamma**steps) * nv[t+steps].item()
        returns.append(G)
    returns = torch.FloatTensor(returns).view(-1, 1)
    values  = critic(states)
    adv = (returns - values).detach()
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    cl = F.mse_loss(values, returns.detach())
    co.zero_grad(); cl.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), mg); co.step()
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
    aloss = torch.stack(al).mean() - ec * torch.stack(ents).mean()
    ao.zero_grad(); aloss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), mg); ao.step()


def train_a3c_once(env: ConcurrentFLRoutingEnv, n_episodes: int,
                   seed: int = None) -> tuple:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    actor  = PolicyNet(env.obs_dim, HIDDEN_DIM, env.action_space_n)
    critic = ValueNet(env.obs_dim, HIDDEN_DIM)
    ao = torch.optim.Adam(actor.parameters(),  lr=5e-4)
    co = torch.optim.Adam(critic.parameters(), lr=1e-3)
    ent_s, ent_e = 0.05, 0.002
    update_every = 3
    curve_r, curve_t = [], []

    for ep in range(1, n_episodes + 1):
        p = ep / n_episodes
        ec = (ent_s - (ent_s-ent_e)*(p/0.7)*0.5 if p < 0.7
              else ent_s*0.5*(1-(p-0.7)/0.3)+ent_e)
        state = env.reset(seed=None)
        buf = {'states':[],'actions':[],'rewards':[],'valid_sets':[],'next_states':[],'dones':[]}
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            with torch.no_grad():
                logits = actor(torch.FloatTensor(state))
                action = valid[torch.distributions.Categorical(logits=logits[valid]).sample().item()]
            ns, r, done, _, _ = env.step(action)
            buf['states'].append(state); buf['actions'].append(action)
            buf['rewards'].append(r); buf['valid_sets'].append(valid)
            buf['next_states'].append(ns); buf['dones'].append(float(done))
            state = ns
            if len(buf['states']) >= update_every or done:
                _update_a3c(actor, critic, ao, co, buf, ec=ec)
                buf = {'states':[],'actions':[],'rewards':[],'valid_sets':[],'next_states':[],'dones':[]}
        res = env.get_episode_result()
        curve_r.append(res['total_reward'])
        curve_t.append(res['T_up'] if res['T_up'] != float('inf') else None)

    actor.eval()
    return actor, {'rewards': curve_r, 'tups': curve_t}


# ======================================================
# DQN 训练（单次）
# ======================================================
def train_dqn_per_once(env: ConcurrentFLRoutingEnv, n_episodes: int,
                       seed: int = None) -> tuple:
    """DQN-PER：优先经验回放，解决5-UAV场景的不稳定问题"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    agent = DQNAgentPER(
        state_dim=env.obs_dim, hidden_dim=HIDDEN_DIM,
        action_dim=env.action_space_n,
        lr=5e-4, gamma=0.98, buffer_size=20000, batch_size=128,
        eps_start=1.0, eps_end=0.05, eps_decay=n_episodes // 2,
        target_update=10, alpha=0.6, beta_start=0.4, beta_end=1.0,
    )
    total_steps_est = n_episodes * 20
    curve_r, curve_t = [], []
    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=None)
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            action = agent.select_action(state, valid)
            ns, r, done, _, _ = env.step(action)
            agent.store(state, action, r, ns, float(done))
            agent.update(total_steps_est)
            state = ns
        res = env.get_episode_result()
        curve_r.append(res['total_reward'])
        curve_t.append(res['T_up'] if res['T_up'] != float('inf') else None)
    return agent, {'rewards': curve_r, 'tups': curve_t}


def train_dqn_once(env: ConcurrentFLRoutingEnv, n_episodes: int,
                   seed: int = None) -> tuple:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    agent = DQNAgent(
        state_dim=env.obs_dim, hidden_dim=HIDDEN_DIM,
        action_dim=env.action_space_n,
        lr=1e-3, gamma=0.98, buffer_size=10000, batch_size=64,
        eps_start=1.0, eps_end=0.05, eps_decay=n_episodes//2, target_update=20,
    )
    curve_r, curve_t = [], []
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
        curve_r.append(res['total_reward'])
        curve_t.append(res['T_up'] if res['T_up'] != float('inf') else None)
    return agent, {'rewards': curve_r, 'tups': curve_t}


# ======================================================
# 评估
# ======================================================
def evaluate(env, policy_fn, n_eval=200, seed_offset=2000):
    tups, divs = [], []
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
        divs.append(r['path_diversity'])
    arr = np.array([t for t in tups if t < 900])
    if not len(arr): arr = np.array([999.0])
    return {
        'mean_T_up':   float(np.mean(arr)),
        'median_T_up': float(np.median(arr)),
        'std_T_up':    float(np.std(arr)),
        'bad_rate':    float(np.mean(arr > 50)),
        'mean_diversity': float(np.mean(divs)),
        'tups': tups,
    }


def lad_policy(env, valid):
    M, L = env.delay_model.cfg.model_size, env.delay_model.cfg.packet_size
    wg = nx.DiGraph()
    for u, v in env.topo.edges:
        bw = max(env.topo.available_bandwidth(u, v), 0.1)
        wg.add_edge(u, v, weight=(M+L)/bw)
    best, mn = valid[0], float('inf')
    for a in valid:
        try:
            d = nx.shortest_path_length(wg, a, env.server_id, weight='weight')
            if d < mn: mn, best = d, a
        except: pass
    return best


# ======================================================
# 平滑 & 绘图
# ======================================================
def smooth(data, w=40):
    arr = np.array([x if x is not None else np.nan for x in data], dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        chunk = arr[max(0,i-w//2):min(len(arr),i+w//2+1)]
        v = chunk[~np.isnan(chunk)]
        if len(v): out[i] = v.mean()
    return out

COLORS = {
    "PPO (Ours)":   "#e63946",
    "A3C":          "#457b9d",
    "DQN":          "#f4a261",
    "LAD-Dijkstra": "#2a9d8f",
}
LS = {"PPO (Ours)": "-", "A3C": "--", "DQN": ":", "LAD-Dijkstra": "-."}


def plot_convergence_rate(conv_stats, num_gbs, save_dir):
    """收敛率对比柱状图（核心稳定性指标）"""
    algs  = list(conv_stats.keys())
    rates = [conv_stats[a]['conv_rate'] * 100 for a in algs]
    cols  = [COLORS.get(a, "#888") for a in algs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(algs, rates, color=cols, alpha=0.85, width=0.5)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, f'{v:.0f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel("Convergence Rate (%)", fontsize=12)
    ax.set_title(f"Convergence Rate Comparison ({num_gbs} UAVs, {conv_stats[algs[0]]['n_runs']} runs)\n"
                 f"Converged = median T_up < {CONV_THRESHOLD[num_gbs]}s",
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/stable_conv_rate_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_multi_run_curves(all_curves, alg_name, num_gbs, save_dir):
    """多次运行的收敛曲线（展示稳定性）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{alg_name} — {num_gbs} UAVs, Multi-Run Convergence",
                 fontsize=12, fontweight='bold')
    color = COLORS.get(alg_name, "#888")

    for ax, key, ylabel, title in zip(
        axes, ['rewards', 'tups'],
        ['Episode Reward', 'Upload Delay T_up (s)'],
        ['(a) Episode Reward', '(b) T_up During Training']
    ):
        for i, curve in enumerate(all_curves):
            sm = smooth(curve[key], 50)
            valid = ~np.isnan(sm)
            alpha = 0.9 if i == 0 else 0.4
            lw    = 2.0 if i == 0 else 1.0
            ax.plot(np.arange(len(sm))[valid], sm[valid],
                    color=color, alpha=alpha, linewidth=lw,
                    label=f"run {i+1}" if i < 3 else None)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/stable_runs_{alg_name.replace(' ', '_').replace('(','').replace(')','').replace('/','_')}_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_eval_comparison(best_results, num_gbs, save_dir):
    """最优模型的评估结果对比（4个指标）"""
    algs = list(best_results.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"Best Model Evaluation — {num_gbs} UAVs",
                 fontsize=12, fontweight='bold')

    for ax, metric, ylabel, title, scale in zip(
        axes,
        ['mean_T_up', 'median_T_up', 'bad_rate', 'mean_diversity'],
        ['Mean T_up (s)', 'Median T_up (s)', 'Bad Rate (%)', 'Path Diversity'],
        ['(a) Mean T_up', '(b) Median T_up', '(c) Bad Scenario Rate', '(d) Path Diversity'],
        [1, 1, 100, 1]
    ):
        vals = [best_results[a][metric] * scale for a in algs]
        cols = [COLORS.get(a, "#888") for a in algs]
        bars = ax.bar(range(len(algs)), vals, color=cols, alpha=0.85, width=0.6)
        for bar, v in zip(bars, vals):
            fmt = f'{v:.1f}{"%" if metric=="bad_rate" else "s" if "T_up" in metric else ""}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    fmt, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels([a.replace(" ", "\n") for a in algs], fontsize=8)
        ax.set_ylabel(ylabel + ('%' if metric=='bad_rate' else ''), fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/stable_eval_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_stability_summary(all_stats, uav_list, save_dir):
    """多规模稳定性汇总图"""
    algs = list(all_stats[uav_list[0]].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stability Comparison: PPO vs A3C vs DQN",
                 fontsize=13, fontweight='bold')

    # 子图1：收敛率 vs UAV数量
    ax = axes[0]
    for alg in algs:
        rates = [all_stats[n][alg]['conv_rate'] * 100 for n in uav_list]
        ax.plot(uav_list, rates, label=alg,
                color=COLORS.get(alg, "#888"), marker='o',
                linewidth=2.5, markersize=8)
    ax.set_xlabel("Number of UAVs", fontsize=11)
    ax.set_ylabel("Convergence Rate (%)", fontsize=11)
    ax.set_title("(a) Convergence Rate", fontsize=11)
    ax.set_xticks(uav_list); ax.set_ylim(0, 105)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 子图2：收敛时median T_up vs UAV数量（仅收敛的run）
    ax = axes[1]
    for alg in algs:
        meds = []
        for n in uav_list:
            conv_meds = all_stats[n][alg]['conv_medians']
            meds.append(np.mean(conv_meds) if conv_meds else 999.0)
        ax.plot(uav_list, meds, label=alg,
                color=COLORS.get(alg, "#888"), marker='o',
                linewidth=2.5, markersize=8)
    ax.set_xlabel("Number of UAVs", fontsize=11)
    ax.set_ylabel("Median T_up (s) when converged", fontsize=11)
    ax.set_title("(b) Median T_up (converged runs only)", fontsize=11)
    ax.set_xticks(uav_list)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/stable_summary.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


# ======================================================
# 主流程
# ======================================================
def run_stability_test(num_gbs, n_episodes, n_eval, n_runs):
    topo_cfg = make_topo_cfg(num_gbs)
    env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)
    threshold = CONV_THRESHOLD[num_gbs]

    print(f"\n{'='*60}")
    print(f"  {num_gbs}-UAV | {n_runs}次运行 | {n_episodes}ep/run | 阈值<{threshold}s")
    print(f"  obs_dim={env.obs_dim}")
    print(f"{'='*60}")

    # 训练函数映射
    train_fns = {
        "A3C":        train_a3c_once,
        "DQN":        train_dqn_once,
        "DQN-PER":    train_dqn_per_once,
    }
    COLORS["DQN-PER"] = "#9b5de5"
    LS["DQN-PER"] = (0, (3, 1, 1, 1))

    stats = {}       # 每个算法的收敛统计
    best_models = {} # 每个算法最优模型
    best_curves  = {} # 每个算法所有运行曲线

    for alg, train_fn in train_fns.items():
        print(f"\n  [{alg}] {n_runs}次独立训练...")
        conv_count = 0
        conv_medians = []
        all_curves = []
        best_med, best_model = float('inf'), None

        for run in range(1, n_runs + 1):
            seed = 42 + run * 100 + num_gbs * 10
            model, curve = train_fn(env, n_episodes, seed=seed)

            # 快速评估（60ep）
            if alg == "A3C":
                policy_fn = lambda e, v, m=model: (
                    v[F.softmax(m(torch.FloatTensor(e._obs()))[v], dim=0).argmax().item()])
            else:  # DQN or DQN-PER
                policy_fn = lambda e, v, m=model: (
                    v[m.q_net(torch.FloatTensor(e._obs()))[v].argmax().item()])

            quick = evaluate(env, policy_fn, 60)
            med = quick['median_T_up']
            converged = med < threshold
            if converged:
                conv_count += 1
                conv_medians.append(med)
            all_curves.append(curve)
            if med < best_med:
                best_med = med; best_model = model
            print(f"    run {run}: median={med:.2f}s {'✅' if converged else '❌'}")

        conv_rate = conv_count / n_runs
        stats[alg] = {
            'conv_rate':    conv_rate,
            'conv_count':   conv_count,
            'n_runs':       n_runs,
            'conv_medians': conv_medians,
            'mean_conv_med': float(np.mean(conv_medians)) if conv_medians else 999.0,
        }
        best_models[alg] = best_model
        best_curves[alg] = all_curves
        print(f"  → 收敛率: {conv_count}/{n_runs} = {conv_rate*100:.0f}%  "
              f"收敛时median均值: {stats[alg]['mean_conv_med']:.2f}s")

        # 绘制多次运行曲线
        plot_multi_run_curves(all_curves, alg, num_gbs, f"{PAPER_DIR}/figures")

    # 绘制收敛率对比
    plot_convergence_rate(stats, num_gbs, f"{PAPER_DIR}/figures")

    # 用最优模型做正式评估
    print(f"\n  [正式评估 {n_eval}ep（最优模型）]")
    best_results = {}
    for alg, model in best_models.items():
        if alg == "A3C":
            pf = lambda e, v, m=model: (
                v[F.softmax(m(torch.FloatTensor(e._obs()))[v], dim=0).argmax().item()])
        else:
            pf = lambda e, v, m=model: (
                v[m.q_net(torch.FloatTensor(e._obs()))[v].argmax().item()])
        best_results[alg] = evaluate(env, pf, n_eval)

    best_results["LAD-Dijkstra"] = evaluate(env, lad_policy, n_eval)

    for alg, res in best_results.items():
        print(f"  {alg:18s} mean={res['mean_T_up']:7.2f}s "
              f"median={res['median_T_up']:6.2f}s "
              f"bad={res['bad_rate']*100:.0f}%")

    plot_eval_comparison(best_results, num_gbs, f"{PAPER_DIR}/figures")

    # 保存最优PPO模型
    if "PPO (Ours)" in best_models and best_models["PPO (Ours)"] is not None:
        torch.save(best_models["PPO (Ours)"].actor.state_dict(),
                   f"{PAPER_DIR}/models/ppo_actor_{num_gbs}uav.pth")

    return stats, best_results


def main(n_episodes=1200, n_eval=300, n_runs=5):
    uav_list = [3, 5]

    print(f"\n{'#'*60}")
    print(f"  稳定收敛实验：PPO vs A3C vs DQN")
    print(f"  {n_runs}次独立运行 | {n_episodes}ep/run | 评估{n_eval}ep")
    print(f"{'#'*60}")

    all_stats   = {}
    all_results = {}

    train_eps = {3: n_episodes, 5: max(n_episodes, 2000)}

    for num_gbs in uav_list:
        stats, best_results = run_stability_test(
            num_gbs, train_eps[num_gbs], n_eval, n_runs)
        all_stats[num_gbs]   = stats
        all_results[num_gbs] = best_results

    # 多规模稳定性汇总图
    plot_stability_summary(all_stats, uav_list, f"{PAPER_DIR}/figures")

    # 保存数据
    safe_stats = {
        str(n): {alg: {k: v for k, v in s.items()}
                 for alg, s in sc.items()}
        for n, sc in all_stats.items()
    }
    safe_results = {
        str(n): {alg: {k: v for k, v in r.items() if k != 'tups'}
                 for alg, r in res.items()}
        for n, res in all_results.items()
    }
    path = f"{PAPER_DIR}/data/stable_results_{TIMESTAMP}.json"
    with open(path, 'w') as f:
        json.dump({'stats': safe_stats, 'best_eval': safe_results}, f, indent=2)

    # 汇总打印
    rl_algs = [a for a in list(all_stats[uav_list[0]].keys())]
    print(f"\n{'='*65}")
    print("  稳定性汇总：收敛率")
    header = " | ".join(f"{a:>12}" for a in rl_algs)
    print(f"  {'UAV':>5} | {header}")
    print(f"  {'-'*60}")
    for n in uav_list:
        row = " | ".join(f"{all_stats[n][a]['conv_rate']*100:>11.0f}%" for a in rl_algs)
        print(f"  {n:>5} | {row}")

    print(f"\n  稳定性汇总：最优模型 median T_up (s)")
    eval_algs = rl_algs + ["LAD-Dijkstra"]
    header2 = " | ".join(f"{a:>12}" for a in eval_algs)
    print(f"  {'UAV':>5} | {header2}")
    print(f"  {'-'*70}")
    for n in uav_list:
        row2 = " | ".join(f"{all_results[n][a]['median_T_up']:>11.2f}s" for a in eval_algs)
        print(f"  {n:>5} | {row2}")

    print(f"\n  数据: {path}")
    return all_stats, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--eval",     type=int, default=300)
    parser.add_argument("--runs",     type=int, default=5)
    args = parser.parse_args()
    main(args.episodes, args.eval, args.runs)
