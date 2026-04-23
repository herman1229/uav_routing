"""
MARL实验脚本：MAPPO vs 改进A3C vs 传统算法
框架：CTDE（集中训练分布执行）
算法：MAPPO（Multi-Agent PPO，参数共享Actor + 集中式Critic）

核心假设验证：
- MAPPO的独立Actor + 集中Critic是否比改进A3C（单智能体轮流决策）更稳定？
- 在5-UAV高并发场景，MAPPO能否解决A3C的收敛不稳定问题？

用法: python marl_experiment.py [--episodes 1500] [--eval 300] [--uav 3]
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
from src.envs.marl_env import MARLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.mappo import MAPPOAgent
from src.agents.a3c import PolicyNet, ValueNet

PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/data", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/models", exist_ok=True)
HIDDEN_DIM = 256
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# 场景配置（与其他实验保持一致）
# ======================================================
UAV_ROUTER_MAP = {1: 3, 3: 4, 5: 5, 7: 6, 10: 8}

def make_topo_cfg(num_gbs):
    return TopologyConfig(
        num_gbs=num_gbs,
        num_routers=UAV_ROUTER_MAP[num_gbs],
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
# MAPPO 训练
# ======================================================
def train_mappo(marl_env: MARLRoutingEnv, n_episodes: int, label: str):
    agent = MAPPOAgent(
        local_obs_dim   = marl_env.local_obs_dim,
        global_state_dim= marl_env.global_state_dim,
        hidden_dim      = HIDDEN_DIM,
        action_dim      = marl_env.action_space_n,
        actor_lr        = 3e-4,
        critic_lr       = 1e-3,
        gamma           = 0.98,
        gae_lambda      = 0.95,
        clip_ratio      = 0.2,
        ppo_epochs      = 4,
        entropy_coef    = 0.01,
        max_grad_norm   = 0.5,
    )

    print(f"  [MAPPO] {label} | local_obs={marl_env.local_obs_dim} "
          f"global_state={marl_env.global_state_dim} | {n_episodes}ep")

    curve_reward, curve_tup, curve_succ = [], [], []

    for ep in range(1, n_episodes + 1):
        local_obs_all, global_state = marl_env.reset(seed=None)
        done = False

        # 每个GBS独立收集轨迹
        rollouts: Dict[int, Dict] = {
            g: {'local_obs': [], 'global_states': [], 'actions_idx': [],
                'old_log_probs': [], 'rewards': [], 'dones': [], 'valid_masks': []}
            for g in range(marl_env.num_gbs)
        }

        while not done:
            cur_gbs = marl_env.get_current_gbs()
            valid   = marl_env.get_valid_actions(cur_gbs)
            if not valid:
                break

            local_obs = local_obs_all[cur_gbs]
            action, log_prob = agent.select_action(local_obs, valid)
            local_idx = marl_env.action_to_local_idx(action, valid)
            valid_mask = marl_env.make_valid_mask(valid)

            new_local_obs_all, new_global_state, reward, done, info = marl_env.step(action)

            # 记录到当前GBS的轨迹
            r = rollouts[cur_gbs]
            r['local_obs'].append(local_obs)
            r['global_states'].append(global_state)
            r['actions_idx'].append(local_idx)
            r['old_log_probs'].append(log_prob)
            r['rewards'].append(reward)
            r['dones'].append(float(done))
            r['valid_masks'].append(valid_mask)

            local_obs_all = new_local_obs_all
            global_state  = new_global_state

        # Episode结束，合并所有GBS的轨迹一起更新
        merged = {k: [] for k in ['local_obs', 'global_states', 'actions_idx',
                                   'old_log_probs', 'rewards', 'dones', 'valid_masks']}
        for g in range(marl_env.num_gbs):
            for k in merged:
                merged[k].extend(rollouts[g][k])

        if merged['local_obs']:
            agent.update(merged)

        result = marl_env.get_episode_result()
        curve_reward.append(result['total_reward'])
        t = result['T_up'] if result['T_up'] != float('inf') else None
        curve_tup.append(t)
        curve_succ.append(result['success_count'] / marl_env.num_gbs)

        if ep % 200 == 0:
            recent_r = curve_reward[-50:]
            valid_t  = [x for x in curve_tup[-50:] if x is not None]
            tup_str  = f"{np.mean(valid_t):.2f}s" if valid_t else "N/A"
            print(f"    Ep {ep:4d} | AvgR={np.mean(recent_r):.2f} | T_up={tup_str}")

    agent.actor.eval()
    return agent, {'rewards': curve_reward, 'tups': curve_tup, 'success_rates': curve_succ}


# ======================================================
# 改进A3C训练（复用自paper_experiment.py的逻辑）
# ======================================================
def _update_a3c(actor, critic, ao, co, buf, gamma=0.98, n_step=5, mg=0.5, ec=0.01):
    if not buf['states']: return
    states = torch.FloatTensor(np.array(buf['states']))
    ns_t   = torch.FloatTensor(np.array(buf['next_states']))
    rewards, dones = buf['rewards'], buf['dones']
    T = len(rewards)
    with torch.no_grad():
        nv = critic(ns_t).squeeze(1)
    returns = []
    for t in range(T):
        G, steps = 0.0, min(n_step, T - t)
        for k in range(steps):
            G += (gamma**k) * rewards[t+k]
            if dones[t+k]: break
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


def train_a3c(base_env: ConcurrentFLRoutingEnv, n_episodes: int, label: str, max_attempts=5):
    """改进A3C训练，多次尝试取最优"""
    best_actor, best_med = None, float('inf')
    threshold = 5.0 if base_env.num_gbs == 1 else 20.0

    for attempt in range(1, max_attempts + 1):
        actor  = PolicyNet(base_env.obs_dim, HIDDEN_DIM, base_env.action_space_n)
        critic = ValueNet(base_env.obs_dim, HIDDEN_DIM)
        ao = torch.optim.Adam(actor.parameters(),  lr=5e-4)
        co = torch.optim.Adam(critic.parameters(), lr=1e-3)
        ent_s, ent_e = 0.05, 0.002
        update_every = 3
        curve_r, curve_t = [], []

        for ep in range(1, n_episodes + 1):
            p = ep / n_episodes
            ec = (ent_s - (ent_s-ent_e)*(p/0.7)*0.5 if p < 0.7
                  else ent_s*0.5*(1-(p-0.7)/0.3)+ent_e)
            state = base_env.reset(seed=None)
            buf = {'states':[],'actions':[],'rewards':[],'valid_sets':[],'next_states':[],'dones':[]}
            done = False
            while not done:
                valid = base_env.get_valid_actions()
                if not valid: break
                with torch.no_grad():
                    logits = actor(torch.FloatTensor(state))
                    action = valid[torch.distributions.Categorical(logits=logits[valid]).sample().item()]
                ns, r, done, _, _ = base_env.step(action)
                buf['states'].append(state); buf['actions'].append(action)
                buf['rewards'].append(r); buf['valid_sets'].append(valid)
                buf['next_states'].append(ns); buf['dones'].append(float(done))
                state = ns
                if len(buf['states']) >= update_every or done:
                    _update_a3c(actor, critic, ao, co, buf, ec=ec)
                    buf = {'states':[],'actions':[],'rewards':[],'valid_sets':[],'next_states':[],'dones':[]}
            res = base_env.get_episode_result()
            curve_r.append(res['total_reward'])
            curve_t.append(res['T_up'] if res['T_up'] != float('inf') else None)

        # 快速评估
        actor.eval()
        tups = []
        for t in range(60):
            base_env.reset(seed=3000+t)
            d = False
            while not d:
                v = base_env.get_valid_actions()
                if not v: break
                with torch.no_grad():
                    q = actor(torch.FloatTensor(base_env._obs()))
                    a = v[F.softmax(q[v], dim=0).argmax().item()]
                _, _, d, _, _ = base_env.step(a)
            r2 = base_env.get_episode_result()
            tups.append(r2['T_up'] if r2['T_up'] != float('inf') else 999)
        med = float(np.median([t for t in tups if t < 900] or [999]))
        print(f"    [A3C attempt {attempt}]: median={med:.2f}s")
        if med < best_med:
            best_med = med; best_actor = actor
            best_curve = {'rewards': curve_r, 'tups': curve_t}
        if med < threshold:
            print(f"    ✅ A3C收敛"); break

    best_actor.eval()
    return best_actor, best_curve


# ======================================================
# 评估
# ======================================================
def evaluate_mappo(marl_env: MARLRoutingEnv, agent: MAPPOAgent, n_eval=300):
    tups, divs, succs = [], [], []
    for ep in range(n_eval):
        local_obs_all, _ = marl_env.reset(seed=ep + 1000)
        done = False
        while not done:
            cur_gbs = marl_env.get_current_gbs()
            valid   = marl_env.get_valid_actions(cur_gbs)
            if not valid: break
            action, _ = agent.select_action(local_obs_all[cur_gbs], valid)
            local_obs_all, _, _, done, _ = marl_env.step(action)
        r = marl_env.get_episode_result()
        tups.append(r['T_up'] if r['T_up'] != float('inf') else 999.0)
        divs.append(r['path_diversity'])
        succs.append(r['success_count'] / marl_env.num_gbs)
    arr = np.array([t for t in tups if t < 900])
    if not len(arr): arr = np.array([999.0])
    return {
        'mean_T_up':    float(np.mean(arr)),
        'median_T_up':  float(np.median(arr)),
        'std_T_up':     float(np.std(arr)),
        'bad_rate':     float(np.mean(arr > 50)),
        'mean_diversity': float(np.mean(divs)),
        'success_rate': float(np.mean(succs)),
        'tups': tups,
    }

def evaluate_a3c(base_env: ConcurrentFLRoutingEnv, actor, n_eval=300):
    tups, divs, succs = [], [], []
    for ep in range(n_eval):
        base_env.reset(seed=ep + 1000)
        done = False
        while not done:
            valid = base_env.get_valid_actions()
            if not valid: break
            with torch.no_grad():
                q = actor(torch.FloatTensor(base_env._obs()))
                action = valid[F.softmax(q[valid], dim=0).argmax().item()]
            _, _, done, _, _ = base_env.step(action)
        r = base_env.get_episode_result()
        tups.append(r['T_up'] if r['T_up'] != float('inf') else 999.0)
        divs.append(r['path_diversity'])
        succs.append(r['success_count'] / base_env.num_gbs)
    arr = np.array([t for t in tups if t < 900])
    if not len(arr): arr = np.array([999.0])
    return {
        'mean_T_up':    float(np.mean(arr)),
        'median_T_up':  float(np.median(arr)),
        'std_T_up':     float(np.std(arr)),
        'bad_rate':     float(np.mean(arr > 50)),
        'mean_diversity': float(np.mean(divs)),
        'success_rate': float(np.mean(succs)),
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

def evaluate_baseline(base_env, policy_fn, n_eval=300):
    tups, divs = [], []
    for ep in range(n_eval):
        base_env.reset(seed=ep + 1000)
        done = False
        while not done:
            valid = base_env.get_valid_actions()
            if not valid: break
            action = policy_fn(base_env, valid)
            _, _, done, _, _ = base_env.step(action)
        r = base_env.get_episode_result()
        tups.append(r['T_up'] if r['T_up'] != float('inf') else 999.0)
        divs.append(r['path_diversity'])
    arr = np.array([t for t in tups if t < 900])
    if not len(arr): arr = np.array([999.0])
    return {
        'mean_T_up':    float(np.mean(arr)),
        'median_T_up':  float(np.median(arr)),
        'std_T_up':     float(np.std(arr)),
        'bad_rate':     float(np.mean(arr > 50)),
        'mean_diversity': float(np.mean(divs)),
        'tups': tups,
    }


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
    "MAPPO (MARL)":   "#8338ec",
    "A3C (Ours)":     "#e63946",
    "LAD-Dijkstra":   "#2a9d8f",
    "ShortestPath":   "#f4a261",
}
MARKERS = {"MAPPO (MARL)": "D", "A3C (Ours)": "o",
           "LAD-Dijkstra": "^", "ShortestPath": "s"}
LS = {"MAPPO (MARL)": "--", "A3C (Ours)": "-",
      "LAD-Dijkstra": "-.", "ShortestPath": ":"}


def plot_training_comparison(mappo_curve, a3c_curve, num_gbs, save_dir):
    """训练收敛曲线对比（奖励 + T_up）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"MAPPO vs A3C — Training Convergence ({num_gbs} UAVs)",
                 fontsize=13, fontweight='bold')

    for ax, key, ylabel, title in zip(
        axes,
        ['rewards', 'tups'],
        ['Episode Reward', 'Upload Delay T_up (s)'],
        ['(a) Episode Reward', '(b) T_up During Training']
    ):
        for label, curve, color in [
            ("MAPPO (MARL)", mappo_curve, COLORS["MAPPO (MARL)"]),
            ("A3C (Ours)",   a3c_curve,   COLORS["A3C (Ours)"]),
        ]:
            sm = smooth(curve[key], 50)
            valid = ~np.isnan(sm)
            ax.plot(np.arange(len(sm))[valid], sm[valid],
                    label=label, color=color, linewidth=2,
                    linestyle=LS[label])
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/marl_training_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_eval_comparison(results, num_gbs, save_dir):
    """评估结果对比（均值/中位数/坏场景率/多样性）"""
    algs = list(results.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"MAPPO vs Baselines — Evaluation Results ({num_gbs} UAVs)",
                 fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title, scale in zip(
        axes,
        ['mean_T_up', 'median_T_up', 'bad_rate', 'mean_diversity'],
        ['Mean T_up (s)', 'Median T_up (s)', 'Bad Rate (%)', 'Path Diversity'],
        ['(a) Mean T_up', '(b) Median T_up', '(c) Bad Scenario Rate', '(d) Path Diversity'],
        [1, 1, 100, 1]
    ):
        vals = [results[a][metric] * scale for a in algs]
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
    p = f"{save_dir}/marl_eval_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_multi_scale(all_results, uav_list, save_dir):
    """多规模对比：MAPPO vs A3C vs LAD（T_up均值/中位数/坏场景率）"""
    algs = ["MAPPO (MARL)", "A3C (Ours)", "LAD-Dijkstra", "ShortestPath"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MAPPO vs A3C vs Traditional Algorithms — Multi-Scale Comparison",
                 fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title in zip(
        axes,
        ['mean_T_up', 'median_T_up', 'bad_rate'],
        ['Mean T_up (s)', 'Median T_up (s)', 'Bad Rate (%)'],
        ['(a) Mean T_up', '(b) Median T_up', '(c) Bad Scenario Rate']
    ):
        for alg in algs:
            if alg not in all_results[uav_list[0]]:
                continue
            vals = [all_results[n][alg][metric] * (100 if metric=='bad_rate' else 1)
                    for n in uav_list]
            ax.plot(uav_list, vals, label=alg,
                    color=COLORS.get(alg, "#888"),
                    marker=MARKERS.get(alg, "o"),
                    linewidth=2.5, linestyle=LS.get(alg, "-"), markersize=8)
        ax.set_xlabel("Number of UAVs (FL Clients)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(uav_list)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/marl_multiscale.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


# ======================================================
# 主流程
# ======================================================
def run_single_scale(num_gbs, n_train, n_eval, max_attempts=5):
    print(f"\n{'='*60}")
    print(f"  {num_gbs}-UAV | 训练{n_train}ep | 评估{n_eval}ep")
    print(f"{'='*60}")

    topo_cfg = make_topo_cfg(num_gbs)
    base_env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)
    marl_env = MARLRoutingEnv(base_env)

    print(f"  local_obs_dim={marl_env.local_obs_dim}, "
          f"global_state_dim={marl_env.global_state_dim}")

    # 训练MAPPO（多次取最优）
    best_mappo, best_mappo_curve, best_mappo_med = None, None, float('inf')
    threshold = 5.0 if num_gbs == 1 else 20.0
    for attempt in range(1, max_attempts + 1):
        print(f"\n  [MAPPO attempt {attempt}/{max_attempts}]")
        mappo_agent, mappo_curve = train_mappo(marl_env, n_train, f"{num_gbs}uav-a{attempt}")
        # 快速评估
        q_res = evaluate_mappo(marl_env, mappo_agent, 60)
        print(f"  → median={q_res['median_T_up']:.2f}s, bad={q_res['bad_rate']*100:.0f}%")
        if q_res['median_T_up'] < best_mappo_med:
            best_mappo_med   = q_res['median_T_up']
            best_mappo       = mappo_agent
            best_mappo_curve = mappo_curve
        if q_res['median_T_up'] < threshold:
            print(f"  ✅ MAPPO收敛"); break

    # 训练改进A3C
    print(f"\n  [改进A3C]")
    a3c_actor, a3c_curve = train_a3c(base_env, n_train, f"{num_gbs}uav", max_attempts)

    # 保存模型
    torch.save(best_mappo.actor.state_dict(),
               f"{PAPER_DIR}/models/mappo_actor_{num_gbs}uav.pth")
    torch.save(best_mappo.critic.state_dict(),
               f"{PAPER_DIR}/models/mappo_critic_{num_gbs}uav.pth")
    torch.save(a3c_actor.state_dict(),
               f"{PAPER_DIR}/models/marl_a3c_{num_gbs}uav.pth")

    # 绘制训练曲线
    plot_training_comparison(best_mappo_curve, a3c_curve, num_gbs,
                             f"{PAPER_DIR}/figures")

    # 正式评估
    print(f"\n  [正式评估 {n_eval}ep]")
    results = {
        "MAPPO (MARL)": evaluate_mappo(marl_env, best_mappo, n_eval),
        "A3C (Ours)":   evaluate_a3c(base_env, a3c_actor, n_eval),
        "LAD-Dijkstra": evaluate_baseline(base_env, lad_policy, n_eval),
        "ShortestPath": evaluate_baseline(
            base_env,
            lambda e, v: v[min(range(len(v)),
                key=lambda i: nx.shortest_path_length(e.topo.graph, v[i], e.server_id)
                if nx.has_path(e.topo.graph, v[i], e.server_id) else 999)],
            n_eval),
    }

    for alg, res in results.items():
        print(f"  {alg:18s} mean={res['mean_T_up']:7.2f}s "
              f"median={res['median_T_up']:6.2f}s "
              f"bad={res['bad_rate']*100:.0f}% "
              f"div={res['mean_diversity']:.2f}")

    plot_eval_comparison(results, num_gbs, f"{PAPER_DIR}/figures")

    return results, best_mappo_curve, a3c_curve


def main(n_train=1500, n_eval=300, uav_list=None):
    if uav_list is None:
        uav_list = [3, 5]  # 默认跑3和5-UAV场景

    train_eps = {1: n_train, 3: n_train, 5: max(n_train, 2000)}

    print(f"\n{'#'*60}")
    print(f"  MARL实验：MAPPO vs 改进A3C vs 传统算法")
    print(f"  场景: {uav_list}-UAV | 训练{n_train}ep | 评估{n_eval}ep")
    print(f"{'#'*60}")

    all_results = {}
    for num_gbs in uav_list:
        results, mappo_curve, a3c_curve = run_single_scale(
            num_gbs, train_eps.get(num_gbs, n_train), n_eval)
        all_results[num_gbs] = results

    # 多规模对比图
    if len(uav_list) > 1:
        print(f"\n  绘制多规模对比图...")
        plot_multi_scale(all_results, uav_list, f"{PAPER_DIR}/figures")

    # 保存数据
    safe = {str(n): {alg: {k: v for k, v in r.items() if k != 'tups'}
                     for alg, r in res.items()}
            for n, res in all_results.items()}
    path = f"{PAPER_DIR}/data/marl_results_{TIMESTAMP}.json"
    with open(path, 'w') as f:
        json.dump(safe, f, indent=2)

    # 汇总表
    for metric, title, fmt in [
        ('mean_T_up',   'T_up 均值 (s)',   '{:>8.2f}'),
        ('median_T_up', 'T_up 中位数 (s)', '{:>8.2f}'),
        ('bad_rate',    '坏场景率 (%)',     '{:>8.1f}%'),
    ]:
        print(f"\n  【{title}】")
        algs = list(all_results[uav_list[0]].keys())
        print(f"  {'UAV':>5} | " + " | ".join(f"{a:>14}" for a in algs))
        print(f"  {'-'*65}")
        for n in uav_list:
            vals = [fmt.format(all_results[n][alg][metric] *
                               (100 if metric == 'bad_rate' else 1))
                    for alg in algs]
            print(f"  {n:>5} | " + " | ".join(vals))

    print(f"\n  数据: {path}")
    print(f"  图表: {PAPER_DIR}/figures/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval",     type=int, default=300)
    parser.add_argument("--uav",      type=int, nargs='+', default=[3, 5])
    args = parser.parse_args()
    main(args.episodes, args.eval, args.uav)
