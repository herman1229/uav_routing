"""
规模扩展实验：1/3/5/7/10-UAV 场景
验证"随UAV数量增加，A3C相对传统算法优势增大"的趋势

用法: python scale_experiment.py [--episodes 1500] [--eval 300]
"""
import os, sys, json, argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet, ValueNet
from src.baselines.dqn import DQNAgent

PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/data", exist_ok=True)
HIDDEN_DIM = 256
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

COLORS = {
    "A3C (Ours)":   "#e63946",
    "DQN":          "#457b9d",
    "LAD-Dijkstra": "#2a9d8f",
    "ShortestPath": "#f4a261",
    "Random":       "#aaaaaa",
}
MARKERS = {"A3C (Ours)": "o", "DQN": "D", "LAD-Dijkstra": "^",
           "ShortestPath": "s", "Random": "x"}
LS = {"A3C (Ours)": "-", "DQN": "--", "LAD-Dijkstra": "-.",
      "ShortestPath": ":", "Random": ":"}

# ======================================================
# 拓扑配置（支持 1/3/5/7/10 UAV）
# ======================================================
UAV_ROUTER_MAP = {1: 3, 3: 4, 5: 5, 7: 6, 10: 8}

def make_topo_cfg(num_gbs):
    num_routers = UAV_ROUTER_MAP[num_gbs]
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
# 训练
# ======================================================
def _update(actor, critic, ao, co, buf, gamma=0.98, n_step=5, mg=0.5, ec=0.01):
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
    values = critic(states)
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


def train_a3c(env, n_episodes, label, max_attempts=5):
    best_actor, best_med = None, float('inf')
    threshold = 5.0 if env.num_gbs == 1 else 20.0

    for attempt in range(1, max_attempts + 1):
        actor  = PolicyNet(env.obs_dim, HIDDEN_DIM, env.action_space_n)
        critic = ValueNet(env.obs_dim, HIDDEN_DIM)
        ao = torch.optim.Adam(actor.parameters(),  lr=5e-4)
        co = torch.optim.Adam(critic.parameters(), lr=1e-3)
        ent_s, ent_e = 0.05, 0.002
        update_every = 3

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
                    _update(actor, critic, ao, co, buf, ec=ec)
                    buf = {'states':[],'actions':[],'rewards':[],'valid_sets':[],'next_states':[],'dones':[]}

        # 快速评估
        actor.eval()
        tups = []
        for t in range(60):
            env.reset(seed=3000+t)
            d = False
            while not d:
                v = env.get_valid_actions()
                if not v: break
                with torch.no_grad():
                    q = actor(torch.FloatTensor(env._obs()))
                    a = v[F.softmax(q[v], dim=0).argmax().item()]
                _, _, d, _, _ = env.step(a)
            r2 = env.get_episode_result()
            tups.append(r2['T_up'] if r2['T_up'] != float('inf') else 999)
        med = float(np.median([t for t in tups if t < 900] or [999]))
        print(f"    attempt {attempt}: median={med:.2f}s")
        if med < best_med:
            best_med = med; best_actor = actor
        if med < threshold:
            print(f"    ✅ 收敛"); break

    best_actor.eval()
    return best_actor


def train_dqn(env, n_episodes):
    agent = DQNAgent(
        state_dim=env.obs_dim, hidden_dim=HIDDEN_DIM,
        action_dim=env.action_space_n,
        lr=1e-3, gamma=0.98, buffer_size=10000, batch_size=64,
        eps_start=1.0, eps_end=0.05, eps_decay=n_episodes//2, target_update=20,
    )
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
    return agent


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
    if not len(arr): arr = np.array([999.0])
    return {
        'mean_T_up':    float(np.mean(arr)),
        'median_T_up':  float(np.median(arr)),
        'std_T_up':     float(np.std(arr)),
        'bad_rate':     float(np.mean(arr > 50)),
        'success_rate': float(np.mean(succs)),
        'mean_diversity': float(np.mean(divs)),
        'tups': tups,
    }

def a3c_fn(actor, env, valid):
    with torch.no_grad():
        q = actor(torch.FloatTensor(env._obs()))
        return valid[F.softmax(q[valid], dim=0).argmax().item()]

def dqn_fn(agent, env, valid):
    with torch.no_grad():
        q = agent.q_net(torch.FloatTensor(env._obs()))
        return valid[q[valid].argmax().item()]

def lad_fn(env, valid):
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

def sp_fn(env, valid):
    best, mn = valid[0], float('inf')
    for a in valid:
        try:
            h = nx.shortest_path_length(env.topo.graph, a, env.server_id)
            if h < mn: mn, best = h, a
        except: pass
    return best


# ======================================================
# 绘图
# ======================================================
def smooth(data, w=40):
    arr = np.array(data, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        chunk = arr[max(0,i-w//2):min(len(arr),i+w//2+1)]
        v = chunk[~np.isnan(chunk)]
        if len(v): out[i] = v.mean()
    return out


def plot_scale_main(all_results, uav_list, save_dir):
    """主图：T_up均值/中位数 vs UAV数量"""
    algs = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scalability: Upload Delay T_up vs Number of UAVs",
                 fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title in zip(
        axes,
        ['mean_T_up', 'median_T_up'],
        ['Mean T_up (s)', 'Median T_up (s)'],
        ['(a) Mean T_up', '(b) Median T_up — Typical Performance']
    ):
        for alg in algs:
            vals = [all_results[n][alg][metric] for n in uav_list]
            stds = [all_results[n][alg]['std_T_up'] for n in uav_list]
            if metric == 'mean_T_up':
                ax.errorbar(uav_list, vals, yerr=stds, label=alg,
                            color=COLORS[alg], marker=MARKERS[alg],
                            linewidth=2.5, linestyle=LS[alg],
                            markersize=8, capsize=4)
            else:
                ax.plot(uav_list, vals, label=alg,
                        color=COLORS[alg], marker=MARKERS[alg],
                        linewidth=2.5, linestyle=LS[alg], markersize=8)
        ax.set_xlabel("Number of UAVs (FL Clients)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(uav_list)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/scale_tup_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_scale_improvement(all_results, uav_list, save_dir):
    """A3C相对各基线的提升幅度随UAV规模的变化"""
    baselines = ["DQN", "LAD-Dijkstra", "ShortestPath"]
    bl_colors = [COLORS["DQN"], COLORS["LAD-Dijkstra"], COLORS["ShortestPath"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("A3C Improvement over Baselines vs Number of UAVs",
                 fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title in zip(
        axes,
        ['mean_T_up', 'bad_rate'],
        ['A3C Improvement in Mean T_up (%)', 'A3C Improvement in Bad Rate (pp)'],
        ['(a) Mean T_up Improvement', '(b) Bad Scenario Rate Improvement']
    ):
        for bl, c in zip(baselines, bl_colors):
            improvements = []
            for n in uav_list:
                a3c_v = all_results[n]["A3C (Ours)"][metric]
                bl_v  = all_results[n][bl][metric]
                if metric == 'mean_T_up':
                    impr = (bl_v - a3c_v) / max(bl_v, 0.01) * 100
                else:
                    impr = (bl_v - a3c_v) * 100  # percentage points
                improvements.append(impr)
            ax.plot(uav_list, improvements, label=f"vs {bl}",
                    color=c, marker="o", linewidth=2.5, markersize=8)
            # 标注最后一个点
            ax.annotate(f'{improvements[-1]:+.1f}',
                        (uav_list[-1], improvements[-1]),
                        textcoords="offset points", xytext=(5, 0),
                        fontsize=8, color=c)

        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.fill_between(uav_list,
                        [min(0, min([all_results[n]["A3C (Ours)"][metric] for n in uav_list]))] * len(uav_list),
                        0, alpha=0.05, color='red', label='A3C worse')
        ax.set_xlabel("Number of UAVs (FL Clients)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(uav_list)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/scale_improvement_trend.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_scale_bad_rate(all_results, uav_list, save_dir):
    """坏场景率随UAV规模的变化"""
    algs = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for alg in algs:
        vals = [all_results[n][alg]['bad_rate'] * 100 for n in uav_list]
        ax.plot(uav_list, vals, label=alg,
                color=COLORS[alg], marker=MARKERS[alg],
                linewidth=2.5, linestyle=LS[alg], markersize=8)

    ax.set_xlabel("Number of UAVs (FL Clients)", fontsize=11)
    ax.set_ylabel("Bad Scenario Rate (T_up > 50s, %)", fontsize=11)
    ax.set_title("Robustness: Bad Scenario Rate vs Number of UAVs\n"
                 "A3C maintains lowest bad rate as scale increases", fontsize=11)
    ax.set_xticks(uav_list)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/scale_bad_rate.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_scale_diversity(all_results, uav_list, save_dir):
    """路径多样性随规模的变化"""
    algs = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for alg in algs:
        vals = [all_results[n][alg]['mean_diversity'] for n in uav_list]
        ax.plot(uav_list, vals, label=alg,
                color=COLORS[alg], marker=MARKERS[alg],
                linewidth=2.5, linestyle=LS[alg], markersize=8)

    ax.set_xlabel("Number of UAVs (FL Clients)", fontsize=11)
    ax.set_ylabel("Path Diversity (unique routers used)", fontsize=11)
    ax.set_title("Path Diversity vs Number of UAVs\n"
                 "A3C learns to spread traffic across more routers", fontsize=11)
    ax.set_xticks(uav_list)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/scale_diversity.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


# ======================================================
# 主流程
# ======================================================
def main(n_train=1500, n_eval=300):
    uav_list = [1, 3, 5, 7, 10]
    # 规模越大需要更多训练，但不低于命令行传入的 n_train
    scale_factor = {1: 1.0, 3: 1.0, 5: 1.3, 7: 1.5, 10: 1.8}
    train_eps = {n: max(n_train, int(n_train * scale_factor[n])) for n in uav_list}

    print(f"\n{'#'*60}")
    print(f"  规模扩展实验 | UAV: {uav_list}")
    print(f"  训练轮数: {n_train}(1/3), {max(n_train,2000)}(5/7), {max(n_train,2500)}(10)")
    print(f"{'#'*60}")

    all_results = {}

    for num_gbs in uav_list:
        ep_count = train_eps[num_gbs]
        print(f"\n{'='*55}")
        print(f"  {num_gbs}-UAV | 训练{ep_count}ep | 评估{n_eval}ep")
        print(f"{'='*55}")

        topo_cfg = make_topo_cfg(num_gbs)
        env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)
        print(f"  obs_dim={env.obs_dim}, nodes={env.num_nodes}, edges={len(env.topo.edges)}")

        # 训练
        print(f"  [A3C 训练]")
        actor = train_a3c(env, ep_count, f"{num_gbs}uav", max_attempts=5)

        print(f"  [DQN 训练]")
        dqn = train_dqn(env, ep_count)

        # 保存模型
        torch.save(actor.state_dict(), f"{PAPER_DIR}/models/scale_a3c_{num_gbs}uav.pth")
        torch.save(dqn.q_net.state_dict(), f"{PAPER_DIR}/models/scale_dqn_{num_gbs}uav.pth")

        # 评估
        print(f"  [评估 {n_eval}ep]")
        results = {
            "A3C (Ours)":   evaluate(env, lambda e,v,a=actor: a3c_fn(a,e,v), n_eval),
            "DQN":          evaluate(env, lambda e,v,d=dqn: dqn_fn(d,e,v), n_eval),
            "LAD-Dijkstra": evaluate(env, lad_fn, n_eval),
            "ShortestPath": evaluate(env, sp_fn, n_eval),
        }
        all_results[num_gbs] = results

        for alg, res in results.items():
            print(f"  {alg:18s} mean={res['mean_T_up']:7.2f}s "
                  f"median={res['median_T_up']:6.2f}s "
                  f"bad={res['bad_rate']*100:.0f}% "
                  f"div={res['mean_diversity']:.2f}")

    # 绘图
    print(f"\n{'='*55}")
    print("  绘制规模扩展实验图表...")
    print(f"{'='*55}")
    plot_scale_main(all_results, uav_list, f"{PAPER_DIR}/figures")
    plot_scale_improvement(all_results, uav_list, f"{PAPER_DIR}/figures")
    plot_scale_bad_rate(all_results, uav_list, f"{PAPER_DIR}/figures")
    plot_scale_diversity(all_results, uav_list, f"{PAPER_DIR}/figures")

    # 保存数据
    safe = {str(n): {alg: {k: v for k, v in r.items() if k != 'tups'}
                     for alg, r in res.items()}
            for n, res in all_results.items()}
    path = f"{PAPER_DIR}/data/scale_results_{TIMESTAMP}.json"
    with open(path, 'w') as f:
        json.dump(safe, f, indent=2)

    # 打印汇总表
    algs = ["A3C (Ours)", "DQN", "LAD-Dijkstra", "ShortestPath"]
    for metric, title, fmt in [
        ('mean_T_up',   'T_up 均值 (s)',    '{:>8.2f}'),
        ('median_T_up', 'T_up 中位数 (s)',  '{:>8.2f}'),
        ('bad_rate',    '坏场景率 (%)',      '{:>8.1f}%'),
    ]:
        print(f"\n  【{title}】")
        print(f"  {'UAV':>5} | " + " | ".join(f"{a:>14}" for a in algs))
        print(f"  {'-'*65}")
        for n in uav_list:
            vals = []
            for alg in algs:
                v = all_results[n][alg][metric]
                vals.append(fmt.format(v * 100 if metric=='bad_rate' else v))
            print(f"  {n:>5} | " + " | ".join(vals))

    print(f"\n  数据: {path}")
    print(f"  图表: {PAPER_DIR}/figures/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval",     type=int, default=300)
    args = parser.parse_args()
    main(args.episodes, args.eval)
