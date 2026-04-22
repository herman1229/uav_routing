"""
并发多流路由：A3C vs 基线公平对比
基线的并发版：每个GBS轮流决策，但每步用当前状态做贪婪选择（不感知其他GBS位置）
"""
import os, sys, json, glob
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet

HIDDEN_DIM = 256
N_EVAL = 300
OUTPUT_DIR = "outputs"

SCENARIOS = {
    "Low Load": TopologyConfig(
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(5, 15),
        init_link_load_range=(0.1, 0.3),
        link_failure_prob=0.05,
        step_failure_prob=0.05,
    ),
    "Medium Load": TopologyConfig(
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(15, 35),
        init_link_load_range=(0.3, 0.65),
        link_failure_prob=0.15,
        step_failure_prob=0.08,
    ),
    "High Load": TopologyConfig(
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(30, 45),
        init_link_load_range=(0.55, 0.85),
        link_failure_prob=0.25,
        step_failure_prob=0.12,
    ),
}

ENV_BASE = dict(
    delay_cfg=DelayConfig(model_size=10.0, t_agg=0.5),
    delta_t=5.0, num_slots=100,
    g_hop=-1.0, alpha_1=0.4, alpha_2=0.1,
    w_delay=0.5, r_success=20.0, r_fail=-5.0,
    r_loop=-2.0, r_compete=-1.5, r_diversity=0.3, beta_tup=3.0,
    max_steps_per_gbs=50,
)


# ------------------------------------------------------------------
# 基线策略（并发版）
# ------------------------------------------------------------------

def random_concurrent(env, valid):
    return random.choice(valid)


def shortest_path_concurrent(env, valid):
    best, min_h = valid[0], float('inf')
    for a in valid:
        try:
            h = nx.shortest_path_length(env.topo.graph, a, env.server_id)
            if h < min_h:
                min_h, best = h, a
        except nx.NetworkXNoPath:
            pass
    return best


def lad_concurrent(env, valid):
    """负载感知Dijkstra（在线版，每步重新计算，但不感知其他GBS）"""
    M = env.delay_model.cfg.model_size
    L = env.delay_model.cfg.packet_size
    wg = nx.DiGraph()
    for u, v in env.topo.edges:
        bw = max(env.topo.available_bandwidth(u, v), 0.1)
        wg.add_edge(u, v, weight=(M + L) / bw)
    best, min_d = valid[0], float('inf')
    for a in valid:
        try:
            d = nx.shortest_path_length(wg, a, env.server_id, weight='weight')
            if d < min_d:
                min_d, best = d, a
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return best


def a3c_concurrent(actor, env, valid):
    with torch.no_grad():
        s = torch.FloatTensor(env._obs())
        logits = actor(s)[valid]
        idx = F.softmax(logits, dim=0).argmax().item()
    return valid[idx]


# ------------------------------------------------------------------
# 运行 episode
# ------------------------------------------------------------------

def run_episodes(env, policy_fn, n, seed_offset=0):
    results = []
    for ep in range(n):
        env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = policy_fn(env, valid)
            _, _, done, _, _ = env.step(action)
        r = env.get_episode_result()
        results.append({
            "T_up": r["T_up"] if r["T_up"] != float('inf') else 999.0,
            "success": r["success_count"] / env.num_gbs,
            "reward": r["total_reward"],
            "diversity": r["path_diversity"],
        })
    return results


def summarize(results, label):
    tups = [r["T_up"] for r in results if r["T_up"] < 900]
    succs = [r["success"] for r in results]
    divs = [r["diversity"] for r in results]
    mean_t = np.mean(tups) if tups else 999.0
    std_t = np.std(tups) if tups else 0.0
    print(f"  {label:22s} | T_up={mean_t:8.3f}±{std_t:.2f}s "
          f"| Succ={np.mean(succs):.3f} | Diversity={np.mean(divs):.2f}")
    return {
        "label": label,
        "mean_T_up": float(mean_t),
        "std_T_up": float(std_t),
        "success_rate": float(np.mean(succs)),
        "mean_diversity": float(np.mean(divs)),
        "tups": [r["T_up"] for r in results],
    }


# ------------------------------------------------------------------
# 绘图
# ------------------------------------------------------------------

def plot_all(all_results, save_dir):
    scenarios = list(all_results.keys())
    algorithms = list(all_results[scenarios[0]].keys())
    colors = {
        "Random": "#aaaaaa",
        "ShortestPath": "#f4a261",
        "LAD-Concurrent": "#2a9d8f",
        "A3C-Concurrent": "#e63946",
    }

    # 柱状图：T_up
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))
    for ax, sc in zip(axes, scenarios):
        means = [all_results[sc][a]["mean_T_up"] for a in algorithms]
        stds  = [all_results[sc][a]["std_T_up"]  for a in algorithms]
        cols  = [colors.get(a, "#888") for a in algorithms]
        bars = ax.bar(algorithms, means, yerr=stds, color=cols, capsize=4, alpha=0.85)
        ax.set_title(sc, fontsize=12)
        ax.set_ylabel("T_up (s)")
        ax.set_xticklabels(algorithms, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{m:.1f}', ha='center', va='bottom', fontsize=7)
    plt.suptitle("Concurrent Multi-Flow Routing: T_up Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = f"{save_dir}/concurrent_tup_bar.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"  saved: {p}")

    # 路径多样性对比
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(scenarios))
    w = 0.2
    for i, alg in enumerate(algorithms):
        divs = [all_results[sc][alg]["mean_diversity"] for sc in scenarios]
        ax.bar(x + i*w, divs, w, label=alg, color=colors.get(alg, "#888"), alpha=0.85)
    ax.set_xticks(x + w*(len(algorithms)-1)/2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Path Diversity (unique routers used)")
    ax.set_title("Path Diversity: A3C learns to spread traffic across routers")
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/concurrent_diversity.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"  saved: {p}")

    # 箱线图（中负载）
    sc = "Medium Load"
    fig, ax = plt.subplots(figsize=(8, 4))
    data, labels = [], []
    for alg in algorithms:
        tups = [t for t in all_results[sc][alg]["tups"] if t < 900]
        if tups:
            data.append(tups); labels.append(alg)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, alg in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(alg, "#888")); patch.set_alpha(0.7)
    ax.set_ylabel("T_up (s)")
    ax.set_title(f"T_up Distribution ({sc})")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/concurrent_tup_box.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"  saved: {p}")


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------

def main():
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

    # 加载A3C模型
    tmp_env = ConcurrentFLRoutingEnv(topo_cfg=SCENARIOS["Medium Load"], **ENV_BASE)
    actor = PolicyNet(tmp_env.obs_dim, HIDDEN_DIM, tmp_env.action_space_n)
    model_files = sorted(glob.glob(f"{OUTPUT_DIR}/models/concurrent_actor_*.pth"))
    if model_files:
        actor.load_state_dict(torch.load(model_files[-1], map_location='cpu'))
        print(f"加载模型: {model_files[-1]}")
    else:
        print("未找到并发模型，请先运行 train_concurrent.py")
        return
    actor.eval()

    all_results = {}
    print("=" * 70)
    print(f"并发多流路由公平对比  每场景{N_EVAL}个episode")
    print("=" * 70)

    for sc_name, topo_cfg in SCENARIOS.items():
        print(f"\n[{sc_name}]")
        env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)
        sc_res = {}
        sc_res["Random"]         = summarize(run_episodes(env, random_concurrent, N_EVAL), "Random")
        sc_res["ShortestPath"]   = summarize(run_episodes(env, shortest_path_concurrent, N_EVAL), "ShortestPath")
        sc_res["LAD-Concurrent"] = summarize(run_episodes(env, lad_concurrent, N_EVAL), "LAD-Concurrent")
        sc_res["A3C-Concurrent"] = summarize(
            run_episodes(env, lambda e, v: a3c_concurrent(actor, e, v), N_EVAL),
            "A3C-Concurrent"
        )
        all_results[sc_name] = sc_res

    print("\n绘制图表...")
    plot_all(all_results, f"{OUTPUT_DIR}/figures")

    with open(f"{OUTPUT_DIR}/logs/concurrent_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("汇总：T_up 均值 (s)")
    print(f"{'Scenario':14s} | {'A3C':>8s} | {'LAD':>8s} | {'SP':>8s} | {'Improvement':>12s}")
    print("-" * 60)
    for sc in SCENARIOS:
        a = all_results[sc]["A3C-Concurrent"]["mean_T_up"]
        l = all_results[sc]["LAD-Concurrent"]["mean_T_up"]
        s = all_results[sc]["ShortestPath"]["mean_T_up"]
        impr = (l - a) / l * 100 if l > 0 else 0
        print(f"{sc:14s} | {a:>8.3f} | {l:>8.3f} | {s:>8.3f} | {impr:>+11.1f}%")
    print("=" * 70)

    print("\n路径多样性（A3C是否学会分散路径）:")
    print(f"{'Scenario':14s} | {'A3C':>6s} | {'LAD':>6s} | {'SP':>6s}")
    print("-" * 40)
    for sc in SCENARIOS:
        a = all_results[sc]["A3C-Concurrent"]["mean_diversity"]
        l = all_results[sc]["LAD-Concurrent"]["mean_diversity"]
        s = all_results[sc]["ShortestPath"]["mean_diversity"]
        print(f"{sc:14s} | {a:>6.2f} | {l:>6.2f} | {s:>6.2f}")


if __name__ == "__main__":
    main()
