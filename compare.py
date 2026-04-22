"""
公平对比评估脚本
- 固定相同随机种子，A3C和基线在完全相同的初始网络状态下对比
- 分低/中/高负载三个场景
- 输出T_up、成功率、路径长度对比
"""
import os
import sys
import json
import numpy as np
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.fl_routing_env import FLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet
from src.baselines.shortest_path import (
    shortest_path_policy, load_aware_dijkstra_policy, random_policy,
    _compute_dijkstra_path
)

# ======================================================
# 场景配置
# ======================================================
SCENARIOS = {
    "低负载": TopologyConfig(
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(5, 15),
        init_link_load_range=(0.1, 0.3),
        link_failure_prob=0.05,
    ),
    "中负载": TopologyConfig(
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(15, 35),
        init_link_load_range=(0.3, 0.65),
        link_failure_prob=0.15,
    ),
    "高负载": TopologyConfig(
        node_capacity=50,
        gbs_to_router_capacity=20.0,
        router_to_router_capacity=40.0,
        router_to_server_capacity=80.0,
        init_node_load_range=(30, 45),
        init_link_load_range=(0.55, 0.85),
        link_failure_prob=0.25,
    ),
}

DELAY_CFG = DelayConfig(model_size=10.0, t_agg=0.5)
ENV_BASE_KWARGS = dict(
    delay_cfg=DELAY_CFG, delta_t=5.0, num_slots=100,
    g_hop=-1.0, alpha_1=0.4, alpha_2=0.1, w_delay=0.5,
    r_success=20.0, r_fail=-5.0, r_loop=-2.0, beta_tup=2.0,
    max_steps_per_gbs=50,
)

N_EVAL = 200   # 每个场景评估次数
HIDDEN_DIM = 256
OUTPUT_DIR = "outputs"


def load_a3c_policy(env: FLRoutingEnv) -> PolicyNet:
    """加载最新训练好的A3C策略"""
    import glob
    # 尝试加载训练好的模型，若无则用随机初始化
    actor = PolicyNet(env.obs_dim, HIDDEN_DIM, env.action_space_n)
    model_files = sorted(glob.glob(f"{OUTPUT_DIR}/models/actor_*.pth"))
    if model_files:
        actor.load_state_dict(torch.load(model_files[-1], map_location='cpu'))
        print(f"  加载模型: {model_files[-1]}")
    else:
        print("  未找到保存的模型，使用当前训练状态（需先运行train_single.py保存模型）")
    actor.eval()
    return actor


def a3c_policy(actor: PolicyNet, env: FLRoutingEnv, valid_actions):
    """A3C贪婪推理（取概率最大的动作，不采样）"""
    with torch.no_grad():
        state = torch.FloatTensor(env._obs())
        logits = actor(state)
        valid_logits = logits[valid_actions]
        probs = F.softmax(valid_logits, dim=0)
        idx = probs.argmax().item()
    return valid_actions[idx]


def run_policy(env, policy_fn, n_episodes, seed_offset=0):
    results = []
    for ep in range(n_episodes):
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
            "path_lens": [len(r["gbs_paths"][g]) - 1 for g in range(env.num_gbs)
                          if r["gbs_success"][g]],
        })
    return results


def run_offline_dijkstra(env, n_episodes, seed_offset=0):
    """
    离线Dijkstra：每个GBS路由开始时计算一次最优路径，然后沿路执行
    不能中途感知链路变化 —— 更贴近真实系统
    """
    results = []
    for ep in range(n_episodes):
        env.reset(seed=ep + seed_offset)
        done = False
        planned_path: List[int] = []
        path_idx = 0
        last_gbs = -1

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            # 新GBS开始时计算路径
            if env.current_gbs != last_gbs:
                last_gbs = env.current_gbs
                planned_path = _compute_dijkstra_path(env, env.current_node)
                path_idx = 1  # 跳过起点

            # 沿预计算路径走；若路径失效则退化为在线Dijkstra
            if path_idx < len(planned_path):
                action = planned_path[path_idx]
                if action in valid:
                    path_idx += 1
                else:
                    # 预计算路径失效，临时用在线Dijkstra
                    action = load_aware_dijkstra_policy(env, valid)
            else:
                action = load_aware_dijkstra_policy(env, valid)

            _, _, done, _, _ = env.step(action)

        r = env.get_episode_result()
        results.append({
            "T_up": r["T_up"] if r["T_up"] != float('inf') else 999.0,
            "success": r["success_count"] / env.num_gbs,
            "reward": r["total_reward"],
            "path_lens": [len(r["gbs_paths"][g]) - 1 for g in range(env.num_gbs)
                          if r["gbs_success"][g]],
        })
    return results


def summarize(results, label):
    tups = [r["T_up"] for r in results if r["T_up"] < 900]
    succs = [r["success"] for r in results]
    path_lens = []
    for r in results:
        path_lens.extend(r["path_lens"])
    print(f"  {label:20s} | T_up={np.mean(tups):7.3f}±{np.std(tups):.2f}s "
          f"| Succ={np.mean(succs):.3f} "
          f"| AvgHops={np.mean(path_lens):.2f}" if path_lens else
          f"  {label:20s} | T_up={np.mean(tups):7.3f}±{np.std(tups):.2f}s "
          f"| Succ={np.mean(succs):.3f}")
    return {
        "label": label,
        "mean_T_up": float(np.mean(tups)) if tups else 999.0,
        "std_T_up": float(np.std(tups)) if tups else 0.0,
        "success_rate": float(np.mean(succs)),
        "mean_hops": float(np.mean(path_lens)) if path_lens else 0.0,
        "tups": [r["T_up"] for r in results],
    }


def plot_comparison(all_scenario_results, save_dir):
    scenarios = list(all_scenario_results.keys())
    algorithms = list(all_scenario_results[scenarios[0]].keys())
    colors = {"Random": "#aaaaaa", "ShortestPath": "#f4a261",
              "LoadAwareDijkstra": "#2a9d8f", "A3C": "#e63946"}

    # 图1：各场景T_up对比（柱状图）
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, sc in zip(axes, scenarios):
        means = [all_scenario_results[sc][alg]["mean_T_up"] for alg in algorithms]
        stds  = [all_scenario_results[sc][alg]["std_T_up"]  for alg in algorithms]
        cols  = [colors.get(alg, "#555") for alg in algorithms]
        bars = ax.bar(algorithms, means, yerr=stds, color=cols, capsize=4, alpha=0.85)
        ax.set_title(f"{sc}场景", fontsize=13)
        ax.set_ylabel("T_up (s)")
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=15, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        # 标注A3C的值
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
    plt.suptitle("FL上传时延 T_up 对比（不同负载场景）", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{save_dir}/tup_bar_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved: {path}")

    # 图2：成功率对比
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scenarios))
    width = 0.2
    for i, alg in enumerate(algorithms):
        succs = [all_scenario_results[sc][alg]["success_rate"] for sc in scenarios]
        ax.bar(x + i*width, succs, width, label=alg,
               color=colors.get(alg, "#555"), alpha=0.85)
    ax.set_xticks(x + width * (len(algorithms)-1)/2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("路由成功率")
    ax.set_ylim(0, 1.1)
    ax.set_title("路由成功率对比", fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/success_bar_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved: {path}")

    # 图3：T_up分布箱线图（中负载场景）
    sc = "中负载"
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for alg in algorithms:
        tups = [t for t in all_scenario_results[sc][alg]["tups"] if t < 900]
        if tups:
            data.append(tups)
            labels.append(alg)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, alg in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(alg, "#555"))
        patch.set_alpha(0.7)
    ax.set_ylabel("T_up (s)")
    ax.set_title(f"T_up分布（{sc}场景）", fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/tup_boxplot_medium.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved: {path}")


def main():
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

    # 构建A3C策略（用临时env获取obs_dim）
    tmp_env = FLRoutingEnv(
        topo_cfg=SCENARIOS["中负载"], **ENV_BASE_KWARGS
    )
    actor = load_a3c_policy(tmp_env)

    all_results = {}
    print("=" * 65)
    print(f"公平对比评估  每场景{N_EVAL}个episode")
    print("=" * 65)

    for sc_name, topo_cfg in SCENARIOS.items():
        print(f"\n【{sc_name}场景】")
        env = FLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE_KWARGS)
        sc_results = {}

        r_random = run_policy(env, random_policy, N_EVAL, seed_offset=0)
        sc_results["Random"] = summarize(r_random, "Random")

        r_sp = run_policy(env, shortest_path_policy, N_EVAL, seed_offset=0)
        sc_results["ShortestPath"] = summarize(r_sp, "ShortestPath")

        r_lad = run_policy(env, load_aware_dijkstra_policy, N_EVAL, seed_offset=0)
        sc_results["LAD-Online"] = summarize(r_lad, "LAD-Online")

        r_lad_off = run_offline_dijkstra(env, N_EVAL, seed_offset=0)
        sc_results["LAD-Offline"] = summarize(r_lad_off, "LAD-Offline")

        r_a3c = run_policy(env, lambda e, v: a3c_policy(actor, e, v), N_EVAL, seed_offset=0)
        sc_results["A3C"] = summarize(r_a3c, "A3C")

        all_results[sc_name] = sc_results

    print("\n绘制对比图表...")
    plot_comparison(all_results, f"{OUTPUT_DIR}/figures")

    with open(f"{OUTPUT_DIR}/logs/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果保存至: {OUTPUT_DIR}/logs/comparison_results.json")

    # 打印汇总表
    print("\n" + "=" * 70)
    print("汇总：各算法 T_up 均值对比（单位s）")
    print(f"{'场景':8s} | {'A3C':>8s} | {'LAD-Online':>12s} | {'LAD-Offline':>13s} | {'SP':>8s}")
    print("-" * 60)
    for sc in SCENARIOS:
        a3c_t  = all_results[sc]["A3C"]["mean_T_up"]
        lad_on = all_results[sc]["LAD-Online"]["mean_T_up"]
        lad_of = all_results[sc]["LAD-Offline"]["mean_T_up"]
        sp_t   = all_results[sc]["ShortestPath"]["mean_T_up"]
        print(f"{sc:8s} | {a3c_t:>8.3f} | {lad_on:>12.3f} | {lad_of:>13.3f} | {sp_t:>8.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
