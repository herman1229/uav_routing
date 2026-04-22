"""
基线对比评估入口
运行三种基线算法并输出对比指标
用法: python evaluate.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.fl_routing_env import FLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.baselines import run_random, run_shortest_path, run_load_aware_dijkstra

N_EPISODES = 500
SEED = 42
OUTPUT_DIR = "outputs"

ENV_KWARGS = dict(
    topo_cfg=TopologyConfig(node_capacity=50, link_capacity=100.0),
    delay_cfg=DelayConfig(model_size=10.0, t_agg=0.5),
    delta_t=5.0, num_slots=100,
    g_hop=-1.0, alpha_1=0.4, alpha_2=0.1, w_delay=0.5,
    r_success=10.0, r_fail=-10.0, max_steps_per_gbs=30,
)


def summarize(results, name):
    rewards = [r["total_reward"] for r in results]
    success = [r["success_count"] / 3 for r in results]
    t_ups = [r["T_up"] for r in results if r["T_up"] != float('inf') and r["T_up"] > 0]
    print(f"\n[{name}]")
    print(f"  平均奖励:   {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  成功率:     {np.mean(success):.3f}")
    if t_ups:
        print(f"  平均T_up:   {np.mean(t_ups):.3f}s ± {np.std(t_ups):.3f}s")
        print(f"  最小T_up:   {np.min(t_ups):.3f}s")
    return {
        "name": name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(success)),
        "mean_T_up": float(np.mean(t_ups)) if t_ups else -1.0,
        "std_T_up": float(np.std(t_ups)) if t_ups else -1.0,
        "rewards": rewards,
        "t_ups": [x if x != float('inf') else -1.0 for x in [r["T_up"] for r in results]],
    }


def main():
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    env = FLRoutingEnv(**ENV_KWARGS)

    print("=" * 60)
    print(f"基线算法评估  Episodes={N_EPISODES}")
    print("=" * 60)

    summaries = []

    print("\n运行 Random...")
    r_random = run_random(env, N_EPISODES, SEED)
    summaries.append(summarize(r_random, "Random"))

    print("\n运行 ShortestPath...")
    r_sp = run_shortest_path(env, N_EPISODES, SEED)
    summaries.append(summarize(r_sp, "ShortestPath"))

    print("\n运行 LoadAwareDijkstra...")
    r_lad = run_load_aware_dijkstra(env, N_EPISODES, SEED)
    summaries.append(summarize(r_lad, "LoadAwareDijkstra"))

    # 保存
    out = f"{OUTPUT_DIR}/logs/baseline_results.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n结果保存至: {out}")

    # 保存numpy
    np.save(f"{OUTPUT_DIR}/logs/baseline_random_rewards.npy", np.array(summaries[0]["rewards"]))
    np.save(f"{OUTPUT_DIR}/logs/baseline_sp_rewards.npy", np.array(summaries[1]["rewards"]))
    np.save(f"{OUTPUT_DIR}/logs/baseline_lad_rewards.npy", np.array(summaries[2]["rewards"]))
    np.save(f"{OUTPUT_DIR}/logs/baseline_random_tup.npy", np.array(summaries[0]["t_ups"]))
    np.save(f"{OUTPUT_DIR}/logs/baseline_sp_tup.npy", np.array(summaries[1]["t_ups"]))
    np.save(f"{OUTPUT_DIR}/logs/baseline_lad_tup.npy", np.array(summaries[2]["t_ups"]))

    return summaries


if __name__ == "__main__":
    main()
