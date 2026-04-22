"""
绘图脚本：对比A3C与基线算法
用法: python plot_results.py --a3c outputs/logs/a3c_rewards_XXXX.npy
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def smooth(data, window=50):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    arr = np.load(files[-1])
    print(f"  loaded: {files[-1]} ({len(arr)} items)")
    return arr


def plot_reward_curves(a3c_rewards, baselines: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    # A3C
    if a3c_rewards is not None:
        sm = smooth(a3c_rewards, 50)
        x = np.arange(len(sm))
        ax.plot(x, sm, label="A3C (ours)", color="tab:blue", linewidth=2)

    # 基线（水平线：均值）
    colors = {"Random": "tab:gray", "ShortestPath": "tab:orange", "LoadAwareDijkstra": "tab:green"}
    for name, rewards in baselines.items():
        if rewards is not None:
            mean_r = np.mean(rewards)
            ax.axhline(mean_r, linestyle="--", color=colors.get(name, "black"),
                       label=f"{name} (mean={mean_r:.2f})", linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("A3C vs Baselines — Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


def plot_tup_curves(a3c_tup, baselines_tup: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    if a3c_tup is not None:
        valid = a3c_tup[a3c_tup > 0]
        if len(valid) > 0:
            sm = smooth(valid, 50)
            ax.plot(np.arange(len(sm)), sm, label="A3C (ours)", color="tab:blue", linewidth=2)

    colors = {"Random": "tab:gray", "ShortestPath": "tab:orange", "LoadAwareDijkstra": "tab:green"}
    for name, tup in baselines_tup.items():
        if tup is not None:
            valid = tup[tup > 0]
            if len(valid) > 0:
                mean_t = np.mean(valid)
                ax.axhline(mean_t, linestyle="--", color=colors.get(name, "black"),
                           label=f"{name} (mean={mean_t:.2f}s)", linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("T_up (s)")
    ax.set_title("A3C vs Baselines — Upload Delay T_up")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


def plot_success_rate(a3c_results_json: str, save_path: str):
    import json
    if not os.path.exists(a3c_results_json):
        return
    with open(a3c_results_json) as f:
        results = json.load(f)
    success = [r["success"] / 3 for r in results]
    sm = smooth(success, 50)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(sm)), sm, color="tab:blue", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("A3C — Routing Success Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3c_rewards", default=None)
    parser.add_argument("--a3c_tup", default=None)
    parser.add_argument("--a3c_json", default=None)
    args = parser.parse_args()

    print("加载数据...")
    a3c_r = np.load(args.a3c_rewards) if args.a3c_rewards else load_latest("outputs/logs/a3c_rewards_*.npy")
    a3c_t = np.load(args.a3c_tup) if args.a3c_tup else load_latest("outputs/logs/a3c_tup_*.npy")

    baselines_r = {
        "Random":           load_latest("outputs/logs/baseline_random_rewards.npy"),
        "ShortestPath":     load_latest("outputs/logs/baseline_sp_rewards.npy"),
        "LoadAwareDijkstra":load_latest("outputs/logs/baseline_lad_rewards.npy"),
    }
    baselines_t = {
        "Random":           load_latest("outputs/logs/baseline_random_tup.npy"),
        "ShortestPath":     load_latest("outputs/logs/baseline_sp_tup.npy"),
        "LoadAwareDijkstra":load_latest("outputs/logs/baseline_lad_tup.npy"),
    }

    print("\n绘图...")
    plot_reward_curves(a3c_r, baselines_r, f"{OUTPUT_DIR}/reward_comparison.png")
    plot_tup_curves(a3c_t, baselines_t, f"{OUTPUT_DIR}/tup_comparison.png")

    a3c_json = args.a3c_json or (sorted(glob.glob("outputs/logs/a3c_results_*.json")) or [None])[-1]
    if a3c_json:
        plot_success_rate(a3c_json, f"{OUTPUT_DIR}/success_rate.png")

    print("\n全部图表已保存至 outputs/figures/")


if __name__ == "__main__":
    main()
