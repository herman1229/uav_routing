"""
稳定收敛实验结果绘图脚本
基于 paper_results/data/stable_results_*.json 生成完整图表集
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)

# 配色（与其他实验保持一致）
COLORS = {
    "A3C":          "#e63946",
    "DQN":          "#457b9d",
    "DQN-PER":      "#9b5de5",
    "LAD-Dijkstra": "#2a9d8f",
    "ShortestPath": "#f4a261",
}
MARKERS = {"A3C": "o", "DQN": "D", "DQN-PER": "s",
           "LAD-Dijkstra": "^", "ShortestPath": "x"}
LS = {"A3C": "-", "DQN": "--", "DQN-PER": "-.", "LAD-Dijkstra": ":", "ShortestPath": ":"}


def load_latest():
    files = sorted(glob.glob(f"{PAPER_DIR}/data/stable_results_*.json"))
    with open(files[-1]) as f:
        return json.load(f)


def plot_convergence_rate_combined(data):
    """图1：3-UAV 和 5-UAV 收敛率并排柱状图"""
    uav_list = [3, 5]
    rl_algs  = ["A3C", "DQN", "DQN-PER"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Convergence Rate: 5 Independent Runs per Algorithm",
                 fontsize=13, fontweight='bold')

    for ax, n in zip(axes, uav_list):
        stats = data['stats'][str(n)]
        rates = [stats[a]['conv_rate'] * 100 for a in rl_algs]
        cols  = [COLORS[a] for a in rl_algs]
        bars  = ax.bar(rl_algs, rates, color=cols, alpha=0.85, width=0.5)
        for bar, v in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.5,
                    f'{v:.0f}%', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.set_ylabel("Convergence Rate (%)", fontsize=11)
        ax.set_title(f"{'(a)' if n==3 else '(b)'} {n} UAVs "
                     f"(threshold: median T_up < 20s)", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        # 标注 n_runs
        n_runs = stats[rl_algs[0]]['n_runs']
        ax.text(0.97, 0.97, f'{n_runs} runs each',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='gray')

    plt.tight_layout()
    p = f"{PAPER_DIR}/figures/stable_fig1_convergence_rate.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_best_model_comparison(data):
    """图2：最优模型评估结果（4指标，两场景并排）"""
    uav_list = [3, 5]
    eval_algs = ["A3C", "DQN", "DQN-PER", "LAD-Dijkstra"]
    metrics = [
        ('mean_T_up',   'Mean T_up (s)',    1,   '(a) Mean T_up'),
        ('median_T_up', 'Median T_up (s)',  1,   '(b) Median T_up'),
        ('bad_rate',    'Bad Rate (%)',      100, '(c) Bad Scenario Rate'),
        ('mean_diversity','Path Diversity',  1,   '(d) Path Diversity'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Best Model Evaluation Results — A3C / DQN / DQN-PER / LAD",
                 fontsize=13, fontweight='bold')

    for row, n in enumerate(uav_list):
        best = data['best_eval'][str(n)]
        for col, (metric, ylabel, scale, title) in enumerate(metrics):
            ax = axes[row][col]
            vals = [best[a][metric] * scale for a in eval_algs]
            cols = [COLORS.get(a, "#888") for a in eval_algs]
            bars = ax.bar(range(len(eval_algs)), vals, color=cols, alpha=0.85, width=0.6)

            # 标注最优值
            best_val = min(vals) if metric != 'mean_diversity' else max(vals)
            for bar, v, a in zip(bars, vals, eval_algs):
                is_best = (v == best_val)
                fmt = f'{v:.1f}{"%" if metric=="bad_rate" else "s" if "T_up" in metric else ""}'
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.015,
                        fmt, ha='center', va='bottom',
                        fontsize=7.5,
                        fontweight='bold' if is_best else 'normal',
                        color='#e63946' if is_best else 'black')

            ax.set_xticks(range(len(eval_algs)))
            ax.set_xticklabels([a.replace('-', '\n') for a in eval_algs], fontsize=7.5)
            ax.set_ylabel(ylabel + ('%' if metric=='bad_rate' else ''), fontsize=9)
            ax.set_title(f"{n}-UAV {title}", fontsize=9)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = f"{PAPER_DIR}/figures/stable_fig2_best_eval.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_stability_vs_performance(data):
    """图3：稳定性 vs 性能散点图（核心结论图）"""
    uav_list = [3, 5]
    rl_algs  = ["A3C", "DQN", "DQN-PER"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stability vs Performance Trade-off\n"
                 "X: Convergence Rate (higher=more stable)  "
                 "Y: Median T_up when converged (lower=better)",
                 fontsize=12, fontweight='bold')

    for ax, n in zip(axes, uav_list):
        stats = data['stats'][str(n)]
        best  = data['best_eval'][str(n)]

        for alg in rl_algs:
            conv_rate = stats[alg]['conv_rate'] * 100
            # 收敛时的median（只用收敛的run的均值）
            conv_meds = stats[alg]['conv_medians']
            med = np.mean(conv_meds) if conv_meds else 999.0

            ax.scatter(conv_rate, med,
                       color=COLORS[alg], marker=MARKERS[alg],
                       s=200, zorder=5, label=alg)
            ax.annotate(f"  {alg}\n  ({conv_rate:.0f}%, {med:.1f}s)",
                        (conv_rate, med),
                        fontsize=8, color=COLORS[alg])

        # LAD 参考线（不是RL，不参与收敛率比较，用虚线标注性能基准）
        lad_med = best['LAD-Dijkstra']['median_T_up']
        ax.axhline(lad_med, color=COLORS['LAD-Dijkstra'],
                   linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'LAD-Dijkstra (median={lad_med:.1f}s)')

        ax.set_xlabel("Convergence Rate (%)", fontsize=11)
        ax.set_ylabel("Median T_up when converged (s)", fontsize=11)
        ax.set_title(f"{'(a)' if n==3 else '(b)'} {n} UAVs", fontsize=11)
        ax.set_xlim(0, 110)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # 标注"理想区域"
        ax.annotate('← Better stability\nLower T_up ↓',
                    xy=(0.98, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8, color='gray',
                    style='italic')

    plt.tight_layout()
    p = f"{PAPER_DIR}/figures/stable_fig3_stability_vs_perf.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_convergence_trend(data):
    """图4：收敛率趋势（3-UAV vs 5-UAV，展示规模对稳定性的影响）"""
    rl_algs = ["A3C", "DQN", "DQN-PER"]
    uav_list = [3, 5]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(uav_list))
    w = 0.25

    for i, alg in enumerate(rl_algs):
        rates = [data['stats'][str(n)][alg]['conv_rate'] * 100 for n in uav_list]
        bars = ax.bar(x + i*w, rates, w, label=alg,
                      color=COLORS[alg], alpha=0.85)
        for bar, v in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{n} UAVs" for n in uav_list], fontsize=12)
    ax.set_ylabel("Convergence Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Convergence Rate vs Number of UAVs\n"
                 "DQN-PER maintains highest stability as scale increases",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 标注降幅
    for i, alg in enumerate(rl_algs):
        r3 = data['stats']['3'][alg]['conv_rate'] * 100
        r5 = data['stats']['5'][alg]['conv_rate'] * 100
        drop = r3 - r5
        if drop > 0:
            ax.annotate(f'↓{drop:.0f}pp',
                        xy=(x[1] + i*w + w/2, r5 + 3),
                        ha='center', fontsize=8, color=COLORS[alg], alpha=0.8)

    plt.tight_layout()
    p = f"{PAPER_DIR}/figures/stable_fig4_conv_trend.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_per_effect(data):
    """图5：PER效果对比（DQN vs DQN-PER，突出PER的贡献）"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Effect of Prioritized Experience Replay (PER)\n"
                 "DQN vs DQN-PER: Stability and Performance",
                 fontsize=12, fontweight='bold')

    uav_list = [3, 5]
    metrics = [
        ('conv_rate_pct', 'Convergence Rate (%)', '(a) Convergence Rate'),
        ('median_T_up',   'Median T_up (s)',       '(b) Median T_up (best run)'),
        ('bad_rate_pct',  'Bad Scenario Rate (%)', '(c) Bad Scenario Rate (best run)'),
    ]

    # 准备数据
    plot_data = {}
    for n in uav_list:
        plot_data[n] = {
            'DQN': {
                'conv_rate_pct': data['stats'][str(n)]['DQN']['conv_rate'] * 100,
                'median_T_up':   data['best_eval'][str(n)]['DQN']['median_T_up'],
                'bad_rate_pct':  data['best_eval'][str(n)]['DQN']['bad_rate'] * 100,
            },
            'DQN-PER': {
                'conv_rate_pct': data['stats'][str(n)]['DQN-PER']['conv_rate'] * 100,
                'median_T_up':   data['best_eval'][str(n)]['DQN-PER']['median_T_up'],
                'bad_rate_pct':  data['best_eval'][str(n)]['DQN-PER']['bad_rate'] * 100,
            }
        }

    x = np.arange(len(uav_list))
    w = 0.35

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for i, alg in enumerate(['DQN', 'DQN-PER']):
            vals = [plot_data[n][alg][metric] for n in uav_list]
            bars = ax.bar(x + i*w, vals, w, label=alg,
                          color=COLORS[alg], alpha=0.85)
            for bar, v in zip(bars, vals):
                unit = '%' if 'pct' in metric or 'rate' in metric else 's'
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.02,
                        f'{v:.1f}{unit}', ha='center', fontsize=8.5, fontweight='bold')

        ax.set_xticks(x + w/2)
        ax.set_xticklabels([f"{n} UAVs" for n in uav_list], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = f"{PAPER_DIR}/figures/stable_fig5_per_effect.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def main():
    print("加载正式实验数据...")
    data = load_latest()
    print(f"  数据: {sorted(glob.glob(PAPER_DIR+'/data/stable_results_*.json'))[-1]}")

    print("\n生成稳定收敛实验图表...")
    plot_convergence_rate_combined(data)
    plot_best_model_comparison(data)
    plot_stability_vs_performance(data)
    plot_convergence_trend(data)
    plot_per_effect(data)

    print(f"\n全部图表已保存至 {PAPER_DIR}/figures/")
    print("文件列表:")
    for f in sorted(glob.glob(f"{PAPER_DIR}/figures/stable_fig*.png")):
        print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
