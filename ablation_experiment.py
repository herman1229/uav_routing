"""
消融实验脚本
验证改进A3C四项创新各自的贡献：
  - A3C-Full:     完整版（本文方法）
  - w/o CompState: 去掉竞争感知状态（其他UAV位置 + 链路活跃流数）
  - w/o CompReward:去掉复合奖励（去掉竞争惩罚r_compete + episode级T_up奖励beta_tup）
  - w/o ConcModel: 去掉并发建模（退化为串行，GBS依次路由，不感知并发竞争）
  - w/o TrainImpv: 去掉训练改进（MC回报 + 无梯度裁剪 + 固定熵）

用法: python ablation_experiment.py [--episodes 1200] [--eval 300] [--uav 3]
"""
import os, sys, json, argparse
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

PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/data", exist_ok=True)
HIDDEN_DIM = 256
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# 场景配置（与paper_experiment.py保持一致）
# ======================================================
def make_topo_cfg(num_gbs):
    num_routers = {1: 3, 3: 4, 5: 5, 7: 6, 10: 8}[num_gbs]
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
# 消融变体配置
# 每个变体只改动一项，其余与Full版相同
# ======================================================
VARIANTS = {
    "A3C-Full": {
        "desc": "完整版（本文方法）",
        "env_override": {},          # 不改环境参数
        "state_mode": "full",        # 完整状态
        "train_mode": "improved",    # 改进训练
    },
    "w/o CompState": {
        "desc": "去掉竞争感知状态",
        "env_override": {},
        "state_mode": "no_comp",     # 状态中去掉其他UAV位置和链路活跃流
        "train_mode": "improved",
    },
    "w/o CompReward": {
        "desc": "去掉竞争惩罚+T_up奖励",
        "env_override": {"r_compete": 0.0, "beta_tup": 0.0},
        "state_mode": "full",
        "train_mode": "improved",
    },
    "w/o ConcModel": {
        "desc": "去掉并发建模（串行路由）",
        "env_override": {},
        "state_mode": "no_concurrent",  # 不感知其他UAV位置
        "train_mode": "improved",
    },
    "w/o TrainImpv": {
        "desc": "去掉训练改进（MC+无裁剪+固定熵）",
        "env_override": {},
        "state_mode": "full",
        "train_mode": "vanilla",     # 原版训练
    },
}

# ======================================================
# 环境包装：支持不同状态模式
# ======================================================
class AblationEnvWrapper:
    """包装ConcurrentFLRoutingEnv，支持消融状态模式"""
    def __init__(self, base_env: ConcurrentFLRoutingEnv, state_mode: str):
        self.env = base_env
        self.state_mode = state_mode
        self.num_gbs = base_env.num_gbs
        self.num_nodes = base_env.num_nodes
        self.server_id = base_env.server_id
        self.topo = base_env.topo
        self.delay_model = base_env.delay_model
        self.action_space_n = base_env.action_space_n

        # 根据状态模式计算obs_dim
        base_dim = (1 + 1 + 1 + 1  # cur_gbs, cur_pos, server_id, slot_norm
                    + self.num_nodes   # node_loads
                    + len(self.topo.edges)  # link_loads
                    + self.num_nodes   # visited_flags
                    + self.num_gbs)    # done_flags

        if state_mode == "full":
            # 完整版：加上其他UAV位置(N) + 链路活跃流(E)
            self.obs_dim = base_dim + self.num_gbs + len(self.topo.edges)
        elif state_mode == "no_comp":
            # 去掉其他UAV位置 + 链路活跃流
            self.obs_dim = base_dim
        elif state_mode == "no_concurrent":
            # 去掉其他UAV位置（但保留链路活跃流）
            self.obs_dim = base_dim + len(self.topo.edges)
        else:
            self.obs_dim = base_dim + self.num_gbs + len(self.topo.edges)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        return self._transform_obs(obs)

    def step(self, action):
        obs, r, done, tr, info = self.env.step(action)
        return self._transform_obs(obs), r, done, tr, info

    def get_valid_actions(self):
        return self.env.get_valid_actions()

    def get_current_gbs(self):
        return self.env.get_current_gbs()

    def get_episode_result(self):
        return self.env.get_episode_result()

    def _obs(self):
        return self._transform_obs(self.env._obs())

    def _transform_obs(self, full_obs):
        """根据消融模式裁剪状态向量"""
        env = self.env
        N = env.num_gbs
        E = len(env.topo.edges)
        n = env.num_nodes

        # full_obs 结构：
        # [0]:cur_gbs [1]:cur_pos [2..N+1]:all_gbs_pos [N+2]:server [N+3]:slot
        # [N+4..N+4+n-1]:node_loads [N+4+n..N+4+n+E-1]:link_loads
        # [N+4+n+E..N+4+n+2E-1]:active_flows [N+4+n+2E..N+4+2n+2E-1]:visited
        # [N+4+2n+2E..N+4+2n+2E+N-1]:done_flags

        if self.state_mode == "full":
            return full_obs

        # 提取各段
        idx = 0
        cur_gbs   = full_obs[idx:idx+1]; idx += 1
        cur_pos   = full_obs[idx:idx+1]; idx += 1
        all_pos   = full_obs[idx:idx+N]; idx += N
        server    = full_obs[idx:idx+1]; idx += 1
        slot      = full_obs[idx:idx+1]; idx += 1
        node_ld   = full_obs[idx:idx+n]; idx += n
        link_ld   = full_obs[idx:idx+E]; idx += E
        act_flows = full_obs[idx:idx+E]; idx += E
        visited   = full_obs[idx:idx+n]; idx += n
        done_fl   = full_obs[idx:idx+N]; idx += N

        if self.state_mode == "no_comp":
            # 去掉 all_pos 和 act_flows
            return np.concatenate([cur_gbs, cur_pos, server, slot,
                                    node_ld, link_ld, visited, done_fl])
        elif self.state_mode == "no_concurrent":
            # 去掉 all_pos（但保留 act_flows）
            return np.concatenate([cur_gbs, cur_pos, server, slot,
                                    node_ld, link_ld, act_flows, visited, done_fl])
        return full_obs


# ======================================================
# 训练函数
# ======================================================
def _update_improved(actor, critic, actor_opt, critic_opt,
                     buf, gamma=0.98, n_step=5, max_grad=0.5, ent_coef=0.01):
    """改进版更新：n-step TD + 梯度裁剪 + Advantage归一化"""
    if not buf['states']: return
    states = torch.FloatTensor(np.array(buf['states']))
    ns_t   = torch.FloatTensor(np.array(buf['next_states']))
    rewards, dones = buf['rewards'], buf['dones']
    T = len(rewards)

    with torch.no_grad():
        nv = critic(ns_t).squeeze(1)
    returns = []
    for t in range(T):
        G = sum((gamma**k) * rewards[t+k] for k in range(min(n_step, T-t))
                if not (k > 0 and dones[t+k-1]))
        if t + n_step < T and not dones[t + min(n_step,T-t) - 1]:
            G += (gamma**min(n_step, T-t)) * nv[t + min(n_step, T-t)].item()
        returns.append(G)
    returns = torch.FloatTensor(returns).view(-1, 1)
    values = critic(states)
    adv = (returns - values).detach()
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    cl = F.mse_loss(values, returns.detach())
    critic_opt.zero_grad(); cl.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad)
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
    aloss = torch.stack(al).mean() - ent_coef * torch.stack(ents).mean()
    actor_opt.zero_grad(); aloss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad)
    actor_opt.step()


def _update_vanilla(actor, critic, actor_opt, critic_opt, buf, gamma=0.98):
    """原版更新：MC回报 + 无梯度裁剪 + 固定熵系数"""
    if not buf['states']: return
    states = torch.FloatTensor(np.array(buf['states']))
    rewards = torch.FloatTensor(buf['rewards'])
    R, returns = 0.0, []
    for r in reversed(rewards.tolist()):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns).view(-1, 1)
    values = critic(states)
    adv = (returns - values).detach()

    cl = F.mse_loss(values, returns.detach())
    critic_opt.zero_grad(); cl.backward(); critic_opt.step()

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
    aloss = torch.stack(al).mean() - 0.01 * torch.stack(ents).mean()
    actor_opt.zero_grad(); aloss.backward(); actor_opt.step()


def train_variant(wrapped_env, n_episodes, train_mode, label):
    actor  = PolicyNet(wrapped_env.obs_dim, HIDDEN_DIM, wrapped_env.action_space_n)
    critic = ValueNet(wrapped_env.obs_dim, HIDDEN_DIM)
    actor_opt  = torch.optim.Adam(actor.parameters(), lr=5e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gamma, n_step, max_grad = 0.98, 5, 0.5
    ent_start, ent_end = 0.05, 0.002
    update_every = 3

    curve_reward, curve_tup = [], []
    for ep in range(1, n_episodes + 1):
        progress = ep / n_episodes
        if progress < 0.7:
            ent_coef = ent_start - (ent_start - ent_end) * (progress / 0.7) * 0.5
        else:
            ent_coef = ent_start * 0.5 * (1 - (progress - 0.7) / 0.3) + ent_end

        state = wrapped_env.reset(seed=None)
        buf = {'states': [], 'actions': [], 'rewards': [],
               'valid_sets': [], 'next_states': [], 'dones': []}
        done = False
        while not done:
            valid = wrapped_env.get_valid_actions()
            if not valid: break
            with torch.no_grad():
                logits = actor(torch.FloatTensor(state))
                dist = torch.distributions.Categorical(logits=logits[valid])
                action = valid[dist.sample().item()]
            ns, r, done, _, _ = wrapped_env.step(action)
            buf['states'].append(state); buf['actions'].append(action)
            buf['rewards'].append(r); buf['valid_sets'].append(valid)
            buf['next_states'].append(ns); buf['dones'].append(float(done))
            state = ns
            if len(buf['states']) >= update_every or done:
                if train_mode == "improved":
                    _update_improved(actor, critic, actor_opt, critic_opt,
                                     buf, gamma, n_step, max_grad, ent_coef)
                else:
                    _update_vanilla(actor, critic, actor_opt, critic_opt, buf, gamma)
                buf = {'states': [], 'actions': [], 'rewards': [],
                       'valid_sets': [], 'next_states': [], 'dones': []}

        res = wrapped_env.get_episode_result()
        curve_reward.append(res['total_reward'])
        t = res['T_up'] if res['T_up'] != float('inf') else None
        curve_tup.append(t)

    actor.eval()
    return actor, curve_reward, curve_tup


def evaluate_variant(wrapped_env, actor, n_eval=200):
    tups, divs = [], []
    for ep in range(n_eval):
        wrapped_env.reset(seed=ep + 2000)
        done = False
        while not done:
            valid = wrapped_env.get_valid_actions()
            if not valid: break
            with torch.no_grad():
                q = actor(torch.FloatTensor(wrapped_env._obs()))
                action = valid[F.softmax(q[valid], dim=0).argmax().item()]
            _, _, done, _, _ = wrapped_env.step(action)
        r = wrapped_env.get_episode_result()
        tups.append(r['T_up'] if r['T_up'] != float('inf') else 999.0)
        divs.append(r['path_diversity'])
    arr = np.array([t for t in tups if t < 900])
    if len(arr) == 0: arr = np.array([999.0])
    return {
        'mean_T_up':    float(np.mean(arr)),
        'median_T_up':  float(np.median(arr)),
        'std_T_up':     float(np.std(arr)),
        'bad_rate':     float(np.mean(arr > 50)),
        'mean_diversity': float(np.mean(divs)),
        'tups': tups,
    }


# ======================================================
# 平滑
# ======================================================
def smooth(data, w=40):
    data = np.array([x if x is not None else np.nan for x in data], dtype=float)
    out = np.full_like(data, np.nan)
    for i in range(len(data)):
        chunk = data[max(0, i-w//2): min(len(data), i+w//2+1)]
        v = chunk[~np.isnan(chunk)]
        if len(v): out[i] = v.mean()
    return out


# ======================================================
# 绘图
# ======================================================
COLORS_ABL = {
    "A3C-Full":       "#e63946",
    "w/o CompState":  "#457b9d",
    "w/o CompReward": "#2a9d8f",
    "w/o ConcModel":  "#f4a261",
    "w/o TrainImpv":  "#9b5de5",
}
LS_ABL = {
    "A3C-Full": "-", "w/o CompState": "--",
    "w/o CompReward": "-.", "w/o ConcModel": ":", "w/o TrainImpv": (0,(3,1,1,1)),
}


def plot_ablation_curves(all_curves, num_gbs, save_dir):
    """训练收敛曲线对比（奖励 + T_up）"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Ablation Study — Training Curves ({num_gbs} UAVs)",
                 fontsize=13, fontweight='bold')

    for ax, key, ylabel, title in zip(
        axes,
        ['rewards', 'tups'],
        ['Episode Reward', 'Upload Delay T_up (s)'],
        ['(a) Episode Reward Convergence', '(b) T_up During Training']
    ):
        for name, curves in all_curves.items():
            data = curves[key]
            sm = smooth(data, 50)
            valid = ~np.isnan(sm)
            ax.plot(np.arange(len(sm))[valid], sm[valid],
                    label=name, color=COLORS_ABL[name],
                    linestyle=LS_ABL[name], linewidth=2)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/ablation_curves_{num_gbs}uav.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_ablation_bar(results, num_gbs, save_dir):
    """消融实验结果柱状图（均值/中位数/坏场景率）"""
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Ablation Study — Evaluation Results ({num_gbs} UAVs)",
                 fontsize=13, fontweight='bold')

    metrics = [
        ('mean_T_up',   'Mean T_up (s)',    '(a) Mean T_up'),
        ('median_T_up', 'Median T_up (s)',  '(b) Median T_up'),
        ('bad_rate',    'Bad Rate (T_up>50s)', '(c) Bad Scenario Rate'),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        vals = [results[n][metric] * (100 if metric == 'bad_rate' else 1)
                for n in names]
        cols = [COLORS_ABL[n] for n in names]
        bars = ax.bar(range(len(names)), vals, color=cols, alpha=0.85, width=0.6)
        # 标注数值
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f'{v:.1f}{"%" if metric=="bad_rate" else "s"}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        # 标注Full版基准线
        full_val = results["A3C-Full"][metric] * (100 if metric == 'bad_rate' else 1)
        ax.axhline(full_val, color='red', linestyle='--', linewidth=1.2,
                   alpha=0.6, label='A3C-Full baseline')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("w/o ", "w/o\n") for n in names],
                           fontsize=8, rotation=0)
        ax.set_ylabel(ylabel + ('%' if metric == 'bad_rate' else ''), fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{save_dir}/ablation_bar_{num_gbs}uav.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


def plot_ablation_contribution(results, num_gbs, save_dir):
    """各改进项对T_up均值的贡献量（相对Full版的退化幅度）"""
    full_mean = results["A3C-Full"]["mean_T_up"]
    full_bad  = results["A3C-Full"]["bad_rate"] * 100
    names_ablated = [n for n in results if n != "A3C-Full"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Contribution of Each Improvement ({num_gbs} UAVs)\n"
                 f"(Degradation when component is removed — higher = more important)",
                 fontsize=11, fontweight='bold')

    for ax, metric, ylabel, full_val, title in zip(
        axes,
        ['mean_T_up', 'bad_rate'],
        ['T_up Degradation (s)', 'Bad Rate Increase (pp)'],
        [full_mean, full_bad],
        ['(a) Mean T_up Degradation', '(b) Bad Rate Degradation']
    ):
        vals = []
        for n in names_ablated:
            v = results[n][metric] * (100 if metric == 'bad_rate' else 1)
            vals.append(v - full_val)
        cols = [COLORS_ABL[n] for n in names_ablated]
        bars = ax.bar(range(len(names_ablated)), vals, color=cols, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01 if v >= 0 else bar.get_height() - max(vals)*0.05,
                    f'+{v:.1f}' if v >= 0 else f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(range(len(names_ablated)))
        ax.set_xticklabels([n.replace("w/o ", "") for n in names_ablated],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/ablation_contribution_{num_gbs}uav.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


# ======================================================
# 主流程
# ======================================================
def run_ablation(num_gbs=3, n_train=1200, n_eval=200, max_attempts=4):
    print(f"\n{'='*60}")
    print(f"  消融实验 | {num_gbs}-UAV | 训练{n_train}ep | 评估{n_eval}ep")
    print(f"{'='*60}")

    topo_cfg = make_topo_cfg(num_gbs)
    base_env_kwargs = {**ENV_BASE}

    all_results = {}
    all_curves  = {}

    for vname, vcfg in VARIANTS.items():
        print(f"\n  [{vname}] {vcfg['desc']}")

        # 构建环境（可能覆盖部分奖励参数）
        env_kwargs = {**base_env_kwargs, **vcfg['env_override']}
        base_env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **env_kwargs)
        wrapped  = AblationEnvWrapper(base_env, vcfg['state_mode'])
        print(f"    obs_dim={wrapped.obs_dim}")

        # 多次训练取最优
        best_actor, best_reward, best_tup = None, None, float('inf')
        for attempt in range(1, max_attempts + 1):
            actor, curve_r, curve_t = train_variant(
                wrapped, n_train, vcfg['train_mode'], f"{vname}-a{attempt}")
            quick = evaluate_variant(wrapped, actor, 50)
            print(f"    attempt {attempt}: median={quick['median_T_up']:.2f}s, "
                  f"bad={quick['bad_rate']*100:.0f}%")
            if quick['median_T_up'] < best_tup:
                best_tup   = quick['median_T_up']
                best_actor = actor
                best_reward = curve_r
                best_tup_curve = curve_t
            threshold = 5.0 if num_gbs == 1 else 20.0
            if quick['median_T_up'] < threshold:
                print(f"    ✅ 收敛 (median={quick['median_T_up']:.2f}s)")
                break

        # 正式评估
        res = evaluate_variant(wrapped, best_actor, n_eval)
        all_results[vname] = res
        all_curves[vname]  = {'rewards': best_reward, 'tups': best_tup_curve}
        print(f"    → mean={res['mean_T_up']:.2f}s median={res['median_T_up']:.2f}s "
              f"bad={res['bad_rate']*100:.0f}% div={res['mean_diversity']:.2f}")

    # 绘图
    print(f"\n  绘制消融实验图表...")
    plot_ablation_curves(all_curves, num_gbs, f"{PAPER_DIR}/figures")
    plot_ablation_bar(all_results, num_gbs, f"{PAPER_DIR}/figures")
    plot_ablation_contribution(all_results, num_gbs, f"{PAPER_DIR}/figures")

    # 保存数据
    safe = {n: {k: v for k, v in r.items() if k != 'tups'}
            for n, r in all_results.items()}
    path = f"{PAPER_DIR}/data/ablation_{num_gbs}uav_{TIMESTAMP}.json"
    with open(path, 'w') as f:
        json.dump(safe, f, indent=2)

    # 打印汇总
    print(f"\n  {'='*55}")
    print(f"  消融实验结果汇总（{num_gbs}-UAV）")
    print(f"  {'变体':<18} {'均值':>8} {'中位数':>8} {'坏场景率':>10} {'多样性':>8}")
    print(f"  {'-'*55}")
    for n, r in all_results.items():
        marker = " ◀ Full" if n == "A3C-Full" else ""
        print(f"  {n:<18} {r['mean_T_up']:>8.2f}s {r['median_T_up']:>7.2f}s "
              f"{r['bad_rate']*100:>9.1f}% {r['mean_diversity']:>7.2f}{marker}")
    print(f"  {'='*55}")
    print(f"  数据: {path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav",      type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--eval",     type=int, default=200)
    args = parser.parse_args()
    run_ablation(args.uav, args.episodes, args.eval)
