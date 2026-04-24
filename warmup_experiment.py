"""
预热实验：传统算法行为克隆（BC）预热 + 强化学习精调
验证：用LAD-Dijkstra的决策数据预训练RL网络，是否能加速收敛并提升效果

实验假设：
  - 纯RL从随机初始化出发，大量时间做随机探索，容易陷入局部最优
  - BC预热让网络先学会"基本路由能力"（接近LAD水平）
  - 从好的起点出发，RL更容易找到比LAD更优的策略

实验设计（5次独立运行，统计收敛率）：
  对比组：
    A3C-Pure:     纯A3C（当前基线）
    A3C-BC:       BC预热 + A3C精调
    DQN-PER-Pure: 纯DQN-PER（当前最稳定基线）
    DQN-PER-BC:   BC预热 + DQN-PER精调

  预热方式：
    Actor预热：用LAD轨迹的(s,a)对做行为克隆（交叉熵损失）
    DQN预热：BC初始化Q网络 + 预填充ReplayBuffer

用法: python warmup_experiment.py [--episodes 1500] [--eval 300] [--runs 5]
"""
import os, sys, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import PolicyNet, ValueNet
from src.baselines.dqn_per import DQNAgentPER, QNetwork

PAPER_DIR = "paper_results"
os.makedirs(f"{PAPER_DIR}/figures", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/data", exist_ok=True)
os.makedirs(f"{PAPER_DIR}/models", exist_ok=True)
HIDDEN_DIM = 256
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
CONV_THRESHOLD = {3: 20.0, 5: 20.0}

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
# LAD-Dijkstra 策略
# ======================================================
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
# 阶段1：行为克隆数据收集
# ======================================================
def collect_bc_data(env: ConcurrentFLRoutingEnv,
                    n_episodes: int = 500,
                    seed_offset: int = 5000) -> Dict:
    """
    用LAD-Dijkstra跑n_episodes个episode，收集行为克隆数据
    返回：
      states:       网络观测
      actions_global: LAD选择的全局动作编号（用于cross_entropy目标）
      valid_masks:  有效动作mask（用于屏蔽无效动作的logit）
    """
    states, actions_global, valid_masks = [], [], []
    rewards_list = []

    for ep in range(n_episodes):
        env.reset(seed=ep + seed_offset)
        done = False
        ep_r = 0.0
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            state = env._obs()
            action = lad_policy(env, valid)   # 全局节点编号

            mask = np.zeros(env.action_space_n, dtype=np.float32)
            for a in valid:
                mask[a] = 1.0

            _, r, done, _, _ = env.step(action)
            states.append(state)
            actions_global.append(action)     # 存全局编号
            valid_masks.append(mask)
            ep_r += r
        rewards_list.append(ep_r)

    print(f"    收集BC数据: {len(states)}步, {n_episodes}个episode, "
          f"平均奖励={np.mean(rewards_list):.2f}")
    return {
        'states':         np.array(states,         dtype=np.float32),
        'actions_global': np.array(actions_global, dtype=np.int64),
        'valid_masks':    np.array(valid_masks,     dtype=np.float32),
    }

# ======================================================
# 阶段2a：Actor 行为克隆预热（用于A3C）
# ======================================================
def bc_pretrain_actor(actor: PolicyNet,
                      bc_data: Dict,
                      bc_epochs: int = 15,
                      bc_lr: float = 1e-3,
                      batch_size: int = 256) -> List[float]:
    """
    用行为克隆损失预训练Actor网络
    目标：minimize CrossEntropy(π(s), a_LAD)
    actions_global: LAD选择的全局节点编号（cross_entropy的目标）
    """
    opt = torch.optim.Adam(actor.parameters(), lr=bc_lr)
    states         = torch.FloatTensor(bc_data['states'])
    actions_global = torch.LongTensor(bc_data['actions_global'])
    valid_masks    = torch.FloatTensor(bc_data['valid_masks'])
    N = len(states)
    losses = []

    for epoch in range(bc_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            s = states[idx]
            a = actions_global[idx]   # 全局编号
            m = valid_masks[idx]

            logits = actor(s)
            masked_logits = logits + (1.0 - m) * (-1e9)
            loss = F.cross_entropy(masked_logits, a)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

    with torch.no_grad():
        logits = actor(states)
        masked = logits + (1.0 - valid_masks) * (-1e9)
        pred = masked.argmax(dim=1)
        accuracy = (pred == actions_global).float().mean().item()
    print(f"    BC预热完成: 最终loss={losses[-1]:.4f}, "
          f"准确率≈{accuracy*100:.1f}%")
    return losses

# ======================================================
# 阶段2b：Q网络行为克隆预热（用于DQN-PER）
# ======================================================
def bc_pretrain_qnet(q_net: QNetwork,
                     bc_data: Dict,
                     bc_epochs: int = 15,
                     bc_lr: float = 1e-3,
                     batch_size: int = 256) -> List[float]:
    """
    用行为克隆损失预训练Q网络
    目标：让Q(s, a_LAD) 在有效动作中最大
    """
    opt = torch.optim.Adam(q_net.parameters(), lr=bc_lr)
    states         = torch.FloatTensor(bc_data['states'])
    actions_global = torch.LongTensor(bc_data['actions_global'])
    valid_masks    = torch.FloatTensor(bc_data['valid_masks'])
    N = len(states)
    losses = []

    for epoch in range(bc_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            s = states[idx]
            a = actions_global[idx]
            m = valid_masks[idx]

            q_vals = q_net(s)
            masked_q = q_vals + (1.0 - m) * (-1e9)
            loss = F.cross_entropy(masked_q, a)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

    with torch.no_grad():
        q_vals = q_net(states)
        masked = q_vals + (1.0 - valid_masks) * (-1e9)
        pred = masked.argmax(dim=1)
        accuracy = (pred == actions_global).float().mean().item()
    print(f"    BC预热完成: 最终loss={losses[-1]:.4f}, "
          f"准确率≈{accuracy*100:.1f}%")
    return losses

# ======================================================
# RL训练函数（A3C / DQN-PER，支持BC预热初始化）
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
        G, steps = 0.0, min(n_step, T-t)
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


def train_a3c_with_warmup(env, n_episodes, bc_data=None, seed=None,
                          bc_epochs=15) -> Tuple:
    """A3C训练，可选BC预热"""
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    actor  = PolicyNet(env.obs_dim, HIDDEN_DIM, env.action_space_n)
    critic = ValueNet(env.obs_dim, HIDDEN_DIM)

    bc_losses = []
    if bc_data is not None:
        bc_losses = bc_pretrain_actor(actor, bc_data, bc_epochs=bc_epochs)

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
    return actor, {'rewards': curve_r, 'tups': curve_t, 'bc_losses': bc_losses}


def train_dqnper_with_warmup(env, n_episodes, bc_data=None, seed=None,
                              bc_epochs=15) -> Tuple:
    """DQN-PER训练，可选BC预热"""
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    agent = DQNAgentPER(
        state_dim=env.obs_dim, hidden_dim=HIDDEN_DIM,
        action_dim=env.action_space_n,
        lr=5e-4, gamma=0.98, buffer_size=20000, batch_size=128,
        eps_start=1.0, eps_end=0.05, eps_decay=n_episodes//2,
        target_update=10, alpha=0.6, beta_start=0.4, beta_end=1.0,
    )

    bc_losses = []
    if bc_data is not None:
        # BC预热Q网络
        bc_losses = bc_pretrain_qnet(agent.q_net, bc_data, bc_epochs=bc_epochs)
        agent.target_net.load_state_dict(agent.q_net.state_dict())

        # 预填充ReplayBuffer（用LAD轨迹数据）
        # 重新收集带奖励的完整轨迹
        print(f"    预填充ReplayBuffer...")
        pre_episodes = min(200, n_episodes // 5)
        for ep in range(pre_episodes):
            env.reset(seed=ep + 9000)
            done = False
            state = env._obs()
            while not done:
                valid = env.get_valid_actions()
                if not valid: break
                action = lad_policy(env, valid)
                ns, r, done, _, _ = env.step(action)
                agent.store(state, action, r, env._obs(), float(done))
                state = env._obs()
        print(f"    Buffer预填充: {len(agent.buffer)}条经验")

        # BC预热后降低初始epsilon（已有好的初始策略，不需要大量随机探索）
        agent.eps_start = 0.3
        agent.steps = 0  # 重置step计数

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

    return agent, {'rewards': curve_r, 'tups': curve_t, 'bc_losses': bc_losses}

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
        'mean_T_up':    float(np.mean(arr)),
        'median_T_up':  float(np.median(arr)),
        'std_T_up':     float(np.std(arr)),
        'bad_rate':     float(np.mean(arr > 50)),
        'mean_diversity': float(np.mean(divs)),
        'tups': tups,
    }

def quick_eval(env, policy_fn, n=60):
    tups = []
    for t in range(n):
        env.reset(seed=3000+t)
        done = False
        while not done:
            v = env.get_valid_actions()
            if not v: break
            a = policy_fn(env, v)
            _, _, done, _, _ = env.step(a)
        r = env.get_episode_result()
        tups.append(r['T_up'] if r['T_up'] != float('inf') else 999)
    arr = np.array([t for t in tups if t < 900])
    return float(np.median(arr)) if len(arr) else 999.0

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
    "A3C-Pure":       "#e63946",
    "A3C-BC":         "#ff9f1c",
    "DQN-PER-Pure":   "#457b9d",
    "DQN-PER-BC":     "#9b5de5",
    "LAD-Dijkstra":   "#2a9d8f",
}
LS = {
    "A3C-Pure": "-", "A3C-BC": "--",
    "DQN-PER-Pure": "-.", "DQN-PER-BC": (0,(3,1,1,1)),
    "LAD-Dijkstra": ":"
}


def plot_convergence_rate(conv_stats, num_gbs, save_dir):
    """收敛率对比柱状图"""
    algs  = [a for a in conv_stats if a != 'LAD-Dijkstra']
    rates = [conv_stats[a]['conv_rate'] * 100 for a in algs]
    cols  = [COLORS[a] for a in algs]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(algs, rates, color=cols, alpha=0.85, width=0.5)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # 标注BC预热带来的提升
    pairs = [("A3C-Pure", "A3C-BC"), ("DQN-PER-Pure", "DQN-PER-BC")]
    for pure, bc in pairs:
        if pure in conv_stats and bc in conv_stats:
            r_pure = conv_stats[pure]['conv_rate'] * 100
            r_bc   = conv_stats[bc]['conv_rate'] * 100
            delta  = r_bc - r_pure
            if delta != 0:
                x_pure = algs.index(pure)
                x_bc   = algs.index(bc)
                y = max(r_pure, r_bc) + 8
                ax.annotate('', xy=(x_bc, y-3), xytext=(x_pure, y-3),
                            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
                ax.text((x_pure+x_bc)/2, y,
                        f'{delta:+.0f}pp', ha='center', fontsize=9,
                        color='green' if delta > 0 else 'red', fontweight='bold')

    ax.set_ylim(0, 120)
    ax.set_ylabel("Convergence Rate (%)", fontsize=12)
    ax.set_title(f"Convergence Rate: BC Warmup Effect ({num_gbs} UAVs, 5 runs)\n"
                 f"Threshold: median T_up < {CONV_THRESHOLD[num_gbs]}s",
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/warmup_fig1_conv_rate_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_training_curves_all(all_curves, num_gbs, save_dir):
    """各算法多次运行的T_up训练曲线"""
    algs = list(all_curves.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Convergence: BC Warmup vs Pure RL ({num_gbs} UAVs)",
                 fontsize=12, fontweight='bold')

    for ax, key, ylabel, title in zip(
        axes, ['rewards', 'tups'],
        ['Episode Reward', 'Upload Delay T_up (s)'],
        ['(a) Episode Reward', '(b) T_up During Training']
    ):
        for alg in algs:
            curves = all_curves[alg]
            for i, curve in enumerate(curves):
                sm = smooth(curve[key], 50)
                valid = ~np.isnan(sm)
                alpha = 0.85 if i == 0 else 0.3
                lw    = 2.0  if i == 0 else 0.8
                ax.plot(np.arange(len(sm))[valid], sm[valid],
                        color=COLORS[alg], alpha=alpha, linewidth=lw,
                        linestyle=LS[alg],
                        label=alg if i == 0 else None)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/warmup_fig2_training_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_eval_comparison(best_results, num_gbs, save_dir):
    """最优模型评估结果对比"""
    algs = list(best_results.keys())
    metrics = [
        ('mean_T_up',   'Mean T_up (s)',    1,   '(a) Mean T_up'),
        ('median_T_up', 'Median T_up (s)',  1,   '(b) Median T_up'),
        ('bad_rate',    'Bad Rate (%)',      100, '(c) Bad Scenario Rate'),
        ('mean_diversity','Path Diversity', 1,   '(d) Path Diversity'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"Best Model Evaluation — BC Warmup Effect ({num_gbs} UAVs)",
                 fontsize=12, fontweight='bold')

    for ax, (metric, ylabel, scale, title) in zip(axes, metrics):
        vals = [best_results[a][metric] * scale for a in algs]
        cols = [COLORS.get(a, "#888") for a in algs]
        bars = ax.bar(range(len(algs)), vals, color=cols, alpha=0.85, width=0.6)
        best_v = min(vals) if metric != 'mean_diversity' else max(vals)
        for bar, v, a in zip(bars, vals, algs):
            is_best = (v == best_v)
            fmt = f'{v:.1f}{"%" if metric=="bad_rate" else "s" if "T_up" in metric else ""}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.015,
                    fmt, ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold' if is_best else 'normal',
                    color='#e63946' if is_best else 'black')
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels([a.replace('-', '\n') for a in algs], fontsize=7)
        ax.set_ylabel(ylabel + ('%' if metric=='bad_rate' else ''), fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/warmup_fig3_eval_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_bc_loss(bc_loss_data, num_gbs, save_dir):
    """BC预热损失曲线"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for alg, losses in bc_loss_data.items():
        if losses:
            ax.plot(range(1, len(losses)+1), losses,
                    color=COLORS[alg], linewidth=2, label=alg, marker='o', markersize=4)
    ax.set_xlabel("BC Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_title(f"Behavioral Cloning Pretraining Loss ({num_gbs} UAVs)\n"
                 f"Lower = Better imitation of LAD-Dijkstra",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/warmup_fig4_bc_loss_{num_gbs}uav.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


def plot_summary(all_stats, uav_list, save_dir):
    """汇总图：BC预热效果 vs UAV规模"""
    pairs = [("A3C-Pure", "A3C-BC"), ("DQN-PER-Pure", "DQN-PER-BC")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BC Warmup Effect Summary: Convergence Rate & Median T_up",
                 fontsize=12, fontweight='bold')

    for ax, metric, ylabel, title in zip(
        axes,
        ['conv_rate', 'conv_median'],
        ['Convergence Rate (%)', 'Median T_up when converged (s)'],
        ['(a) Convergence Rate', '(b) Median T_up (converged runs)']
    ):
        x = np.arange(len(uav_list))
        w = 0.2
        for i, (pure, bc) in enumerate(pairs):
            pure_vals, bc_vals = [], []
            for n in uav_list:
                stats = all_stats[n]
                if metric == 'conv_rate':
                    pure_vals.append(stats[pure]['conv_rate'] * 100)
                    bc_vals.append(stats[bc]['conv_rate'] * 100)
                else:
                    pm = stats[pure]['conv_medians']
                    bm = stats[bc]['conv_medians']
                    pure_vals.append(np.mean(pm) if pm else 999.0)
                    bc_vals.append(np.mean(bm) if bm else 999.0)

            offset = (i - 0.5) * w * 2
            ax.bar(x + offset,       pure_vals, w, label=pure, color=COLORS[pure], alpha=0.85)
            ax.bar(x + offset + w,   bc_vals,   w, label=bc,   color=COLORS[bc],   alpha=0.85)

        ax.set_xticks(x + w/2)
        ax.set_xticklabels([f"{n} UAVs" for n in uav_list], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = f"{save_dir}/warmup_fig5_summary.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {p}")


# ======================================================
# 主流程
# ======================================================
def run_warmup_test(num_gbs, n_episodes, n_eval, n_runs,
                    bc_episodes=500, bc_epochs=15):
    topo_cfg = make_topo_cfg(num_gbs)
    env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)
    threshold = CONV_THRESHOLD[num_gbs]

    print(f"\n{'='*60}")
    print(f"  {num_gbs}-UAV | {n_runs}次运行 | {n_episodes}ep/run")
    print(f"  BC预热: {bc_episodes}ep数据, {bc_epochs}个epoch")
    print(f"  obs_dim={env.obs_dim}, 收敛阈值<{threshold}s")
    print(f"{'='*60}")

    # 收集BC数据（只需收集一次，所有runs共用）
    print(f"\n  [收集LAD行为克隆数据]")
    bc_data = collect_bc_data(env, n_episodes=bc_episodes)

    # 定义实验组
    experiments = {
        "A3C-Pure":     (train_a3c_with_warmup,    None),
        "A3C-BC":       (train_a3c_with_warmup,    bc_data),
        "DQN-PER-Pure": (train_dqnper_with_warmup, None),
        "DQN-PER-BC":   (train_dqnper_with_warmup, bc_data),
    }

    stats       = {}
    best_models = {}
    all_curves  = {k: [] for k in experiments}
    bc_loss_data= {}

    for alg, (train_fn, bc_d) in experiments.items():
        warmup_tag = "（BC预热）" if bc_d is not None else "（纯RL）"
        print(f"\n  [{alg}]{warmup_tag} {n_runs}次独立训练...")
        conv_count, conv_medians = 0, []
        best_med, best_model = float('inf'), None
        run_bc_losses = []

        for run in range(1, n_runs + 1):
            seed = 42 + run * 100 + num_gbs * 10
            model, curve = train_fn(env, n_episodes, bc_data=bc_d,
                                    seed=seed, bc_epochs=bc_epochs)

            # 记录BC损失（只记第一次）
            if run == 1 and curve.get('bc_losses'):
                run_bc_losses = curve['bc_losses']

            # 快速评估
            if alg.startswith("A3C"):
                pf = lambda e, v, m=model: (
                    v[F.softmax(m(torch.FloatTensor(e._obs()))[v], dim=0).argmax().item()])
            else:
                pf = lambda e, v, m=model: (
                    v[m.q_net(torch.FloatTensor(e._obs()))[v].argmax().item()])

            med = quick_eval(env, pf)
            converged = med < threshold
            if converged:
                conv_count += 1; conv_medians.append(med)
            all_curves[alg].append(curve)
            if med < best_med:
                best_med = med; best_model = model
            print(f"    run {run}: median={med:.2f}s {'✅' if converged else '❌'}")

        stats[alg] = {
            'conv_rate':    conv_count / n_runs,
            'conv_count':   conv_count,
            'n_runs':       n_runs,
            'conv_medians': conv_medians,
            'mean_conv_med': float(np.mean(conv_medians)) if conv_medians else 999.0,
        }
        best_models[alg] = best_model
        bc_loss_data[alg] = run_bc_losses
        print(f"  → 收敛率: {conv_count}/{n_runs}={conv_count/n_runs*100:.0f}%  "
              f"收敛时median均值: {stats[alg]['mean_conv_med']:.2f}s")

    # 绘制图表
    print(f"\n  绘制图表...")
    plot_convergence_rate(stats, num_gbs, f"{PAPER_DIR}/figures")
    plot_training_curves_all(all_curves, num_gbs, f"{PAPER_DIR}/figures")
    plot_bc_loss({k: v for k, v in bc_loss_data.items() if v},
                 num_gbs, f"{PAPER_DIR}/figures")

    # 正式评估
    print(f"\n  [正式评估 {n_eval}ep（最优模型）]")
    best_results = {}
    for alg, model in best_models.items():
        if alg.startswith("A3C"):
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

    return stats, best_results, all_curves, bc_loss_data


def main(n_episodes=1500, n_eval=300, n_runs=5,
         bc_episodes=500, bc_epochs=15):
    uav_list = [3, 5]
    train_eps = {3: n_episodes, 5: max(n_episodes, 2000)}

    print(f"\n{'#'*60}")
    print(f"  BC预热实验：传统算法预热 + RL精调")
    print(f"  {n_runs}次独立运行 | BC数据{bc_episodes}ep | BC训练{bc_epochs}epoch")
    print(f"{'#'*60}")

    all_stats   = {}
    all_results = {}
    all_curves_by_uav = {}
    all_bc_losses = {}

    for num_gbs in uav_list:
        stats, best_results, all_curves, bc_losses = run_warmup_test(
            num_gbs, train_eps[num_gbs], n_eval, n_runs,
            bc_episodes, bc_epochs)
        all_stats[num_gbs]          = stats
        all_results[num_gbs]        = best_results
        all_curves_by_uav[num_gbs]  = all_curves
        all_bc_losses[num_gbs]      = bc_losses

    # 汇总图
    print(f"\n  绘制汇总图...")
    plot_summary(all_stats, uav_list, f"{PAPER_DIR}/figures")

    # 保存数据
    safe = {
        'stats': {str(n): {a: {k: v for k, v in s.items()}
                            for a, s in sc.items()}
                  for n, sc in all_stats.items()},
        'best_eval': {str(n): {a: {k: v for k, v in r.items() if k != 'tups'}
                                for a, r in res.items()}
                      for n, res in all_results.items()},
    }
    path = f"{PAPER_DIR}/data/warmup_results_{TIMESTAMP}.json"
    with open(path, 'w') as f:
        json.dump(safe, f, indent=2)

    # 汇总打印
    print(f"\n{'='*70}")
    print("  BC预热效果汇总：收敛率")
    print(f"  {'UAV':>5} | {'A3C-Pure':>10} | {'A3C-BC':>10} | "
          f"{'DQN-PER-Pure':>14} | {'DQN-PER-BC':>12}")
    print(f"  {'-'*60}")
    for n in uav_list:
        row = [f"{all_stats[n][a]['conv_rate']*100:.0f}%"
               for a in ["A3C-Pure","A3C-BC","DQN-PER-Pure","DQN-PER-BC"]]
        print(f"  {n:>5} | {row[0]:>10} | {row[1]:>10} | {row[2]:>14} | {row[3]:>12}")

    print(f"\n  BC预热效果汇总：最优模型 median T_up (s)")
    print(f"  {'UAV':>5} | {'A3C-Pure':>10} | {'A3C-BC':>10} | "
          f"{'DQN-PER-Pure':>14} | {'DQN-PER-BC':>12} | {'LAD':>8}")
    print(f"  {'-'*70}")
    for n in uav_list:
        algs = ["A3C-Pure","A3C-BC","DQN-PER-Pure","DQN-PER-BC","LAD-Dijkstra"]
        row = [f"{all_results[n][a]['median_T_up']:.2f}s" for a in algs]
        print(f"  {n:>5} | {row[0]:>10} | {row[1]:>10} | {row[2]:>14} | "
              f"{row[3]:>12} | {row[4]:>8}")

    print(f"\n  数据: {path}")
    print(f"  图表: {PAPER_DIR}/figures/warmup_fig*.png")
    return all_stats, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int, default=1500)
    parser.add_argument("--eval",        type=int, default=300)
    parser.add_argument("--runs",        type=int, default=5)
    parser.add_argument("--bc_episodes", type=int, default=500)
    parser.add_argument("--bc_epochs",   type=int, default=15)
    args = parser.parse_args()
    main(args.episodes, args.eval, args.runs, args.bc_episodes, args.bc_epochs)
