"""5-GBS 训练诊断脚本"""
import sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '/Users/wuenshuai/Documents/school/yanjiushengbishe/uav_routing')

from scalability_experiment import (
    make_topo_cfg, ENV_BASE, HIDDEN_DIM,
    lad_policy, shortest_path_policy,
    evaluate, _update_a3c
)
from src.envs.concurrent_fl_env import ConcurrentFLRoutingEnv
from src.agents.a3c import PolicyNet, ValueNet

num_gbs = 5
topo_cfg = make_topo_cfg(num_gbs)
env = ConcurrentFLRoutingEnv(topo_cfg=topo_cfg, **ENV_BASE)
print(f'obs_dim={env.obs_dim}, nodes={env.num_nodes}')
print(f'Server出口: {[u for u,v in env.topo.edges if v==env.server_id]}')

# 先看LAD和DQN在固定种子下的中位数
print('\n=== 基线中位数分析（固定种子）===')
r_lad = evaluate(env, lad_policy, 200, seed_offset=0)
r_sp  = evaluate(env, shortest_path_policy, 200, seed_offset=0)
print(f'LAD: median={r_lad["median_T_up"]:.1f}s, bad={r_lad["bad_rate"]*100:.0f}%')
print(f'SP:  median={r_sp["median_T_up"]:.1f}s,  bad={r_sp["bad_rate"]*100:.0f}%')

# 分析5-GBS拓扑的结构性问题
print('\n=== 5-GBS 拓扑结构分析 ===')
import networkx as nx
env.reset(seed=42)
for g in range(num_gbs):
    paths = list(nx.all_simple_paths(env.topo.graph, g, env.server_id, cutoff=4))
    short_paths = sorted(paths, key=len)[:3]
    print(f'GBS{g}: 最短{len(short_paths[0])-1}跳路径={short_paths[0]}')

# 找出Server出口链路的带宽分布
print('\n=== Server出口链路带宽分布（200个episode）===')
exit_bws = {u: [] for u,v in env.topo.edges if v==env.server_id}
for ep in range(200):
    env.reset(seed=ep)
    for u,v in env.topo.edges:
        if v == env.server_id:
            exit_bws[u].append(env.topo.available_bandwidth(u, v))
for u, bws in exit_bws.items():
    arr = np.array(bws)
    print(f'  R{u}->Server: mean={arr.mean():.1f}, min={arr.min():.1f}, p10={np.percentile(arr,10):.1f}Mbps')

# 关键：分析当T_up>50s时，是哪个GBS是掉队者
print('\n=== 掉队者分析（LAD，T_up>50s的场景）===')
straggler_count = {g: 0 for g in range(num_gbs)}
for ep in range(200):
    env.reset(seed=ep)
    done = False
    while not done:
        valid = env.get_valid_actions()
        if not valid: break
        action = lad_policy(env, valid)
        _, _, done, _, _ = env.step(action)
    r = env.get_episode_result()
    if r['T_up'] > 50 and r['up_per_gbs']:
        straggler = max(r['up_per_gbs'], key=r['up_per_gbs'].get)
        straggler_count[straggler] += 1

print('掉队者分布:', straggler_count)
