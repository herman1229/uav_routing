"""
基线算法1：最短路径（跳数最少）
基线算法2：负载感知Dijkstra（以链路时延为边权，每步重新计算 = 在线自适应版）
基线算法3：负载感知Dijkstra（路由开始时计算一次路径，沿路执行 = 离线版，更贴近真实）
基线算法4：随机路由
"""
import networkx as nx
import numpy as np
import random
from typing import List, Optional, Dict

from ..envs.fl_routing_env import FLRoutingEnv


def _run_episode(env: FLRoutingEnv, policy_fn) -> dict:
    """通用episode运行框架（每步调用policy_fn）"""
    state = env.reset()
    done = False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        action = policy_fn(env, valid)
        state, _, done, _, _ = env.step(action)
    return env.get_episode_result()


def _compute_dijkstra_path(env: FLRoutingEnv, src: int) -> List[int]:
    """计算从src到server的Dijkstra最短路径（以当前链路时延为权重）"""
    M = env.delay_model.cfg.model_size
    L = env.delay_model.cfg.packet_size
    weighted_graph = nx.DiGraph()
    for u, v in env.topo.edges:
        bw = max(env.topo.available_bandwidth(u, v), 0.1)
        weight = (M + L) / bw + env.delay_model.cfg.hop_delay
        weighted_graph.add_edge(u, v, weight=weight)
    try:
        path = nx.shortest_path(weighted_graph, src, env.server_id, weight='weight')
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# ------------------------------------------------------------------
# 随机策略
# ------------------------------------------------------------------

def random_policy(env: FLRoutingEnv, valid_actions: List[int]) -> int:
    return random.choice(valid_actions)


def run_random(env: FLRoutingEnv, n_episodes: int = 500, seed: int = 0) -> List[dict]:
    random.seed(seed)
    np.random.seed(seed)
    results = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        result = _run_episode(env, random_policy)
        result["episode"] = ep
        results.append(result)
    return results


# ------------------------------------------------------------------
# 最短路径（跳数）
# ------------------------------------------------------------------

def shortest_path_policy(env: FLRoutingEnv, valid_actions: List[int]) -> int:
    """贪心选择距离Server跳数最少的下一跳"""
    best_action = valid_actions[0]
    min_hops = float('inf')
    for a in valid_actions:
        try:
            hops = nx.shortest_path_length(env.topo.graph, a, env.server_id)
            if hops < min_hops:
                min_hops = hops
                best_action = a
        except nx.NetworkXNoPath:
            pass
    return best_action


def run_shortest_path(env: FLRoutingEnv, n_episodes: int = 500, seed: int = 0) -> List[dict]:
    random.seed(seed)
    np.random.seed(seed)
    results = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        result = _run_episode(env, shortest_path_policy)
        result["episode"] = ep
        results.append(result)
    return results


# ------------------------------------------------------------------
# 负载感知Dijkstra（以链路时延为边权）
# ------------------------------------------------------------------

def load_aware_dijkstra_policy(env: FLRoutingEnv, valid_actions: List[int]) -> int:
    """
    以当前链路时延为边权，选择到Server时延最小的下一跳
    边权 = (M + L) / available_bandwidth
    """
    M = env.delay_model.cfg.model_size
    L = env.delay_model.cfg.packet_size

    # 构建带权图
    weighted_graph = nx.DiGraph()
    for u, v in env.topo.edges:
        bw = max(env.topo.available_bandwidth(u, v), 0.1)
        weight = (M + L) / bw + env.delay_model.cfg.hop_delay
        weighted_graph.add_edge(u, v, weight=weight)

    best_action = valid_actions[0]
    min_delay = float('inf')
    for a in valid_actions:
        try:
            delay = nx.shortest_path_length(weighted_graph, a, env.server_id, weight='weight')
            if delay < min_delay:
                min_delay = delay
                best_action = a
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return best_action


def run_load_aware_dijkstra(env: FLRoutingEnv, n_episodes: int = 500, seed: int = 0) -> List[dict]:
    random.seed(seed)
    np.random.seed(seed)
    results = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        result = _run_episode(env, load_aware_dijkstra_policy)
        result["episode"] = ep
        results.append(result)
    return results
