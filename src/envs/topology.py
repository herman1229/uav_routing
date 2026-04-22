"""
网络拓扑定义模块
节点编号: GBS(0~num_gbs-1), Router(num_gbs~num_gbs+num_routers-1), Server(最后一个)

改进：
- 不均匀链路容量：GBS->Router窄（接入瓶颈），Router间中等，Router->Server宽
- 链路随机失效：每个时隙有概率某条链路带宽大幅降低（模拟UAV移动）
- 更高初始负载：让拥塞效应更明显
- 更大负载提交量：每次成功传输对网络影响更大
"""
import networkx as nx
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TopologyConfig:
    num_gbs: int = 3
    num_routers: int = 4
    num_servers: int = 1
    node_capacity: int = 50
    # 各类链路容量（Mbps）
    gbs_to_router_capacity: float = 20.0   # 接入链路（窄，形成瓶颈）
    router_to_router_capacity: float = 40.0 # 骨干链路（中等）
    router_to_server_capacity: float = 80.0 # 汇聚链路（宽）
    init_node_load_range: Tuple[int, int] = (15, 35)   # 更高初始负载
    init_link_load_range: Tuple[float, float] = (0.3, 0.65)  # 更高初始链路负载
    bandwidth_change_std: float = 4.0      # 时隙间带宽波动
    load_change_std: float = 3.0
    link_failure_prob: float = 0.15        # 每时隙每条链路失效概率
    link_failure_severity: float = 0.75    # 失效时带宽降低比例
    step_failure_prob: float = 0.08        # 每步（episode内）链路随机波动概率
    commit_bw_fraction: float = 0.18       # 每次成功传输占用带宽比例（更大）
    commit_load_increment: float = 2.0     # 每次成功传输节点负载增量


class NetworkTopology:
    """
    网络拓扑管理：节点/链路定义、负载状态、时隙更新
    拓扑结构（固定）：
      GBS(0,1,2) -> Router(3,4,5,6) -> Server(7)
      GBS0->Router3,Router4  (窄接入链路)
      GBS1->Router4,Router5
      GBS2->Router5,Router6
      Router3<->Router4<->Router5<->Router6（中等骨干链路）
      Router5->Server7, Router6->Server7  (宽汇聚链路)
    """

    def __init__(self, cfg: TopologyConfig = None):
        self.cfg = cfg or TopologyConfig()
        self.num_gbs = self.cfg.num_gbs
        self.num_routers = self.cfg.num_routers
        self.server_id = self.num_gbs + self.num_routers  # 7
        self.num_nodes = self.server_id + 1               # 8

        self.graph = self._build_graph()
        self.edges = list(self.graph.edges())
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        # 各链路的最大容量（不均匀）
        self.link_max_capacity: Dict[Tuple, float] = self._init_link_capacities()

        # 节点/链路状态（运行时）
        self.node_load: Dict[int, float] = {}
        self.link_bandwidth: Dict[Tuple, float] = {}
        self.link_load: Dict[Tuple, float] = {}

        self._init_state()

    # ------------------------------------------------------------------
    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for i in range(self.num_nodes):
            g.add_node(i)
        connections = [
            (0, 3), (0, 4),
            (1, 4), (1, 5),
            (2, 5), (2, 6),
            (3, 4), (4, 3),
            (3, 5), (5, 3),
            (4, 5), (5, 4),
            (4, 6), (6, 4),
            (5, 6), (6, 5),
            (5, 7), (6, 7),
        ]
        for u, v in connections:
            g.add_edge(u, v)
        return g

    def _init_link_capacities(self) -> Dict[Tuple, float]:
        """为不同类型链路分配不均匀容量"""
        caps = {}
        for u, v in self.edges:
            if u < self.num_gbs:
                # GBS -> Router：接入链路，窄
                caps[(u, v)] = self.cfg.gbs_to_router_capacity
            elif v == self.server_id:
                # Router -> Server：汇聚链路，宽
                caps[(u, v)] = self.cfg.router_to_server_capacity
            else:
                # Router <-> Router：骨干链路，中等
                caps[(u, v)] = self.cfg.router_to_router_capacity
        return caps

    def _init_state(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        lo, hi = self.cfg.init_node_load_range
        for i in range(self.num_nodes):
            self.node_load[i] = float(random.randint(lo, hi))
        blo, bhi = self.cfg.init_link_load_range
        for e in self.edges:
            load = random.uniform(blo, bhi)
            self.link_load[e] = load
            self.link_bandwidth[e] = self.link_max_capacity[e] * (1.0 - load)

    def reset(self, seed: int = None):
        self._init_state(seed)

    def update_timeslot(self):
        """时隙切换：带宽波动 + 随机链路失效事件"""
        std_bw = self.cfg.bandwidth_change_std
        std_load = self.cfg.load_change_std

        for e in self.edges:
            cap = self.link_max_capacity[e]
            # 随机失效：带宽大幅降低
            if random.random() < self.cfg.link_failure_prob:
                new_bw = self.link_bandwidth[e] * (1.0 - self.cfg.link_failure_severity)
            else:
                delta = np.random.normal(0, std_bw)
                new_bw = self.link_bandwidth[e] + delta
            new_bw = float(np.clip(new_bw, 0.5, cap))
            self.link_bandwidth[e] = new_bw
            self.link_load[e] = 1.0 - new_bw / cap

        for i in range(self.num_nodes):
            delta = np.random.normal(0, std_load)
            self.node_load[i] = float(np.clip(self.node_load[i] + delta, 0, self.cfg.node_capacity))

    def commit_path(self, path_nodes: List[int],
                    load_increment: float = None, bw_fraction: float = None):
        """路由成功后提交负载更新（使用配置中的较大值）"""
        inc = load_increment if load_increment is not None else self.cfg.commit_load_increment
        frac = bw_fraction if bw_fraction is not None else self.cfg.commit_bw_fraction
        for n in path_nodes:
            self.node_load[n] = min(self.node_load[n] + inc, self.cfg.node_capacity)
        for i in range(len(path_nodes) - 1):
            e = (path_nodes[i], path_nodes[i + 1])
            if e in self.link_load:
                cap = self.link_max_capacity[e]
                self.link_load[e] = min(self.link_load[e] + frac, 1.0)
                self.link_bandwidth[e] = cap * (1.0 - self.link_load[e])

    def step_fluctuation(self):
        """每步调用：小概率对部分链路做随机波动，模拟UAV实时移动"""
        if self.cfg.step_failure_prob <= 0:
            return
        for e in self.edges:
            if random.random() < self.cfg.step_failure_prob:
                cap = self.link_max_capacity[e]
                # 随机波动：±20%
                delta = random.uniform(-0.2, 0.2) * cap
                new_bw = float(np.clip(self.link_bandwidth[e] + delta, 0.5, cap))
                self.link_bandwidth[e] = new_bw
                self.link_load[e] = 1.0 - new_bw / cap

    # ------------------------------------------------------------------
    # 查询接口
    def successors(self, node: int) -> List[int]:
        return list(self.graph.successors(node))

    def available_bandwidth(self, u: int, v: int) -> float:
        return self.link_bandwidth.get((u, v), 0.0)

    def max_capacity(self, u: int, v: int) -> float:
        return self.link_max_capacity.get((u, v), 1.0)

    def node_load_ratio(self, node: int) -> float:
        return self.node_load[node] / self.cfg.node_capacity

    def link_load_ratio(self, u: int, v: int) -> float:
        return self.link_load.get((u, v), 0.0)

    def get_node_feature_vector(self) -> np.ndarray:
        return np.array([self.node_load_ratio(i) for i in range(self.num_nodes)], dtype=np.float32)

    def get_link_feature_vector(self) -> np.ndarray:
        # 同时提供负载率和归一化可用带宽，给agent更丰富的信息
        feats = []
        for e in self.edges:
            feats.append(self.link_load_ratio(*e))
        return np.array(feats, dtype=np.float32)

    def node_type(self, node: int) -> str:
        if node < self.num_gbs:
            return "GBS"
        elif node < self.server_id:
            return "Router"
        return "Server"
