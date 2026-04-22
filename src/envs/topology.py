"""
网络拓扑定义模块
节点编号: GBS(0~num_gbs-1), Router(num_gbs~num_gbs+num_routers-1), Server(最后一个)
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
    link_capacity: float = 100.0
    init_node_load_range: Tuple[int, int] = (5, 15)
    init_link_load_range: Tuple[float, float] = (0.2, 0.4)
    bandwidth_change_std: float = 5.0
    load_change_std: float = 2.0


class NetworkTopology:
    """
    网络拓扑管理：节点/链路定义、负载状态、时隙更新
    拓扑结构（固定）：
      GBS(0,1,2) -> Router(3,4,5,6) -> Server(7)
      GBS0->Router3,Router4
      GBS1->Router4,Router5
      GBS2->Router5,Router6
      Router3<->Router4<->Router5<->Router6（全双向）
      Router5->Server7, Router6->Server7
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

        # 节点/链路状态（运行时）
        self.node_load: Dict[int, float] = {}
        self.link_bandwidth: Dict[Tuple, float] = {}  # 当前可用带宽 Mbps
        self.link_load: Dict[Tuple, float] = {}       # 当前负载率 [0,1]

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
            self.link_bandwidth[e] = self.cfg.link_capacity * (1.0 - load)

    def reset(self, seed: int = None):
        self._init_state(seed)

    def update_timeslot(self):
        """时隙切换时扰动网络状态，模拟动态变化"""
        std_bw = self.cfg.bandwidth_change_std
        std_load = self.cfg.load_change_std
        for e in self.edges:
            delta = np.random.normal(0, std_bw)
            new_bw = float(np.clip(self.link_bandwidth[e] + delta, 1.0, self.cfg.link_capacity))
            self.link_bandwidth[e] = new_bw
            self.link_load[e] = 1.0 - new_bw / self.cfg.link_capacity
        for i in range(self.num_nodes):
            delta = np.random.normal(0, std_load)
            self.node_load[i] = float(np.clip(self.node_load[i] + delta, 0, self.cfg.node_capacity))

    def commit_path(self, path_nodes: List[int], load_increment: float = 1.0, bw_fraction: float = 0.08):
        """路由成功后提交负载更新"""
        for n in path_nodes:
            self.node_load[n] = min(self.node_load[n] + load_increment, self.cfg.node_capacity)
        for i in range(len(path_nodes) - 1):
            e = (path_nodes[i], path_nodes[i + 1])
            if e in self.link_load:
                self.link_load[e] = min(self.link_load[e] + bw_fraction, 1.0)
                self.link_bandwidth[e] = self.cfg.link_capacity * (1.0 - self.link_load[e])

    # ------------------------------------------------------------------
    # 查询接口
    def successors(self, node: int) -> List[int]:
        return list(self.graph.successors(node))

    def available_bandwidth(self, u: int, v: int) -> float:
        return self.link_bandwidth.get((u, v), 0.0)

    def node_load_ratio(self, node: int) -> float:
        return self.node_load[node] / self.cfg.node_capacity

    def link_load_ratio(self, u: int, v: int) -> float:
        return self.link_load.get((u, v), 0.0)

    def get_node_feature_vector(self) -> np.ndarray:
        return np.array([self.node_load_ratio(i) for i in range(self.num_nodes)], dtype=np.float32)

    def get_link_feature_vector(self) -> np.ndarray:
        return np.array([self.link_load_ratio(*e) for e in self.edges], dtype=np.float32)

    def node_type(self, node: int) -> str:
        if node < self.num_gbs:
            return "GBS"
        elif node < self.server_id:
            return "Router"
        return "Server"
