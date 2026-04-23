"""
MARL环境适配层
将ConcurrentFLRoutingEnv包装为支持CTDE的多智能体环境

关键设计：
- 局部观测（Actor输入）：当前GBS自己的位置 + 网络状态 + visited标记
  不包含其他GBS的具体位置（符合分布式执行）
- 全局状态（Critic输入）：所有GBS的局部观测拼接
  训练时可见，执行时不需要

局部观测维度：
  [cur_pos(1), server_id(1), slot_norm(1),
   node_loads(N_nodes), link_loads(N_edges),
   link_active_flows(N_edges), visited_flags(N_nodes)]
  = 3 + N_nodes + 2*N_edges + N_nodes = 3 + 2*N_nodes + 2*N_edges

全局状态维度：
  num_gbs × local_obs_dim
"""
import numpy as np
from typing import List, Dict, Tuple

from .concurrent_fl_env import ConcurrentFLRoutingEnv
from .topology import TopologyConfig
from .delay_model import DelayConfig


class MARLRoutingEnv:
    """
    MARL包装器：在ConcurrentFLRoutingEnv上提供多智能体接口
    """

    def __init__(self, base_env: ConcurrentFLRoutingEnv):
        self.env = base_env
        self.num_gbs    = base_env.num_gbs
        self.num_nodes  = base_env.num_nodes
        self.server_id  = base_env.server_id
        self.topo       = base_env.topo
        self.delay_model= base_env.delay_model
        self.action_space_n = base_env.action_space_n

        n = self.num_nodes
        e = len(self.topo.edges)
        # 局部观测：cur_pos(1) + server_id(1) + slot(1) + node_loads(n)
        #           + link_loads(e) + active_flows(e) + visited(n)
        self.local_obs_dim  = 3 + n + 2 * e + n
        # 全局状态：所有GBS的局部观测拼接
        self.global_state_dim = self.num_gbs * self.local_obs_dim

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self._get_local_obs_all(), self._get_global_state()

    def step(self, action: int):
        """单步：当前GBS执行一跳"""
        obs, reward, done, truncated, info = self.env.step(action)
        local_obs_all = self._get_local_obs_all()
        global_state  = self._get_global_state()
        return local_obs_all, global_state, reward, done, info

    def get_current_gbs(self) -> int:
        return self.env.get_current_gbs()

    def get_valid_actions(self, gbs: int = None) -> List[int]:
        return self.env.get_valid_actions(gbs)

    def get_episode_result(self) -> dict:
        return self.env.get_episode_result()

    def get_local_obs(self, gbs_id: int) -> np.ndarray:
        """获取指定GBS的局部观测"""
        env = self.env
        n = self.num_nodes
        e = len(self.topo.edges)

        cur_pos  = env.gbs_pos.get(gbs_id, gbs_id)
        slot_norm= env.current_slot / max(env.num_slots, 1)

        node_loads   = self.topo.get_node_feature_vector()
        link_loads   = self.topo.get_link_feature_vector()
        active_flows = np.array(
            [env.link_active_flows.get(e_key, 0) / max(self.num_gbs, 1)
             for e_key in self.topo.edges], dtype=np.float32)
        visited = np.array(
            [1.0 if i in env.visited.get(gbs_id, set()) else 0.0
             for i in range(n)], dtype=np.float32)

        obs = np.concatenate([
            [float(cur_pos) / n],
            [float(self.server_id) / n],
            [slot_norm],
            node_loads,
            link_loads,
            active_flows,
            visited,
        ], dtype=np.float32)
        return obs

    def _get_local_obs_all(self) -> Dict[int, np.ndarray]:
        return {g: self.get_local_obs(g) for g in range(self.num_gbs)}

    def _get_global_state(self) -> np.ndarray:
        """全局状态 = 所有GBS局部观测拼接"""
        parts = [self.get_local_obs(g) for g in range(self.num_gbs)]
        return np.concatenate(parts, dtype=np.float32)

    def make_valid_mask(self, valid_actions: List[int]) -> np.ndarray:
        """生成动作mask向量 [action_dim]，有效动作为1"""
        mask = np.zeros(self.action_space_n, dtype=np.float32)
        for a in valid_actions:
            mask[a] = 1.0
        return mask

    def action_to_local_idx(self, action: int, valid_actions: List[int]) -> int:
        """将全局动作编号转换为在valid_actions中的局部索引"""
        return valid_actions.index(action)
