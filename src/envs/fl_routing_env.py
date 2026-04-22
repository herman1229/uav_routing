"""
FL路由环境（v2 - 挑战性增强版）
改进：
- 不均匀链路容量（接入瓶颈）+ 链路随机失效
- 环路检测与惩罚（visited集合）
- Episode级T_up奖励（引导A3C直接优化FL时延）
- 状态加入visited节点标记
"""
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Set

from .topology import NetworkTopology, TopologyConfig
from .delay_model import DelayModel, DelayConfig


class FLRoutingEnv:
    """
    Episode = 一轮FL通信（上传阶段）
    3个GBS依次向Server路由

    状态向量：
      [flow_id(1), current_node(1), server_id(1), model_size_norm(1),
       time_slot_norm(1), node_load_features(8), link_load_features(18),
       visited_flags(8)]
      共 5 + 8 + 18 + 8 = 39 维
    """

    def __init__(self, topo_cfg: TopologyConfig = None, delay_cfg: DelayConfig = None,
                 delta_t: float = 5.0, num_slots: int = 100,
                 g_hop: float = -1.0, alpha_1: float = 0.4, alpha_2: float = 0.1,
                 w_delay: float = 0.5, r_success: float = 10.0, r_fail: float = -10.0,
                 r_loop: float = -2.0, beta_tup: float = 1.0,
                 max_steps_per_gbs: int = 30):

        self.topo = NetworkTopology(topo_cfg)
        self.delay_model = DelayModel(delay_cfg)

        # 时隙参数
        self.delta_t = delta_t
        self.num_slots = num_slots
        self.current_slot = 0
        self.real_time = 0.0

        # 奖励参数
        self.g_hop = g_hop
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.w_delay = w_delay
        self.r_success = r_success
        self.r_fail = r_fail
        self.r_loop = r_loop         # 环路惩罚
        self.beta_tup = beta_tup     # episode级T_up奖励权重
        self.max_steps_per_gbs = max_steps_per_gbs

        # 空间定义
        self.num_nodes = self.topo.num_nodes
        self.server_id = self.topo.server_id
        self.num_gbs = self.topo.num_gbs
        self.action_space_n = self.num_nodes

        # obs_dim: 5标量 + node_loads(8) + link_loads(18) + visited_flags(8)
        self.obs_dim = 5 + self.num_nodes + len(self.topo.edges) + self.num_nodes

        # Episode 状态
        self.current_gbs = 0
        self.current_node = 0
        self.episode_step = 0
        self.gbs_paths: Dict[int, List[int]] = {}
        self.gbs_rewards: Dict[int, float] = {}
        self.gbs_success: Dict[int, bool] = {}
        self.gbs_step_count: Dict[int, int] = {}
        self.gbs_delays: Dict[int, float] = {}
        self.visited: Set[int] = set()  # 当前GBS的已访问节点

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.topo.reset(seed)
        self.current_gbs = 0
        self.current_node = 0
        self.episode_step = 0
        self.gbs_paths = {i: [i] for i in range(self.num_gbs)}
        self.gbs_rewards = {i: 0.0 for i in range(self.num_gbs)}
        self.gbs_success = {i: False for i in range(self.num_gbs)}
        self.gbs_step_count = {i: 0 for i in range(self.num_gbs)}
        self.gbs_delays = {i: 0.0 for i in range(self.num_gbs)}
        self.visited = {0}  # GBS0已访问
        return self._obs()

    def step(self, action: int):
        self.episode_step += 1
        self.gbs_step_count[self.current_gbs] += 1

        # episode内步级链路波动（模拟UAV实时移动）
        self.topo.step_fluctuation()

        valid_actions = self.topo.successors(self.current_node)
        action_valid = action in valid_actions
        reached = action_valid and (action == self.server_id)

        reward, reward_info = self._calc_reward(self.current_node, action, action_valid, reached)

        if action_valid:
            self.gbs_paths[self.current_gbs].append(action)
            self.current_node = action
            self.visited.add(action)
        self.gbs_rewards[self.current_gbs] += reward

        gbs_done = (self.current_node == self.server_id)
        gbs_timeout = (self.gbs_step_count[self.current_gbs] >= self.max_steps_per_gbs)

        if gbs_done or gbs_timeout:
            if gbs_done:
                self.gbs_success[self.current_gbs] = True
                path = self.gbs_paths[self.current_gbs]
                self.topo.commit_path(path)
                self.gbs_delays[self.current_gbs] = self.delay_model.path_delay(path, self.topo)
            else:
                self.gbs_success[self.current_gbs] = False
                self.gbs_rewards[self.current_gbs] = self.r_fail
                self.gbs_delays[self.current_gbs] = float('inf')

            self.current_gbs += 1
            all_done = self.current_gbs >= self.num_gbs
            if not all_done:
                # 重置visited集合，开始新GBS的路由
                self.current_node = self.current_gbs
                self.visited = {self.current_gbs}
            else:
                # 所有GBS完成：加入episode级T_up奖励
                success_paths = {
                    g: self.gbs_paths[g]
                    for g in range(self.num_gbs) if self.gbs_success[g]
                }
                if success_paths:
                    t_up, _ = self.delay_model.upload_delay(success_paths, self.topo)
                    if t_up < float('inf'):
                        # T_up越小额外奖励越高，归一化后缩放
                        tup_bonus = self.beta_tup * (1.0 - self.delay_model.normalize_delay(t_up))
                        reward += tup_bonus
                        self.gbs_rewards[self.current_gbs - 1] += tup_bonus
            terminated = all_done
        else:
            terminated = False

        obs = self._obs()
        info = {
            "all_gbs_done": terminated,
            "current_gbs": min(self.current_gbs, self.num_gbs - 1),
            "action_valid": action_valid,
            "reward_info": reward_info,
        }
        return obs, reward, terminated, False, info

    def get_episode_result(self) -> dict:
        """Episode结束后调用，返回完整FL round指标"""
        success_paths = {
            gbs: self.gbs_paths[gbs]
            for gbs in range(self.num_gbs) if self.gbs_success[gbs]
        }
        if success_paths:
            t_up, up_per_gbs = self.delay_model.upload_delay(success_paths, self.topo)
        else:
            t_up, up_per_gbs = float('inf'), {}

        t_round = t_up + self.delay_model.cfg.t_agg

        # 更新真实时间与时隙
        if t_up < float('inf'):
            self._advance_time(t_up)

        return {
            "total_reward": sum(self.gbs_rewards.values()),
            "success_count": sum(self.gbs_success.values()),
            "gbs_success": dict(self.gbs_success),
            "gbs_paths": {k: list(v) for k, v in self.gbs_paths.items()},
            "gbs_rewards": dict(self.gbs_rewards),
            "gbs_delays": dict(self.gbs_delays),
            "T_up": t_up,
            "T_round": t_round,
            "up_per_gbs": up_per_gbs,
            "real_time": self.real_time,
            "current_slot": self.current_slot,
        }

    def get_valid_actions(self, node: int = None) -> List[int]:
        n = node if node is not None else self.current_node
        return self.topo.successors(n)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _advance_time(self, elapsed: float):
        """推进真实时间，检查是否跨越时隙"""
        self.real_time += elapsed
        new_slot = int(self.real_time / self.delta_t)
        if new_slot > self.current_slot and new_slot < self.num_slots:
            self.topo.update_timeslot()
            self.current_slot = new_slot

    def _calc_reward(self, cur: int, nxt: int, valid: bool, reached: bool) -> Tuple[float, dict]:
        if not valid:
            return self.r_fail * 0.1, {"reason": "invalid"}

        # 环路检测：已访问节点给惩罚
        if nxt in self.visited and nxt != self.server_id:
            return self.r_loop, {"reason": "loop"}

        if reached:
            return self.r_success, {"reason": "reached"}

        # 负载感知步骤奖励
        node_load_r = 1.0 - self.topo.node_load_ratio(nxt)
        link_load_r = 1.0 - self.topo.link_load_ratio(cur, nxt)
        distr_cost = self._distribution_cost(nxt)

        # 链路时延惩罚（基于实际可用带宽，不均匀容量更有区分度）
        bw = max(self.topo.available_bandwidth(cur, nxt), 0.1)
        link_delay = (self.delay_model.cfg.model_size + self.delay_model.cfg.packet_size) / bw
        delay_penalty = self.delay_model.normalize_delay(link_delay)

        r = (self.g_hop
             + self.alpha_1 * (node_load_r + link_load_r)
             - self.alpha_2 * distr_cost
             - self.w_delay * delay_penalty)

        return r, {
            "reason": "step",
            "node_load_r": node_load_r,
            "link_load_r": link_load_r,
            "distr_cost": distr_cost,
            "delay_penalty": delay_penalty,
            "total": r,
        }

    def _distribution_cost(self, node: int) -> float:
        successors = [n for n in self.topo.successors(node) if n in self.topo.node_load]
        if not successors or node not in self.topo.node_load:
            return 0.0
        avg = sum(self.topo.node_load_ratio(n) for n in successors) / len(successors)
        return (2 / math.pi) * math.atan(self.topo.node_load_ratio(node) - avg)

    def _obs(self) -> np.ndarray:
        gbs_id = min(self.current_gbs, self.num_gbs - 1)
        slot_norm = self.current_slot / max(self.num_slots, 1)
        model_norm = self.delay_model.cfg.model_size / 100.0
        visited_flags = np.array(
            [1.0 if i in self.visited else 0.0 for i in range(self.num_nodes)],
            dtype=np.float32
        )
        obs = np.concatenate([
            [float(gbs_id) / self.num_gbs],
            [float(self.current_node) / self.num_nodes],
            [float(self.server_id) / self.num_nodes],
            [model_norm],
            [slot_norm],
            self.topo.get_node_feature_vector(),
            self.topo.get_link_feature_vector(),
            visited_flags,
        ], dtype=np.float32)
        return obs
