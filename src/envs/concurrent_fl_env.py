"""
并发多流FL路由环境

与串行版的核心区别：
- 3个GBS同时出发，每step轮流让一个GBS走一跳
- 链路带宽按当前活跃流数分割（竞争建模）
- 状态包含所有GBS的当前位置，让agent感知竞争
- T_up = max(所有GBS到达时间)，体现掉队者效应

Dijkstra的弱点：独立为每个GBS计算最优路径，
多个GBS会选同一条路径（如都经过Router5->Server），
造成带宽竞争，T_up被最慢的GBS拖慢。

A3C的优势：感知其他GBS位置，学会主动分散路径，
避免带宽竞争，降低T_up。
"""
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Set, Optional

from .topology import NetworkTopology, TopologyConfig
from .delay_model import DelayModel, DelayConfig


class ConcurrentFLRoutingEnv:
    """
    并发多流路由环境

    状态向量 (60维)：
      [cur_gbs_id(1), cur_pos(1), all_gbs_pos(3), server_id(1), slot_norm(1),
       node_loads(8), link_loads(18), link_active_flows(18), visited_flags(8),
       gbs_done_flags(3)]
      共 1+1+3+1+1 + 8+18+18+8+3 = 62维
    """

    def __init__(self,
                 topo_cfg: TopologyConfig = None,
                 delay_cfg: DelayConfig = None,
                 delta_t: float = 5.0,
                 num_slots: int = 100,
                 g_hop: float = -1.0,
                 alpha_1: float = 0.4,
                 alpha_2: float = 0.1,
                 w_delay: float = 0.5,
                 r_success: float = 20.0,
                 r_fail: float = -5.0,
                 r_loop: float = -2.0,
                 r_compete: float = -1.5,   # 链路竞争额外惩罚（加强）
                 r_diversity: float = 0.3,  # 选择其他GBS未用路径的奖励
                 beta_tup: float = 3.0,     # episode级T_up奖励（加强）
                 max_steps_per_gbs: int = 50):

        self.topo = NetworkTopology(topo_cfg)
        self.delay_model = DelayModel(delay_cfg)

        self.delta_t = delta_t
        self.num_slots = num_slots
        self.current_slot = 0
        self.real_time = 0.0

        self.g_hop = g_hop
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.w_delay = w_delay
        self.r_success = r_success
        self.r_fail = r_fail
        self.r_loop = r_loop
        self.r_compete = r_compete
        self.r_diversity = r_diversity
        self.beta_tup = beta_tup
        self.max_steps_per_gbs = max_steps_per_gbs

        self.num_nodes = self.topo.num_nodes
        self.server_id = self.topo.server_id
        self.num_gbs = self.topo.num_gbs
        self.action_space_n = self.num_nodes

        # obs: 标量5 + 所有gbs位置3 + node_loads8 + link_loads18
        #      + link_active_flows18 + visited8 + done_flags3 = 63维
        self.obs_dim = (1 + 1 + self.num_gbs + 1 + 1
                        + self.num_nodes
                        + len(self.topo.edges)
                        + len(self.topo.edges)
                        + self.num_nodes
                        + self.num_gbs)

        # 运行时状态
        self.gbs_pos: Dict[int, int] = {}          # 每个GBS当前位置
        self.gbs_paths: Dict[int, List[int]] = {}  # 每个GBS走过的路径
        self.gbs_done: Dict[int, bool] = {}        # 是否已到达
        self.gbs_success: Dict[int, bool] = {}
        self.gbs_step_count: Dict[int, int] = {}
        self.gbs_rewards: Dict[int, float] = {}
        self.gbs_arrival_step: Dict[int, int] = {} # 到达时的step编号
        self.visited: Dict[int, Set[int]] = {}     # 每个GBS的已访问节点

        # 链路活跃流计数（并发竞争核心）
        self.link_active_flows: Dict[Tuple, int] = {}

        # 决策顺序
        self.decision_queue: List[int] = []  # 当前轮次待决策的GBS列表
        self.current_decision_idx: int = 0   # 当前轮到第几个
        self.global_step: int = 0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.topo.reset(seed)
        self.current_slot = 0
        self.real_time = 0.0
        self.global_step = 0

        for g in range(self.num_gbs):
            self.gbs_pos[g] = g
            self.gbs_paths[g] = [g]
            self.gbs_done[g] = False
            self.gbs_success[g] = False
            self.gbs_step_count[g] = 0
            self.gbs_rewards[g] = 0.0
            self.gbs_arrival_step[g] = -1
            self.visited[g] = {g}

        for e in self.topo.edges:
            self.link_active_flows[e] = 0

        # 决策队列：每轮所有未完成的GBS各走一跳
        self._rebuild_queue()
        return self._obs()

    def step(self, action: int):
        """当前轮到的GBS执行一跳"""
        self.global_step += 1
        gbs = self.decision_queue[self.current_decision_idx]
        self.gbs_step_count[gbs] += 1

        # 步级链路波动
        self.topo.step_fluctuation()

        cur = self.gbs_pos[gbs]
        valid = self.topo.successors(cur)
        action_valid = action in valid
        reached = action_valid and (action == self.server_id)

        reward, _ = self._calc_reward(gbs, cur, action, action_valid, reached)

        if action_valid:
            # 释放旧链路活跃流（从上一个节点到当前节点的链路）
            if len(self.gbs_paths[gbs]) >= 2:
                prev_e = (self.gbs_paths[gbs][-2], self.gbs_paths[gbs][-1])
                if prev_e in self.link_active_flows:
                    self.link_active_flows[prev_e] = max(0, self.link_active_flows[prev_e] - 1)

            # 占用新链路
            new_e = (cur, action)
            if new_e in self.link_active_flows:
                self.link_active_flows[new_e] += 1

            self.gbs_paths[gbs].append(action)
            self.gbs_pos[gbs] = action
            self.visited[gbs].add(action)

        self.gbs_rewards[gbs] += reward

        # 检查当前GBS是否完成
        if reached:
            self.gbs_done[gbs] = True
            self.gbs_success[gbs] = True
            self.gbs_arrival_step[gbs] = self.global_step
            self.topo.commit_path(self.gbs_paths[gbs])
            # 释放该GBS占用的链路
            self._release_gbs_links(gbs)

        elif not action_valid or self.gbs_step_count[gbs] >= self.max_steps_per_gbs:
            self.gbs_done[gbs] = True
            self.gbs_success[gbs] = False
            self.gbs_rewards[gbs] = self.r_fail
            self._release_gbs_links(gbs)

        # 移到下一个待决策GBS
        self.current_decision_idx += 1
        if self.current_decision_idx >= len(self.decision_queue):
            # 一轮结束，重建队列（只含未完成的GBS）
            self._rebuild_queue()

        all_done = all(self.gbs_done.values())

        if all_done:
            reward += self._episode_bonus()

        info = {
            "all_done": all_done,
            "current_gbs": gbs,
            "action_valid": action_valid,
            "gbs_done": dict(self.gbs_done),
        }
        return self._obs(), reward, all_done, False, info

    def get_current_gbs(self) -> int:
        if self.current_decision_idx < len(self.decision_queue):
            return self.decision_queue[self.current_decision_idx]
        return 0

    def get_valid_actions(self, gbs: int = None) -> List[int]:
        g = gbs if gbs is not None else self.get_current_gbs()
        return self.topo.successors(self.gbs_pos[g])

    def get_episode_result(self) -> dict:
        success_paths = {
            g: self.gbs_paths[g]
            for g in range(self.num_gbs) if self.gbs_success[g]
        }
        if success_paths:
            # 并发场景：T_up基于各GBS路径时延（考虑竞争后的有效带宽）
            t_up, up_per_gbs = self.delay_model.upload_delay(success_paths, self.topo)
        else:
            t_up, up_per_gbs = float('inf'), {}

        if t_up < float('inf'):
            self._advance_time(t_up)

        return {
            "total_reward": sum(self.gbs_rewards.values()),
            "success_count": sum(self.gbs_success.values()),
            "gbs_success": dict(self.gbs_success),
            "gbs_paths": {k: list(v) for k, v in self.gbs_paths.items()},
            "gbs_rewards": dict(self.gbs_rewards),
            "T_up": t_up,
            "T_round": t_up + self.delay_model.cfg.t_agg if t_up < float('inf') else float('inf'),
            "up_per_gbs": up_per_gbs,
            "real_time": self.real_time,
            "current_slot": self.current_slot,
            # 路径多样性：3条路径中经过不同Router的数量
            "path_diversity": self._path_diversity(),
        }

    def close(self):
        pass

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _rebuild_queue(self):
        """重建决策队列：未完成的GBS"""
        self.decision_queue = [g for g in range(self.num_gbs) if not self.gbs_done[g]]
        self.current_decision_idx = 0

    def _link_used_by_others(self, gbs: int, u: int, v: int) -> bool:
        """检查其他GBS的历史路径中是否已经使用过链路(u,v)"""
        for other in range(self.num_gbs):
            if other == gbs:
                continue
            path = self.gbs_paths[other]
            for i in range(len(path) - 1):
                if path[i] == u and path[i+1] == v:
                    return True
        return False

    def _release_gbs_links(self, gbs: int):
        """释放某GBS占用的所有链路活跃流"""
        path = self.gbs_paths[gbs]
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            if e in self.link_active_flows:
                self.link_active_flows[e] = max(0, self.link_active_flows[e] - 1)

    def _effective_bandwidth(self, u: int, v: int) -> float:
        """考虑活跃流竞争后的有效带宽"""
        base_bw = self.topo.available_bandwidth(u, v)
        flows = self.link_active_flows.get((u, v), 0)
        # 带宽按活跃流数平均分配（+1避免除零）
        return max(base_bw / (flows + 1), 0.1)

    def _calc_reward(self, gbs: int, cur: int, nxt: int,
                     valid: bool, reached: bool) -> Tuple[float, dict]:
        if not valid:
            return self.r_fail * 0.1, {"reason": "invalid"}

        if nxt in self.visited[gbs] and nxt != self.server_id:
            return self.r_loop, {"reason": "loop"}

        if reached:
            return self.r_success, {"reason": "reached"}

        # 链路竞争惩罚：其他GBS也在使用同一条链路
        flows_on_link = self.link_active_flows.get((cur, nxt), 0)
        compete_penalty = self.r_compete * flows_on_link

        # 路径多样性奖励（可配置，默认关闭）
        other_paths_used = self._link_used_by_others(gbs, cur, nxt)
        diversity_r = -self.r_diversity if other_paths_used else self.r_diversity

        # 有效带宽（考虑竞争后的实际可用带宽）
        eff_bw = self._effective_bandwidth(cur, nxt)
        link_delay = (self.delay_model.cfg.model_size + self.delay_model.cfg.packet_size) / eff_bw
        delay_penalty = self.delay_model.normalize_delay(link_delay)

        node_load_r = 1.0 - self.topo.node_load_ratio(nxt)
        # 链路负载奖励：基于有效带宽而非原始负载率，更准确反映竞争影响
        link_load_r = eff_bw / max(self.topo.max_capacity(cur, nxt), 1.0)
        distr_cost = self._distribution_cost(nxt)

        r = (self.g_hop
             + self.alpha_1 * (node_load_r + link_load_r)
             - self.alpha_2 * distr_cost
             - self.w_delay * delay_penalty
             + compete_penalty
             + diversity_r)

        return r, {"reason": "step", "compete": flows_on_link,
                   "diversity": diversity_r, "delay_penalty": delay_penalty}

    def _episode_bonus(self) -> float:
        """Episode结束时的T_up奖励"""
        success_paths = {
            g: self.gbs_paths[g]
            for g in range(self.num_gbs) if self.gbs_success[g]
        }
        if not success_paths:
            return 0.0
        t_up, _ = self.delay_model.upload_delay(success_paths, self.topo)
        if t_up == float('inf'):
            return 0.0
        # 使用指数衰减让T_up奖励更有区分度：T_up越小奖励越高，差距更大
        # normalize_delay 用 max_delay=50s 让奖励在常见范围内有更大梯度
        norm = min(t_up / 50.0, 1.0)
        return self.beta_tup * math.exp(-2.0 * norm)

    def _distribution_cost(self, node: int) -> float:
        succs = [n for n in self.topo.successors(node) if n in self.topo.node_load]
        if not succs:
            return 0.0
        avg = sum(self.topo.node_load_ratio(n) for n in succs) / len(succs)
        if node not in self.topo.node_load:
            return 0.0
        return (2 / math.pi) * math.atan(self.topo.node_load_ratio(node) - avg)

    def _advance_time(self, elapsed: float):
        self.real_time += elapsed
        new_slot = int(self.real_time / self.delta_t)
        if new_slot > self.current_slot and new_slot < self.num_slots:
            self.topo.update_timeslot()
            self.current_slot = new_slot

    def _path_diversity(self) -> int:
        """统计3条路径经过的不同中间节点数（越大越分散）"""
        routers_used = set()
        for g in range(self.num_gbs):
            for node in self.gbs_paths[g]:
                if self.num_gbs <= node < self.server_id:
                    routers_used.add(node)
        return len(routers_used)

    def _obs(self) -> np.ndarray:
        cur_gbs = self.get_current_gbs()
        slot_norm = self.current_slot / max(self.num_slots, 1)

        # 所有GBS的归一化位置（已完成的标记为server_id）
        all_pos = np.array(
            [float(self.gbs_pos[g]) / self.num_nodes for g in range(self.num_gbs)],
            dtype=np.float32
        )

        # 链路活跃流数（归一化，最多num_gbs条流）
        link_flows = np.array(
            [self.link_active_flows.get(e, 0) / max(self.num_gbs, 1)
             for e in self.topo.edges],
            dtype=np.float32
        )

        visited_flags = np.array(
            [1.0 if i in self.visited[cur_gbs] else 0.0
             for i in range(self.num_nodes)],
            dtype=np.float32
        )

        done_flags = np.array(
            [1.0 if self.gbs_done[g] else 0.0 for g in range(self.num_gbs)],
            dtype=np.float32
        )

        obs = np.concatenate([
            [float(cur_gbs) / self.num_gbs],
            [float(self.gbs_pos[cur_gbs]) / self.num_nodes],
            all_pos,
            [float(self.server_id) / self.num_nodes],
            [slot_norm],
            self.topo.get_node_feature_vector(),
            self.topo.get_link_feature_vector(),
            link_flows,
            visited_flags,
            done_flags,
        ], dtype=np.float32)
        return obs
