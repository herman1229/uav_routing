import numpy as np
import networkx as nx
import random
import math
from typing import Dict, List, Tuple, Any

class UAVRoutingEnv:
    """
    多GBS无人机路由环境
    在一个episode内依次完成3个GBS的路由决策
    """
    
    def __init__(self):
        # 网络参数
        self.num_nodes = 8  # 3个GBS + 4个Router + 1个Server
        self.num_gbs = 3
        self.num_routers = 4
        self.server_id = 7  # 聚合参数服务器ID
        
        # 网络容量参数
        self.node_capacity = 5  # 流表项
        self.link_capacity = 1000  # Mbps（固定链路容量）
        
        # 奖励函数参数
        self.g_hop = -1
        self.alpha_1 = 0.4
        self.alpha_2 = 0.1
        self.failure_penalty = -1  # 失败惩罚
        
        # 初始化网络拓扑（单向：GBS -> Router）
        self._init_network_topology()
        
        # 初始化网络状态
        self._init_network_state()
        
        # 定义动作和观察空间
        self._define_spaces()
        
        # 环境状态
        self.current_gbs = 0  # 当前正在决策的GBS
        self.current_node = None
        self.current_time = 0
        self.episode_step = 0
        self.max_steps_per_gbs = 100  # 每个GBS的最大步数限制
        
        # Episode级别的状态
        self.gbs_paths = {}  # 存储每个GBS的路径
        self.gbs_rewards = {}  # 存储每个GBS的奖励
        self.gbs_success = {}  # 存储每个GBS是否成功
        self.gbs_step_rewards = {}  # 存储每个GBS每一步的奖励
        self.gbs_step_reward_details = {}  # 存储每个GBS每一步的奖励计算详情
        
    def _init_network_topology(self):
        """初始化网络拓扑结构（有向图：GBS->Router单向）"""
        # 创建有向图
        self.network = nx.DiGraph()
        
        # 添加节点
        for i in range(self.num_nodes):
            self.network.add_node(i)
        
        # 定义连接关系（单向拓扑）
        connections = [
            # GBS到Router连接（单向：GBS -> Router）
            (0, 3), (0, 4),  # GBS0 -> Router1, Router2
            (1, 4), (1, 5),  # GBS1 -> Router2, Router3
            (2, 5), (2, 6),  # GBS2 -> Router3, Router4
            
            # Router之间连接（双向）
            (3, 4), (4, 3),  # Router1 <-> Router2
            (3, 5), (5, 3),  # Router1 <-> Router3
            (4, 5), (5, 4),  # Router2 <-> Router3
            (4, 6), (6, 4),  # Router2 <-> Router4
            (5, 6), (6, 5),  # Router3 <-> Router4
            
            # Router到Server连接（单向：Router -> Server）
            (5, 7), (6, 7)   # Router3, Router4 -> Server
        ]
        
        for u, v in connections:
            self.network.add_edge(u, v)
        
        # 验证每个GBS至少有出边到Router
        for gbs_id in range(self.num_gbs):
            successors = list(self.network.successors(gbs_id))
            has_router = any(node >= self.num_gbs and node < self.server_id for node in successors)
            assert has_router, f"GBS {gbs_id} 必须有出边到Router"
    
    def _init_network_state(self):
        """初始化网络状态"""
        # 节点负载 (当前流表项使用数)
        self.node_load = {i: random.randint(0, 1) for i in range(self.num_nodes)}
        
        # 链路负载 (当前带宽使用率)
        self.link_load = {}
        for edge in self.network.edges():
            self.link_load[edge] = random.uniform(0.2, 0.4)  # 初始负载20%-40%
    
    def _define_spaces(self):
        """定义动作和观察空间"""
        self.action_space_n = self.num_nodes
        
        # 观察空间：[当前GBS_ID, 当前节点, 目标节点, 网络特征向量]
        network_feature_dim = self.num_nodes + len(self.network.edges())
        self.obs_dim = 3 + network_feature_dim  # GBS_ID + 当前节点 + 目标节点 + 网络特征
    
    def _get_adjacent_nodes(self, node_id):
        """获取节点的后继节点（有向图）"""
        return list(self.network.successors(node_id))
    
    def _calculate_node_load_reward(self, node_id):
        """计算节点负载奖励"""
        load_ratio = self.node_load[node_id] / self.node_capacity
        return 1 - min(load_ratio, 1.0)
    
    def _calculate_link_load_reward(self, node_from, node_to):
        """计算链路负载奖励"""
        edge = (node_from, node_to)  # 有向边
        load = self.link_load.get(edge, 0.2)
        return 1 - min(load, 1.0)
    
    def _calculate_distribution_cost(self, node_id):
        """计算节点负载分布成本"""
        successors = self._get_adjacent_nodes(node_id)
        if not successors:
            return 0
        
        avg_load = sum(self.node_load[n] for n in successors) / len(successors)
        load_diff = self.node_load[node_id] - avg_load
        cost = (2 / math.pi) * math.atan(load_diff)
        return cost
    
    def _calculate_reward(self, current_node, next_node, action_valid, reached_destination):
        """计算奖励函数"""
        if not action_valid:
            return -1, {
                'base_reward': -1,
                'node_load_reward': 0,
                'link_load_reward': 0,
                'distribution_cost': 0,
                'total_reward': -1,
                'reason': 'invalid_action'
            }
        
        if reached_destination:
            return 1, {
                'base_reward': 1,
                'node_load_reward': 0,
                'link_load_reward': 0,
                'distribution_cost': 0,
                'total_reward': 1,
                'reason': 'reached_destination'
            }
        
        base_reward = self.g_hop
        node_load_reward = self._calculate_node_load_reward(next_node)
        link_load_reward = self._calculate_link_load_reward(current_node, next_node)
        distribution_cost = self._calculate_distribution_cost(next_node)
        
        total_reward = base_reward + self.alpha_1 * (node_load_reward + link_load_reward) - self.alpha_2 * distribution_cost
        
        reward_details = {
            'base_reward': base_reward,
            'node_load_reward': node_load_reward,
            'link_load_reward': link_load_reward,
            'distribution_cost': distribution_cost,
            'alpha_1': self.alpha_1,
            'alpha_2': self.alpha_2,
            'total_reward': total_reward,
            'reason': 'normal_step'
        }
        
        return total_reward, reward_details
    
    def _get_network_features(self):
        """获取网络特征向量"""
        features = []
        
        # 节点负载特征（归一化）
        for i in range(self.num_nodes):
            features.append(self.node_load[i] / self.node_capacity)
        
        # 链路负载特征
        for edge in sorted(self.network.edges()):
            features.append(self.link_load.get(edge, 0.2))
        
        return np.array(features, dtype=np.float32)
    
    def _get_observation(self):
        """获取当前观察"""
        network_features = self._get_network_features()
        
        obs = np.concatenate([
            [self.current_gbs],        # 当前GBS ID
            [self.current_node],       # 当前节点
            [self.server_id],          # 目标节点
            network_features           # 网络特征
        ], dtype=np.float32)
        
        return obs
    
    def _update_network_load(self, path_nodes):
        """
        更新网络负载状态（基于完整路径）
        只有成功到达目标时才调用
        """
        # 更新节点负载
        for node in path_nodes:
            self.node_load[node] = min(self.node_load[node] + 1, self.node_capacity)
        
        # 更新链路负载
        for i in range(len(path_nodes) - 1):
            edge = (path_nodes[i], path_nodes[i+1])
            current_load = self.link_load.get(edge, 0.2)
            self.link_load[edge] = min(current_load + 0.08, 1.0)
    
    def reset(self, seed=None):
        """重置环境（开始新的episode）"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 重置网络状态
        self._init_network_state()
        
        # 重置episode状态
        self.current_gbs = 0
        self.current_node = 0  # 从GBS0开始
        self.current_time = 0
        self.episode_step = 0
        
        # 重置GBS级别的记录
        self.gbs_paths = {i: [] for i in range(self.num_gbs)}
        self.gbs_rewards = {i: 0.0 for i in range(self.num_gbs)}
        self.gbs_success = {i: False for i in range(self.num_gbs)}
        self.gbs_step_count = {i: 0 for i in range(self.num_gbs)}
        self.gbs_step_rewards = {i: [] for i in range(self.num_gbs)}
        self.gbs_step_reward_details = {i: [] for i in range(self.num_gbs)}
        
        return self._get_observation()
    
    def step(self, action):
        """
        执行一步动作
        在当前GBS的决策过程中执行
        """
        self.episode_step += 1
        self.gbs_step_count[self.current_gbs] += 1
        
        # 检查动作有效性
        adjacent_nodes = self._get_adjacent_nodes(self.current_node)
        action_valid = action in adjacent_nodes
        
        # 检查是否到达目标
        reached_destination = (action == self.server_id and self.current_node != self.server_id)
        
        # 计算即时奖励
        step_reward, reward_details = self._calculate_reward(self.current_node, action, action_valid, reached_destination)
        
        # 记录当前GBS的步骤奖励详情
        self.gbs_step_rewards[self.current_gbs].append(step_reward)
        self.gbs_step_reward_details[self.current_gbs].append({
            'step': self.gbs_step_count[self.current_gbs],
            'from_node': self.current_node,
            'to_node': action,
            'action_valid': action_valid,
            'reached_destination': reached_destination,
            'reward_details': reward_details
        })
        
        # 更新当前GBS的状态
        if action_valid:
            self.gbs_paths[self.current_gbs].append((self.current_node, action))
            self.current_node = action
        
        self.gbs_rewards[self.current_gbs] += step_reward
        
        # 检查当前GBS是否完成
        gbs_terminated = (self.current_node == self.server_id)
        gbs_truncated = (self.gbs_step_count[self.current_gbs] >= self.max_steps_per_gbs)
        
        # 当前GBS完成的处理
        if gbs_terminated or gbs_truncated:
            if gbs_terminated:
                self.gbs_success[self.current_gbs] = True
                # 更新网络负载
                path_nodes = [self.current_gbs] + [p[1] for p in self.gbs_paths[self.current_gbs]]
                self._update_network_load(path_nodes)
            else:
                # 失败：不更新网络负载，但给予惩罚奖励
                self.gbs_success[self.current_gbs] = False
                self.gbs_rewards[self.current_gbs] = self.failure_penalty
            
            # 切换到下一个GBS
            self.current_gbs += 1
            
            # 检查是否所有GBS都完成了
            if self.current_gbs >= self.num_gbs:
                # Episode结束
                terminated = True
                truncated = False
            else:
                # 开始下一个GBS的决策
                self.current_node = self.current_gbs
                terminated = False
                truncated = False
        else:
            terminated = False
            truncated = False
        
        # 获取新观察
        obs = self._get_observation()
        
        # 信息
        info = {
            'current_gbs': self.current_gbs if self.current_gbs < self.num_gbs else self.num_gbs - 1,
            'current_node': self.current_node,
            'action_valid': action_valid,
            'gbs_paths': self.gbs_paths.copy(),
            'gbs_rewards': self.gbs_rewards.copy(),
            'gbs_success': self.gbs_success.copy(),
            'all_gbs_done': self.current_gbs >= self.num_gbs
        }
        
        return obs, step_reward, terminated, truncated, info
    
    def get_episode_reward(self):
        """获取整个episode的总奖励（3个GBS的奖励之和）"""
        return sum(self.gbs_rewards.values())
    
    def get_episode_info(self):
        """获取episode的详细信息"""
        return {
            'gbs_paths': self.gbs_paths.copy(),
            'gbs_rewards': self.gbs_rewards.copy(),
            'gbs_success': self.gbs_success.copy(),
            'gbs_step_rewards': self.gbs_step_rewards.copy(),
            'gbs_step_reward_details': self.gbs_step_reward_details.copy(),
            'total_reward': self.get_episode_reward(),
            'success_count': sum(self.gbs_success.values()),
            'failure_count': self.num_gbs - sum(self.gbs_success.values())
        }
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Episode Step: {self.episode_step}")
            print(f"Current GBS: {self.current_gbs}")
            print(f"Current Node: {self.current_node}")
            print(f"GBS Paths: {self.gbs_paths}")
            print(f"GBS Rewards: {self.gbs_rewards}")
            print("-" * 50)
    
    def close(self):
        """关闭环境"""
        pass


def test_environment():
    """测试环境功能"""
    env = UAVRoutingEnv()
    
    print("环境测试开始...")
    print(f"节点数: {env.num_nodes}")
    print(f"网络拓扑边数: {len(env.network.edges())}")
    print(f"网络类型: {'有向图' if isinstance(env.network, nx.DiGraph) else '无向图'}")
    print(f"观察维度: {env.obs_dim}")
    print(f"动作维度: {env.action_space_n}")
    
    # 测试GBS的出边
    print("\nGBS出边测试:")
    for gbs_id in range(env.num_gbs):
        successors = list(env.network.successors(gbs_id))
        print(f"  GBS{gbs_id} → {successors}")
    
    # 测试一个完整的episode（3个GBS依次决策）
    print("\n" + "="*60)
    print("测试完整Episode（3个GBS依次决策）")
    print("="*60)
    
    obs = env.reset(seed=42)
    episode_terminated = False
    
    while not episode_terminated:
        current_gbs = min(env.current_gbs, env.num_gbs - 1)
        current_node = int(obs[1])  # obs[1]是当前节点
        
        print(f"\n当前决策GBS: {current_gbs}, 当前节点: {current_node}")
        
        # 获取有效动作
        valid_actions = list(env.network.successors(current_node))
        print(f"  有效动作: {valid_actions}")
        
        if valid_actions:
            # 简单策略：选择朝向Server的动作
            best_action = None
            min_dist = float('inf')
            
            for action in valid_actions:
                try:
                    dist = nx.shortest_path_length(env.network, action, env.server_id)
                    if dist < min_dist:
                        min_dist = dist
                        best_action = action
                except:
                    pass
            
            action = best_action if best_action is not None else valid_actions[0]
        else:
            action = env.action_space_n - 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  动作: {action}, 奖励: {reward:.3f}")
        print(f"  所有GBS完成: {info['all_gbs_done']}")
        
        if info['all_gbs_done']:
            episode_terminated = True
            print("\n" + "="*60)
            print("Episode完成！")
            episode_info = env.get_episode_info()
            print(f"总奖励: {episode_info['total_reward']:.2f}")
            print(f"成功GBS数: {episode_info['success_count']}/3")
            
            for gbs_id in range(env.num_gbs):
                path = [gbs_id] + [p[1] for p in env.gbs_paths[gbs_id]]
                status = "SUCCESS" if env.gbs_success[gbs_id] else "FAILED"
                print(f"  GBS{gbs_id}: {' -> '.join(map(str, path))} | "
                      f"Reward={env.gbs_rewards[gbs_id]:.2f} | {status}")
            print("="*60)
    
    print("\n环境测试完成")


if __name__ == "__main__":
    test_environment()
