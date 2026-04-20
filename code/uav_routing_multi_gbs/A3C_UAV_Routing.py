import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import util
import os
from datetime import datetime
from queue import Empty
from A3C_UAV_Routing_Env import UAVRoutingEnv

GLOBAL_MAX_EPISODE = 1200
GAMMA = 0.98

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class A3Cagent:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, env, numOfCPU):
        self.global_actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.global_critic = ValueNet(state_dim, hidden_dim)
        self.global_actor.share_memory()
        self.global_critic.share_memory()
        self.global_critic_optimizer = SharedAdam(self.global_critic.parameters(), lr=critic_lr, betas=(0.92, 0.999))
        self.global_actor_optimizer = SharedAdam(self.global_actor.parameters(), lr=actor_lr, betas=(0.92, 0.999))
        
        self.env = env
        self.global_episode = mp.Value('i', 0)
        self.global_episode_reward = mp.Value('d', 0.)
        self.res_queue = mp.Queue()
        self.log_queue = mp.Queue()
        
        self.workers = [Worker(i, self.global_actor, self.global_critic, 
                              self.global_critic_optimizer, self.global_actor_optimizer,
                              self.env, state_dim, hidden_dim, action_dim,
                              self.global_episode, self.global_episode_reward, 
                              self.res_queue, self.log_queue) 
                       for i in range(numOfCPU)]

    def train(self):
        [w.start() for w in self.workers]
        res = []
        finished_workers = 0
        numOfCPU = len(self.workers)
        
        while finished_workers < numOfCPU:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                finished_workers += 1
        
        [w.join() for w in self.workers]
        return res


class Worker(mp.Process):
    def __init__(self, name, global_actor, global_critic, global_critic_optimizer, 
                 global_actor_optimizer, env, state_dim, hidden_dim, action_dim,
                 global_episode, global_episode_reward, res_queue, log_queue):
        super(Worker, self).__init__()
        self.id = name
        self.name = 'w%02i' % name
        self.env = env
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.res_queue = res_queue
        self.log_queue = log_queue
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_critic_optimizer = global_critic_optimizer
        self.global_actor_optimizer = global_actor_optimizer
        self.local_actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.local_critic = ValueNet(state_dim, hidden_dim)

    def take_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.local_actor(state)
        
        current_node = int(state[1].item())  # obs[1]是当前节点
        valid_actions = list(self.env.network.successors(current_node))
        
        if not valid_actions:
            return current_node
        
        valid_logits = torch.tensor([logits[i].item() for i in valid_actions])
        dist = F.softmax(valid_logits, dim=0)
        probs = torch.distributions.Categorical(dist)
        action_idx = probs.sample().detach().item()
        action_idx = min(action_idx, len(valid_actions) - 1)
        action = valid_actions[action_idx]
        return action

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)

        discounted_rewards = [torch.sum(torch.FloatTensor([GAMMA**i for i in range(rewards[j:].size(0))]) \
             * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1)
        critic_loss = F.mse_loss(self.local_critic(states), value_targets.detach())

        logits = self.local_actor(states)
        dists = F.softmax(logits, dim=1)
        probs = torch.distributions.Categorical(dists)
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist + 1e-8)))
        entropy = torch.stack(entropy).sum()

        advantage = value_targets - self.local_critic(states)
        actor_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        actor_loss = actor_loss.mean() - entropy * 0.001

        self.global_critic_optimizer.zero_grad()
        critic_loss.backward()
        for local_params, global_params in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            global_params._grad = local_params._grad
        self.global_critic_optimizer.step()

        self.global_actor_optimizer.zero_grad()
        actor_loss.backward()
        for local_params, global_params in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            global_params._grad = local_params._grad
        self.global_actor_optimizer.step()
    
    def sync_with_global(self):
        self.local_critic.load_state_dict(self.global_critic.state_dict())
        self.local_actor.load_state_dict(self.global_actor.state_dict())

    def run(self):
        while self.global_episode.value < GLOBAL_MAX_EPISODE:
            # 开始新的episode（包含3个GBS的决策）
            state = self.env.reset(seed=self.id)
            episode_terminated = False
            transition_dict = {
                'states': [],
                'actions': [],
                'rewards': [],
            }
            
            # 在一个episode内完成3个GBS的决策
            while not episode_terminated and self.global_episode.value < GLOBAL_MAX_EPISODE:
                action = self.take_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['rewards'].append(reward)
                
                # 检查是否所有GBS都完成了
                if info['all_gbs_done']:
                    episode_terminated = True
                
                state = next_state
            
            # Episode结束：获取总奖励
            episodeReward = self.env.get_episode_reward()
            episode_info = self.env.get_episode_info()
            
            # 更新全局计数
            with self.global_episode.get_lock():
                self.global_episode.value += 1
            
            # 构建日志
            log_lines = []
            log_lines.append("=" * 80)
            log_lines.append(f"{self.name} | Episode: {self.global_episode.value} | Total Reward: {episodeReward:.2f}")
            log_lines.append(f"成功GBS: {episode_info['success_count']}/3")
            log_lines.append("-" * 80)
            
            # 每个GBS的详细信息
            for gbs_id in range(self.env.num_gbs):
                path = [gbs_id] + [p[1] for p in episode_info['gbs_paths'][gbs_id]]
                status = "SUCCESS" if episode_info['gbs_success'][gbs_id] else "FAILED"
                gbs_reward = episode_info['gbs_rewards'][gbs_id]
                
                log_lines.append(f"GBS{gbs_id}: {' -> '.join(map(str, path))} | "
                                f"Reward={gbs_reward:.2f} | {status}")
                
                # 详细步骤和奖励计算过程
                step_rewards = episode_info['gbs_step_rewards'][gbs_id]
                step_details = episode_info['gbs_step_reward_details'][gbs_id]
                
                for step_idx, (from_node, to_node) in enumerate(episode_info['gbs_paths'][gbs_id]):
                    node_type_from = self._get_node_type(from_node)
                    node_type_to = self._get_node_type(to_node)
                    step_reward = step_rewards[step_idx] if step_idx < len(step_rewards) else 0
                    
                    log_lines.append(f"  Step {step_idx+1}: {from_node}({node_type_from}) -> {to_node}({node_type_to}) | Reward: {step_reward:.3f}")
                    
                    # 奖励计算详情
                    if step_idx < len(step_details):
                        detail = step_details[step_idx]
                        reward_info = detail['reward_details']
                        
                        if reward_info['reason'] == 'normal_step':
                            log_lines.append(f"    奖励计算: 基础={reward_info['base_reward']:.3f}, "
                                           f"节点负载={reward_info['node_load_reward']:.3f}, "
                                           f"链路负载={reward_info['link_load_reward']:.3f}, "
                                           f"分布成本={reward_info['distribution_cost']:.3f}")
                            log_lines.append(f"    最终奖励: {reward_info['base_reward']:.3f} + "
                                           f"{reward_info['alpha_1']:.1f}*({reward_info['node_load_reward']:.3f}+{reward_info['link_load_reward']:.3f}) - "
                                           f"{reward_info['alpha_2']:.1f}*{reward_info['distribution_cost']:.3f} = {reward_info['total_reward']:.3f}")
                        elif reward_info['reason'] == 'reached_destination':
                            log_lines.append(f"    奖励计算: 到达目标服务器，奖励={reward_info['total_reward']:.3f}")
                        elif reward_info['reason'] == 'invalid_action':
                            log_lines.append(f"    奖励计算: 无效动作，惩罚={reward_info['total_reward']:.3f}")
                
                # 如果GBS失败，显示失败原因
                if not episode_info['gbs_success'][gbs_id]:
                    if len(step_details) > 0:
                        last_detail = step_details[-1]
                        if last_detail['reward_details']['reason'] == 'invalid_action':
                            log_lines.append(f"    失败原因: 无效动作")
                        else:
                            log_lines.append(f"    失败原因: 超过最大步数限制({self.env.max_steps_per_gbs}步)")
                    else:
                        log_lines.append(f"    失败原因: 无有效动作")
            
            log_lines.append("=" * 80)
            
            # 发送日志
            log_msg = '\n'.join(log_lines)
            self.log_queue.put(log_msg)
            
            # 控制台简要输出
            print(f"{self.name} | Ep: {self.global_episode.value} | "
                  f"Reward: {episodeReward:.2f} | Success: {episode_info['success_count']}/3")
            
            # 发送奖励
            self.res_queue.put(episodeReward)
            
            # 更新网络
            self.update(transition_dict)
            self.sync_with_global()
        
        # Worker结束
        self.res_queue.put(None)
        self.log_queue.put(None)
        print(f"{self.name} finished!")
    
    def _get_node_type(self, node_id):
        if node_id < 3:
            return "GBS"
        elif node_id < 7:
            return "Router"
        else:
            return "Server"


def test01():
    # 创建环境
    env = UAVRoutingEnv()
    
    numOfCPU = 8
    agent = A3Cagent(state_dim=env.obs_dim,
                     hidden_dim=256,
                     action_dim=env.action_space_n,
                     actor_lr=1e-3,
                     critic_lr=1e-3,
                     env=env,
                     numOfCPU=numOfCPU)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"A3C_MultiGBS_Training_{timestamp}.txt")
    
    # 开始信息
    header_info = []
    header_info.append("="*80)
    header_info.append("A3C多GBS无人机路由训练（Episode内3个GBS依次决策）")
    header_info.append("="*80)
    header_info.append(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header_info.append(f"配置: {numOfCPU}个Worker, 最大{GLOBAL_MAX_EPISODE}个Episodes")
    header_info.append(f"网络: 有向图, GBS->Router单向, Router间双向")
    header_info.append(f"规则: 1个Episode包含3个GBS的依次决策")
    header_info.append(f"      每个GBS最多{env.max_steps_per_gbs}步")
    header_info.append(f"      成功才更新网络负载，失败给予{env.failure_penalty}惩罚")
    header_info.append(f"日志文件: {log_filename}")
    header_info.append("="*80)
    header_info.append("")
    
    for line in header_info:
        print(line)
    
    # 启动训练
    [w.start() for w in agent.workers]
    returnLists1 = []
    all_logs = header_info.copy()
    finished_workers = 0
    log_finished_workers = 0
    
    # 收集奖励和日志
    while finished_workers < numOfCPU or log_finished_workers < numOfCPU:
        try:
            r = agent.res_queue.get(timeout=0.1)
            if r is not None:
                returnLists1.append(r)
            else:
                finished_workers += 1
        except Empty:
            pass
        
        try:
            log = agent.log_queue.get(timeout=0.1)
            if log is not None:
                all_logs.append(log)
                all_logs.append("")
            else:
                log_finished_workers += 1
        except Empty:
            pass
        
        if finished_workers >= numOfCPU and log_finished_workers >= numOfCPU:
            break
    
    [w.join() for w in agent.workers]
    print(f"\n训练完成! 总Episodes: {len(returnLists1)}")
    
    # 统计成功率
    # 从日志中统计（这里简化处理）
    summary_info = []
    summary_info.append("\n" + "="*80)
    summary_info.append("训练完成总结")
    summary_info.append("="*80)
    summary_info.append(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_info.append(f"总Episodes: {len(returnLists1)}")
    
    if len(returnLists1) > 0:
        summary_info.append(f"平均奖励: {np.mean(returnLists1):.2f}")
        summary_info.append(f"最大奖励: {np.max(returnLists1):.2f}")
        summary_info.append(f"最小奖励: {np.min(returnLists1):.2f}")
        summary_info.append(f"奖励标准差: {np.std(returnLists1):.2f}")
    
    summary_info.append("="*80)
    
    for line in summary_info:
        print(line)
    
    all_logs.extend(summary_info)
    
    # 保存日志
    print(f"\n保存训练日志到: {log_filename}")
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_logs))
    print("日志保存完成!")
    
    # 保存训练数据和绘图
    if len(returnLists1) > 0:
        ReturnList = []
        ReturnList.append(util.smooth([returnLists1], sm=50))
        labelList = ['A3C_MultiGBS_Routing']
        util.PlotReward(len(returnLists1), ReturnList, labelList, 'UAV_MultiGBS_Routing')
        np.save("A3C_MultiGBS_Routing_rewards.npy", returnLists1)
        print("训练数据已保存!")
    
    env.close()

if __name__ == '__main__':
    test01()
