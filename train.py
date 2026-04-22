"""
A3C 训练入口
用法: python train.py
"""
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.fl_routing_env import FLRoutingEnv
from src.envs.topology import TopologyConfig
from src.envs.delay_model import DelayConfig
from src.agents.a3c import A3CAgent

# ======================================================
# 实验参数（集中配置）
# ======================================================
CFG = {
    # 网络
    "num_gbs": 3,
    "num_routers": 4,
    "node_capacity": 50,
    "link_capacity": 100.0,
    # FL
    "model_size": 10.0,
    "t_agg": 0.5,
    # 时隙
    "delta_t": 5.0,
    "num_slots": 100,
    # 奖励
    "g_hop": -1.0,
    "alpha_1": 0.4,
    "alpha_2": 0.1,
    "w_delay": 0.2,
    "r_success": 20.0,
    "r_fail": -5.0,
    "max_steps_per_gbs": 50,
    # A3C
    "hidden_dim": 256,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "gamma": 0.98,
    "num_workers": 8,
    "max_episodes": 300,   # 快速验证用，正式实验改为2000+
    # 输出
    "output_dir": "outputs",
    "seed": 42,
}


def build_env_kwargs():
    return dict(
        topo_cfg=TopologyConfig(
            num_gbs=CFG["num_gbs"],
            num_routers=CFG["num_routers"],
            node_capacity=CFG["node_capacity"],
            link_capacity=CFG["link_capacity"],
        ),
        delay_cfg=DelayConfig(
            model_size=CFG["model_size"],
            t_agg=CFG["t_agg"],
        ),
        delta_t=CFG["delta_t"],
        num_slots=CFG["num_slots"],
        g_hop=CFG["g_hop"],
        alpha_1=CFG["alpha_1"],
        alpha_2=CFG["alpha_2"],
        w_delay=CFG["w_delay"],
        r_success=CFG["r_success"],
        r_fail=CFG["r_fail"],
        max_steps_per_gbs=CFG["max_steps_per_gbs"],
    )


def main():
    os.makedirs(CFG["output_dir"] + "/logs", exist_ok=True)
    os.makedirs(CFG["output_dir"] + "/figures", exist_ok=True)
    os.makedirs(CFG["output_dir"] + "/models", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 获取状态/动作维度
    env_kwargs = build_env_kwargs()
    _env = FLRoutingEnv(**env_kwargs)
    state_dim = _env.obs_dim
    action_dim = _env.action_space_n
    print(f"state_dim={state_dim}, action_dim={action_dim}")

    print("=" * 60)
    print("A3C FL路由训练（时隙化 + FL时延建模）")
    print(f"Workers={CFG['num_workers']}, Episodes={CFG['max_episodes']}")
    print(f"model_size={CFG['model_size']}Mb, delta_t={CFG['delta_t']}s")
    print("=" * 60)

    agent = A3CAgent(
        state_dim=state_dim,
        hidden_dim=CFG["hidden_dim"],
        action_dim=action_dim,
        actor_lr=CFG["actor_lr"],
        critic_lr=CFG["critic_lr"],
        num_workers=CFG["num_workers"],
        max_episodes=CFG["max_episodes"],
        gamma=CFG["gamma"],
        env_kwargs=env_kwargs,
    )

    results = agent.train()

    # 保存原始结果
    out_path = f"{CFG['output_dir']}/logs/a3c_results_{timestamp}.json"
    # T_up=inf 序列化处理
    for r in results:
        if r.get("T_up") == float('inf'):
            r["T_up"] = -1.0
        if r.get("T_round") == float('inf'):
            r["T_round"] = -1.0
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # 统计摘要
    rewards = [r["reward"] for r in results]
    t_ups = [r["T_up"] for r in results if r["T_up"] > 0]
    success_rates = [r["success"] / CFG["num_gbs"] for r in results]

    print("\n" + "=" * 60)
    print("训练完成")
    print(f"总Episodes: {len(results)}")
    print(f"平均奖励: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"平均成功率: {np.mean(success_rates):.3f}")
    if t_ups:
        print(f"平均T_up: {np.mean(t_ups):.3f}s ± {np.std(t_ups):.3f}s")
    print(f"结果保存至: {out_path}")
    print("=" * 60)

    # 保存numpy数组供绘图
    np.save(f"{CFG['output_dir']}/logs/a3c_rewards_{timestamp}.npy", np.array(rewards))
    if t_ups:
        np.save(f"{CFG['output_dir']}/logs/a3c_tup_{timestamp}.npy", np.array(t_ups))

    return results


if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
