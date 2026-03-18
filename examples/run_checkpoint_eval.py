import os
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from flow.utils.rllib import get_rllib_config, get_flow_params
from flow.utils.registry import make_create_env

# ====================================================
# 1️⃣ 修改以下路径为你自己的 checkpoint 位置
# ====================================================
result_dir = "/home/sdc_joran/ray_results/grid_0_3x3_i4800_multiagent/PPO_MultiTrafficLightGridPOEnv-v1_0a4a3ca0_2025-11-04_13-51-54_yhihbzy"
checkpoint_num = "3454"  # 比如 checkpoint_100

# ====================================================
# 2️⃣ 初始化 Ray
# ====================================================
ray.init(num_cpus=1, ignore_reinit_error=True)

# ====================================================
# 3️⃣ 从结果目录读取配置与环境参数
# ====================================================
config = get_rllib_config(result_dir)
flow_params = get_flow_params(config)
create_env, env_name = make_create_env(params=flow_params, version=0)
register_env(env_name, create_env)

# ====================================================
# 4️⃣ 创建 PPOTrainer 并加载 checkpoint
# ====================================================
trainer = PPOTrainer(env=env_name, config=config)
checkpoint_path = os.path.join(result_dir, f"checkpoint_{checkpoint_num}",
                               f"checkpoint-{checkpoint_num}")
print(f"Loading checkpoint: {checkpoint_path}")
trainer.restore(checkpoint_path)

# ====================================================
# 5️⃣ 创建环境并开始 rollout
# ====================================================
env = create_env()
obs = env.reset()
horizon = flow_params["env"].horizon

print(">>> Starting evaluation rollout with frozen checkpoint policy ...")

for t in range(horizon):
    action = {}
    for agent_id in obs.keys():
        # 从 checkpoint 中加载的策略计算动作
        action[agent_id] = trainer.compute_action(obs[agent_id], policy_id="av")

    obs, reward, done, _ = env.step(action)

    # 打印平均奖励或速度
    if t % 50 == 0:
        avg_rew = np.mean(list(reward.values()))
        print(f"[t={t:04d}] mean reward = {avg_rew:.3f}")

    if done.get("__all__", False):
        print(f"Simulation finished early at step {t}")
        break

env.terminate()
print(">>> Simulation complete.")
