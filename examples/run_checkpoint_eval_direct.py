import os
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from multiagent_traffic_light_grid_Worst import flow_params, env_name, create_env

# ====================================================
# 1️⃣ 修改 checkpoint 路径
# ====================================================
result_dir = "/home/sdc_joran/ray_results/grid_0_3x3_i3200_multiagent_2025-11-10_09-00-00"
checkpoint_num = "100"

# ====================================================
# 2️⃣ 初始化 Ray
# ====================================================
ray.init(num_cpus=1, ignore_reinit_error=True)

# ====================================================
# 3️⃣ 注册环境（使用你定义的 flow_params）
# ====================================================
register_env(env_name, create_env)
env = create_env()
obs = env.reset()
horizon = flow_params["env"].horizon

# ====================================================
# 4️⃣ 创建 PPOTrainer 并加载 checkpoint
# ====================================================
config = {
    "env": env_name,
    "num_workers": 0,
    "multiagent": {
        "policies": {
            "av": (None, env.observation_space, env.action_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "av",
    },
}
trainer = PPOTrainer(env=env_name, config=config)

checkpoint_path = os.path.join(result_dir, f"checkpoint_{checkpoint_num}", f"checkpoint-{checkpoint_num}")
print(f"Loading checkpoint from: {checkpoint_path}")
trainer.restore(checkpoint_path)

# ====================================================
# 5️⃣ Rollout 评估（冻结策略）
# ====================================================
print(">>> Starting frozen-policy simulation...")
for t in range(horizon):
    action = {}
    for agent_id in obs.keys():
        action[agent_id] = trainer.compute_action(obs[agent_id], policy_id="av")
    obs, reward, done, _ = env.step(action)
    if t % 50 == 0:
        avg_rew = np.mean(list(reward.values()))
        print(f"[t={t:04d}] mean reward = {avg_rew:.3f}")
    if done.get("__all__", False):
        print(f"Simulation finished early at step {t}")
        break

env.terminate()
print(">>> Simulation complete.")
