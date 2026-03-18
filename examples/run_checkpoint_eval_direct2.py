import os
import sys
import time
import importlib.util
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer


# ========== 1) 你可以只改这两行 ==========
RESULT_DIR = "/home/sdc_joran/ray_results/grid_0_3x3_i4800_multiagent/PPO_MultiTrafficLightGridPOEnv-v1_0a4a3ca0_2025-11-04_13-51-54_yhihbzy"
CHECKPOINT_NUM = "3454"
# ========================================

# ========== 2) 自动定位 multiagent_traffic_light_grid_Worst.py ==========
CANDIDATES = [
    "/home/sdc_joran/flow/examples/exp_configs/rl/multiagent/multiagent_traffic_light_grid_Worst.py"
]


# 允许通过环境变量手工指定
ENV_HINT = os.environ.get("WORST_ENV_FILE")
if ENV_HINT:
    CANDIDATES.insert(0, ENV_HINT)

ENV_FILE = None
for p in CANDIDATES:
    if os.path.isfile(p):
        ENV_FILE = p
        break

if ENV_FILE is None:
    raise FileNotFoundError(
        "找不到 multiagent_traffic_light_grid_Worst.py，请把文件放到当前目录或 ~/flow/examples，"
        "或设置环境变量 WORST_ENV_FILE 指向该文件的绝对路径。"
    )

print(f"[INFO] 使用环境文件: {ENV_FILE}")

def _import_by_path(py_path, module_name="worst_env_module"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

worst_mod = _import_by_path(ENV_FILE)  # 动态导入
# 期望 worst_mod 里有 flow_params, create_env, env_name
flow_params = worst_mod.flow_params
create_env = worst_mod.create_env
env_name = worst_mod.env_name

# ========== 3) 一些显示/兼容设置，可选 ==========
# 打开 GUI（如果想要）
try:
    flow_params["sim"].render = True
except Exception:
    pass

# 让评估时使用 evaluate 奖励（可选）
try:
    flow_params["env"].evaluate = True
except Exception:
    pass

# ========== 4) 注册环境并创建 Trainer（不训练，只恢复权重）==========
ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=True)

register_env(env_name, create_env)
env = create_env()
obs = env.reset()
horizon = flow_params["env"].horizon

# RLlib 旧版 TF 策略兼容：明确指定 framework=tf，并手动声明 multiagent
config = {
    "env": env_name,
    "num_workers": 0,
    # 旧 Flow 通常是 TF
    "framework": "tf",
    "multiagent": {
        # 这里用 None 让 RLlib 用默认 PPO TF Policy 占位（加载 checkpoint 后会覆盖权重）
        "policies": {
            "av": (None, env.observation_space, env.action_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "av",
    },
    # 禁止任何训练相关采样并行（本就 num_workers=0 了）
    "log_level": "WARN",
}

trainer = PPOTrainer(env=env_name, config=config)

ckpt_path = os.path.join(RESULT_DIR, f"checkpoint_{CHECKPOINT_NUM}", f"checkpoint-{CHECKPOINT_NUM}")
print(f"[INFO] 加载 checkpoint: {ckpt_path}")
trainer.restore(ckpt_path)

# ========== 5) 冻结策略执行仿真 + 指标统计 ==========
print("[INFO] 开始冻结策略评估仿真 ...")
vel_hist = []
final_outflows, final_inflows = [], []

t0 = time.time()
for t in range(horizon):
    # 收集速度用于统计
    try:
        vehicles = env.unwrapped.k.vehicle
        speeds = vehicles.get_speed(vehicles.get_ids())
        if speeds:
            vel_hist.append(np.mean(speeds))
    except Exception:
        pass

    # 计算动作（不训练）
    action = {aid: trainer.compute_action(obs[aid], policy_id="av") for aid in obs.keys()}
    obs, reward, done, _ = env.step(action)

    if t % 50 == 0:
        avg_rew = np.mean(list(reward.values())) if reward else 0.0
        print(f"[t={t:04d}] mean reward = {avg_rew:.3f}")

    if done.get("__all__", False):
        print(f"[INFO] 提前结束 at step {t}")
        break

# 统计最后 500 秒的出入流（若可用）
try:
    vehicles = env.unwrapped.k.vehicle
    outflow = vehicles.get_outflow_rate(500)
    inflow = vehicles.get_inflow_rate(500)
    final_outflows.append(outflow)
    final_inflows.append(inflow)
except Exception:
    pass

env.terminate()

# ========== 6) 打印结果摘要 ==========
print("\n==== Summary ====")
if vel_hist:
    print(f"Mean speed (m/s): mean={np.mean(vel_hist):.3f}, std={np.std(vel_hist):.3f}")
else:
    print("Mean speed: N/A")

if final_inflows and final_outflows:
    te = (final_outflows[0] / final_inflows[0]) if final_inflows[0] > 1e-6 else 0.0
    print(f"Outflow (veh/hr): {final_outflows[0]:.2f}")
    print(f"Inflow  (veh/hr): {final_inflows[0]:.2f}")
    print(f"Throughput efficiency: {te:.3f}")
else:
    print("Outflow/Inflow: N/A")

print(f"Wall time: {time.time()-t0:.1f}s")
print("[INFO] 评估完成。")
