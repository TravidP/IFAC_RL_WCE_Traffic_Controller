import os
import gym
import numpy as np
import ray
import pandas as pd
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# =========== CHECKPOINT PATH ===========
default_policy_dir = (
    "/home/sdc_joran//ray_results/grid_0_3x3_robust_multiagent/"
    "PPO_DRMultiTrafficLightGridPOEnv-v1_c844a966_2025-11-26_10-17-51x_hrrw8b/"
    "checkpoint_4018"
) 

checkpoint_path = os.path.join(default_policy_dir, "checkpoint-4018")

# =========== Dummy env ===========
class DummyEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(79,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8)

    def reset(self):
        return {"av": np.zeros(79, np.float32)}

    def step(self, action_dict):
        return {"av": np.zeros(79)}, {"av": 0.0}, {"av": True, "__all__": True}, {"av": {}}

# =========== PPO Trainer config ===========
config = {
    "num_workers": 0,
    "env": DummyEnv,
    "model": {
        "fcnet_hiddens": [128, 64, 32],
        "fcnet_activation": "tanh",
        "vf_share_layers": False,
    },
    "multiagent": {
        "policies": {
            "av": (None, DummyEnv().observation_space, DummyEnv().action_space, {})
        },
        "policy_mapping_fn": lambda id: "av",
    },
}

# Ray init
ray.shutdown()
ray.init(ignore_reinit_error=True)

PPO = get_agent_class("PPO")
trainer = PPO(env=DummyEnv, config=config)

print("[INFO] Restoring checkpoint...")
trainer.restore(checkpoint_path)
print("[OK] Restore completed!")

policy = trainer.get_policy("av")
weights = policy.get_weights()

# ========== EXPORT ALL WEIGHTS INTO ONE CSV ==========
output_csv = "dr_marl_all_weights.csv"

rows = []  # store all rows to write

for key, w in weights.items():
    arr = np.array(w)

    # header row: layer name + shape
    rows.append([f"Layer: {key}", f"Shape: {arr.shape}"])

    # convert vector -> row
    if arr.ndim == 1:
        rows.append(arr.tolist())

    # matrix -> each row written separately
    elif arr.ndim == 2:
        for r in arr:
            rows.append(r.tolist())

    # empty line between layers
    rows.append([])

# write to CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, header=False, index=False)

print(f"\n[OK] All weights exported to: {output_csv}")
