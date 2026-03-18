import os
import gym
import numpy as np
import ray
import pandas as pd
from ray.rllib.agents.registry import get_agent_class

# =========== CHECKPOINT PATH ===========
default_policy_dir = (
    "/home/sdc_joran/ray_results/grid_0_3x3_worst_estimator_internal_inflow/"
    "PPO_WorstEstimatorTrafficEnv-v0_a4bd6d64_2025-11-19_16-51-42yf2h4ygu/"
    "checkpoint_10/"
)
checkpoint_path = os.path.join(default_policy_dir, "checkpoint-10")

# =========== Dummy single-agent env ===========
class DummyEnv(gym.Env):
    def __init__(self, config=None):
        # IMPORTANT: match the obs dimension of the WCE training (18 from the error message)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(16)

    def reset(self):
        # Return a dummy obs with the correct length
        return np.zeros(18, dtype=np.float32)

    def step(self, action):
        obs = np.zeros(18, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info

# =========== PPO Trainer config ===========
config = {
    "num_workers": 0,
    "env": DummyEnv,
    "model": {
        "fcnet_hiddens": [32, 32, 32],   # must match training config
        "fcnet_activation": "tanh",
        "vf_share_layers": False,
    },
    # no "multiagent" here!
}

ray.shutdown()
ray.init(ignore_reinit_error=True)

PPO = get_agent_class("PPO")
trainer = PPO(env=DummyEnv, config=config)

print("[INFO] Restoring checkpoint...")
trainer.restore(checkpoint_path)
print("[OK] Restore completed!")

policy = trainer.get_policy()  # default_policy
weights = policy.get_weights()

# ========== EXPORT ALL WEIGHTS INTO ONE CSV ==========
output_csv = "worst_case_estimator_weights_10.csv"

rows = []
for key, w in weights.items():
    arr = np.array(w)
    rows.append([f"Layer: {key}", f"Shape: {arr.shape}"])
    if arr.ndim == 1:
        rows.append(arr.tolist())
    elif arr.ndim == 2:
        for r in arr:
            rows.append(r.tolist())
    rows.append([])  # blank line

df = pd.DataFrame(rows)
df.to_csv(output_csv, header=False, index=False)
print(f"\n[OK] All weights exported to: {output_csv}")
