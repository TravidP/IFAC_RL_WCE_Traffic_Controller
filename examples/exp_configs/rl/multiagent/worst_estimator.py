"""
worst_estimator.py
Evaluate OD groups in multiagent_traffic_light_grid environment,
simulate 15 minutes each, and find the worst-performing traffic group.
"""

import os
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from flow.utils.registry import make_create_env
from flow.core.params import EnvParams, SumoParams, InitialConfig, NetParams
from flow.networks import TrafficLightGridNetwork
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.controllers.Routes import load_od_csv_matrix
from flow.controllers.Routes import ROW_ORDER, COL_ORDER, mix_od_matrices
import time

# === Config ===
SIM_TIME = 900          # 15 minutes = 900 seconds
GROUP_CSV_DIR = "data/od_groups"   # directory containing multiple CSVs
MODEL_CHECKPOINT = "metadata/checkpoint_latest"  # trained RL policy path
RESULTS_LOG = "results_worst_estimator.csv"

# === Reward weight ===
ALPHA_SPEED = 1.0
BETA_WAIT = 1.0


def evaluate_group(env_creator, trainer, group_csv):
    """Run one 15-min simulation for a given OD group and compute reward."""
    print(f"\n=== Evaluating group: {group_csv} ===")
    od_matrix = load_od_csv_matrix(group_csv)
    env = env_creator()

    obs = env.reset()
    total_wait = 0.0
    total_speed = 0.0
    veh_count = 0

    for step in range(int(SIM_TIME / env.sim_step)):
        actions = {}
        for agent_id, obs_i in obs.items():
            action = trainer.compute_action(obs_i, policy_id="default_policy")
            actions[agent_id] = action

        obs, rewards, dones, infos = env.step(actions)

        # Collect metrics
        veh_ids = env.k.vehicle.get_ids()
        speeds = np.array(env.k.vehicle.get_speed(veh_ids))
        waits = np.array(env.k.vehicle.get_accumulated_waiting_time(veh_ids))

        if len(veh_ids) > 0:
            total_speed += np.mean(speeds)
            total_wait += np.mean(waits)
            veh_count += 1

        if dones["__all__"]:
            break

    env.close()

    avg_speed = total_speed / max(veh_count, 1)
    avg_wait = total_wait / max(veh_count, 1)

    reward = ALPHA_SPEED * avg_speed - BETA_WAIT * avg_wait

    print(f"[Result] Avg Speed: {avg_speed:.2f} m/s, Avg Wait: {avg_wait:.2f}s, Reward: {reward:.3f}")
    return reward, avg_speed, avg_wait


def run_worst_case_search():
    """Iterate over all group CSVs and find the worst-case distribution."""
    # === Load trained model ===
    print(f"Loading PPO model from {MODEL_CHECKPOINT}...")
    trainer = PPOTrainer(config={"framework": "torch"})
    trainer.restore(MODEL_CHECKPOINT)

    # === Build environment ===
    create_env, _ = make_create_env({
        "env_name": "multiagent_traffic_light_grid",
        "version": 0
    })
    env_creator = create_env

    results = []

    for csv_file in sorted(os.listdir(GROUP_CSV_DIR)):
        if not csv_file.endswith(".csv"):
            continue
        csv_path = os.path.join(GROUP_CSV_DIR, csv_file)
        reward, speed, wait = evaluate_group(env_creator, trainer, csv_path)
        results.append((csv_file, reward, speed, wait))

    # sort and show
    results.sort(key=lambda x: x[1])  # sort by reward (ascending)
    print("\n=== Summary of all groups ===")
    for name, r, s, w in results:
        print(f"{name:25s}  reward={r:.3f}  speed={s:.2f}  wait={w:.2f}")

    # save results
    np.savetxt(RESULTS_LOG, results, fmt="%s,%.4f,%.4f,%.4f",
               header="group,reward,avg_speed,avg_wait", comments='')

    worst = results[0]
    print(f"\n💀 Worst-case group: {worst[0]}  (Reward={worst[1]:.3f})")


if __name__ == "__main__":
    start = time.time()
    run_worst_case_search()
    print(f"\nTotal runtime: {(time.time()-start)/60:.1f} min")
