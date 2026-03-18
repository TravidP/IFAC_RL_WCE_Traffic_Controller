# worst_estimator_policy.py
# Use your TRAINED traffic-light PPO policy to control signals during simulation
# while collecting data to train a separate worst-case estimator NN (MLP).
#
# CLI:
#   1) Collect dataset:
#      python worst_estimator_policy.py collect \
#         --groups_dir /path/to/od_groups \
#         --out data/wce_with_policy.npz \
#         --policy_ckpt /path/to/rllib/checkpoint_XXXX
#
#   2) Train estimator:
#      python worst_estimator_policy.py train \
#         --data data/wce_with_policy.npz \
#         --ckpt runs/wce.pt
#
#   3) Select worst group (no sim; estimator forward pass):
#      python worst_estimator_policy.py select \
#         --ckpt runs/wce.pt \
#         --groups_dir /path/to/od_groups
#
# Notes:
# - This script REBUILDS a Flow environment per-CSV using the SAME structure
#   as your multiagent_traffic_light_grid config, but swaps the OD matrix/inflows.
# - During simulation, signals are controlled by your loaded PPO policy (policy_id="av").

import os
import argparse
import time
from typing import Dict, List, Tuple
import numpy as np

# --- RLlib (load trained PPO) ---
from ray.rllib.agents.ppo import PPOTrainer

# --- Flow core pieces / your helpers ---
from flow.utils.registry import make_create_env
from flow.networks import TrafficLightGridNetwork
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams, TrafficLightParams
from flow.controllers import SimCarFollowingController, SimLaneChangeController
from flow.controllers.routing_controllers import OuterEdgeODRouter
from fnmatch import fnmatch

# --- Your routes & OD utilities (from Routes.py) ---
from flow.controllers.Routes import (
    ROW_ORDER, COL_ORDER, Group_routes_G1,
    read_triplist_csv_to_od_rates, _normalize_weights, load_od_csv_matrix
)

# --- Your TLS preset (hold-all-red) ---
from examples.exp_configs.rl.multiagent.tls_presets import build_two_lane_hold_tls

# --- Import your env module to reuse its constants like RAW_NODE_COORDS if desired ---
# If your file lives elsewhere, adjust this import path accordingly.
from examples.exp_configs.rl.multiagent import multiagent_traffic_light_grid as tlgrid

# --- PyTorch worst-estimator (separate NN; DOES NOT control traffic lights) ---
import torch
import torch.nn as nn
import torch.optim as optim

# ===================== General sim/metric config =====================

HORIZON_S = 900            # 15 minutes
SIM_STEP  = 1.0            # keep aligned with your env (you set sim_step=1)
ALPHA_SPEED = 1.0          # reward = ALPHA_SPEED*avg_speed - BETA_WAIT*avg_wait
BETA_WAIT  = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Grid/network constants (match your env config) ===
N_ROWS = 3
N_COLS = 3
INNER_LENGTH = 300
SHORT_LENGTH = 300
LONG_LENGTH = 100
V_ENTER = 13.89            # 50 km/h

# === Lanes / TLS ===
H_LANES = 4
V_LANES = 4
SWITCH_TIME = 5.0          # as in your env additional_params
DISCRETE_ACTIONS = True    # <- if your trained policy is continuous, set to False

# ===================== Build flow_params for a given OD-CSV =====================

def _build_inflows_from_csv(csv_path: str, horizon_s: int) -> InFlows:
    """Recreate your per-OD inflow logic from the pasted env file."""
    od_rates = read_triplist_csv_to_od_rates(csv_path)  # veh/hour per (origin,dest)
    inflow = InFlows()
    for origin in ROW_ORDER:
        for dest in COL_ORDER:
            rate = float(od_rates[origin][dest])
            if rate <= 0:
                continue

            variants = _normalize_weights(Group_routes_G1.get(origin, {}).get(dest, []))
            if not variants:
                # single stream; router will shortest-path to dest
                period = 3600.0 / rate
                inflow.add(
                    veh_type="human",
                    edge=origin,
                    period=period,
                    name=f"OD_{origin}_to_{dest}",
                    depart_lane="best",
                    depart_speed=V_ENTER,
                    departPos="free",
                    begin=1.0, end=horizon_s
                )
            else:
                for k, v in enumerate(variants):
                    var_rate = rate * float(v["weight"])
                    if var_rate <= 0:
                        continue
                    period = 3600.0 / var_rate
                    inflow.add(
                        veh_type="human",
                        edge=origin,
                        period=period,
                        name=f"OD_{origin}_to_{dest}_k{k}",
                        depart_lane="best",
                        depart_speed=V_ENTER,
                        departPos="free",
                        begin=1.0, end=horizon_s
                    )
    return inflow


def _build_vehicles() -> VehicleParams:
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,
            tau=1.0,
            sigma=0.5,
            speed_mode="right_of_way",
        ),
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(OuterEdgeODRouter, {
            "seed": 42,
            "use_hardcoded": True,
            "hardcoded_routes": Group_routes_G1,
            "use_precomputed_cycle": True,
            "debug_origin": "right0_0",
        }),
        num_vehicles=0
    )
    return vehicles


def _build_flow_params_for_group(csv_path: str) -> dict:
    """Construct a flow_params dict for THIS CSV group, mirroring your env file."""
    inflow = _build_inflows_from_csv(csv_path, HORIZON_S)
    tl_logic = build_two_lane_hold_tls(
        n_rows=N_ROWS,
        n_cols=N_COLS,
        vertical_lanes=2,   # your TLS preset expects 2 here (phase size calc)
        horizontal_lanes=2
    )

    flow_params = dict(
        exp_tag=f"grid_{N_ROWS}x{N_COLS}_group_{os.path.basename(csv_path)}",
        env_name=MultiTrafficLightGridPOEnv,
        network=TrafficLightGridNetwork,
        simulator='traci',
        tls=tl_logic,
        sim=SumoParams(
            restart_instance=True,
            sim_step=SIM_STEP,
            render=False,
            print_warnings=False,
        ),
        env=EnvParams(
            horizon=HORIZON_S,
            sims_per_step=1,
            additional_params={
                "target_velocity": 50,
                "switch_time": SWITCH_TIME,
                "num_observed": 4,
                "discrete": DISCRETE_ACTIONS,
                "tl_type": "controlled",        # <=== KEY: controlled by PPO policy
                "num_local_edges": 4,
                "num_local_lights": 4,
                "queue_window_m": 50.0,
                "queue_speed_thresh": 0.2,
                "queue_cap_per_lane": 10,
                "queue_penalty_gain": 0.2,
                "spawn_from_rou": False,
            },
        ),
        net=NetParams(
            inflows=inflow,
            additional_params={
                "speed_limit": V_ENTER,
                "horizontal_lanes": H_LANES,
                "vertical_lanes": V_LANES,
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLS,
                    "cars_left": 1, "cars_right": 1, "cars_top": 1, "cars_bot": 1,
                },
                "turn_speed": 8.9,
                "node_coordinates": getattr(tlgrid, "RAW_NODE_COORDS", None),  # reuse your real coords
                "normalize_to_center0": True,
            },
        ),
        veh=_build_vehicles(),
        initial=InitialConfig(spacing='custom', shuffle=True),
    )
    return flow_params


# ===================== Observations & metrics =====================

def _edges_into_node(env, node_idx: int) -> List[str]:
    return env.network.node_mapping[node_idx][1]

def _compute_obs_vector(env) -> np.ndarray:
    """
    Per-intersection features:
      [avg_lane_speed_i (normalized by max limit), avg_lane_density_i]
    Concatenate over all intersections -> shape (2 * N_ROWS * N_COLS,)
    """
    num_nodes = N_ROWS * N_COLS
    feats = []
    max_speed_lim = max(env.k.network.speed_limit(e) for e in env.k.network.get_edge_list())
    max_speed_lim = max(float(max_speed_lim), 1e-6)

    for i in range(num_nodes):
        edges = _edges_into_node(env, i)
        veh_speeds = []
        total_len, total_veh = 0.0, 0
        for e in edges:
            L = float(env.k.network.edge_length(e) or 1.0)
            vids = env.k.vehicle.get_ids_by_edge(e)
            if vids:
                speeds = env.k.vehicle.get_speed(vids)
                veh_speeds.extend(speeds)
                total_veh += len(vids)
            total_len += L

        avg_spd = (np.mean(veh_speeds) / max_speed_lim) if veh_speeds else 0.0
        dens = min(1.0, total_veh / max(total_len, 1e-6))
        feats.extend([float(avg_spd), float(dens)])

    return np.array(feats, dtype=np.float32)


def _policy_actions(trainer: PPOTrainer, obs_dict: Dict[str, np.ndarray], env) -> Dict[str, np.ndarray]:
    """Build multi-agent actions from the trained policy (policy_id='av')."""
    actions = {}
    for aid, ob in obs_dict.items():
        act = trainer.compute_action(ob, policy_id="av")  # <-- your env uses 'av'
        actions[aid] = act
    return actions


def run_episode_with_policy(flow_params: dict, trainer: PPOTrainer, sim_seconds: int) -> Tuple[float, float]:
    """Create env for given flow_params, run sim_seconds with PPO policy controlling TLS."""
    create_env, _ = make_create_env(params=flow_params, version=0)
    env = create_env()
    obs = env.reset()

    total_speed, total_wait, frames = 0.0, 0.0, 0
    steps = int(sim_seconds / SIM_STEP)

    for _ in range(steps):
        acts = _policy_actions(trainer, obs, env)
        obs, rews, dones, infos = env.step(acts)

        veh_ids = env.k.vehicle.get_ids()
        if len(veh_ids) > 0:
            speeds = np.array(env.k.vehicle.get_speed(veh_ids), dtype=np.float32)
            waits  = np.array(env.k.vehicle.get_accumulated_waiting_time(veh_ids), dtype=np.float32)
            total_speed += float(np.mean(speeds))
            total_wait  += float(np.mean(waits))
            frames += 1

        if dones.get("__all__", False):
            break

    env.close()
    avg_speed = total_speed / max(frames, 1)
    avg_wait  = total_wait  / max(frames, 1)
    return avg_speed, avg_wait


# ===================== Dataset collection (with PPO policy) =====================

def collect_dataset_with_policy(groups_dir: str, out_path: str, policy_ckpt: str,
                                episodes_per_group: int = 1) -> None:
    """
    For each OD CSV in groups_dir:
      - Rebuild env with that CSV (controlled by PPO policy)
      - Run 15 min to get targets (avg_speed, avg_wait, reward)
      - Snapshot a global obs vector X at reset time
      - Save dataset: X, A(one-hot), Y
    """
    csv_paths = sorted([os.path.join(groups_dir, f) for f in os.listdir(groups_dir) if f.endswith(".csv")])
    if not csv_paths:
        raise ValueError(f"No CSV files found in {groups_dir}")

    # Load trained PPO policy
    trainer = PPOTrainer(config={"framework": "torch"})
    trainer.restore(policy_ckpt)
    print(f"[collect] Using PPO policy checkpoint: {policy_ckpt}")

    Xs, As, Ys, names = [], [], [], []
    n_groups = len(csv_paths)
    t0 = time.time()

    for gi, path in enumerate(csv_paths):
        name = os.path.basename(path)
        print(f"\n=== Group {gi+1}/{n_groups}: {name} ===")

        # Build flow_params for this group
        fp = _build_flow_params_for_group(path)

        # Snapshot obs vector at reset (cheap)
        create_env, _ = make_create_env(params=fp, version=0)
        env = create_env(); _ = env.reset()
        x = _compute_obs_vector(env)
        env.close()

        # Run the 15-min policy-controlled episode for targets
        avg_speed, avg_wait = run_episode_with_policy(fp, trainer, HORIZON_S)
        reward = ALPHA_SPEED * avg_speed - BETA_WAIT * avg_wait

        a = np.zeros(n_groups, dtype=np.float32); a[gi] = 1.0

        Xs.append(x); As.append(a); Ys.append([avg_speed, avg_wait, reward]); names.append(name)
        print(f"[collect] speed={avg_speed:.2f} m/s, wait={avg_wait:.2f} s, reward={reward:.3f}")

    X = np.stack(Xs, axis=0).astype(np.float32)
    A = np.stack(As, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)
    names = np.array(names)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, A=A, Y=Y, names=names)
    print(f"\n[collect] Saved dataset to {out_path}  (X{X.shape}, A{A.shape}, Y{Y.shape}) in {(time.time()-t0)/60:.1f} min")


# ===================== Estimator (MLP) =====================

class EstimatorMLP(nn.Module):
    def __init__(self, x_dim: int, a_dim: int, hidden=(256, 128, 64)):
        super().__init__()
        layers, in_dim = [], x_dim + a_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 3)  # [avg_speed, avg_wait, reward]

    def forward(self, x, a):
        z = torch.cat([x, a], dim=-1)
        return self.head(self.backbone(z))


def train_estimator(data_npz: str, ckpt_out: str, epochs=50, lr=1e-3, batch=64, val_split=0.2, seed=0):
    data = np.load(data_npz, allow_pickle=True)
    X, A, Y = data["X"].astype(np.float32), data["A"].astype(np.float32), data["Y"].astype(np.float32)
    n, x_dim, a_dim = X.shape[0], X.shape[1], A.shape[1]

    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_val = int(n * val_split)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr, Atr, Ytr = X[tr_idx], A[tr_idx], Y[tr_idx]
    Xva, Ava, Yva = X[val_idx], A[val_idx], Y[val_idx]

    model = EstimatorMLP(x_dim, a_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    def batches(X_, A_, Y_):
        m = X_.shape[0]; order = np.random.permutation(m)
        for s in range(0, m, batch):
            b = order[s:s+batch]
            yield (torch.from_numpy(X_[b]).to(DEVICE),
                   torch.from_numpy(A_[b]).to(DEVICE),
                   torch.from_numpy(Y_[b]).to(DEVICE))

    best = 1e9
    os.makedirs(os.path.dirname(ckpt_out), exist_ok=True)
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0
        for xb, ab, yb in batches(Xtr, Atr, Ytr):
            opt.zero_grad()
            pred = model(xb, ab)
            loss = mse(pred, yb)
            loss.backward(); opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= max(1, Xtr.shape[0])

        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xva).to(DEVICE)
            ab = torch.from_numpy(Ava).to(DEVICE)
            yb = torch.from_numpy(Yva).to(DEVICE)
            val = float(mse(model(xb, ab), yb).item())

        print(f"[train] epoch {ep:03d}  train MSE={tr_loss:.4f}  val MSE={val:.4f}")
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(),
                        "x_dim": x_dim, "a_dim": a_dim}, ckpt_out)
            print(f"  ↳ saved best to {ckpt_out} (val={best:.4f})")

    print(f"\n[train] done. best val MSE={best:.4f}")


def load_estimator(ckpt: str) -> EstimatorMLP:
    ck = torch.load(ckpt, map_location=DEVICE)
    m = EstimatorMLP(ck["x_dim"], ck["a_dim"]).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def select_worst(ckpt_estimator: str, groups_dir: str):
    """
    Predict reward for each single group (one-hot A) and choose minimum.
    For a more faithful snapshot, we optionally build a quick X per group by
    constructing an env and computing the observation vector at reset.
    """
    names = sorted([f for f in os.listdir(groups_dir) if f.endswith(".csv")])
    if not names:
        raise ValueError(f"No CSV files found in {groups_dir}")

    est = load_estimator(ckpt_estimator)

    results = []
    for i, name in enumerate(names):
        # Build X from a quick env snapshot to reflect geometry/lane setup
        fp = _build_flow_params_for_group(os.path.join(groups_dir, name))
        create_env, _ = make_create_env(params=fp, version=0)
        env = create_env(); _ = env.reset()
        x_np = _compute_obs_vector(env)
        env.close()

        X = torch.from_numpy(x_np[None, :]).float().to(DEVICE)
        A = np.zeros((1, len(names)), dtype=np.float32); A[0, i] = 1.0
        A = torch.from_numpy(A).to(DEVICE)

        with torch.no_grad():
            pred = est(X, A)[0].cpu().numpy()
        avg_speed, avg_wait, reward = pred.tolist()
        print(f"[predict] {name:25s}  reward={reward:.3f}  speed={avg_speed:.2f}  wait={avg_wait:.2f}")
        results.append((name, reward))

    results.sort(key=lambda t: t[1])
    print(f"\n💀 Worst group (predicted): {results[0][0]}  reward={results[0][1]:.3f}")


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Worst-Case Estimator (uses trained PPO policy during sim).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("collect", help="Collect dataset by simulating each OD-CSV with trained PPO policy.")
    ap_c.add_argument("--groups_dir", required=True)
    ap_c.add_argument("--out", required=True)
    ap_c.add_argument("--policy_ckpt", required=True, help="Path to RLlib PPO checkpoint (policy id 'av').")
    ap_c.add_argument("--episodes_per_group", type=int, default=1)

    ap_t = sub.add_parser("train", help="Train the worst-case estimator network.")
    ap_t.add_argument("--data", required=True)
    ap_t.add_argument("--ckpt", required=True)
    ap_t.add_argument("--epochs", type=int, default=50)
    ap_t.add_argument("--lr", type=float, default=1e-3)
    ap_t.add_argument("--batch", type=int, default=64)
    ap_t.add_argument("--val_split", type=float, default=0.2)
    ap_t.add_argument("--seed", type=int, default=0)

    ap_s = sub.add_parser("select", help="Use estimator to pick the worst group (no sim).")
    ap_s.add_argument("--ckpt", required=True)
    ap_s.add_argument("--groups_dir", required=True)

    args = ap.parse_args()

    if args.cmd == "collect":
        collect_dataset_with_policy(args.groups_dir, args.out, args.policy_ckpt, episodes_per_group=args.episodes_per_group)
    elif args.cmd == "train":
        train_estimator(args.data, args.ckpt, epochs=args.epochs, lr=args.lr, batch=args.batch,
                        val_split=args.val_split, seed=args.seed)
    elif args.cmd == "select":
        select_worst(args.ckpt, args.groups_dir)
    else:
        raise ValueError("Unknown subcommand")

if __name__ == "__main__":
    main()
