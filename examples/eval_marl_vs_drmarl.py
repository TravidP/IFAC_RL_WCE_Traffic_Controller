"""
Evaluation script for baseline MARL vs DR-MARL traffic light controllers.

- Exposes each controller to each of the 8 traffic demand groups (trip_00..07.csv)
  by forcing the WCE in DRMultiTrafficLightGridPOEnv to output a fixed one-hot
  weight vector for that group.
- For every (controller, group) pair it runs several rollouts and records:
    * network-wide queue length over time
    * network-wide average speed over time
- It then:
    * saves per-step means across rollouts
    * saves per-group averages over time (for your LaTeX table)
    * generates 6 figures:
        1. baseline: queue length over time, 8 groups
        2. baseline: avg speed over time, 8 groups
        3. DR-MARL: queue length over time, 8 groups
        4. DR-MARL: avg speed over time, 8 groups
        5. worst group comparison (baseline vs DR) on queue length
        6. worst group comparison (baseline vs DR) on avg speed
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Helper: load CSV weights
# =========================

def parse_rllib_csv_layers(path: str) -> Dict[str, np.ndarray]:
    """Parse a 'all_weights_in_one_file.csv'-style file into numpy arrays.

    The file is expected to be structured in blocks:

        Layer: av/fc_1/kernel,"Shape: (79, 128)",...
        w11, w12, ...
        w21, w22, ...
        ...
        <blank line>
        Layer: av/fc_1/bias,"Shape: (128,)",...
        b1, b2, ...

    Returns
    -------
    dict
        Mapping from layer name (e.g. 'av/fc_1/kernel') to np.ndarray with
        the given shape.
    """
    import re

    layers: Dict[str, np.ndarray] = {}          # layer name -> matrix
    current_name = None                         # name of the layer we’re currently reading
    current_shape: Tuple[int, ...] = None       # the shape for this layer (like (79, 128) or (128,)).
    buffer: List[float] = []                    # a flat list of all numbers we’ve read for this layer so far.

    def finalize():
        nonlocal current_name, current_shape, buffer, layers # modify these outer variables, not create new locals.
        if current_name is None:
            return # if not in the middle of the layer, nothing to do

        # manual product of shape dimensions (works in Python 3.7)
        # Compute how many values we need for this layer
        total = 1
        for d in current_shape:
            total *= d

        vals = buffer                            # list of all the floats we’ve collected for this layer
        # If any numbers are missing, pad with zeros; if too many, cut off extra
        if len(vals) < total:
            # pad with zeros if truncated
            vals = vals + [0.0] * (total - len(vals))
        elif len(vals) > total:
            # truncate if there is noise at the end
            vals = vals[:total]


        arr = np.asarray(vals, dtype=np.float32).reshape(current_shape) # Build the numpy array
        layers[current_name] = arr                                      # Store in layers dict
        # Reset state for next layer
        current_name = None
        current_shape = None
        buffer = []

    # Read the CSV file line by line
    with open(path, newline="") as f:
        reader = csv.reader(f)
        # Looping over the rows
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            
            # Detecting a layer header
            if first.startswith("Layer: "):
                # finish previous block
                finalize()

                # parse new layer header and shape
                current_name = first[len("Layer: ") :].strip()
                joined = ",".join(row[1:])
                m = re.search(r"\((.*?)\)", joined)
                if not m:
                    raise ValueError(f"Could not parse shape from row: {row}")
                parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
                current_shape = tuple(int(p) for p in parts)
            
            # Read weight numbers
            else:
                # numeric row for the current layer
                if current_name is None:
                    continue                # not in a layer block yet, skip
                for cell in row:
                    cell = cell.strip()
                    if not cell:
                        continue
                    try:
                        buffer.append(float(cell))
                    except ValueError:
                        # ignore non-numerical noise
                        continue

        # finalize last
        finalize()
    return layers


# =========================
# 2. Frozen shared TL policy
# =========================

class FrozenSharedTLPolicy:
    """Simple feed-forward NN that mimics the RLlib PPO policy.

    Assumes the following layers exist in the CSV:

        av/fc_1/kernel, av/fc_1/bias
        av/fc_2/kernel, av/fc_2/bias
        av/fc_3/kernel, av/fc_3/bias
        av/fc_out/kernel, av/fc_out/bias

    (value-function layers are ignored).
    """

    # initialize the layers from the given dict
    def __init__(self, layer_dict: Dict[str, np.ndarray]):
        self.W1 = layer_dict["av/fc_1/kernel"]
        self.b1 = layer_dict["av/fc_1/bias"]
        self.W2 = layer_dict["av/fc_2/kernel"]
        self.b2 = layer_dict["av/fc_2/bias"]
        self.W3 = layer_dict["av/fc_3/kernel"]
        self.b3 = layer_dict["av/fc_3/bias"]
        self.Wo = layer_dict["av/fc_out/kernel"]
        self.bo = layer_dict["av/fc_out/bias"]

        # Dimenions of the observation and actions spaces
        self.obs_dim = self.W1.shape[0]
        self.act_dim = self.Wo.shape[1]

    # compute the action logits for a single observation
    def _forward_logits(self, obs: np.ndarray) -> np.ndarray:
        """Return unnormalized action logits for a single observation."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim) # ensure shape (1, obs_dim)
        h1 = np.tanh(x @ self.W1 + self.b1)                             # first hidden layer
        h2 = np.tanh(h1 @ self.W2 + self.b2)                            # second hidden layer                       
        h3 = np.tanh(h2 @ self.W3 + self.b3)                            # third hidden layer
        logits = h3 @ self.Wo + self.bo                                 # output layer
        # shape: (1, act_dim)
        return logits[0]

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return an action index in [0, act_dim).

        If deterministic is False, sample from the softmax distribution.
        """
        logits = self._forward_logits(obs)                              # obtain logits
        if deterministic:
            return int(np.argmax(logits))                               # greedy: argmax

        # stochastic: softmax sampling
        z = logits - np.max(logits)
        probs = np.exp(z)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))


# =========================
# 3. Metric computation
# =========================

def compute_queue_and_speed(env) -> Tuple[float, float]:
    """Compute global queue length and average speed at the current step.

    Queue definition:
        - vehicles within `queue_window_m` of the end of their current edge
        - AND with speed <= `queue_speed_thresh`
    """
    k = env.k                                      # Flow kernel (access to vehicles, network, etc.)
    veh_ids = k.vehicle.get_ids()                  # all vehicle IDs in the network
    speeds: List[float] = []                       # store speeds to compute global avg

    add_params = env.env_params.additional_params  # extra env parameters
    q_window = add_params.get("queue_window_m", 300.0)      # queue window (m)
    v_thresh = add_params.get("queue_speed_thresh", 0.2)    # queue speed threshold (m/s)

    queue_len = 0                                  # total number of queued vehicles
    for vid in veh_ids:
        v = k.vehicle.get_speed(vid)               # vehicle speed
        if v is None:                              # skip vehicle for safety
            continue
        speeds.append(float(v))                    # collect for avg speed

        edge = k.vehicle.get_edge(vid)             # current edge of the vehicle
        if edge is None or edge.startswith(":"):
            # skip internal / junction edges
            continue

        try:
            edge_len = k.network.edge_length(edge) # edge length (m)
            pos = k.vehicle.get_position(vid)      # position along the edge (m)
        except Exception:
            # if SUMO lookup fails, skip this vehicle
            continue

        dist_to_end = edge_len - pos               # distance to end of edge (m)
        # count vehicle as queued if close to end AND slow
        if dist_to_end <= q_window and v <= v_thresh:
            queue_len += 1

    # compute average speed over all vehicles (0 if no vehicles)
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    return float(queue_len), avg_speed



# =========================
# 4. Constant WCE 'mock' to fix group
# =========================

class ConstantWCE:
    """Drop-in replacement for env._wce_model that forces a single group."""

    def __init__(self, group_index: int, num_groups: int = 8):
        # store which group to force (0..num_groups-1)
        self.group_index = int(group_index)
        # total number of groups (length of weight vector w)
        self.num_groups = int(num_groups)

    def predict(self, obs18: np.ndarray) -> np.ndarray:
        # create a zero vector of length num_groups
        w = np.zeros(self.num_groups, dtype=np.float32)
        # set exactly one entry (the chosen group) to 1
        w[self.group_index] = 1.0
        # return the one-hot weight vector
        return w



# =========================
# 5. Rollout loop
# =========================

@dataclass
class RolloutResult:
    queue_ts: List[List[float]]  # list over rollouts, each is list over time
    speed_ts: List[List[float]]  # same for average speeds


def run_rollouts_for_policy(
    create_env_fn,
    policy: FrozenSharedTLPolicy,
    group_index: int,
    num_rollouts: int,
    horizon: int,
    deterministic: bool = True,
) -> RolloutResult:
    # collected time series over *successful* rollouts
    all_queue: List[List[float]] = []
    all_speed: List[List[float]] = []

    # keep going until we have num_rollouts *successful* runs
    rollouts_done = 0
    max_attempts = num_rollouts * 10  # safety cap to avoid infinite loops

    while rollouts_done < num_rollouts and max_attempts > 0:
        max_attempts -= 1

        env = create_env_fn()  # fresh env instance

        # fix OD group for this rollout (disable Dirichlet/WCE randomness)
        env.forced_group_index = group_index

        # Also override WCE if it is used internally
        try:
            add_params = env.env_params.additional_params
            group_csvs = add_params.get("group_csvs", [])
            num_groups = len(group_csvs) if group_csvs else 8
        except Exception:
            num_groups = 8
        if hasattr(env, "_wce_model"):
            env._wce_model = ConstantWCE(group_index=group_index, num_groups=num_groups)

        obs = env.reset()  # start new episode

        done = {"__all__": False}
        t = 0
        q_series: List[float] = []  # queue length per time step
        v_series: List[float] = []  # avg speed per time step

        # track whether episode ended too early (crash, etc.)
        crashed_or_early = False

        # run the rollout up to horizon steps
        while t < horizon:
            # build multi-agent action dict
            actions = {}
            for agent_id, o in obs.items():
                actions[agent_id] = policy.act(o, deterministic=deterministic)

            # advance env by one step
            obs, rewards, done, info = env.step(actions)

            # compute and record metrics
            q, v = compute_queue_and_speed(env)
            q_series.append(q)
            v_series.append(v)

            t += 1

            if done.get("__all__", False):
                # if done before horizon, treat as crashed / early termination
                if t < horizon:
                    crashed_or_early = True
                break

        # clean up environment
        if hasattr(env, "terminate"):
            env.terminate()
        elif hasattr(env, "close"):
            env.close()

        # only keep runs that reached full horizon
        if not crashed_or_early and t >= horizon:
            all_queue.append(q_series)
            all_speed.append(v_series)
            rollouts_done += 1
            print(f"[EVAL] accepted rollout {rollouts_done}/{num_rollouts} for group {group_index} (t={t}s)")
        else:
            print(f"[EVAL] discarding rollout (ended at t={t}s, horizon={horizon}) for group {group_index}")

    # warn if we couldn't obtain enough non-crash rollouts
    if rollouts_done < num_rollouts:
        print(f"[EVAL] WARNING: only obtained {rollouts_done}/{num_rollouts} non-crash rollouts for group {group_index}")

    return RolloutResult(queue_ts=all_queue, speed_ts=all_speed)





# =========================
# 6. Aggregation helpers
# =========================

def pad_and_stack(series_list: List[List[float]]) -> np.ndarray:
    """Pad 1D series with their final value and stack into 2D array."""
    # find maximum length over all series
    max_len = max(len(s) for s in series_list)
    # allocate 2D array [num_series, max_len]
    arr = np.zeros((len(series_list), max_len), dtype=np.float32)
    for i, s in enumerate(series_list):
        if not s:
            continue
        # copy existing values
        arr[i, : len(s)] = s
        # pad any remaining positions with the last value
        if len(s) < max_len:
            arr[i, len(s) :] = s[-1]
    return arr


@dataclass
class GroupStats:
    # per-time-step stats
    queue_mean: np.ndarray
    queue_std: np.ndarray
    queue_min: np.ndarray
    queue_max: np.ndarray

    speed_mean: np.ndarray
    speed_std: np.ndarray
    speed_min: np.ndarray
    speed_max: np.ndarray

    # overall averages (for the table)
    queue_overall_mean: float
    speed_overall_mean: float



def save_raw_timeseries(
    result: RolloutResult,
    controller_name: str,
    group_index: int,
    output_dir: str,
):
    """Save full per-rollout time series to CSV.

    For each (controller, group) we write two files:

        {controller_name}_group{g}_queue_raw.csv
        {controller_name}_group{g}_speed_raw.csv

    Each file has shape (T rows) x (1 + R columns):

        t, rollout_0, rollout_1, ..., rollout_{R-1}
    """
    # Make rectangular arrays [R, T], padding if needed
    Q = pad_and_stack(result.queue_ts)   # queue values
    V = pad_and_stack(result.speed_ts)   # speed values

    R, T = Q.shape                      # R = num_rollouts, T = horizon
    t_axis = np.arange(T, dtype=int)    # time indices 0..T-1

    # Queue CSV
    fname_q = os.path.join(
        output_dir, f"{controller_name}_group{group_index}_queue_raw.csv"
    )
    with open(fname_q, "w", newline="") as f:
        writer = csv.writer(f)
        # header: t, rollout_0, ..., rollout_{R-1}
        header = ["t"] + [f"rollout_{i}" for i in range(R)]
        writer.writerow(header)
        # one row per time step
        for ti in range(T):
            row = [t_axis[ti]] + [Q[r, ti] for r in range(R)]
            writer.writerow(row)

    # Speed CSV
    fname_v = os.path.join(
        output_dir, f"{controller_name}_group{group_index}_speed_raw.csv"
    )
    with open(fname_v, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + [f"rollout_{i}" for i in range(R)]
        writer.writerow(header)
        for ti in range(T):
            row = [t_axis[ti]] + [V[r, ti] for r in range(R)]
            writer.writerow(row)

    print(
        f"[EVAL] saved raw time series for {controller_name}, "
        f"group {group_index} to:\n  {fname_q}\n  {fname_v}"
    )


def summarize_group(result: RolloutResult) -> GroupStats:
    Q = pad_and_stack(result.queue_ts)  # shape (n_rollouts, T)
    V = pad_and_stack(result.speed_ts)

    # per-time-step stats across rollouts
    queue_mean = Q.mean(axis=0)
    queue_std  = Q.std(axis=0)
    queue_min  = Q.min(axis=0)
    queue_max  = Q.max(axis=0)

    speed_mean = V.mean(axis=0)
    speed_std  = V.std(axis=0)
    speed_min  = V.min(axis=0)
    speed_max  = V.max(axis=0)

    return GroupStats(
        queue_mean=queue_mean,
        queue_std=queue_std,
        queue_min=queue_min,
        queue_max=queue_max,
        speed_mean=speed_mean,
        speed_std=speed_std,
        speed_min=speed_min,
        speed_max=speed_max,
        queue_overall_mean=float(Q.mean()),
        speed_overall_mean=float(V.mean()),
    )




# =========================
# 7. Plotting
# =========================

def plot_groups(
    stats_per_group: Dict[int, GroupStats],
    title: str,
    ylabel: str,
    save_path: str,
):
    # create a new figure (one controller, all groups)
    plt.figure(figsize=(10, 5))
    # loop over all groups (sorted by group index)
    for g, st in sorted(stats_per_group.items()):
        T = np.arange(len(st.queue_mean))  # time axis 0..T-1

        # decide which metric to plot based on ylabel
        if ylabel.lower().startswith("queue"):
            # plot mean queue curve for this group (no shading)
            plt.plot(T, st.queue_mean, label=f"group {g}")
        else:
            # plot mean speed curve for this group (no shading)
            plt.plot(T, st.speed_mean, label=f"group {g}")

    # bigger axis labels and tick labels
    plt.xlabel("simulation step", fontsize=18)   # x-axis: time steps (seconds)
    plt.ylabel(ylabel, fontsize=18)             # y-axis label (queue length or speed)
    plt.title(title, fontsize=18)               # plot title
    plt.xticks(fontsize=18)                     # tick label size
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14)                             # show one line per group
    plt.tight_layout()                          # adjust layout to fit labels
    plt.savefig(save_path)                      # save figure to file
    plt.close()                                 # close figure to free memory


def plot_worst_group_comparison(
    stats_baseline: GroupStats,
    stats_dr: GroupStats,
    title: str,
    ylabel: str,
    save_path: str,
):
    # create a new figure (compare worst baseline vs worst DR-MARL)
    plt.figure(figsize=(10, 5))

    # time axes for baseline and DR-MARL (usually same length)
    T_b = np.arange(len(stats_baseline.queue_mean))
    T_d = np.arange(len(stats_dr.queue_mean))

    # choose queue or speed metric based on ylabel
    if ylabel.lower().startswith("queue"):
        # baseline worst group: mean queue + min–max band
        line_b, = plt.plot(T_b, stats_baseline.queue_mean, label="baseline worst group")
        color_b = line_b.get_color()
        plt.fill_between(
            T_b,
            stats_baseline.queue_min,
            stats_baseline.queue_max,
            alpha=0.2,
            color=color_b,
        )

        # DR-MARL worst group: mean queue + min–max band
        line_d, = plt.plot(T_d, stats_dr.queue_mean, label="DR-MARL worst group")
        color_d = line_d.get_color()
        plt.fill_between(
            T_d,
            stats_dr.queue_min,
            stats_dr.queue_max,
            alpha=0.2,
            color=color_d,
        )
    else:
        # baseline worst group: mean speed + min–max band
        line_b, = plt.plot(T_b, stats_baseline.speed_mean, label="baseline worst group")
        color_b = line_b.get_color()
        plt.fill_between(
            T_b,
            stats_baseline.speed_min,
            stats_baseline.speed_max,
            alpha=0.2,
            color=color_b,
        )

        # DR-MARL worst group: mean speed + min–max band
        line_d, = plt.plot(T_d, stats_dr.speed_mean, label="DR-MARL worst group")
        color_d = line_d.get_color()
        plt.fill_between(
            T_d,
            stats_dr.speed_min,
            stats_dr.speed_max,
            alpha=0.2,
            color=color_d,
        )

    # bigger axis labels and tick labels
    plt.xlabel("simulation step", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()







# =========================
# 8. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_csv",
        type=str,
        required=True,
        help="CSV with weights for the baseline MARL TL controller.",
    )
    parser.add_argument(
        "--dr_csv",
        type=str,
        required=True,
        help="CSV with weights for the DR-MARL TL controller.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=3,
        help="Rollouts per (controller, group) pair.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16 * 600,
        help="Max steps per rollout (should match EnvParams.horizon).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to store figures and summary CSV.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy actions (argmax logits). Default: stochastic softmax.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Import the robust experiment config and its create_env
    from exp_configs.rl.multiagent import multiagent_traffic_light_grid_robust as robust_cfg
    import gym

    create_env_fn = lambda: gym.make(robust_cfg.env_name)

    # Load both policies
    layers_base = parse_rllib_csv_layers(args.baseline_csv)
    layers_dr = parse_rllib_csv_layers(args.dr_csv)

    baseline_policy = FrozenSharedTLPolicy(layers_base)
    dr_policy = FrozenSharedTLPolicy(layers_dr)

    controllers = {
        "baseline": baseline_policy,
        "dr_marl": dr_policy,
    }

    # Evaluate each controller on each of the 8 groups
    all_stats: Dict[str, Dict[int, GroupStats]] = {}

    for name, policy in controllers.items():
        stats_for_controller: Dict[int, GroupStats] = {}
        for g in range(1):
            print(f"Evaluating controller={name}, group={g} ...")
            res = run_rollouts_for_policy(
                create_env_fn=create_env_fn,
                policy=policy,
                group_index=g,
                num_rollouts=args.num_rollouts,
                horizon=args.horizon,
                deterministic=args.deterministic,
            )

            # NEW: save full per-rollout time series for this (controller, group)
            save_raw_timeseries(
                result=res,
                controller_name=name,
                group_index=g,
                output_dir=args.output_dir,
            )

            # Existing: aggregate to per-time-step means and overall means
            stats_for_controller[g] = summarize_group(res)
        all_stats[name] = stats_for_controller


    # ==== 6 figures ====

    # 1–2: baseline, 8 groups
    plot_groups(
        all_stats["baseline"],
        title="Baseline MARL – queue length by group",
        ylabel="queue length [veh]",
        save_path=os.path.join(args.output_dir, "baseline_queue_by_group.png"),
    )
    plot_groups(
        all_stats["baseline"],
        title="Baseline MARL – average speed by group",
        ylabel="average speed [m/s]",
        save_path=os.path.join(args.output_dir, "baseline_speed_by_group.png"),
    )

    # 3–4: DR-MARL, 8 groups
    plot_groups(
        all_stats["dr_marl"],
        title="DR-MARL – queue length by group",
        ylabel="queue length [veh]",
        save_path=os.path.join(args.output_dir, "drmarl_queue_by_group.png"),
    )
    plot_groups(
        all_stats["dr_marl"],
        title="DR-MARL – average speed by group",
        ylabel="average speed [m/s]",
        save_path=os.path.join(args.output_dir, "drmarl_speed_by_group.png"),
    )

    # Worst group per controller (by queue_overall_mean)
    def find_worst(stats_dict: Dict[int, GroupStats]) -> Tuple[int, GroupStats]:
        worst_g = max(
            stats_dict.items(), key=lambda kv: kv[1].queue_overall_mean
        )[0]
        return worst_g, stats_dict[worst_g]

    worst_g_base, worst_stats_base = find_worst(all_stats["baseline"])
    worst_g_dr, worst_stats_dr = find_worst(all_stats["dr_marl"])

    print(f"Worst baseline group (by queue length): {worst_g_base}")
    print(f"Worst DR-MARL group (by queue length): {worst_g_dr}")

    # 5–6: comparisons
    plot_worst_group_comparison(
        worst_stats_base,
        worst_stats_dr,
        title=f"Worst groups – queue length (base g={worst_g_base}, DR g={worst_g_dr})",
        ylabel="queue length [veh]",
        save_path=os.path.join(args.output_dir, "worst_groups_queue.png"),
    )
    plot_worst_group_comparison(
        worst_stats_base,
        worst_stats_dr,
        title=f"Worst groups – average speed (base g={worst_g_base}, DR g={worst_g_dr})",
        ylabel="average speed [m/s]",
        save_path=os.path.join(args.output_dir, "worst_groups_speed.png"),
    )

    # Save summary averages (for LaTeX table)
    summary_csv = os.path.join(args.output_dir, "summary_averages.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["controller", "group", "queue_overall_mean", "speed_overall_mean"]
        )
        for ctrl_name, stats_dict in all_stats.items():
            for g, st in sorted(stats_dict.items()):
                writer.writerow(
                    [ctrl_name, g, st.queue_overall_mean, st.speed_overall_mean]
                )

    print(f"Saved summary averages to {summary_csv}")
    print(f"Figures saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
