#!/usr/bin/env python3
"""
Re-plot evaluation results *only* from raw CSV files
where each group may live in a different folder.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0. CONFIG – EDIT THIS
# =========================

# Root path shortcut
FLOW_HOME = "/home/sdc_joran/flow"

# Folders for each group (you can mix and match freely)
GROUP_DIRS = {
    0: os.path.join(FLOW_HOME, "eval_results_9_3"),
    1: os.path.join(FLOW_HOME, "eval_results_9_1"),
    2: os.path.join(FLOW_HOME, "eval_results_9_3"),
    3: os.path.join(FLOW_HOME, "eval_results_9_2"),
    4: os.path.join(FLOW_HOME, "eval_results_9_3"),
    5: os.path.join(FLOW_HOME, "eval_results_9_3"),
    6: os.path.join(FLOW_HOME, "eval_results_9_2"),
    7: os.path.join(FLOW_HOME, "eval_results_group7"),
    8: os.path.join(FLOW_HOME, "eval_results_9_1"),
}

# Controller names used in filenames
CONTROLLERS = ["baseline", "dr_marl"]

# Output folder for combined figures
OUTPUT_DIR = os.path.join(FLOW_HOME, "eval_results_Sioux_Falls_plots")


# =========================
# 1. Data containers
# =========================

@dataclass
class GroupStats:
    queue_mean: np.ndarray
    queue_std: np.ndarray
    queue_min: np.ndarray
    queue_max: np.ndarray
    speed_mean: np.ndarray
    speed_std: np.ndarray
    speed_min: np.ndarray
    speed_max: np.ndarray
    queue_overall_mean: float
    speed_overall_mean: float


# =========================
# 2. Loading helpers
# =========================

def load_raw_matrix(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    t = df["t"].to_numpy()
    vals = df.loc[:, df.columns != "t"].to_numpy()  # (T, R)
    X = vals.T  # (R, T)
    return t, X


def load_group_stats(controller: str, group_index: int, folder: str) -> Tuple[np.ndarray, GroupStats]:
    """Load queue/speed CSVs for one (controller, group) from its folder."""
    q_path = os.path.join(folder, f"{controller}_group{group_index}_queue_raw.csv")
    v_path = os.path.join(folder, f"{controller}_group{group_index}_speed_raw.csv")

    if not os.path.exists(q_path) or not os.path.exists(v_path):
        raise FileNotFoundError(f"Missing files for {controller}, group {group_index} in {folder}")

    t_q, Q = load_raw_matrix(q_path)
    t_v, V = load_raw_matrix(v_path)
    t = t_q

    queue_mean, queue_std, queue_min, queue_max = Q.mean(0), Q.std(0), Q.min(0), Q.max(0)
    speed_mean, speed_std, speed_min, speed_max = V.mean(0), V.std(0), V.min(0), V.max(0)

    stats = GroupStats(
        queue_mean, queue_std, queue_min, queue_max,
        speed_mean, speed_std, speed_min, speed_max,
        float(Q.mean()), float(V.mean())
    )
    return t, stats


# =========================
# 3. Plotting helpers
# =========================

def plot_groups(stats_per_group: Dict[int, GroupStats], title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(10, 5))
    for g, st in sorted(stats_per_group.items()):
        T = np.arange(len(st.queue_mean))
        if ylabel.lower().startswith("queue"):
            plt.plot(T, st.queue_mean, label=f"group {g}")
        else:
            plt.plot(T, st.speed_mean, label=f"group {g}")

    plt.xlabel("simulation step", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_worst_group_comparison(stats_b: GroupStats, stats_d: GroupStats,
                                title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(10, 5))
    T_b = np.arange(len(stats_b.queue_mean))
    T_d = np.arange(len(stats_d.queue_mean))

    if ylabel.lower().startswith("queue"):
        line_b, = plt.plot(T_b, stats_b.queue_mean, label="MARL")
        color_b = line_b.get_color()
        plt.fill_between(T_b, stats_b.queue_min, stats_b.queue_max, alpha=0.2, color=color_b)

        line_d, = plt.plot(T_d, stats_d.queue_mean, label="DR-MARL")
        color_d = line_d.get_color()
        plt.fill_between(T_d, stats_d.queue_min, stats_d.queue_max, alpha=0.2, color=color_d)
    else:
        line_b, = plt.plot(T_b, stats_b.speed_mean, label="MARL")
        color_b = line_b.get_color()
        plt.fill_between(T_b, stats_b.speed_min, stats_b.speed_max, alpha=0.2, color=color_b)

        line_d, = plt.plot(T_d, stats_d.speed_mean, label="DR-MARL")
        color_d = line_d.get_color()
        plt.fill_between(T_d, stats_d.speed_min, stats_d.speed_max, alpha=0.2, color=color_d)

    plt.xlabel("simulation step", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def find_worst_group(stats_per_group: Dict[int, GroupStats]) -> int:
    return max(stats_per_group.items(), key=lambda kv: kv[1].queue_overall_mean)[0]


# =========================
# 4. Main
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_all: Dict[str, Dict[int, GroupStats]] = {}

    for ctrl in CONTROLLERS:
        ctrl_stats: Dict[int, GroupStats] = {}
        for g, folder in GROUP_DIRS.items():
            print(f"[LOAD] {ctrl}, group {g} from {folder}")
            t, st = load_group_stats(ctrl, g, folder)
            ctrl_stats[g] = st
        stats_all[ctrl] = ctrl_stats

    baseline_stats = stats_all["baseline"]
    dr_stats = stats_all["dr_marl"]

    # by-group plots
    plot_groups(baseline_stats, "Baseline MARL – queue length by group",
                "queue length [veh]", os.path.join(OUTPUT_DIR, "baseline_queue_by_group.png"))
    plot_groups(baseline_stats, "Baseline MARL – average speed by group",
                "average speed [m/s]", os.path.join(OUTPUT_DIR, "baseline_speed_by_group.png"))
    plot_groups(dr_stats, "DR-MARL – queue length by group",
                "queue length [veh]", os.path.join(OUTPUT_DIR, "drmarl_queue_by_group.png"))
    plot_groups(dr_stats, "DR-MARL – average speed by group",
                "average speed [m/s]", os.path.join(OUTPUT_DIR, "drmarl_speed_by_group.png"))

    # =========================
    # MANUAL WORST-GROUP COMPARISONS
    # =========================

    # --- Queue length comparison ---
    b_queue_group = 8     # baseline group for queue comparison
    d_queue_group = 8     # DR-MARL group for queue comparison

    print(f"[INFO] manual queue comparison: baseline g={b_queue_group}, DR g={d_queue_group}")

    plot_worst_group_comparison(
        baseline_stats[b_queue_group],
        dr_stats[d_queue_group],
        title=f"Queue length over time - MARL vs DR-MARL (Sioux Falls)",
        ylabel="queue length [veh]",
        save_path=os.path.join(OUTPUT_DIR, "manual_worst_queue.png"),
    )

    # --- Speed comparison ---
    b_speed_group = 8     # baseline group for speed comparison
    d_speed_group = 8     # DR-MARL group for speed comparison

    print(f"[INFO] manual speed comparison: baseline g={b_speed_group}, DR g={d_speed_group}")

    plot_worst_group_comparison(
        baseline_stats[b_speed_group],
        dr_stats[d_speed_group],
        title=f"Average speed over time - MARL vs DR-MARL (Sioux Falls)",
        ylabel="average speed [m/s]",
        save_path=os.path.join(OUTPUT_DIR, "manual_worst_speed.png"),
    )


if __name__ == "__main__":
    main()
