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

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.titlesize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.4,
    "legend.edgecolor": "0.4",
    "legend.facecolor": "white",
    "legend.fontsize": 15,
    "legend.labelspacing": 0.2,
})
# =========================
# 0. CONFIG – EDIT THIS
# =========================

# Root path shortcut
FLOW_HOME = "/home/sdc_joran/flow"

# Folders for each group (you can mix and match freely)
GROUP_DIRS = {
    0: os.path.join(FLOW_HOME, "eval_results_uniform"),
    1: os.path.join(FLOW_HOME, "eval_results_9_3"),     # Done
    2: os.path.join(FLOW_HOME, "eval_results_9_1"),     # Done
    3: os.path.join(FLOW_HOME, "eval_results_9_3"),     # Done
    4: os.path.join(FLOW_HOME, "eval_results_9_2"),     # Done
    5: os.path.join(FLOW_HOME, "eval_results_9_3"),     # Done
    6: os.path.join(FLOW_HOME, "eval_results_9_3"),     # Done
    7: os.path.join(FLOW_HOME, "eval_results_9_2"),     # Done
    8: os.path.join(FLOW_HOME, "eval_results_group7"),  # Done
    9: os.path.join(FLOW_HOME, "eval_results_9_1"),     # Done
}

# Controller names used in filenames
CONTROLLERS = ["baseline", "dr_marl"]

# Output folder for combined figures
OUTPUT_DIR = os.path.join(FLOW_HOME, "eval_results_combined")


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

def plot_groups(stats_per_group: Dict[int, GroupStats], title: str, ylabel: str, save_path: str,  legend_ncol: int | None = None):
    plt.figure(figsize=(10, 5))

    for g, st in sorted(stats_per_group.items()):
        # -------- choose which metric to plot --------
        if ylabel.lower().startswith("queue"):
            y_mean = st.queue_mean
            y_low  = st.queue_mean - st.queue_std   # or st.queue_min
            y_high = st.queue_mean + st.queue_std   # or st.queue_max
        else:
            y_mean = st.speed_mean
            y_low  = st.speed_mean - st.speed_std   # or st.speed_min
            y_high = st.speed_mean + st.speed_std   # or st.speed_max

        T = np.arange(len(y_mean))

        # main line
        line, = plt.plot(T, y_mean, label=f"Group {g}")
        color = line.get_color()

        # shaded error window
        plt.fill_between(T, y_low, y_high, alpha=0.2, color=color)

    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks()
    plt.yticks()
    if legend_ncol is None:
        plt.legend()
    else:
        plt.legend(ncol=legend_ncol)

    plt.xlim(0, len(y_mean) + 20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()



def plot_worst_group_comparison(stats_b: GroupStats, stats_d: GroupStats,
                                title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(10, 5))
    T_b = np.arange(len(stats_b.queue_mean))
    T_d = np.arange(len(stats_d.queue_mean))

    if ylabel.lower().startswith("queue"):
        line_b, = plt.plot(T_b, stats_b.queue_mean, label="Baseline Worst Group")
        color_b = line_b.get_color()
        plt.fill_between(T_b, stats_b.queue_min, stats_b.queue_max, alpha=0.2, color=color_b)

        line_d, = plt.plot(T_d, stats_d.queue_mean, label="DR-MARL Worst Group")
        color_d = line_d.get_color()
        plt.fill_between(T_d, stats_d.queue_min, stats_d.queue_max, alpha=0.2, color=color_d)
    else:
        line_b, = plt.plot(T_b, stats_b.speed_mean, label="Baseline Worst Group")
        color_b = line_b.get_color()
        plt.fill_between(T_b, stats_b.speed_min, stats_b.speed_max, alpha=0.2, color=color_b)

        line_d, = plt.plot(T_d, stats_d.speed_mean, label="DR-MARL Worst Group")
        color_d = line_d.get_color()
        plt.fill_between(T_d, stats_d.speed_min, stats_d.speed_max, alpha=0.2, color=color_d)

    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks()
    plt.yticks()
    plt.legend()
    Tmax = min(len(stats_b.queue_mean), len(stats_d.queue_mean))
    plt.xlim(0, Tmax + 20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
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

    plot_groups(baseline_stats, "Baseline MARL – Average Speed by Group",
                "Average Speed [m/s]", os.path.join(OUTPUT_DIR, "baseline_speed_by_group.pdf"),
                legend_ncol=3)

    plot_groups(dr_stats, "DR-MARL – Queue Length by Group",
                "Queue Length [veh]", os.path.join(OUTPUT_DIR, "drmarl_queue_by_group.pdf"),
                legend_ncol=3)

    plot_groups(dr_stats, "DR-MARL – Average Speed by Group",
                "Average Speed [m/s]", os.path.join(OUTPUT_DIR, "drmarl_speed_by_group.pdf"),
                legend_ncol=3)

    plot_groups(baseline_stats, "Baseline MARL – Queue Length by Group",
                "Queue Length [veh]", os.path.join(OUTPUT_DIR, "baseline_queue_by_group.pdf"),
                legend_ncol=3
                )

    # =========================
    # MANUAL WORST-GROUP COMPARISONS
    # =========================

    # --- Queue Length comparison ---
    b_queue_group = 7     # baseline group for queue comparison
    d_queue_group = 5     # DR-MARL group for queue comparison

    print(f"[INFO] manual queue comparison: Baseline Group {b_queue_group}, DR Group {d_queue_group}")

    plot_worst_group_comparison(
        baseline_stats[b_queue_group],
        dr_stats[d_queue_group],
        title=f"Queue Length comparison (Baseline Group {b_queue_group}, DR Group {d_queue_group})",
        ylabel="Queue Length [veh]",
        save_path=os.path.join(OUTPUT_DIR, "manual_worst_queue.pdf"),
    )

    # --- Speed comparison ---
    b_speed_group = 6     # baseline group for speed comparison
    d_speed_group = 5     # DR-MARL group for speed comparison

    print(f"[INFO] manual speed comparison: Baseline Group {b_speed_group}, DR Group {d_speed_group}")

    plot_worst_group_comparison(
        baseline_stats[b_speed_group],
        dr_stats[d_speed_group],
        title=f"Average Speed Comparison (baseline Group {b_speed_group}, DR Group {d_speed_group})",
        ylabel="Average Speed [m/s]",
        save_path=os.path.join(OUTPUT_DIR, "manual_worst_speed.pdf"),
    )


if __name__ == "__main__":
    main()
