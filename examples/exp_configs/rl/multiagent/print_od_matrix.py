#!/usr/bin/env python3
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt

# ===== Fixed orders you specified =====
ROW_ORDER = [
    "right0_0", "right0_1", "right0_2",
    "top0_3", "top1_3", "top2_3",
    "left3_2", "left3_1", "left3_0",
    "bot2_0", "bot1_0", "bot0_0",
]

COL_ORDER = [
    "left0_0", "left0_1", "left0_2",
    "bot0_3", "bot1_3", "bot2_3",
    "right3_2", "right3_1", "right3_0",
    "top2_0", "top1_0", "top0_0",
]

# ===== Your mapping (unchanged) =====
EDGE_ID_MAP = {
    # --- Outer edges ---
    # Left side
    "e_g1_n2": "bot2_0", "e_n2_g1": "top2_0",
    "e_g2_n1": "bot1_0", "e_n1_g2": "top1_0",
    "e_g4_n0": "bot0_0", "e_n0_g4": "top0_0",

    # Bottom
    "e_g3_n0": "right0_0", "e_n0_g3": "left0_0",
    "e_g5_n3": "right0_1", "e_n3_g5": "left0_1",
    "e_g6_n6": "right0_2", "e_n6_g6": "left0_2",

    # Right side
    "e_g7_n6": "top0_3", "e_n6_g7": "bot0_3",
    "e_g8_n7": "top1_3", "e_n7_g8": "bot1_3",
    "e_g10_n8": "top2_3", "e_n8_g10": "bot2_3",

    # Top
    "e_g9_n8": "left3_2", "e_n8_g9": "right3_2",
    "e_g11_n5": "left3_1", "e_n5_g11": "right3_1",
    "e_g0_n2": "left3_0", "e_n2_g0": "right3_0",

    # --- Inner edges ---
    # Column lanes
    "e_n2_n1": "left2_0", "e_n1_n2": "right2_0",
    "e_n1_n0": "left1_0", "e_n0_n1": "right1_0",
    "e_n5_n4": "left2_1", "e_n4_n5": "right2_1",
    "e_n4_n3": "left1_1", "e_n3_n4": "right1_1",
    "e_n8_n7": "left2_2", "e_n7_n8": "right2_2",
    "e_n7_n6": "left1_2", "e_n6_n7": "right1_2",

    # Row lanes
    "e_n0_n3": "bot0_1", "e_n3_n0": "top0_1",
    "e_n3_n6": "bot0_2", "e_n6_n3": "top0_2",
    "e_n1_n4": "bot1_1", "e_n4_n1": "top1_1",
    "e_n4_n7": "bot1_2", "e_n7_n4": "top1_2",
    "e_n2_n5": "bot2_1", "e_n5_n2": "top2_1",
    "e_n5_n8": "bot2_2", "e_n8_n5": "top2_2",
}

def _zero_matrix():
    return {o: {d: 0.0 for d in COL_ORDER} for o in ROW_ORDER}

def read_triplist_csv_to_od_rates(csv_path: str):
    """
    CSV columns: origin, destination, count (veh/h). Headers present.
    Origin/destination are e_* ids; map them to internal names via EDGE_ID_MAP.
    """
    od = _zero_matrix()
    unknown = set()
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        ok, dk, vk = r.fieldnames[:3]
        for row in r:
            o_raw = row[ok].strip()
            d_raw = row[dk].strip()
            try:
                val = float(row[vk])
            except Exception:
                continue
            o = EDGE_ID_MAP.get(o_raw)
            d = EDGE_ID_MAP.get(d_raw)
            if o in od and d in od[o]:
                od[o][d] += val
            else:
                unknown.add((o_raw, d_raw))
    return od, unknown

def print_matrix(od_rates: dict):
    def fmt(x):
        # print as integer if it's integer-ish, else 2 decimals
        return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"

    header = ["origin\\dest"] + COL_ORDER + ["row_sum"]
    print("\t".join(header))
    for o in ROW_ORDER:
        row_vals = [od_rates[o][d] for d in COL_ORDER]
        row_sum = sum(row_vals)
        print("\t".join([o] + [fmt(v) for v in row_vals] + [fmt(row_sum)]))
    # column sums + grand total
    col_sums = [sum(od_rates[o][d] for o in ROW_ORDER) for d in COL_ORDER]
    print("\t".join(["col_sum"] + [fmt(v) for v in col_sums] + [fmt(sum(col_sums))]))

def od_rates_to_episode_counts(od_rates: dict, horizon_s: int):
    """
    Integer counts per OD using Hamilton largest-remainder per row.
    Guarantees each row's episode total equals round(row_rate*h/3600).
    """
    counts = _zero_matrix()
    for o in ROW_ORDER:
        quotas = [(d, od_rates[o][d] * horizon_s / 3600.0) for d in COL_ORDER]
        floors = {d: math.floor(q) for d, q in quotas}
        rema  = [(d, quotas[i][1] - floors[d]) for i, (d, _) in enumerate(quotas)]
        target = int(round(sum(q for _, q in quotas)))
        curr = sum(floors.values())
        need = max(0, target - curr)
        rema.sort(key=lambda x: (-x[1], COL_ORDER.index(x[0])))
        alloc = floors.copy()
        for k in range(need):
            alloc[rema[k][0]] += 1
        counts[o] = alloc
    return counts

# ---------- NEW: plotting helpers ----------

def _matrix_to_2dlist(matrix: dict):
    """Convert {row: {col: val}} to 2D list in ROW_ORDER/COL_ORDER."""
    return [[matrix[o][d] for d in COL_ORDER] for o in ROW_ORDER]

# For heat map with colors
# def plot_matrix(matrix: dict, title: str, filename: str, value_fmt="{:.0f}"):
#     """
#     Save a heatmap of the matrix to 'filename'.
#     """
#     data = _matrix_to_2dlist(matrix)

#     fig, ax = plt.subplots(figsize=(8, 7))
#     im = ax.imshow(data)

#     # Axis labels
#     ax.set_xticks(range(len(COL_ORDER)))
#     ax.set_yticks(range(len(ROW_ORDER)))
#     ax.set_xticklabels(COL_ORDER, rotation=45, ha="right")
#     ax.set_yticklabels(ROW_ORDER)

#     ax.set_xlabel("Destination")
#     ax.set_ylabel("Origin")
#     ax.set_title(title)

#     # Annotate each cell with the value
#     for i, o in enumerate(ROW_ORDER):
#         for j, d in enumerate(COL_ORDER):
#             val = matrix[o][d]
#             ax.text(
#                 j, i,
#                 value_fmt.format(val),
#                 ha="center", va="center",
#                 fontsize=7,
#             )

#     fig.tight_layout()
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     # High resolution for presentations
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# only numbers
def plot_matrix(matrix: dict, title: str, filename: str, value_fmt="{:.0f}"):
    """
    Draw a clean table-style matrix:
    - no heatmap colors
    - grid lines
    - centered text
    - perfect for presentations
    """
    data = _matrix_to_2dlist(matrix)

    n_rows = len(ROW_ORDER)
    n_cols = len(COL_ORDER)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw an empty table (white background)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()  # make row 0 at top
    ax.axis("off")

    # Draw grid lines
    for i in range(n_rows + 1):
        ax.plot([0, n_cols], [i, i], color="black", linewidth=0.7)
    for j in range(n_cols + 1):
        ax.plot([j, j], [0, n_rows], color="black", linewidth=0.7)

    # Put numbers
    for i, o in enumerate(ROW_ORDER):
        for j, d in enumerate(COL_ORDER):
            val = matrix[o][d]
            ax.text(
                j + 0.5,
                i + 0.5,
                value_fmt.format(val),
                ha="center",
                va="center",
                fontsize=8,
            )

    # Column labels
    for j, d in enumerate(COL_ORDER):
        ax.text(
            j + 0.5, -0.3,
            d, ha="center", va="center",
            fontsize=9, rotation=45
        )

    # Row labels
    for i, o in enumerate(ROW_ORDER):
        ax.text(
            -0.3, i + 0.5,
            o, ha="right", va="center",
            fontsize=9
        )

    # Title
    ax.text(
        n_cols / 2, -1.2,
        title, ha="center", va="center",
        fontsize=14, fontweight="bold"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    CSV_PATH = "/home/sdc_joran/Athena_Data/data/aoi_od_pairs_group1.csv"  # adjust if needed
    HORIZON  = 900  # seconds, only used for the optional counts printout

    od_rates, unknown = read_triplist_csv_to_od_rates(CSV_PATH)
    if unknown:
        print(f"WARNING: {len(unknown)} OD pairs were outside the 12x12 outer-edge grid (ignored).")

    print("\n=== OD RATE MATRIX (veh/h) ===")
    print_matrix(od_rates)

    # Optional: also show the episode integer counts for the chosen horizon
    counts = od_rates_to_episode_counts(od_rates, HORIZON)
    print(f"\n=== OD EPISODE COUNTS for HORIZON={HORIZON}s ===")
    def fmt_int(x): return str(int(x))
    header = ["origin\\dest"] + COL_ORDER + ["row_total"]
    print("\t".join(header))
    for o in ROW_ORDER:
        row_vals = [counts[o][d] for d in COL_ORDER]
        print("\t".join([o] + [fmt_int(v) for v in row_vals] + [fmt_int(sum(row_vals))]))
    col_sums = [sum(counts[o][d] for o in ROW_ORDER) for d in COL_ORDER]
    print("\t".join(["col_total"] + [fmt_int(v) for v in col_sums] + [fmt_int(sum(col_sums))]))

    # ------- NEW: create images -------
    plot_matrix(od_rates,
                title="OD Rate Matrix (veh/h)",
                filename="od_rates_heatmap.png",
                value_fmt="{:.0f}")

    plot_matrix(counts,
                title=f"OD Episode Counts (H={HORIZON}s)",
                filename="od_episode_counts_heatmap.png",
                value_fmt="{:.0f}")

    print("\nSaved figures: 'od_rates_heatmap.png' and 'od_episode_counts_heatmap.png'")
