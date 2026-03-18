# ===== OD utilities & deterministic plan builder =====
from collections import defaultdict
import math

# Define the entries of the matrix
# Fixed origin (rows) and destination (columns) orders
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

# Map the data edges to flow's edges
# Your provided mapping (module-level so it’s built once)
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

# Initialize the matrix
def _ordered_zero_matrix() -> dict:
    """Return {origin: {dest: 0}} in the fixed ROW_ORDER × COL_ORDER."""
    return {o: {d: 0.0 for d in COL_ORDER} for o in ROW_ORDER}

# Read the data from the csv file and add a value of zero to missing OD pairs
def read_triplist_csv_to_od_rates(csv_path: str) -> dict:
    """
    Read a trip list CSV with columns: 'origin', 'destination', 'count'.
    Map e_* edge ids to internal names, aggregate to {origin: {dest: veh/h}}.
    Missing pairs are filled with 0.
    """
    import csv
    od = _ordered_zero_matrix()
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            o_raw = row[r.fieldnames[0]].strip()
            d_raw = row[r.fieldnames[1]].strip()
            val   = float(row[r.fieldnames[2]])
            o = EDGE_ID_MAP.get(o_raw, o_raw)
            d = EDGE_ID_MAP.get(d_raw, d_raw)
            if o in od and d in od[o]:
                od[o][d] += val
    return od

# Change veh/hour to veh/episode
def od_rates_to_episode_counts(od_rates: dict, horizon_s: int) -> dict:
    """
    Deterministic integer counts per OD for the episode using Hamilton rounding
    per origin row so row sums are preserved exactly.
    """
    counts = _ordered_zero_matrix()
    for o in ROW_ORDER:
        # quotas for this row
        quotas = [(d, od_rates[o][d] * horizon_s / 3600.0) for d in COL_ORDER]
        floors = {d: math.floor(q) for d, q in quotas}
        rems   = [(d, (q - floors[d])) for d, q in quotas]
        target = int(round(sum(q for _, q in quotas)))
        curr   = sum(floors.values())
        # allocate remaining by largest remainders (stable tie-breaker = COL_ORDER)
        need = max(0, target - curr)
        rems.sort(key=lambda x: (-x[1], COL_ORDER.index(x[0])))
        alloc = floors.copy()
        for k in range(need):
            alloc[rems[k][0]] += 1
        counts[o] = alloc
    return counts

# --- OD CSV mixing utilities (for worst-case estimator) -------------------
import csv, math, os
from typing import List, Dict

# Assumes you already have ROW_ORDER and COL_ORDER in this module.
# Values in CSV are floats (counts or rates). The first column is the origin id.
# The header row has destination ids (ideally matching COL_ORDER).

def load_od_csv_matrix(path: str) -> Dict[str, Dict[str, float]]:
    """
    Load one OD matrix from CSV into {origin: {dest: value}}.
    - Header: dest ids (first cell can be empty/'origin'); remaining headers are COL_ORDER-aligned.
    - Each row: first cell = origin id; remaining cells = float values.
    Missing cells are treated as 0.0. Unknown origins/dests are ignored.
    """
    mat = {o: {d: 0.0 for d in COL_ORDER} for o in ROW_ORDER}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        # Map header cols to dests
        # If header includes COL_ORDER exactly, align by name; else fall back to position.
        # Build index mapping: col_idx -> dest_id or None.
        dest_map = {}
        for j, h in enumerate(header[1:], start=1):
            hj = (h or "").strip()
            if hj in COL_ORDER:
                dest_map[j] = hj
            else:
                # positional fallback
                pos = j - 1
                if 0 <= pos < len(COL_ORDER):
                    dest_map[j] = COL_ORDER[pos]
        for row in reader:
            if not row:
                continue
            o = (row[0] or "").strip()
            if o not in ROW_ORDER:
                continue
            for j in range(1, len(row)):
                d = dest_map.get(j)
                if not d:
                    continue
                try:
                    val = float(row[j])
                except Exception:
                    val = 0.0
                mat[o][d] = val
    return mat

def normalize_alphas(alphas: List[float]) -> List[float]:
    """Clamp negatives to 0 and normalize so sum(alpha)=1 (or all zeros -> even split)."""
    nonneg = [max(0.0, float(a)) for a in alphas]
    s = sum(nonneg)
    if s <= 0.0:
        # even split (avoid divide-by-zero)
        n = len(nonneg)
        return [1.0/n]*n if n > 0 else []
    return [a/s for a in nonneg]

def mix_od_matrices(mats: List[Dict[str, Dict[str, float]]],
                    alphas: List[float]) -> Dict[str, Dict[str, float]]:
    """
    Weighted mixture: OD_mix = Σ_i alpha_i * OD_i, element-wise over (origin,dest).
    Returns a matrix in the same dict-of-dicts shape as inputs.
    """
    α = normalize_alphas(alphas)
    assert len(α) == len(mats), "alphas and mats length mismatch"
    out = {o: {d: 0.0 for d in COL_ORDER} for o in ROW_ORDER}
    for w, M in zip(α, mats):
        if w == 0.0:  # skip
            continue
        for o in ROW_ORDER:
            row = M.get(o, {})
            for d in COL_ORDER:
                out[o][d] += w * float(row.get(d, 0.0))
    return out



# ==== Define routings for each OD pair ====
Group_routes_G1 = {
  "right0_0": {
    "top0_0": [
      {"weight": 1, "edges": ["right0_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1, "edges": ["right0_0", "right1_0", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1, "edges": ["right0_0", "right1_0", "right2_0", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1, "edges": ["right0_0", "right1_0", "right2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "right2_0", "bot2_1", "right3_1"]},
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "bot1_1", "right2_1", "right3_1"]},
      {"weight": 1/3, "edges": ["right0_0", "bot0_1", "right1_1", "right2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["right0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["right0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1/3, "edges": ["right0_0", "bot0_1", "bot0_2", "right1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["right0_0", "bot0_1", "right1_1", "bot1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["right0_0", "right1_0", "bot1_1", "bot1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1, "edges": ["right0_0", "bot0_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1, "edges": ["right0_0", "bot0_1", "bot0_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1, "edges": ["right0_0", "bot0_1", "left0_1"]},
    ],
    "left0_0":[
      {"weight":1/3, "edges": ["right0_0", "left0_0"]},
      {"weight":1/3, "edges": ["right0_0", "bot0_1", "top0_1", "left0_0"]},
      {"weight":1/3, "edges": ["right0_0", "right1_0", "left1_0", "left0_0"]},
        ]
  },

  "right0_1": {
    "left0_0": [
      {"weight": 1, "edges": ["right0_1", "top0_1", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1, "edges": ["right0_1", "top0_1", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/2, "edges": ["right0_1", "top0_1", "right1_0", "top1_0"]},
      {"weight": 1/2, "edges": ["right0_1", "right1_1", "top1_1", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1/3, "edges": ["right0_1", "top0_1", "right1_0", "right2_0", "top2_0"]},
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "top1_1", "right2_0", "top2_0"]},
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "right2_1", "top2_1", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1/3, "edges": ["right0_1", "top0_1", "right1_0", "right2_0", "right3_0"]},
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "top1_1", "right2_0", "right3_0"]},
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "right2_1", "top2_1", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1, "edges": ["right0_1", "right1_1", "right2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "right2_1", "bot2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "bot1_2", "right2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["right0_1", "bot0_2", "right1_2", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "right2_1", "bot2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["right0_1", "right1_1", "bot1_2", "right2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["right0_1", "bot0_2", "right1_2", "right2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1/2, "edges": ["right0_1", "bot0_2", "right1_2", "bot1_3"]},
      {"weight": 1/2, "edges": ["right0_1", "right1_1", "bot1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1, "edges": ["right0_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1, "edges": ["right0_1", "bot0_2", "left0_2"]},
    ],
    "left0_1":[
      {"weight": 1/4, "edges": ["right0_1", "bot0_2", "top0_2", "left0_1"]},
      {"weight": 1/4, "edges": ["right0_1", "top0_1", "bot0_1", "left0_1"]},
      {"weight": 1/4, "edges": ["right0_1", "right1_1", "left1_1", "left0_1"]},
      {"weight": 1/4, "edges": ["right0_1", "left0_1"]},
    ]
  },

  "right0_2": {
    "left0_1": [
      {"weight": 1, "edges": ["right0_2", "top0_2", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1, "edges": ["right0_2", "top0_2", "top0_1", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1, "edges": ["right0_2", "top0_2", "top0_1", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "right1_1", "top1_1", "top1_0"]},
      {"weight": 1/3, "edges": ["right0_2", "right1_2", "top1_2", "top1_1", "top1_0"]},
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "top0_1", "right1_0", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "right1_1", "right2_1", "top2_1", "top2_0"]},
      {"weight": 1/3, "edges": ["right0_2", "right1_2", "right2_2", "top2_2", "top2_1", "top2_0"]},
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "top0_1", "right1_0", "right2_0", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "right1_1", "right2_1", "top2_1", "right3_0"]},
      {"weight": 1/3, "edges": ["right0_2", "right1_2", "right2_2", "top2_2", "top2_1", "right3_0"]},
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "top0_1", "right1_0", "right2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1/3, "edges": ["right0_2", "right1_2", "right2_2", "top2_2", "right3_1"]},
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "right1_1", "right2_1", "right3_1"]},
      {"weight": 1/3, "edges": ["right0_2", "right1_2", "top1_2", "right2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1, "edges": ["right0_2", "right1_2", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1, "edges": ["right0_2", "right1_2", "right2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1, "edges": ["right0_2", "right1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1, "edges": ["right0_2", "bot0_3"]},
    ],
    "left0_2":[
      {"weight": 1/3, "edges": ["right0_2", "left0_2"]},
      {"weight": 1/3, "edges": ["right0_2", "right1_2", "left1_2", "left0_2"]},
      {"weight": 1/3, "edges": ["right0_2", "top0_2", "bot0_2", "left0_2"]},
    ]
  },

  "top0_3": {
    "left0_2": [
      {"weight": 1, "edges": ["top0_3", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1, "edges": ["top0_3", "top0_2", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1, "edges": ["top0_3", "top0_2", "top0_1", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1, "edges": ["top0_3", "top0_2", "top0_1", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/3, "edges": ["top0_3", "top0_2", "right1_1", "top1_1", "top1_0"]},
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "top1_2", "top1_1", "top1_0"]},
      {"weight": 1/3, "edges": ["top0_3", "top0_2", "top0_1", "right1_0", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "right2_2", "top2_2", "top2_1", "top2_0"]},
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "top1_2", "top1_1", "right2_0", "top2_0"]},
      {"weight": 1/3, "edges": ["top0_3", "top0_2", "top0_1", "right1_0", "right2_0", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "right2_2", "top2_2", "top2_1", "right3_0"]},
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "top1_2", "top1_1", "right2_0", "right3_0"]},
      {"weight": 1/3, "edges": ["top0_3", "top0_2", "top0_1", "right1_0", "right2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "right2_2", "top2_2", "right3_1"]},
      {"weight": 1/3, "edges": ["top0_3", "top0_2", "right1_1", "right2_1", "right3_1"]},
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "top1_2", "right2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1, "edges": ["top0_3", "right1_2", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1, "edges": ["top0_3", "right1_2", "right2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1, "edges": ["top0_3", "right1_2", "bot1_3"]},
    ],
    "bot0_3":[
      {"weight": 1/3, "edges": ["top0_3", "right1_2", "left1_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["top0_3", "top0_2", "bot0_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["top0_3", "bot0_3"]},
    ]
  },

  "top1_3": {
    "bot0_3": [
      {"weight": 1, "edges": ["top1_3", "left1_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1, "edges": ["top1_3", "left1_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1/2, "edges": ["top1_3", "top1_2", "left1_1", "left0_1"]},
      {"weight": 1/2, "edges": ["top1_3", "left1_2", "top0_2", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "left1_1", "top0_1", "left0_0"]},
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "top1_1", "left1_0", "left0_0"]},
      {"weight": 1/3, "edges": ["top1_3", "left1_2", "top0_2", "top0_1", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "left1_1", "top0_1", "top0_0"]},
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "top1_1", "left1_0", "top0_0"]},
      {"weight": 1/3, "edges": ["top1_3", "left1_2", "top0_2", "top0_1", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1, "edges": ["top1_3", "top1_2", "top1_1", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "right2_1", "top2_1", "top2_0"]},
      {"weight": 1/3, "edges": ["top1_3", "right2_2", "top2_2", "top2_1", "top2_0"]},
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "top1_1", "right2_0", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "right2_1", "top2_1", "right3_0"]},
      {"weight": 1/3, "edges": ["top1_3", "right2_2", "top2_2", "top2_1", "right3_0"]},
      {"weight": 1/3, "edges": ["top1_3", "top1_2", "top1_1", "right2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1/2, "edges": ["top1_3", "top1_2", "right2_1", "right3_1"]},
      {"weight": 1/2, "edges": ["top1_3", "right2_2", "top2_2", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1, "edges": ["top1_3", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1, "edges": ["top1_3", "right2_2", "bot2_3"]},
    ],
    "bot1_3":[
      {"weight": 1/4, "edges": ["top1_3", "right2_2", "left2_2", "bot1_3"]},
      {"weight": 1/4, "edges": ["top1_3", "left1_2", "right1_2", "bot1_3"]},
      {"weight": 1/4, "edges": ["top1_3", "top1_2", "bot1_2", "bot1_3"]},
      {"weight": 1/4, "edges": ["top1_3", "bot1_3"]},
    ]
  },

  "top2_3": {
    "bot1_3": [
      {"weight": 1, "edges": ["top2_3", "left2_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1, "edges": ["top2_3", "left2_2", "left1_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1, "edges": ["top2_3", "left2_2", "left1_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1/3, "edges": ["top2_3", "left2_2", "left1_2", "top0_2", "left0_1"]},
      {"weight": 1/3, "edges": ["top2_3", "left2_2", "top1_2", "left1_1", "left0_1"]},
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "left2_1", "left1_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1/3, "edges": ["top2_3", "left2_2", "left1_2", "top0_2", "top0_1", "left0_0"]},
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "left2_1", "left1_1", "top0_1", "left0_0"]},
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "top2_1", "left2_0", "left1_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1/3, "edges": ["top2_3", "left2_2", "left1_2", "top0_2", "top0_1", "top0_0"]},
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "left2_1", "left1_1", "top0_1", "top0_0"]},
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "top2_1", "left2_0", "left1_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "top2_1", "left2_0", "top1_0"]},
      {"weight": 1/3, "edges": ["top2_3", "left2_2", "top1_2", "top1_1", "top1_0"]},
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "left2_1", "top1_1", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1, "edges": ["top2_3", "top2_2", "top2_1", "top2_0"]},
    ],
    "right3_2": [
      {"weight": 1, "edges": ["top2_3", "top2_2", "top2_1", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1, "edges": ["top2_3", "top2_2", "right3_1"]},
    ],
    "right3_0": [
      {"weight": 1, "edges": ["top2_3", "right3_2"]},
    ],
    "bot2_3":[
      {"weight": 1/3, "edges": ["top2_3", "top2_2", "bot2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["top2_3", "left2_2", "right2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["top2_3", "bot2_3"]},
    ]
  },

  "left3_2": {
    "bot2_3": [
      {"weight": 1, "edges": ["left3_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1, "edges": ["left3_2", "left2_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1, "edges": ["left3_2", "left2_2", "left1_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1, "edges": ["left3_2", "left2_2", "left1_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1/3, "edges": ["left3_2", "left2_2", "left1_2", "top0_2", "left0_1"]},
      {"weight": 1/3, "edges": ["left3_2", "left2_2", "top1_2", "left1_1", "left0_1"]},
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "left2_1", "left1_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1/3, "edges": ["left3_2", "left2_2", "left1_2", "top0_2", "top0_1", "left0_0"]},
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "left2_1", "left1_1", "top0_1", "left0_0"]},
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "top2_1", "left2_0", "left1_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1/3, "edges": ["left3_2", "left2_2", "left1_2", "top0_2", "top0_1", "top0_0"]},
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "left2_1", "left1_1", "top0_1", "top0_0"]},
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "top2_1", "left2_0", "left1_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "top2_1", "left2_0", "top1_0"]},
      {"weight": 1/3, "edges": ["left3_2", "left2_2", "top1_2", "top1_1", "top1_0"]},
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "left2_1", "top1_1", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1, "edges": ["left3_2", "top2_2", "top2_1", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1, "edges": ["left3_2", "top2_2", "top2_1", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1, "edges": ["left3_2", "top2_2", "right3_1"]},
    ],
    "right3_2":[
      {"weight": 1/3, "edges": ["left3_2", "top2_2", "bot2_2", "right3_2"]},   
      {"weight": 1/3, "edges": ["left3_2", "left2_2", "right2_2", "right3_2"]},   
      {"weight": 1/3, "edges": ["left3_2", "right3_2"]},   
    ]
  },

  "left3_1": {
    "right3_2": [
      {"weight": 1, "edges": ["left3_1", "bot2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1, "edges": ["left3_1", "bot2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1/2, "edges": ["left3_1", "bot2_2", "left2_2", "bot1_3"]},
      {"weight": 1/2, "edges": ["left3_1", "left2_1", "bot1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1/3, "edges": ["left3_1", "bot2_2", "left2_2", "left1_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "bot1_2", "left1_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "left1_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1/3, "edges": ["left3_1", "bot2_2", "left2_2", "left1_2", "left0_2"]},
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "bot1_2", "left1_2", "left0_2"]},
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "left1_1", "bot0_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1, "edges": ["left3_1", "left2_1", "left1_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "left1_1", "top0_1", "left0_0"]},
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "top1_1", "left1_0", "left0_0"]},
      {"weight": 1/3, "edges": ["left3_1", "top2_1", "left2_0", "left1_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "left1_1", "top0_1", "top0_0"]},
      {"weight": 1/3, "edges": ["left3_1", "left2_1", "top1_1", "left1_0", "top0_0"]},
      {"weight": 1/3, "edges": ["left3_1", "top2_1", "left2_0", "left1_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/2, "edges": ["left3_1", "top2_1", "left2_0", "top1_0"]},
      {"weight": 1/2, "edges": ["left3_1", "left2_1", "top1_1", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1, "edges": ["left3_1", "top2_1", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1, "edges": ["left3_1", "top2_1", "right3_0"]},
    ],
    "right3_1":[
      {"weight": 1/4, "edges": ["left3_1", "top2_1", "bot2_1", "right3_1"]}, 
      {"weight": 1/4, "edges": ["left3_1", "left2_1", "right2_1", "right3_1"]}, 
      {"weight": 1/4, "edges": ["left3_1", "bot2_2", "top2_2", "right3_1"]}, 
      {"weight": 1/4, "edges": ["left3_1", "right3_1"]}, 
    ]
  },

  "left3_0": {
    "right3_1": [
      {"weight": 1, "edges": ["left3_0", "bot2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1, "edges": ["left3_0", "bot2_1", "bot2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1, "edges": ["left3_0", "bot2_1", "bot2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "left2_1", "bot1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "bot2_2", "left2_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["left3_0", "left2_0", "bot1_1", "bot1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["left3_0", "left2_0", "left1_0", "bot0_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "left0_2"]},
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "left0_2"]},
      {"weight": 1/3, "edges": ["left3_0", "left2_0", "left1_0", "bot0_1", "bot0_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1/3, "edges": ["left3_0", "left2_0", "left1_0", "bot0_1", "left0_1"]},
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "left2_1", "left1_1", "left0_1"]},
      {"weight": 1/3, "edges": ["left3_0", "left2_0", "bot1_1", "left1_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1, "edges": ["left3_0", "left2_0", "left1_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1, "edges": ["left3_0", "left2_0", "left1_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1, "edges": ["left3_0", "left2_0", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1, "edges": ["left3_0", "top2_0"]},
    ],
    "right3_0":[
      {"weight": 1/3, "edges": ["left3_0", "bot2_1", "top2_1", "right3_0"]},
      {"weight": 1/3, "edges": ["left3_0", "left2_0", "right2_0", "right3_0"]},
      {"weight": 1/3, "edges": ["left3_0", "right3_0"]},
    ]
  },

  "bot2_0": {
    "right3_0": [
      {"weight": 1, "edges": ["bot2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1, "edges": ["bot2_0", "bot2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1, "edges": ["bot2_0", "bot2_1", "bot2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1, "edges": ["bot2_0", "bot2_1", "bot2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "left2_1", "bot1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["bot2_0", "left2_0", "bot1_1", "bot1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "bot2_2", "left2_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["bot2_0", "left2_0", "left1_0", "bot0_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "left0_2"]},
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "left0_2"]},
      {"weight": 1/3, "edges": ["bot2_0", "left2_0", "left1_0", "bot0_1", "bot0_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1/3, "edges": ["bot2_0", "left2_0", "left1_0", "bot0_1", "left0_1"]},
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "left2_1", "left1_1", "left0_1"]},
      {"weight": 1/3, "edges": ["bot2_0", "left2_0", "bot1_1", "left1_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1, "edges": ["bot2_0", "left2_0", "left1_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1, "edges": ["bot2_0", "left2_0", "left1_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1, "edges": ["bot2_0", "left2_0", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1/3, "edges": ["bot2_0", "left2_0", "right2_0", "top2_0"]},
      {"weight": 1/3, "edges": ["bot2_0", "bot2_1", "top2_1", "top2_0"]},
      {"weight": 1/3, "edges": ["bot2_0", "top2_0"]},
    ]
  },

  "bot1_0": {
    "top2_0": [
      {"weight": 1, "edges": ["bot1_0", "right2_0", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1, "edges": ["bot1_0", "right2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1/2, "edges": ["bot1_0", "right2_0", "bot2_1", "right3_1"]},
      {"weight": 1/2, "edges": ["bot1_0", "bot1_1", "right2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1/3, "edges": ["bot1_0", "right2_0", "bot2_1", "bot2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "right2_1", "bot2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "bot1_2", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1/3, "edges": ["bot1_0", "right2_0", "bot2_1", "bot2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "right2_1", "bot2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "bot1_2", "right2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1, "edges": ["bot1_0", "bot1_1", "bot1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "bot1_2", "left1_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "left1_1", "bot0_2", "bot0_3"]},
      {"weight": 1/3, "edges": ["bot1_0", "left1_0", "bot0_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "bot1_2", "left1_2", "left0_2"]},
      {"weight": 1/3, "edges": ["bot1_0", "bot1_1", "left1_1", "bot0_2", "left0_2"]},
      {"weight": 1/3, "edges": ["bot1_0", "left1_0", "bot0_1", "bot0_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1/2, "edges": ["bot1_0", "bot1_1", "left1_1", "left0_1"]},
      {"weight": 1/2, "edges": ["bot1_0", "left1_0", "bot0_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1, "edges": ["bot1_0", "left1_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1, "edges": ["bot1_0", "left1_0", "top0_0"]},
    ],
    "top1_0": [
      {"weight": 1/4, "edges": ["bot1_0", "left1_0", "right1_0", "top1_0"]},  
      {"weight": 1/4, "edges": ["bot1_0", "right2_0", "left2_0", "top1_0"]},
      {"weight": 1/4, "edges": ["bot1_0", "bot1_1", "top1_1", "top1_0"]},
      {"weight": 1/4, "edges": ["bot1_0", "top1_0"]},
    ]
  },

  "bot0_0": {
    "top1_0": [
      {"weight": 1, "edges": ["bot0_0", "right1_0", "top1_0"]},
    ],
    "top2_0": [
      {"weight": 1, "edges": ["bot0_0", "right1_0", "right2_0", "top2_0"]},
    ],
    "right3_0": [
      {"weight": 1, "edges": ["bot0_0", "right1_0", "right2_0", "right3_0"]},
    ],
    "right3_1": [
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "right2_0", "bot2_1", "right3_1"]},
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "bot1_1", "right2_1", "right3_1"]},
      {"weight": 1/3, "edges": ["bot0_0", "bot0_1", "right1_1", "right2_1", "right3_1"]},
    ],
    "right3_2": [
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "right3_2"]},
      {"weight": 1/3, "edges": ["bot0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "right3_2"]},
    ],
    "bot2_3": [
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "bot2_3"]},
      {"weight": 1/3, "edges": ["bot0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "bot2_3"]},
    ],
    "bot1_3": [
      {"weight": 1/3, "edges": ["bot0_0", "bot0_1", "bot0_2", "right1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "bot1_1", "bot1_2", "bot1_3"]},
      {"weight": 1/3, "edges": ["bot0_0", "bot0_1", "right1_1", "bot1_2", "bot1_3"]},
    ],
    "bot0_3": [
      {"weight": 1, "edges": ["bot0_0", "bot0_1", "bot0_2", "bot0_3"]},
    ],
    "left0_2": [
      {"weight": 1, "edges": ["bot0_0", "bot0_1", "bot0_2", "left0_2"]},
    ],
    "left0_1": [
      {"weight": 1, "edges": ["bot0_0", "bot0_1", "left0_1"]},
    ],
    "left0_0": [
      {"weight": 1, "edges": ["bot0_0", "left0_0"]},
    ],
    "top0_0": [
      {"weight": 1/3, "edges": ["bot0_0", "bot0_1", "top0_1", "top0_0"]},   
      {"weight": 1/3, "edges": ["bot0_0", "right1_0", "left1_0", "top0_0"]}, 
      {"weight": 1/3, "edges": ["bot0_0", "top0_0"]}, 
    ]
  },
}

# Make per-variant weights a valid probability vector.
# - If 'weight' is present, rescale so they sum to 1.0 (handles 1/3, rounding, zeros).
# - If no weights are given, assign equal shares.
# This prevents skew/divide-by-zero when splitting OD demand or sampling routes.
def _normalize_weights(routes):
    if not routes:
        return []
    if all("weight" in r for r in routes):
        s = sum(max(0.0, float(r["weight"])) for r in routes)
        return [dict(r, weight=(float(r["weight"])/s if s > 0 else 0.0)) for r in routes]
    # equal split
    w = 1.0 / len(routes)
    return [dict(r, weight=w) for r in routes]

# Return all route variants for (origin, dest) with weights normalized so they sum to 1.0.
# If the OD pair doesn't exist, returns an empty list.
def get_od_variants(group_routes: dict, origin: str, dest: str):
    """Return normalized list of route variants for this OD, or [] if none."""
    variants = group_routes.get(origin, {}).get(dest, [])
    return _normalize_weights(variants)

# Convenience accessor: get the edge list for variant k of (origin, dest).
# Returns None if the OD pair or variant index is missing/out of range.
def get_variant_edges_or_none(group_routes: dict, origin: str, dest: str, k: int):
    """Return edges list for variant k of this OD, or None if missing."""
    variants = group_routes.get(origin, {}).get(dest, [])
    if 0 <= (k or 0) < len(variants):
        return list(variants[k]["edges"])
    return None


# Compile per-OD route variants + per-episode OD integer counts into a flat, router-friendly
# spec: origin -> list[{weight, edges}]. For each origin, we:
#   1) Sum episode counts over all destinations to get total_o.
#   2) For each (origin,dest), normalize variant weights, then split the integer OD count n_od
#      across variants deterministically via Hamilton apportionment (floors + largest remainders).
#   3) Append each variant with weight = allocated_count / total_o so that per-origin weights sum to 1.
# If an OD has no explicit variants, it’s skipped (router falls back to shortest/manhattan).
def compile_hardcoded_from_group_and_counts(group_routes: dict, od_counts: dict) -> dict:
    """
    Build a flat 'hardcoded_routes' dict origin -> list[{weight, edges}],
    where weights reflect the *episode* split across all dests and their routes.
    (Used as fallback and for compatibility with current router.)
    """
    out = {}
    for o in ROW_ORDER:
        flat = []
        # total vehicles from origin this episode:
        total_o = sum(od_counts[o][d] for d in COL_ORDER)
        if total_o <= 0:
            out[o] = flat
            continue
        for d in COL_ORDER:
            n_od = int(od_counts[o][d])
            if n_od <= 0:
                continue
            variants = _normalize_weights(group_routes.get(o, {}).get(d, []))
            if not variants:
                # No explicit route; leave to router fallback (shortest/manhattan)
                continue
            # split n_od across variants deterministically (Hamilton again)
            quotas = [(i, v["weight"] * n_od) for i, v in enumerate(variants)]
            floors = {i: math.floor(q) for i, q in quotas}
            rems   = [(i, quotas[i][1] - floors[i]) for i, _ in quotas]
            target = n_od
            curr   = sum(floors.values())
            need   = max(0, target - curr)
            rems.sort(key=lambda x: (-x[1], x[0]))
            alloc  = floors.copy()
            for k in range(need):
                alloc[rems[k][0]] += 1
            # append with weights proportional to episode counts over total_o
            for i, v in enumerate(variants):
                k_i = alloc.get(i, 0)
                if k_i > 0:
                    flat.append({"weight": k_i / float(total_o), "edges": v["edges"]})
        out[o] = flat
    return out


# Build a deterministic, well-mixed departure sequence per origin from episode OD counts.
# For each (origin, dest):
#   • Normalize variant weights to sum to 1.0.
#   • Split the integer count n_od across variants via Hamilton apportionment
#     (floors + largest remainders; stable tie-break by index) to get k_i per variant.
#   • Expand each variant into k_i copies of its full edge list.
# Then round-robin interleave all (dest,variant) sublists so departures from the same
# origin are mixed (reducing clumping). Returns:
#   { origin: [edges_list, edges_list, ...] }
# Deterministic by construction (fixed ROW_ORDER/COL_ORDER, no RNG).
def build_precomputed_cycle(group_routes: dict, od_counts: dict) -> dict:
    """
    Build a deterministic cycle per origin: a list of full routes (edges list),
    expanded to exact episode counts and interleaved in round-robin order so
    departures from the same origin are well mixed.
    """
    per_origin_sequences = {}
    for o in ROW_ORDER:
        bucket = []  # list of lists, one sublist per (d,variant)
        # For stable interleave we keep the natural COL_ORDER, and route-index order
        for d in COL_ORDER:
            n_od = int(od_counts[o][d])
            if n_od <= 0:
                continue
            variants = _normalize_weights(group_routes.get(o, {}).get(d, []))
            if not variants:
                continue
            # split n_od across variants (Hamilton)
            quotas = [(i, v["weight"] * n_od) for i, v in enumerate(variants)]
            floors = {i: math.floor(q) for i, q in quotas}
            rems   = [(i, quotas[i][1] - floors[i]) for i, _ in quotas]
            target = n_od
            curr   = sum(floors.values())
            need   = max(0, target - curr)
            rems.sort(key=lambda x: (-x[1], x[0]))
            alloc  = floors.copy()
            for k in range(need):
                alloc[rems[k][0]] += 1
            # Expand each variant i into a list of that many copies
            for i, v in enumerate(variants):
                k_i = alloc.get(i, 0)
                if k_i > 0:
                    bucket.append([list(v["edges"])] * k_i)

        # Interleave round-robin
        seq = []
        still = True
        while still:
            still = False
            for lst in bucket:
                if lst:
                    seq.append(lst.pop())
                    still = True
        per_origin_sequences[o] = seq
    return per_origin_sequences

# ===== Global cycle storage used by the router for deterministic selection ====
_PRECOMPUTED_CYCLE = {}  # origin -> [route_edges, route_edges, ...] length = total vehicles from origin
_CYCLE_PTR = defaultdict(int)

def set_precomputed_cycle(cycle: dict):
    global _PRECOMPUTED_CYCLE, _CYCLE_PTR
    _PRECOMPUTED_CYCLE = {o: list(seq) for o, seq in cycle.items()}
    _CYCLE_PTR = defaultdict(int)  # reset pointers

def next_precomputed_route(origin: str):
    seq = _PRECOMPUTED_CYCLE.get(origin)
    if not seq:
        return None
    i = _CYCLE_PTR[origin]
    if not seq:
        return None
    route = seq[i % len(seq)]
    _CYCLE_PTR[origin] = i + 1
    return route

def prepare_group_from_csv(group_routes: dict, csv_path: str, horizon_s: int):
    """
    Convenience one-shot:
      - read CSV → OD rates
      - rates → episode OD integer counts
      - counts → (hardcoded_routes, precomputed_cycle)
      - also return inflow per hour per origin (row sums of rates)
    """
    od_rates = read_triplist_csv_to_od_rates(csv_path)
    od_counts = od_rates_to_episode_counts(od_rates, horizon_s)
    hardcoded = compile_hardcoded_from_group_and_counts(group_routes, od_counts)
    precomp   = build_precomputed_cycle(group_routes, od_counts)
    inflow_per_hour = {o: sum(od_rates[o][d] for d in COL_ORDER) for o in ROW_ORDER}
    return hardcoded, inflow_per_hour, precomp, od_counts  # return counts too, if you want to log




HARDCODED_ROUTES = {
        "right0_0": [
            # route_1 = right0_0 -> top0_0
            {"weight": 1/11, "edges": ["right0_0", "top0_0"]},

            # route_2 = right0_0 -> top1_0
            {"weight": 1/11, "edges": ["right0_0", "right1_0", "top1_0"]},

            # route 3 = right0_0 -> top2_0
            {"weight": 1/11, "edges": ["right0_0", "right1_0", "right2_0", "top2_0"]},

            # route 4 = right0_0 -> right3_0
            {"weight": 1/11, "edges": ["right0_0", "right1_0", "right2_0", "right3_0"]},

            # route 5 = right0_0 -> right 3_1  
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "right2_0", "bot2_1", "right3_1"]},
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "bot1_1", "right2_1", "right3_1"]},            
            {"weight": 1/33, "edges": ["right0_0", "bot0_1", "right1_1", "right2_1", "right3_1"]},


            # route 6 = right0_0 -> right3_2
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["right0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "right3_2"]},

            # route 7 = right0_0 -> bot2_3 
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["right0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "bot2_3"]},

            # route 8 =right0_0 -> bot1_3
            {"weight": 1/33, "edges": ["right0_0", "bot0_1", "bot0_2", "right1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["right0_0", "bot0_1", "right1_1", "bot1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["right0_0", "right1_0", "bot1_1", "bot1_2", "bot1_3"]},

            # route 9 = right0_0 -> bot0_3
            {"weight": 1/11, "edges": ["right0_0", "bot0_1", "bot0_2", "bot0_3"]},

            # route 10 = right0_0 -> left0_2
            {"weight": 1/11, "edges": ["right0_0", "bot0_1", "bot0_2", "left0_2"]},

            #route 11 = right0_0 -> left0_1
            {"weight": 1/11, "edges": ["right0_0", "bot0_1", "left0_1"]},
        ],

        "right0_1": [
            # route 1 = right0_1 -> left0_0
            {"weight": 1/11, "edges": ["right0_1", "top0_1", "left0_0"]},

            # route 2 = right0_1 -> top0_0
            {"weight": 1/11, "edges": ["right0_1", "top0_1", "top0_0"]},

            # route 3 = right0_1 -> top1_0
            {"weight": 1/22, "edges": ["right0_1", "top0_1", "right1_0", "top1_0"]},
            {"weight": 1/22, "edges": ["right0_1", "right1_1", "top1_1", "top1_0"]},

            # route 4 = right0_1 -> top2_0
            {"weight": 1/33, "edges": ["right0_1", "top0_1", "right1_0", "right2_0", "top2_0"]},
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "top1_1", "right2_0", "top2_0"]},
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "right2_1", "top2_1", "top2_0"]},

            # route 5 = right0_1 -> right3_0
            {"weight": 1/33, "edges": ["right0_1", "top0_1", "right1_0", "right2_0", "right3_0"]},
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "top1_1", "right2_0", "right3_0"]},
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "right2_1", "top2_1", "right3_0"]},   

            # route 6 = right0_1 -> right3_1
            {"weight": 1/11, "edges": ["right0_1", "right1_1", "right2_1", "right3_1"]},

            # route 7 = right0_1 -> right3_2
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "right2_1", "bot2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "bot1_2", "right2_2", "right3_2"]},            
            {"weight": 1/33, "edges": ["right0_1", "bot0_2", "right1_2", "right2_2", "right3_2"]},

            # route 8 = right0_1 -> bot2_3
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "right2_1", "bot2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["right0_1", "right1_1", "bot1_2", "right2_2", "bot2_3"]},            
            {"weight": 1/33, "edges": ["right0_1", "bot0_2", "right1_2", "right2_2", "bot2_3"]},    

            # route 9 = right0_1 -> bot1_3 
            {"weight": 1/22, "edges": ["right0_1", "bot0_2", "right1_2", "bot1_3"]},  
            {"weight": 1/22, "edges": ["right0_1", "right1_1", "bot1_2", "bot1_3"]},

            # route 10 = right0_1 -> bot0_3
            {"weight": 1/11, "edges": ["right0_1", "bot0_2", "bot0_3"]},

            # route 11 = right0_1 -> left0_2
            {"weight": 1/11, "edges": ["right0_1", "bot0_2", "left0_2"]},

        ],

        "right0_2": [ 
            # route 1 = right0_2 -> left0_1
            {"weight": 1/11, "edges": ["right0_2", "top0_2", "left0_1"]},

            # route 2 = right0_2 -> left0_0
            {"weight": 1/11, "edges": ["right0_2", "top0_2", "top0_1", "left0_0"]},

            # route 3 = right0_2 -> top0_0
            {"weight": 1/11, "edges": ["right0_2", "top0_2", "top0_1", "top0_0"]},

            # route 4 = right0_2 -> top1_0
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "right1_1", "top1_1", "top1_0"]},
            {"weight": 1/33, "edges": ["right0_2", "right1_2", "top1_2", "top1_1", "top1_0"]},
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "top0_1", "right1_0", "top1_0"]},

            # route 5 = right0_2 -> top2_0
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "right1_1", "right2_1", "top2_1", "top2_0"]},
            {"weight": 1/33, "edges": ["right0_2", "right1_2", "right2_2", "top2_2", "top2_1", "top2_0"]},
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "top0_1", "right1_0", "right2_0", "top2_0"]},

            # route 6 = right0_2 -> right3_0
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "right1_1", "right2_1", "top2_1", "right3_0"]},
            {"weight": 1/33, "edges": ["right0_2", "right1_2", "right2_2", "top2_2", "top2_1", "right3_0"]},
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "top0_1", "right1_0", "right2_0", "right3_0"]},

            # route 7 = right0_2 -> right3_1
            {"weight": 1/33, "edges": ["right0_2", "right1_2", "right2_2", "top2_2", "right3_1"]},
            {"weight": 1/33, "edges": ["right0_2", "top0_2", "right1_1", "right2_1", "right3_1"]},
            {"weight": 1/33, "edges": ["right0_2", "right1_2", "top1_2", "right2_1", "right3_1"]},

            # route 8 = right0_2 -> right3_2
            {"weight": 1/11, "edges": ["right0_2", "right1_2", "right2_2", "right3_2"]},

            # route 9 = right0_2 -> bot2_3
            {"weight": 1/11, "edges": ["right0_2", "right1_2", "right2_2", "bot2_3"]},

            # route 10 = right0_2 -> bot1_3
            {"weight": 1/11, "edges": ["right0_2", "right1_2", "bot1_3"]},

            # route 11 = right0_2 -> bot0_3
            {"weight": 1/11, "edges": ["right0_2", "bot0_3"]},
        ],

        "top0_3": [
            #route 1 = top0_3 -> left0_2
            {"weight": 1/11, "edges": ["top0_3", "left0_2"]},

            # route 2 = top0_3 -> left0_1
            {"weight": 1/11, "edges": ["top0_3", "top0_2", "left0_1"]},

            # route 3 = top0_3 -> left0_0
            {"weight": 1/11, "edges": ["top0_3", "top0_2", "top0_1", "left0_0"]},

            # route 4 = top0_3 -> top0_0
            {"weight": 1/11, "edges": ["top0_3", "top0_2", "top0_1", "top0_0"]},

            # route 5 = top0_3 -> top1_0
            {"weight": 1/33, "edges": ["top0_3", "top0_2", "right1_1", "top1_1", "top1_0"]},
            {"weight": 1/33, "edges": ["top0_3", "right1_2", "top1_2", "top1_1", "top1_0"]},
            {"weight": 1/33, "edges": ["top0_3", "top0_2", "top0_1", "right1_0", "top1_0"]},

            # route 6 = top0_3 -> top2_0
            {"weight" : 1/33, "edges": ["top0_3", "right1_2", "right2_2", "top2_2", "top2_1", "top2_0"]},
            {"weight" : 1/33, "edges": ["top0_3", "right1_2", "top1_2", " top1_1", "right2_0", "top2_0"]},
            {"weight" : 1/33, "edges": ["top0_3", "top0_2", "top0_1", "right1_0", "right2_0", "top2_0"]},

            # route 7 = top0_3 -> right3_0
            {"weight" : 1/33, "edges": ["top0_3", "right1_2", "right2_2", "top2_2", "top2_1", "right3_0"]},
            {"weight" : 1/33, "edges": ["top0_3", "right1_2", "top1_2", " top1_1", "right2_0", "right3_0"]},
            {"weight" : 1/33, "edges": ["top0_3", "top0_2", "top0_1", "right1_0", "right2_0", "right3_0"]},

            # route 8 = top0_3 -> right3_1
            {"weight": 1/33, "edges": ["top0_3", "right1_2", "right2_2", "top2_2", "right3_1"]},
            {"weight": 1/33, "edges": ["top0_3", "top0_2", "right1_1", "right2_1", "right3_1"]},
            {"weight": 1/33, "edges": ["top0_3", "right1_2", "top1_2", "right2_1", "right3_1"]},

            # route 9 = top0_3 -> right3_2
            {"weight": 1/11, "edges": ["top0_3", "right1_2", "right2_2", "right3_2"]},

            # route 10 = top0_3 -> bot2_3
            {"weight": 1/11, "edges": ["top0_3", "right1_2", "right2_2", "bot2_3"]},

            # route 11 = top0_3 -> bot1_3
            {"weight": 1/11, "edges": ["top0_3", "right1_2", "bot1_3"]},
        ],

        "top1_3": [
            # route 1 = top1_3 -> bot0_3
            {"weight": 1/11, "edges": ["top1_3", "right1_2", "bot0_3"]},

            # route 2 = top1_3 -> left0_2
            {"weight": 1/11, "edges": ["top1_3", "right1_2", "left0_2"]},

            # route 3 = top1_3 -> left0_1
            {"weight": 1/22, "edges": ["top1_3", "top1_2", "left1_1", "left0_1"]},
            {"weight": 1/22, "edges": ["top1_3", "right1_2", "top0_2", "left0_1"]},

            # route 4 = top1_3 -> left0_0
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "left1_1", "top0_1", "left0_0"]},
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "top1_1", "left1_0", "left0_0"]},
            {"weight": 1/33, "edges": ["top1_3", "right1_2", "top0_2", "top0_1", "left0_0"]},

            # route 5 = top1_3 -> top0_0
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "left1_1", "top0_1", "top0_0"]},
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "top1_1", "left1_0", "top0_0"]},
            {"weight": 1/33, "edges": ["top1_3", "right1_2", "top0_2", "top0_1", "top0_0"]},

            # route 6 = top1_3 -> top1_0
            {"weight": 1/11, "edges": ["top1_3", "top1_2", "top1_1", "top1_0"]},

            # route 7 = top1_3 -> top2_0
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "right2_1", "top2_1", "top2_0"]},
            {"weight": 1/33, "edges": ["top1_3", "right2_2", "top2_2", "top2_1", "top2_0"]},
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "top1_1", "right2_0", "top2_0"]},

            # route 8 = top1_3 -> right3_0
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "right2_1", "top2_1", "right3_0"]},
            {"weight": 1/33, "edges": ["top1_3", "right2_2", "top2_2", "top2_1", "right3_0"]},
            {"weight": 1/33, "edges": ["top1_3", "top1_2", "top1_1", "right2_0", "right3_0"]},   

            # route 9 = top1_3 -> right3_1
            {"weight": 1/22, "edges": ["top1_3", "top1_2", "right2_1", "right3_1"]},
            {"weight": 1/22, "edges": ["top1_3", "right2_2", "top2_2", "right3_1"]},

            # route 10 = top1_3 -> right3_2
            {"weight": 1/11, "edges": ["top1_3", "right2_2", "right3_2"]},

            # route 11 = top1_3 -> bot2_3 
            {"weight": 1/11, "edges": ["top1_3", "right2_2", "bot2_3"]},         
        ],

        "top2_3": [
            # route 1 = top2_3 -> bot1_3
            {"weight": 1/11, "edges": ["top2_3", "left2_2", "bot1_3"]},

            # route 2 = top2_3 -> bot0_3
            {"weight": 1/11, "edges": ["top2_3", "left2_2", "left2_1", "bot0_3"]},

            # route 3 = top2_3 -> left0_2
            {"weight": 1/11, "edges": ["top2_3", "left2_2", "left1_2", "left0_2"]},

            # route 4 = top2_3 -> left0_1
            {"weight": 1/33, "edges": ["top2_3", "left2_2", "left1_2", "top0_2", "left0_1"]},
            {"weight": 1/33, "edges": ["top2_3", "left2_2", "top1_2", "left1_1", "left0_1"]},
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "left1_2", "left1_1", "left0_1"]},

            # route 5 = top2_3 -> left0_0
            {"weight": 1/33, "edges": ["top2_3", "left2_2", "left1_2", "top0_2", "top0_1", "left0_0"]},
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "left2_1", "left1_1", "top0_1", "left0_0"]},
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "top2_1", "left2_0", "left1_0", "left0_0"]},

            # route 6 = top2_3 -> top0_0
            {"weight": 1/33, "edges": ["top2_3", "left2_2", "left1_2", "top0_2", "top0_1", "top0_0"]},
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "left2_1", "left1_1", "top0_1", "top0_0"]},
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "top2_1", "left2_0", "left1_0", "top0_0"]},

            # route 7 = top2_3 -> top1_0
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "top2_1", "left2_0", "top1_0"]},
            {"weight": 1/33, "edges": ["top2_3", "left2_2", "top1_2", "top1_1", "top1_0"]},
            {"weight": 1/33, "edges": ["top2_3", "top2_2", "left2_1", "top1_1", "top1_0"]},

            # route 8 = top2_3 -> top2_0
            {"weight": 1/11, "edges": ["top2_3", "top2_2", "top2_1", "top2_0"]},

            # route 9 = top2_3 -> right3_2
            {"weight": 1/11, "edges": ["top2_3", "top2_2", "top2_1", "right3_2"]},

            # route 10 = top2_3 -> right3_1
            {"weight": 1/11, "edges": ["top2_3", "top2_2", "right3_1"]},

            # route 11 = top2_3 -> right3_0
            {"weight": 1/11, "edges": ["top2_3", "right3_0"]},
        ],

        "left3_2": [
            # route 1 = left3_2 -> bot2_3
            {"weight": 1/11, "edges": ["left3_2", "bot2_3"]},

            # route 2 = left3_2 -> bot1_3
            {"weight": 1/11, "edges": ["left3_2", "left2_2", "bot1_3"]},

            # route 3 = left3_2 -> bot0_3
            {"weight": 1/11, "edges": ["left3_2", "left2_2", "left1_2", "bot0_3"]},

            # route 4 = left3_2 -> left0_2
            {"weight": 1/11, "edges": ["left3_2", "left2_2", "left1_2", "left0_2"]},

            # route 5 = left3_2 -> left0_1
            {"weight": 1/33, "edges": ["left3_2", "left2_2", "left1_2", "top0_2", "left0_1"]},
            {"weight": 1/33, "edges": ["left3_2", "left2_2", "top1_2", "left1_1", "left0_1"]},
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "left2_1", "left1_1", "left0_1"]},

            # route 6 = left3_2 -> left0_0
            {"weight": 1/33, "edges": ["left3_2", "left2_2", "left1_2", "top0_2", "top0_1", "left0_0"]},
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "left2_1", "left1_1", "top0_1", "left0_0"]},
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "top2_1", "left2_0", "left1_0", "left0_0"]},

            # route 7 = left3_2 -> top0_0
            {"weight": 1/33, "edges": ["left3_2", "left2_2", "left1_2", "top0_2", "top0_1", "top0_0"]},
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "left2_1", "left1_1", "top0_1", "top0_0"]},
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "top2_1", "left2_0", "left1_0", "top0_0"]},

            # route 8 = left3_2 -> top1_0
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "top2_1", "left2_0", "top1_0"]},
            {"weight": 1/33, "edges": ["left3_2", "left2_2", "top1_2", "top1_1", "top1_0"]},
            {"weight": 1/33, "edges": ["left3_2", "top2_2", "left2_1", "top1_1", "top1_0"]},

            # route 9 = left3_2 -> top2_0
            {"weight": 1/11, "edges": ["left3_2", "top2_2", "top2_1", "top2_0"]},

            # route 10 = left3_2 -> right3_0
            {"weight": 1/11, "edges": ["left3_2", "top2_2", "top2_1", "right3_0"]},

            # route 11 = left3_2 -> right3_1
            {"weight": 1/11, "edges": ["left3_2", "top2_2", "right3_1"]},
        ],

        "left3_1": [
            #route 1 = left3_1 -> right3_2
            {"weight": 1/11, "edges": ["left3_1", "bot2_2", "right3_2"]},

            # route 2 = left3_1 -> bot2_3
            {"weight": 1/11, "edges": ["left3_1", "bot2_2", "bot2_3"]},

            # route 3 = left3_1 -> bot1_3
            {"weight": 1/22, "edges": ["left3_1", "bot2_2", "left2_2", "bot1_3"]},
            {"weight": 1/22, "edges": ["left3_1", "left2_1", "bot1_2", "bot1_3"]},

            # route 4 = left3_1 -> bot0_3
            {"weight": 1/33, "edges": ["left3_1", "bot2_2", "left2_2", "left1_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "bot1_2", "left1_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "left1_1", "bot0_2", "bot0_3"]},

            # route 5 = left3_1 -> left0_2
            {"weight": 1/33, "edges": ["left3_1", "bot2_2", "left2_2", "left1_2", "left0_2"]},
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "bot1_2", "left1_2", "left0_2"]},
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "left1_1", "bot0_2", "left0_2"]},

            # route 6 = left3_1 -> left0_1
            {"weight": 1/11, "edges": ["left3_1", "left2_1", "left1_1", "left0_1"]},

            # route 7 = left3_1 -> left0_0
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "left1_1", "top0_1", "left0_0"]},
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "top1_1", "left1_0", "left0_0"]},
            {"weight": 1/33, "edges": ["left3_1", "top2_1", "left2_0", "left1_0", "left0_0"]},

            # route 8 = left3_1 -> top0_0
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "left1_1", "top0_1", "top0_0"]},
            {"weight": 1/33, "edges": ["left3_1", "left2_1", "top1_1", "left1_0", "top0_0"]},
            {"weight": 1/33, "edges": ["left3_1", "top2_1", "left2_0", "left1_0", "top0_0"]},

            # route 9 = left3_1 -> top1_0
            {"weight": 1/22, "edges": ["left3_1", "top2_1", "left2_0", "top1_0"]},
            {"weight": 1/22, "edges": ["left3_1", "left2_1", "top1_1", "top1_0"]},

            # route 10 = left3_1 -> top2_0
            {"weight": 1/11, "edges": ["left3_1", "top2_1", "top2_0"]},

            # route 11 = left3_1 -> right3_0
            {"weight": 1/11, "edges": ["left3_1", "right3_0"]},
        ],

        "left3_0": [
            # route 1 = left3_0 -> right3_1
            {"weight": 1/11, "edges": ["left3_0", "bot2_1", "right3_1"]},

            # route 2 = left3_0 -> right3_2
            {"weight": 1/11, "edges": ["left3_0", "bot2_1", "bot2_2", "right3_2"]},

            # route 3 = left3_0 -> bot2_3
            {"weight": 1/11, "edges": ["left3_0", "bot2_1", "bot2_2", "bot2_3"]},

            # route 4 = left3_0 -> bot1_3
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "left2_1", "bot1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "bot2_2", "left2_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["left3_0", "left2_0", "bot1_1", "bot1_2", "bot1_3"]},

            # route 5 = left3_0 -> bot0_3
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["left3_0", "left2_0","left1_0", "bot0_1", "bot0_2", "bot0_3"]},

            # route 6 = left3_0 -> left0_2
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "left0_2"]},
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "left0_2"]},
            {"weight": 1/33, "edges": ["left3_0", "left2_0","left1_0", "bot0_1", "bot0_2", "left0_2"]},

            # route 7 = left3_0 -> left0_1
            {"weight": 1/33, "edges": ["left3_0", "left2_0", "left1_0", "bot0_1", "left0_1"]},
            {"weight": 1/33, "edges": ["left3_0", "bot2_1", "left2_1", "left1_1", "left0_1"]},
            {"weight": 1/33, "edges": ["left3_0", "left2_0", "bot1_1", "left1_1", "left0_1"]},

            # route 8 = left3_0 -> left0_0
            {"weight": 1/11, "edges": ["left3_0", "left2_0", "left1_0", "left0_0"]},

            # route 9 = left3_0 -> top0_0
            {"weight": 1/11, "edges": ["left3_0", "left2_0", "left1_0", "top0_0"]},

            # route 10 = left3_0 -> top1_0
            {"weight": 1/11, "edges": ["left3_0", "left2_0", "top1_0"]},

            # route 11 = left3_0 -> top2_0
            {"weight": 1/11, "edges": ["left3_0", "top2_0"]},
        ],

        "bot2_0": [
            # route 1 = bot2_0 -> right3_0
            {"weight": 1/11, "edges": ["bot2_0", "right3_0"]},

            # route 2 = bot2_0 -> right3_1
            {"weight": 1/11, "edges": ["bot2_0", "bot2_1", "right3_1"]},

            # route 3 = bot2_0 -> right3_2
            {"weight": 1/11, "edges": ["bot2_0", "bot2_1", "bot2_2", "right3_2"]},

            # route 4 = bot2_0 -> bot2_3
            {"weight": 1/11, "edges": ["bot2_0", "bot2_1", "bot2_2", "bot2_3"]},

            # route 5 = bot2_0 -> bot1_3
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "left2_1", "bot1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["bot2_0", "left2_0", "bot1_1", "bot1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "bot2_2", "left2_2", "bot1_3"]},

            # route 6 = bot2_0 -> bot0_3
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["bot2_0", "left2_0","left1_0", "bot0_1", "bot0_2", "bot0_3"]},

            # route 7 = bot2_0 -> left0_2
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "bot2_2", "left2_2", "left1_2", "left0_2"]},
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "left2_1", "left1_1", "bot0_2", "left0_2"]},
            {"weight": 1/33, "edges": ["bot2_0", "left2_0","left1_0", "bot0_1", "bot0_2", "left0_2"]},

            # route 8 = bot2_0 -> left0_1
            {"weight": 1/33, "edges": ["bot2_0", "left2_0", "left1_0", "bot0_1", "left0_1"]},
            {"weight": 1/33, "edges": ["bot2_0", "bot2_1", "left2_1", "left1_1", "left0_1"]},
            {"weight": 1/33, "edges": ["bot2_0", "left2_0", "bot1_1", "left1_1", "left0_1"]},

            # route 9 = bot2_0 -> left0_0
            {"weight": 1/11, "edges": ["bot2_0", "left2_0", "left1_0", "left0_0"]},

            # route 10 = bot2_0 -> top0_0
            {"weight": 1/11, "edges": ["bot2_0", "left2_0", "left1_0", "top0_0"]},

            # route 11 = bot2_0 -> top1_0
            {"weight": 1/11, "edges": ["bot2_0", "left2_0", "top1_0"]},
        ],

        "bot1_0": [
            # route 1 = bot1_0 -> top2_0
            {"weight": 1/11, "edges": ["bot1_0", "right2_0", "top2_0"]},

            # route 2 = bot1_0 -> right3_0
            {"weight": 1/11, "edges": ["bot1_0", "right2_0", "right3_0"]},

            # route 3 = bot1_0 -> right3_1
            {"weight": 1/22, "edges": ["bot1_0", "right2_0", "bot2_1", "right3_1"]},
            {"weight": 1/22, "edges": ["bot1_0", "bot1_1", "right2_1", "right3_1"]},

            # route 4 = bot1_0 -> right3_2
            {"weight": 1/33, "edges": ["bot1_0", "right2_0", "bot2_1", "bot2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "right2_1", "bot2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "bot1_2", "right2_2", "right3_2"]},

            # route 5 = bot1_0 -> bot2_3
            {"weight": 1/33, "edges": ["bot1_0", "right2_0", "bot2_1", "bot2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "right2_1", "bot2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "bot1_2", "right2_2", "bot2_3"]},

            # route 6 = bot1_0 -> bot1_3
            {"weight": 1/11, "edges": ["bot1_0", "bot1_1", "bot1_2", "bot1_3"]},

            # route 7 = bot1_0 -> bot0_3
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "bot1_2", "left1_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "left1_1", "bot0_2", "bot0_3"]},
            {"weight": 1/33, "edges": ["bot1_0", "left1_0", "bot0_1", "bot0_2", "bot0_3"]},

            # route 8 = bot1_0 -> left0_2
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "bot1_2", "left1_2", "left0_2"]},
            {"weight": 1/33, "edges": ["bot1_0", "bot1_1", "left1_1", "bot0_2", "left0_2"]},
            {"weight": 1/33, "edges": ["bot1_0", "left1_0", "bot0_1", "bot0_2", "left0_2"]},

            # route 9 = bot1_0 -> left0_1
            {"weight": 1/22, "edges": ["bot1_0", "bot1_1", "left1_1", "left0_1"]},
            {"weight": 1/22, "edges": ["bot1_0", "left1_0", "bot0_1", "left0_1"]},

            # route 10 = bot1_0 -> left0_0
            {"weight": 1/11, "edges": ["bot1_0", "left1_0", "left0_0"]},

            # route 11 = bot1_0 -> top0_0
            {"weight": 1/11, "edges": ["bot1_0", "left1_0", "top0_0"]},
        ],

        "bot0_0": [
            # route 1 = bot0_0 -> top1_0
            {"weight": 1/11, "edges": ["bot0_0", "right1_0", "top1_0"]},

            # route 2 = bot0_0 -> top2_0
            {"weight": 1/11, "edges": ["bot0_0", "right1_0", "right2_0", "top2_0"]},

            # route 3 = bot0_0 -> right3_0
            {"weight": 1/11, "edges": ["bot0_0", "right1_0", "right2_0", "right3_0"]},

            # route 4 = bot0_0 -> right3_1
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "right2_0", "bot2_1", "right3_1"]},
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "bot1_1", "right2_1", "right3_1"]},
            {"weight": 1/33, "edges": ["bot0_0", "bot0_1", "right1_1", "right2_1", "right3_1"]},

            # route 5 = bot0_0 -> right3_2
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "right3_2"]},
            {"weight": 1/33, "edges": ["bot0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "right3_2"]},

            # route 6 = bot0_0 -> bot2_3
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "right2_0", "bot2_1", "bot2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "bot1_1", "bot1_2", "right2_2", "bot2_3"]},
            {"weight": 1/33, "edges": ["bot0_0", "bot0_1", "bot0_2", "right1_2", "right2_2", "bot2_3"]},

            # route 7 = bot0_0 -> bot1_3
            {"weight": 1/33, "edges": ["bot0_0", "bot0_1", "bot0_2", "right1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["bot0_0", "right1_0", "bot1_1", "bot1_2", "bot1_3"]},
            {"weight": 1/33, "edges": ["bot0_0", "bot0_1", "right1_1", "bot1_2", "bot1_3"]},

            # route 8 = bot0_0 -> bot0_3
            {"weight": 1/11, "edges": ["bot0_0", "bot0_1", "bot0_2", "bot0_3"]},

            # route 9 = bot0_0 -> left0_2
            {"weight": 1/11, "edges": ["bot0_0", "bot0_1", "bot0_2", "left0_2"]},

            # route 10 = bot0_0 -> left0_1
            {"weight": 1/11, "edges": ["bot0_0", "bot0_1", "left0_1"]},

            # route 11 = bot0_0 -> left0_0
            {"weight": 1/11, "edges": ["bot0_0", "left0_0"]}
        ],
    }