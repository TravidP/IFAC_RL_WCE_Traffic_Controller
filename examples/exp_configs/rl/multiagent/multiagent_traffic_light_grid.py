"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams, TrafficLightParams
from flow.controllers import SimCarFollowingController, GridRouter, SimLaneChangeController
from flow.controllers.routing_controllers import OuterEdgeODRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from .tls_presets import build_two_lane_hold_tls
from fnmatch import fnmatch
from pathlib import Path
from flow.controllers.Routes import (
    ROW_ORDER, COL_ORDER,
    prepare_group_from_csv, Group_routes_G1,  # use your chosen group variable here
    set_precomputed_cycle
)
from flow.controllers.Routes import read_triplist_csv_to_od_rates, ROW_ORDER, COL_ORDER, Group_routes_G1, _normalize_weights



# Experiment parameters
N_ROLLOUTS = 1  # number of rollouts per training iteration
N_CPUS = 1 # number of parallel workers

# Environment parameters
HORIZON = 900  # time horizon of a single rollout
V_ENTER = 13.89  # enter speed for departing vehicles 50 km/h
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on

N_ROWS = 3     # number of rows of bidirectional lanes
N_COLUMNS = 3  # number of columns of bidirectional lanes

#new 
# === OD data source for this group ===
OD_CSV = "/home/sdc_joran/Athena Data/data/aoi_od_pairs_group1.csv"   # your uploaded file
HC_ROUTES, EDGE_INFLOW, PRECOMP_CYCLE, OD_COUNTS = prepare_group_from_csv(
    Group_routes_G1,  # pick the group you filled
    OD_CSV,
    HORIZON
)
# Install the cycle for deterministic per-vehicle route choice
set_precomputed_cycle(PRECOMP_CYCLE)


# ===== Real-world node coordinates (WGS-like or projected units) =====
RAW_NODE_COORDS = {
    # inner nodes
    "n0": (740092.609, 4206840.109),
    "n1": (739843.986, 4207215.429),
    "n2": (739784.542, 4207302.469),
    "n3": (740202.636, 4206910.988),
    "n4": (739961.565, 4207256.932),
    "n5": (739890.232, 4207349.964),
    "n6": (740345.290, 4207010.310),
    "n7": (740139.197, 4207361.506),
    "n8": (740074.736, 4207448.396),
    # outer nodes (each serves both short & long for that side)
    "g0": (739612.0855075686, 4207547.945594033),
    "g1": (739485.5036382437, 4207326.470212429),
    "g2": (739571.0385821164, 4207090.931180458),
    "g3": (740302.5262034187, 4206625.78502162),
    "g4": (739802.3217088071, 4206764.390517634),
    "g5": (740367.3558577141, 4206660.2543395655),
    "g6": (740538.3215322343, 4206780.660642144),
    "g7": (740597.1240073298, 4207173.342612541),
    "g8": (740388.0134816014, 4207529.107785443),
    "g9": (739933.8791491276, 4207713.2720985105),
    "g10": (740312.7081826181, 4207631.068494645),
    "g11": (739733.7687455552, 4207605.931283082),
}

# inflow rate of vehicles at every edge
UNIFORM_EDGE_INFLOW = 400
EDGE_INFLOW = {
    "right0_0": 400, "right0_1": 400, "right0_2": 400,  # inflow rate of vehicles from the southern edges
    "left3_0": 400, "left3_1": 400, "left3_2": 400,     # inflow rate of vehicles from the northern edges
    "top0_3": 400, "top1_3": 400, "top2_3": 400,        # inflow rate of vehicles from the eastern edges
    "bot0_0": 400, "bot1_0": 400, "bot2_0": 400,        # inflow rate of vehicles from the western edges
}

def inflow_for_edge(edge_id: str) -> float:
    """Return vehs/hour for this edge, honoring exact and wildcard overrides."""
    # exact match wins
    if edge_id in EDGE_INFLOW:
        return float(EDGE_INFLOW[edge_id])
    # then wildcard matches (first match wins)
    for pat, val in EDGE_INFLOW.items():
        if "*" in pat and fnmatch(edge_id, pat):
            return float(val)
    # fallback
    return float(UNIFORM_EDGE_INFLOW)

# new
def period_for_edge(edge_id: str, horizon=HORIZON) -> float:
    """
    Deterministic spacing: period = horizon / N_origin, where
    N_origin = round(row_sum_rate * horizon / 3600).
    If N_origin == 0, return a huge period to effectively disable spawning.
    """
    rate_h = EDGE_INFLOW.get(edge_id, 0.0)  # veh/h
    n = int(round(rate_h * horizon / 3600.0))
    if n <= 0:
        return 1e9  # effectively none this episode
    return float(horizon) / float(n)


# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        tau = 1.0, # reaction time
        sigma=0.5, # imperfection
        speed_mode="right_of_way",
    ),
    lane_change_controller=(SimLaneChangeController, {}),
    #routing_controller=(Right00ToTop00Router, {"seed": 1}),
    routing_controller=(OuterEdgeODRouter, {
        "seed": 1,
        "use_hardcoded": True,          # False when using real data
        # both new 
        "hardcoded_routes": Group_routes_G1,    # <- compiled from OD + your Group_routes
        "seed": 42,
        "use_precomputed_cycle": True,    # (not read directly, but useful for clarity)
        "debug_origin": "right0_0",  # for debug output,
        # Add the following when using real data
        # "routes_xml_path": "/home/sdc_joran/flow/examples/routes.rou_with_uturns.xml",  # with u-turns
        # "routes_xml_path": "/home/sdc_joran/flow/examples/routes.rou.xml",  # without u-turs
    }),
    num_vehicles=0
    )

# Define route file
#ROU = str(Path(__file__).resolve().parents[2] / "/home/sdc_joran/flow/examples/routes.rou.xml")  
ROU = str(Path(__file__).resolve().parents[2] / "/home/sdc_joran/flow/examples/routes.rou_with_uturns.xml")  

# --------------------------------- Used for when having inflow per OD pair
# New
# === Build per-OD (and per-variant) inflows ===
OD_CSV = "/home/sdc_joran/Athena Data/data/aoi_od_pairs_group1.csv"   # OD-pairs file
od_rates = read_triplist_csv_to_od_rates(OD_CSV)

inflow = InFlows()
for origin in ROW_ORDER:
    for dest in COL_ORDER:
        rate = od_rates[origin][dest]  # veh/h for this OD
        if rate <= 0:
            continue

        variants = _normalize_weights(Group_routes_G1.get(origin, {}).get(dest, []))

        if not variants:
            # No explicit edges: still spawn this OD; router will use dest lock + shortest-path.
            period = 3600.0 / rate
            inflow.add(
                veh_type="human",
                edge=origin,
                period=period,
                name=f"OD_{origin}_to_{dest}",   # <- tag the OD
                depart_lane="best",
                depart_speed=V_ENTER,
                departPos="free",
                begin=0, end=HORIZON
            )
        else:
            # One stream per route variant
            for k, v in enumerate(variants):
                var_rate = rate * float(v["weight"])
                if var_rate <= 0:
                    continue
                period = 3600.0 / var_rate

                # Optional: small deterministic offset to avoid same-second clashes
                phase_offset = (hash(f"{origin}->{dest}#{k}") & 255) * 0.01  # 0..2.55 s

                inflow.add(
                    veh_type="human",
                    edge=origin,
                    period=period,
                    name=f"OD_{origin}_to_{dest}_k{k}",  # <- include variant index
                    depart_lane="best",
                    depart_speed=V_ENTER,
                    departPos="free",
                    begin=1.0,
                    end=HORIZON
                )

# print("[OD DEBUG] added flows per origin:",
#       {o: sum(1 for d in COL_ORDER if od_rates[o][d] > 0) for o in ROW_ORDER})

# print("Loaded OD inflows:")
# for f in inflow.get():  # works if InFlows has a get() or just print manually if not
#     print(f)



# # Uncomment when using synthetic data

# ------------------ Used when having set inflows for an edge ----------------------------------
# # inflows of vehicles are placed on all outer edges (listed here)
# outer_edges = []
# outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
# outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
# outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
# outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
# inflow = InFlows()
# for edge in outer_edges:
#     inflow.add(
#         veh_type="human",
#         edge=edge,
#         # period=period_for_edge(edge, HORIZON), # new
#         vehs_per_hour=inflow_for_edge(edge),
#         depart_lane="best",  # can be "best"
#         depart_speed=V_ENTER,
#         departPos="free",
#         )

# Define traffic light logic:
tl_logic = build_two_lane_hold_tls(
    n_rows=N_ROWS,
    n_cols=N_COLUMNS,
    vertical_lanes=2,
    horizontal_lanes=2
)


flow_params = dict(
    # name of the experiment
    exp_tag = f"grid_0_{N_ROWS}x{N_COLUMNS}_i{int(sum(EDGE_INFLOW.values()))}_multiagent",
    #exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

    # name of the flow environment the experiment is running on
    env_name=MultiTrafficLightGridPOEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # traffic light logic that is being used
    tls=tl_logic,

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1, #defaults to 0.2, maybe change again
        render=False,
        print_warnings=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=1, # NEW: was 1
        additional_params={
            "target_velocity": 50,
            "switch_time": 7,
            "num_observed": 4,
            "discrete": True,
            "tl_type": "controlled",   # can be actuated (SUMO controlled) or controlled (RL controlled)
            "num_local_edges": 4,
            "num_local_lights": 4,
            # "switch_penalty": 0.05,   # penalize each phase change
            # "standstill_gain": 1,     # same as your previous hardcoded gain
            # "all_red_penalty": 0.20,  # keep or set to 0.0 to disable
            "queue_window_m": 50.0,     # length of the queue window (m)
            "queue_speed_thresh": 0.2,  # threshold speed (m/s) to consider a vehicle queuing
            "queue_cap_per_lane": 10,   # max vehicles to count per lane
            "queue_penalty_gain": 0.2,  # tune: 0.1–0.4 typically works well
            "rou_file": ROU,
            "spawn_from_rou": False,     # False when using synthetic data
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow, # should be inflow if using inflows for spawning or None for real data
        additional_params={
            "speed_limit": V_ENTER,  # inherited from grid0 benchmark
            "horizontal_lanes": 4,       # Change to 2 for 2 lanes
            "vertical_lanes": 4,         # Change to 2 for 2 lanes
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
            },
            "turn_speed": 8.9,  # speed for turning maneuvers
            # >>> NEW: real coordinates <<<
            "node_coordinates": RAW_NODE_COORDS,
            # (optional) explicit control; default is True
            "normalize_to_center0": True,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization
    # or reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']

