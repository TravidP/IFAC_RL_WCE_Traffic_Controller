"""Distributionally Robust Multi-Agent Traffic Light Controller (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent.traffic_light_grid_robust import DRMultiTrafficLightGridPOEnv   
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, SimLaneChangeController
from flow.controllers.routing_controllers import OuterEdgeODRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from .tls_presets import build_two_lane_hold_tls
from flow.controllers.Routes import Group_routes_G1


# Experiment parameters
N_ROLLOUTS = 1  # number of rollouts per training iteration
N_CPUS = 1 # number of parallel workers

# Environment parameters
HORIZON = 16*600  # time horizon of a single rollout
V_ENTER = 13.89  # enter speed for departing vehicles 50 km/h
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
N_ROWS = 3     # number of rows of bidirectional lanes
N_COLUMNS = 3  # number of columns of bidirectional lanes


# ===== Real-world node coordinates =====
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


# Define vehicles, but do not spawn in using initial config
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
    routing_controller=(OuterEdgeODRouter, {
        "use_hardcoded": True,
        "hardcoded_routes": Group_routes_G1,   
        "seed": 42,
        "use_precomputed_cycle": True,  
    }),
    num_vehicles=0              # Don't spawn in using initial config
    )

# Define traffic light logic:
tl_logic = build_two_lane_hold_tls(
    n_rows=N_ROWS,
    n_cols=N_COLUMNS,
    vertical_lanes=4,
    horizontal_lanes=4
)


flow_params = dict(
    # name of the experiment
    exp_tag = f"grid_0_{N_ROWS}x{N_COLUMNS}_robust_multiagent",

    # name of the flow environment the experiment is running on
    env_name=DRMultiTrafficLightGridPOEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # traffic light logic that is being used
    tls=tl_logic,

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1, 
        render=False,
        print_warnings=False,           # suppress SUMO warnings
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "switch_time": 5,                
            "tl_type": "controlled",         
            "discrete": True,                
            "num_observed": 4,
            "target_velocity": V_ENTER,
            "num_local_lights": 4,
            "num_local_edges": 4,
            "queue_window_m": 50.0,
            "queue_speed_thresh": 0.2,
            "queue_cap_per_lane": 10,
            "queue_penalty_gain": 0.1,
            # WCE-driven spawning
            "mix_window_s": 600,
            "worst_estimator_csv": "/home/sdc_joran/flow/worst_case_estimator_weights_36.csv",
            "group_csvs": [
                "/home/sdc_joran/Athena_Data/data/trip_00.csv",
                "/home/sdc_joran/Athena_Data/data/trip_01.csv",
                "/home/sdc_joran/Athena_Data/data/trip_02.csv",
                "/home/sdc_joran/Athena_Data/data/trip_03.csv",
                "/home/sdc_joran/Athena_Data/data/trip_04.csv",
                "/home/sdc_joran/Athena_Data/data/trip_05.csv",
                "/home/sdc_joran/Athena_Data/data/trip_06.csv",
                "/home/sdc_joran/Athena_Data/data/trip_07.csv",
                "/home/sdc_joran/Athena_Data/data/trip_08.csv",           
            ],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=None, 
        additional_params={
            "speed_limit": V_ENTER,  
            "horizontal_lanes": 4,       
            "vertical_lanes": 4,        
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
            },
            "turn_speed": 8.9,  # speed for turning maneuvers
            "node_coordinates": RAW_NODE_COORDS,
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

