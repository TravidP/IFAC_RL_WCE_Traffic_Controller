"""Multi-agent Sioux Falls network environment configuration file."""

import os

# Flow imports
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent.traffic_light_sioux_falls import MultiTrafficLightSiouxFallsPOEnv
from flow.networks.sioux_falls_network import SiouxFallsNetwork
from flow.core.params import (
    VehicleParams,
    NetParams,
    InitialConfig,
    EnvParams,
    SumoParams,
    VehicleParams,
    TrafficLightParams
)

from flow.controllers.SiouxFalls_routes import OD_SIOUX_FALLS, build_sioux_falls_od_rates
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env


# Experiment parameters
N_ROLLOUTS = 1  # number of rollouts per training iteration
N_CPUS = 1 # number of parallel workers

# Environment parameters
HORIZON = 900  # time horizon of a single rollout
V_ENTER = 13.89  # enter speed for departing vehicles 50 km/h


N_ZONES = len(OD_SIOUX_FALLS)
identity_weights = [1.0] * N_ZONES
SIOUX_FALLS_OD_RATES = build_sioux_falls_od_rates(
    OD_SIOUX_FALLS,
    scale=1.0,
    weights= identity_weights,
)

# Uncomment when using WCE to generate weights
# new_od_rates = build_sioux_falls_od_rates(OD_SIOUX_FALLS, HORIZON, weights=w)



# 1. Basic parameters ----------------------------------------------------------

# SUMO simulation parameters
sim_params = SumoParams(
    render=True,   # <-- this tells Flow to use the GUI so you can visualize it
    sim_step=1,    # 1 second per simulation step
)

# Environment parameters 
env_params = EnvParams(
    horizon= HORIZON,
    additional_params={
        "switch_time":6,
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
        "base_od_rates": SIOUX_FALLS_OD_RATES,
    }
)

# Initial config – we don’t need to specify edges because routes come from the .rou.xml
initial_config = InitialConfig(
    spacing='uniform', # was custom
    shuffle=True,
)

# VehicleParams – leave empty, because vehicles are defined in your SUMO templates
vehicles = VehicleParams()

# 2. Point to your SUMO files --------------------------------------------------

SUMO_DIR = "/home/sdc_joran/SiouxFalls_Network"

net_params = NetParams(
    template={
        # Network geometry
        "net": os.path.join(SUMO_DIR, "SiouxFalls.net.xml"),

        # Route file(s) – must be a *list*, even if you only have one file
        "rou": [os.path.join(SUMO_DIR, "SiouxFalls.rou.xml")],
        # Vehicle type definitions
        "vtype": os.path.join(SUMO_DIR, "SiouxFalls.vtype.xml"),
    },
    additional_params={
        "turn_speed": 8.9,

        # ---- dummy grid_array required by TrafficLightGridEnv base class ----
        "grid_array": {
            "short_length": 300.0,
            "long_length": 300.0,
            "inner_length": 300.0,
            "row_num": 0,
            "col_num": 0,
        },
    },
)

# 3. Define traffic light logic -----------------------------------------------
tl_logic = TrafficLightParams(baseline=False)

# 4. Pack everything into flow_params -----------------------------------------

flow_params = dict(
    # Name of the experiment
    exp_tag=   "multiagent_traffic_light_sioux_falls",

    # name of the flow environment the experiment is running on
    env_name=MultiTrafficLightSiouxFallsPOEnv,      

    # name of the network class the experiment is running on
    network=SiouxFallsNetwork,       

    # simulator that is used by the experiment
    simulator="traci",     # SUMO via TraCI

    # Traffi light logic that is being used
    tls=tl_logic,

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=sim_params,

    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization
    # or reset (see flow.core.params.InitialConfig)
    initial=initial_config,
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


