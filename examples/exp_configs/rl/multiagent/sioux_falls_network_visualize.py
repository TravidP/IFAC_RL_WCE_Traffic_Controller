"""Run the Sioux Falls SUMO network in Flow using the provided files."""

import os
import ray  # ensure Ray is imported before anything that might import pyarrow

# Import TestEnv directly to avoid pulling in all RL envs
from flow.envs.test import TestEnv
from flow.core.experiment import Experiment
from flow.networks import Network
from flow.core.params import (
    VehicleParams,
    NetParams,
    InitialConfig,
    EnvParams,
    SumoParams,
)

# 1. Basic parameters ----------------------------------------------------------

sim_params = SumoParams(
    render=True,   # opens SUMO-GUI for visualization
    sim_step=1,    # 1 second per simulation step
)

env_params = EnvParams()

# No need to specify initial edges; SUMO routes handle vehicle placement
initial_config = InitialConfig()

# Vehicles are defined in the SUMO templates (rou + vtype),
# so this can stay empty.
vehicles = VehicleParams()

# 2. Point to your Sioux Falls SUMO files -------------------------------------

SUMO_DIR = "/home/sdc_joran/SiouxFalls_Network"

net_params = NetParams(
    template={
        # Network geometry (built from nod/edg/con via netconvert)
        "net": os.path.join(SUMO_DIR, "SiouxFalls.net.xml"),

        # Route file(s): who drives where and when
        "rou": [os.path.join(SUMO_DIR, "SiouxFalls.rou.xml")],

        # Vehicle type definitions (the file you just created)
        "vtype": os.path.join(SUMO_DIR, "SiouxFalls.vtype.xml"),
    }
)

# 3. Pack everything into flow_params -----------------------------------------

flow_params = dict(
    exp_tag="sioux_falls_template",
    env_name=TestEnv,
    network=Network,
    simulator="traci",   # SUMO via TraCI
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

# Length of each rollout (in simulation steps)
flow_params["env"].horizon = 3600  # e.g. 1 simulated hour

# 4. Run the experiment --------------------------------------------------------

if __name__ == "__main__":
    exp = Experiment(flow_params)
    _ = exp.run(1)
