import numpy as np
import pandas as pd
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from collections import defaultdict
import torch, os, json
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.registry import get_agent_class
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.utils.rllib import get_rllib_config, get_rllib_pkl
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.controllers.Routes import read_triplist_csv_to_od_rates, Group_routes_G1
import random
import ray
from ray.tune.registry import register_env
group_routes=Group_routes_G1
ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet

    # --- NEW: queue-length & per-lane stats ---
    # meters upstream from stop line to consider a vehicle "in queue"
    "queue_window_m": 50.0,
    # speed (m/s) below which a vehicle is considered stopped (queued)
    "queue_speed_thresh": 0.2,
    # normalization cap for per-lane queue length (cars per lane within window)
    "queue_cap_per_lane": 10,
    # reward penalty weight for queue length (>=0). Higher => stronger push to clear queues.
    "queue_penalty_gain": 0.1,
    "load_tl_policy": True,
    "policy_dir": "/home/sdc_joran/flow/all_weights_in_one_file.csv"

}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1
import gym
import torch
import torch.nn as nn
import tensorflow as tf


class TLTorchPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- policy net ----
        self.fc1 = nn.Linear(79, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.logits = nn.Linear(32, 8)

        # ---- value net ----
        self.v1 = nn.Linear(79, 128)
        self.v2 = nn.Linear(128, 64)
        self.v3 = nn.Linear(64, 32)
        self.v_out = nn.Linear(32, 1)

    def forward(self, obs):
        # policy forward
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        logits = self.logits(h)

        # value forward
        v = torch.tanh(self.v1(obs))
        v = torch.tanh(self.v2(v))
        v = torch.tanh(self.v3(v))
        v = self.v_out(v)

        return logits, v

    def act(self, obs):
        logits, _ = self.forward(obs)
        action = torch.argmax(logits).item()
        return action
    



def load_all_weights_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)

    layers = {}
    current_key = None
    buffer = []

    for _, row in df.iterrows():
        line = row[0]

        if isinstance(line, str) and line.startswith("Layer:"):
            # 保存上一层
            if current_key is not None:
                layers[current_key] = np.array(buffer)
            # 开启新层
            part = line.split(",")[0]
            current_key = part.replace("Layer: ", "").strip()
            buffer = []
        else:
            # 读取矩阵行
            if not row.isnull().all():
                buffer.append(row.dropna().values.astype(float))

    # 最后一层
    if current_key is not None:
        layers[current_key] = np.array(buffer)

    return layers

def assign_weights_to_model(model, layers):
    mapping = {
        "av/fc_1/kernel": ("fc1.weight", True),
        "av/fc_1/bias": ("fc1.bias", False),

        "av/fc_2/kernel": ("fc2.weight", True),
        "av/fc_2/bias": ("fc2.bias", False),

        "av/fc_3/kernel": ("fc3.weight", True),
        "av/fc_3/bias": ("fc3.bias", False),

        "av/fc_out/kernel": ("logits.weight", True),
        "av/fc_out/bias": ("logits.bias", False),

        "av/fc_value_1/kernel": ("v1.weight", True),
        "av/fc_value_1/bias": ("v1.bias", False),

        "av/fc_value_2/kernel": ("v2.weight", True),
        "av/fc_value_2/bias": ("v2.bias", False),

        "av/fc_value_3/kernel": ("v3.weight", True),
        "av/fc_value_3/bias": ("v3.bias", False),

        "av/value_out/kernel": ("v_out.weight", True),
        "av/value_out/bias": ("v_out.bias", False),
    }

    for key, array in layers.items():
        if key not in mapping:
            continue
        torch_key, is_weight = mapping[key]
        tensor = torch.tensor(array, dtype=torch.float32)

        # transpose kernel for PyTorch (TF kernel is [in, out])
        if is_weight:
            tensor = tensor.t()

        getattr(model, torch_key.split(".")[0]).__getattr__(torch_key.split(".")[1]).data[:] = tensor


class WorstEstimatorTrafficEnv(MultiTrafficLightGridPOEnv):
    """
    环境目标：
      - Agent 动作：选择一个 traffic group ID（1,2,3,...）
      - 内部运行：使用固定 traffic light 网络 (pretrained .pth)
      - 返回：负的 traffic performance（比如负平均速度、负throughput）
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        
        super().__init__(env_params, sim_params, network, simulator)

        # ---------------------------------------------------------------
        # 1) Traffic-light PPO policy loading settings
        # ---------------------------------------------------------------

        # whether to load a fixed traffic-light PPO policy
        self.load_tl_policy = env_params.additional_params.get("load_tl_policy", True)

        # path to PPO checkpoint directory (usually checkpoint_xxxx/)
        # expand "~" automatically
        default_policy_dir = (
           "/home/sdc_joran/flow/all_weights_in_one_file.csv"
        )
        self.policy_dir =env_params.additional_params.get("policy_dir", default_policy_dir)
        

        # PPO agent instance will be loaded lazily inside reset()
        self.trained_agent = None

        self.spawn_schedule = []
        
        # === 可选 traffic groups ===
        self.group_csvs = [
            "/home/sdc_joran/Athena_Data/data/trip_00.csv",
            "/home/sdc_joran/Athena_Data/data/trip_01.csv",
            "/home/sdc_joran/Athena_Data/data/trip_02.csv",
            "/home/sdc_joran/Athena_Data/data/trip_03.csv",
            "/home/sdc_joran/Athena_Data/data/trip_04.csv",
            "/home/sdc_joran/Athena_Data/data/trip_05.csv",
            "/home/sdc_joran/Athena_Data/data/trip_06.csv",
            "/home/sdc_joran/Athena_Data/data/trip_07.csv",
        ]
        self.group_mats = [read_triplist_csv_to_od_rates(p) for p in self.group_csvs]
        self.group_list = list(self.group_mats) 
        self.mix_window_s = 600
        self._next_mix_at = self.mix_window_s
        
        # === 当前 group 状态 ===
        self.current_group = None
        self.obs_traffic = None
        self.obs = None
        # initial_weights = [1,1,1,1] # or random initialization
        # initial_weights = np.random.rand(8).tolist()
        K = 3   # 随机激活 3 个 OD pattern
        mask = np.zeros(8)
        idx = np.random.choice(8, K, replace=False)
        mask[idx] = 1
        initial_weights = mask.tolist()

        self.current_group= self.group_mixture(initial_weights)


        print("[INIT] WorstEstimatorTrafficEnv initialized.")
        print(f"[INIT] traffic-light policy load   = {self.load_tl_policy}")
        print(f"[INIT] traffic-light policy dir    = {self.policy_dir}")
        print(f"[INIT] initial group mixture ready.")

    # def _load_trained_policy(self, result_dir, algo_name="PPO"):
    #     """
    #     Load RLlib TF checkpoint and convert into a pure PyTorch model (no RLlib!).
    #     """
    #     model = TLTorchPolicy(input_dim=79, hidden=[128, 64, 32], output_dim=8)
    #     model = load_tf_checkpoint_as_torch(result_dir, model)
    #     model.eval()
    #     return model


    def _load_trained_policy(self, csv_path, algo_name="PPO"):
        print("[INFO] Loading TL fixed policy from CSV...")

        # 1. 建立 PyTorch 网络结构
        model = TLTorchPolicy()

        # 2. 加载 CSV 权重
        layers = load_all_weights_from_csv(csv_path)
        assign_weights_to_model(model, layers)

        model.eval()
        return model





    def reset(self):
        """Reset 时默认不动，等待 worst estimator 选择 group。
        # updates the warmup procedure"""
        # TODO: Add warmup training
             # === 加载固定 traffic light 策略 ===
        # self.trained_agent = self._load_trained_policy(
        #     result_dir="~/ray_results/grid_0_3x3_i4800_multiagent/PPO_MultiTrafficLightGridPOEnv-v1_0a4a3ca0_2025-11-04_13-51-54_yhihbzy/checkpoint_3454",
        #     algo_name="PPO"
        # )
        if self.trained_agent is None and self.load_tl_policy:
            print("[INFO] Loading fixed TL RL policy...")
            self.trained_agent = self._load_trained_policy(
                self.policy_dir,
                algo_name="PPO"
            )
        obs = super().reset()
        
        self.sim_time = 0.0
        # print("\n[DEBUG] RESET OBSERVATION =====================")
        # print("obs:", obs)
        # print("type(obs):", type(obs))
        # if isinstance(obs, dict):
        #     for k, v in obs.items():
        #         print(f"  agent_id = {k}, type = {type(v)}, shape = {getattr(v, 'shape', None)}")
        # print("================================================\n")
                
        return obs 
    
    @property
    def observation_space(self):
        return Box(low=0., high=1., shape=(18,), dtype=np.float32)
        # change the observation
    @property
    def action_space(self):
        """See class definition."""
        # To do list: the output will be an integer in {0, 1, 2, 3, 4, 5, 6, 7}
    
        return Box(
            low=0,
            high=1,
            shape=(8,),
            dtype=np.float32)
    
    def get_state(self):
        """
        Centralized global observation for a single-agent controller.

        Each intersection contributes two normalized features:
            [avg_speed_norm, avg_density_norm]
        Normalization scheme matches get_traffic_state():
            - speed_norm = mean(vehicle_speed) / max_speed
            - density_norm = (num_vehicles / (edge_length * num_lanes)) / 0.2
            (where 0.2 veh/m/lane ≈ 5m headway, typical saturation density)

        Final observation is concatenated over all intersections:
            shape = (num_tls * 2,)
        """

        # --- Traffic light (intersection) IDs ---
        tl_ids = self.k.traffic_light.get_ids()
  

        # --- Global limits for normalization ---
        max_speed = max(self.k.network.speed_limit(e) for e in self.k.network.get_edge_list())

  
        # --- Store global features ---
        all_features = []

        for tl_id in tl_ids:
            idx = int(tl_id.split("center")[1])
            local_edges = self.network.node_mapping[idx][1]

            local_speeds = []
            local_densities = []

            for edge in local_edges:
                nlanes = self.k.network.num_lanes(edge)
                edge_len_m = self.k.network.edge_length(edge)
                vehs = self.k.vehicle.get_ids_by_edge(edge)

                if len(vehs) > 0:
                    # --- 速度归一化 ---
                    avg_v = np.mean([self.k.vehicle.get_speed(v) for v in vehs]) / max_speed

                    # --- 本车道密度 ---
                    density = len(vehs) / max(edge_len_m * nlanes, 1e-6)  # veh/m/lane
                    density_norm = np.clip(density / 0.2, 0.0, 1.0)  # 0.2 veh/m/lane ≈ 饱和密度
                else:
                    avg_v, density_norm = 0.0, 0.0

                local_speeds.append(avg_v)
                local_densities.append(density_norm)

            # --- 取该路口平均值 ---
            avg_speed_norm = np.mean(local_speeds) if local_speeds else 0.0
            avg_density_norm = np.mean(local_densities) if local_densities else 0.0

            all_features.extend([avg_speed_norm, avg_density_norm])

        # --- 拼接并归一化 ---
        obs = np.array(all_features, dtype=np.float32)
        np.clip(obs, 0.0, 1.0, out=obs)
        self.obs = obs
        print("DEBUG: Global obs for WCE:", obs)
        return {"worst":obs}


    def get_traffic_state(self):

        """
        Custom observation structure.

        Each traffic light agent observes:
        1. 8 incoming lanes, each described by 8 features:
           [lane_number, first_car_dist, first_car_speed, second_car_dist, second_car_speed, lane_density, lane_avg_speed, lane_queue_frac]
        2. Self + 4 neighboring traffic lights:
           [last_change_time, current_phase, currently_yellow]
        Total observation dim = 8 * 8 + 5 * 3 = 79.
        """

        obs = {}
        max_speed = max(
            self.k.network.speed_limit(e) for e in self.k.network.get_edge_list()
        )
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(
            grid_array["short_length"],
            grid_array["long_length"],
            grid_array["inner_length"],
        )

        # some params for density calculation
        queue_window_m = float(
            self.env_params.additional_params.get("queue_window_m", 50.0)
        )
        v_stop = float(
            self.env_params.additional_params.get("queue_speed_thresh", 0.2)
        )
        Q_CAP = int(
            self.env_params.additional_params.get("queue_cap_per_lane", 10)
        )

        node_to_edges = self.network.node_mapping
        num_tls = len(self.k.traffic_light.get_ids())
        lane_id_den = float(max(1, num_tls * 8 - 1))   # total lanes - 1, maps last lane to 1.0
        all_agents_observed_ids = []
        for rl_id in self.k.traffic_light.get_ids():
            rl_idx = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_idx][1]
            # print(f"DEBUG: Agent {rl_id} (idx {rl_idx}) local edges: {local_edges}")
            lane_features = []

            # === Gather lane-level features (up to 8 lanes) ===
            lane_count = 0
            observed_ids_agent = []
            for edge in local_edges:
                nlanes = self.k.network.num_lanes(edge)
                edge_len_m = self.k.network.edge_length(edge)
                veh_ids = self.k.vehicle.get_ids_by_edge(edge)

                # bucket vehicles by lane index
                by_lane = [[] for _ in range(nlanes)]
                for vid in veh_ids:
                    ln = self.k.vehicle.get_lane(vid)
                    if 0 <= ln < nlanes:
                        pos = self.k.vehicle.get_position(vid)
                        dist_to_light = max(0.0, edge_len_m - pos)
                        v = self.k.vehicle.get_speed(vid)
                        by_lane[ln].append((dist_to_light, v, vid))

                # ---- AGGREGATE (0,1) as SR group; (2,3) as LEFT group ----
                groups = [
                    [0, 1],  # SR group (two lanes together)
                    [2, 3],  # Left group (two lanes together)
                ]

                for lanes_in_group in groups:
                    # Assign a unique "virtual lane id" for normalization and ordering
                    lane_id = rl_idx * 8 + lane_count
                    lane_count += 1
                    if lane_count > 8:
                        break

                    # flatten vehicles across all lanes in the group
                    vehs = []
                    for ln in lanes_in_group:
                        if ln < nlanes:
                            vehs.extend(by_lane[ln])

                    # pick nearest and second-nearest across the group
                    vehs_sorted = sorted(vehs, key=lambda x: x[0])
                    if len(vehs_sorted) > 0:
                        first_dist, first_speed, vid1 = vehs_sorted[0]
                    else:
                        first_dist, first_speed, vid1 = edge_len_m, 0.0, None

                    if len(vehs_sorted) > 1:
                        second_dist, second_speed, vid2 = vehs_sorted[1]
                    else:
                        second_dist, second_speed, vid2 = edge_len_m, 0.0, None

                    observed_ids_agent.extend([vid1, vid2])

                    # normalize distances by THIS edge’s length
                    Lnorm = max(self.network.edge_lengths.get(edge, self.network.max_edge_length), 1e-3)
                    first_dist  /= Lnorm
                    second_dist /= Lnorm
                    first_speed /= max_speed
                    second_speed/= max_speed

                    # density & avg speed over the whole group
                    if len(vehs) > 0:
                        density = len(vehs) / max(edge_len_m, 1e-6)   # cars per meter (clipped later)
                        avg_v   = np.mean([v for _, v, _ in vehs]) / max_speed
                    else:
                        density, avg_v = 0.0, 0.0

                    # queue fraction within window (scale cap by number of lanes in the group)
                    q_cnt = 0
                    if len(vehs) > 0:
                        for d, v, _ in vehs:
                            if d <= queue_window_m and v <= v_stop:
                                q_cnt += 1
                    cap = Q_CAP * max(1, sum(1 for ln in lanes_in_group if ln < nlanes))  # e.g., 2*Q_CAP if both lanes exist
                    q_val = min(q_cnt / max(cap, 1), 1.0)

                    lane_features.append([
                        lane_id / lane_id_den,  # virtual-lane id in [0,1]
                        first_dist, first_speed,
                        second_dist, second_speed,
                        density, avg_v, q_val
                    ])

                    if lane_count >= 8:
                        break


            # pad to 8 lanes if fewer
            while len(lane_features) < 8:
                lane_features.append([0.0] * 8)
                observed_ids_agent.extend([None, None])

            lane_features = np.array(lane_features, dtype=np.float32).flatten()

            # === Collect self + 4 neighbor TL states ===
            def _get_idx(rel):
                idx = self._get_relative_node(rl_id, rel)
                return idx 

            neighbors = [rl_idx,
                         _get_idx("top"),
                         _get_idx("bottom"),
                         _get_idx("left"),
                         _get_idx("right")]

            tl_states = []
            for idx in neighbors:
                if idx == -1:
                    # Border defaults
                    tl_states.extend([
                        1.0,   # self.min_switch_time / self.min_switch_time
                        1.0,   # 5 / 5
                        1.0    # yellow flag
                    ])
                else:
                    tl_states.extend([
                        self.last_change[idx] / self.min_switch_time,  # normalized time since last change
                        self.last_phase[idx] / 8.0,                    # phase normalized to [0,1]
                        float(self.currently_yellow[idx])              # yellow flag
                    ])

            tl_states = np.array(tl_states, dtype=np.float32)

            # === Final concatenated observation ===
            observation = np.concatenate([lane_features, tl_states], dtype=np.float32)
            obs[rl_id] = observation
            all_agents_observed_ids.append(observed_ids_agent)
        
        self.observed_ids = all_agents_observed_ids  # used by additional_command()

        # ---- FINAL CLAMP to avoid float spillover (keeps within [0,1]) ----
        for agent_id, ob in obs.items():
            np.clip(ob, 0.0, 1.0, out=ob)

        # ---- DEBUG: Check if any observation value > 1.0 ----
        for agent_id, ob in obs.items():
            if np.any(ob > 1.0) or np.any(ob < 0.0):
                print(f"[WARN] Observation out of [0,1] range for {agent_id}: "
                    f"min={ob.min():.3f}, max={ob.max():.3f}")
        
        self.obs_traffic=obs
        return obs



    def traffic_step(self, traffic_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : dict of array_like
            agent's observation of the current environment
        reward : dict of floats
            amount of reward associated with the previous state/action pair
        done : dict of bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    accel_contr = self.k.vehicle.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))
            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self._apply_traffic_actions(traffic_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

        states = self.get_traffic_state()
        self.obs_traffic=states
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or (self.time_counter >= self.env_params.sims_per_step *
                     (self.env_params.warmup_steps + self.env_params.horizon)):
            done['__all__'] = True
        else:
            done['__all__'] = False
        infos = {key: {} for key in states.keys()}

        for rl_id in self.k.vehicle.get_arrived_rl_ids(self.env_params.sims_per_step):
            done[rl_id] = True
            states[rl_id] = np.zeros(self.observation_space.shape[0])

        
        return states, done, infos, crash

    def _apply_traffic_actions(self, traffic_actions):
        self._switched_tls = set()  # reset for this step

        # --- NEW: ensure we have a buffer for per-node transition states ---
        if not hasattr(self, "pending_trans_state"):
            # length large enough for tls indices used in rl_id (center<i>)
            n_tls = len(self.k.traffic_light.get_ids())
            self.pending_trans_state = [""] * max(n_tls, 64)  # 64 is a safe cushion; adjust if you like

        # --- helper: which movements can be preserved for a (cur_phase, next_phase) pair? ---
        def _preserve_indices_for_pair(grp, cur_p, next_p):
            """
            Returns a set of link indices that may remain GREEN during the yellow clearing
            for the (cur_p, next_p) pair, based on your list.
            We derive the exact indices by intersecting the per-side 'all' group with
            the axis+movement group (SR or LEFT).
            """
            # map your side names to the grouping keys present in grp
            side_key = {
                "bot":   "west_all",   # west  (incoming from west)
                "right": "south_all",  # south (incoming from south)
                "top":   "east_all",   # east  (incoming from east)
                "left":  "north_all",  # north (incoming from north)
            }

            # table of preservable movements (pair is undirected, works both directions)
            pair = frozenset({int(cur_p), int(next_p)})
            preserve_spec = {
                frozenset({4, 1}): [("bot",   "sr")],   # west SR
                frozenset({5, 0}): [("right", "sr")],   # south SR
                frozenset({6, 1}): [("top",   "sr")],   # east SR
                frozenset({7, 0}): [("left",  "sr")],   # north SR

                frozenset({2, 5}): [("right", "left")], # south lefts
                frozenset({2, 7}): [("left",  "left")], # north lefts
                frozenset({3, 4}): [("bot",   "left")], # west lefts
                frozenset({3, 6}): [("top",   "left")], # east lefts
            }

            if pair not in preserve_spec:
                return set()  # no special preserve for this pair -> fall back to ALL_Y

            keep = set()
            for side, mov in preserve_spec[pair]:
                side_indices = set(grp.get(side_key[side], []))

                if mov == "sr":
                    # SR: use axis SR group: NS_SR for north/south sides, EW_SR for west/east sides
                    if side in ("left", "right"):  # north/south sides
                        mov_set = set(grp.get("ns_sr_lane1", []))
                    else:  # bot/top (west/east sides)
                        mov_set = set(grp.get("ew_sr_lane1", []))
                else:  # mov == "left"
                    # LEFT: use axis left group: NS_LEFT for north/south sides, EW_LEFT for west/east sides
                    if side in ("left", "right"):  # north/south sides
                        mov_set = set(grp.get("ns_left_lane0", []))
                    else:  # bot/top (west/east sides)
                        mov_set = set(grp.get("ew_left_lane0", []))

                keep.update(side_indices.intersection(mov_set))

            return keep
 
        # --- DEBUG: print all actions received ---
        # print(f"\n=== RL ACTIONS RECEIVED === {rl_actions}")

        for rl_id, rl_action in traffic_actions.items():
            # parse node index (kept from your code)
            try:
                i = int(rl_id.replace("center", ""))
            except Exception:
                i = int("".join(ch for ch in rl_id if ch.isdigit()))
            node_id = f"center{i}"

            curr = self.k.traffic_light.get_state(node_id=node_id)
            L = len(curr)
            if L == 0:
                v = self.net_params.additional_params.get("vertical_lanes", 1)
                h = self.net_params.additional_params.get("horizontal_lanes", 1)
                L = 2 * v + 2 * h

            grp = self._classify_links_for_node(node_id)

            L = grp["L"]
            if L <= 0:
                continue

            # build your phase states exactly as before
            NS_SR   = self._build_state(L, grp["ns_sr_lane1"])
            EW_SR   = self._build_state(L, grp["ew_sr_lane1"])
            NS_LEFT = self._build_state(L, grp["ns_left_lane0"])
            EW_LEFT = self._build_state(L, grp["ew_left_lane0"])
            W_ALL   = self._build_state(L, grp["west_all"])
            S_ALL   = self._build_state(L, grp["south_all"])
            E_ALL   = self._build_state(L, grp["east_all"])
            N_ALL   = self._build_state(L, grp["north_all"])
            ALL_RED = "r" * L
            ALL_Y   = "y" * L

            phase_states = {
                0: NS_SR,
                1: EW_SR,
                2: NS_LEFT,
                3: EW_LEFT,
                4: W_ALL,
                5: S_ALL,
                6: E_ALL,
                7: N_ALL,
            }

            # robustly parse action (kept)
            a = rl_action
            if isinstance(a, (list, tuple, np.ndarray)):
                a = int(a[0])
            else:
                a = int(a)
            a = max(0, min(7, a))

            # graceful fallbacks for missing groups (kept)
            if a == 2 and len(grp["ns_left_lane0"]) == 0:
                a = 0
            if a == 3 and len(grp["ew_left_lane0"]) == 0:
                a = 1
            if a == 4 and len(grp["north_all"]) == 0:
                a = 0
            if a == 5 and len(grp["south_all"]) == 0:
                a = 0
            if a == 6 and len(grp["east_all"]) == 0:
                a = 1
            if a == 7 and len(grp["west_all"]) == 0:
                a = 1

            # --- DEBUG: show parsed action before applying ---
            # print(f"Node {node_id}: RL action={rl_action} -> parsed={a}")

            # ========== ONLY CHANGES START HERE ==========
            if self.currently_yellow[i] == 1:
                # still in clearing; either commit or keep showing the stored transition state
                self.last_change[i] += self.sim_step
                if self.last_change[i] >= self.min_switch_time:
                    state = phase_states.get(int(self.pending_phase[i]), ALL_RED)
                    self.k.traffic_light.set_state(node_id=node_id, state=state)
                    self.last_phase[i] = int(self.pending_phase[i])
                    self.currently_yellow[i] = 0
                    # (optional) clear stored transition
                    self.pending_trans_state[i] = ""
                    # print(f"  → Switch complete: phase {self.last_phase[i]} | state={state}")
                else:
                    # NEW: if we have a custom transition for this node, use it; otherwise ALL_Y
                    trans = self.pending_trans_state[i]
                    self.k.traffic_light.set_state(node_id=node_id, state=(trans if trans else ALL_Y))
                    # print(f"  → YELLOW clearing... ({trans if trans else ALL_Y})")
            else:
                if a != int(self.last_phase[i]):
                    tls_id = self.tl_ids[i] if hasattr(self, "tl_ids") else node_id
                    if not self.currently_yellow[i]:
                        self._switched_tls.add(tls_id)

                    # Decide whether this pair has preservable movements
                    cur_p = int(self.last_phase[i])
                    next_p = int(a)
                    keep_idxs = _preserve_indices_for_pair(grp, cur_p, next_p)

                    if keep_idxs:
                        # Build a transition state that keeps only those indices green.
                        cur_state   = curr if len(curr) == L else ("r" * L)
                        final_state = phase_states.get(next_p, ALL_RED)

                        trans_chars = ["r"] * L
                        for k in range(L):
                            c = cur_state[k]
                            f = final_state[k]
                            if c == "G" and f == "r":
                                trans_chars[k] = "G" if (k in keep_idxs) else "y"
                            elif c == "G" and f == "G":
                                trans_chars[k] = "G"
                            elif c != "G" and f == "G":
                                trans_chars[k] = "r"  # will go G after clearing
                            else:
                                trans_chars[k] = "r"

                        trans_state = "".join(trans_chars)
                        self.k.traffic_light.set_state(node_id=node_id, state=trans_state)
                        self.pending_trans_state[i] = trans_state  # remember for the clearing window
                    else:
                        # No special preserve → original behavior
                        self.k.traffic_light.set_state(node_id=node_id, state=ALL_Y)
                        self.pending_trans_state[i] = ""  # nothing special to replay

                    self.pending_phase[i] = a
                    self.last_change[i] = 0.0
                    self.currently_yellow[i] = 1
                    # print(f"  → Switching to phase {a} (YELLOW)")
                else:
                    self.k.traffic_light.set_state(node_id=node_id, state=phase_states[a])

    def _apply_group_flow(self, group_distribution, mode='replace'):
        """
        Update the departure (dispatch) plan according to the new group inflow.
        If mode='replace', clear the old inflow and use the new distribution for dispatch;
        If mode='accumulate', then add (accumulate) on top of the existing inflow.
        """
        traci_conn = self.k.kernel_api
        sim_time = traci_conn.simulation.getTime()

        if mode == 'replace':
            self.spawn_schedule.clear()

        new_schedule = []
        for origin, dests in group_distribution.items():
            for dest, rate in dests.items():
                if rate <= 0:
                    continue

                period = 3600.0 / rate
                next_spawn = sim_time + np.random.uniform(0, period)

                new_schedule.append({
                    "origin": origin,
                    "dest": dest,
                    "period": period,
                    "next_spawn": next_spawn
                })

        self.spawn_schedule.extend(new_schedule)
        print(f"[INFO] Installed new inflow schedule ({len(new_schedule)} OD pairs, mode={mode})")


    def step(self, rl_actions):
        """
        1. action ∈ [0, N_groups)
        2. Switch the traffic inflow group according to the action
        3. Run an episode under a fixed control policy
        4. Compute the reward
        """
                # ----- 自动适配单智能体动作 -----
        if not isinstance(rl_actions, dict):
            rl_actions = {"worst": rl_actions}
        # ---------------------------------
         # 1. 提取 worst-estimator 的 8维动作
        action = rl_actions["worst"]     # ← 这是 array([8])

        self.current_group= self.group_mixture(action)

        # 构建新的 inflow/OD 数据
        self._apply_group_flow(self.current_group)

        sim_steps = int(10 * 60 / self.sim_step)  # 10 min at Δt=1s
        total_wait_time = 0.0
        total_avg_speeds = []
        
        for _ in range(sim_steps):
        # === rollout simulation for fixed policy ===
            self._spawn_step()
            traffic_actions=self._simulate_with_fixed_policy(self.obs_traffic)
            self.obs_traffic,_,_,crash=self.traffic_step(traffic_actions)
            veh_ids = self.k.vehicle.get_ids()
            if not veh_ids:
                continue

            # 平均速度（当前 step）
            speeds = [self.k.vehicle.get_speed(vid) for vid in veh_ids]
            avg_speed = np.mean(speeds) 
            total_avg_speeds.append(avg_speed)
            

            # 累积等待时间
            for vid in veh_ids:
                v = self.k.vehicle.get_speed(vid)
                if v <= 0.1:  # 视为等待
                    total_wait_time=  total_wait_time+ self.sim_step
        
        # reward = negative of performance metric (e.g. negative avg speed)
        acc_speed = np.sum(total_avg_speeds) if total_avg_speeds else 0.0
        λ = 0
        reward = total_wait_time - λ * acc_speed
        print("reward:", reward)
        # observation: 可以是最近 group 的统计或空
        obs = self.get_state() 
        self.obs=obs
        next_observation = np.copy(obs)
        done = (self.time_counter >= self.env_params.sims_per_step *
                (self.env_params.warmup_steps + self.env_params.horizon)
                or crash)
        info = {}
        return (
            obs,               # obs
            {"worst": reward},                 # reward
            {"worst": done, "__all__": done},  # done
            {"worst": {}}                      # info
        )


    

    def group_mixture(self, weights):
        """
        根据多个 group（每个是 {origin: {dest: rate}} 的字典）按权重加权混合。
        weights: list/array, 长度等于 group_mats 数量
        return: dict(origin -> dict(dest -> mixed_rate))
        """
        # ---- 1️⃣ 归一化权重 ----
        w = np.maximum(np.array(weights, dtype=float), 0.0)
        if w.sum() == 0:
            w = np.ones_like(w)
        w /= w.sum()
        print(w)
        n_groups = len(self.group_mats)
        assert len(w) == n_groups, f"Expected {n_groups} weights, got {len(w)}"

        # ---- 2️⃣ 初始化空矩阵 ----
        mixed = defaultdict(lambda: defaultdict(float))

        # ---- 3️⃣ 逐组按权重加权求和 ----
        for gi, group in enumerate(self.group_mats):
            for origin, dests in group.items():
                for dest, rate in dests.items():
                    mixed[origin][dest] += w[gi] * rate

        # ---- 4️⃣ 转换成普通 dict ----
        mixed = {o: dict(dests) for o, dests in mixed.items()}

        # ---- 5️⃣ 调试输出（可选）----
        # print("[INFO] Mixed inflow distribution (first 3 origins):")
        # for o, dests in list(mixed.items())[:3]:
        #     print(f"  {o}: {dests}")

        return mixed
    



    def pick_weighted_route(self, origin, dest):
        """
        从 group_routes[origin][dest] 中按权重随机选择一条 edges 路径。
        返回 edges 列表，如 ["right0_0", "right1_0", "top1_0"]
        """
        if origin not in group_routes:
            raise KeyError(f"Origin {origin} not in route group")
        if dest not in group_routes[origin]:
            raise KeyError(f"Destination {dest} not in route group[{origin}]")

        candidates = group_routes[origin][dest]
        weights = [float(r.get("weight", 1.0)) for r in candidates]
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"No valid route weights for {origin}->{dest}")

        # 归一化 & 随机抽样
        r = random.random() * total
        cum = 0.0
        for cand, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                return cand["edges"]

        # 如果没选中（极少发生，防止浮点误差）
        return candidates[-1]["edges"]
    

    # def _spawn_step(self):
    #     """检查是否该发车，并插入新车"""
    #     traci_conn = self.k.kernel_api
    #     sim_time = traci_conn.simulation.getTime()

    #     for schedule in self.spawn_schedule:
    #         if sim_time >= schedule["next_spawn"]:
    #             origin, dest = schedule["origin"], schedule["dest"]
    #             veh_id = f"{origin}_{dest}_{int(sim_time)}"
    #             route_id = self.pick_weighted_route(origin,dest)

    #             # 如果没有该路线，则动态添加
    #             if route_id not in traci_conn.route.getIDList():
    #                 try:
    #                     edges = self.network.find_path(origin, dest)
    #                     traci_conn.route.add(route_id, edges)
    #                 except Exception:
    #                     continue

    #             try:
    #                 traci_conn.vehicle.add(
    #                     vehID=veh_id,
    #                     routeID=route_id,
    #                     typeID="human",
    #                     departLane="best",
    #                     departSpeed=13.89,
    #                 )
    #             except Exception:
    #                 pass

    #             # 安排下一辆
    #             schedule["next_spawn"] = sim_time + schedule["period"]
    def _spawn_step(self):
        """检查是否该发车，并插入新车（带完整 debug 输出）"""
        traci_conn = self.k.kernel_api
        sim_time = traci_conn.simulation.getTime()

        for schedule in self.spawn_schedule:

            # --- Debug: 打印 schedule 当前状态 ---
            # print(f"[SPAWN DEBUG] sim_time={sim_time:.1f} | origin={schedule['origin']} "
            #     f"dest={schedule['dest']} | next_spawn={schedule['next_spawn']} "
            #     f"| period={schedule['period']}")

            # 不到时间就跳过
            if sim_time < schedule["next_spawn"]:
                continue

            origin, dest = schedule["origin"], schedule["dest"]
            veh_id = f"{origin}_{dest}_{int(sim_time)}"

            # --- Debug 输出 ---
            # print(f"[SPAWN DEBUG] --> Ready to spawn veh_id={veh_id}")

      # 选择 route（其实返回的是 edges 列表）
            try:
                route_edges = self.pick_weighted_route(origin, dest)  # e.g. ["bot2_0","top2_0"]
                # print(f"[SPAWN DEBUG] route_edges chosen = {route_edges}")
            except Exception as e:
                print(f"[SPAWN ERROR] pick_weighted_route failed: {e}")
                continue

            # 生成稳定的 route ID（字符串）
            route_id_str = f"r__{origin}__{dest}__{'_'.join(route_edges)}"

            # 若该 route 尚未在 SUMO 注册，则添加
            if route_id_str not in traci_conn.route.getIDList():
                # 校验 edges 是否都存在
                invalid = [e for e in route_edges if e not in traci_conn.edge.getIDList()]
                if invalid:
                    print(f"[SPAWN ERROR] edges not in SUMO: {invalid}")
                    continue
                try:
                    traci_conn.route.add(route_id_str, route_edges)
                    # print(f"[SPAWN DEBUG] Route added to SUMO: id={route_id_str} | edges={route_edges}")
                except Exception as e:
                    print(f"[SPAWN ERROR] traci.route.add failed: {e}")
                    continue
            # else:
            #     print(f"[SPAWN DEBUG] route {route_id_str} already exists in SUMO.")

            # 发车
            try:
                traci_conn.vehicle.add(
                    vehID=veh_id,
                    routeID=route_id_str,
                    typeID="human",
                    departLane="best",
                    departSpeed="13.89",  # 这就是 50 km/h
                )
                # print(f"[SPAWN SUCCESS] Vehicle added: {veh_id} on {route_id_str}")
            except Exception as e:
                print(f"[SPAWN ERROR] vehicle.add failed: {e}")
                schedule["next_spawn"] = sim_time + schedule["period"]
                continue

            # 最终更新下一次发车时间
            schedule["next_spawn"] = sim_time + schedule["period"]
            # print(f"[SPAWN DEBUG] next_spawn updated → {schedule['next_spawn']}")

  
    # def _simulate_with_fixed_policy(self,observation):
    #     """
    #     用训练好的红绿灯策略计算当前 observation 的动作。
    #     兼容单智能体 / 多智能体。
    #     """
    #     actions = {}
    #     if isinstance(observation, dict):
    #         for agent_id, obs in observation.items():
    #             obs = np.array(obs, dtype=np.float32).reshape(-1)
    #             actions[agent_id] = self.trained_agent.compute_action(
    #                 obs, policy_id="av"  # policy_id 可根据你训练时定义调整
    #             )
    #     else:
    #         obs = np.array(observation, dtype=np.float32).reshape(-1)
    #         actions = self.trained_agent.compute_action(obs)
    #     return actions

    def _simulate_with_fixed_policy(self, observation):
        """
        Return traffic light actions from fixed PyTorch policy.
        """
        actions = {}
        model = self.trained_agent  # now this is TLTorchPolicy

        if isinstance(observation, dict):
            for agent_id, obs in observation.items():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                actions[agent_id] = model.act(obs_t)
        else:
            obs_t = torch.tensor(observation, dtype=torch.float32)
            actions = model.act(obs_t)

        return actions
