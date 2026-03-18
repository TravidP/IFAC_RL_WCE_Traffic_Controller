"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
from flow.envs.multiagent import MultiEnv
from collections import defaultdict
from xml.etree import ElementTree as ET
from flow.controllers.routing_controllers import convert_edge_id  # your helper



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
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1

class MultiTrafficLightGridPOEnv(TrafficLightGridPOEnv, MultiEnv):
    """Multiagent shared model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self._printed_init_once = False

        # required params
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.format(p))
        self.num_local_lights = env_params.additional_params.get("num_local_lights", 4)
        self.num_local_edges  = env_params.additional_params.get("num_local_edges", 4)

        # controller state
        self.last_phase    = np.zeros(self.num_traffic_lights, dtype=np.int8)
        self.pending_phase = np.zeros(self.num_traffic_lights, dtype=np.int8)

        # link-group caches (created once here)
        self._tls_groups = {}
        self._tls_groups_printed = set()

        # penalties and timeouts
        self._switch_penalty = self.env_params.additional_params.get("switch_penalty", 0.0)
        self._switched_tls = set()  # TLS IDs that STARTED a switch this step
        self._veh_wait_times = {}   # vid -> seconds stopped/slow (v <= 0.1 m/s)

        # minimum time between switches
        self.time_in_phase = np.zeros(self.num_traffic_lights, dtype=np.float32)

        # Initiliaze the routes and routes file
        self._rou_schedule = []
        self._rou_idx = 0
        self._spawn_from_rou = bool(env_params.additional_params.get("spawn_from_rou", False))
        rou_file = env_params.additional_params.get("rou_file", None)
        if self._spawn_from_rou and rou_file:
            self._rou_schedule = self._load_rou_schedule(rou_file)



    def reset(self):
        obs = super().reset()

        # clear caches for the new episode
        self._tls_groups.clear()
        self._tls_groups_printed.clear()

        # start in NS straight/right (phase 0), not yellow
        self.last_phase[:]       = 0
        self.pending_phase[:]    = 0
        self.currently_yellow[:] = 0
        self.last_change[:]      = self.min_switch_time # allow immediate commit
        # IDs of all signalized intersections
        self.tl_ids = self.k.traffic_light.get_ids()
        n_tls = len(self.tl_ids)

        # Create or reset the per-TL time-in-phase timer (seconds)
        if not hasattr(self, "time_in_phase") or len(self.time_in_phase) != n_tls:
            import numpy as np
            self.time_in_phase = np.zeros(n_tls, dtype=np.float32)
        else:
            self.time_in_phase.fill(0.0)


        # assert initial state per node
        for i in range(self.rows * self.cols):
            node_id = f"center{i}"
            grp = self._classify_links_for_node(node_id)

            # robust L
            L = grp["L"]
            if L <= 0:
                L = len(self.k.kernel_api.trafficlight.getControlledLinks(node_id)) or 8

            ns_sr = self._build_state(L, grp["ns_sr_lane1"])
            if ns_sr.count("G") == 0:
                # fallback: try EW straight/right; if thatâ€™s also empty, all-red
                ew_sr = self._build_state(L, grp["ew_sr_lane1"])
                ns_sr = ew_sr if ew_sr.count("G") else ("r" * L)

            self.k.traffic_light.set_state(node_id=node_id, state=ns_sr)

        # >>> ADD THIS <<<
        if getattr(self, "_spawn_from_rou", False):
            # re-run the same departure schedule each episode
            self._rou_idx = 0
            # optional: if you are NOT restarting SUMO on reset, give IDs a prefix per episode:
            # self._rou_episode = getattr(self, "_rou_episode", 0) + 1
            # self._rou_id_prefix = f"ep{self._rou_episode}_"
            # (and use this prefix when adding vehicles/routes)
        # <<< ADD THIS >>>

        return obs
        

    def _build_state(self, L, idx_list):
        s = ["r"] * L
        for k in idx_list:
            if 0 <= k < L:
                s[k] = "G"
        return "".join(s)

    @property
    def observation_space(self):
        return Box(low=0., high=1., shape=(79,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        # the output will be an integer in {0, 1, 2, 3, 4, 5, 6, 7}
        if self.discrete:
            return Discrete(8)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

    def _select_per_lane(self, edge_id, per_lane=2):
        """
        Return a list of up to (num_lanes(edge_id) * per_lane) vehicle IDs:
        the 'per_lane' closest-to-intersection vehicles from each lane.
        """
        vids = self.k.vehicle.get_ids_by_edge(edge_id)
        nlanes = self.k.network.num_lanes(edge_id)
        edge_len = self.k.network.edge_length(edge_id)

        by_lane = defaultdict(list)
        for vid in vids:
            ln = self.k.vehicle.get_lane(vid)
            pos = self.k.vehicle.get_position(vid)
            dist_to_node = max(0.0, edge_len - pos)
            by_lane[ln].append((dist_to_node, vid))

        selected = []
        for ln in range(nlanes):
            # closest first
            lane_sel = sorted(by_lane.get(ln, []), key=lambda t: t[0])[:per_lane]
            selected.extend([vid for _, vid in lane_sel])

        return selected  # may be <, =, or > self.num_observed (we’ll pad/trim later)

    def get_state(self):
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

        return obs



    
    # Extra helper
    def _classify_links_for_node(self, node_id):
        """
        Return indices grouped for 4-phase control using lane policy:
        lane 0: straight + right (SR)
        lane 1: left-only (L)
        Groups (kept compatible with existing call sites):
        - ns_sr_lane1 : incoming vertical (left/right) on lane0
        - ew_sr_lane1 : incoming horizontal (top/bot) on lane0
        - ns_left_lane0 : incoming vertical (left/right) on lane1
        - ew_left_lane0 : incoming horizontal (top/bot) on lane1
        - L : total number of controlled links
        """
        if not hasattr(self, "_tls_groups"):
            self._tls_groups = {}
        if node_id in self._tls_groups:
            return self._tls_groups[node_id]

        links = self.k.kernel_api.trafficlight.getControlledLinks(node_id)  # list of groups
        L = len(links)

        def split_lane(lane_id: str):
            # robustly split "..._<laneIndex>" even if names have extra underscores
            # e.g., "bot2_1_3" -> ("bot2_1", 3)
            edge, lane = lane_id.rsplit("_", 1)
            return edge, int(lane)

        def in_approach(edge_name: str):
            if edge_name.startswith("left"):   return "N"
            if edge_name.startswith("right"):  return "S"
            if edge_name.startswith("top"):    return "E"
            if edge_name.startswith("bot"):    return "W"
            return None

        def in_axis(edge_name: str):
            # vertical if left*/right*, horizontal if top*/bot*
            if edge_name.startswith("left") or edge_name.startswith("right"):
                return "NS"
            else:
                return "EW"

        def out_prefix(edge_name: str):
            # we only need the cardinal prefix to know the movement type
            if edge_name.startswith("left"):   return "left"
            if edge_name.startswith("right"):  return "right"
            if edge_name.startswith("top"):    return "top"
            if edge_name.startswith("bot"):    return "bot"
            return None

        def movement_type(appr: str, out_pref: str):
            """
            Return 'SR' for straight/right movements, 'L' for left-turn movements.
            Mapping follows the way specify_connections() builds out-edges.
            """
            if appr == "W":   # incoming 'bot'
                if out_pref in ("bot", "left"):  return "SR"  # straight E, right S
                if out_pref == "right":          return "L"   # left N
                if out_pref == "top":            return "L"   # u-turn W
            elif appr == "E": # incoming 'top'
                if out_pref in ("top", "right"): return "SR"  # straight W, right N
                if out_pref == "left":           return "L"   # left S
                if out_pref == "bot":            return "L"   # u-turn E
            elif appr == "N": # incoming 'left'
                if out_pref in ("left", "top"):  return "SR"  # straight S, right W
                if out_pref == "bot":            return "L"   # left E
                if out_pref == "right":          return "L"   # u-turn N
            elif appr == "S": # incoming 'right'
                if out_pref in ("right", "bot"): return "SR"  # straight N, right E
                if out_pref == "top":            return "L"   # left W
                if out_pref == "left":           return "L"   # u-turn S
            return None

        ns_sr, ew_sr, ns_left, ew_left = [], [], [], []
        north_all, south_all, east_all, west_all = [], [], [], []

        # SUMO groups multiple lane-to-lane connections that share a TLS index.
        # Use the *first* triple as representative for the movement (OK for SR/Left).
        for idx, group in enumerate(links):
            if not group:    # some TLS indices may have no connections
                continue
            in_lane_id, out_lane_id, via = group[0]
            in_edge, _  = split_lane(in_lane_id)
            out_edge, _ = split_lane(out_lane_id)

            appr = in_approach(in_edge)         # N/S/E/W
            axis = in_axis(in_edge)             # NS or EW
            outp = out_prefix(out_edge)         # 'left'/'right'/'top'/'bot'
            mtyp = movement_type(appr, outp)    # 'SR' or 'L'

            # Single-approach buckets (for N_ALL/S_ALL/E_ALL/W_ALL actions)
            if appr == "N":   north_all.append(idx)
            elif appr == "S": south_all.append(idx)
            elif appr == "E": east_all.append(idx)
            elif appr == "W": west_all.append(idx)

            # Axis + movement buckets (for NS_SR/EW_SR and NS_LEFT/EW_LEFT)
            if mtyp == "SR":
                if axis == "NS": ns_sr.append(idx)
                else:            ew_sr.append(idx)
            elif mtyp == "L":
                if axis == "NS": ns_left.append(idx)
                else:            ew_left.append(idx)


        groups = dict(
            ns_sr_lane1=ns_sr,
            ew_sr_lane1=ew_sr,
            ns_left_lane0=ns_left,
            ew_left_lane0=ew_left,
            north_all=north_all, south_all=south_all,
            east_all=east_all,   west_all=west_all,
            L=L,
        )

        self._tls_groups[node_id] = groups

        # # One-time debug
        # if not hasattr(self, "_tls_groups_printed"):
        #     self._tls_groups_printed = set()
        # if node_id not in self._tls_groups_printed:
        #     self._tls_groups_printed.add(node_id)
        #     print(f"[TLS {node_id}] L={L} "
        #         f"N/S/E/W={list(map(len,[north_all,south_all,east_all,west_all]))} "
        #         f"NS_SR/EW_SR={[len(ns_sr),len(ew_sr)]} "
        #         f"NS_L/EW_L={[len(ns_left),len(ew_left)]}")

        return groups

    def _apply_rl_actions(self, rl_actions):
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

        for rl_id, rl_action in rl_actions.items():
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
                    # print(f"  → Holding phase {a} | state={phase_states[a]}")``
            # ========== ONLY CHANGES END HERE ==========

    
    # def compute_reward(self, rl_actions, **kwargs):
    #     """Compute *local* rewards for each traffic light agent."""
    #     if rl_actions is None:
    #         return {}

    #     veh_ids = self.k.vehicle.get_ids()
    #     if len(veh_ids) == 0:
    #         # No vehicles = no reward signal
    #         return {rl_id: 0.0 for rl_id in rl_actions.keys()}

    #     # For normalization
    #     v_top = max(self.k.network.speed_limit(e) for e in self.k.network.get_edge_list())
    #     eps = 1e-6

    #     # === Compute per-agent rewards ===
    #     rews = {}

    #     for rl_id in rl_actions.keys():
    #         rl_idx = int(rl_id.split("center")[1])  # extract the index number
    #         local_edges = self.network.node_mapping[rl_idx][1]

    #         local_speeds = []
    #         # local_waits = []
    #         local_queue = 0
    #         total_lanes = 0

    #         for edge in local_edges:
    #             vehs_on_edge = self.k.vehicle.get_ids_by_edge(edge)
    #             total_lanes += self.k.network.num_lanes(edge)
    #             for vid in vehs_on_edge:
    #                 v = self.k.vehicle.get_speed(vid)
    #                 local_speeds.append(v)

    #                 # Waiting time or near-zero speed → penalize
    #                 # if hasattr(self, "_veh_wait_times"):
    #                 #     local_waits.append(self._veh_wait_times.get(vid, 0.0))
    #                 # else:
    #                 #     # Approximate wait from low speed
    #                 #     if v < 0.1:
    #                 #         local_waits.append(1.0)
    #                 #     else:
    #                 #         local_waits.append(0.0)

    #                 # Simple queue approximation
    #                 if v <= 0.2:
    #                     local_queue += 1

    #         # --- compute metrics for this agent ---
    #         if len(local_speeds) > 0:
    #             avg_speed = np.mean(local_speeds)
    #         else:
    #             avg_speed = 0.0

    #         # if len(local_waits) > 0:
    #         #     avg_wait = np.mean(local_waits)
    #         # else:
    #         #     avg_wait = 0.0

    #         # --- normalize ---
    #         speed_ratio = avg_speed / (v_top + eps)
    #         queue_ratio = local_queue / (total_lanes * 10.0 + eps)  # assume 10 cars/lane cap


    #         # --- final local reward ---
    #         # rew = (2.0 * speed_ratio) - (0.05 * avg_wait) - (0.2 * queue_ratio)
    #         rew = (2.0 * speed_ratio)- (0.2 * queue_ratio)

    #         # --- fairness term (optional) ---
    #         # fairness_penalty = (self.sim_time - self.last_change[rl_idx]) / (self.min_switch_time + eps)
    #         # rew -= 0.001 * fairness_penalty

    #         rews[rl_id] = rew

    #     return rews
    
    def compute_reward(self, rl_actions, **kwargs):
        """Compute *local* rewards for each traffic light agent."""
        if rl_actions is None:
            return {}

        veh_ids = self.k.vehicle.get_ids()
        if len(veh_ids) == 0:
            return {rl_id: 0.0 for rl_id in rl_actions.keys()}

        # --- NEW: update per-vehicle waiting timers ---
        sim_dt = getattr(self, "sim_step", 1.0)
        # prune missing vehicles
        for vid in list(self._veh_wait_times.keys()):
            if vid not in veh_ids:
                self._veh_wait_times.pop(vid, None)
        # update alive vehicles
        for vid in veh_ids:
            v = self.k.vehicle.get_speed(vid)
            if v <= 0.1:
                self._veh_wait_times[vid] = self._veh_wait_times.get(vid, 0.0) + sim_dt
            else:
                # reset when it moves again
                self._veh_wait_times[vid] = 0.0

        # For normalization
        v_top = max(self.k.network.speed_limit(e) for e in self.k.network.get_edge_list())
        eps = 1e-6

        # penalty weight (tunable)
        wait_penalty_gain = float(self.env_params.additional_params.get("wait_penalty_gain", 0.0))
        # optional safety cap to avoid explosion; set high if you really want it unbounded
        t_cap = float(self.env_params.additional_params.get("wait_time_cap_s", 60.0))

        rews = {}
        # we'll rely on observed_ids computed in get_state; if missing, we'll rebuild on the fly
        have_obs_ids = hasattr(self, "observed_ids") and isinstance(self.observed_ids, list)

        for rl_id in rl_actions.keys():
            rl_idx = int(rl_id.split("center")[1])
            local_edges = self.network.node_mapping[rl_idx][1]

            local_speeds = []
            local_queue = 0
            total_lanes = 0

            # --- base speed/queue part (unchanged) ---
            for edge in local_edges:
                vehs_on_edge = self.k.vehicle.get_ids_by_edge(edge)
                total_lanes += self.k.network.num_lanes(edge)
                for vid in vehs_on_edge:
                    v = self.k.vehicle.get_speed(vid)
                    local_speeds.append(v)
                    if v <= 0.2:
                        local_queue += 1

            avg_speed = np.mean(local_speeds) if local_speeds else 0.0
            speed_ratio = avg_speed / (v_top + eps)
            queue_ratio = local_queue / (total_lanes * 10.0 + eps)

            # --- NEW: first-car exponential waiting penalty ---
            wait_pen_sum = 0.0

            def _sum_firstcar_penalty_for_agent():
                nonlocal wait_pen_sum
                # Case A: use IDs cached by get_state (fast path)
                if have_obs_ids and rl_idx < len(self.observed_ids):
                    ids_list = self.observed_ids[rl_idx]
                    # observed_ids stores [first, second] for SR, then [first, second] for LEFT, per edge
                    # So every 2 elements, the 0th is the first car of that group
                    for j in range(0, len(ids_list), 2):
                        vid1 = ids_list[j]
                        if vid1 is None:
                            continue
                        v1 = self.k.vehicle.get_speed(vid1)
                        if v1 <= 0.1:
                            t = min(self._veh_wait_times.get(vid1, 0.0), t_cap)
                            wait_pen_sum += float(np.exp(0.001 * t))
                    return

                # Case B: rebuild first-car per group here (fallback)
                for edge in local_edges:
                    nlanes = self.k.network.num_lanes(edge)
                    edge_len = self.k.network.edge_length(edge)
                    # group SR: lanes [0,1]; group LEFT: lanes [2,3]
                    for lanes_in_group in ([0, 1], [2, 3]):
                        # pick nearest vehicle across lanes in the group
                        best = None
                        for ln in lanes_in_group:
                            if ln >= nlanes:
                                continue
                            vids = [vid for vid in self.k.vehicle.get_ids_by_edge(edge)
                                    if self.k.vehicle.get_lane(vid) == ln]
                            if not vids:
                                continue
                            # nearest = min dist_to_stopline
                            for vid in vids:
                                pos = self.k.vehicle.get_position(vid)
                                dist_to_light = max(0.0, edge_len - pos)
                                if (best is None) or (dist_to_light < best[0]):
                                    best = (dist_to_light, vid)
                        if best is not None:
                            vid1 = best[1]
                            v1 = self.k.vehicle.get_speed(vid1)
                            if v1 <= 0.1:
                                t = min(self._veh_wait_times.get(vid1, 0.0), t_cap)
                                wait_pen_sum += float(np.exp(0.001 * t))

            _sum_firstcar_penalty_for_agent()

            # --- final reward ---
            rew = (2.0 * speed_ratio) - (0.2 * queue_ratio) - (wait_penalty_gain * wait_pen_sum)
            rews[rl_id] = rew

        return rews

  
    def _load_rou_schedule(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        schedule = []
        for veh in root.findall(".//vehicle"):
            vid = veh.attrib["id"]
            depart = float(veh.attrib.get("depart", "0"))
            # we ignore the XML vType and just use Flow's default 'human'
            vtype = "human"
            route_el = veh.find("route")
            edges = route_el.attrib["edges"].split()
            edges = [convert_edge_id(e) for e in edges]  # <- your mapping
            schedule.append((depart, vid, vtype, edges))
        schedule.sort(key=lambda x: x[0])
        return schedule

    def additional_command(self):
        # keep the original behavior
        super().additional_command()

        if not self._spawn_from_rou or not self._rou_schedule:
            return

        # current simulation time from SUMO (seconds)
        t = self.k.kernel_api.simulation.getTime()

        # spawn all vehicles whose depart time has passed
        while self._rou_idx < len(self._rou_schedule) and self._rou_schedule[self._rou_idx][0] <= t + 1e-6:
            depart, vid, vtype, edges = self._rou_schedule[self._rou_idx]
            rid = f"r_{vid}"
            try:
                # define the route and then add the vehicle now
                self.k.kernel_api.route.add(rid, edges)
                self.k.kernel_api.vehicle.addFull(
                    vehID=vid, routeID=rid, typeID=vtype,
                    departPos="base", departSpeed="max", departLane="best"
                )
            except Exception as e:
                print(f"[ROUSpawner] failed to add {vid} at t={t}: {e}")
            self._rou_idx += 1



    # def additional_command(self):
    #     """See class definition."""
    #     for veh_ids in self.observed_ids:
    #         for veh_id in veh_ids:
    #             if veh_id is not None:  # NEW guard
    #                 self.k.vehicle.set_observed(veh_id)