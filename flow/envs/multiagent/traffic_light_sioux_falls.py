# multiagent_traffic_light_sioux_falls_env.py

from collections import defaultdict
import numpy as np

from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv


class MultiTrafficLightSiouxFallsPOEnv(MultiTrafficLightGridPOEnv):
    """
    Multi-agent TL environment for the Sioux Falls network.

    It reuses all logic from MultiTrafficLightGridPOEnv (79-dim obs,
    yellow phases, reward, etc.), but adapts:
      - how TL indices are computed (no 'center' prefix),
      - how node_mapping (idx -> incoming edges) is built,
      - how neighbors are handled (for now: no neighbors, only self TL;
        we can refine this later once the SUMO network is finalized).
    """

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)

        # Will be filled after SUMO is up and we know all TL/edge IDs
        self._sioux_structures_built = False
        self._id_to_idx = {}          # tl_id -> 0..num_tls-1
        self._idx_to_id = {}          # inverse mapping
        self._neighbors_by_idx = {}   # idx -> list of neighbor idx (max 4)

        # OD-based spawning state
        self._od_rates = env_params.additional_params.get("base_od_rates", {})
        self._veh_counter = 0
        self._route_ids = set()

    # ------------------------------------------------------------------
    # Define excluded agents and neighbour mapping
    # ------------------------------------------------------------------
    EXCLUDED_TL_IDS = {"1", "2", "7", "13"}
    SIOUX_NEIGHBORS = {
        "3": ["1", "4", "12"],
        "4": ["3", "5", "11"],
        "5": ["4", "6", "9"],
        "6": ["2", "5", "8"],
        "8": ["6", "7", "9", "16"],
        "9": ["5", "8", "10"],
        "10": ["9", "11", "15", "16"],
        "11": ["4", "10", "12", "14"],
        "12": ["3", "11", "13"],
        "14": ["11", "15", "23"],
        "15": ["10", "14", "19", "22"],
        "16": ["8", "10", "18", "19"],
        "18": ["7", "16", "20"],
        "19": ["15", "16", "20"],
        "20": ["18", "19", "21", "22"],
        "21": ["20", "22", "24"],
        "22": ["15", "20", "21", "23"],
        "23": ["14", "22", "24"],
        "24": ["13", "21", "23"],
    }

    SIOUX_EDGE_DIR = {
        # node 3
        "1": "N",
        "7": "E",
        "34": "S",
        # node 4
        "10": "E",
        "30": "S",
        "5": "W",
        # node 5
        "14": "E",
        "22": "S",
        "8": "W",
        # node 6
        "3": "N",
        "18": "S",
        "11": "W",
        # node 8
        "15": "N",
        "26": "E",
        "46": "S",
        "23": "W",
        # node 9
        "12": "N",
        "20": "E",
        "25": "S",
        # node 10
        "24": "N",
        "47": "E",
        "42": "S",
        "31": "W",
        # node 11
        "9": "N",
        "26": "E",
        "39": "S",
        "35": "W",
        # node 12
        "6": "N",
        "32": "E",
        "37": "S",
        # node 14
        "33": "N",
        "43": "E",
        "70": "S",
        # node 15
        "27": "N",
        "56": "E",
        "66": "S",
        "40": "W",
        # node 16
        "21": "N",
        "54": "E",
        "10002": "S",
        "28": "W",
        # node 18
        "17": "N",
        "59": "S",
        "49": "W",
        # node 19
        "10001": "N",
        "60": "S",
        "44": "W",
        # node 20
        "67": "N",
        "58": "E",
        "55": "S",
        "63": "W",
        # node 21
        "68": "N",
        "61": "E",
        "74": "W",
        # node 22
        "45": "N",
        "62": "E",
        "64": "S",
        "71": "W",
        # node 23
        "41": "N",
        "69": "E",
        "75": "S",
        # node 24
        "72": "N",
        "65": "E",
        "38": "W",
    }
    # ------------------------------------------------------------------
    # 1) Build Sioux-Falls specific mappings once SUMO is running
    # ------------------------------------------------------------------
    def _build_sioux_structures(self):
        """
        Build Sioux-Falls specific mappings once SUMO is running.

        - defines ordering of RL-controlled TLs
        - builds node_mapping[idx] = (tl_id, incoming_edge_ids)
        using junction incLanes from the SUMO net.xml
        - builds neighbor map using SIOUX_NEIGHBORS
        - loads all route definitions from the SiouxFalls.rou.xml
        """
        from xml.etree import ElementTree as ET

        # 1) TL IDs from SUMO
        all_tl_ids = list(self.k.traffic_light.get_ids())
        if len(all_tl_ids) == 0:
            raise RuntimeError(
                "No traffic lights found in SiouxFalls.net.xml. "
                "Regenerate the net with junctions type='traffic_light' "
                "(except nodes 1,2,7,13)."
            )

        tl_ids = [tid for tid in all_tl_ids if tid not in self.EXCLUDED_TL_IDS]

        # deterministic order (numeric if possible)
        try:
            tl_ids.sort(key=lambda x: int(x))
        except Exception:
            tl_ids.sort()

        self.tl_ids = tl_ids
        self.num_traffic_lights = len(tl_ids)

        self._id_to_idx = {tid: i for i, tid in enumerate(tl_ids)}
        self._idx_to_id = {i: tid for tid, i in self._id_to_idx.items()}

        # 2) Parse net.xml to get incoming edges per junction from incLanes
        net_path = self.network.net_params.template.get("net", None)
        if net_path is None:
            raise RuntimeError("Missing net file path: net_params.template['net'].")

        root = ET.parse(net_path).getroot()

        inc_edges_by_node = {}
        for j in root.findall("junction"):
            jid = j.get("id", "")
            if not jid or jid.startswith(":"):
                continue

            inc_lanes = j.get("incLanes", "")
            lanes = inc_lanes.split() if inc_lanes else []

            edges = []
            seen = set()
            for ln in lanes:
                if "_" not in ln:
                    continue
                edge_id = ln.rsplit("_", 1)[0]
                if edge_id not in seen:
                    edges.append(edge_id)
                    seen.add(edge_id)

            try:
                edges.sort(key=lambda x: int(x))
            except Exception:
                edges.sort()

            inc_edges_by_node[jid] = edges

        # 3) node_mapping: idx -> (tl_id, incoming_edge_ids)
        node_mapping = {}
        for tl_id, idx in self._id_to_idx.items():
            node_mapping[idx] = (tl_id, inc_edges_by_node.get(tl_id, []))
        self.network.node_mapping = node_mapping

        # 4) Neighbors
        self._neighbors_by_idx = {}
        for tl_id, idx in self._id_to_idx.items():
            neighbor_ids = self.SIOUX_NEIGHBORS.get(tl_id, [])
            self._neighbors_by_idx[idx] = [
                self._id_to_idx[nid] for nid in neighbor_ids
                if nid in self._id_to_idx
            ]

        # ------------------------------------------------------------------
        # 5) ROUTE IMPORT: read SiouxFalls.rou.xml and register all routes
        # ------------------------------------------------------------------
        traci = self.k.kernel_api
        route_ids = set(traci.route.getIDList())

        rou_files = self.network.net_params.template.get("rou", [])
        if isinstance(rou_files, str):
            rou_files = [rou_files]

        for rou_path in rou_files:
            try:
                r_root = ET.parse(rou_path).getroot()
            except Exception as e:
                print(f"[Sioux OD DEBUG] Failed to parse rou file {rou_path}: {e}",
                    flush=True)
                continue

            for r in r_root.findall("route"):
                rid = r.get("id")
                edges_str = r.get("edges", "")
                if not rid or not edges_str:
                    continue

                # only add if SUMO doesn't know this ID yet
                if rid in route_ids:
                    continue

                edges = edges_str.split()
                try:
                    traci.route.add(rid, edges)
                    route_ids.add(rid)
                except Exception:
                    # ignore duplicates or invalid edges
                    pass

        self._route_ids = route_ids
        self._sioux_structures_built = True

        # ------------------------------------------------------------------
        # 6) OD spawn bookkeeping + one-time debug print
        # ------------------------------------------------------------------
        self._od_rates = self.env_params.additional_params.get("base_od_rates", {})
        self._veh_counter = 0

        if not getattr(self, "_printed_od_debug", False):
            total_pairs = sum(len(dests) for dests in self._od_rates.values())
            # print(
            #     "[Sioux OD DEBUG] origins:", len(self._od_rates),
            #     "pairs:", total_pairs,
            #     "route_ids:", len(self._route_ids),
            #     "sample route ids:", list(self._route_ids)[:10],
            #     flush=True,
            # )
            if self._od_rates:
                first_o = next(iter(self._od_rates))
                # print(
                #     "[Sioux OD DEBUG] sample origin", first_o,
                #     ":", self._od_rates[first_o],
                #     flush=True,
                # )
            self._printed_od_debug = True




    # ------------------------------------------------------------------
    # 2) Ensure mappings exist on reset
    # ------------------------------------------------------------------
    def reset(self):
        # Force rebuild on every reset (safe)
        self._sioux_structures_built = False

        # This will internally call self.get_state(), which is now safe
        _ = TrafficLightGridPOEnv.reset(self)

        # Ensure initialized (again, cheap/idempotent)
        self._ensure_sioux_initialized()

        # Optional: print TLS IDs once (this prints in the worker log!)
        if not getattr(self, "_printed_tls", False):
            # print("TLS IDs:", self.k.kernel_api.trafficlight.getIDList(), flush=True)
            self._printed_tls = True

        # Set starting phase for all controlled TLS
        for i, tl_id in enumerate(self.tl_ids):
            try:
                self.k.kernel_api.trafficlight.setPhase(tl_id, 0)
            except Exception:
                pass
            self.last_phase[i] = 0

        # OD spawn setup
        self._od_rates = self.env_params.additional_params.get("base_od_rates", {})
        self._veh_counter = 0
        try:
            self._route_ids = set(self.k.kernel_api.route.getIDList())
        except Exception:
            self._route_ids = set()

        return self.get_state()




    # ------------------------------------------------------------------
    # 3) OD-based vehicle spawning helpers
    # ------------------------------------------------------------------
    def _pick_route_id_for_od(self, origin_id: str, dest_id: str):
        """
        Choose an existing SUMO route ID for a given (origin, dest) pair.

        We first try a routeDistribution ID "rd_o_d", then fall back to
        a simple route "r_o_d" if that exists. If neither exists, return
        None and skip spawning for that pair.
        """
        if not self._route_ids:
            # route IDs may not be cached yet
            try:
                self._route_ids = set(self.k.kernel_api.route.getIDList())
            except Exception:
                return None

        rd_id = f"rd_{origin_id}_{dest_id}"
        r_id = f"r_{origin_id}_{dest_id}"

        if rd_id in self._route_ids:
            return rd_id
        if r_id in self._route_ids:
            return r_id

        # No suitable route found in the .rou file
        return None
    
    def _spawn_vehicles_from_od(self):
        od_rates = getattr(self, "_od_rates", None)
        if not od_rates:
            return

        dt = float(getattr(self, "sim_step", 1.0))
        if dt <= 0:
            dt = 1.0
        factor = dt / 3600.0

        traci = self.k.kernel_api
        type_id = str(self.env_params.additional_params.get("spawn_type_id", "car"))

        def pick_route_id(o: str, d: str):
            rd = f"rd_{o}_{d}"
            r = f"r_{o}_{d}"
            if rd in self._route_ids: return rd
            if r in self._route_ids: return r
            prefix = f"rd_{o}_{d}_"
            cand = [rid for rid in self._route_ids if rid.startswith(prefix)]
            return np.random.choice(cand) if cand else None

        for o, dests in od_rates.items():
            for d, rate_per_hour in dests.items():
                lam = float(rate_per_hour) * factor
                if lam <= 0:
                    continue
                n_new = np.random.poisson(lam)
                if n_new <= 0:
                    continue

                route_id = pick_route_id(str(o), str(d))
                if route_id is None:
                    continue

                for _ in range(int(n_new)):
                    veh_id = f"od_{o}_{d}_{self._veh_counter}"
                    self._veh_counter += 1
                    try:
                        traci.vehicle.addFull(
                            vehID=veh_id,
                            routeID=route_id,
                            typeID=type_id,
                            departLane="best",
                            departPos="base",
                            departSpeed="max",
                        )
                    except Exception:
                        try:
                            traci.vehicle.add(
                                vehID=veh_id,
                                routeID=route_id,
                                typeID=type_id,
                                depart="now",
                                departLane="best",
                                departPos="base",
                                departSpeed="max",
                            )
                        except Exception:
                            pass


    def _ensure_sioux_initialized(self):
        """Build mappings + size TL state arrays if they are missing or wrong size."""
        if not getattr(self, "_sioux_structures_built", False):
            self._build_sioux_structures()

        n = len(getattr(self, "tl_ids", []))
        if n == 0:
            raise RuntimeError(
                "No RL-controlled traffic lights found (tl_ids empty). "
                "Check that SUMO exposes TLS IDs and EXCLUDED_TL_IDS filtering is correct."
            )

        # (Re)initialize TL state arrays if missing or wrong length
        if (not hasattr(self, "last_change")) or (getattr(self.last_change, "size", 0) != n):
            self.num_traffic_lights = n
            self.last_phase        = np.zeros(n, dtype=np.int8)
            self.pending_phase     = np.zeros(n, dtype=np.int8)
            self.currently_yellow  = np.zeros(n, dtype=np.int8)
            self.last_change       = np.full(n, self.min_switch_time, dtype=np.float32)
            self.time_in_phase     = np.zeros(n, dtype=np.float32)
            self.pending_trans_state = [""] * n
            self._switched_tls = set()

    # ------------------------------------------------------------------
    # 4) Override get_state to be ID-agnostic and use our neighbors map
    # ------------------------------------------------------------------
    def get_state(self):
        """
        Copy of your 79-dim get_state, with two small changes:
          - instead of rl_idx = int(rl_id.split('center')[...]),
            we use rl_idx = self._id_to_idx[rl_id]
          - neighbors are taken from self._neighbors_by_idx, not from
            _get_relative_node (grid-only).
        """
        self._ensure_sioux_initialized()

        # Make sure mappings are ready
        if not self._sioux_structures_built:
            self._build_sioux_structures()

        obs = {}
        max_speed = max(
            self.k.network.speed_limit(e) for e in self.k.network.get_edge_list()
        )

        # params for density / queue calculations
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
        num_tls = len(self.tl_ids)
        lane_id_den = float(max(1, num_tls * 8 - 1))

        all_agents_observed_ids = []
        for rl_id in self.tl_ids:
            # --- Sioux: use explicit mapping instead of 'center' split ---
            rl_idx = self._id_to_idx[rl_id]
            local_edges = node_to_edges[rl_idx][1]

            lane_features = []
            observed_ids_agent = []

            # ========= per-lane features (unchanged from your version) =========
            lane_count = 0
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
                    Lnorm = max(edge_len_m, 1e-3)
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


            # ========= TL self + neighbors (modified) =========
            # In the grid env this used self._get_relative_node(...).
            # For Sioux we instead use a precomputed neighbor list.
            neighbors_idxs = [rl_idx] + self._neighbors_by_idx.get(rl_idx, [])
            # pad to length 5 (self + 4 neighbors)
            while len(neighbors_idxs) < 5:
                neighbors_idxs.append(-1)

            tl_states = []
            for idx in neighbors_idxs[:5]:
                if idx == -1:
                    # border / no-neighbor default
                    tl_states.extend([
                        1.0,   # normalized last_change
                        1.0,   # normalized phase
                        1.0,   # yellow flag
                    ])
                else:
                    tl_states.extend([
                        self.last_change[idx] / self.min_switch_time,
                        self.last_phase[idx] / 8.0,
                        float(self.currently_yellow[idx]),
                    ])

            tl_states = np.array(tl_states, dtype=np.float32)

            # === Final concatenated observation ===
            observation = np.concatenate([lane_features, tl_states], dtype=np.float32)
            obs[rl_id] = observation
            all_agents_observed_ids.append(observed_ids_agent)

        # cache for the reward function
        self.observed_ids = all_agents_observed_ids

        return obs


    # ------------------------------------------------------------------
    # 5) Override _classify_links_for_node to match Sioux Falls layout
    # ------------------------------------------------------------------
    def _classify_links_for_node(self, node_id):
        """
        Classify each TLS link at a junction into:
          - approach direction: north/south/east/west
          - movement type: straight/right (SR) vs left (L)

        This uses lane geometry from SUMO instead of the manual
        SIOUX_EDGE_DIR mapping, so straight/right/left are determined
        from the actual shapes of incoming and outgoing lanes.
        """
        import math

        tls = self.k.kernel_api.trafficlight
        lane_api = self.k.kernel_api.lane

        # list-of-lists; links[i] is a list of (inLane, outLane, viaLane)
        links = tls.getControlledLinks(node_id)
        n_links = len(links)

        # Initialize all mask vectors
        north_all = [0] * n_links
        south_all = [0] * n_links
        east_all  = [0] * n_links
        west_all  = [0] * n_links

        ns_sr   = [0] * n_links
        ns_left = [0] * n_links
        ew_sr   = [0] * n_links
        ew_left = [0] * n_links

        def lane_dir(lane_id):
            """Unit direction vector of lane travel (from start to end)."""
            try:
                shape = lane_api.getShape(lane_id)
            except Exception:
                return None
            if not shape or len(shape) < 2:
                return None
            x0, y0 = shape[0]
            x1, y1 = shape[-1]
            dx, dy = x1 - x0, y1 - y0
            n = math.hypot(dx, dy)
            if n < 1e-6:
                return None
            return dx / n, dy / n

        def cardinal_from_vec(vec):
            """Map a direction vector to N/S/E/W based on dominant component."""
            if vec is None:
                return None
            dx, dy = vec
            if abs(dy) >= abs(dx):
                # mostly vertical
                return "S" if dy > 0 else "N"
            else:
                # mostly horizontal
                return "E" if dx > 0 else "W"

        def movement_type(in_vec, out_vec):
            """
            Given incoming and outgoing travel vectors, return:
              - 'SR' for straight or right
              - 'L' for left
              - None if undefined
            """
            if in_vec is None or out_vec is None:
                return None
            dx1, dy1 = in_vec
            dx2, dy2 = out_vec
            dot = dx1 * dx2 + dy1 * dy2
            cross = dx1 * dy2 - dy1 * dx2
            ang = math.atan2(cross, dot)  # signed angle [-pi, pi]
            deg = ang * 180.0 / math.pi

            # Straight if the angle is small
            if abs(deg) < 30.0:
                return "SR"
            # For SUMO's coordinate system, positive angle here corresponds
            # to a left turn (e.g. N -> E gave about +90 deg in tests).
            if deg > 30.0:
                return "L"
            # Negative angles are right turns -> treat with straight
            return "SR"

        for idx, group in enumerate(links):
            if not group:
                continue

            in_lane, out_lane, _ = group[0]

            in_vec = lane_dir(in_lane)
            out_vec = lane_dir(out_lane)

            appr_dir = cardinal_from_vec(in_vec)
            if appr_dir is None:
                continue

            # Determine axis and fill "all approaches" masks
            if appr_dir == "N":
                north_all[idx] = 1
                axis = "NS"
            elif appr_dir == "S":
                south_all[idx] = 1
                axis = "NS"
            elif appr_dir == "E":
                east_all[idx] = 1
                axis = "EW"
            else:  # "W"
                west_all[idx] = 1
                axis = "EW"

            mtype = movement_type(in_vec, out_vec)
            if mtype is None:
                continue

            if axis == "NS":
                if mtype == "SR":
                    ns_sr[idx] = 1
                elif mtype == "L":
                    ns_left[idx] = 1
            else:  # axis == "EW"
                if mtype == "SR":
                    ew_sr[idx] = 1
                elif mtype == "L":
                    ew_left[idx] = 1

        return {
            "north_all": north_all,
            "south_all": south_all,
            "east_all":  east_all,
            "west_all":  west_all,
            "ns_sr":     ns_sr,
            "ns_left":   ns_left,
            "ew_sr":     ew_sr,
            "ew_left":   ew_left,
            "n_links":   n_links,
        }

    
    def _apply_rl_actions(self, rl_actions):
        """
        Sioux Falls version of the traffic-light update logic.

        It is functionally similar to MultiTrafficLightGridPOEnv._apply_rl_actions
        but uses SUMO TLS ids (e.g. "3", "10", ...) directly instead of assuming
        names of the form "center<i>" and matches the masks returned by
        _classify_links_for_node().
        """
        import numpy as np  # safe even if already imported at top

        # Ensure Sioux-specific mappings and arrays are ready
        self._ensure_sioux_initialized()

        # keep track of which lights changed this step (optional, used by some plots)
        self._switched_tls = set()

        # safety: handle None
        if rl_actions is None:
            return

        # simulation time step (defaults to 1s if not present)
        sim_dt = getattr(self, "sim_step", 1.0)

        def make_state(n_links, mask):
            """Return a TL state string ('G'/'r') of length n_links."""
            chars = ["r"] * n_links
            for idx, flag in enumerate(mask):
                if flag:
                    chars[idx] = "G"
            return "".join(chars)

        for rl_id, rl_action in rl_actions.items():
            # Map agent id -> internal index [0, num_tls)
            if rl_id not in self._id_to_idx:
                # Unknown agent id; skip it
                continue
            i = self._id_to_idx[rl_id]

            # In Sioux Falls, rl_id *is* the SUMO TLS id
            node_id = rl_id

            # --- parse and clamp action to [0, 7] ---
            a = rl_action
            if isinstance(a, (list, tuple)):
                a = a[0]
            if isinstance(a, np.ndarray):
                # RLlib often passes actions as 1D arrays
                a = a.item()
            a = int(a)
            if a < 0:
                a = 0
            if a > 7:
                a = 7

            # --- build phase states for this node ---
            grp = self._classify_links_for_node(node_id)
            L = grp["n_links"]
            if L <= 0:
                # no controlled links here (should not happen)
                continue

            NS_SR   = make_state(L, grp["ns_sr"])
            EW_SR   = make_state(L, grp["ew_sr"])
            NS_LEFT = make_state(L, grp["ns_left"])
            EW_LEFT = make_state(L, grp["ew_left"])
            W_ALL   = make_state(L, grp["west_all"])
            S_ALL   = make_state(L, grp["south_all"])
            E_ALL   = make_state(L, grp["east_all"])
            N_ALL   = make_state(L, grp["north_all"])
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

            # Fallbacks for missing movements
            if a == 2 and not any(grp["ns_left"]):
                a = 0
            if a == 3 and not any(grp["ew_left"]):
                a = 1
            if a == 4 and not any(grp["north_all"]):
                a = 0
            if a == 5 and not any(grp["south_all"]):
                a = 0
            if a == 6 and not any(grp["east_all"]):
                a = 1
            if a == 7 and not any(grp["west_all"]):
                a = 1

            cur_phase = int(self.last_phase[i])
            is_yellow = bool(self.currently_yellow[i])

            # Update timers
            if is_yellow:
                self.last_change[i] += sim_dt
            else:
                self.time_in_phase[i] += sim_dt

            # --- yellow-phase handling ---
            if is_yellow:
                if self.last_change[i] >= self.min_switch_time:
                    # Commit to pending phase
                    next_p = int(self.pending_phase[i])
                    state = phase_states.get(next_p, ALL_RED)
                    self.k.traffic_light.set_state(node_id=node_id, state=state)

                    self.last_phase[i] = next_p
                    self.currently_yellow[i] = 0
                    self.last_change[i] = 0.0
                    self.time_in_phase[i] = 0.0
                    self._switched_tls.add(node_id)
                else:
                    # still within yellow; keep yellow for all links
                    self.k.traffic_light.set_state(node_id=node_id, state=ALL_Y)
                # no further processing for this node in this step
                continue

            # --- not currently yellow: decide whether to switch ---
            if a != cur_phase:
                # request a new phase: go through yellow first
                self.pending_phase[i] = a
                self.currently_yellow[i] = 1
                self.last_change[i] = 0.0
                self.k.traffic_light.set_state(node_id=node_id, state=ALL_Y)
                self._switched_tls.add(node_id)
            else:
                # hold current phase
                state = phase_states.get(cur_phase, ALL_RED)
                self.k.traffic_light.set_state(node_id=node_id, state=state)



    # ------------------------------------------------------------------
    # 6) Override compute_reward to use _id_to_idx instead of 'center'
    # ------------------------------------------------------------------
    def compute_reward(self, rl_actions, **kwargs):
        """Per-agent reward based on local speed and queue length."""
        if rl_actions is None:
            return {}

        veh_ids = self.k.vehicle.get_ids()
        if len(veh_ids) == 0:
            # no vehicles → zero reward for all active agents
            return {rl_id: 0.0 for rl_id in rl_actions.keys()}

        # For normalization
        v_top = max(
            self.k.network.speed_limit(e) for e in self.k.network.get_edge_list()
        )
        eps = 1e-6

        rews = {}

        for rl_id in rl_actions.keys():
            # Sioux: use explicit mapping instead of splitting "center"
            rl_idx = self._id_to_idx[rl_id]
            local_edges = self.network.node_mapping[rl_idx][1]

            local_speeds = []
            local_queue = 0
            total_lanes = 0

            # collect speeds and queue counts on edges controlled by this TL
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

            # crude queue proxy: vehicles below 0.2 m/s per lane, normalized
            queue_ratio = local_queue / (total_lanes * 10.0 + eps)

            # final reward: favor high speed, penalize queues
            rew = (2.0 * speed_ratio) - (0.2 * queue_ratio)
            rews[rl_id] = rew

        return rews
    
    def additional_command(self):
        """
        Extend the parent 'additional_command' with OD-based spawning.

        This is called once per simulation step by Flow.
        """
        # Call parent behavior (yellow-phase timing etc.)
        super().additional_command()

        # Then inject OD-based vehicles
        if not self._sioux_structures_built:
            self._build_sioux_structures()

        self._spawn_vehicles_from_od()


