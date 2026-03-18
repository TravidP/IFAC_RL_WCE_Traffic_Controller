"""Contains a list of custom routing controllers."""
import random
import bisect
import numpy as np
import zlib  
from .Routes import HARDCODED_ROUTES as _HC_ROUTES,  get_od_variants, get_variant_edges_or_none
from flow.controllers.base_routing_controller import BaseRouter


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed ring.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif edge == current_route[-1]:
            # choose one of the available routes based on the fraction of times
            # the given route can be chosen
            num_routes = len(env.available_routes[edge])
            frac = [val[1] for val in env.available_routes[edge]]
            route_id = np.random.choice(
                [i for i in range(num_routes)], size=1, p=frac)[0]

            # pass the chosen route
            return env.available_routes[edge][route_id][0]
        else:
            return None


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity network.

    This class allows the vehicle to pick a random route at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            while veh_next_edge[0][0][0] == not_an_edge:
                veh_next_edge = env.k.network.next_edge(
                    veh_next_edge[random_route][0],
                    veh_next_edge[random_route][1])
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle in a traffic light grid environment.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.k.vehicle.get_edge(self.veh_id) == \
                env.k.vehicle.get_route(self.veh_id)[-1]:
            return [env.k.vehicle.get_edge(self.veh_id)]
        else:
            return None


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route


class I210Router(ContinuousRouter):
    """Assists in choosing routes in select cases for the I-210 sub-network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        # vehicles on these edges in lanes 4 and 5 are not going to be able to
        # make it out in time
        if edge == "119257908#1-AddedOffRampEdge" and lane in [5, 4, 3]:
            new_route = env.available_routes[
                "119257908#1-AddedOffRampEdge"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route

# put this at module level so the dict isn’t rebuilt every call
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
    # col 0
    "e_n2_n1": "left2_0", "e_n1_n2": "right2_0",
    "e_n1_n0": "left1_0", "e_n0_n1": "right1_0",  

    # col 1
    "e_n5_n4": "left2_1", "e_n4_n5": "right2_1",
    "e_n4_n3": "left1_1", "e_n3_n4": "right1_1",

    # col 2
    "e_n8_n7": "left2_2", "e_n7_n8": "right2_2",
    "e_n7_n6": "left1_2", "e_n6_n7": "right1_2",

    # Row lanes
    # row 0 
    "e_n0_n3": "bot0_1", "e_n3_n0": "top0_1",
    "e_n3_n6": "bot0_2", "e_n6_n3": "top0_2",

    # row 1
    "e_n1_n4": "bot1_1", "e_n4_n1": "top1_1",
    "e_n4_n7": "bot1_2", "e_n7_n4": "top1_2",

    # row 2
    "e_n2_n5": "bot2_1", "e_n5_n2": "top2_1",
    "e_n5_n8": "bot2_2", "e_n8_n5": "top2_2",
}


def convert_edge_id(edge_id: str) -> str:
    """
    Convert old edge IDs (like 'e_g1_n2') to new ones (like 'bot2_0').

    Returns the original ID if no mapping exists.
    """
    return EDGE_ID_MAP.get(edge_id, edge_id)

import xml.etree.ElementTree as ET

# load routes file (an idea, might not work)
def load_real_routes(xml_path: str) -> dict:
    """
    Parse a SUMO routes.rou.xml file and return a mapping of origin -> list of {weight, edges}.
    Each vehicle in the XML defines one route (equal weight for now).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    routes_by_origin = {}
    vehicles = root.findall("vehicle")
    n = len(vehicles)
    weight = 1.0 / max(1, n)

    for veh in vehicles:
        route_elem = veh.find("route")
        if route_elem is None:
            continue
        edge_ids = route_elem.attrib["edges"].split()
        converted = [convert_edge_id(e) for e in edge_ids]
        if not converted:
            continue
        origin = converted[0]
        routes_by_origin.setdefault(origin, []).append({
            "weight": weight,
            "edges": converted,
        })

    return routes_by_origin

class OuterEdgeODRouter(BaseRouter):
    """
    OD router for grid networks:
    - Spawn happens on *inflow* outer edges: left{R}_*, right0_*, bot*_0, top*_C
    - Once inside (off inflow), pick a destination among *exit* outer edges:
        left0_*, top*_0, right{R}_*, bot*_{C}
    - Try SUMO shortest_path; if missing, build a Manhattan route.
    - Validate connectivity; only then setRoute (with curr_edge first).
    - NEW: before setting the route, align the current lane to the next planned
      turn so vehicles don’t get stuck in a lane that has no link to that turn.
      ALSO: keep aligning at each new edge while following the route.
      NEWER: while still on the inflow stub (before first junction), pick the
      destination early and align into the lane that connects to the *first*
      out-edge needed at the first junction.

    - NEWEST: Hard-coded OD mode with weighted per-origin routes.
      Define explicit edge sequences for each origin with a "weight".
      Example below for origin 'right0_0' (fill in the other 131 routes here too).
    """


    HARDCODED_ROUTES = _HC_ROUTES
    # ========================================

    def __init__(self, veh_id, router_params=None):
        super().__init__(veh_id, router_params or {})
        self.dest_edge = None
        self._route_set = False

        base = self.router_params.get("seed", None)
        if base is None:
            self._rng = random.Random()  # system randomness
        else:
            # stable per-vehicle mixing (reproducible across runs)
            h = zlib.crc32(str(veh_id).encode("utf-8")) & 0xFFFFFFFF
            mixed = (h ^ int(base)) & 0xFFFFFFFF
            self._rng = random.Random(mixed)

        # Hard-coded routing control (unchanged)
        self._use_hardcoded = bool(self.router_params.get("use_hardcoded", True))
        self._routes_spec = self.router_params.get("hardcoded_routes", None)
        if self._routes_spec is None:
            self._routes_spec = dict(OuterEdgeODRouter.HARDCODED_ROUTES)
        self._hc_compiled = False
        self._hc_by_origin = {}
        self._origin_edge = None
        self._hc_full_route = None
        self._debug_origin = self.router_params.get("debug_origin", "right0_0")

        # added initilization for real data, comment if incorrect
        self._routes_spec = self.router_params.get("hardcoded_routes", None)
        if self._routes_spec is None:
            xml_path = self.router_params.get("routes_xml_path", None)
            if xml_path:
                try:
                    self._routes_spec = load_real_routes(xml_path)
                    # print(f"[OuterEdgeODRouter] Loaded real routes from {xml_path} ({len(self._routes_spec)} origins)")
                except Exception as e:
                    # print(f"[OuterEdgeODRouter] Failed to load {xml_path}: {e}")
                    self._routes_spec = dict(OuterEdgeODRouter.HARDCODED_ROUTES)
            else:
                self._routes_spec = dict(OuterEdgeODRouter.HARDCODED_ROUTES)


    def _log(self, msg):
        try:
            # only log for the chosen origin
            if not self._debug_origin:
                return
            if self._origin_edge != self._debug_origin:
                return
            # print(msg)  # commented: suppress all debug logs; keep only right0_0 route picks below
        except Exception:
            pass
    # -------------------- edge id parsing + node helpers --------------------
    @staticmethod
    def _parse_edge_id(eid):
        # 'bot2_1' -> ('bot', 2, 1)
        if   eid.startswith("bot"):   kind = "bot"
        elif eid.startswith("top"):   kind = "top"
        elif eid.startswith("left"):  kind = "left"
        elif eid.startswith("right"): kind = "right"
        else: return None
        rest = eid[len(kind):]
        r, c = rest.split("_")
        return kind, int(r), int(c)

    @staticmethod
    def _to_node_of_edge(kind, r, c):
        # Node that this edge ENTERS (its end node / junction):
        # incoming to (r,c): bot(r,c), right(r,c), top(r,c+1), left(r+1,c)
        if   kind == "bot":   return (r, c)       # (r,c-1) -> (r,c)
        elif kind == "right": return (r, c)       # (r-1,c) -> (r,c)
        elif kind == "top":   return (r, c-1)     # (r,c)   -> (r,c-1)
        elif kind == "left":  return (r-1, c)     # (r,c)   -> (r-1,c)
        else: return None

    @staticmethod
    def _from_node_of_edge(kind, r, c):
        # Node the edge STARTS from (tail). Must be consistent with _start_node_of_edge.
        if   kind == "bot":   return (r, c-1)   # (r,c-1) -> (r,c)
        elif kind == "right": return (r-1, c)   # (r-1,c) -> (r,c)
        elif kind == "top":   return (r, c)     # (r,c)   -> (r,c-1)
        elif kind == "left":  return (r, c)     # (r,c)   -> (r-1,c)
        else: return None

    # ---------- precise edge-id geometry (kept in sync with the above) ----------
    @staticmethod
    def _start_node_of_edge(kind, r, c):
        # Start node (tail). Must match _from_node_of_edge exactly.
        if   kind == "bot":   return (r, c-1)
        elif kind == "right": return (r-1, c)
        elif kind == "top":   return (r, c)     # FIXED: was (r, c+1)
        elif kind == "left":  return (r, c)     # FIXED: was (r+1, c)
        else:                 return None

    @staticmethod
    def _end_node_of_edge(kind, r, c):
        # End node (head), i.e., junction the edge enters. Must match _to_node_of_edge.
        if   kind == "bot":   return (r, c)
        elif kind == "right": return (r, c)
        elif kind == "top":   return (r, c-1)
        elif kind == "left":  return (r-1, c)
        else:                 return None

    @staticmethod
    def _connected(e_from, e_to):
        pf = OuterEdgeODRouter._parse_edge_id(e_from)
        pt = OuterEdgeODRouter._parse_edge_id(e_to)
        if not pf or not pt:
            return False
        return OuterEdgeODRouter._end_node_of_edge(*pf) == \
               OuterEdgeODRouter._start_node_of_edge(*pt)
    
    # Parse OD tag from flow/vehicle id:
    # "OD:<origin>-><dest>#k<idx>_<flowCounter>.<vehCounter>" or "OD:<origin>-><dest>"
    def _parse_od_tag(self, vid):
        try:
            core = vid.split('.')[0]
            base = core.rsplit('_', 1)[0] if '_' in core else core
            if not base.startswith("OD:"):
                return None
            body = base[3:]
            k = None
            if "#k" in body:
                body, ktxt = body.split("#k", 1)
                k = int(ktxt)
            origin, dest = body.split("->")
            return {"origin": origin, "dest": dest, "k": k}
        except Exception:
            return None

    def _pick_route_for_od_variant(self, origin, dest, k):
        """Pick exact variant k edges for (origin,dest), or weighted fallback, or None."""
        spec = self.router_params.get("hardcoded_routes", {})
        od_list = spec.get(origin, {}).get(dest, [])
        if not od_list:
            return None
        # normalize weights
        tmp = []
        tot = 0.0
        for r in od_list:
            w = float(r.get("weight", 1.0))
            tmp.append({"edges": list(r["edges"]), "weight": max(0.0, w)})
            tot += max(0.0, w)
        if k is not None and 0 <= k < len(tmp):
            return tmp[k]["edges"]
        # weighted fallback if k is None/out of range
        tot = tot or 1.0
        u = self._rng.random() * tot
        s = 0.0
        for r in tmp:
            s += r["weight"]
            if u <= s:
                return r["edges"]
        return tmp[-1]["edges"]



    # -------------------- outer-edge list + destination choice --------------
    def _outer_edges(self, env):
        R = env.net_params.additional_params["grid_array"]["row_num"]
        C = env.net_params.additional_params["grid_array"]["col_num"]
        outs = []
        outs += [f"left0_{j}"     for j in range(C)]   # SOUTH boundary exits
        outs += [f"top{i}_0"      for i in range(R)]   # WEST  boundary exits
        outs += [f"right{R}_{j}"  for j in range(C)]   # NORTH boundary exits
        outs += [f"bot{i}_{C}"    for i in range(R)]   # EAST  boundary exits
        return outs

    # (kept for fallback paths when no hardcoded is available)
    def _choose_dest(self, env, origin):
        choices = [e for e in self._outer_edges(env) if e != origin]
        return self._rng.choice(choices)

    # ---------- detect inflow outer stubs (do NOT route yet here) ----------
    def _is_inflow_outer_edge(self, env, eid):
        p = self._parse_edge_id(eid)
        if not p:
            return False
        kind, r, c = p
        R = env.net_params.additional_params["grid_array"]["row_num"]
        C = env.net_params.additional_params["grid_array"]["col_num"]
        return ((kind == "left"  and r == R) or   # left{R}_*
                (kind == "right" and r == 0) or   # right0_*
                (kind == "bot"   and c == 0) or   # bot*_0
                (kind == "top"   and c == C))     # top*_C

    # -------------------- choose correct out-edge for a node step -----------
    @staticmethod
    def _edge_for_turn(prev_node, curr_node, next_node):
        r0, c0 = prev_node
        r,  c  = curr_node
        r1, c1 = next_node
        ar, ac = (r - r0, c - c0)       # approach vector (prev->curr)
        mr, mc = (r1 - r,  c1 - c)      # move vector (curr->next)

        bot   = lambda rr, cc: f"bot{rr}_{cc}"
        top   = lambda rr, cc: f"top{rr}_{cc}"
        left  = lambda rr, cc: f"left{rr}_{cc}"
        right = lambda rr, cc: f"right{rr}_{cc}"

        # came from WEST (prev=(r,c-1) -> curr=(r,c))  (approach eastbound)
        if ar == 0 and ac == 1:
            if   (mr, mc) == (0,  1):  # straight east
                return bot(r, c+1)
            elif (mr, mc) == (-1, 0):  # turn north
                return right(r+1, c)
            elif (mr, mc) == (1,  0):  # turn south
                return left(r, c)

        # came from EAST (prev=(r,c+1) -> curr=(r,c))  (approach westbound)
        if ar == 0 and ac == -1:
            if   (mr, mc) == (0, -1):  # straight west
                return top(r, c)
            elif (mr, mc) == (-1, 0):  # turn north
                return right(r+1, c)
            elif (mr, mc) == (1,  0):  # turn south
                return left(r, c)

        # came from SOUTH (prev=(r-1,c) -> curr=(r,c)) (approach southbound)
        if ar == 1 and ac == 0:
            if   (mr, mc) == (1,  0):  # straight south
                return right(r+1, c)
            elif (mr, mc) == (0,  1):  # turn east
                return bot(r, c+1)
            elif (mr, mc) == (0, -1):  # turn west
                return top(r, c)

        # came from NORTH (prev=(r+1,c) -> curr=(r,c)) (approach northbound)
        if ar == -1 and ac == 0:
            if   (mr, mc) == (-1, 0):  # straight north
                return left(r, c)
            elif (mr, mc) == (0, -1):  # turn west
                return top(r, c)
            elif (mr, mc) == (0,  1):  # turn east
                return bot(r, c+1)

        return None

    # -------------------- route builders (kept for fallback) ----------------
    def _shortest_path(self, env, src_edge, dst_edge):
        try:
            p = env.k.network.shortest_path(src_edge, dst_edge)
            if p and len(p) >= 1:
                return p
        except Exception:
            pass
        return None

    def _manhattan_route_edges(self, env, curr_edge, dest_edge):
        """
        Build a node path from the node after curr_edge to the node where
        dest_edge begins; then translate each node-step to an out-edge with
        _edge_for_turn(prev, curr, next). Finally append dest_edge itself.
        """
        # Nodes tied to edges
        k0, r0, c0 = self._parse_edge_id(curr_edge)
        curr_node  = self._to_node_of_edge(k0, r0, c0)       # junction we enter
        prev_node  = self._from_node_of_edge(k0, r0, c0)     # node we came from

        kd, rd, cd = self._parse_edge_id(dest_edge)
        dest_start = self._from_node_of_edge(kd, rd, cd)     # junction where dest edge starts

        # Manhattan path in node-space (H first, then V)
        path_nodes = [curr_node]
        r, c = curr_node
        rt, ct = dest_start
        while c < ct:  path_nodes.append((r, c+1)); c += 1
        while c > ct:  path_nodes.append((r, c-1)); c -= 1
        while r < rt:  path_nodes.append((r+1, c)); r += 1
        while r > rt:  path_nodes.append((r-1, c)); r -= 1

        # Translate node-steps into edges
        route_edges = []
        for i in range(1, len(path_nodes)):
            nxt = path_nodes[i]
            e = self._edge_for_turn(prev_node, path_nodes[i-1], nxt)
            if e is None:
                route_edges = []
                break
            route_edges.append(e)
            prev_node = path_nodes[i-1]

        # then traverse the destination outer edge fully
        route_edges.append(dest_edge)
        return [e for e in route_edges if e]

    # -------------------- lane alignment to next turn -----------------------
    def _edge_from_lane_id(self, lane_id):
        # SUMO lane IDs look like "<edge_id>_<laneIndex>", e.g., "right0_0_1".
        # We need just the edge_id portion (safe even with underscores in edge_id).
        return lane_id.rsplit("_", 1)[0]

    def _align_lane_to_next_turn(self, env, vid, curr_edge, next_edge, duration=60, force=False):
        """
        Pick a lane on curr_edge that actually reaches `next_edge`, allowing for
        chains of internal lanes (':center...') between the approach and the
        outgoing edge. Request a lane change for `duration` seconds.
        If `force=True` (we use this on *inflow* edges), also call moveTo(...)
        to guarantee placement even when there is no gap to merge.
        """
        try:
            if curr_edge.startswith(":"):
                return

            ka = env.k.kernel_api
            num_lanes = ka.edge.getLaneNumber(curr_edge)
            if num_lanes <= 1:
                return

            cur_lane_idx = ka.vehicle.getLaneIndex(vid)

            def reaches_target_edge(start_lane_id, target_edge, max_depth=6):
                """BFS over lane links until we hit a non-internal edge."""
                seen = set([start_lane_id])
                frontier = [start_lane_id]
                depth = 0
                while frontier and depth < max_depth:
                    nxt_frontier = []
                    for lid in frontier:
                        e = self._edge_from_lane_id(lid)
                        if not e.startswith(":"):
                            if e == target_edge:
                                return True
                            continue  # do not expand beyond a non-internal lane
                        try:
                            links = ka.lane.getLinks(lid)
                        except Exception:
                            links = []
                        for link in links:
                            nlid = link[0]  # next lane id in SUMO
                            if nlid not in seen:
                                seen.add(nlid)
                                nxt_frontier.append(nlid)
                    frontier = nxt_frontier
                    depth += 1
                return False

            def lane_reaches(li):
                """Return True iff approach lane `li` on curr_edge can reach next_edge."""
                approach_lane = f"{curr_edge}_{li}"
                try:
                    first_links = ka.lane.getLinks(approach_lane)
                except Exception:
                    first_links = []
                if not first_links:
                    return reaches_target_edge(approach_lane, next_edge)
                for link in first_links:
                    if reaches_target_edge(link[0], next_edge):
                        return True
                return False

            # --- choose desired lane: prefer least crowded among all eligible (ties -> random) ---
            eligible = [li for li in range(num_lanes) if lane_reaches(li)]
            if not eligible:
                return

            # If already on an eligible lane, stay to avoid churn
            if 0 <= cur_lane_idx < num_lanes and cur_lane_idx in eligible:
                desired = cur_lane_idx
            else:
                lane_loads = []
                for li in eligible:
                    try:
                        n = ka.lane.getLastStepVehicleNumber(f"{curr_edge}_{li}")
                    except Exception:
                        n = float("inf")  # if query fails, make it least attractive
                    lane_loads.append((n, li))
                min_n = min(n for n, _ in lane_loads)
                candidates = [li for n, li in lane_loads if n == min_n]
                desired = random.choice(candidates) if len(candidates) > 1 else candidates[0]

            if desired == cur_lane_idx:
                return

            ka.vehicle.changeLane(vid, desired, duration)

            if force:
                try:
                    pos = ka.vehicle.getLanePosition(vid)
                    try:
                        L = ka.lane.getLength(f"{curr_edge}_{desired}")
                        # keep a small safety buffer from the end of the lane
                        safe_end = max(0.0, L - 2.0)
                    except Exception:
                        safe_end = None
                    # only move if we're not too close to the stop line
                    if safe_end is None or pos < safe_end:
                        ka.vehicle.moveTo(vid, curr_edge, pos, desired)
                except Exception:
                    pass


            # Optional: detailed align debug (suppressed)
            # try:
            #     self._log(f"[AlignLane] {vid} on {curr_edge} lane {cur_lane_idx} -> {desired} to reach {next_edge}")
            # except Exception:
            #     pass

        except Exception:
            pass

# ---------- compile & pick hardcoded routes -----------------------------
    def _compile_hardcoded(self):
        if self._hc_compiled:
            return
        self._hc_by_origin.clear()

        def _validate_route(route_edges):
            if not route_edges or len(route_edges) < 2:
                return False, "too short (<2 edges)"
            prev = route_edges[0]
            for e in route_edges[1:]:
                if not self._connected(prev, e):
                    return False, f"disconnected hop: {prev} -> {e}"
                prev = e
            return True, None

        # Helper: compile a list of spec entries (dicts or bare edge lists)
        def _compile_spec_list(origin, spec_list):
            routes, cumw, total = [], [], 0.0
            for spec in spec_list:
                if isinstance(spec, dict):
                    w = float(spec.get("weight", 1.0))
                    edges = list(spec.get("edges", []))
                else:
                    w = 1.0
                    edges = list(spec)
                if w <= 0.0 or not edges:
                    continue
                if edges[0] != origin:
                    edges = [origin] + edges
                ok, reason = _validate_route(edges)
                if not ok:
                    continue
                routes.append(edges)
                total += w
                cumw.append(total)
            return routes, cumw, total

        for origin, specs in (self._routes_spec or {}).items():
            origin_bucket = {"by_dest": {}}
            total_dropped = 0

            if isinstance(specs, dict):
                # --- nested format {dest: [specs]} ---
                agg_routes, agg_cumw, agg_total = [], [], 0.0
                for dest, spec_list in specs.items():
                    routes, cumw, total = _compile_spec_list(origin, spec_list)
                    if routes:
                        origin_bucket["by_dest"][dest] = {
                            "routes": routes,
                            "cumw": cumw,
                            "total": total,
                        }
                        agg_routes.extend(routes)
                        agg_cumw.extend([agg_total + x for x in cumw])
                        agg_total += total
                if agg_routes and agg_total > 0.0:
                    origin_bucket["routes"] = agg_routes
                    origin_bucket["cumw"] = agg_cumw

            else:
                # --- flat format [specs] ---
                routes, cumw, total = _compile_spec_list(origin, specs)
                if routes and total > 0.0:
                    origin_bucket["routes"] = routes
                    origin_bucket["cumw"] = cumw

            if origin_bucket.get("routes") or origin_bucket["by_dest"]:
                self._hc_by_origin[origin] = origin_bucket

        self._hc_compiled = True



    def _pick_hardcoded_route(self, origin, dest=None, rng=None):
        """
        Returns a list of edges or None. Emits debug prints showing why sampling
        might fail (missing bucket, empty routes, zero cumw, bad index).
        """
        if not self._hc_compiled:
            self._compile_hardcoded()
        bucket = self._hc_by_origin.get(origin)
        if not bucket:
            return None

        # Prefer a by-dest sub-bucket if dest is provided and present
        label = f"{origin}"
        routes, cumw = None, None
        if dest and bucket.get("by_dest") and dest in bucket["by_dest"]:
            sub = bucket["by_dest"][dest]
            routes, cumw = sub.get("routes"), sub.get("cumw")
            label = f"{origin}->{dest}"
        else:
            routes, cumw = bucket.get("routes"), bucket.get("cumw")
            label = f"{origin} (agg)"

        if not routes or not cumw:
            return None
        if len(routes) != len(cumw):
            return None

        total = cumw[-1]
        if total <= 0.0:
            return None

        u = (rng.random() if rng else random.random()) * total
        idx = bisect.bisect_left(cumw, u)
        if idx >= len(routes):
            # Shouldn’t happen, but log loudly and clamp
            idx = len(routes) - 1

        chosen = routes[idx]
        return chosen


    def _route_from_current(self, full_route, curr_edge):
        """Trim a full route so it starts at curr_edge, preserving order."""
        if not full_route:
            return None
        try:
            idx = full_route.index(curr_edge)
        except ValueError:
            return None
        return full_route[idx:]

    # ---------- NEW: compute the very first out-edge needed at first junction ----------
    def _first_step_out_edge_toward_dest(self, inflow_edge, dest_edge):
        """
        From the *inflow* edge and a chosen dest_edge, decide which out-edge
        must be taken at the upcoming junction (Manhattan logic: horizontal
        first, then vertical).
        """
        p0 = self._parse_edge_id(inflow_edge)
        pd = self._parse_edge_id(dest_edge)
        if not p0 or not pd:
            return None

        k0, r0, c0 = p0
        kd, rd, cd = pd

        curr_node = self._to_node_of_edge(k0, r0, c0)
        prev_node = self._from_node_of_edge(k0, r0, c0)
        dest_start = self._from_node_of_edge(kd, rd, cd)

        if not curr_node or not prev_node or not dest_start:
            return None

        r, c = curr_node
        rt, ct = dest_start

        # Degenerate case: destination edge starts at the very first junction.
        if (r, c) == (rt, ct):
            return dest_edge

        if c < ct:      next_node = (r, c + 1)
        elif c > ct:    next_node = (r, c - 1)
        elif r < rt:    next_node = (r + 1, c)
        elif r > rt:    next_node = (r - 1, c)
        else:
            next_node = curr_node  # safe no-op (handled above)

        return self._edge_for_turn(prev_node, curr_node, next_node)

    # ---------- NEW: prefer true route to decide first out-edge -------------
    def _first_step_from_route(self, env, curr_edge, dest_edge):
        """
        Use the real planned route to choose the very first out-edge:
        try SUMO shortest_path (preferred), then Manhattan as fallback.
        (Kept for fallback behavior; not used in hardcoded mode.)
        """
        p = self._shortest_path(env, curr_edge, dest_edge)
        if p and len(p) >= 2:
            return p[1]
        p = self._manhattan_route_edges(env, curr_edge, dest_edge)
        if p and len(p) >= 2:
            return p[1]
        return None

    # ---------- choose_route: hardcoded mode + original behavior fallback ----------
    def choose_route(self, env):
        vid = self.veh_id
        v = env.k.vehicle
        ka = env.k.kernel_api

        # If we already set a route earlier, keep aligning lanes to the *next* edge
        if self._route_set:
            try:
                cur_edge = v.get_edge(vid)
                if not cur_edge or cur_edge.startswith(":"):
                    return None
                route = ka.vehicle.getRoute(vid)
                idx   = ka.vehicle.getRouteIndex(vid)
                if route and 0 <= idx < len(route) - 1:
                    nxt_edge = route[idx + 1]
                    if self._connected(cur_edge, nxt_edge):
                        self._align_lane_to_next_turn(env, vid, cur_edge, nxt_edge, duration=30, force=False)
            except Exception:
                pass
            return None

        # First-time routing below
        curr_edge = v.get_edge(vid)
        if not curr_edge or curr_edge.startswith(":"):
            return None

        # -------- HARD-CODED MODE --------
        if self._use_hardcoded and self._is_inflow_outer_edge(env, curr_edge):
            if self._origin_edge is None:
                self._origin_edge = curr_edge
            # if self._hc_full_route is None:
            #     chosen = self._pick_hardcoded_route(self._origin_edge)
            #     if chosen:
            #         self._hc_full_route = chosen
            #         self.dest_edge = chosen[-1]

            if self._hc_full_route is None:
                od = self._parse_od_tag(vid)
                if od and od["origin"] == curr_edge:
                    chosen = self._pick_route_for_od_variant(od["origin"], od["dest"], od["k"])
                    if chosen:
                        self._hc_full_route = chosen
                        self.dest_edge = chosen[-1]
                    else:
                        # No explicit edges: at least lock the destination so shortest-path aims correctly
                        self.dest_edge = od["dest"]
                else:
                    # fallback: old origin-level pick if no OD tag present
                    chosen = self._pick_hardcoded_route(self._origin_edge)
                    if chosen:
                        self._hc_full_route = chosen
                        self.dest_edge = chosen[-1]
                    # if self._origin_edge == "right0_1":
                    #     print(f"[HardcodedRoutes] PICK vid={vid} origin={self._origin_edge} route={self._hc_full_route}")
            # NEW: always set the route immediately (even on long inflow stubs)
            if self._hc_full_route:
                route = self._route_from_current(self._hc_full_route, curr_edge)
                if route and route[0] != curr_edge:
                    route = [curr_edge] + route
                if route:
                    # connectivity guard
                    prev = route[0]
                    for e in route[1:]:
                        if not self._connected(prev, e):
                            # print(f"[Router ERROR] Non-connected edges in chosen route: {prev} -> {e}")  # commented: suppress errors
                            return None
                        prev = e
                    ka.vehicle.setRoute(vid, route)
                    # uncomment when checking routes
                    # try:
                    #     print(f"[ROUTE] {vid} dest={route[-1]} len={len(route)} : {' -> '.join(route)}")
                    # except Exception:
                    #     pass
                    self._route_set = True
                    if len(route) >= 2:
                        # align but do not force-teleport onto the stop line
                        self._align_lane_to_next_turn(env, vid, route[0], route[1], duration=60, force=False)
                return None

        # -------- OFF INFLOW (or no hardcoded available) --------
        if self._use_hardcoded and self._hc_full_route:
            # We already have a chosen full route; trim from current edge and set.
            route = self._route_from_current(self._hc_full_route, curr_edge)
            if not route:
                # fallback if current edge not in planned route
                route = None
            if route:
                if route[0] != curr_edge:
                    route = [curr_edge] + route
                # connectivity guard
                prev = route[0]
                for e in route[1:]:
                    if not self._connected(prev, e):
                        # print(f"[Router ERROR] Non-connected edges (off-inflow) {prev} -> {e}")  # commented
                        return None
                    prev = e
                if len(route) >= 2:
                    self._align_lane_to_next_turn(env, vid, route[0], route[1], duration=30, force=False)
                ka.vehicle.setRoute(vid, route)
                # uncomment when checking routes
                # try:
                #     print(f"[ROUTE] {vid} dest={route[-1]} len={len(route)} : {' -> '.join(route)}")
                # except Exception:
                #     pass
                self._route_set = True
                return None
            # fall through to original behavior if something went wrong

        # -------- ORIGINAL BEHAVIOR (fallback) --------
        if self.dest_edge is None:
            self.dest_edge = self._choose_dest(env, curr_edge)

        # 0) Still on the inflow outer stub (original plan)
        if self._is_inflow_outer_edge(env, curr_edge):
            edge_len = ka.lane.getLength(f"{curr_edge}_0")
            if edge_len < 10.0:  # short stub: set route now
                route = self._shortest_path(env, curr_edge, self.dest_edge)
                if not route:
                    route = self._manhattan_route_edges(env, curr_edge, self.dest_edge)
                if route and route[0] != curr_edge:
                    route = [curr_edge] + route
                if route:
                    ka.vehicle.setRoute(vid, route)
                    # uncomment when checking routes
                    # try:
                    #     print(f"[ROUTE] {vid} dest={route[-1]} len={len(route)} : {' -> '.join(route)}")
                    # except Exception:
                    #     pass
                    self._route_set = True
                    if len(route) >= 2:
                        self._align_lane_to_next_turn(env, vid, route[0], route[1], duration=60, force=True)
                    try:
                        cur_lane_idx = ka.vehicle.getLaneIndex(vid)
                    except Exception:
                        cur_lane_idx = "?"
                    next_edge = route[1] if len(route) >= 2 else "?"
                    # self._log(f"[Router Debug] {vid} on {curr_edge} lane {cur_lane_idx}, aligning to {next_edge} (short-stub)")  # commented
                    return None
            else:
                first_out = self._first_step_from_route(env, curr_edge, self.dest_edge)
                if first_out:
                    self._align_lane_to_next_turn(env, vid, curr_edge, first_out, duration=60, force=True)
                try:
                    cur_lane_idx = ka.vehicle.getLaneIndex(vid)
                except Exception:
                    cur_lane_idx = "?"
                next_edge = first_out if first_out else "?"
                # self._log(f"[Router Debug] {vid} on {curr_edge} lane {cur_lane_idx}, aligning to {next_edge}")  # commented
            return None

        # 1) we're now off the inflow stub -> compute and set the route
        route = self._shortest_path(env, curr_edge, self.dest_edge)

        # 1a) retry a few alternative exits if SP fails
        tries = 0
        while (not route) and tries < 8:
            self.dest_edge = self._choose_dest(env, curr_edge)
            route = self._shortest_path(env, curr_edge, self.dest_edge)
            tries += 1

        # 1b) fallback to Manhattan if still no luck
        if not route:
            route = self._manhattan_route_edges(env, curr_edge, self.dest_edge)
        if not route:
            return None

        # 2) SUMO requires current edge first
        if route[0] != curr_edge:
            route = [curr_edge] + route

        # 3) connectivity guard
        prev = route[0]
        for e in route[1:]:
            if not self._connected(prev, e):
                return None
            prev = e

        # 3b) align the lane on the current edge to the next planned edge
        if len(route) >= 2:
            self._align_lane_to_next_turn(env, vid, route[0], route[1], duration=30, force=False)

        # 4) safe to set
        ka.vehicle.setRoute(vid, route)
        # uncomment when checking routes
        # try:
        #     print(f"[ROUTE] {vid} dest={route[-1]} len={len(route)} : {' -> '.join(route)}")
        # except Exception:
        #     pass
        self._route_set = True
        return None

