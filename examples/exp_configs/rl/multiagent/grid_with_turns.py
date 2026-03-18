# grid_with_turns.py
from flow.networks import TrafficLightGridNetwork

class TwoLaneTurnsGridNetwork(TrafficLightGridNetwork):
    """
    2-lane approaches at each junction.
      - Lane 0 (rightmost): straight + right
      - Lane 1 (inner):     left-only
    Edge naming in Flow grid:
      bot = eastbound,  top = westbound,  right = southbound,  left = northbound.
    """

    def specify_connections(self, net_params):
        conns = []

        R = net_params.additional_params["grid_array"]["row_num"]
        C = net_params.additional_params["grid_array"]["col_num"]

        def bot(r, c):   return f"bot{r}_{c}"    # eastbound segment starting at (r,c) -> (r,c+1)
        def top(r, c):   return f"top{r}_{c}"    # westbound segment starting at (r,c) -> (r,c-1)
        def right(r, c): return f"right{r}_{c}"  # southbound segment starting at (r,c) -> (r+1,c)
        def left(r, c):  return f"left{r}_{c}"   # northbound segment starting at (r,c) -> (r-1,c)

        for r in range(R):
            for c in range(C):
                # Node (r,c)

                # 1) Incoming from WEST into (r,c): edge is bot(r, c-1)
                if c - 1 >= 0:
                    in_e = bot(r, c - 1)
                    # lane0 straight → east (out of (r,c))
                    conns.append({"from": in_e, "to": bot(r, c), "fromLane": "0", "toLane": "0"})
                    # lane0 right  → south
                    if r + 1 < R:
                        conns.append({"from": in_e, "to": right(r, c), "fromLane": "0", "toLane": "0"})
                    # lane1 left   → north
                    if r - 1 >= 0:
                        conns.append({"from": in_e, "to": left(r, c), "fromLane": "1", "toLane": "1"})

                # 2) Incoming from EAST into (r,c): edge is top(r, c + 1)
                if c + 1 < C:
                    in_e = top(r, c + 1)
                    # lane0 straight → west
                    conns.append({"from": in_e, "to": top(r, c), "fromLane": "0", "toLane": "0"})
                    # lane0 right   → north
                    if r - 1 >= 0:
                        conns.append({"from": in_e, "to": left(r, c), "fromLane": "0", "toLane": "0"})
                    # lane1 left    → south
                    if r + 1 < R:
                        conns.append({"from": in_e, "to": right(r, c), "fromLane": "1", "toLane": "1"})

                # 3) Incoming from SOUTH into (r,c): edge is left(r, c)  (northbound, from (r+1,c)→(r,c))
                if r + 1 < R:
                    in_e = left(r, c)
                    # lane0 straight → north (leaves (r,c) toward (r-1,c)): left(r-1, c)
                    if r - 1 >= 0:
                        conns.append({"from": in_e, "to": left(r - 1, c), "fromLane": "0", "toLane": "0"})
                    # lane0 right   → east
                    conns.append({"from": in_e, "to": bot(r, c), "fromLane": "0", "toLane": "0"})
                    # lane1 left    → west
                    if c - 1 >= 0:
                        conns.append({"from": in_e, "to": top(r, c), "fromLane": "1", "toLane": "1"})

                # 4) Incoming from NORTH into (r,c): edge is right(r - 1, c) (southbound, from (r-1,c)→(r,c))
                if r - 1 >= 0:
                    in_e = right(r - 1, c)
                    # lane0 straight → south (leaves (r,c) toward (r+1,c)): right(r, c)
                    if r + 1 < R:
                        conns.append({"from": in_e, "to": right(r, c), "fromLane": "0", "toLane": "0"})
                    # lane0 right   → west
                    if c - 1 >= 0:
                        conns.append({"from": in_e, "to": top(r, c), "fromLane": "0", "toLane": "0"})
                    # lane1 left    → east
                    conns.append({"from": in_e, "to": bot(r, c), "fromLane": "1", "toLane": "1"})

        return conns
