# tls_presets.py
from flow.core.params import TrafficLightParams

def build_two_lane_hold_tls(n_rows: int,
                                    n_cols: int,
                                    vertical_lanes: int = 4,
                                    horizontal_lanes: int = 4,
                                    program_id: str = "hold") -> TrafficLightParams:
    """
    Build an actuated TLS program whose phase strings match the number of
    controlled connections when there are *straight-only* connections
    and two lanes per approach (configurable via args).

    Assumes Flow's grid node ids are 'center0', 'center1', ..., row-major order.
    """
    # number of connections per intersection
    n_conn = 24 # 12 for 2 lanes, 20 for 4 lanes
    
    # Following needed to make sure different directions go at the same time and crashes happen
    phases = [
        {"duration": "3600000", "state": "r" * n_conn}   # 1000h all-red placeholder
    ]
    tl = TrafficLightParams(baseline=False)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            tl.add(
                node_id=f"center{idx}",
                tls_type="static", # was actuated
                programID=program_id,
                phases=phases
            )
    return tl