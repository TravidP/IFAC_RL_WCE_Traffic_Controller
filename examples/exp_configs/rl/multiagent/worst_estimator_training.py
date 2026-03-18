"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs import WorstEstimatorTrafficEnv             # ✅ 用你自定义的环境
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter, SimLaneChangeController
from flow.controllers.routing_controllers import OuterEdgeODRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from .tls_presets import build_two_lane_hold_tls
from flow.controllers.Routes import Group_routes_G1

# ========= 实验/网络参数 =========
N_ROLLOUTS = 8
N_CPUS = 8

# 关键：一小时一轮
HORIZON = 16*600                      # ✅ 1 小时
WARMUP_SEC = 600                    # ✅ 10 分钟 warmup

V_ENTER = 13.89
INNER_LENGTH = 300
LONG_LENGTH = 100
SHORT_LENGTH = 300
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1
N_ROWS, N_COLUMNS = 3, 3
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
    "g11": (739733.7687455552, 4207605.931283082),}
# ========= 车辆与路由（行进时使用；真正的发车由环境内部控制）=========
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,
        tau=1.0,
        sigma=0.5,
        speed_mode="right_of_way",
    ),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(OuterEdgeODRouter, {
        "seed": 42,
        "use_hardcoded": True,
        "hardcoded_routes": Group_routes_G1,   # 供路由器参考；真正的发车由环境 _spawn_step 生成
        "use_precomputed_cycle": True,
    }),
    num_vehicles=0,
)

# ========= 信号灯逻辑 =========
tl_logic = build_two_lane_hold_tls(
    n_rows=N_ROWS,
    n_cols=N_COLUMNS,
    vertical_lanes=2,
    horizontal_lanes=2
)

# ========= Flow 参数 =========
flow_params = dict(
    exp_tag=f"grid_0_{N_ROWS}x{N_COLUMNS}_worst_estimator_internal_inflow",

    # ✅ 用你写的环境，内部会：每次 step 跑 10 分钟仿真、依据动作混合 inflow
    env_name=WorstEstimatorTrafficEnv,

    network=TrafficLightGridNetwork,
    simulator='traci',
    tls=tl_logic,

    # ✅ 每个 rollout 重启一次 SUMO；但同一 episode 内部不会为 10 分钟窗口重启
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=True,
        print_warnings=False,
    ),

    # ✅ 一小时 + 十分钟 warmup；其余保持
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=1,
        warmup_steps=WARMUP_SEC,               # ✅
        additional_params={
            "target_velocity": 50,
            "switch_time": 5,
            "num_observed": 4,
            "discrete": True,
            "tl_type": "controlled",
            "num_local_edges": 4,
            "num_local_lights": 4,
            "queue_window_m": 50.0,
            "queue_speed_thresh": 0.2,
            "queue_cap_per_lane": 10,
            "queue_penalty_gain": 0.2,

            # ✅ 不从 ROU 发车；发车由 WorstEstimatorTrafficEnv._spawn_step 完成
            "spawn_from_rou": False,
            # 不要传 "rou_file"
            "load_tl_policy": True,
            "policy_dir": "/home/sdc_joran/flow/all_weights_in_one_file.csv"

        },
    ),

    # ✅ 关键：外部 inflow 取消，交给环境内部控制
    net=NetParams(
        inflows=None,                            # ✅ 不用 NetParams.inflows
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
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "turn_speed": 8.9,
            # 你的坐标/归一化等保持原样（如需）
            "node_coordinates": RAW_NODE_COORDS,
            "normalize_to_center0": True,
        },
    ),

    veh=vehicles,

    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)

# ========================= Environment Setup ==============================
create_env, env_name = make_create_env(params=flow_params, version=0)
from ray.tune.registry import register_env
register_env(env_name, create_env)

if __name__ == "__main__":
    env = create_env()
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)
