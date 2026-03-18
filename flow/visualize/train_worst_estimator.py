# train_worst_estimator.py
import os, sys, gym, numpy as np
from typing import Dict, List, Tuple

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
# 旧版本 ray 不需要这个类型定义，直接忽略
EnvType = object
import os
os.environ["SUMO_HOME"] = "/usr/share/sumo"  # 如果没有，可手动指定 sumo 安装路径
os.environ["LIBSUMO_AS_TRACI"] = "1"
os.environ["SUMO_BINARY"] = "sumo"  # 🚫 不用 GUI，防止卡死

from ray.tune.registry import register_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# ===== 引入你已有的Flow构建 =====
# 直接复用 multiagent_traffic_light_grid_Worst.py 里对 Flow 的构建与常量
from examples.exp_configs.rl.multiagent.multiagent_traffic_light_grid_Worst import (
    HORIZON,  # 900s = 15min
    make_create_env, register_env as _ignored,  # 已经注册过一次，这里只用 make_create_env
    flow_params,  # 我们会基于此克隆并改 inflows
    Group_routes_G1, read_triplist_csv_to_od_rates, _normalize_weights,
    ROW_ORDER, COL_ORDER
)
# ^ 来自你的文件，含有 OD 读入、变体拆分逻辑、HORIZON=900 等  :contentReference[oaicite:4]{index=4}

# ====== 固定信号策略载入（来自你的 visualizer_rllib 思路） ======
try:
    # 新版/旧版兼容导入
    from ray.rllib.agents.registry import get_agent_class
except ImportError:
    # 极旧版本 fallback
    from ray.rllib.agents import get_agent_class

from flow.utils.registry import make_create_env as _make_create_env
from flow.utils.rllib import get_flow_params as _get_flow_params
from flow.utils.rllib import get_rllib_config as _get_rllib_config
# 参考 visualizer_rllib.py 的加载 & 推理流程  :contentReference[oaicite:5]{index=5}

import traci  # SUMO TraCI 接口（Flow 内部也用）

# ============ 你可在这里列出可混合的 OD 分组（示例：仅 1 组可以先跑通） ============
GROUP_LIB = {
    # 例：可以把多个 Group_routes_G* 填进来
    "G1": Group_routes_G1,
    # "G2": Group_routes_G2,
    # "G3": Group_routes_G3,
}
GROUP_KEYS = list(GROUP_LIB.keys())
G = len(GROUP_KEYS)

# ============ 固定红绿灯策略封装 ============
class FixedTLPolicy:
    """加载你训练好的 RLlib 多智能体红绿灯策略，用于仿真时出动作（不再训练）。"""
    def __init__(self, result_dir: str, checkpoint_num: str):
        # === 兼容旧版 RLlib: 修复 multiagent.policies ===
        config = _get_rllib_config(result_dir)

        ma_conf = config.get("multiagent", {})
        policies = ma_conf.get("policies", {})

        # 某些 Flow 旧实验会保存成 {"tl0": {}}，不是 tuple
        # 这里动态修复成 (PolicyClass, obs_space, act_space, config_dict)
        for k, v in list(policies.items()):
            if not isinstance(v, tuple):
                try:
                    # 导入基础 Policy 类占位
                    from ray.rllib.policy.policy import Policy
                except Exception:
                    class Policy(object):
                        pass
                policies[k] = (Policy, None, None, {})

        ma_conf["policies"] = policies
        config["multiagent"] = ma_conf
        # ===================================================

        run = config.get("env_config", {}).get("run", None)
        if run is None:
            raise RuntimeError("params.json 中缺少 'run' 字段，请像 visualizer_rllib.py 一样传入/保存。")
        agent_cls = get_agent_class(run)
        self.agent = agent_cls(env=None, config=config)
        ckpt = os.path.join(result_dir, f"checkpoint_{checkpoint_num}", f"checkpoint-{checkpoint_num}")
        self.agent.restore(ckpt)

        # 多智能体策略映射
        self.policy_map = config["multiagent"]["policy_mapping_fn"]
        self.policies = list(config["multiagent"]["policies"].keys())
        self.use_lstm = bool(config.get("model", {}).get("use_lstm", False))
        if self.use_lstm:
            size = config["model"]["lstm_cell_size"]
            self.state_init = {pid: [np.zeros(size, np.float32), np.zeros(size, np.float32)]
                               for pid in self.policies}

    def compute_actions(self, obs_multiagent: Dict[str, np.ndarray]) -> Dict[str, int]:
        """逐路口出离散相位动作"""
        actions = {}
        for agent_id, ob in obs_multiagent.items():
            policy_id = self.policy_map(agent_id)
            if self.use_lstm:
                act, self.state_init[policy_id], _ = self.agent.compute_action(ob, state=self.state_init[policy_id],
                                                                               policy_id=policy_id)
            else:
                act = self.agent.compute_action(ob, policy_id=policy_id)
            actions[agent_id] = act
        return actions

# ============ 工具：按混合系数合成 inflows ============
def build_inflows_from_mixture(base_od_csv: str,
                               group_weights: Dict[str, float],
                               horizon: int):
    """
    读取基础 OD 率（或按你的 CSV 术语：triplist -> od_rates），
    再将 group 权重作用到各 OD->route variants 上，构造 Flow 的 InFlows。
    """
    from flow.core.params import InFlows, VehicleParams, SumoCarFollowingParams
    from flow.controllers import SimCarFollowingController, SimLaneChangeController
    from flow.controllers.Routes import Group_routes_G1  # 仅示意，下面用 GROUP_LIB

    od_rates = read_triplist_csv_to_od_rates(base_od_csv)  # 车辆/小时矩阵
    inflow = InFlows()

    # 每个OD的“变体”来源于各 group 的 routes 定义，这里按 group 权重线性叠加
    # 简化策略：对每个 OD，把不同 group 的变体列表合并，权重乘上 group weight 并归一化
    for origin in ROW_ORDER:
        for dest in COL_ORDER:
            # 汇总所有group的变体
            variants_all = []
            for gk, g_weight in group_weights.items():
                variants = GROUP_LIB[gk].get(origin, {}).get(dest, [])
                for v in variants:
                    variants_all.append({"route": v["route"], "weight": g_weight * float(v.get("weight", 1.0))})

            # 若没有显式变体，仍按 od_rates 出车（路由走最短路/锁定目的地）
            rate = od_rates[origin][dest]  # veh/h
            if rate <= 0:
                continue

            if not variants_all:
                period = 3600.0 / rate
                inflow.add(
                    veh_type="human",
                    edge=origin,
                    period=period,
                    name=f"OD_{origin}_to_{dest}",
                    depart_lane="best",
                    depart_speed=13.89,  # 与文件一致 V_ENTER=13.89 m/s  :contentReference[oaicite:6]{index=6}
                    departPos="free",
                    begin=0, end=horizon
                )
            else:
                # 合并后的变体先按权重归一化
                total_w = sum(max(1e-8, v["weight"]) for v in variants_all)
                normed = [{"route": v["route"], "weight": v["weight"] / total_w} for v in variants_all]

                for k, v in enumerate(normed):
                    var_rate = rate * v["weight"]
                    if var_rate <= 0:
                        continue
                    period = 3600.0 / var_rate
                    inflow.add(
                        veh_type="human",
                        edge=origin,
                        period=period,
                        name=f"OD_{origin}_to_{dest}_mix{k}",
                        depart_lane="best",
                        depart_speed=13.89,
                        departPos="free",
                        begin=0, end=horizon
                    )
    return inflow

# ============ 观测统计：每路口每车道 平均速度 + 密度 ============
def collect_region_stats(env) -> np.ndarray:
    """
    返回 shape = (num_intersections * lanes_per_intersection * 2,) 的向量：
    对每条 lane: [avg_speed, density]。
    密度用 车数/车道长度 的粗略近似（单位 veh/m），足够作为学习信号。
    """
    k = env.unwrapped.k
    lanes = k.network.get_edge_list()  # 也可按你的“局部边集”收集
    feats = []
    for edge in lanes:
        try:
            lane_ids = k.kernel_api.edge.getLanes(edge)  # TraCI
        except Exception:
            lane_ids = []
        for lid in lane_ids:
            vehs = k.kernel_api.lane.getLastStepVehicleIDs(lid)
            speeds = [k.kernel_api.vehicle.getSpeed(vid) for vid in vehs] if vehs else []
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            length = k.kernel_api.lane.getLength(lid)
            density = (len(vehs) / max(1.0, length))  # veh/m
            feats.extend([avg_speed, density])
    return np.array(feats, dtype=np.float32)

# ============ CWT & AvgSpeed 统计（单个 15min 段内） ============
class SegmentStats:
    def __init__(self):
        self.total_wait = 0.0   # seconds
        self.speed_sum = 0.0
        self.speed_cnt = 0

    def update(self, env):
        k = env.unwrapped.k
        vids = k.vehicle.get_ids()
        # waitingTime：SUMO 定义（速度极低/停止时累计）
        wt = sum(k.kernel_api.vehicle.getWaitingTime(v) for v in vids) if vids else 0.0
        self.total_wait += wt
        spds = k.vehicle.get_speed(vids)
        if spds:
            self.speed_sum += float(np.sum(spds))
            self.speed_cnt += len(spds)

    @property
    def avg_speed(self):
        return (self.speed_sum / max(1, self.speed_cnt))

# === 连续仿真版本：不重启 SUMO，只在窗口内动态投放车流 ===
import gym, numpy as np, time, itertools
from typing import Dict, List

# 假设沿用你已有：HORIZON=900, GROUP_LIB, ROW_ORDER, COL_ORDER, FixedTLPolicy, collect_region_stats, SegmentStats 等
# 以及 read_triplist_csv_to_od_rates(...) 可读基础 OD 率（veh/hour）

# ====== 连续仿真版 OnlineSpawner（按秒泊松投放）======
import itertools
class OnlineSpawner:
    def __init__(self, kernel_api, group_lib, row_order, col_order):
        self.traci = kernel_api
        self.group_lib = group_lib
        self.ROW = row_order
        self.COL = col_order
        self._veh_counter = itertools.count(1)
        self.window_variants = []

    def _ensure_route(self, route_id, edges):
        try:
            if route_id not in set(self.traci.route.getIDList()):
                self.traci.route.add(route_id, edges)
        except Exception:
            # route 模块可能不存在（极老 SUMO），那就跳过动态注册，依赖已有 routes
            pass

    def plan_window(self, base_od_rates, group_weights):
        variants = []
        for o in self.ROW:
            for d in self.COL:
                rate_h = float(base_od_rates.get(o, {}).get(d, 0.0))
                if rate_h <= 0:
                    continue
                pool = []
                for gk, gw in group_weights.items():
                    odv = self.group_lib[gk].get(o, {}).get(d, [])
                    for v in odv:
                        pool.append({
                            "route_id": v["route"],
                            "edges": v.get("edges", []),
                            "w": float(gw) * float(v.get("weight", 1.0))
                        })
                if not pool:
                    # 没有变体就让静态 inflow 继续，不做动态注入
                    continue
                tot = sum(max(1e-12, p["w"]) for p in pool)
                for p in pool:
                    share = p["w"] / tot
                    lam = (rate_h * share) / 3600.0  # veh/s
                    if p["edges"]:
                        self._ensure_route(p["route_id"], p["edges"])
                    variants.append({
                        "route_id": p["route_id"],
                        "lambda_per_sec": lam,
                        "departLane": "best",
                        "departPos": "free",
                        "departSpeed": 13.89
                    })
        self.window_variants = variants

    def tick(self, sim_time_s: int):
        for v in self.window_variants:
            lam = v["lambda_per_sec"]
            if lam <= 0:
                continue
            n_new = np.random.poisson(lam)
            for _ in range(n_new):
                vid = f"dyn_{next(self._veh_counter)}"
                try:
                    self.traci.vehicle.add(
                        vehID=vid, routeID=v["route_id"],
                        departLane=v["departLane"],
                        departPos=v["departPos"],
                        departSpeed=v["departSpeed"]
                    )
                except Exception:
                    # 放不进去就跳过
                    pass


# ====== 连续仿真 WorstEstimatorEnv（不重启、不依赖 .unwrapped/.last_obs）======
import gym
class WorstEstimatorEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, env_config):
        super().__init__()
        self.base_od_csv   = env_config.get("od_csv")
        self.alpha         = float(env_config.get("alpha", 1.0))
        self.beta          = float(env_config.get("beta", 1.0))
        self.tl_result_dir = env_config["tl_result_dir"]
        self.tl_ckpt       = env_config["tl_ckpt"]
        self.horizon       = int(HORIZON)

        # 只创建一次 Flow 环境；不再 reset
        create_env, _ = make_create_env(params=flow_params, version=0)
        self.env = create_env()
        self.env.reset()

        # kernel 句柄（兼容不同 Flow 版本：env.k 或 env.k.kernel_api）
        self.k = getattr(self.env, "k", None)
        if self.k is None:
            raise RuntimeError("Flow env has no attribute 'k'")
        self.kernel_api = getattr(self.k, "kernel_api", None)
        if self.kernel_api is None:
            # 极老版本可能直接是 traci
            self.kernel_api = getattr(self.env, "kernel_api", None)
        if self.kernel_api is None:
            raise RuntimeError("Cannot access kernel_api (TraCI) from Flow env")

        # 固定红绿灯策略
        try:
            from ray.rllib.agents.registry import get_agent_class
        except ImportError:
            from ray.rllib.agents import get_agent_class
        # 复用 visualizer_rllib 的思路：读取 params.json 等，如果你已有 helper 就直接用之
        from flow.utils.rllib import get_rllib_config as _get_rllib_config
        config = _get_rllib_config(self.tl_result_dir)
        run = config.get("env_config", {}).get("run", None)
        if run is None:
            raise RuntimeError("params.json 缺少 env_config.run（与 visualizer_rllib.py 一致的保存方式）")
        agent_cls = get_agent_class(run)
        self.tl_agent = agent_cls(env=None, config=config)
        ckpt = os.path.join(self.tl_result_dir, f"checkpoint_{self.tl_ckpt}", f"checkpoint-{self.tl_ckpt}")
        self.tl_agent.restore(ckpt)
        self.policy_map = config["multiagent"]["policy_mapping_fn"]
        self.use_lstm = bool(config.get("model", {}).get("use_lstm", False))
        if self.use_lstm:
            size = config["model"]["lstm_cell_size"]
            self.state_init = {}
            for pid in config["multiagent"]["policies"]:
                self.state_init[pid] = [np.zeros(size, np.float32), np.zeros(size, np.float32)]

        # 观测/动作空间
        sample_obs = collect_region_stats(self.env)
        self.observation_space = gym.spaces.Box(low=-1e9, high=1e9, shape=sample_obs.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(len(GROUP_KEYS),), dtype=np.float32)

        # 注车器 & 基础 OD
        self.base_od_rates = read_triplist_csv_to_od_rates(self.base_od_csv)
        self.spawner = OnlineSpawner(self.kernel_api, GROUP_LIB, ROW_ORDER, COL_ORDER)

        # 连续仿真计时
        self._sim_time = 0
        # 让系统先“热身”一个窗口，保证第一步前有车辆与稳定观测
        self._warmup_one_window()

    def _warmup_one_window(self):
        warm_w = {gk: 1.0/len(GROUP_KEYS) for gk in GROUP_KEYS}
        self.spawner.plan_window(self.base_od_rates, warm_w)
        for _ in range(self.horizon):
            self.spawner.tick(self._sim_time)
            tl_actions = self._compute_tl_actions(self._safe_last_ma_obs())
            _, _, done, _ = self.env.step(tl_actions)
            self._sim_time += 1
            if self._done_true(done):
                break

    def _done_true(self, done):
        return (isinstance(done, dict) and done.get("__all__", False)) or (isinstance(done, bool) and done)

    def _safe_last_ma_obs(self):
        """
        Flow 的多智能体 env 每步返回的是 dict obs；
        这里通过 env.step 的返回来串联，而不是访问 .unwrapped/.last_obs 这种不稳定属性。
        简单做法：在 step 循环中把上一步 state 缓存；首次时用 env.reset() 的返回。
        """
        # 为了简化，在本实现中：tl 动作用上一步的 state（在 step() 循环里维护）
        return getattr(self, "_last_state_dict", {})

    def _compute_tl_actions(self, ma_obs_dict):
        actions = {}
        if not isinstance(ma_obs_dict, dict):
            return actions
        for agent_id, ob in ma_obs_dict.items():
            policy_id = self.policy_map(agent_id)
            if self.use_lstm:
                act, self.state_init[policy_id], _ = self.tl_agent.compute_action(
                    ob, state=self.state_init[policy_id], policy_id=policy_id)
            else:
                act = self.tl_agent.compute_action(ob, policy_id=policy_id)
            actions[agent_id] = act
        return actions

    def reset(self):
        # 连续仿真：不 reset SUMO，仅返回当前观测
        return collect_region_stats(self.env)

    def step(self, action):
        # 1) simplex 权重
        w = np.exp(action - np.max(action))
        w = w / np.clip(np.sum(w), 1e-8, None)
        group_weights = {gk: float(w[i]) for i, gk in enumerate(GROUP_KEYS)}

        # 2) 规划下一窗口的注车率
        self.spawner.plan_window(self.base_od_rates, group_weights)

        # 3) 连续推进 900 步
        stats = SegmentStats()
        last_state = None
        for _ in range(self.horizon):
            self.spawner.tick(self._sim_time)
            tl_actions = self._compute_tl_actions(last_state if isinstance(last_state, dict) else {})
            state, _, done, _ = self.env.step(tl_actions)
            self._last_state_dict = state if isinstance(state, dict) else {}
            stats.update(self.env)
            self._sim_time += 1
            last_state = state
            if self._done_true(done):
                break

        obs_next = collect_region_stats(self.env)
        seg_cwt = stats.total_wait
        seg_avs = stats.avg_speed
        reward = float(self.alpha * seg_cwt - self.beta * seg_avs)
        info = {"weights": group_weights, "CWT": seg_cwt, "AvgSpeed": seg_avs, "sim_time": self._sim_time}
        return obs_next, reward, False, info

   

# ============ RLlib 注册 & 训练入口 ============
def env_creator(env_config) -> EnvType:
    return WorstEstimatorEnv(env_config)

def main():
    # ✅ 让 RLlib 在主进程运行，直接打印错误
    ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=True)

    register_env("WorstEstimatorEnv", env_creator)

    TL_RESULT_DIR = "/home/sdc_joran/ray_results/grid_0_3x3_i4800_multiagent/PPO_MultiTrafficLightGridPOEnv-v1_0a4a3ca0_2025-11-04_13-51-54_yhihbzy"
    TL_CKPT_NUM   = "3454"
    OD_CSV = "/home/sdc_joran/Athena Data/data/aoi_od_pairs_group1.csv"

    config = {
        "env": "WorstEstimatorEnv",
        "env_config": {
            "od_csv": OD_CSV,
            "alpha": 1.0,
            "beta": 1.0,
            "tl_result_dir": TL_RESULT_DIR,
            "tl_ckpt": TL_CKPT_NUM,
        },
        "num_workers": 0,      # 👈 让一切在主进程执行
        "num_gpus": 0,
        "train_batch_size": 2,
        "sgd_minibatch_size": 2,
        "num_sgd_iter": 2,
        "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
    }

    stop = {"training_iteration": 3}

    # ✅ 加 verbose 和容错
    tune.run(
        PPOTrainer,
        config=config,
        stop=stop,
        local_dir="./ray_results_worst_estimator",
        raise_on_failed_trial=False,
        verbose=3
    )


if __name__ == "__main__":
    main()
