#!/usr/bin/env python3
"""ROS2 wrapper for the EVO-MPCC core (acados / IPOPT backends).

This is the **skeleton** of the ROS2 node — full feature parity with the
original ROS1 `Nonlinear_MPC_node.py` (56 KB) is reached incrementally.
What's here today:

  - Parameter declaration (mirroring `ddrx_unified_params.yaml`)
  - Subscriptions: odom, pose, goal, clicked_point, external_obstacles
  - Publications: ackermann cmd, MPC trajectory (Path + MPCTrajectory),
                  boundary marker, debug topics
  - 40 Hz control timer that pulls latest state, runs `mpc.solve(...)`,
    publishes outputs
  - boundary_hook adapter that builds MarkerArray from the points the
    core hands back
  - on_set_parameters_callback that pushes live-tunable weights into
    `mpc.q_*_live` attributes (replaces ROS1 dynamic_reconfigure)

What's intentionally still TODO (carried over from the ROS1 node, ported
in follow-up commits):

  - Track data loading: read centerline / boundary / waypoints CSV,
    build CasADi spline interpolants, call `mpc.set_track_data(...)`.
    See `ros1_source/nonlinear_mpc_casadi/scripts/Nonlinear_MPC_node.py`
    (`load_track_data` / `_build_*_lut`).
  - Static-obstacle preload from `~static_obstacles` param.
  - Lap-counter integration (consume LapData).
  - Joy / state machine integration.
  - `/behavior_strategy` direct publish bypass.
  - Visualization: `/center_path`, `/right_path`, `/left_path`,
    `/reference_path` (latched), trajectory marker arrows.

The numerical MPC code (`mpc_core.acados_kinematic.MPC`) is unchanged
from the ROS1 version after Phase 1's ROS-decoupling — the wrapper here
only handles ROS2 IO. Replace `MPC` with `mpc_core.ipopt_kinematic.MPC`
to swap the backend (mirrors the ROS1 `mpc_backend` parameter).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from .track_loader import TrackData, build_track_from_wpnts, find_current_arc_length, load_track
from .mpc_core.model_policy import A_MIN_DYN, clamp_a_lat_to_grip, lmpc_anchor_s
# 2026-05-28 #18 LMPC integration
from .mpc_core.lmpc.lap_database import LapDatabase
from .mpc_core.lmpc.safe_set import SafeSetLookup
from .mpc_core.lmpc.nominal_dynamics import predict_next

from std_msgs.msg import Bool, ColorRGBA, Float32MultiArray, Float64, Header, Int32
from geometry_msgs.msg import (
    Point, PointStamped, PoseArray, PoseStamped, PoseWithCovarianceStamped,
    Quaternion, Vector3,
)
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

from osuf1_common.msg import MPCTrajectory, MPCPrediction
from f110_msgs.msg import LapData, WpntArray  # noqa: F401  (used in TODO sections)


# ──────────────────────────────────────────────────────────────────────────
# Backend selection — keep symmetric with ROS1 `mpc_backend` parameter
# ──────────────────────────────────────────────────────────────────────────
def _load_mpc_backend(name: str):
    if name == 'acados':
        from .mpc_core.acados_kinematic import MPC
        return MPC
    if name == 'ipopt':
        from .mpc_core.ipopt_kinematic import MPC
        return MPC
    raise ValueError(f"unknown mpc_backend: {name!r} (expected 'acados' or 'ipopt')")


# ──────────────────────────────────────────────────────────────────────────
# Adapter: rclpy logger → mpc_core logger interface
# ──────────────────────────────────────────────────────────────────────────
class _RclpyLoggerAdapter:
    """Bridge rclpy node logger to the mpc_core logger contract.

    `mpc_core` calls `info(msg, *args)` with %-style format args (matches
    the rospy convention). rclpy's logger expects a pre-formatted string,
    so we format here. Throttled variants emulate `rospy.logwarn_throttle`
    by tracking last-fire time per message template.
    """
    def __init__(self, node_logger):
        self._lg = node_logger
        self._last: dict[str, float] = {}

    @staticmethod
    def _fmt(msg, args):
        try:
            return msg % args if args else msg
        except Exception:
            return msg + " " + " ".join(str(a) for a in args)

    def info(self, msg, *args):    self._lg.info(self._fmt(msg, args))
    def warn(self, msg, *args):    self._lg.warn(self._fmt(msg, args))
    def warning(self, msg, *args): self._lg.warn(self._fmt(msg, args))
    def error(self, msg, *args):   self._lg.error(self._fmt(msg, args))
    def debug(self, msg, *args):   self._lg.debug(self._fmt(msg, args))

    def _throttled(self, fn, period: float, msg, args):
        now = time.monotonic()
        if now - self._last.get(msg, 0.0) >= period:
            self._last[msg] = now
            fn(msg, *args)

    def info_throttle(self, period, msg, *args):
        self._throttled(self.info, period, msg, args)

    def warn_throttle(self, period, msg, *args):
        self._throttled(self.warn, period, msg, args)


# ──────────────────────────────────────────────────────────────────────────
# Live-tunable parameters — names match `mpc_core` `*_live` attributes.
# Updated via `on_set_parameters_callback` (replaces ROS1 dyn_reconfigure).
# ──────────────────────────────────────────────────────────────────────────
LIVE_PARAMS = [
    ('q_cte_live', 8.0),
    ('q_lag_live', 200.0),
    ('q_d_delta_live', 25.0),
    ('R_safe_live', 0.3),
    ('M_slack_live', 2.0e4),
    ('a_lat_safe_live', 6.0),
    ('D_detour_live', 0.15),
    ('D_apex_live', 0.22),
    ('R_car_live', 0.0),
    ('commit_dist_live', 10.0),
    ('cost_spike_thr_live', 500.0),
    ('alpha_steer_live', 0.6),
    # acados-specific scale slots
    ('q_cte_scale_live', 1.0),
    ('q_lag_scale_live', 1.0),
    ('q_psi_scale_live', 1.0),
    ('q_v_scale_live', 1.0),
    ('q_dd_scale_live', 1.0),
    ('q_p_scale_live', 1.0),
    ('q_drate_scale_live', 1.0),
    ('q_dv_scale_live',    1.0),   # 2026-05-27 #8 — a_x penalty
]


class MPCNode(Node):
    CONTROLLER_FREQ = 40.0  # Hz — matches ROS1 default

    def __init__(self):
        super().__init__('mpc_node')
        self._latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('mpc_backend', 'acados')
        self.declare_parameter('odom_topic_name', '/car_state/odom')
        self.declare_parameter('localized_pose_topic_name', '/car_state/pose')
        self.declare_parameter('goal_topic_name', '/move_base_simple/goal')
        self.declare_parameter('cmd_vel_topic_name',
                               '/vesc/high_level/ackermann_cmd_mux/input/nav_1')
        # Track / MPC sizing
        self.declare_parameter('track_name', 'f')
        self.declare_parameter('track_dir', '')   # empty → use share/tracks/
        self.declare_parameter('vel_scale', 0.95)
        self.declare_parameter('inflation_factor', 1.2)
        self.declare_parameter('extend_part', 2)
        # Fixed-width corridor (centerline ± half). >0 = enabled, 0 = raw
        # wpnt.d_left/d_right + inflation (기존 동작). 자세한 설명: ddrx_unified_params.yaml.
        self.declare_parameter('mpc_corridor_half_width', 0.0)
        # Corridor-width speed cap (complements κ-cap; fixes narrow-but-straight
        # wall clips). v_floor<=0 disables. width=usable corridor (d_l+d_r,
        # post-margin); ramp speed from floor (at tight) to v_max (at wide).
        self.declare_parameter('corridor_v_floor', 0.0)
        self.declare_parameter('corridor_v_tight_width', 1.0)
        self.declare_parameter('corridor_v_wide_width', 1.6)
        self.declare_parameter('vehicle_L', 0.307)
        self.declare_parameter('max_speed', 6.0)
        self.declare_parameter('max_speed_p', 6.0)
        self.declare_parameter('mpc_max_steering', 0.4)
        self.declare_parameter('dT', 0.025)
        self.declare_parameter('N_horizon', 18)
        # ── Cold-start / low-speed conditioning (2026-05-29) ──
        # MPPI init_vel 기법: 정지·저속에서 dynamic Pacejka 모델은 vx→0 에서
        # slip atan2 항이 ill-conditioned → cost surface 가 δ 에 거의 flat →
        # solver 가 ±zigzag (진단: v<2 에서 steer chatter 2.6배). 해결책:
        #   (a) solver 에 넘기는 x0 의 vx 를 floor (모델이 항상 well-conditioned
        #       영역에서 풂) → 저속 chatter 제거.
        #   (b) 실측 vx 가 floor 미만이면 출력 speed 를 startup_speed 로 floor
        #       → 차가 가속해 floor 를 넘김 (PP warmup ramp 불필요, t=0 부터 MPCC).
        # floor=0 → 비활성 (기존 동작). 둘 다 같은 root cause (저속 ill-cond) 해결.
        self.declare_parameter('cold_start_vx_floor', 2.0)
        self.declare_parameter('startup_speed', 1.5)
        # multi-dt time_steps mode: 'uniform' (모든 step dt=dT) | 'pyramidal' (가까이 sharp, 멀리 long).
        # pyramidal 은 horizon 늘어남 — 다음 코너까지 보여 racing line 발견 ↑.
        self.declare_parameter('time_steps_mode', 'uniform')
        self.declare_parameter('integration_mode', 'Euler')
        self.declare_parameter('params_file', 'BO_params_LTM')
        # Track source: 'centerline' (/centerline_waypoints) or 'raceline'
        # (/global_waypoints, IQP 사전 최적 raceline). raceline 사용 시 mpc
        # 가 racing line 추종 — 코너에서 자연스럽게 apex 통과 (EVO-MPCC /
        # Liniger MPCC 의 표준 패턴).
        self.declare_parameter('track_source', 'centerline')
        # auto_tune=true → max_speed 만 사용자 설정, 나머지 cost weights /
        # corridor / EMA 등은 _auto_tune_from_max_speed() 의 heuristic 매핑
        # 으로 자동 결정. mpc init 시 1회 적용.
        self.declare_parameter('auto_tune', False)
        # C: MLP weight scaler (학습된 모델로 q_*_scale_live 결정).
        # use_ml_scale=true 면 B heuristic 대신 NN 추론 사용.
        # ml_model_path: TorchScript .pt 경로. 빈 문자열이면 default 위치.
        self.declare_parameter('use_ml_scale', False)
        self.declare_parameter('ml_model_path', '')
        # BO sweep / 수동 override 모드.
        #   'off'        → MLP/B 그대로 (기본)
        #   'fixed'      → override_q_*_scale 4개 고정값 적용
        #   'bucketed'   → κ_abs 기반 3 bucket × 4 scale = 12 param 적용 (BO 학습용)
        #   'polynomial' → BO 결과로 학습된 poly(v) → bucket 별 scale (실시간 deploy)
        self.declare_parameter('override_mode', 'off')
        self.declare_parameter('poly_path', '')   # polynomial JSON 경로 (override_mode='polynomial')
        # ── 모델 선택 (2026-06-10 unified 8-state layout — 양 모드 동일 구조) ──
        # use_dynamic=False (kinematic f_kin, 8-state): tire slip 미모델 (고그립 차용)
        # use_dynamic=True  (dynamic blended Pacejka, 8-state): tire slip 정확 (저그립 차용)
        # 둘 다 [x,y,ψ,vx,vy,r,s,δ_prev] + u=[a_x,δ,p_v] → LMPC/BO/GP/B4' 모두 동일 동작
        # dyn_tire_model: 'linear' (안전·빠름) / 'tanh' (saturation) / 'pacejka' (정확·불안정 가능)
        self.declare_parameter('use_dynamic', False)
        self.declare_parameter('nlp_solver_iters', 1)   # 1=SQP_RTI; >1=SQP(capped)+globalization (저속 RTI 함정 탈출)
        self.declare_parameter('dyn_tire_model', 'linear')
        # ── 그립 정직화 (2026-06-10) ──
        # dyn_mu: 모델 타이어 μ. sim_params.yaml 의 mu(또는 실차 노면)와 일치시켜야
        # 모델이 정직 — 저그립 실험은 dyn_mu:=0.6. codegen-time 상수 (mu 가 codegen
        # 디렉토리 태그에 키잉되므로 전환 시 수동 rm 불필요).
        self.declare_parameter('dyn_mu', 1.0489)
        # ellipse_frac: friction ellipse headroom η. a_lim = μ·g·η.
        self.declare_parameter('ellipse_frac', 0.95)
        # acados Levenberg-Marquardt regularization for the dynamic model.
        # Higher = better-conditioned QP (fewer ACADOS_MINSTEP / QP_Failure)
        # and more consistent cycle-to-cycle predictions, but over-damped if
        # too large. Exposed as a param so it can be swept / BO-tuned.
        self.declare_parameter('lm_dynamic', 1.0)
        # Phase D — GP residual learning (L4acados ResidualLearningMPC wrap).
        # use_gp_residual=true & gp_ckpt_path 존재 → setup_MPC 후 GP 적용.
        self.declare_parameter('use_gp_residual', False)
        self.declare_parameter('gp_ckpt_path', '')
        # Phase D (closed-form) — adds the trained GP posterior mean as a pure
        # CasADi expression to the dynamics ONLY (cost/p_sym untouched).
        # Independent of use_gp_residual (l4acados). Default OFF.
        self.declare_parameter('use_gp_casadi', False)
        self.declare_parameter('use_error_regression', False)
        self.declare_parameter('err_regr_bandwidth', 1.0)   # Epanechnikov h (× max neighbour vel-dist)
        self.declare_parameter('err_regr_ema', 0.8)          # B4' e_corr temporal EMA: β·prev+(1-β)·new
        # R3 decouple max_speed: 0 → derive from max_speed (behaviour unchanged).
        self.declare_parameter('speed_target', 0.0)          # q_p progress target (0 → max_speed)
        self.declare_parameter('lookahead_m', 0.0)           # ref_v κ window [m] (0 → max(6, v²/6))
        # brake-distance-aware 속도명령의 가정 감속도 [m/s²]. 낮을수록 코너를 더
        # 일찍/느리게 진입 (코너 안에선 friction-circle 로 못 서니까 미리 줄이려면 ↓).
        self.declare_parameter('brake_anticip_a', 2.0)
        # 실차 모드: /sim/initialpose 없음. STUCK recovery 시도해도 무의미 + 위험.
        # sim=true (default) / real=false.
        self.declare_parameter('enable_sim_reset', True)
        # fixed mode (W1) — 단일 4 scale.
        self.declare_parameter('override_scales', False)  # legacy: true → 'fixed' 와 동일
        self.declare_parameter('override_q_cte_scale', 1.0)
        self.declare_parameter('override_q_lag_scale', 1.0)
        self.declare_parameter('override_q_v_scale', 1.0)
        self.declare_parameter('override_q_drate_scale', 1.0)
        # bucketed mode (W2) — bucket boundaries (κ_abs):
        #   b0: κ ∈ [0,         bucket_kappa_b01)    → 직선/완만
        #   b1: κ ∈ [b01,       bucket_kappa_b12)    → medium corner
        #   b2: κ ∈ [b12,       +∞)                  → hairpin
        self.declare_parameter('bucket_kappa_b01', 0.3)
        self.declare_parameter('bucket_kappa_b12', 0.6)
        for b in (0, 1, 2):
            self.declare_parameter(f'override_q_cte_scale_b{b}',   1.0)
            self.declare_parameter(f'override_q_lag_scale_b{b}',   1.0)
            self.declare_parameter(f'override_q_v_scale_b{b}',     1.0)
            self.declare_parameter(f'override_q_drate_scale_b{b}', 1.0)
        # 데이터 수집 모드: lap 마다 effective max_speed 자동 증가.
        # codegen v_max (yaml max_speed) 는 end_speed 로 고정 → ubu cap.
        # 시작 시 mpc.v_max = start_speed → 매 lap_per_step 마다 step 증가.
        self.declare_parameter('auto_step_enable', False)
        self.declare_parameter('auto_step_start', 4.0)
        self.declare_parameter('auto_step_end', 12.0)
        self.declare_parameter('auto_step_size', 1.0)
        self.declare_parameter('auto_step_laps', 5)

        # ── LMPC (Learning MPC) parameters (2026-05-28 #18) ──
        self.declare_parameter('use_lmpc', False)
        self.declare_parameter('lmpc_w', 1.0)
        self.declare_parameter('lmpc_alpha', 1.0)
        self.declare_parameter('lmpc_beta', 0.05)
        self.declare_parameter('lmpc_reg_w', 0.001)
        self.declare_parameter('lmpc_K_points', 10)
        self.declare_parameter('lmpc_K_laps', 4)
        self.declare_parameter('lmpc_slice_window', 50)
        self.declare_parameter('lmpc_enable_after_real_laps', 1)
        self.declare_parameter('lmpc_load_path', '')
        self.declare_parameter('lmpc_save_path', '')
        self.declare_parameter('lmpc_seed_from_raceline', True)
        # IQP raceline json for the apex seed (grip-clamped). Set by launch to
        # maps/<map>/global_waypoints.json. Empty → centerline seed fallback.
        self.declare_parameter('lmpc_raceline_json', '')
        self.declare_parameter('lmpc_max_resets', 0)  # 2026-06-08: teleport된 랩은 safe-set에서 거부
        self.declare_parameter('lmpc_max_abs_ec_m', 1.0)
        self.declare_parameter('lmpc_max_lap_time_ratio', 1.5)
        self.declare_parameter('lmpc_max_stuck_seconds', 5.0)
        self.declare_parameter('lmpc_buffer_per_bucket', 10)
        # Vehicle dynamics (only `l_wb` consumed in kinematic mode; full set
        # used by acados dynamic Pacejka mode — kept here so swapping
        # use_dynamic doesn't require config changes.)
        self.declare_parameter('vehicle.l_wb', 0.307)
        self.declare_parameter('vehicle.l_f', 0.162)
        self.declare_parameter('vehicle.l_r', 0.145)
        self.declare_parameter('vehicle.m', 3.54)
        for name, default in LIVE_PARAMS:
            self.declare_parameter(name, default)

        backend_name = self.get_parameter('mpc_backend').value
        odom_topic = self.get_parameter('odom_topic_name').value
        pose_topic = self.get_parameter('localized_pose_topic_name').value
        goal_topic = self.get_parameter('goal_topic_name').value
        cmd_topic = self.get_parameter('cmd_vel_topic_name').value

        # ── MPC instance (numerical core) ───────────────────────────
        MPC = _load_mpc_backend(backend_name)
        self._mpc_log = _RclpyLoggerAdapter(self.get_logger())
        self.mpc = MPC(cost_type=None, system_model=None, logger=self._mpc_log)
        self.mpc.boundary_hook = self._on_boundary_points
        self._push_live_params_to_mpc()

        # ── Publications ────────────────────────────────────────────
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)
        self.mpc_traj_pub = self.create_publisher(Path, '/mpc_trajectory', 10)
        # MarkerArray twin of /mpc_trajectory — small magenta spheres, one per
        # mpc stage. Matches the visual style of /center_path etc. The Path
        # version is kept for mpc_debug_logger / other consumers.
        self.mpc_traj_markers_pub = self.create_publisher(
            MarkerArray, '/mpc_trajectory/markers', 10)
        self.boundary_pub = self.create_publisher(MarkerArray, '/boundary_marker', 10)
        self.prediction_pub = self.create_publisher(MPCTrajectory, '/mpc/prediction', 1)
        self.cost_pub = self.create_publisher(Float64, '/mpc/cost', 10)
        self.solve_time_pub = self.create_publisher(Float64, '/mpc/solve_time', 10)
        self.feasible_pub = self.create_publisher(Bool, '/mpc/is_feasible', 10)
        self.mpc_debug_pub = self.create_publisher(Float32MultiArray, '/mpc_debug', 10)
        # Agent A round-3 fix: gym 의 in_collision latch (base_classes.py:288) 가
        # wall contact 시 integration 완전 skip → reverse cmd 도 무시. /sim/initialpose
        # (gym_bridge_launch.py 의 remap target) publish → ego_reset_callback →
        # env.reset() 로 latch 해제. 5s cooldown 으로 폭주 방지.
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/sim/initialpose', 10)
        self._stuck_release_total = 0
        # Per-lap teleport (safe-reset) count for LMPC lap-acceptance. Counted in
        # _publish_safe_reset (the single choke point both kinematic & dynamic
        # branches pass through), read+reset at each lap boundary. Replaces the
        # old _stuck_release_total sync which only incremented in the kinematic
        # branch → dynamic-mode teleports counted as 0 → crash laps polluted SS.
        self._lmpc_lap_teleports = 0
        self._last_reset_t = 0.0
        # Latched (TRANSIENT_LOCAL) viz publishers — small-sphere MarkerArray
        # in the style of race-stack's /centerline_waypoints/markers (type=2
        # SPHERE, scale 0.05). Different color per topic so RViz tells them
        # apart at a glance.
        #   /center_path   raw centerline (white)
        #   /right_path    raw right boundary (red)
        #   /left_path     raw left boundary (green)
        #   /reference_path mpc reference = centerline (yellow)
        self.center_path_pub    = self.create_publisher(MarkerArray, '/center_path',    self._latched_qos)
        self.right_path_pub     = self.create_publisher(MarkerArray, '/right_path',     self._latched_qos)
        self.left_path_pub      = self.create_publisher(MarkerArray, '/left_path',      self._latched_qos)
        self.reference_path_pub = self.create_publisher(MarkerArray, '/reference_path', self._latched_qos)

        # ── Subscriptions ───────────────────────────────────────────
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 1)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 1)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 1)
        self.create_subscription(PointStamped, '/clicked_point',
                                 self._clicked_point_cb, 4)
        self.create_subscription(PoseArray, '/external_obstacles',
                                 self._external_obs_cb, 1)
        # ── mux 제어권 추적 (2026-06-11) ─────────────────────────────
        # 실차 버그: 조이 수동 전환 중에도 MPC는 계속 solve → STUCK 오감지·
        # warm-start stale·_last_sensor_s 점프(lap_count 꼬임). 수동→자동
        # 복귀 시 teleport 와 동일한 latch 리셋을 수행.
        self._mux_mpcc_active: bool | None = None   # None=신호 미수신(구버전 mux 호환)
        self._mux_inactive_since: float | None = None
        self.create_subscription(Bool, '/mux/mpcc_active',
                                 self._mux_active_cb, 1)

        # ── State cache (most-recent message snapshots) ─────────────
        self._last_odom: Odometry | None = None
        self._last_pose: PoseStamped | None = None
        self._goal: PoseStamped | None = None
        self._obstacles: list[tuple[float, float]] = []
        self._obstacles_stamp: float = 0.0
        self._lap_count: int = 0  # TODO: subscribe to LapData
        self._mpc_ready: bool = False
        self._track: TrackData | None = None

        # ── Live parameter callback (replaces dynamic_reconfigure) ──
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Track ingestion: prefer live race-stack /centerline_waypoints
        # (so mpc shares the sim's coordinate frame — origin / map.yaml
        # mismatches between CSV-baked tracks and race-stack maps would
        # otherwise put the reference path in the wrong world location).
        # CSV fallback fires after `_track_csv_fallback_sec` if no wpnts
        # arrive — preserves standalone-CSV use case (bag replay, dry run).
        self._track_init_done = False
        self._track_csv_fallback_sec = 5.0
        # global_republisher publishes both /centerline_waypoints and
        # /global_waypoints with default QoS (RELIABLE + VOLATILE).
        # track_source param selects which one mpc uses as reference:
        #   centerline → /centerline_waypoints (864 wpnts, raw centerline)
        #   raceline   → /global_waypoints     (766 wpnts, IQP raceline)
        track_source = str(self.get_parameter('track_source').value).strip().lower()
        if track_source not in ('centerline', 'raceline'):
            self.get_logger().warn(
                f"unknown track_source={track_source!r} — falling back to 'centerline'")
            track_source = 'centerline'
        self._track_source = track_source
        ref_topic = ('/global_waypoints' if track_source == 'raceline'
                     else '/centerline_waypoints')
        self.get_logger().info(
            f"mpc reference path: {track_source} ({ref_topic})")
        self.create_subscription(
            WpntArray, ref_topic, self._on_centerline_wpnts, 10)
        self._track_fallback_timer = self.create_timer(
            self._track_csv_fallback_sec, self._track_csv_fallback_cb)

        # ── C: ML weight scaler load (use_ml_scale=true 면) ────────────
        self._load_ml_scaler()
        # ── D: BO-learned polynomial coefficients (override_mode='polynomial' 시) ──
        self._load_polynomial()

        # ── 자동 max_speed stepper (auto_step_enable=true 면) ──
        # 같은 launch 안에서 lap 카운트 기반으로 effective v_max 자동 증가.
        # codegen ubu cap 은 yaml max_speed (예: 12) 로 고정 — 그게 절대 상한.
        # mpc.v_max (cost target) 만 step 마다 변경.
        self._auto_step_state = {
            'enabled':      bool(self.get_parameter('auto_step_enable').value),
            'start':        float(self.get_parameter('auto_step_start').value),
            'end':          float(self.get_parameter('auto_step_end').value),
            'size':         float(self.get_parameter('auto_step_size').value),
            'laps':         int(self.get_parameter('auto_step_laps').value),
            'last_lap':     -1,
            'lap_at_step':  0,
            'current_step': 0,
        }
        # /mpc/lap_count subscription drives BOTH auto_step max_speed ramp AND
        # LMPC lap-end buffering (_on_lap_count_step → _lmpc_on_lap_end).
        # BUG (2026-05-29): this was gated on auto_step only, so with auto_step
        # off (BO/normal default) LMPC never received a lap-end event → the
        # safe set never accumulated real laps → LMPC was silently a no-op
        # (laps stayed flat at ~19s, 0 buffered). Subscribe if EITHER needs it.
        if self._auto_step_state['enabled'] or bool(self.get_parameter('use_lmpc').value):
            self.create_subscription(Int32, '/mpc/lap_count',
                                     self._on_lap_count_step, 10)
        if self._auto_step_state['enabled']:
            # 시작 시 mpc.v_max 를 start_speed 로 (auto_tune 동시 트리거).
            # 단 mpc 초기화 후에야 self.mpc 존재 → wpnt callback 안의 후처리
            # 가 더 안전하므로 여기선 state 만 저장. 실제 적용은 _initialize_mpc
            # 끝에서 _set_effective_max_speed() 호출 (아래 추가).
            self.get_logger().info(
                f"[auto_step] enabled: {self._auto_step_state['start']:.1f} → "
                f"{self._auto_step_state['end']:.1f} m/s, "
                f"+{self._auto_step_state['size']:.1f} / "
                f"{self._auto_step_state['laps']} lap")

        # ── LMPC infrastructure (2026-05-28 #18) ────────────────────
        self._lmpc_use = bool(self.get_parameter('use_lmpc').value)
        self._lmpc_db = LapDatabase(
            buffer_per_bucket=int(self.get_parameter('lmpc_buffer_per_bucket').value),
            max_resets_accept=int(self.get_parameter('lmpc_max_resets').value),
            min_lap_steps=50,
            max_lap_time_ratio=float(self.get_parameter('lmpc_max_lap_time_ratio').value),
            max_abs_ec_m=float(self.get_parameter('lmpc_max_abs_ec_m').value),
            max_stuck_seconds=float(self.get_parameter('lmpc_max_stuck_seconds').value),
        )
        self._lmpc_ss = SafeSetLookup(
            self._lmpc_db,
            K_points=int(self.get_parameter('lmpc_K_points').value),
            K_laps=int(self.get_parameter('lmpc_K_laps').value),
            slice_window=int(self.get_parameter('lmpc_slice_window').value),
        )
        self._lmpc_enable_after = int(self.get_parameter('lmpc_enable_after_real_laps').value)
        self._lmpc_load_path = str(self.get_parameter('lmpc_load_path').value).strip()
        self._lmpc_save_path = str(self.get_parameter('lmpc_save_path').value).strip()
        self._lmpc_seed_raceline = bool(self.get_parameter('lmpc_seed_from_raceline').value)
        # cost weights (passed through mpc.lmpc_*_live attrs every cycle)
        self._lmpc_w_target     = float(self.get_parameter('lmpc_w').value)
        self._lmpc_alpha        = float(self.get_parameter('lmpc_alpha').value)
        self._lmpc_beta         = float(self.get_parameter('lmpc_beta').value)
        self._lmpc_reg_w        = float(self.get_parameter('lmpc_reg_w').value)
        # Per-lap state/input/timestep accumulator (cleared on lap end)
        self._lmpc_lap_buf = {
            'state': [], 'input': [], 'time': [],
            'lap_start_t': None, 'last_lap': -1,
            'max_abs_ec': 0.0, 'stuck_seconds_accum': 0.0,
        }
        self.get_logger().info(
            f"[LMPC] use_lmpc={self._lmpc_use}  enable_after_real_laps={self._lmpc_enable_after}  "
            f"seed_raceline={self._lmpc_seed_raceline}  load={self._lmpc_load_path or '(none)'}  "
            f"save={self._lmpc_save_path or '(none)'}")
        # Defer raceline seed + npz load until after track loaded (mpc setup) —
        # called in `_initialize_mpc` end-of-setup hook.

        # ── Control loop ────────────────────────────────────────────
        # period = yaml dT 와 일치 (MPC 의 첫 step dt 와 같게).
        # multi-dt 면 dT 는 가까이 step 의 dt (sharp control rate).
        period = float(self.get_parameter('dT').value)
        # CONTROLLER_FREQ 는 1/dT 로 동기화 (display 용).
        self.CONTROLLER_FREQ = 1.0 / period
        self.control_timer = self.create_timer(period, self._control_loop_cb)

        self.get_logger().info(
            f"MPC node up — backend={backend_name}, "
            f"rate={self.CONTROLLER_FREQ:.0f} Hz, ready={self._mpc_ready} "
            f"(waiting up to {self._track_csv_fallback_sec:.1f}s for "
            "/centerline_waypoints, then CSV fallback)")

    # ─────────────────────────────────────────────────────────────────
    # Track + MPC setup
    # ─────────────────────────────────────────────────────────────────
    def _share_dir(self) -> str:
        return get_package_share_directory(__package__)

    def _resolve_track_dir(self) -> str:
        explicit = str(self.get_parameter('track_dir').value or '').strip()
        if explicit:
            return explicit
        # Default: share/tracks/ inside this installed package
        return os.path.join(self._share_dir(), 'tracks')

    def _resolve_bo_params(self) -> dict:
        """Load Bayesian-Opt MPC weights (BO_params_*.json) from share/config/mpc/."""
        params_file = self.get_parameter('params_file').value
        json_path = os.path.join(self._share_dir(), 'config', 'mpc', f'{params_file}.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"MPC weights JSON not found: {json_path}")
        with open(json_path) as f:
            return json.load(f)

    def _a_lat_eff_for_track(self) -> float:
        """track ref_v 용 a_lat — μgη clamp (BO 가 뭘 요청하든 물리 그립 상한)."""
        eff, clamped = clamp_a_lat_to_grip(
            float(self.get_parameter('a_lat_safe_live').value),
            float(self.get_parameter('dyn_mu').value),
            float(self.get_parameter('ellipse_frac').value))
        if clamped and not getattr(self, '_grip_track_warned', False):
            self._grip_track_warned = True
            self.get_logger().warn(
                f"[grip] a_lat_safe_live > μgη → track ref_v 는 {eff:.2f} 로 clamp")
        return eff

    def _build_param_dict(self, bo: dict) -> dict:
        """Match `getPreDefinedParas()` from the ROS1 node — keys mpc_core consumes."""
        gp = self.get_parameter
        return {
            'is_jit': False,
            'mpc_v_track':    float(bo.get('q_v', 0.0)),
            'mpc_w_cte':      float(bo['q_cte']),
            'mpc_w_lag':      float(bo['q_lag']),
            'mpc_w_accel':    float(bo['q_dv']),
            'mpc_w_delta_d':  float(bo['q_d_delta']),
            'mpc_w_delta_vp': float(bo['q_dvp']),
            'mpc_vp_project': float(bo['gamma']),
            'N': int(gp('N_horizon').value),
            'dT': float(gp('dT').value),
            'time_steps_mode': str(gp('time_steps_mode').value) if self.has_parameter('time_steps_mode') else 'uniform',
            'theta_max': float(gp('mpc_max_steering').value),
            'v_max': float(gp('max_speed').value),
            'p_min': 0.0,
            'p_max': float(gp('max_speed_p').value),
            'INTEGRATION_MODE': str(gp('integration_mode').value),
            'L': float(gp('vehicle_L').value),
            'x_min': -200.0, 'x_max': 200.0,
            'y_min': -200.0, 'y_max': 200.0,
            'psi_min': -1000.0, 'psi_max': 1000.0,
            's_min': 0.0, 's_max': 200.0,        # s_max overwritten after track load
            'Vbias_max': 10.0,
            'q_mu': 0.5,
            'M_slack': 100.0,
            'R_safe': 0.5,
            'a_lat_max': 12.0,
            'w_alat': 0.0,
        }

    def _on_centerline_wpnts(self, msg: WpntArray) -> None:
        """Race-stack `/centerline_waypoints` ingestion. Fires once; subsequent
        messages are ignored (the same static track is republished at 0.5Hz).
        Cancels the CSV fallback timer."""
        if self._track_init_done:
            return
        self._track_init_done = True
        if self._track_fallback_timer is not None:
            self._track_fallback_timer.cancel()
            self._track_fallback_timer = None
        try:
            self._initialize_mpc(wpnts=msg.wpnts)
        except Exception as e:
            self.get_logger().error(
                f"MPC init from /centerline_waypoints failed: {e}. Control loop will idle.")
            self._mpc_ready = False

    def _track_csv_fallback_cb(self) -> None:
        """No `/centerline_waypoints` arrived within the grace window — fall
        back to bundled CSV (standalone use, bag replay, IFAC dry-run)."""
        if self._track_init_done:
            return
        self._track_init_done = True
        if self._track_fallback_timer is not None:
            self._track_fallback_timer.cancel()
            self._track_fallback_timer = None
        self.get_logger().warn(
            f"/centerline_waypoints not seen in {self._track_csv_fallback_sec:.1f}s — falling back to CSV")
        try:
            self._initialize_mpc(wpnts=None)
        except Exception as e:
            self.get_logger().error(
                f"MPC CSV-fallback init failed: {e}. Control loop will idle.")
            self._mpc_ready = False

    def _set_effective_max_speed(self, v: float) -> None:
        """Live update of mpc cost target v_max (acados ubu cap unchanged).
        auto_tune 매핑도 재계산하여 q_lag/q_cte/EMA 등 effective 기반 갱신."""
        v = float(v)
        try:
            self.mpc.v_max = v       # cost target (p_v - v_max) residual
            self.mpc.p_max = v       # progress upper bound
        except Exception:
            pass
        # auto_tune 매핑 즉시 적용 (q_lag, q_cte, alpha 등)
        if bool(self.get_parameter('auto_tune').value):
            self._auto_tune_from_max_speed(v_override=v)
        self.get_logger().info(f"[auto_step] effective v_max → {v:.2f} m/s")

    # =================================================================
    # LMPC integration helpers (2026-05-28 #18)
    # =================================================================
    def _lmpc_load_or_seed(self) -> None:
        """After mpc setup, optionally load SS from npz or seed from raceline."""
        if not hasattr(self, 'mpc') or self.mpc is None:
            return
        # 1) npz load (highest priority — pre-trained SS)
        if self._lmpc_load_path:
            from os.path import expanduser
            path = expanduser(self._lmpc_load_path)
            try:
                self._lmpc_db.load_all(path)
                self.get_logger().info(f"[LMPC] loaded SS from {path}: {self._lmpc_db.summary()}")
            except Exception as e:
                self.get_logger().warn(f"[LMPC] load failed ({path}): {e}")

        # 2) raceline seed (cold start). Step 3: PREFER the IQP raceline (apex line)
        #    with GRIP-CLAMPED speed min(v_max, √(a_lat/|κ_raceline|)) — gives the
        #    SS Q-function a feasible FASTER target than centerline. Falls back to
        #    centerline geometry if no raceline json. (joint-α + soft anchor + hard
        #    corridor keeps the car drivable even where the raceline is outside it.)
        if self._lmpc_seed_raceline and self._track is not None:
            try:
                v_max_now = float(getattr(self.mpc, 'v_max', 5.0))
                _eff_fn = getattr(self.mpc, 'a_lat_safe_eff', None)  # μgη clamp (5번째 소비 지점)
                a_lat = float(_eff_fn()) if _eff_fn else float(getattr(self.mpc, 'a_lat_safe_live', 9.0))
                tr = self._track
                xy = psi = s_arr = v_arr = None
                rl_json = str(self.get_parameter('lmpc_raceline_json').value)
                if rl_json and os.path.exists(rl_json):
                    import json as _json
                    gw = _json.load(open(rl_json))
                    w = gw.get('global_traj_wpnts_iqp', {}).get('wpnts', [])
                    if len(w) >= 10:
                        xy = np.array([[p['x_m'], p['y_m']] for p in w], dtype=float)
                        psi = np.array([p['psi_rad'] for p in w], dtype=float)
                        s_arr = np.array([p['s_m'] for p in w], dtype=float)
                        kap = np.abs(np.array([p['kappa_radpm'] for p in w], dtype=float))
                        v_arr = np.minimum(v_max_now, np.sqrt(a_lat / np.maximum(kap, 1e-3)))
                        # Clamp seed LATERAL to the corridor (raceline reaches ±0.97 >
                        # corridor ±0.75): SS ⊂ corridor → α target ⊂ corridor → the car
                        # cuts apex only to the corridor edge (no wall overshoot).
                        _clh = max(0.10, float(self.get_parameter('mpc_corridor_half_width').value) - 0.15)
                        _cl = np.asarray(tr.center_lane)
                        _ang = getattr(tr, 'center_point_angles', None)
                        for _i in range(len(xy)):
                            _j = int(np.argmin(np.linalg.norm(_cl[:, :2] - xy[_i, :2], axis=1)))
                            _ps = float(_ang[_j]) if (_ang is not None and _j < len(_ang)) else 0.0
                            _lat = math.sin(_ps) * (xy[_i, 0] - _cl[_j, 0]) - math.cos(_ps) * (xy[_i, 1] - _cl[_j, 1])
                            if abs(_lat) > _clh:
                                _ex = _lat - max(-_clh, min(_clh, _lat))
                                xy[_i, 0] -= _ex * math.sin(_ps)
                                xy[_i, 1] -= _ex * (-math.cos(_ps))
                        self.get_logger().info(
                            f"[LMPC] IQP apex seed: {len(w)} pts, v∈"
                            f"[{v_arr.min():.1f},{v_arr.max():.1f}] grip-clamped @ a_lat={a_lat:.1f}")
                if xy is None:
                    cl = tr.raw_center_lane if tr.raw_center_lane is not None else tr.center_lane
                    xy = np.asarray(cl, dtype=float)
                    s_arr = np.asarray(tr.element_arc_lengths_orig, dtype=float)[:len(xy)]
                    if tr.center_point_angles is not None and len(tr.center_point_angles) >= len(xy):
                        psi = np.asarray(tr.center_point_angles, dtype=float)[:len(xy)]
                    else:
                        dx = np.diff(xy[:, 0], append=xy[0, 0]); dy = np.diff(xy[:, 1], append=xy[0, 1])
                        psi = np.arctan2(dy, dx)
                    try:
                        v_arr = np.array([float(tr.lut_ref_v(s)) for s in s_arr])
                    except Exception:
                        v_arr = np.full(len(xy), v_max_now * 0.8)
                    self.get_logger().info("[LMPC] centerline seed (no IQP raceline json)")
                ok = self._lmpc_db.seed_from_raceline(v_max_now, xy, psi, v_arr, s_arr)
                if ok:
                    self.get_logger().info(
                        f"[LMPC] raceline-seeded synthetic lap @ v={v_max_now:.1f}: "
                        f"{self._lmpc_db.summary()}")
                else:
                    self.get_logger().warn("[LMPC] raceline seed rejected (filter)")
            except Exception as e:
                import traceback
                self.get_logger().warn(f"[LMPC] raceline seed failed: {e}\n{traceback.format_exc()}")

    def _lmpc_update_per_cycle(self, x0: np.ndarray, s_now: float) -> None:
        """Every control cycle: (a) accumulate state into lap buffer, (b) SS query → set mpc attrs."""
        # Cleared each cycle; set only when a state is buffered for this cycle. The
        # post-solve lockstep append consumes it, so a raise / early-return / failed
        # solve leaves no stale state and never appends a state without its input.
        self._lmpc_pending_state = None
        if not self._lmpc_use:
            # LMPC OFF — ensure weights are 0 (in case toggled at runtime)
            self.mpc.lmpc_w_live = 0.0
            self.mpc._e_corr = np.zeros(3)
            return

        # Accumulate per-lap stuck time so the lmpc_max_stuck_seconds filter is
        # actually fed (previously unwired → meta['stuck_seconds'] stayed 0 →
        # filter inert). One dT per cycle the car is in stuck-recovery.
        if getattr(self.mpc, '_stuck_release_active', False):
            self._lmpc_lap_buf['stuck_seconds_accum'] += float(self.get_parameter('dT').value)

        # (a) Accumulate this cycle's 8-state x0 (extended) + estimate e_c.
        # x0 from _current_state_4 is 7-dim [x,y,ψ,vx,vy,r,s] (unified layout,
        # both modes) — we need 8-state for LapDatabase (… + δ_prev).
        state8 = np.zeros(8)
        state8[0:7] = x0[0:7]
        state8[7] = float(getattr(self.mpc, '_v_cmd_for_stuck', 0.0))   # δ_prev proxy
        # Defer the state/time append to the post-solve lockstep block (see the
        # input append in the control loop). Appending here pre-solve meant a
        # solve exception left a state with no paired input → every later
        # (state,input) pair in the lap was misaligned by one.
        self._b4_state8 = state8.copy()   # B4' pred-error gate: paired with u_seq[0] after solve
        self._lmpc_pending_state = state8.copy()
        self._lmpc_pending_time = time.monotonic()
        # max |e_c| running max (e_c estimated via CasADi center LUT)
        try:
            cx = float(self._track.center_lut_x(s_now))
            cy = float(self._track.center_lut_y(s_now))
            ec_est = float(np.hypot(state8[0] - cx, state8[1] - cy))
            self._lmpc_lap_buf['max_abs_ec'] = max(self._lmpc_lap_buf['max_abs_ec'], ec_est)
        except Exception:
            pass

        # (b) SS query → set mpc attrs (these are picked up by p_arr builder
        # at next set call in mpc.solve()).
        v_bucket = float(getattr(self.mpc, 'v_max', 5.0))
        # Activate only after enough real laps in this v bucket
        n_real = self._lmpc_db.n_real_laps(v_bucket)
        if n_real < self._lmpc_enable_after:
            self.mpc.lmpc_w_live = 0.0
            self.mpc._e_corr = np.zeros(3)
            return

        # Query SS at a TRACK-ANCHORED point one horizon ahead of the CURRENT
        # car position — NOT at the previous solve's predicted x_N. The terminal
        # cost attracts x_N toward ss_states[:,0]; we still want that attractor
        # ~one horizon ahead (forward pull), but anchoring it at x_N made the
        # query depend on x_N, which the cost was simultaneously moving — an
        # anchor-less positive-feedback loop: a lateral drift in x_N moved the
        # query → moved the attractor onto the drift → pulled x_N further out, so
        # ec grew monotonically (0.22→0.47 m) until wedge (use_lmpc off, 3691fdb).
        # Pinning the query to track arc length (current s + v·N·dT look-ahead)
        # breaks the loop while keeping the carrot ahead. z_t is built from track
        # geometry at q_s (centerline px,py,ψ + ref_v) so the weighted-L2 SS
        # match is to where the car SHOULD be, not where x_N drifted to.
        try:
            track_L = float(self._track.element_arc_lengths_orig[-1]) if (
                self._track.element_arc_lengths_orig is not None
                and len(self._track.element_arc_lengths_orig) > 0
            ) else 80.0
            v_now = float(state8[3])
            n_h = int(getattr(self.mpc, 'N', None) or self.get_parameter('N_horizon').value)
            dT = float(self.get_parameter('dT').value)
            q_s = lmpc_anchor_s(s_now, v_now, n_h, dT, track_L)
            psi_a = float(np.arctan2(self._track.center_lut_dy(q_s),
                                     self._track.center_lut_dx(q_s)))
            v_a = float(self.mpc.ref_v(q_s)) if self.mpc.ref_v is not None else v_now
            q_state = np.array([
                float(self._track.center_lut_x(q_s)),
                float(self._track.center_lut_y(q_s)),
                psi_a, v_a, 0.0, 0.0, q_s, 0.0,
            ], dtype=float)
            res = self._lmpc_ss.query(q_state, v_bucket, s_curr=q_s, track_length=track_L)
        except Exception as e:
            self.get_logger().warn(f"[LMPC] query failed: {e}", throttle_duration_sec=2.0)
            self.mpc.lmpc_w_live = 0.0
            self.mpc._e_corr = np.zeros(3)
            return

        if not res.is_ready:
            self.mpc.lmpc_w_live = 0.0
            self.mpc._e_corr = np.zeros(3)
            return

        # Pack SS into (4, K) state matrix (px, py, ψ, vx) + (K,) Q
        K = 10  # acados-side hardcoded
        ss_states = np.zeros((4, K))
        ss_Q = np.full(K, 1e6)  # padding = 1e6 → exp(-β·1e6)≈0 자연 무시
        for i in range(min(res.K, K)):
            ss_states[0, i] = res.states[i, 0]
            ss_states[1, i] = res.states[i, 1]
            ss_states[2, i] = res.states[i, 2]
            ss_states[3, i] = res.states[i, 3]
            ss_Q[i] = res.cost_to_go[i]
        self.mpc._lmpc_ss_states = ss_states
        self.mpc._lmpc_ss_Q = ss_Q
        # B4'.3: local error regression over the SAME SS neighbours just queried.
        if getattr(self.mpc, '_err_regr', False) and res.residuals.shape[0] > 0:
            from .mpc_core.lmpc.error_regression import epanechnikov_e_corr
            _dt = float(self.get_parameter('dT').value)
            # e_corr enters f_expl (a RATE, ẋ). The stored residual is a velocity
            # delta (actual_next − nominal_next). acados integrates x+dt·f_expl, so
            # to make dt·e_corr cancel the residual we inject residual/dt (a rate).
            # R1: weight by VELOCITY-space distance (current vx,vy,r ↔ neighbour
            # vx,vy,r = res.states[:,3:6]), NOT the position-heavy SS distance —
            # so the Epanechnikov kernel picks velocity-local neighbours (the
            # regime e_corr actually corrects). Reuses the same K neighbours;
            # leaves the LMPC position-near terminal query untouched. h scales to
            # the velocity neighbourhood (kernel spans the K returned points).
            _qv = np.asarray(state8, float)[3:6]
            _nv = np.asarray(res.states, float)[:, 3:6]
            _vel_d = np.linalg.norm(_nv - _qv[None, :], axis=1)
            _h = float(self.get_parameter('err_regr_bandwidth').value) * max(float(_vel_d.max()), 1e-6)
            _newc = epanechnikov_e_corr(res.residuals, _vel_d, h=_h) / max(_dt, 1e-6)
            # R2: temporal EMA low-pass — SS neighbours change cycle-to-cycle, so
            # raw e_corr jitters → the injected model offset (hence the predicted
            # trajectory) shakes. Smooth it.
            _beta = float(self.get_parameter('err_regr_ema').value)
            _prev = np.asarray(getattr(self.mpc, '_e_corr', np.zeros(3)), float)
            self.mpc._e_corr = _beta * _prev + (1.0 - _beta) * _newc
        else:
            self.mpc._e_corr = np.zeros(3)
        # Post-crash robustness: while STUCK / in stuck-recovery the state x0 is bad
        # (against a wall, vx≈0) → SQP_RTI prediction diverges and the LMPC terminal
        # fights the recovery. Gate LMPC OFF during recovery so pure-MPCC (the proven
        # stuck-release) pulls the car out; re-enable once moving again.
        _vx_low = (x0 is not None and len(x0) > 3 and float(x0[3]) < 0.3)
        _stuck = bool(getattr(self.mpc, '_stuck_release_active', False)) or _vx_low
        self.mpc.lmpc_w_live     = 0.0 if _stuck else self._lmpc_w_target
        self.mpc.lmpc_alpha_live = self._lmpc_alpha
        self.mpc.lmpc_beta_live  = self._lmpc_beta
        self.mpc.lmpc_reg_w_live = self._lmpc_reg_w

    def _b4_pred_error_log(self, state8_now, u_now, dt):
        """B4' correctness gate: compare the PREVIOUS cycle's nominal vs
        corrected one-step velocity prediction against the realized current
        state. 'Working' iff mean corrected-error < mean nominal-error over a
        window. e_corr is paired with the control it was active for."""
        prev = getattr(self, '_b4_prev', None)
        if prev is not None:
            ps, pu, pe, pdt = prev
            pred = predict_next(ps, pu, pdt)
            now_v = np.asarray(state8_now, float)[3:6]
            nominal_err = float(np.linalg.norm(now_v - pred[3:6]))
            corrected_err = float(np.linalg.norm(now_v - (pred[3:6] + pdt * np.asarray(pe))))
            self._b4_nom_acc = getattr(self, '_b4_nom_acc', 0.0) + nominal_err
            self._b4_cor_acc = getattr(self, '_b4_cor_acc', 0.0) + corrected_err
            self._b4_cnt = getattr(self, '_b4_cnt', 0) + 1
            if self._b4_cnt % 100 == 0:
                self.get_logger().info(
                    f"[B4'-pred] mean|err| nominal={self._b4_nom_acc / self._b4_cnt:.4f} "
                    f"corrected={self._b4_cor_acc / self._b4_cnt:.4f} (n={self._b4_cnt})")
        self._b4_prev = (np.asarray(state8_now, float).copy(),
                         np.asarray(u_now, float).copy(),
                         np.asarray(getattr(self.mpc, '_e_corr', np.zeros(3)), float).copy(),
                         float(dt))

    def _lmpc_on_lap_end(self, lap_idx: int) -> None:
        """Called from _on_lap_count_step when lap counter increments —
        finalize current lap buffer, push to LapDatabase if filters pass."""
        buf = self._lmpc_lap_buf
        if not buf['state'] or buf['last_lap'] < 0:
            # First lap_count event ever — reset buffer for next lap
            buf['state'] = []; buf['input'] = []; buf['time'] = []
            buf['lap_start_t'] = time.monotonic()
            buf['max_abs_ec'] = 0.0
            buf['stuck_seconds_accum'] = 0.0   # reset per-lap (wired: incremented in _lmpc_update_per_cycle while stuck)
            self._lmpc_lap_teleports = 0       # start counting teleports for lap 1
            buf['last_lap'] = lap_idx
            return

        # Build arrays
        states = np.array(buf['state'])
        T = states.shape[0]
        # B4'.2: real applied-control log (3-wide [a_x,delta,p_v]). Truncate to the
        # common prefix so state[t]→state[t+1] always has its control input[t].
        inlog = np.array(buf['input'], dtype=float) if buf['input'] else np.zeros((0, 3))
        if inlog.ndim != 2 or inlog.shape[1] != 3:
            inlog = inlog.reshape(-1, 3) if inlog.size else np.zeros((0, 3))
        n = min(states.shape[0], inlog.shape[0])
        if n >= 2:
            states = states[:n]
            inputs = inlog[:n - 1]          # input[t] pairs state[t]->state[t+1]
            T = states.shape[0]
        else:
            inputs = np.zeros((max(T - 1, 1), 3))   # not enough logged input → safe fallback
        t_arr = np.array(buf['time']) - buf['lap_start_t'] if buf['lap_start_t'] else np.linspace(0, T * 0.04, T)
        t_arr = t_arr[:T]   # keep length == T after possible state truncation
        lap_time = float(t_arr[-1] - t_arr[0]) if T > 1 else 0.0
        # n_resets in this lap = teleport(safe-reset) count THIS lap, counted in
        # _publish_safe_reset. With lmpc_max_resets=0 any teleport rejects the lap.
        n_resets = int(self._lmpc_lap_teleports)

        v_bucket = float(getattr(self.mpc, 'v_max', 5.0))
        meta = {
            'lap_idx': lap_idx,
            'max_abs_ec': float(buf['max_abs_ec']),
            'stuck_seconds': float(buf.get('stuck_seconds_accum', 0.0)),
        }
        ok = self._lmpc_db.add_lap(v_bucket, states, inputs, t_arr,
                                    lap_time, n_resets=n_resets,
                                    dt=float(self.get_parameter('dT').value),
                                    metadata=meta)
        reject_reason = getattr(self._lmpc_db, 'last_reject_reason', '') if not ok else ''
        self.get_logger().info(
            f"[LMPC] lap {lap_idx} buffered: v_bucket={v_bucket:.1f} T={T} "
            f"lap_time={lap_time:.2f}s n_resets={n_resets} accepted={ok}"
            + (f" reject_reason='{reject_reason}'" if reject_reason else "")
            + f" max_abs_ec={meta['max_abs_ec']:.2f}m\n  db: {self._lmpc_db.summary()}")

        # Reset buffer for next lap
        buf['state'] = []; buf['input'] = []; buf['time'] = []
        buf['lap_start_t'] = time.monotonic()
        buf['max_abs_ec'] = 0.0
        buf['stuck_seconds_accum'] = 0.0   # reset per-lap (wired — see _lmpc_update_per_cycle)
        self._lmpc_lap_teleports = 0       # start counting teleports for the next lap
        buf['last_lap'] = lap_idx

    def _lmpc_save_on_shutdown(self) -> None:
        if not self._lmpc_save_path:
            return
        try:
            from os.path import expanduser
            self._lmpc_db.save_all(expanduser(self._lmpc_save_path))
            self.get_logger().info(f"[LMPC] saved SS to {self._lmpc_save_path}")
        except Exception as e:
            self.get_logger().warn(f"[LMPC] save failed: {e}")

    def _on_lap_count_step(self, msg) -> None:
        """매 `auto_step_laps` lap 마다 effective max_speed 증가."""
        lap = int(msg.data)
        # LMPC: lap end detection + buffer flush
        try:
            self._lmpc_on_lap_end(lap)
        except Exception as e:
            self.get_logger().warn(f"[LMPC] lap_end failed: {e}")
        st = self._auto_step_state
        # auto_step v_max ramp runs ONLY when explicitly enabled. The
        # /mpc/lap_count subscription may now exist solely for LMPC lap-end
        # buffering above; without this guard the v_max ramp would fire even
        # with auto_step disabled (observed: v ramped 5→6 during a fixed-v=5
        # LMPC test, splitting laps across buckets).
        if not st['enabled']:
            return
        if lap == st['last_lap']:
            return
        # lap 0→1 첫 진입은 무시 (시작 상태 유지)
        if st['last_lap'] >= 0 and lap > 0 and (lap - st['lap_at_step']) >= st['laps']:
            st['current_step'] += 1
            new_v = st['start'] + st['current_step'] * st['size']
            if new_v > st['end'] + 1e-6:
                self.get_logger().info(
                    f"[auto_step] reached end ({st['end']:.1f}). holding.")
                st['last_lap'] = lap
                return
            st['lap_at_step'] = lap
            # 2026-05-28 #20: v step 시 LMPC SS warm_transfer
            # 이전 v bucket 의 best lap → 새 v bucket 의 seed.
            # 새 bucket 의 첫 real lap 도착 전까지 LMPC 활성 (synthetic 보다 real warm-transfer 가 좋음).
            old_v = st['start'] + (st['current_step'] - 1) * st['size']
            try:
                ok = self._lmpc_db.warm_transfer(old_v, new_v)
                if ok:
                    self.get_logger().info(
                        f"[LMPC] warm_transfer v={old_v:.1f}→v={new_v:.1f} (best from old bucket)"
                    )
            except Exception as e:
                self.get_logger().warn(f"[LMPC] warm_transfer failed: {e}")
            self._set_effective_max_speed(new_v)
        st['last_lap'] = lap

    def _auto_tune_from_max_speed(self, v_override: float | None = None) -> None:
        """v_max 한 변수만 사용자 설정. 나머지 cost weights / corridor / EMA
        를 heuristic 매핑으로 자동 결정. `on_set_parameters_callback` 가 자동
        호출되어 live attrs (mpc.q_*_live 등) 도 같이 갱신됨.

        매핑 식 (kinematic + raceline 가정, 안전 위주):
            max_speed_p          = v_max * 0.95
            q_lag_live           = max(50, 12 * v_max)
            q_cte_live           = max(2, 8 - v_max * 0.4)
            q_d_delta_live       = 25 (ROS1 검증값 고정 — 동적 조정은 MLP q_drate_scale 이 담당)
            alpha_steer_live     = max(0.4, 0.7 - v_max * 0.02)  # 빠를수록 smooth
            mpc_corridor_half_width = yaml 값 그대로 (auto override 제거 — 사용자 직접 결정)
            D_apex_live          = 0 (raceline 자체가 apex 통과)
            cost_spike_thr_live  = max(100, 250 - 5 * v_max)
        """
        from rclpy.parameter import Parameter as P
        v = float(v_override) if v_override is not None else float(self.get_parameter('max_speed').value)
        track_src = str(self.get_parameter('track_source').value).strip().lower()
        # Agent R3 fix: D_apex=0.35 centerline 기본값이 random 트랙의 uniform 3m
        # corridor 에서 effective margin (0.95 − 0.15 − 0.35 = 0.45) 너무 빠듯 →
        # cost slack 폭증 → solver 가 감속 회피 → stuck. 0 으로 고정 (BO 가 13번째
        # 차원으로 학습한 D_apex 값이 우선).
        d_apex = 0.0
        # 주의: mpc_corridor_half_width 는 auto 매핑에서 제외 (yaml 값 보존).
        #       이전 auto 식 max(0.3, 0.8 - 0.03*v) 가 yaml 의 1.0 등 큰 값을 0.68 로
        #       덮어써서 좌우 반경이 항상 작게 강제되는 회귀가 있었음. 사용자가
        #       yaml 에서 직접 1.0~1.5 등으로 키우면 그대로 적용되어야 함.
        auto = {
            'max_speed_p':              v * 0.95,
            'q_lag_live':               max(50.0, 12.0 * v),
            'q_cte_live':               max(2.0, 8.0 - v * 0.4),
            'q_d_delta_live':           25.0,
            'alpha_steer_live':         max(0.4, 0.7 - v * 0.02),
            'D_apex_live':              d_apex,
            # Agent R-round2 Fix 2: cost_spike_thr_live 매핑 제거.
            # 옛 식 `max(100, 250 - 5·v_max)` = 230 @ v=4 → 정상 cost ~330 도
            # 매 cycle spurious fallback → MPC 의 정상 plan 거의 ignored.
            # yaml 의 600 (N=120 시대 검증) 또는 N=80 면 yaml 에서 ~400 으로
            # 직접 set. v_max 와 무관 (weight scale / N_horizon 에 의존).
        }
        params = [P(k, P.Type.DOUBLE, float(val)) for k, val in auto.items()]
        self.set_parameters(params)
        self.get_logger().info(
            f"[auto_tune] v_max={v:.1f} → " +
            ", ".join(f"{k.replace('_live', '')}={val:.2f}" for k, val in auto.items()))

    def _initialize_mpc(self, wpnts=None) -> None:
        if bool(self.get_parameter('auto_tune').value):
            self._auto_tune_from_max_speed()
        vel_scale = float(self.get_parameter('vel_scale').value)
        inflation = float(self.get_parameter('inflation_factor').value)
        extend_part = int(self.get_parameter('extend_part').value)

        if wpnts is not None:
            corridor_half = float(self.get_parameter('mpc_corridor_half_width').value)
            mode_desc = (f"fixed corridor ±{corridor_half:.2f}m"
                         if corridor_half > 1e-3
                         else f"raw d_left/d_right + inflation {inflation}")
            self.get_logger().info(
                f"building track from /centerline_waypoints ({len(wpnts)} wpnts, "
                f"vel_scale={vel_scale}, {mode_desc})")
            self._track = build_track_from_wpnts(
                wpnts, vel_scale=vel_scale,
                inflation_factor=inflation, extend_part=extend_part,
                default_v=float(self.get_parameter('max_speed').value),
                corridor_half_width=corridor_half,
                a_lat_max=self._a_lat_eff_for_track(),
                a_long_max=abs(A_MIN_DYN),
                corridor_v_floor=float(self.get_parameter('corridor_v_floor').value),
                corridor_v_tight=float(self.get_parameter('corridor_v_tight_width').value),
                corridor_v_wide=float(self.get_parameter('corridor_v_wide_width').value))
            try:
                import numpy as _np
                _rv = _np.asarray(self._track.ref_v)
                _vx = [getattr(w, 'vx_mps', 0.0) for w in wpnts]
                self.get_logger().warn(
                    f"[DBG-refv] track ref_v min={_rv.min():.2f} max={_rv.max():.2f} mean={_rv.mean():.2f} n={len(_rv)} "
                    f"| default_v={float(self.get_parameter('max_speed').value):.1f} a_lat={self._a_lat_eff_for_track():.2f} "
                    f"corridor={corridor_half:.2f} | wpnt vx max={max(_vx):.2f} mean={sum(_vx)/max(1,len(_vx)):.2f}")
            except Exception as _e:
                self.get_logger().warn(f"[DBG-refv] fail: {_e}")
        else:
            track_dir = self._resolve_track_dir()
            track_name = str(self.get_parameter('track_name').value)
            self.get_logger().info(
                f"loading track '{track_name}' from {track_dir} "
                f"(vel_scale={vel_scale}, inflation={inflation})")
            self._track = load_track(track_dir, track_name,
                                     vel_scale=vel_scale,
                                     inflation_factor=inflation,
                                     extend_part=extend_part)
        L_orig = float(self._track.element_arc_lengths_orig[-1])
        L_ext = float(self._track.element_arc_lengths[-1])
        self.get_logger().info(
            f"track loaded: N_orig={len(self._track.element_arc_lengths_orig)}, "
            f"L_orig={L_orig:.2f} m, L_ext={L_ext:.2f} m")

        bo = self._resolve_bo_params()
        param = self._build_param_dict(bo)
        param['s_max'] = L_ext  # match ROS1 (extended)
        is_ot = bool(bo.get('overtaking', False))

        # Vehicle dynamics dict — kinematic mode only consumes `l_wb`
        vheid = {
            'l_wb': float(self.get_parameter('vehicle.l_wb').value),
            'l_f':  float(self.get_parameter('vehicle.l_f').value),
            'l_r':  float(self.get_parameter('vehicle.l_r').value),
            'm':    float(self.get_parameter('vehicle.m').value),
        }

        self.mpc.set_initial_params(param, vheid, is_ot)
        # 그립 정직화 — set_track_data(κ_eq/bf)·setup_MPC(ellipse/codegen tag) 둘 다
        # dyn_mu 를 소비하므로 track 빌드 전에 push (push 순서 의존).
        self.mpc.dyn_mu = float(self.get_parameter('dyn_mu').value)
        self.mpc.ellipse_frac = float(self.get_parameter('ellipse_frac').value)
        # κ-lookahead window MUST be set before set_track_data — the brake-aware
        # ref_v cap (κ_eq, forward window) is built inside set_track_data.
        # BUG (2026-06-15): lookahead_m was only assigned ~40 lines below, AFTER
        # set_track_data, so config `lookahead_m` never reached the cap → it always
        # fell back to the v_max²/6 default. Set it here so the config takes effect.
        _lk0 = float(self.get_parameter('lookahead_m').value)
        self.mpc.lookahead_m = _lk0 if _lk0 > 0 else None
        self.mpc.set_track_data(
            self._track.center_lut_x, self._track.center_lut_y,
            self._track.center_lut_dx, self._track.center_lut_dy,
            self._track.right_lut_x, self._track.right_lut_y,
            self._track.left_lut_x,  self._track.left_lut_y,
            self._track.element_arc_lengths,
            float(self._track.element_arc_lengths_orig[-1]),
            self._track.lut_ref_v,
        )
        # ── Dynamic Pacejka 모드 토글 — setup_MPC() 전에 적용해야 codegen 반영 ──
        self.mpc.use_dynamic = bool(self.get_parameter('use_dynamic').value)
        self.mpc.nlp_solver_iters = int(self.get_parameter('nlp_solver_iters').value)
        self.mpc.dyn_tire_model = str(self.get_parameter('dyn_tire_model').value)
        # LM regularization (read before setup_MPC so codegen picks it up).
        self.mpc.lm_dynamic = float(self.get_parameter('lm_dynamic').value)
        self.mpc.dyn_use_pacejka = (self.mpc.dyn_tire_model == 'pacejka')
        # 2026-06-10 unified layout: kinematic mode builds the SAME 8-state
        # layout (slot 3 = vx in both modes) → LMPC is allowed for both.
        # effective_lmpc() stays the single policy point (re-gates if a future
        # model layout diverges).
        from .mpc_core.model_policy import effective_lmpc
        _req_lmpc = bool(self.get_parameter('use_lmpc').value)
        _eff_lmpc = effective_lmpc(self.mpc.use_dynamic, _req_lmpc)
        if _req_lmpc and not _eff_lmpc:
            self.get_logger().warn(
                "[LMPC] use_lmpc=true but effective_lmpc() gated it OFF "
                "(model layout incompatible with safe-set packing).")
        # Reconcile the runtime LMPC bookkeeping gate too: _lmpc_use was set from
        # the raw param in __init__ (before use_dynamic was known). Without this,
        # kinematic mode would still record laps / run the SS query (slot 3 = s
        # packed as vx) and poison the LapDatabase. _lmpc_use gates the per-cycle
        # update (→ empty buffer → lap-end no-ops) and the query-state stash.
        self._lmpc_use = _eff_lmpc
        # LMPC codegen flag (Step 4): when LMPC active, drop the terminal contour/yaw
        # W_e emphasis so the joint-α apex target — not centerline tracking — sets x_N.
        self.mpc._lmpc_codegen = _eff_lmpc
        # Phase B3: joint-α LMPC (α as state + convex-α terminal). Tie activation
        # to use_lmpc so the OCP is built with the augmented state (nx=8+K).
        self.mpc._lmpc_joint = _eff_lmpc
        # R3: decouple max_speed — set progress target / κ-lookahead independently
        # of the hard cap BEFORE setup_MPC (codegen-time). 0 → derive from v_max.
        _st = float(self.get_parameter('speed_target').value)
        _lk = float(self.get_parameter('lookahead_m').value)
        self.mpc.speed_target = _st if _st > 0 else None
        self.mpc.lookahead_m  = _lk if _lk > 0 else None
        # Phase D closed-form CasADi GP residual flag — set BEFORE setup_MPC()
        # so the dynamics codegen picks it up. Independent of use_gp_residual.
        self.mpc.use_gp_casadi = bool(self.get_parameter('use_gp_casadi').value)
        # B4' error regression. Coupled to use_lmpc (it reuses the SS-neighbour
        # query). With use_lmpc off -> _err_regr off -> pure baseline f_expl.
        _use_err = bool(self.get_parameter('use_error_regression').value)
        self.mpc._err_regr = _use_err and _eff_lmpc
        if _use_err and self.mpc.use_gp_casadi:
            # e_corr and GP residual BOTH add to f_expl velocity rows -> double
            # correction. They are alternatives (B4' supersedes the GP residual).
            self.get_logger().warn(
                "[B4'] use_error_regression AND use_gp_casadi both set -> "
                "disabling GP residual to avoid double-correcting f_expl.")
            self.mpc.use_gp_casadi = False
        # 이전에 set_track_data 가 print 한 "MODEL = KINEMATIC" 은 toggle 이전 상태라 잘못됨.
        # 명확하게 다시 announce.
        self.get_logger().info(
            f"[mpc] ACTUAL MODEL FOR CODEGEN = "
            f"{'DYNAMIC 8-state, tire=' + self.mpc.dyn_tire_model if self.mpc.use_dynamic else 'KINEMATIC-UNIFIED 8-state (f_kin)'}")

        self.get_logger().info("calling mpc.setup_MPC() — first run codegens acados (~30s)…")

        # — setup_MPC 직후 실제 codegen 된 model 확인 (use_dynamic 토글이 정말 먹혔는지)
        self.mpc.setup_MPC()
        # === 실제 codegen 된 model 검증 (use_dynamic 토글 안 먹은 경우 여기서 발각) ===
        actual_n_states = int(self.mpc.n_states)
        actual_n_controls = int(self.mpc.n_controls)
        actual_dyn = bool(self.mpc.use_dynamic)
        if actual_n_states == 8 and actual_dyn:
            kind = "DYNAMIC ✓ (8-state Pacejka)"
        elif actual_n_states == 8 and not actual_dyn:
            kind = "KINEMATIC ✓ (8-state unified f_kin)"
        else:
            kind = f"UNEXPECTED (n_states={actual_n_states}, use_dynamic={actual_dyn})"
        self.get_logger().info(
            f"=== MODEL POST-CODEGEN: {kind}, n_controls={actual_n_controls} ===")
        # 사용자가 토픽으로 확인할 수 있게 /mpc/model_info 도 latched 발행
        from std_msgs.msg import String as _String
        if not hasattr(self, '_model_info_pub'):
            self._model_info_pub = self.create_publisher(_String, '/mpc/model_info',
                self._latched_qos if hasattr(self, '_latched_qos') else 10)
        self._model_info_pub.publish(_String(data=f"{kind} | n_states={actual_n_states} | n_controls={actual_n_controls}"))

        # ── LMPC: raceline seed + npz load (2026-05-28 #18) ─────────────
        try:
            self._lmpc_load_or_seed()
        except Exception as _e:
            self.get_logger().warn(f"[LMPC] seed/load skipped: {_e}")

        # ── Phase D: GP residual learning wrap ──────────────────────────
        # use_gp_residual=true 면 setup_MPC() 결과 (acados ocp) 를 L4acados
        # 의 ResidualLearningMPC 로 wrap. 미설정 시 plain acados 사용.
        # 추가 codegen ~30s (첫 launch 만). gp_residual.pt 로드 실패 시
        # 자동 fallback (plain acados 그대로).
        if bool(self.get_parameter('use_gp_residual').value):
            from .mpc_core.gp_residual_wrapper import wrap_solver_with_gp
            ckpt = str(self.get_parameter('gp_ckpt_path').value).strip() \
                   or os.path.expanduser('~/bo_results/gp_residual.pt')
            ok = wrap_solver_with_gp(self.mpc, ckpt, logger=self.get_logger())
            if ok:
                self.get_logger().info("[Phase D] GP residual ACTIVE")
            else:
                self.get_logger().warn("[Phase D] GP wrap failed — plain acados fallback")

        self._mpc_ready = True
        # Startup ramp — MPC ready 직후 N 초 동안은 MPC output 무시하고 작은
        # forward push (0.5 m/s, steer=0) 강제. 정지에서 dynamic 모드의
        # Pacejka singularity (vx=0) 회피 + solver 가 plausible warm-start
        # 형성하는 동안 vehicle 이 안전하게 출발.
        self._startup_t = time.monotonic()
        self.get_logger().info(
            "MPC ready — control loop active (cold-start: vx-floor + speed-floor, "
            "MPCC steering from t=0)")
        # auto_step 활성 시, codegen 끝난 직후 effective v_max = start 로 강제.
        # (codegen ubu cap 은 yaml max_speed = end_speed 그대로 12 인 상태)
        if getattr(self, '_auto_step_state', {}).get('enabled', False):
            self._set_effective_max_speed(self._auto_step_state['start'])

        # Latched RViz viz: raw (un-inflated) lanes. /boundary_marker (cycle)
        # shows inflated corridor — comparing the two reveals how much
        # safety margin mpc applies vs the actual track walls.
        self._publish_track_viz()

    def _publish_track_viz(self) -> None:
        """One-shot publish of /center_path /right_path /left_path /reference_path
        as small-sphere MarkerArrays. Latched (TRANSIENT_LOCAL) so late RViz
        attaches still receive. Ports ROS1 `preprocess_track_data` viz publish."""
        if self._track is None:
            return
        cl = self._track.raw_center_lane
        rl = self._track.raw_right_lane
        ll = self._track.raw_left_lane
        if cl is None or rl is None or ll is None:
            self.get_logger().warn("track has no raw_*_lane — skipping viz publish")
            return
        #              (r,   g,   b)   ns
        self.center_path_pub.publish(self._np_to_markers(cl, (1.0, 1.0, 1.0), 'center'))
        self.right_path_pub.publish(self._np_to_markers(rl, (1.0, 0.2, 0.2), 'right'))
        self.left_path_pub.publish(self._np_to_markers(ll, (0.2, 1.0, 0.2), 'left'))
        self.reference_path_pub.publish(self._np_to_markers(cl, (1.0, 1.0, 0.2), 'reference'))
        self.get_logger().info(
            f"published latched /center_path /right_path /left_path /reference_path "
            f"({len(cl)} sphere markers each)")

    def _np_to_markers(self, pts, rgb, ns: str) -> MarkerArray:
        """Build a MarkerArray of small spheres (type=2, scale=0.05) — matches
        the visual style of `/centerline_waypoints/markers` so RViz looks
        consistent. One Marker per waypoint with id=i."""
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, p in enumerate(pts):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = ns
            m.id = int(i)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            m.color.a = 1.0
            ma.markers.append(m)
        return ma

    # ─────────────────────────────────────────────────────────────────
    # Live parameter sync (rclpy parameter callback → mpc.* attributes)
    # ─────────────────────────────────────────────────────────────────
    def _push_live_params_to_mpc(self) -> None:
        for name, _default in LIVE_PARAMS:
            val = float(self.get_parameter(name).value)
            setattr(self.mpc, name, val)

    def _on_param_change(self, params: list[Parameter]):
        from rcl_interfaces.msg import SetParametersResult
        live_names = {n for n, _ in LIVE_PARAMS}
        for p in params:
            if p.name in live_names:
                try:
                    setattr(self.mpc, p.name, float(p.value))
                except Exception as e:
                    return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    # ─────────────────────────────────────────────────────────────────
    # Sub callbacks — cache only; heavy work runs on the control timer
    # ─────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        # Teleport detection — RViz /initialpose 로 차 이동 시 odom 위치가 점프.
        # 이전 위치와 1m 이상 차이 나면 MPC warm-start reset (옛 traj 의 풀-스티어
        # spin 방지). 정상 주행 dx 는 보통 v·dt = 5·0.025 = 0.125m 라 1m threshold
        # 충분히 보수적.
        if self._last_odom is not None and getattr(self, 'mpc', None) is not None:
            p1 = msg.pose.pose.position
            p0 = self._last_odom.pose.pose.position
            jump = ((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5
            if jump > 1.0:
                self._reset_mpc_latches(f'teleport 감지 ({jump:.2f}m jump)')
        self._last_odom = msg

    def _reset_mpc_latches(self, reason: str) -> None:
        """Teleport·수동(조이)→자동 복귀 등 '연속 주행 가정이 깨진' 시점의 공통 리셋.

        warm-start(옛 traj 풀-스티어 spin 방지) + s-tracking(_last_sensor_s 점프로
        lap_count 꼬임 방지) + STUCK latch(수동 정차를 STUCK 으로 오감지한 잔여
        release/카운터) + startup ramp 재가동.
        """
        if getattr(self, 'mpc', None) is None:
            return
        self.get_logger().warn(f"[mpc] {reason} — warm-start/latch reset")
        self.mpc.WARM_START = False
        # _pos_for_v 는 (x,y,t) 3-tuple — solve() 가 unpack → None 로 reset.
        # _last_sensor_s 도 None reset 으로 lap_count 누적 안 해야 함.
        NONE_RESET = {'_last_sensor_s', '_pos_for_v', '_last_sensor_yaw'}
        for attr in ('_last_sensor_s', '_pos_for_v', '_last_sensor_yaw',
                     '_v_cmd_for_stuck', '_stuck_count', '_last_delta_applied',
                     '_stuck_release_remaining', '_stuck_release_total'):
            if hasattr(self.mpc, attr):
                setattr(self.mpc, attr, None if attr in NONE_RESET else 0.0)
        if hasattr(self.mpc, '_has_moved'):
            self.mpc._has_moved = False
        # startup ramp 다시 활성화 (정지에서 출발 안정화)
        self._startup_t = time.monotonic()

    def _mux_active_cb(self, msg: Bool) -> None:
        # simple_mux 가 매 사이클 publish 하는 제어권 신호. 수동(조이) 구간이
        # 1초 이상 지속된 뒤 MPCC 로 복귀하면 teleport 와 동일한 latch 리셋.
        now = time.monotonic()
        active = bool(msg.data)
        if self._mux_mpcc_active is True and not active:
            self._mux_inactive_since = now
        elif active and self._mux_mpcc_active is False:
            dur = now - (self._mux_inactive_since or now)
            if dur > 1.0:
                self._reset_mpc_latches(f'mux 제어권 복귀 (수동 {dur:.1f}s)')
            self._mux_inactive_since = None
        self._mux_mpcc_active = active

    def _pose_cb(self, msg: PoseStamped):
        self._last_pose = msg

    def _goal_cb(self, msg: PoseStamped):
        self._goal = msg

    def _clicked_point_cb(self, msg: PointStamped):
        # Mirrors ROS1: RViz click adds a static obstacle for dev
        self._obstacles.append((msg.point.x, msg.point.y))
        self._obstacles_stamp = time.monotonic()
        self.get_logger().info(
            f"clicked obstacle ({msg.point.x:.2f}, {msg.point.y:.2f}); "
            f"now {len(self._obstacles)} obstacles")

    def _external_obs_cb(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y) for p in msg.poses]
        self._obstacles_stamp = time.monotonic()

    # ─────────────────────────────────────────────────────────────────
    # Boundary visualization hook — called by mpc_core every solve()
    # ─────────────────────────────────────────────────────────────────
    def _on_boundary_points(self, shifted_points: list[tuple[float, float]]) -> None:
        ma = MarkerArray()
        for i, (xx, yy) in enumerate(shifted_points):
            mk = Marker()
            mk.header.frame_id = 'map'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.scale = Vector3(x=0.10, y=0.10, z=0.10)
            mk.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
            mk.pose.position = Point(x=float(xx), y=float(yy), z=0.0)
            mk.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ma.markers.append(mk)
        self.boundary_pub.publish(ma)

    # ─────────────────────────────────────────────────────────────────
    # Control loop @ 40 Hz
    # ─────────────────────────────────────────────────────────────────
    def _current_state_4(self) -> np.ndarray | None:
        """Assemble state for MPC solver. 2026-06-10 unified 8-state layout:
        ALWAYS [x, y, psi, vx, vy, r, s] (7-dim, solver appends δ_prev → 8),
        in both kinematic and dynamic modes. vx/vy/r fall back to 0 when odom
        is missing. Returns None if not ready."""
        if self._last_pose is None and self._last_odom is None:
            return None
        if self._last_pose is not None:
            p = self._last_pose.pose.position
            q = self._last_pose.pose.orientation
        else:
            p = self._last_odom.pose.pose.position
            q = self._last_odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        psi = math.atan2(siny_cosp, cosy_cosp)
        s = 0.0
        if self._track is not None:
            s, _ = find_current_arc_length(self._track, np.array([p.x, p.y]))

        # unified 8-state: vx/vy/r 도 필요 (양 모드). odom 의 twist 에서
        # 추출 (body frame). odom 없으면 0 폴백 (startup 직후 한정).
        vx = vy = r = 0.0
        if self._last_odom is not None:
            tw = self._last_odom.twist.twist
            vx = float(tw.linear.x)
            vy = float(tw.linear.y)
            r  = float(tw.angular.z)
        return np.array([p.x, p.y, psi, vx, vy, r, s], dtype=float)

    def _load_ml_scaler(self) -> None:
        """C: TorchScript WeightScaleMLP 로드. mpc_node init 시 1회 호출.
        실패 시 self._ml_scaler = None — B heuristic 폴백."""
        self._ml_scaler = None
        if not bool(self.get_parameter('use_ml_scale').value):
            return
        try:
            from .ml.inference import WeightScaleInference, default_model_path
            path = self.get_parameter('ml_model_path').value or default_model_path()
            self._ml_scaler = WeightScaleInference(path)
            self.get_logger().info(f"[ml] WeightScaleMLP loaded from {path}")
        except Exception as e:
            self.get_logger().warn(
                f"[ml] WeightScaleMLP load failed: {e} — falling back to B heuristic")

    def _load_polynomial(self) -> None:
        """BO 결과로 학습된 polynomial coefficients 로드 (override_mode='polynomial').
        실패 시 self._poly = None — bucketed/B 폴백."""
        self._poly = None
        path = str(self.get_parameter('poly_path').value).strip()
        if not path:
            return
        try:
            import json, os
            if not os.path.isfile(path):
                self.get_logger().warn(f"[poly] file not found: {path} — fallback")
                return
            with open(path) as f:
                d = json.load(f)
            self._poly = {
                'polys': {k: list(map(float, v)) for k, v in d['polys'].items()},
                'bounds': tuple(d.get('scale_bounds', (0.3, 5.0))),
                'speed_range': d.get('speed_range', [0, 100]),
            }
            self.get_logger().info(
                f"[poly] loaded from {path}: degree {d.get('degree', '?')}, "
                f"speed_range {d.get('speed_range', '?')}, "
                f"{len(self._poly['polys'])} polynomials")
        except Exception as e:
            self.get_logger().warn(f"[poly] load failed: {e} — fallback")
            self._poly = None

    # ── Corner-exit detection (Method C: spatial-aware q_cte) ──
    # 코너에서 직선으로 전환되는 시점 (s_exit) 기록. 그 후 EXIT_BLEND_M 동안
    # q_cte 점진적 회복 (0 → 1). 코너 직후 centerline 강제 복귀 → 휘청 방지.
    EXIT_BLEND_M = 5.0           # 출구 후 5m 동안 점진 회복
    EXIT_KAPPA_HIGH = 0.3        # 코너 in/out 경계
    EXIT_KAPPA_LOW  = 0.15

    def _corner_exit_factor(self, s_now: float, kappa_abs_now: float) -> float:
        """코너 출구 후 q_cte 감소 factor 계산.
        반환: 0~1 (코너 출구 직후 0, EXIT_BLEND_M 후 1).
        코너 안 또는 멀리 직선이면 1.0 (변경 X).
        """
        # 이전 cycle 의 kappa hi/lo 트래킹
        prev_kappa = getattr(self, '_prev_kappa_abs', 0.0)
        self._prev_kappa_abs = kappa_abs_now
        # 코너→직선 전환 감지
        if prev_kappa > self.EXIT_KAPPA_HIGH and kappa_abs_now < self.EXIT_KAPPA_LOW:
            self._last_exit_s = s_now   # 출구 시점 기록
        # 출구 이후 거리
        last_exit_s = getattr(self, '_last_exit_s', None)
        if last_exit_s is None:
            return 1.0
        d = s_now - last_exit_s
        if d < 0 or d > self.EXIT_BLEND_M:
            return 1.0
        # 0 → 1 점진 회복
        return float(d / self.EXIT_BLEND_M)

    def _adaptive_weight_update(self, s_now: float) -> None:
        """B: κ-aware online weight adaptation. 매 cycle 호출.

        mpc_core 의 q_*_scale_live 가 cost residual multiplier (default 1.0).
        forward-max κ_lookahead (6m 앞) 에 따라 동적 매핑:
          - 직선 (κ ≈ 0): scale = 1.0 (yaml 기본값 유지)
          - 코너 (κ ↑): cte/lag 약화, v 추종 강화, rate cost 강화
                       → mpc 가 자동으로 코너 진입 감속 + smooth steering

        매핑 (heuristic, kinematic + raceline 가정):
          q_cte_scale   = max(0.3, 1 − 2·κ)   # 코너 path 자유도 ↑
          q_lag_scale   = max(0.5, 1 − 1.5·κ) # 코너 progress 압박 ↓
          q_v_scale     = 1 + 2·κ              # 코너 ref_v(=raceline IQP) 추종 ↑ → 감속
          q_drate_scale = 1 + 3·κ              # 코너 steering rate cost ↑ → smooth
        """
        if self._track is None or not getattr(self, '_mpc_ready', False):
            return
        # BO sweep / 수동 override 분기 — auto_tune 보다 우선. 외부 평가용.
        # 결정: override_mode == 'bucketed' OR ('fixed' / legacy override_scales=true).
        mode = str(self.get_parameter('override_mode').value).lower()
        if mode in ('fixed',) or bool(self.get_parameter('override_scales').value):
            self.mpc.q_cte_scale_live   = float(self.get_parameter('override_q_cte_scale').value)
            self.mpc.q_lag_scale_live   = float(self.get_parameter('override_q_lag_scale').value)
            self.mpc.q_v_scale_live     = float(self.get_parameter('override_q_v_scale').value)
            self.mpc.q_drate_scale_live = float(self.get_parameter('override_q_drate_scale').value)
            return
        if mode == 'bucketed':
            # 현재 cycle 의 κ_abs 로 bucket 결정.
            try:
                L = float(self._track.element_arc_lengths_orig[-1])
                kappa_abs_now = float(self.mpc.abs_kappa_lut(s_now % L))
            except Exception:
                kappa_abs_now = 0.0
            b01 = float(self.get_parameter('bucket_kappa_b01').value)
            b12 = float(self.get_parameter('bucket_kappa_b12').value)
            b = 0 if kappa_abs_now < b01 else (1 if kappa_abs_now < b12 else 2)
            self.mpc.q_cte_scale_live   = float(self.get_parameter(f'override_q_cte_scale_b{b}').value)
            self.mpc.q_lag_scale_live   = float(self.get_parameter(f'override_q_lag_scale_b{b}').value)
            self.mpc.q_v_scale_live     = float(self.get_parameter(f'override_q_v_scale_b{b}').value)
            self.mpc.q_drate_scale_live = float(self.get_parameter(f'override_q_drate_scale_b{b}').value)
            # ── C: Corner-exit spatial blending (2026-05-26 비활성) ──
            # try/except 로 안전화 + 비활성 (BO startup fail 원인 진단 중).
            # 원인 파악 후 재활성. 현재 BO 가 spatial-aware 없는 Hybrid 그대로.
            # try:
            #     exit_factor = self._corner_exit_factor(s_now, kappa_abs_now)
            #     self.mpc.q_cte_scale_live *= exit_factor
            # except Exception:
            #     pass
            return
        if mode == 'polynomial' and getattr(self, '_poly', None) is not None:
            # BO-learned polynomial: (v, κ_bucket) → 4 scales. 매 cycle eval, ~수 μs.
            try:
                L = float(self._track.element_arc_lengths_orig[-1])
                kappa_abs_now = float(self.mpc.abs_kappa_lut(s_now % L))
            except Exception:
                kappa_abs_now = 0.0
            b01 = float(self.get_parameter('bucket_kappa_b01').value)
            b12 = float(self.get_parameter('bucket_kappa_b12').value)
            b = 0 if kappa_abs_now < b01 else (1 if kappa_abs_now < b12 else 2)
            v = float(getattr(self.mpc, 'v_max', 4.0))  # 현재 cost target (auto_step aware)
            lo, hi = self._poly['bounds']
            def _eval(key: str) -> float:
                import numpy as _np
                return float(_np.clip(_np.polyval(self._poly['polys'][key], v), lo, hi))
            self.mpc.q_cte_scale_live   = _eval(f'q_cte_b{b}')
            self.mpc.q_lag_scale_live   = _eval(f'q_lag_b{b}')
            self.mpc.q_v_scale_live     = _eval(f'q_v_b{b}')
            self.mpc.q_drate_scale_live = _eval(f'q_drate_b{b}')
            return
        if not bool(self.get_parameter('auto_tune').value):
            return  # auto_tune=false 면 yaml 의 yaml 기본 scale 그대로
        try:
            L = float(self._track.element_arc_lengths_orig[-1])
            kappa = float(self.mpc.abs_kappa_lut(s_now % L))
            kappa_signed = float(self.mpc.signed_kappa_lut(s_now % L))
        except Exception:
            return

        # C: NN inference 우선 (use_ml_scale=true 면)
        if getattr(self, '_ml_scaler', None) is not None:
            try:
                v_act = self._last_odom.twist.twist.linear.x if self._last_odom else 0.0
                ref_v = float(self.mpc.ref_v(s_now % L))
                v_max_cost = float(getattr(self.mpc, 'v_max', 0.0))
                qcte, qlag, qv, qdrate = self._ml_scaler(
                    abs(kappa), kappa_signed, v_act, ref_v, v_max_cost)
                self.mpc.q_cte_scale_live   = qcte
                self.mpc.q_lag_scale_live   = qlag
                self.mpc.q_v_scale_live     = qv
                self.mpc.q_drate_scale_live = qdrate
                return
            except Exception as e:
                self._mpc_log.warn_throttle(5.0, "[ml] inference failed: %s — fallback to B", str(e))

        # B fallback (heuristic) — train.py 의 target 공식과 동일하게 유지!
        # MLP 가 학습한 매핑과 일관성 — MLP 실패 시 폴백이 같은 동작.
        k = min(max(abs(kappa), 0.0), 1.0)  # cap
        self.mpc.q_cte_scale_live   = max(0.3, 1.0 - 2.0 * k)  # 코너 path 자유도 ↑
        self.mpc.q_lag_scale_live   = max(0.5, 1.0 - 1.5 * k)  # 코너 progress 압박 ↓
        self.mpc.q_v_scale_live     = 1.5 + 1.5 * k             # ref_v 추종 강화
        self.mpc.q_drate_scale_live = 1.5 + 4.0 * k             # 강화 (2.5→4.0). κ=0.82→4.78

    def _publish_safe_reset(self, x0) -> None:
        """Agent A: gym `in_collision` latch escape.

        Compute a safe respawn point 2m ahead of the current stuck position
        on the centerline (so the car appears in clear track, facing forward),
        publish `/initialpose` → gym_bridge.ego_reset_callback → env.reset()
        → latch cleared, dynamics integration resumes. Also clears MPC warm-
        start via the existing teleport detector.
        """
        if self._track is None:
            return
        try:
            x = float(x0[0]); y = float(x0[1])
            s_now, _ = find_current_arc_length(
                self._track, np.array([x, y]))
            L = float(self._track.element_arc_lengths_orig[-1])
            s_safe = (s_now + 2.0) % L     # 2m ahead on centerline
            rx = float(self._track.center_lut_x(s_safe))
            ry = float(self._track.center_lut_y(s_safe))
            dx = float(self._track.center_lut_dx(s_safe))
            dy = float(self._track.center_lut_dy(s_safe))
            yaw = math.atan2(dy, dx)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"[stuck-recover] safe pose calc failed: {e}")
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = rx
        msg.pose.pose.position.y = ry
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.initialpose_pub.publish(msg)
        # Single source of truth for LMPC lap rejection: every teleport, in any
        # control branch, lands here. Count it so the current lap is flagged as
        # corrupted and rejected from the safe set.
        self._lmpc_lap_teleports += 1
        self.get_logger().warn(
            f"[stuck-recover] /initialpose → ({rx:.2f},{ry:.2f},{yaw:.2f}) "
            f"after {self._stuck_release_total} stuck cycles "
            f"(lap teleports={self._lmpc_lap_teleports})")

    # Recovery-steer sign during reverse. +1 / -1, validated empirically.
    # 후진 중 steer→yaw 가 전진과 반대 → 이 부호로 검증/뒤집기.
    _RECOVERY_STEER_SIGN = -1.0

    def _recovery_steer(self, x0):
        """후진(STUCK release) 중 centerline 으로 되돌아가도록 조향각 계산.

        car (x,y,psi) 를 centerline 에 투영해 signed lateral error e_c 와
        heading error 를 구하고, 이에 비례하는 조향각을 ±max_steer 로 clamp.
        벽에서 멀어지는 방향으로 후진하게 만들어 재끼임 CASCADE 를 차단.
        """
        if self._track is None or x0 is None or len(x0) < 3:
            return 0.0
        try:
            x = float(x0[0]); y = float(x0[1]); psi = float(x0[2])
            s_now, _ = find_current_arc_length(self._track, np.array([x, y]))
            cx = float(self._track.center_lut_x(s_now))
            cy = float(self._track.center_lut_y(s_now))
            dx = float(self._track.center_lut_dx(s_now))
            dy = float(self._track.center_lut_dy(s_now))
            tnorm = math.hypot(dx, dy)
            if tnorm < 1e-9:
                return 0.0
            dx /= tnorm; dy /= tnorm
            # signed lateral error: (car - center) · left-normal(-dy, dx)
            # e_c > 0 → 차가 centerline 의 왼쪽 → 오른쪽으로 복귀해야 함.
            e_c = (x - cx) * (-dy) + (y - cy) * dx
            # heading error vs centerline tangent (전진 기준), wrap to [-pi,pi]
            psi_t = math.atan2(dy, dx)
            e_psi = math.atan2(math.sin(psi - psi_t), math.cos(psi - psi_t))
            max_steer = float(getattr(self.mpc, 'theta_max', 0.3) or 0.3)
            # lateral + heading 항을 합쳐 복귀 조향 산출.
            k_lat = 0.6; k_psi = 0.5
            raw = -(k_lat * e_c + k_psi * e_psi)
            steer = self._RECOVERY_STEER_SIGN * raw
            return float(max(-max_steer, min(max_steer, steer)))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"[stuck-recover] recovery steer calc failed: {e}",
                throttle_duration_sec=2.0)
            return 0.0

    def _control_loop_cb(self):
        if not self._mpc_ready:
            return  # waiting on track loader (TODO)
        x0 = self._current_state_4()
        if x0 is None:
            return
        # unified 7-dim state [x,y,ψ,vx,vy,r,s] (both modes) → s = x0[6].
        s_now = float(x0[6])
        # B: κ-aware online weight adaptation (auto_tune=true 시 활성)
        self._adaptive_weight_update(s_now)
        # 2026-05-28 #18 LMPC: 매 cycle SS query → set mpc._lmpc_* attrs
        try:
            self._lmpc_update_per_cycle(x0, s_now)
        except Exception as _e:
            self.get_logger().warn(f"[LMPC] cycle update skipped: {_e}",
                                    throttle_duration_sec=2.0)
        # ── Cold-start vx floor (MPPI init_vel) — STARTUP ONLY ──
        # 정지 출발 시 dynamic Pacejka 가 ill-conditioned → solver x0 의 vx 를 floor.
        # ★ 2026-05-29 BUG FIX: 기존엔 startup 게이트 없이 매 cycle raw_vx<floor 면
        # 적용 → 코너 감속/wedge 정지(v=0)에서도 solver 에 vx=2.0 거짓 전달 →
        # solver 가 멈춘 걸 못 봐 탈출 못 함(s≈19 영구 wedge) + floor on/off 토글로
        # 입력 비일관(chatter). → `_has_moved`(차가 한 번이라도 실제로 움직였나, mpc
        # 의 stuck-detector latch)로 게이트: 출발 전(true standstill)만 floor, 그 후엔
        # solver 가 진짜 속도를 봄. unified layout: 양 모드 적용 (f_kin 은
        # ill-conditioning 은 없지만 동일 거동 유지가 원칙).
        vx_floor = float(self.get_parameter('cold_start_vx_floor').value) \
            if self.has_parameter('cold_start_vx_floor') else 0.0
        raw_vx = float(x0[3])
        x0_solve = x0
        if (vx_floor > 0.0 and raw_vx < vx_floor
                and not getattr(self.mpc, '_has_moved', False)):
            x0_solve = x0.copy()
            x0_solve[3] = vx_floor
        t0 = time.monotonic()
        try:
            con_first, traj, u_seq, opti_value = self.mpc.solve(x0_solve, self._obstacles)
        except Exception as e:
            import traceback
            self.get_logger().error(f"mpc.solve raised: {e}\n{traceback.format_exc()}")
            return
        solve_dt = time.monotonic() - t0

        # Joint-α terminal debug — confirms whether the terminal actually targets
        # the APEX (tgt_lat≈±0.9) or centerline. (The SS query is now anchored to
        # track arc length in _lmpc_update_per_cycle, so the previous horizon-end
        # x_N stash is gone — it was the drift-feedback source.)
        if getattr(self, '_lmpc_use', False) and traj is not None and traj.shape[0] > 1:
            try:
                ss = getattr(self.mpc, '_lmpc_ss_states', None)
                if traj.shape[1] >= 18 and ss is not None and self._track is not None:
                    a = np.asarray(traj[-1, 8:18], dtype=float); a = a / (a.sum() + 1e-6)
                    ssm = np.asarray(ss)            # (4,K)
                    tgt = ssm @ a                   # (4,)
                    cl = np.asarray(self._track.center_lane)
                    ang = getattr(self._track, 'center_point_angles', None)

                    def _lat(xy):
                        i = int(np.argmin(np.linalg.norm(cl[:, :2] - xy[:2], axis=1)))
                        psi = float(ang[i]) if (ang is not None and i < len(ang)) else 0.0
                        return math.sin(psi) * (xy[0] - cl[i, 0]) - math.cos(psi) * (xy[1] - cl[i, 1])
                    ssl = [_lat(ssm[:2, j]) for j in range(ssm.shape[1])]
                    self.get_logger().info(
                        f"[αdbg] tgt_lat={_lat(tgt):+.2f} car_lat={_lat(np.asarray(x0_solve)):+.2f} "
                        f"amax={a.max():.2f}@{int(a.argmax())} ssLat[{min(ssl):+.2f},{max(ssl):+.2f}]",
                        throttle_duration_sec=0.5)
            except Exception:
                pass

        # B4'.2: log the applied control [a_x,delta,p_v] in lockstep with the
        # per-cycle state append (state appended pre-solve in _lmpc_update_per_cycle;
        # u_seq[0] is the control applied from that state). Needed for the lap
        # residual = actual - f_expl(state, u); a zero/stub input would make the
        # residual absorb the control effect instead of model error.
        try:
            if (getattr(self, '_lmpc_use', False) and u_seq is not None and len(u_seq) > 0
                    and getattr(self, '_lmpc_pending_state', None) is not None):
                # Lockstep: append state[t], time[t], input[t] together, ONLY after a
                # successful solve, so the lap buffer stays 1:1 aligned. (state/time
                # were deferred from pre-solve; a failed solve appends nothing.)
                self._lmpc_lap_buf['state'].append(self._lmpc_pending_state)
                self._lmpc_lap_buf['time'].append(self._lmpc_pending_time)
                self._lmpc_lap_buf['input'].append(np.asarray(u_seq[0], float).copy())
                self._lmpc_pending_state = None
        except Exception:
            pass

        # B4' prediction-error gate: realized state8 (stashed this cycle) vs the
        # prediction from the previous (state, control). Guarded — never breaks control.
        try:
            if getattr(self, '_b4_state8', None) is not None and u_seq is not None and len(u_seq) > 0:
                self._b4_pred_error_log(
                    self._b4_state8, np.asarray(u_seq[0], float),
                    float(self.get_parameter('dT').value))
        except Exception:
            pass

        # Output: AckermannDrive
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        # 2026-06-10 unified layout: u[0] = a_x in BOTH modes — drive.speed 로
        # 보내면 안 됨! 대신 traj 의 예측 vx (state[3]) 사용.
        if traj is not None and traj.shape[0] > 1:
            # gym 의 PID 가 추종할 target vx (both modes).
            #
            # Agent R-round2 Fix 1: v_now + floor 를 solver v_max cap 안에 강제.
            # 이전 R1 (v_now + 1.0 무제한) 가 v=9.62 over-shoot → 첫 코너 진입에
            # 못 회전 → boundary 박힘 → systemic stuck. solver 가 v_max=3.80 으로
            # 제약됐는데 actuator cmd 가 escape 했던 게 root cause.
            #
            # Fix 3: STUCK release active → cmd=0/steer=0 강제 (wall contact 해제).
            v_now = float(x0[3]) if (x0 is not None and len(x0) > 3) else 0.0
            # C-SPEED startup ramp (2026-06-19): ramp the MPC's COST TARGET
            # (mpc.v_max / p_max) — not just the output speed — from 2.5→v_max
            # over RAMP_T s. The veer comes from the MPC PLANNING aggressively for
            # high ref_v ahead while still acquiring the line at launch; ramping
            # the cost target makes the plan itself gentle so the car settles on
            # the line first. _startup_t resets on each stuck-recovery.
            _rt = time.monotonic() - getattr(self, '_startup_t', time.monotonic())
            _RAMP_T = 1.5
            _vm_full = float(self.get_parameter('max_speed').value)
            try:
                if _rt < _RAMP_T:
                    _vm_r = 2.5 + max(0.0, _vm_full - 2.5) * max(0.0, _rt / _RAMP_T)
                    self.mpc.v_max = _vm_r
                    self.mpc.p_max = _vm_r
                elif float(getattr(self.mpc, 'v_max', 0.0)) < _vm_full:
                    self.mpc.v_max = _vm_full
                    self.mpc.p_max = _vm_full
            except Exception:
                pass
            v_cap = float(getattr(self.mpc, 'v_max', 4.0))
            # Agent A round-4 fix: gym in_collision latch 은 reverse 도 못 풀어줌.
            # cumulative count wait 가 release_active reset 으로 누적 못 함.
            # → stuck 감지 즉시 /initialpose publish (5s cooldown).
            if getattr(self.mpc, '_stuck_release_active', False):
                self._stuck_release_total += 1
                cmd.drive.speed = -0.5
                # ── Recovery steer toward centerline (was steering=0) ──
                # 이전: reverse 중 steer=0 → 같은 나쁜 heading 으로 직선 후진 →
                # 같은 각도로 벽 재접근 → 재끼임 CASCADE.
                # Fix: 후진 중 centerline 방향으로 조향해 벽에서 멀어지게 함.
                # ★ 후진 시 steer→yaw 관계가 전진과 반대 → 부호 검증 필요.
                cmd.drive.steering_angle = self._recovery_steer(x0)
                # 즉시 reset (5s cooldown 만 유지). 실차 (enable_sim_reset=false)
                # 에선 publish 해도 무의미 → skip + warn 만.
                if (time.monotonic() - self._last_reset_t) > 1.0:   # 2026-06-02: 5s→1s. 5s 쿨다운이 첫 리스폰 후 재시도를 막아 stuck-loop 지속. 1s 면 gym latch 안 풀려도 빠르게 재리스폰.
                    if bool(self.get_parameter('enable_sim_reset').value):
                        self._publish_safe_reset(x0)
                    else:
                        self._mpc_log.warn_throttle(2.0,
                            "[real] STUCK detected but enable_sim_reset=false — "
                            "publishing reverse cmd only (수동 e-stop 권장)")
                    self._last_reset_t = time.monotonic()
                    # Do NOT zero _stuck_release_total on teleport — let it
                    # accumulate over the whole stuck episode (reset happens only
                    # on recovery, in the non-stuck else branch) so it reflects
                    # true severity and we can escalate instead of teleporting
                    # at 1 Hz forever with no give-up.
                    if self._stuck_release_total > 400:   # ~10 s at 40 Hz, no recovery
                        self.get_logger().error(
                            "[stuck-recover] still stuck after %d cycles — recovery "
                            "not progressing; manual intervention recommended"
                            % self._stuck_release_total,
                            throttle_duration_sec=3.0)
            else:
                self._stuck_release_total = 0
                # 2026-05-29 structural-bug fix: the target vx sent to the gym
                # PID must reflect the solver's NEAR-TERM plan, not traj[-1, 3]
                # (horizon-END vx). At a corner the solver brakes now and
                # re-accelerates after the apex, so traj[-1, 3] stays high
                # (~post-corner speed) while the immediate plan (a_x<0,
                # near-step vx) is braking. Commanding the horizon-end vx
                # ignored the brake entirely → the car plowed into the apex at
                # full speed and wedged (s≈19 on rand_a). Use the MINIMUM over
                # the near horizon so a planned brake is honored promptly while
                # still covering the actuator/PID lag (~0.16 s). A tiny forward
                # assist applies ONLY from near-standstill so the car can launch
                # from v=0 — it never overrides braking once the car is moving.
                # Look at the planned vx over steps 1..n_near (exclude index 0,
                # which is just the current state). Taking the MIN honors a
                # planned brake immediately; on a straight where the plan
                # accelerates, every near step ≥ v_now so the command still
                # rises (no stall).
                # near-horizon MIN honors a planned brake promptly (covers ~0.16s
                # actuator lag) while still rising on straights (min of an
                # increasing plan = traj[1] ≈ current+step → gradual accel). The
                # earlier traj[-1] (horizon-end) ignored the brake → plowed into
                # apex → wedge. (Note: a prior "min kills accel" theory was a
                # red herring — the slowness was a leftover max_speed=3 config.)
                # ── 2026-06-12 직선 저속 수정: 플랜트 P-제어 역모델 preview ──
                # gym PID 는 accl = kp·(v_cmd − v_now), 가속 kp = 10·a_max/v_max
                # = 10·9.51/20 ≈ 4.76 /s (dynamic_models.pid). 이전 near-min
                # (steps 1..4 ≈ 0.16 s) 은 가속 중 v_cmd ≈ v_now+0.16 → 전달
                # 가속 4.76×0.16 ≈ 0.76 m/s² = 계획 a_x(4.0)의 19%. CSV 실측:
                # 직선 ref_v 4.87 인데 v_actual 2.88 천장 (mpc_20260611_131959).
                # 수정: 가속 구간은 PID 시정수 1/kp ≈ 0.21 s 만큼 미리보기한
                # 계획속도 traj[τ] 를 명령 → 플랜트가 계획 a_x 를 실제로 냄.
                # 브레이크 정직성 유지: preview 창 안에 v_now 아래로 내려가는
                # 스테이지가 하나라도 있으면 그 MIN 을 즉시 명령 (옛 traj[-1]
                # horizon-끝 방식이 브레이크 무시 → apex 돌진/wedge 했던 버그
                # 의 재발 방지 — min-가드가 그 오류를 막는다).
                n_prev = min(7, traj.shape[0])           # steps 1..6 ≈ 0.24 s ≈ 1/kp
                if n_prev > 1:
                    v_brk = float(np.min(traj[1:n_prev, 3]))
                    if v_brk < v_now - 0.05:             # 감속 계획 → 즉시 브레이크
                        v_plan = v_brk
                    else:                                # 가속/유지 → τ-preview 속도
                        v_plan = float(traj[n_prev - 1, 3])
                else:
                    v_plan = float(traj[-1, 3])
                # 2026-06-15 BRAKE-DISTANCE-AWARE 속도명령 (실차 bag 진단):
                # 기존 v_plan=앞 0.24초 plan min ≈ 현재속도 → 발행 속도명령이 코너
                # 감속을 못 봄 (bag: 명령 deficit vs 실속도 ≤0.28 m/s, 강제동 0회) →
                # VESC 순항 → 코스팅 ~0.8 → 코너 못 줄여 박음. (VESC는 명령만 낮으면
                # 제동함 — 벤치+bag 확인.) 수정: plan 의 모든 미래 stage 속도 v[i]
                # (거리 d[i] 앞)에 대해 "지금부터 a_brake 로 제동하면 도달가능한 최고속"
                # = √(v[i]²+2·a_brake·d[i]) 의 min 을 명령 → 코너를 제동거리만큼 일찍
                # 보고 명령이 떨어짐 → VESC 가 제때 능동제동. d 는 plan 의 s(state[6]).
                try:
                    _sH = np.asarray(traj[:, 6], dtype=float)
                    _vH = np.asarray(traj[:, 3], dtype=float)
                    _d  = np.maximum(0.0, _sH - _sH[0])
                    _ab = max(0.3, float(self.get_parameter('brake_anticip_a').value))
                    _vallow = np.sqrt(np.maximum(0.0, _vH * _vH + 2.0 * _ab * _d))
                    # FUTURE stages only (i>=1). Including i=0 (d=0 → v_allow=v_now)
                    # would cap the command at the current speed → blocks all
                    # acceleration (car crawls). Only upcoming slowdowns should bind.
                    if _vallow.shape[0] > 1:
                        v_plan = min(v_plan, float(np.min(_vallow[1:])))
                except Exception:
                    pass
                if v_now < 0.5:                          # launch-from-rest assist
                    v_plan = max(v_plan, min(v_cap, v_now + 0.5))
                cmd.drive.speed = max(0.0, min(v_cap, v_plan))
        else:
            # traj missing (defensive — solve() always returns one). u[0] is a_x
            # in the unified layout, NOT a speed: never forward it as drive.speed.
            cmd.drive.speed = 0.0
        cmd.drive.steering_angle = float(con_first[1])
        # (2026-06-10 unified layout: the old kinematic-only stuck-escalation
        # block was removed — the traj-based branch above, incl. its teleport
        # escalation, now serves BOTH modes.)

        # ── Cold-start output speed floor (replaces old 3s PP warmup ramp) ──
        # 실측 vx 가 floor 미만이면 forward push 보장 (차가 가속해 floor 넘김).
        # 단 steering 은 t=0 부터 SOLVER (filtered) 출력 사용 → PP handoff 없이
        # MPCC 가 처음부터 조향. 위 vx-floor 덕에 정지에서도 solver 출력 정상.
        # stuck-release(후진) 중이면 floor 적용 안 함. unified: 양 모드.
        startup_speed = float(self.get_parameter('startup_speed').value) \
            if self.has_parameter('startup_speed') else 0.0
        if (vx_floor > 0.0 and raw_vx < vx_floor
                and not getattr(self.mpc, '_stuck_release_active', False)):
            cmd.drive.speed = max(float(cmd.drive.speed), startup_speed)

        # C-STEER (2026-06-19): hard OUTPUT steering-rate limit to servo slew.
        # The solver's first control jumps full-lock on the flat cost surface
        # (documented sign-flip hunting) → launch swerve + crash. Clamp published
        # steer to ±sv_max·Δt of the last applied → servo-feasible → hunting
        # killed at the actuator. Time-based (robust to publish rate).
        _now_t = self.get_clock().now().nanoseconds * 1e-9
        _sv_max = 3.2
        _prev_st = getattr(self, '_last_steer_pub', None)
        _prev_t = getattr(self, '_last_steer_t', None)
        if _prev_st is not None and _prev_t is not None and _now_t > _prev_t:
            _dmax = _sv_max * min(0.1, _now_t - _prev_t)
            _s = float(cmd.drive.steering_angle)
            cmd.drive.steering_angle = float(
                max(_prev_st - _dmax, min(_prev_st + _dmax, _s)))
        self._last_steer_pub = float(cmd.drive.steering_angle)
        self._last_steer_t = _now_t

        self.ackermann_pub.publish(cmd)

        # Diagnostics
        self.cost_pub.publish(Float64(data=float(opti_value)))
        # ROS1 호환: ms 단위 (Nonlinear_MPC_node.py:328). logger [dbg] 출력
        # 도 "ms" 표기라 seconds 그대로 publish 하면 0 으로 보임.
        self.solve_time_pub.publish(Float64(data=float(solve_dt * 1000.0)))
        self.feasible_pub.publish(Bool(data=bool(opti_value < 1e8)))

        # MPC trajectory as nav_msgs/Path
        path = Path()
        path.header.stamp = cmd.header.stamp
        path.header.frame_id = 'map'
        for k in range(traj.shape[0]):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=float(traj[k, 0]), y=float(traj[k, 1]), z=0.0)
            siny = math.sin(float(traj[k, 2]) * 0.5)
            cosy = math.cos(float(traj[k, 2]) * 0.5)
            ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=siny, w=cosy)
            path.poses.append(ps)
        self.mpc_traj_pub.publish(path)

        # ── Prediction-consistency metric (SQP_RTI shake) ──
        # Compare this cycle's predicted (x,y) path against the PREVIOUS
        # cycle's prediction advanced by one node (one dt has elapsed). A
        # consistent solver reproduces nearly the same geometry shifted by one
        # step; the residual RMS quantifies the cycle-to-cycle "예측선 흔들림".
        # Logged at ~1 Hz so a run's shake level can be read straight from the
        # sim log (grep "pred-consistency"). Skipped during stuck-release
        # (reverse) cycles where traj is an override, not a real prediction.
        try:
            cur_xy = traj[:, 0:2]
            prev_xy = getattr(self, '_prev_pred_xy', None)
            in_release = bool(getattr(self.mpc, '_stuck_release_active', False))
            if prev_xy is not None and not in_release:
                m = min(cur_xy.shape[0] - 1, prev_xy.shape[0] - 1)
                if m > 1:
                    d = np.linalg.norm(cur_xy[:m] - prev_xy[1:m + 1], axis=1)
                    rms = float(np.sqrt(np.mean(d * d)))
                    e = getattr(self, '_pred_shake_ema', None)
                    self._pred_shake_ema = rms if e is None else 0.9 * e + 0.1 * rms
                    self.get_logger().info(
                        "[pred-consistency] rms=%.4f ema=%.4f" % (rms, self._pred_shake_ema),
                        throttle_duration_sec=1.0)
            self._prev_pred_xy = cur_xy.copy()
        except Exception:
            pass

        # MarkerArray twin — 속도 색상 LINE_STRIP (2026-06-11, 구 sphere 대체).
        # 파랑(0)→초록(½v_max)→빨강(v_max) per-vertex 색으로 예측궤적의 속도
        # 프로파일이 rviz에서 한눈에 보임. 끝점은 흰 구로 horizon 끝 표시.
        markers = MarkerArray()
        # DELETEALL first so leftover markers from previous runs/horizons
        # don't linger (defensive; horizon size is fixed but cheap to guard).
        clear = Marker()
        clear.header = path.header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        line = Marker()
        line.header = path.header
        line.ns = 'mpc_traj'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.12  # line width (더 굵게)
        vmax = max(1.0, float(getattr(self.mpc, 'v_max', 8.0)))
        for k in range(traj.shape[0]):
            line.points.append(Point(x=float(traj[k, 0]), y=float(traj[k, 1]),
                                     z=0.05))
            h = min(1.0, max(0.0, float(traj[k, 3]) / vmax))  # vx = state idx 3
            line.colors.append(ColorRGBA(r=h, g=1.0 - abs(2.0 * h - 1.0),
                                         b=1.0 - h, a=1.0))
        markers.markers.append(line)
        end = Marker()
        end.header = path.header
        end.ns = 'mpc_traj'
        end.id = 1
        end.type = Marker.SPHERE
        end.action = Marker.ADD
        end.pose.position.x = float(traj[-1, 0])
        end.pose.position.y = float(traj[-1, 1])
        end.pose.position.z = 0.05
        end.pose.orientation.w = 1.0
        end.scale.x = end.scale.y = end.scale.z = 0.18
        end.color.r = end.color.g = end.color.b = end.color.a = 1.0
        markers.markers.append(end)
        self.mpc_traj_markers_pub.publish(markers)

        # MPC prediction (raw state + inputs per stage) → osuf1_common/MPCTrajectory
        traj_msg = MPCTrajectory()
        traj_msg.header = path.header
        for k in range(traj.shape[0] - 1):
            pred = MPCPrediction()
            pred.state = traj[k].astype(np.float32).tolist()
            pred.inputs = u_seq[k].astype(np.float32).tolist()
            traj_msg.trajectory.append(pred)
        self.prediction_pub.publish(traj_msg)

        # /mpc_debug Float32MultiArray — 16 fields matching ROS1 DBG_FIELDS
        # order (see mpc_debug_logger.DBG_FIELDS). Consumed by the logger
        # node for CSV / event-dump anomaly detection.
        v_actual = 0.0
        if self._last_odom is not None:
            v_actual = float(self._last_odom.twist.twist.linear.x)
        current_s, near_idx = 0.0, 0
        if self._track is not None:
            current_s, near_idx = find_current_arc_length(
                self._track, np.array([x0[0], x0[1]]))
        try:
            ref_v_now = float(self.mpc.ref_v(current_s % float(self._track.element_arc_lengths_orig[-1])))
        except Exception:
            ref_v_now = 0.0
        # C-1 학습 데이터: κ_lookahead + B 의 동적 scale 4개 추가 (총 22 필드)
        kappa_abs, kappa_signed = 0.0, 0.0
        if self._track is not None:
            try:
                L = float(self._track.element_arc_lengths_orig[-1])
                kappa_abs = float(self.mpc.abs_kappa_lut(current_s % L))
                kappa_signed = float(self.mpc.signed_kappa_lut(current_s % L))
            except Exception:
                pass
        dbg = Float32MultiArray()
        dbg.data = [
            float(con_first[0]),       # 0  a_x cmd (unified u[0]; was v_cmd pre-2026-06-10)
            float(con_first[1]),       # 1  steer_cmd
            v_actual,                  # 2  v_actual
            float(x0[0]),              # 3  car_x
            float(x0[1]),              # 4  car_y
            float(x0[2]),              # 5  car_yaw
            float(current_s),          # 6  current_s
            float(near_idx),           # 7  near_idx
            ref_v_now,                 # 8  ref_v
            float(len(self._obstacles)),  # 9  n_obs_in
            0.0,                       # 10 sel_dmin  (obstacle TODO)
            0.0,                       # 11 sel_x
            0.0,                       # 12 sel_y
            0.0,                       # 13 side_pref
            float(opti_value),         # 14 opti_value
            float(solve_dt * 1000.0),  # 15 solve_ms
            # C-1 추가 — MLP 학습 input (state) + B output (scale):
            kappa_abs,                 # 16 kappa_abs (forward-max κ, B input)
            kappa_signed,              # 17 kappa_signed (apex 방향)
            float(getattr(self.mpc, 'q_cte_scale_live', 1.0)),    # 18
            float(getattr(self.mpc, 'q_lag_scale_live', 1.0)),    # 19
            float(getattr(self.mpc, 'q_v_scale_live', 1.0)),      # 20
            float(getattr(self.mpc, 'q_drate_scale_live', 1.0)),  # 21
            float(getattr(self.mpc, 'v_max', 0.0)),                # 22 v_max_cost
        ]
        self.mpc_debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
