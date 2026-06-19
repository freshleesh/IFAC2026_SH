"""HJ 풀 racing stack + f1tenth_gym sim 통합 launch.

체인:
  gym_bridge (sim) → /car_state/odom + /scan + /car_state/pose
    → frenet_odom_republisher → /car_state/odom_frenet
    → state_machine (FSM) → /local_waypoints + /behavior_strategy
    → simple_pp (pure pursuit, vx_mps 그대로) → /vesc/high_level/ackermann_cmd
    → simple_mux → /vesc/ackermann_cmd → gym_bridge (driving)
    (mac 실차의 low_level_mac 과 토픽 이름이 다름:
       sim   mux in = /vesc/high_level/ackermann_cmd        / out = /vesc/ackermann_cmd
       mac   mux in = /vesc/high_level/.../input/nav_1       / out = /ackermann_cmd)

LaunchArgs:
  map (f): 맵 이름 — stack_master/maps/<name>/{<name>.{png,yaml}, global_waypoints.json}
  racecar_version (SIM): 차량 설정 이름
  mode (timetrial | overtake | mpcc): 운영 모드
    - timetrial: GB_TRACK 만, 추월 분기 비활성 (n_obstacles=0). 검증된 기본 모드.
    - overtake : OVERTAKE 분기 + spliner 정적 회피 + 가짜 장애물 (n_obstacles=4).
    - mpcc     : controller_manager 대신 nonlinear_mpc_acados (MPCC) 사용.
                 timetrial 인프라 + mpc_node + joy_node + auto-engage helper.
                 IFAC 데모용. 자체 reference 추종, state_machine은 GB_TRACK 강제.
  n_obstacles (auto): 명시 시 mode 와 무관하게 강제. 0=정적 장애물 발생 안 함.

기동 순서 (TimerAction):
  t=0:  global_republisher + low_level (gym_bridge + simple_mux + obstacle + rviz)
  t=2:  frenet_conversion_server + frenet_odom_republisher
  t=3:  fake_topic_relay + random_obstacle_publisher
  t=4:  spliner (overtake 모드일 때만)
  t=5:  state_machine
  t=6:  mpcc 모드면 mpc_node + mpc_debug_logger
  t=7:  simple_pp (timetrial/overtake) — mpcc 모드면 생략
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _build(context: LaunchContext, *_args, **_kwargs):
    map_name = LaunchConfiguration("map").perform(context)
    racecar_version = LaunchConfiguration("racecar_version").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    controller = LaunchConfiguration("controller").perform(context)

    if mode not in ("timetrial", "overtake", "avoid", "mpcc"):
        raise ValueError(f"mode must be 'timetrial' / 'overtake' / 'avoid' / 'mpcc', got {mode!r}")
    if controller not in ("pp_heading", "simple_pp", "mppi"):
        raise ValueError(f"controller must be 'pp_heading' / 'simple_pp' / 'mppi', got {controller!r}")

    # 'avoid' = 'overtake' alias — 정적 장애물 회피 의도 명시. 코드 동작 동일.
    if mode == "avoid":
        mode = "overtake"

    # mode=mpcc 는 controller 인자보다 우선 (기존 mpcc 분기 유지).
    is_mpcc = (mode == "mpcc")
    use_mppi = (not is_mpcc) and (controller == "mppi")
    use_pp_heading = (not is_mpcc) and (controller == "pp_heading")
    # MPCC mode: state_machine 은 GB_TRACK 강제 (mpc 가 자체 reference 추종).
    timetrials_only = (mode == "timetrial") or is_mpcc
    force_gbtrack = (mode == "timetrial") or is_mpcc
    ot_planner = "spliner" if mode == "overtake" else ""

    sm_share = get_package_share_directory("stack_master")
    # canonical map storage (src side — fast_livo SaveMap 가 새 파일 쓰는 곳).
    # Ubuntu dev: 레포 == colcon ws / Mac mini 실차: 레포가 ws/src/IFAC2026_SH 로 클론됨.
    _ws = os.path.normpath(os.path.join(sm_share, '..', '..', '..', '..'))
    _repo = next(
        (r for r in (_ws, os.path.join(_ws, 'src', 'IFAC2026_SH'))
         if os.path.isdir(os.path.join(r, 'src', 'system', 'stack_master'))),
        _ws)
    maps_src_root = os.path.join(_repo, 'src', 'system', 'stack_master', 'maps')
    controller_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "sim_controller_params.yaml"
    )

    # ── low_level ──
    low_level = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(sm_share, "launch", "low_level.launch.xml")),
        launch_arguments={"sim": "true", "map": map_name, "maps_root": maps_src_root}.items(),
    )

    # ── global_republisher ──
    global_repub = Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[
            os.path.join(sm_share, "config", "global_republisher.yaml"),
            {
                "map": map_name,
                "map_path": os.path.join(maps_src_root, map_name, "global_waypoints.json"),
            },
        ],
        output="screen",
    )

    # ── frenet ──
    frenet_server = TimerAction(period=2.0, actions=[Node(
        package="frenet_conversion",
        executable="frenet_conversion_server",
        name="frenet_conversion_server",
        output="screen",
    )])
    frenet_odom_repub = TimerAction(period=2.0, actions=[Node(
        package="frenet_odom_republisher",
        executable="frenet_odom_republisher",
        name="frenet_odom_republisher",
        remappings=[
            ("/odom", "/car_state/odom"),
            ("/odom_frenet", "/car_state/odom_frenet"),
            ("/odom_frenet_fixed", "/car_state/odom_frenet_fixed"),
        ],
        output="screen",
    )])

    # ── fake topic relay + obstacle pub ──
    fake_relay = TimerAction(period=3.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])
    # random_obs = TimerAction(period=3.0, actions=[Node(
    #     package="random_obstacle_publisher",
    #     executable="random_obstacle_publisher",
    #     name="random_obstacle_publisher",
    #     parameters=[{"n_obstacles": n_obstacles, "rate_hz": 20.0}],
    #     remappings=[("/obstacles", "/tracking/obstacles")],
    #     output="screen",
    # )])

    # ── overtake 분기 노드 (spliner) ──
    actions = [
        low_level,
        global_repub,
        frenet_server, frenet_odom_repub,
        fake_relay,
        #random_obs,
    ]
    if mode == "overtake":
        spliner_node = TimerAction(period=4.0, actions=[Node(
            package="spliner",
            executable="static_avoidance_node",  # 정적 + 동적 회피 spliner
            name="spliner",
            output="screen",
        )])
        actions.append(spliner_node)

    # ── state_machine ──
    sm_node = TimerAction(period=5.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            "racecar_version": racecar_version,
            "map": map_name,    # ROS2: state_machine init 의 ot_sectors.yaml fallback 용
            "state_machine.rate": 50.0,
            "state_machine.n_loc_wpnts": 80,
            "state_machine.ot_planner": ot_planner,
            "state_machine.timetrials_only": timetrials_only,
            "state_machine.gb_ego_width_m": 0.3,
            # OVERTAKE ↔ GB_TRACK 진동 방지 — sim hysteresis 강화
            "state_machine.gb_horizon_m": 5.0,         # 1.0 → 5.0 (enemy_in_front 더 길게 True)
            "state_machine.lateral_width_gb_m": 0.3,
            "state_machine.interest_horizon_m": 20.0,
            "state_machine.use_force_trailing": False,
            "state_machine.splini_ttl": 5.0,            # 2.0 → 5.0 (회피 wpnts freshness)
            "state_machine.pred_splini_ttl": 0.2,
            "state_machine.overtaking_horizon_m": 6.9,
            "state_machine.lateral_width_ot_m": 0.3,
            "state_machine.splini_hyst_timer_sec": 3.0,  # 0.75 → 3.0
            "state_machine.emergency_break_horizon": 1.1,
            "state_machine.ftg_speed_mps": 1.0,
            "state_machine.ftg_timer_sec": 3.0,
            "state_machine.ftg_active": False,
            "state_machine.force_GBTRACK": force_gbtrack,
            "state_machine.overtaking_ttl_sec": 10.0,    # 3.0 → 10.0 (OVERTAKE 종료 지연)
            "state_machine.volt_threshold": 10.0,
            "/global_republisher/track_length": 25.0,
            "measure": False,
            "sim": True,
        }],
        output="screen",
    )])
    actions.append(sm_node)

    if use_pp_heading:
        # ── pp_heading_controller (현행 메인: PP + friction-circle + heading PID) ─
        # sub /car_state/odom, /local_waypoints  ·  pub mux input nav_1
        pp_heading_yaml = os.path.join(
            get_package_share_directory("controller"), "config", "pp_heading_params.yaml"
        )
        controller_node = TimerAction(period=7.0, actions=[Node(
            package="controller",
            executable="pp_heading_controller",
            name="pp_heading_controller",
            parameters=[pp_heading_yaml, {"drive_topic": "/vesc/high_level/ackermann_cmd"}],
            output="screen",
        )])
        actions.append(controller_node)
    elif not is_mpcc and not use_mppi:
        # ── simple_pp (minimal pure-pursuit, vx_mps 그대로) ────────────
        # 기존 controller_manager (L1 + lat_err/accel_lim 후처리) 가 vx_mps 를
        # 깎는 문제 디버깅용 교체. middle_level_mac 과 동일 노드/파라미터.
        friction_circle_yaml = os.path.join(
            get_package_share_directory("controller"), "config", "friction_circle.yaml"
        )
        controller_node = TimerAction(period=7.0, actions=[Node(
            package="controller",
            executable="fc_node",
            name="friction_circle_controller",
            parameters=[friction_circle_yaml],
            output="screen",
        )])
        actions.append(controller_node)
    elif use_mppi:
        # ── MPPI (simple_pp 자리 drop-in 교체) ────────────────────────
        # 인터페이스 계약 = simple_pp 와 동일:
        #   sub /car_state/odom,  pub /vesc/high_level/ackermann_cmd
        # raceline / wall sdf 자동 매칭: stack_master/maps/<map>/ 단일 소스.
        #   raceline: stack_master/maps/<map>/raceline.csv
        #   wall sdf: stack_master/maps/<map>/<map>.yaml
        # 둘 다 launch 인자로 override 가능.
        mppi_share = get_package_share_directory("mppi_bringup")
        sm_share_mppi = get_package_share_directory("stack_master")
        mppi_params = LaunchConfiguration("mppi_params_file").perform(context)
        if not mppi_params:
            mppi_params = os.path.join(mppi_share, "config", "params_sim_mac.yaml")
        mppi_wpt = LaunchConfiguration("mppi_wpt_path").perform(context)
        if not mppi_wpt:
            mppi_wpt = os.path.join(sm_share_mppi, "maps", map_name, "raceline.csv")
            if not os.path.exists(mppi_wpt):
                raise FileNotFoundError(
                    f"mppi raceline 자동 매칭 실패: {mppi_wpt} 없음. "
                    f"mppi_wpt_path:=<csv 경로> 로 명시하거나 "
                    f"stack_master/maps/{map_name}/raceline.csv 생성 필요."
                )
        mppi_wall_map = LaunchConfiguration("mppi_wall_map").perform(context)
        if not mppi_wall_map:
            mppi_wall_map = os.path.join(sm_share_mppi, "maps", map_name, f"{map_name}.yaml")
            if not os.path.exists(mppi_wall_map):
                # wall sdf 자동 매칭 실패면 비워서 SDF off — fatal 아님.
                mppi_wall_map = ""

        mppi_overrides = {
            # full_sim 통합 시나리오 override — yaml 의
            # 단독 sim_full_mac default 를 덮음.
            "pose_topic": "/car_state/odom",
            "drive_topic": "/vesc/high_level/ackermann_cmd",
            "wpt_path_absolute": True,
            "wpt_path": mppi_wpt,
        }
        if mppi_wall_map:
            mppi_overrides["wall_cost_map_yaml"] = mppi_wall_map
        else:
            mppi_overrides["wall_cost_enabled"] = False
            mppi_overrides["wall_cost_map_yaml"] = ""
        mppi_node = TimerAction(period=7.0, actions=[Node(
            package="mppi_example",
            executable="mppi_node",
            name="lmppi_node",
            output="log",
            parameters=[mppi_params, mppi_overrides],
        )])
        actions.append(mppi_node)
    else:
        # ── MPCC 모드: nonlinear_mpc_acados 가 controller 자리 대체 ──
        mpc_share = get_package_share_directory("nonlinear_mpc_acados")
        mpc_params = os.path.join(mpc_share, "config", "ddrx_unified_params.yaml")
        # ACADOS env (libacados.so / Tera renderer / generated solver dlopen).
        # 사용자가 export 했다면 그 값 우선, 없으면 ~/acados.
        acados_dir = os.environ.get("ACADOS_SOURCE_DIR") or os.path.expanduser("~/acados")
        ld_extra = os.path.join(acados_dir, "lib")
        actions.append(SetEnvironmentVariable("ACADOS_SOURCE_DIR", acados_dir))
        actions.append(SetEnvironmentVariable(
            "LD_LIBRARY_PATH",
            ld_extra + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        ))
        actions.append(SetEnvironmentVariable(
            "DYLD_LIBRARY_PATH",
            ld_extra + ":" + os.environ.get("DYLD_LIBRARY_PATH", ""),
        ))

        mpc_node = TimerAction(period=6.0, actions=[Node(
            package="nonlinear_mpc_acados",
            executable="mpc_node",
            name="mpc_node",
            parameters=[
                mpc_params,
                {
                    "mpc_backend": "acados",
                    # simple_mux in_topic 과 매칭: mpc → /vesc/high_level/ackermann_cmd
                    "cmd_vel_topic_name": "/vesc/high_level/ackermann_cmd",
                },
            ],
            output="screen",
        )])
        actions.append(mpc_node)

        # ── mpc_debug_logger: 매 cycle CSV (~/mpc_logs/) + 죽는 순간 자동
        # event dump (~/mpc_logs/events/event_<reason>_*.csv). 별도 노드라
        # mpc_node 동작에 영향 없음. ROS1 unicorn 의 동명 logger 포팅.
        debug_logger = TimerAction(period=6.0, actions=[Node(
            package="nonlinear_mpc_acados",
            executable="mpc_debug_logger",
            name="mpc_debug_logger",
            output="log",
        )])
        actions.append(debug_logger)

        # ── joy (수동/자동 토글). USB joystick 없으면 idle. ──
        joy_node = Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[{"deadzone": 0.05, "autorepeat_rate": 20.0}],
            output="log",
        )
        actions.append(joy_node)

        # ── auto-engage helper: mpc 가 solver codegen 끝낼 충분한 시간 (~40s)
        # 후에 joy RB(buttons[5])=1 한 번 publish → simple_mux 의 autodrive_latched
        # rising-edge 트리거. 이후엔 joy LB 로 수동 takeover, RB 로 다시 autodrive.
        auto_engage = TimerAction(period=40.0, actions=[ExecuteProcess(
            cmd=[
                "ros2", "topic", "pub", "--once", "/joy",
                "sensor_msgs/msg/Joy",
                "{header: {frame_id: 'auto_engage'}, "
                "axes: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "
                "buttons: [0, 0, 0, 0, 0, 1, 0, 0]}",
            ],
            output="log",
        )])
        actions.append(auto_engage)

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value="f"),
        DeclareLaunchArgument("racecar_version", default_value="SIM"),
        DeclareLaunchArgument("mode", default_value="timetrial",
                              description="timetrial | overtake | mpcc"),
        DeclareLaunchArgument("controller", default_value="pp_heading",
                              description="pp_heading | simple_pp | mppi (mode=mpcc 시 무시)"),
        DeclareLaunchArgument("mppi_params_file", default_value="",
                              description="mppi yaml override. 빈 문자열이면 mppi_bringup/config/params_sim_mac.yaml."),
        DeclareLaunchArgument("mppi_wpt_path", default_value="",
                              description="mppi raceline csv 절대경로. 빈 문자열이면 houston_main5.csv."),
        DeclareLaunchArgument("mppi_wall_map", default_value="",
                              description="mppi wall SDF map yaml. 빈 문자열이면 houston_main.yaml."),
        # DeclareLaunchArgument("n_obstacles", default_value="auto",
        #                       description="0=강제 비활성, auto=mode 기준, 정수=강제"),
        OpaqueFunction(function=_build),
    ])
