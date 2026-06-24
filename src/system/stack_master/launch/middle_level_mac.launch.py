"""fast_livo localization + PP 자동주행 통합 launch (middle level).

map argument 하나로 모든 데이터 로드 — 기본값 my_seat. stack_master/maps/<map>/
디렉터리에 prior_map(.pcd/.fmap) + global_waypoints.json 이 같이 들어있는 구조.

low_level_mac 포함, 한 줄로 모든 체인 기동:
  low_level_mac (vesc/livox/cam/joy)
    └─ /livox/lidar, /odom, /joy
  fast_livo localization (mid360_localization.yaml, prior_map_dir=<map>)
    └─ /aft_mapped_to_init  →remap→  /car_state/odom (~10Hz)
  static TF camera_init→map (identity; fast_livo frame ↔ race-stack frame)
  global_republisher (<map>/global_waypoints.json)
    └─ /global_waypoints + /global_waypoints/vel_markers_tuned
  frenet_odom_republisher (cartesian ↔ frenet)
    └─ /car_state/odom_frenet
  fake_topic_relay (state_machine 의존 토픽 stub)
    └─ /global_waypoints_scaled, /global_waypoints/overtaking, /car_state/pose 등
  state_machine (timetrial_only + force_GBTRACK)
    └─ /local_waypoints, /behavior_strategy
  controller_manager (L1 = pure pursuit)
    └─ /vesc/high_level/ackermann_cmd_mux/input/nav_1
  simple_mux (joy mux: RB=autodrive, LB=humandrive)
    └─ /ackermann_cmd (out_topic) → ackermann_to_vesc → motor/servo
  rviz (relocalization.rviz, global path 포함, 옵션)

전제:
  - sb 로 워크스페이스 source (DYLD_LIBRARY_PATH 채워짐)
  - cb 후 ros_fix_install 동작 (conda env/lib symlink, rpath, xattr strip)
  - VESC USB 연결, livox/camera 정상
  - <map> prior map (cloudGlobal.pcd, prior_map.fmap, global_waypoints.json)
    가 stack_master/maps/<map>/ 에 존재

사용:
  ros2 launch stack_master middle_level_mac.launch.py
  ros2 launch stack_master middle_level_mac.launch.py map:=<other_map>
  ros2 launch stack_master middle_level_mac.launch.py use_low_level:=false   # low_level_mac 따로
  ros2 launch stack_master middle_level_mac.launch.py use_rviz_livo:=false   # rviz 없이

조작:
  - RB (right bumper): autodrive ON — controller 명령이 motor/servo 로 흐름
  - LB (left bumper) + sticks: humandrive — joystick 직접 조종
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Args ───────────────────────────────────────────────────────────
    map_arg          = DeclareLaunchArgument("map", default_value="f")
    low_level_arg    = DeclareLaunchArgument("use_low_level", default_value="true",
                                             description="low_level_mac (vesc/livox/cam/joy) 포함 여부")
    rviz_low_arg     = DeclareLaunchArgument("use_rviz_low_level", default_value="false")
    rviz_livo_arg    = DeclareLaunchArgument("use_rviz_livo", default_value="true")
    joy_arg          = DeclareLaunchArgument("use_joy", default_value="true")
    use_cc_arg       = DeclareLaunchArgument("use_current_control", default_value="false",
                                             description="low_level_mac 으로 forward: 전류 PID 제어")
    cc_kp_arg        = DeclareLaunchArgument("cc_kp", default_value="3.0")
    cc_ki_arg        = DeclareLaunchArgument("cc_ki", default_value="6.0")
    cc_kd_arg        = DeclareLaunchArgument("cc_kd", default_value="0.0")
    cc_imax_arg      = DeclareLaunchArgument("cc_current_max", default_value="30.0")
    cc_imin_arg      = DeclareLaunchArgument("cc_current_min", default_value="-20.0")
    cc_dimin_arg     = DeclareLaunchArgument("cc_driver_current_min", default_value="-55.0")
    cc_dimax_arg     = DeclareLaunchArgument("cc_driver_current_max", default_value="90.0")
    cc_ssign_arg     = DeclareLaunchArgument("cc_speed_sign", default_value="1.0")
    cc_csign_arg     = DeclareLaunchArgument("cc_current_sign", default_value="1.0")
    cc_mabs_arg      = DeclareLaunchArgument("cc_max_abs_speed", default_value="12.0")
    controller_arg   = DeclareLaunchArgument("controller", default_value="simple_pp",
                                             description="simple_pp | mppi — drop-in 교체")
    mppi_params_arg  = DeclareLaunchArgument("mppi_params_file", default_value="",
                                             description="mppi yaml. 빈 문자열이면 params_real_mac.yaml.")
    mppi_wpt_arg     = DeclareLaunchArgument("mppi_wpt_path", default_value="",
                                             description="mppi raceline csv 절대경로. 빈 문자열이면 houston_main5.csv (실차 첫 통합용 기본 — 실제 사용시 map 에 맞게 override).")
    mppi_wall_arg    = DeclareLaunchArgument("mppi_wall_map", default_value="",
                                             description="mppi wall SDF map yaml. 빈 문자열이면 SDF off.")

    map_name        = LaunchConfiguration("map")
    use_low_level   = LaunchConfiguration("use_low_level")
    use_rviz_low    = LaunchConfiguration("use_rviz_low_level")
    use_rviz_livo   = LaunchConfiguration("use_rviz_livo")
    use_joy         = LaunchConfiguration("use_joy")
    use_cc          = LaunchConfiguration("use_current_control")

    # ── Resolved paths ─────────────────────────────────────────────────
    sm_share        = get_package_share_directory("stack_master")
    livo_share      = get_package_share_directory("fast_livo")
    controller_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "sim_controller_params.yaml"
    )
    livo_main       = os.path.join(livo_share, "config", "mid360_localization.yaml")
    livo_cam        = os.path.join(livo_share, "config", "camera_see3cam.yaml")
    livo_rviz_cfg   = os.path.join(livo_share, "rviz_cfg", "relocalization.rviz")
    # Canonical map storage — stack_master/maps/<name>/ (src side, writable).
    # cloudGlobal.pcd / prior_map.fmap / <name>.yaml / global_waypoints.json /
    # raceline.csv 모두 같은 디렉토리. `map:=<name>` 하나로 localization +
    # global_repub + mppi raceline + wall sdf 다 매칭.
    src_maps_root = "/Users/mini/ros2_ws/src/IFAC2026_SH/src/system/stack_master/maps"

    # ── 1. low_level_mac (vesc + livox + camera + joy + DYLD_LIBRARY_PATH) ──
    low_level = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sm_share, "launch", "low_level_mac.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz_low,
            "joy": use_joy,
            "use_current_control":   use_cc,
            "cc_kp":                 LaunchConfiguration("cc_kp"),
            "cc_ki":                 LaunchConfiguration("cc_ki"),
            "cc_kd":                 LaunchConfiguration("cc_kd"),
            "cc_current_max":        LaunchConfiguration("cc_current_max"),
            "cc_current_min":        LaunchConfiguration("cc_current_min"),
            "cc_driver_current_min": LaunchConfiguration("cc_driver_current_min"),
            "cc_driver_current_max": LaunchConfiguration("cc_driver_current_max"),
            "cc_speed_sign":         LaunchConfiguration("cc_speed_sign"),
            "cc_current_sign":       LaunchConfiguration("cc_current_sign"),
            "cc_max_abs_speed":      LaunchConfiguration("cc_max_abs_speed"),
        }.items(),
        condition=IfCondition(use_low_level),
    )

    # ── 2. fast_livo localization (prior_map_dir 은 `map` 인자로 override) ─
    # /aft_mapped_to_init → /car_state/odom remap 으로 frenet/state_machine 체인 진입.
    from launch.substitutions import PathJoinSubstitution
    livo_prior_map_dir = PathJoinSubstitution([src_maps_root, map_name])
    livo_node = Node(
        package="fast_livo",
        executable="fastlivo_mapping",
        name="laserMapping",
        parameters=[
            livo_main,
            livo_cam,
            {"relocalization.prior_map_dir": livo_prior_map_dir},
        ],
        remappings=[("/aft_mapped_to_init", "/car_state/odom")],
        output="screen",
    )
    livo_rviz = Node(
        condition=IfCondition(use_rviz_livo),
        package="rviz2",
        executable="rviz2",
        name="rviz2_livo",
        arguments=["-d", livo_rviz_cfg],
        output="screen",
    )

    # ── 2b. global init (KISS-Matcher) ─────────────────────────────────
    # auto_start=false: trigger 대기. zsh alias `pose` 로 호출하면
    # ~/livox/lidar 누적 → KISS-Matcher → /initialpose 발행 → fast_livo 초기화.
    gi_share = get_package_share_directory("fast_livo_global_init")
    gi_yaml = os.path.join(gi_share, "config", "global_init.yaml")
    gi_pcd_path = PathJoinSubstitution([src_maps_root, map_name, "cloudGlobal.pcd"])
    global_init_node = Node(
        package="fast_livo_global_init",
        executable="global_init_node",
        name="fast_livo_global_init",
        parameters=[
            gi_yaml,
            {
                "prior_map.pcd_path": gi_pcd_path,
                "control.auto_start": False,
            },
        ],
        output="screen",
    )

    # ── 3. static TF camera_init → map (identity) ──────────────────────
    # fast_livo 는 world/camera_init frame, race-stack 의 path 마커는 map frame.
    # 둘을 동일 frame 으로 묶어주기 위한 static transform.
    static_tf_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_camera_init_to_map",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "camera_init", "--child-frame-id", "map",
        ],
        output="log",
    )

    # ── 4. global_republisher: <src>/maps/<map>/global_waypoints.json ──
    # PathJoinSubstitution lets us use the launch-arg `map` at runtime so
    # `ros2 launch ... map:=garage` just works without rebuilding.
    from launch.substitutions import PathJoinSubstitution
    waypoints_path = PathJoinSubstitution([src_maps_root, map_name, "global_waypoints.json"])
    global_repub = Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[{
            "map": map_name,
            "map_path": waypoints_path,
        }],
        output="screen",
    )

    # ── 5. frenet_odom_republisher (TimerAction: localization init 후) ─
    frenet_odom_repub = TimerAction(period=4.0, actions=[Node(
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

    # ── 6. fake_topic_relay (state_machine init 직전에 stub 발행 시작) ──
    fake_relay = TimerAction(period=5.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])

    # ── 7. state_machine (timetrial_only + force_GBTRACK) ──────────────
    sm_node = TimerAction(period=6.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            "racecar_version": "SIM",
            "map": map_name,
            "state_machine.rate": 50.0,
            "state_machine.n_loc_wpnts": 80,
            "state_machine.ot_planner": "",
            "state_machine.timetrials_only": True,
            "state_machine.force_GBTRACK": True,
            "state_machine.gb_ego_width_m": 0.3,
            "state_machine.gb_horizon_m": 5.0,
            "state_machine.lateral_width_gb_m": 0.3,
            "state_machine.interest_horizon_m": 20.0,
            "state_machine.use_force_trailing": False,
            "state_machine.splini_ttl": 5.0,
            "state_machine.pred_splini_ttl": 0.2,
            "state_machine.overtaking_horizon_m": 6.9,
            "state_machine.lateral_width_ot_m": 0.3,
            "state_machine.splini_hyst_timer_sec": 3.0,
            "state_machine.emergency_break_horizon": 1.1,
            "state_machine.ftg_speed_mps": 1.0,
            "state_machine.ftg_timer_sec": 3.0,
            "state_machine.ftg_active": False,
            "state_machine.overtaking_ttl_sec": 10.0,
            "state_machine.volt_threshold": 10.0,
            "measure": False,
            "sim": False,
        }],
        output="screen",
    )])

    # ── 8. controller 분기 (simple_pp | mppi) ─────────────────────────
    # simple_pp = 순수 pure-pursuit (vx_mps 그대로). mppi = JAX 샘플링.
    # 둘 다 인터페이스 동일 (/car_state/odom in, /vesc/.../nav_1 out).
    def _pick_controller(context: LaunchContext, *_args, **_kwargs):
        which = LaunchConfiguration("controller").perform(context)
        if which == "simple_pp":
            return [TimerAction(period=7.0, actions=[Node(
                package="controller",
                executable="simple_pp",
                name="simple_pp",
                parameters=[{
                    "lookahead_distance": 1.2,
                    "wheelbase": 0.33,
                    "max_steering_rad": 0.4,
                    "max_speed_mps": 8.0,
                    "speed_scale": 1.0,
                    "control_rate_hz": 50.0,
                    "drive_topic": "/vesc/high_level/ackermann_cmd_mux/input/nav_1",
                }],
                output="screen",
            )])]
        if which == "mppi":
            mppi_share = get_package_share_directory("mppi_bringup")
            sm_share_local = get_package_share_directory("stack_master")
            map_str = LaunchConfiguration("map").perform(context)
            params_path = LaunchConfiguration("mppi_params_file").perform(context) or \
                os.path.join(mppi_share, "config", "params_real_mac.yaml")
            wpt_path = LaunchConfiguration("mppi_wpt_path").perform(context)
            if not wpt_path:
                wpt_path = os.path.join(sm_share_local, "maps", map_str, "raceline.csv")
                if not os.path.exists(wpt_path):
                    raise FileNotFoundError(
                        f"mppi raceline 자동 매칭 실패: {wpt_path} 없음. "
                        f"mppi_wpt_path:=<csv 경로> 로 명시하거나 "
                        f"stack_master/maps/{map_str}/raceline.csv 생성 필요."
                    )
            wall_map = LaunchConfiguration("mppi_wall_map").perform(context)
            if not wall_map:
                wall_map = os.path.join(sm_share_local, "maps", map_str, f"{map_str}.yaml")
                if not os.path.exists(wall_map):
                    wall_map = ""
            overrides = {
                "pose_topic": "/car_state/odom",
                "drive_topic": "/vesc/high_level/ackermann_cmd_mux/input/nav_1",
                "wpt_path_absolute": True,
                "wpt_path": wpt_path,
            }
            if wall_map:
                overrides["wall_cost_map_yaml"] = wall_map
            else:
                # 빈 문자열이면 SDF off (yaml 의 wall_cost_enabled=true 인 경우
                # 안전을 위해 명시적으로 disable).
                overrides["wall_cost_enabled"] = False
                overrides["wall_cost_map_yaml"] = ""
            return [TimerAction(period=7.0, actions=[Node(
                package="mppi_example",
                executable="mppi_node",
                name="lmppi_node",
                output="log",
                parameters=[params_path, overrides],
            )])]
        raise ValueError(f"controller must be 'simple_pp' or 'mppi', got {which!r}")

    controller_branch = OpaqueFunction(function=_pick_controller)

    # simple_mux 는 low_level_mac 에서 띄움 (low 만으로도 joy 직접 조작 가능하도록).
    # middle_level 은 controller 명령(/vesc/.../nav_1) 을 추가하기만 함.

    return LaunchDescription([
        map_arg, low_level_arg, rviz_low_arg, rviz_livo_arg, joy_arg,
        controller_arg, mppi_params_arg, mppi_wpt_arg, mppi_wall_arg,
        use_cc_arg, cc_kp_arg, cc_ki_arg, cc_kd_arg, cc_imax_arg, cc_imin_arg,
        cc_dimin_arg, cc_dimax_arg, cc_ssign_arg, cc_csign_arg, cc_mabs_arg,
        low_level,
        livo_node, livo_rviz,
        global_init_node,
        static_tf_map,
        global_repub,
        frenet_odom_repub,
        fake_relay,
        sm_node,
        controller_branch,
    ])
