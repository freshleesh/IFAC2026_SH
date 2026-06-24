"""macOS 실차 HW + sensor 통합 low_level launch.

camera_lidar_mac.launch.py (camera + livox, use_rviz 옵션) +
vesc_all.launch.xml 의 vesc 4 노드를 한 번에 띄움. macOS 실차 bringup 전용.

기동 노드:
  - opencv_cam (see3cam 1280×720, headless=false 일 때만 / macOS GUI 세션 필요)
  - livox_ros_driver2 (Mid-360) + RViz (use_rviz 따라)
  - vesc_driver_node                 (시리얼 통신)
  - vesc_ackermann/ackermann_to_vesc_node  (ackermann → 모터/서보)
  - vesc_ackermann/vesc_to_odom_node       (VESC state → odom + TF)
  - joy_mac/joy_node                       (macOS GameController → /joy)
    (humandrive/autodrive 변환은 simple_mux 에서 담당)

사용:
  # 기본 (camera + livox + rviz + vesc + joy 전부)
  ros2 launch stack_master low_level_mac.launch.py

  # rviz 끄고 (sensor + HW 만)
  ros2 launch stack_master low_level_mac.launch.py use_rviz:=false

  # SSH (headless — camera skip, rviz 도 끄려면 use_rviz:=false 도 같이)
  ros2 launch stack_master low_level_mac.launch.py headless:=true use_rviz:=false

  # joy 별도 터미널에서
  ros2 launch stack_master low_level_mac.launch.py joy:=false
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _farg(name):
    """launch arg 를 float 파라미터로 강제 변환 (정수 입력도 DOUBLE 로)."""
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    # ── Args ──
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')
    camera_index = LaunchConfiguration('camera_index')
    camera_width = LaunchConfiguration('camera_width')
    camera_height = LaunchConfiguration('camera_height')
    camera_fps = LaunchConfiguration('camera_fps')
    vesc_config = LaunchConfiguration('vesc_config')
    use_joy = LaunchConfiguration('joy')
    use_cc = LaunchConfiguration('use_current_control')   # True=전류PID, False=기존 속도(eRPM)

    # ── Sensor bringup (camera + livox + optional rviz; DYLD_LIBRARY_PATH 포함) ──
    cam_lidar_share = get_package_share_directory('camera_lidar_calibration')
    sensor_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cam_lidar_share, 'launch', 'camera_lidar_mac.launch.py')
        ),
        launch_arguments={
            'headless': headless,
            'use_rviz': use_rviz,
            'camera_index': camera_index,
            'camera_width': camera_width,
            'camera_height': camera_height,
            'camera_fps': camera_fps,
        }.items(),
    )

    # ── VESC HW (시리얼 통신 + 변환 + odom) ──
    # 변환 단을 토글: use_current_control=false → ackermann_to_vesc(속도→eRPM, 기존),
    #                 true → speed_pid_to_current(속도 PID→전류, vesc_current_control).
    # 토픽이 루트(/ackermann_cmd /sensors/core /commands/motor/* /commands/servo/*)라 remap 불필요.

    # vesc_driver — 속도모드 (vesc_config 그대로: current_min=0)
    vesc_driver_speed = Node(
        package='vesc_driver', executable='vesc_driver_node', name='vesc_driver_node',
        output='screen', parameters=[vesc_config],
        condition=UnlessCondition(use_cc),
    )
    # vesc_driver — 전류모드 (회생 위해 current_min/max override)
    vesc_driver_current = Node(
        package='vesc_driver', executable='vesc_driver_node', name='vesc_driver_node',
        output='screen',
        parameters=[vesc_config, {
            'current_min': _farg('cc_driver_current_min'),
            'current_max': _farg('cc_driver_current_max'),
        }],
        condition=IfCondition(use_cc),
    )

    # 변환단 A: ackermann_to_vesc (기존 속도/eRPM)
    ackermann_to_vesc = Node(
        package='vesc_ackermann', executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node', output='screen', parameters=[vesc_config],
        condition=UnlessCondition(use_cc),
    )
    # 변환단 B: speed_pid_to_current (전류 PID) — servo 변환도 흡수
    speed_pid = Node(
        package='vesc_current_control', executable='speed_pid_to_current',
        name='speed_pid_to_current_node', output='screen',
        parameters=[{
            'speed_to_erpm_gain': 3423.0, 'speed_sign': 1.0,
            'current_sign': _farg('cc_current_sign'),
            'kp': _farg('cc_kp'), 'ki': _farg('cc_ki'), 'kd': _farg('cc_kd'),
            'current_max': _farg('cc_current_max'), 'current_min': _farg('cc_current_min'),
            'max_abs_speed': _farg('cc_max_abs_speed'),
            # servo 변환 (vesc_config 의 ackermann_to_vesc 값과 동일)
            'steering_angle_to_servo_gain': 0.5135,
            'steering_angle_to_servo_offset': 0.445,
            'servo_max': 0.85, 'servo_min': 0.15,
        }],
        condition=IfCondition(use_cc),
    )
    vesc_to_odom = Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        output='screen',
        parameters=[vesc_config],
    )

    # ── joy_mac/joy_node (macOS GameController → /joy) ──
    # NOTE: 기존 vesc_driver_mac/teleop_joy 는 제거 — simple_mux 안의
    # humandrive (LB+stick) 경로와 동일 기능을 갖고 동일 토픽 /ackermann_cmd
    # 으로 publish 해서 충돌했음. simple_mux 가 표준 mux 역할 담당.
    joy_node = Node(
        package='joy_mac',
        executable='joy_node',
        name='joy_node',
        output='screen',
        condition=IfCondition(use_joy),
    )

    # ── simple_mux: joy(LB=humandrive, RB=autodrive) mux → /ackermann_cmd ──
    # low_level 만으로도 joy 직접 조작이 되어야 하므로 여기에 둠. middle_level
    # 에서 자동주행 chain 만 추가하면 됨. autodrive input(/vesc/.../nav_1) 가
    # 없을 때(=low_level 단독)는 LB humandrive 만 동작.
    sm_share = get_package_share_directory('stack_master')
    vehicle_config = os.path.join(sm_share, 'config', 'vehicle_config.yaml')
    # joy_max_speed [m/s]. ackermann_to_vesc 에서 ×3423 → ERPM.
    # vesc_driver speed_max(46500 ERPM) 까지 풀로 쓰려면 ≤ 13.58.
    simple_mux = Node(
        package='stack_master',
        executable='simple_mux_node.py',
        name='simple_mux',
        parameters=[
            vehicle_config,
            {
                'in_topic':      '/vesc/high_level/ackermann_cmd_mux/input/nav_1',
                'out_topic':     '/ackermann_cmd',
                'joy_max_speed': 13.58,
                'use_estop':     False,
                'sim':           False,
            },
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Skip camera (macOS GUI 세션 없을 때 / SSH).'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Run RViz (livox view). headless 와 독립.'),
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('camera_width', default_value='1280'),
        DeclareLaunchArgument('camera_height', default_value='720'),
        DeclareLaunchArgument('camera_fps', default_value='60'),
        DeclareLaunchArgument(
            'vesc_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('vesc_driver_mac'), 'config', 'vesc_config.yaml',
            ]),
            description='VESC config YAML (serial port, gains, etc.)',
        ),
        DeclareLaunchArgument('joy', default_value='true',
                              description='Start joy_mac/joy_node. false 면 별도 터미널에서 실행.'),

        # ── 전류 PID 제어 토글 (vesc_current_control 필요) ──
        DeclareLaunchArgument('use_current_control', default_value='false',
                              description='true=속도 PID→전류(vesc_current_control), false=기존 속도(eRPM).'),
        # 전류모드 파라미터 (보수적 기본 — 검증 후 레이스용으로 상향). 정수 입력도 float 변환.
        DeclareLaunchArgument('cc_kp', default_value='3.0'),
        DeclareLaunchArgument('cc_ki', default_value='6.0'),
        DeclareLaunchArgument('cc_kd', default_value='0.0'),
        DeclareLaunchArgument('cc_current_max', default_value='30.0',
                              description='PID 출력 전류 상한[A]. 레이스시 상향(드라이버 캡 안쪽).'),
        DeclareLaunchArgument('cc_current_min', default_value='-20.0',
                              description='PID 출력 전류 하한[A](음수=회생).'),
        DeclareLaunchArgument('cc_driver_current_min', default_value='-55.0',
                              description='vesc_driver current_min override(회생, 펌웨어 -60 안쪽).'),
        DeclareLaunchArgument('cc_driver_current_max', default_value='90.0'),
        DeclareLaunchArgument('cc_current_sign', default_value='1.0'),
        DeclareLaunchArgument('cc_max_abs_speed', default_value='12.0',
                              description='폭주 가드[m/s]. 레이스 최고속 위로.'),

        sensor_bringup,
        vesc_driver_speed,
        vesc_driver_current,
        ackermann_to_vesc,
        speed_pid,
        vesc_to_odom,
        joy_node,
        simple_mux,
    ])
