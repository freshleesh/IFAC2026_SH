#!/usr/bin/env python3
"""ftg_fallback_node — independent reactive Follow-The-Gap controller.

Always-on safety net: MPCC 가 죽거나 멈춰도 이 노드는 /scan 받아서 안전한
gap-follow 명령을 자체 토픽 `/vesc/ftg_fallback` 으로 20Hz publish.

simple_mux 가 watchdog 으로 MPCC cmd 의 staleness 감지 → 끊기면 자동으로
이 fallback 토픽으로 switch. MPCC 가 복귀하면 다시 switch back.

레이싱 안전성: MPCC solver 가 가끔 터지거나 (acados 한계) infeasible 빠지면
차가 그냥 멈춰서 다른 차에 받힘. fallback 항상 살아있으면 즉시 reactive 로
빠져나옴.

설계:
  - /scan 구독 (sensor_msgs/LaserScan, BEST_EFFORT)
  - /vesc/ftg_fallback publish (ackermann_msgs/AckermannDriveStamped, 20Hz timer)
  - controller.ftg.FTG 재사용 (이미 검증된 구현체)
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

from controller.ftg import FTG


class FtgFallbackNode(Node):
    def __init__(self):
        super().__init__('ftg_fallback')
        self.declare_parameter('scan_topic',     '/scan')
        self.declare_parameter('cmd_topic',      '/vesc/ftg_fallback')
        self.declare_parameter('rate_hz',        20.0)
        # FTG 자체의 안전속도 cap (MPCC 보다 보수적이어야 — fallback 용).
        # 0 < x ≤ 4 권장. 높이면 race speed 도 나오지만 위험 ↑.
        self.declare_parameter('max_speed',      2.5)

        scan_topic = str(self.get_parameter('scan_topic').value)
        cmd_topic  = str(self.get_parameter('cmd_topic').value)
        rate_hz    = float(self.get_parameter('rate_hz').value)
        max_speed  = float(self.get_parameter('max_speed').value)

        # FTG 재사용 — node=None 으로 default 값 사용 + viz pub 비활성 (FTG class 의
        # _get_param_or_default 가 controller_manager 전용 helper 라 여기서 못 쓰니까).
        self.ftg = FTG(mapping=False, node=None)
        # 2026-05-28 #14: ftg.py 의 default range_offset=0.0 (FLOAT) → _preprocess_lidar 의
        # ranges[0.0:-0.0] 에서 TypeError "slice indices must be integers". 명시적으로 int set.
        # F1Tenth gym scan_size 1080 (270°/0.25°), 30 = 7.5° trim (앞 90° 정상 view 보존).
        self.ftg.range_offset = 30
        # 2026-05-28 #16: SAFETY_RADIUS class default 0.3 (FLOAT, "m" 단위 의도) 도 ftg.py 의
        # `range(self.SAFETY_RADIUS)` (frame count 자리) 에서 TypeError. 5 = ~1.25° lidar bin.
        self.ftg.SAFETY_RADIUS = 5
        # FTG 의 MAX_SPEED 를 param 으로 override (class attr 가 SAFE cap 결정)
        self.ftg.MAX_SPEED = max_speed
        # scale 도 보수적으로 (corner 더 천천히)
        self.ftg.CORNERS_SPEED        = 0.35 * max_speed * 0.65
        self.ftg.MILD_CORNERS_SPEED   = 0.50 * max_speed * 0.65
        self.ftg.STRAIGHTS_SPEED      = 0.85 * max_speed * 0.65
        self.ftg.ULTRASTRAIGHTS_SPEED = 1.00 * max_speed * 0.65

        self._last_scan = None
        scan_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, scan_qos)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)
        self.create_timer(1.0 / rate_hz, self._tick)

        # Startup grace — 처음 N 초 동안은 zero cmd. spawn 위치가 트랙 밖이거나
        # MPCC codegen 중인 동안 FTG 가 이상한 scan 보고 풀스티어 내는 거 방지.
        import time as _t
        self._start_t = _t.monotonic()
        self.startup_grace_sec = 5.0

        self.get_logger().info(
            f'[ftg_fallback] up. scan={scan_topic} cmd={cmd_topic} '
            f'rate={rate_hz:.0f}Hz max_speed={max_speed:.1f} '
            f'startup_grace={self.startup_grace_sec:.1f}s')

    def _scan_cb(self, msg: LaserScan):
        self._last_scan = msg

    def _tick(self):
        if self._last_scan is None:
            return
        # Startup grace: 처음 N 초 동안은 zero cmd 만 publish (스폰 위치 weirdness
        # + MPCC codegen 중 FTG 의 풀스티어 발생 방지).
        import time as _t
        elapsed = _t.monotonic() - self._start_t
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        if elapsed < self.startup_grace_sec:
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
            self.cmd_pub.publish(msg)
            return
        try:
            speed, steer = self.ftg.process_lidar(self._last_scan.ranges)
        except Exception as e:
            self.get_logger().warn(f'[ftg_fallback] FTG raised: {e}', throttle_duration_sec=2.0)
            return
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FtgFallbackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
