#!/usr/bin/env python3
"""pp_fallback_node — Pure-Pursuit fallback for MPCC.

Always-on safety net (FTG 의 reactive 보다 더 안정적인 fallback).
MPCC 가 죽거나 stale 하면 simple_mux 가 자동으로 이 토픽으로 switch.
PP 는 centerline lookahead point 를 향해 bicycle 모델 curvature 로 steer 계산
→ 항상 centerline 따라가서 충돌/in_collision lock 회피.

설계:
  - /centerline_waypoints (WpntArray) 구독 — wpnt list 캐싱 (1Hz)
  - /car_state/odom (Odometry) 구독 — 차 pose (50Hz+)
  - /vesc/pp_fallback (AckermannDriveStamped) publish — 20Hz timer
  - PP: nearest wpnt → lookahead_dist 앞 wpnt → world→vehicle frame →
    curvature κ = 2y/L² → steer = atan(wheelbase · κ)
  - speed: 보수적 cap (max_speed param), straight 에선 더 빠름
"""
from __future__ import annotations
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray


class PpFallbackNode(Node):
    def __init__(self):
        super().__init__('pp_fallback')
        self.declare_parameter('wpnts_topic',   '/centerline_waypoints')
        self.declare_parameter('odom_topic',    '/car_state/odom')
        self.declare_parameter('cmd_topic',     '/vesc/pp_fallback')
        self.declare_parameter('rate_hz',       20.0)
        self.declare_parameter('max_speed',     4.0)        # PP launch cap (was 2.5) — faster start
        self.declare_parameter('lookahead',     1.5)        # meters
        self.declare_parameter('wheelbase',     0.307)
        self.declare_parameter('s_max',         0.4)        # steer abs cap

        self.wpnts_topic = str(self.get_parameter('wpnts_topic').value)
        self.odom_topic  = str(self.get_parameter('odom_topic').value)
        self.cmd_topic   = str(self.get_parameter('cmd_topic').value)
        self.rate_hz     = float(self.get_parameter('rate_hz').value)
        self.max_speed   = float(self.get_parameter('max_speed').value)
        self.lookahead   = float(self.get_parameter('lookahead').value)
        self.wheelbase   = float(self.get_parameter('wheelbase').value)
        self.s_max       = float(self.get_parameter('s_max').value)

        self._wpnts_xy = None    # (N, 2) numpy
        self._wpnts_s  = None    # (N,) cumulative arc length
        self._last_pose = None   # (x, y, yaw)

        qos_be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(WpntArray, self.wpnts_topic, self._wpnts_cb, 10)
        self.create_subscription(Odometry,  self.odom_topic,  self._odom_cb,  qos_be)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        # Startup grace — 처음 N 초 동안 zero cmd (sim spawn 위치 weirdness 방지).
        import time as _t
        self._start_t = _t.monotonic()
        self.startup_grace_sec = float(self.declare_parameter('startup_grace_sec', 1.0).value)  # was 3.0 — faster launch (벽 안박을 만큼)

        self.get_logger().info(
            f'[pp_fallback] up. wpnts={self.wpnts_topic} odom={self.odom_topic} '
            f'cmd={self.cmd_topic} rate={self.rate_hz:.0f}Hz '
            f'max_speed={self.max_speed:.1f} lookahead={self.lookahead:.2f}m')

    def _wpnts_cb(self, msg: WpntArray):
        if not msg.wpnts:
            return
        xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=float)
        # cumulative arc length (close the loop for nearest search)
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(d)))
        self._wpnts_xy = xy
        self._wpnts_s = s

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion → yaw (assume z-up, no roll/pitch)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self._last_pose = (float(p.x), float(p.y), yaw)

    def _tick(self):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        # Grace period — zero cmd
        import time as _t
        if _t.monotonic() - self._start_t < self.startup_grace_sec:
            self.cmd_pub.publish(cmd)
            return

        if self._wpnts_xy is None or self._last_pose is None:
            self.cmd_pub.publish(cmd)
            return

        try:
            cx, cy, cyaw = self._last_pose
            xy = self._wpnts_xy
            # nearest wpnt
            d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
            i_near = int(np.argmin(d2))
            # lookahead idx — walk forward by lookahead distance
            s = self._wpnts_s
            target_s = s[i_near] + self.lookahead
            # wrap (closed track)
            if target_s > s[-1]:
                target_s -= s[-1]
            # find idx by s
            i_la = int(np.searchsorted(s, target_s) % len(s))
            lx, ly = float(xy[i_la, 0]), float(xy[i_la, 1])

            # world → vehicle frame
            dx_w = lx - cx
            dy_w = ly - cy
            cos_y, sin_y = math.cos(-cyaw), math.sin(-cyaw)
            dx_v = cos_y * dx_w - sin_y * dy_w
            dy_v = sin_y * dx_w + cos_y * dy_w

            Ld_sq = dx_v * dx_v + dy_v * dy_v
            if Ld_sq < 0.01 or dx_v < 0:
                # too close or behind — slow heading P-fallback (target yaw = next wpnt tangent)
                cmd.drive.speed = 0.5
                cmd.drive.steering_angle = 0.0
                self.cmd_pub.publish(cmd)
                return

            curvature = 2.0 * dy_v / Ld_sq
            steer = math.atan2(self.wheelbase * curvature, 1.0)
            steer = max(-self.s_max, min(self.s_max, steer))

            # speed: straight (|κ| small) 일수록 빠르게, 코너에선 보수적.
            kappa_abs = abs(curvature)
            if kappa_abs < 0.1:
                speed = self.max_speed              # 직선
            elif kappa_abs < 0.3:
                speed = self.max_speed * 0.75       # mild corner
            else:
                speed = self.max_speed * 0.5        # hairpin

            cmd.drive.speed = float(speed)
            cmd.drive.steering_angle = float(steer)
            self.cmd_pub.publish(cmd)
        except Exception as e:
            self.get_logger().warn(f'[pp_fallback] PP raised: {e}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PpFallbackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
