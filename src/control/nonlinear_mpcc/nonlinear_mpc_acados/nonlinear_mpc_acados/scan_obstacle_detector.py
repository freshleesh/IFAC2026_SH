#!/usr/bin/env python3
"""LiDAR → /external_obstacles 브릿지 — 실차 장애물 인식의 빠진 고리 (2026-06-11).

실차에는 MPCC 가 구독하는 /external_obstacles (PoseArray) 를 publish 하는
노드가 없었다 (sim 은 RViz 클릭 기반 static_obstacle_manager 가 담당).
이 노드가 그 역할을 한다:

  /scan (LaserScan) + odom(map frame) + /map (OccupancyGrid)
    → 스캔점을 map frame 으로 변환
    → 정적 맵에서 벽 근처(점유+팽창) 점 제거 → 트랙 "안"의 점만 생존
    → 스캔順 클러스터링 → 지름 필터 → 클러스터 중심 = 장애물
    → PoseArray publish (빈 리스트도 매 사이클 — mpc 는 덮어쓰기 의미론)

의존: rclpy + numpy 뿐 (grid_filter/cv2/scipy 불필요) — 실차 ws 에 그대로 포팅 가능.

sim 검증 (static_obstacle_manager 와 publisher 충돌 피하려면 출력 remap):
  python3 scan_obstacle_detector.py --ros-args -p output_topic:=/detected_obstacles
실차: 기본값 그대로 실행 (output_topic=/external_obstacles).
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan


class ScanObstacleDetector(Node):
    def __init__(self):
        super().__init__('scan_obstacle_detector')

        self.declare_parameter('output_topic', '/external_obstacles')
        self.declare_parameter('odom_topic', '/car_state/odom')
        self.declare_parameter('scan_topic', '/scan')
        # base_link → LiDAR 전방 오프셋 [m] (f1tenth 표준 ~0.27)
        self.declare_parameter('scan_offset_x', 0.27)
        self.declare_parameter('max_range', 8.0)       # 이 너머는 무시
        self.declare_parameter('wall_margin', 0.20)    # 점유셀 팽창 반경 [m]
        self.declare_parameter('cluster_gap', 0.20)    # 스캔順 이웃점 분리 임계 [m]
        self.declare_parameter('min_points', 3)        # 노이즈 컷
        self.declare_parameter('max_diameter', 0.8)    # 이보다 크면 벽/차폭 가정 → 제외
        self.declare_parameter('publish_rate', 20.0)
        # 깜빡임 억제: 직전 N 프레임 중 M 프레임 이상 같은 자리(0.3m)면 채택
        self.declare_parameter('persist_frames', 3)
        self.declare_parameter('persist_min', 2)

        g = lambda n: self.get_parameter(n).value
        self._offset_x = float(g('scan_offset_x'))
        self._max_range = float(g('max_range'))
        self._wall_margin = float(g('wall_margin'))
        self._cluster_gap = float(g('cluster_gap'))
        self._min_points = int(g('min_points'))
        self._max_diam = float(g('max_diameter'))
        self._persist_frames = int(g('persist_frames'))
        self._persist_min = int(g('persist_min'))

        self._pub = self.create_publisher(PoseArray, str(g('output_topic')), 1)

        # /map 은 latched → transient_local QoS 필수
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_subscription(Odometry, str(g('odom_topic')),
                                 self._odom_cb, 1)
        self.create_subscription(LaserScan, str(g('scan_topic')),
                                 self._scan_cb, 1)

        self._free = None          # bool [H,W] — True = 트랙 안 (벽 margin 제외)
        self._map_info = None
        self._pose = None          # (x, y, yaw) in map frame
        self._recent: list[list[tuple[float, float]]] = []  # 최근 프레임 검출들
        self._last_pub_t = 0.0
        self._min_period = 1.0 / float(g('publish_rate'))
        self.get_logger().info(
            f"[detector] scan→{g('output_topic')} 대기 (margin={self._wall_margin}m, "
            f"diam≤{self._max_diam}m, persist {self._persist_min}/{self._persist_frames})")

    # ── /map: free-space 마스크 + 벽 팽창 (numpy 만으로) ────────────
    def _map_cb(self, msg: OccupancyGrid):
        # 첫 맵만 사용 — sim 의 gym_bridge 가 "장애물 포함" /map 을 1Hz 재발행
        # 하는데, 그걸 받으면 장애물이 벽으로 마스킹돼 검출 불가. 실차 맵은
        # 정적이므로 첫 수신만으로 충분.
        if self._free is not None:
            return
        h, w = msg.info.height, msg.info.width
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        occ = (grid > 50) | (grid < 0)   # 점유 or unknown → 벽 취급
        free = ~occ
        # 벽 margin 만큼 free 를 침식 = 점유를 팽창. 분리축 박스 침식(빠름·충분).
        r = max(1, int(round(self._wall_margin / msg.info.resolution)))
        f = free.copy()
        for axis in (0, 1):
            g2 = f.copy()
            for s in range(1, r + 1):
                g2[tuple(slice(s, None) if a == axis else slice(None)
                         for a in (0, 1))] &= np.take(f, range(0, f.shape[axis] - s), axis)
                g2[tuple(slice(None, -s) if a == axis else slice(None)
                         for a in (0, 1))] &= np.take(f, range(s, f.shape[axis]), axis)
            f = g2
        self._free = f
        self._map_info = msg.info
        self.get_logger().info(
            f"[detector] map {w}x{h} res={msg.info.resolution:.3f} — free mask 준비")

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (p.x, p.y, yaw)

    # ── /scan: 변환 → 필터 → 클러스터 → publish ─────────────────────
    def _scan_cb(self, msg: LaserScan):
        if self._free is None or self._pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_pub_t < self._min_period:
            return
        self._last_pub_t = now

        x0, y0, yaw = self._pose
        lx = x0 + self._offset_x * math.cos(yaw)
        ly = y0 + self._offset_x * math.sin(yaw)
        rng = np.asarray(msg.ranges, dtype=np.float64)
        ang = msg.angle_min + np.arange(rng.size) * msg.angle_increment + yaw
        ok = np.isfinite(rng) & (rng > msg.range_min) & (rng < self._max_range)
        px = lx + rng * np.cos(ang)
        py = ly + rng * np.sin(ang)

        # map free-mask 안의 점만 (벽 점은 margin 팽창에 걸려 탈락)
        info = self._map_info
        cx = ((px - info.origin.position.x) / info.resolution).astype(int)
        cy = ((py - info.origin.position.y) / info.resolution).astype(int)
        inb = ok & (cx >= 0) & (cx < info.width) & (cy >= 0) & (cy < info.height)
        keep = np.zeros_like(ok)
        keep[inb] = self._free[cy[inb], cx[inb]]

        # 스캔順 클러스터링 (인접 빔 거리 gap 으로 분할)
        centers = []
        idx = np.flatnonzero(keep)
        if idx.size:
            cuts = np.flatnonzero(
                (np.diff(idx) > 2) |
                (np.hypot(np.diff(px[idx]), np.diff(py[idx])) > self._cluster_gap))
            for seg in np.split(idx, cuts + 1):
                if seg.size < self._min_points:
                    continue
                sx, sy = px[seg], py[seg]
                diam = math.hypot(sx.max() - sx.min(), sy.max() - sy.min())
                if diam > self._max_diam:
                    continue
                # 중심 = 점 평균 + 반경 보정 (라이다는 앞면만 봄 → 빔 방향으로 반지름 밀기)
                mx, my = float(sx.mean()), float(sy.mean())
                d = math.hypot(mx - lx, my - ly)
                rr = max(0.05, diam / 2.0)
                centers.append((mx + rr * (mx - lx) / max(d, 1e-6),
                                my + rr * (my - ly) / max(d, 1e-6)))

        # 시간 일관성: 최근 persist_frames 중 persist_min 프레임에서 0.3m 내 재검출
        self._recent.append(centers)
        if len(self._recent) > self._persist_frames:
            self._recent.pop(0)
        stable = []
        for (ox, oy) in centers:
            n = sum(any(math.hypot(ox - qx, oy - qy) < 0.3 for qx, qy in fr)
                    for fr in self._recent)
            if n >= self._persist_min:
                stable.append((ox, oy))

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'map'
        for (ox, oy) in stable:
            p = Pose()
            p.position.x, p.position.y = ox, oy
            p.orientation.w = 1.0
            out.poses.append(p)
        self._pub.publish(out)   # 빈 리스트도 publish — mpc 덮어쓰기 의미론


def main(args=None):
    rclpy.init(args=args)
    node = ScanObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == '__main__':
    main()
