#!/usr/bin/env python3
"""MPC debug logger — separate ROS2 node, no changes to main MPC node.

Subscribes to `/mpc_debug` (Float32MultiArray with DBG_FIELDS), `/mpc_trajectory`
(Path, predicted N-step trajectory), `/mpc/solve_time`, `/mpc/cost`,
`/mpc/is_feasible`, and `/boundary_marker`. Per-cycle:

  1. dumps every cycle to CSV at ~/mpc_logs/mpc_<timestamp>.csv
  2. detects anomalies (infeasible/cost_spike/vcmd_jerk/stuck/slow_solve) and
     auto-writes pre/post event-window CSVs into ~/mpc_logs/events/
  3. detects lap rollover (current_s big-to-small jump), increments lap counter,
     re-publishes the *driven* per-lap path on /mpc/driven_lap<N>
  4. publishes /mpc/lap_count (Int32, latched) and /mpc/lateral_margin_min (Float64)
  5. prints a 1-Hz one-line summary

Ported from ROS1 `evo_mpcc/nonlinear_mpc_casadi/scripts/mpc_debug_logger.py`.
ROS1 → ROS2 changes: rospy → rclpy.Node, latched pubs → TRANSIENT_LOCAL QoS,
timer callbacks lose the event arg.
"""
from __future__ import annotations

import os
import csv
import time
import collections

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import Float32MultiArray, Float64, Bool, Int32
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray


# Must match Nonlinear_MPC_node.DBG_FIELDS order (ROS1 source of truth).
# C-1 추가: kappa_abs/signed + B 의 q_*_scale (16~21) — MLP 학습 데이터 용도.
DBG_FIELDS = [
    "v_cmd", "steer_cmd", "v_actual",
    "car_x", "car_y", "car_yaw",
    "current_s", "near_idx", "ref_v",
    "n_obs_in", "sel_dmin", "sel_x", "sel_y",
    "side_pref", "opti_value", "solve_ms",
    # C-1 MLP 학습용 추가 필드
    "kappa_abs", "kappa_signed",
    "q_cte_scale", "q_lag_scale", "q_v_scale", "q_drate_scale",
    "v_max_cost",  # mpc.v_max (cost target). auto_step 이 동적 변경.
]

LAP_JUMP_THRESHOLD = 5.0  # current_s big-to-small drop triggers lap +1


class MPCDebugLogger(Node):
    def __init__(self):
        super().__init__('mpc_debug_logger')

        # ── State ──────────────────────────────────────────────────
        self.lap = 0
        self.last_s: float | None = None
        self.last_dbg: dict | None = None
        self.last_pred_path: Path | None = None
        self.last_solve_ms = float("nan")
        self.last_cost = float("nan")
        self.last_feasible = True
        self.last_mpcc_active = False
        self.last_switch_count = 0
        self.last_boundary: list[tuple[float, float]] | None = None
        # True simulator velocity states from /car_state/odom (twist).
        # GP-residual experiment: prefer these over finite-diff estimates.
        self.last_vx_odom = float("nan")
        self.last_vy_odom = float("nan")
        self.last_r_odom = float("nan")
        self.driven_paths: dict[int, list[tuple[float, float]]] = {0: []}
        self.start_time = time.time()
        self.row_count = 0

        # ── Anomaly detection (auto event dump) ────────────────────
        self.ring_before: collections.deque = collections.deque(maxlen=20)
        self.event_capture_after = 15
        self.active_event: dict | None = None
        self._last_event_time: dict[str, float] = {}
        self._event_throttle_s = 2.0
        self.events_dir = os.path.expanduser("~/mpc_logs/events")
        os.makedirs(self.events_dir, exist_ok=True)
        self.event_count = 0

        # ── CSV ────────────────────────────────────────────────────
        log_dir = os.path.expanduser("~/mpc_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"mpc_{ts}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        header = ["t", "lap"] + DBG_FIELDS + [
            "feasible", "min_lateral_margin",
            "pred_dx_n0", "pred_dy_n0",
            "pred_x_end", "pred_y_end",
            "mpcc_active", "switch_count",
            "vx_odom", "vy_odom", "r_odom",
        ]
        self.csv.writerow(header)
        self.csv_file.flush()
        self.get_logger().info(f"[mpc_debug_logger] CSV → {self.csv_path}")

        # ── Publishers (latched analogue: TRANSIENT_LOCAL) ─────────
        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.lap_pubs: dict[int, object] = {}
        self.lap_count_pub = self.create_publisher(Int32, "/mpc/lap_count", latched)
        self.lateral_margin_pub = self.create_publisher(Float64, "/mpc/lateral_margin_min", 10)

        # ── Subscriptions ──────────────────────────────────────────
        self.create_subscription(Float32MultiArray, "/mpc_debug", self.cb_dbg, 1)
        self.create_subscription(Path, "/mpc_trajectory", self.cb_pred, 1)
        self.create_subscription(Float64, "/mpc/solve_time", self.cb_solve, 1)
        self.create_subscription(Float64, "/mpc/cost", self.cb_cost, 1)
        self.create_subscription(Bool, "/mpc/is_feasible", self.cb_feasible, 1)
        self.create_subscription(MarkerArray, "/boundary_marker", self.cb_boundary, 1)
        # True velocity states (sim ground truth) for GP-residual clean targets.
        self.create_subscription(Odometry, "/car_state/odom", self.cb_odom, 1)
        # mux 가 매 cycle publish — last 값만 잡아두고 CSV 매 row 에 같이 기록.
        self.create_subscription(Bool,  "/mux/mpcc_active",  self.cb_mpcc_active,  10)
        self.create_subscription(Int32, "/mux/switch_count", self.cb_switch_count, 10)

        # 1Hz status tick
        self.create_timer(1.0, self._tick)
        self._latched_qos = latched  # cached for per-lap publishers

    # ── callbacks ─────────────────────────────────────────────────
    def cb_dbg(self, msg: Float32MultiArray):
        if len(msg.data) < len(DBG_FIELDS):
            return
        d = dict(zip(DBG_FIELDS, msg.data))
        self.last_dbg = d

        s = float(d.get("current_s", 0.0))
        if self.last_s is not None and (self.last_s - s) > LAP_JUMP_THRESHOLD:
            now_t = time.time()
            prev_lap_t = getattr(self, '_lap_start_t', self.start_time)
            lap_time_sec = now_t - prev_lap_t
            self.lap += 1
            self._lap_start_t = now_t
            self.driven_paths[self.lap] = []
            self.get_logger().info(
                f"[mpc_debug_logger] >>> LAP {self.lap} started — "
                f"이전 lap 시간: {lap_time_sec:.2f}s "
                f"(s rollover {self.last_s:.1f} -> {s:.1f})")
            self.lap_count_pub.publish(Int32(data=self.lap))
            # /mpc/lap_time 발행 (Float64)
            if not hasattr(self, 'lap_time_pub'):
                self.lap_time_pub = self.create_publisher(Float64, '/mpc/lap_time', 10)
            self.lap_time_pub.publish(Float64(data=lap_time_sec))
        self.last_s = s

        x, y = float(d.get("car_x", 0.0)), float(d.get("car_y", 0.0))
        self.driven_paths.setdefault(self.lap, []).append((x, y))

        margin = self._compute_margin(x, y)
        if margin is not None:
            self.lateral_margin_pub.publish(Float64(data=margin))

        pred_dx0 = pred_dy0 = pred_xe = pred_ye = float("nan")
        if self.last_pred_path is not None and len(self.last_pred_path.poses) >= 2:
            p0 = self.last_pred_path.poses[0].pose.position
            p1 = self.last_pred_path.poses[1].pose.position
            pe = self.last_pred_path.poses[-1].pose.position
            pred_dx0, pred_dy0 = p1.x - p0.x, p1.y - p0.y
            pred_xe, pred_ye = pe.x, pe.y
        row = [time.time() - self.start_time, self.lap] + \
              [d.get(k, 0.0) for k in DBG_FIELDS] + \
              [int(self.last_feasible),
               margin if margin is not None else float("nan"),
               pred_dx0, pred_dy0, pred_xe, pred_ye,
               int(self.last_mpcc_active), int(self.last_switch_count),
               self.last_vx_odom, self.last_vy_odom, self.last_r_odom]
        self.csv.writerow(row)
        self.row_count += 1
        if self.row_count % 50 == 0:
            self.csv_file.flush()

        self.ring_before.append(row)
        reason = self._detect_anomaly(d)
        if reason is not None and self._can_fire_event(reason):
            self._open_event_dump(reason, row)
        if self.active_event is not None:
            self.active_event['writer'].writerow(row)
            self.active_event['remaining'] -= 1
            if self.active_event['remaining'] <= 0:
                self._close_event_dump()

        self._publish_lap_path(self.lap)

    def cb_pred(self, msg: Path):
        self.last_pred_path = msg

    def cb_solve(self, msg: Float64):
        self.last_solve_ms = float(msg.data)

    def cb_cost(self, msg: Float64):
        self.last_cost = float(msg.data)

    def cb_feasible(self, msg: Bool):
        self.last_feasible = bool(msg.data)

    def cb_mpcc_active(self, msg: Bool):
        self.last_mpcc_active = bool(msg.data)

    def cb_switch_count(self, msg: Int32):
        self.last_switch_count = int(msg.data)

    def cb_odom(self, msg: Odometry):
        # Ground-truth velocity states from the simulator (body frame twist).
        self.last_vx_odom = float(msg.twist.twist.linear.x)
        self.last_vy_odom = float(msg.twist.twist.linear.y)
        self.last_r_odom = float(msg.twist.twist.angular.z)

    def cb_boundary(self, msg: MarkerArray):
        pts = []
        for mk in msg.markers:
            for p in mk.points:
                pts.append((p.x, p.y))
        self.last_boundary = pts

    # ── anomaly detection ─────────────────────────────────────────
    def _detect_anomaly(self, d):
        """Return a short reason string if this cycle is anomalous, else None."""
        if not self.last_feasible:
            return "infeasible"
        if self.last_solve_ms == self.last_solve_ms and self.last_solve_ms > 100.0:
            return "slow_solve"
        if len(self.ring_before) >= 3:
            past = list(self.ring_before)[-3:]
            idx_v_cmd = 2 + 0
            idx_opti  = 2 + 14
            past_costs = [r[idx_opti] for r in past]
            past_med = float(np.median(past_costs)) if past_costs else 0.0
            cur_cost = float(d.get('opti_value', 0.0))
            if past_med > 1.0 and cur_cost > 10.0 * past_med + 100.0:
                return "cost_spike"
            past_v = [r[idx_v_cmd] for r in past]
            if past_v and abs(float(d.get('v_cmd', 0.0)) - past_v[-1]) > 2.0:
                return "vcmd_jerk"
        v_actual = float(d.get('v_actual', 0.0))
        v_cmd = float(d.get('v_cmd', 0.0))
        if v_actual < 0.3 and v_cmd > 1.0:
            return "stuck"
        return None

    def _can_fire_event(self, reason):
        last = self._last_event_time.get(reason, 0.0)
        now = time.time()
        if self.active_event is not None:
            return False
        if (now - last) < self._event_throttle_s:
            return False
        self._last_event_time[reason] = now
        return True

    def _open_event_dump(self, reason, trigger_row):
        ts = time.strftime("%H%M%S")
        self.event_count += 1
        path = os.path.join(self.events_dir,
                            f"event_{ts}_{reason}_{self.event_count:03d}.csv")
        f = open(path, "w", newline="")
        w = csv.writer(f)
        header = ["t", "lap"] + DBG_FIELDS + [
            "feasible", "min_lateral_margin",
            "pred_dx_n0", "pred_dy_n0", "pred_x_end", "pred_y_end",
            "trigger",
        ]
        w.writerow(header)
        for r in self.ring_before:
            w.writerow(list(r) + [""])
        self.active_event = {
            'file': f, 'writer': w,
            'remaining': self.event_capture_after,
            'reason': reason, 'path': path,
        }
        self.get_logger().warn(
            f"[mpc_debug_logger] >>> EVENT {reason} @ t={trigger_row[0]:.1f} → {path}")

    def _close_event_dump(self):
        if self.active_event is None:
            return
        try:
            self.active_event['file'].flush()
            self.active_event['file'].close()
        except Exception:
            pass
        self.get_logger().info(
            f"[mpc_debug_logger] event dump closed: {self.active_event['path']}")
        self.active_event = None

    # ── helpers ───────────────────────────────────────────────────
    def _compute_margin(self, x, y):
        if not self.last_boundary:
            return None
        dx = np.array([p[0] - x for p in self.last_boundary])
        dy = np.array([p[1] - y for p in self.last_boundary])
        d = np.hypot(dx, dy)
        return float(np.min(d)) if len(d) else None

    def _publish_lap_path(self, lap):
        if lap not in self.lap_pubs:
            self.lap_pubs[lap] = self.create_publisher(
                Path, f"/mpc/driven_lap{lap}", self._latched_qos)
            self.get_logger().info(
                f"[mpc_debug_logger] publishing /mpc/driven_lap{lap}")
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.driven_paths[lap]:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.lap_pubs[lap].publish(path)

    def _tick(self):
        if self.last_dbg is None:
            self.get_logger().info("[dbg] (no /mpc_debug yet)")
            return
        d = self.last_dbg
        margin = self._compute_margin(d.get("car_x", 0), d.get("car_y", 0))
        margin_str = f"{margin:.3f}m" if margin is not None else "n/a"
        self.get_logger().info(
            f"[dbg] lap={self.lap}  s={float(d.get('current_s', 0)):6.2f}  "
            f"v={float(d.get('v_actual', 0)):4.2f}  "
            f"vcmd={float(d.get('v_cmd', 0)):4.2f}  "
            f"steer={float(d.get('steer_cmd', 0)):+5.3f}  "
            f"solve={self.last_solve_ms:5.1f}ms  cost={self.last_cost:8.2f}  "
            f"feas={'Y' if self.last_feasible else 'N'}  margin={margin_str}")

    def _shutdown(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(
                f"[mpc_debug_logger] CSV closed ({self.row_count} rows) at {self.csv_path}")
        except Exception:
            pass
        self._close_event_dump()
        self.get_logger().info(
            f"[mpc_debug_logger] event dumps written: {self.event_count} in {self.events_dir}")


def main():
    rclpy.init()
    node = MPCDebugLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
