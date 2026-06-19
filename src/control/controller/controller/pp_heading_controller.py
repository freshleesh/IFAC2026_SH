"""PP + Friction Circle Controller.

Steering law (each tick):  δ_cmd = clip(δ_geo + δ_us + δ_damp + δ_pid, ±δ_max)

  δ_geo  — Pure-pursuit base, friction-circle limited.
             kappa_pp  = 2·ly / dist²            (curvature to lookahead point)
             kappa_max = a_total_max / vx²        (friction-circle curvature limit)
             kappa_geo = clip(kappa_pp, ±kappa_max)
             δ_geo     = atan(L · kappa_geo)      (valid at ALL speeds, incl. standstill)

  δ_us   — Understeer feedforward (tire slip in fast corners).
             δ_us = k_understeer · a_lat_geo,  a_lat_geo = vx²·kappa_geo

  δ_damp — Yaw-rate damping (optional; default off, replaced by δ_pid's Kd).
             δ_damp = −k_yaw_damp · (omega_z − vx·kappa_geo)

  δ_pid  — Heading-error PID on the lookahead bearing e_psi = atan2(ly, lx).
             δ_pid = Kp·e_psi + Ki·∫e_psi + Kd·ė_psi(filtered)
             Kp pulls the car onto the line faster; Kd damps the resulting
             overshoot. The derivative is low-pass filtered (heading_d_tau) so Kd
             is usable — raw 50 Hz de/dt is too noisy and forces Kd=0, leaving P
             alone → hairpin oscillation.

Speed law:
  a_long_budget = min(a_long_max, √(a_total_max² − (v_wp²·κ_now)²))   (raceline-based)
  v_ref         = min( raceline vx_mps, √(a_lat_corner/|κ|) )  over the braking window
  v_cmd         = min(v_ref, vx + a_long_budget·align·t_cmd_horizon)   [accel]
  v_cmd         = v_ref                                                [brake]
  align         = heading-alignment gate (no power until pointed along the path).

IFAC2026_SH port notes:
  - odom      : /car_state/odom            (IFAC convention; was /vesc/odom)
  - waypoints : /local_waypoints           (f110_msgs/WpntArray, state_machine 출력)
  - drive     : drive_topic param, default /vesc/high_level/ackermann_cmd_mux/input/nav_1
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

PARAMS = {
    # vehicle
    'wheelbase_L':     0.33,
    'delta_max':       0.41,
    'v_min':           0.5,
    'v_max':           8.0,
    'v_min_for_ik':    0.5,    # [m/s] friction-circle / cross-track speed floor
    # control rate
    'control_rate_hz': 50.0,
    # adaptive lookahead: ld = clip(k_v·vx / (1 + k_k·|κ_eff|), ld_min, ld_max)
    'ld_min':          1.2,
    'ld_max':          2.5,
    'k_v':             0.3,    # [s]      vx → lookahead gain
    'k_k':             10.0,   # [m/rad]  corner κ → lookahead reduction
    # friction circle
    'a_total_max':     6.7,    # [m/s²]  μ·g  (실측 6.68)
    'a_long_max':      2.0,    # [m/s²]  최대 가속도 (drive-wheel traction)
    # understeer feedforward: δ_us = k_understeer · a_lat
    'k_understeer':    0.010,  # [rad/(m/s²)]  0 = off
    # yaw-rate damping: δ_damp = −k · (omega_z − vx·kappa_geo)
    'k_yaw_damp':      0.0,    # [s]  0 = off (heading PID below replaces it)
    # heading-error PID on the lookahead bearing e_psi = atan2(ly, lx).
    # δ_pid = Kp·e + Ki·∫e + Kd·ė(filtered).  Kp → line convergence, Kd → damp.
    'heading_kp':      0.4,    # [-]
    'heading_ki':      0.0,    # [1/s]
    'heading_kd':      0.05,   # [s]
    'heading_i_max':   0.2,    # [rad]  integral clamp (anti-windup)
    'heading_d_tau':   0.05,   # [s]    derivative low-pass time constant
    # speed
    'a_brake_max':     3.5,    # [m/s²]  제동거리 윈도우 크기
    't_cmd_horizon':   0.3,    # [s]     가속 명령 horizon
    'a_lat_corner':    2.0,    # [m/s²]  곡률 속도 캡: v ≤ √(a_lat_corner/|κ|)
    # heading-alignment acceleration gate
    'yaw_gate_min':    0.05,   # [rad]  ≤ this: full accel
    'yaw_gate_max':    0.40,   # [rad]  ≥ this: no accel
}

# /pp_debug Float64MultiArray field indices
_D_A_LAT_PP   = 0   # [m/s²]  PP lateral demand  (vx²·κ_pp, before clip)
_D_A_LAT_GEO  = 1   # [m/s²]  friction-circle-clipped a_lat
_D_DELTA_GEO  = 2   # [deg]   pure-pursuit base steering
_D_DELTA_US   = 3   # [deg]   understeer feedforward
_D_DELTA_DAMP = 4   # [deg]   yaw-rate damping
_D_DELTA_PID  = 5   # [deg]   heading-error PID trim
_D_E_PSI      = 6   # [rad]   lookahead bearing (heading error)
_D_LD         = 7   # [m]     adaptive lookahead distance
_D_V_REF      = 8   # [m/s]   velocity reference (after curvature cap)
_D_V_CMD      = 9   # [m/s]   commanded speed
_D_DELTA_CMD  = 10  # [deg]   total commanded steering
_D_YAW_ERR    = 11  # [rad]   heading misalignment (yaw − path tangent)
_D_ALIGN      = 12  # [-]     accel gate factor (0=blocked, 1=full)
_DEBUG_LEN    = 13

_N_KAPPA_AVG = 5


class PPHeadingController(Node):

    def __init__(self):
        super().__init__('pp_heading_controller')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        # drive topic (IFAC mux input by default; string param so declared separately)
        self.declare_parameter(
            'drive_topic', '/vesc/high_level/ackermann_cmd_mux/input/nav_1')
        drive_topic = str(self.get_parameter('drive_topic').value)

        self.wheelbase     = float(p('wheelbase_L'))
        self.delta_max     = float(p('delta_max'))
        self.v_min         = float(p('v_min'))
        self.v_max         = float(p('v_max'))
        self.v_min_for_ik  = float(p('v_min_for_ik'))
        self._dt           = 1.0 / float(p('control_rate_hz'))
        self.ld_min        = float(p('ld_min'))
        self.ld_max        = float(p('ld_max'))
        self.k_v           = float(p('k_v'))
        self.k_k           = float(p('k_k'))
        self.a_total_max   = float(p('a_total_max'))
        self.a_long_max    = float(p('a_long_max'))
        self.k_understeer  = float(p('k_understeer'))
        self.k_yaw_damp    = float(p('k_yaw_damp'))
        self.heading_kp    = float(p('heading_kp'))
        self.heading_ki    = float(p('heading_ki'))
        self.heading_kd    = float(p('heading_kd'))
        self.heading_i_max = float(p('heading_i_max'))
        self.heading_d_tau = float(p('heading_d_tau'))
        self.a_brake_max   = float(p('a_brake_max'))
        self.t_cmd_horizon = float(p('t_cmd_horizon'))
        self.a_lat_corner  = float(p('a_lat_corner'))
        self.yaw_gate_min  = float(p('yaw_gate_min'))
        self.yaw_gate_max  = float(p('yaw_gate_max'))

        self.odom         = None
        self.waypoints    = []
        self._nearest_idx = None
        self._last_pos    = None
        self._he_int      = 0.0    # heading PID integral
        self._he_prev     = None   # previous e_psi (None = first tick)
        self._he_deriv    = 0.0    # filtered derivative state

        # cached numpy arrays — rebuilt only when waypoints change
        self._wx      = None
        self._wy      = None
        self._s_vals  = None
        self._psi_wp  = None
        self._N       = 0
        self._s_total = 0.0

        self.create_subscription(Odometry,  '/car_state/odom',  self._odom_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self._wp_cb,   10)
        self.drive_pub     = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)
        self.debug_pub     = self.create_publisher(Float64MultiArray, '/pp_debug', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(self._dt, self._loop)

        self.get_logger().info(
            f'[PP] drive_topic={drive_topic}  '
            f'k_v={self.k_v}  k_k={self.k_k}  ld=[{self.ld_min},{self.ld_max}]  '
            f'a_total_max={self.a_total_max}  a_long_max={self.a_long_max}  '
            f'a_lat_corner={self.a_lat_corner}  '
            f'k_us={self.k_understeer}  k_yaw_damp={self.k_yaw_damp}  '
            f'Kp={self.heading_kp}  Ki={self.heading_ki}  Kd={self.heading_kd}  '
            f'd_tau={self.heading_d_tau}  dt={self._dt*1000:.1f} ms'
        )

    def _odom_cb(self, msg): self.odom = msg

    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts
        self._wx      = np.array([wp.x_m     for wp in self.waypoints])
        self._wy      = np.array([wp.y_m     for wp in self.waypoints])
        self._s_vals  = np.array([wp.s_m     for wp in self.waypoints])
        self._psi_wp  = np.array([wp.psi_rad for wp in self.waypoints])
        self._N       = len(self.waypoints)
        self._s_total = float(self._s_vals[-1]) if self._N > 0 else 0.0
        self._nearest_idx = None
        self._he_int   = 0.0       # reset PID state on path change
        self._he_prev  = None
        self._he_deriv = 0.0

    # ── arc-length forward walk ──────────────────────────────────────────────

    def _advance(self, start: int, dist: float,
                 wx, wy, s_vals, s_total: float, N: int) -> int:
        """Index ~dist metres ahead of start along the path (forward only)."""
        acc = 0.0
        i   = start
        for _ in range(N - 1):
            j   = (i + 1) % N
            seg = (s_vals[j] - s_vals[i]) % s_total
            if seg <= 1e-6:
                seg = math.hypot(wx[j] - wx[i], wy[j] - wy[i])
            acc += seg
            if acc >= dist:
                return j
            i = j
        return i

    # ── main loop ────────────────────────────────────────────────────────────

    def _loop(self):
        if self.odom is None or self._wx is None or self._N == 0:
            return

        vx      = abs(self.odom.twist.twist.linear.x)
        omega_z = float(self.odom.twist.twist.angular.z)
        p_x = self.odom.pose.pose.position.x
        p_y = self.odom.pose.pose.position.y
        q   = self.odom.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        wx      = self._wx
        wy      = self._wy
        s_vals  = self._s_vals
        N       = self._N
        s_total = self._s_total

        # ── 1. Nearest waypoint ──────────────────────────────────────────────
        if self._last_pos is not None:
            dx = p_x - self._last_pos[0]
            dy = p_y - self._last_pos[1]
            if dx * dx + dy * dy > 0.25:    # teleport > 0.5 m → reset
                self._nearest_idx = None
        self._last_pos = (p_x, p_y)

        if self._nearest_idx is None:
            d2      = (wx - p_x) ** 2 + (wy - p_y) ** 2
            aligned = np.cos(yaw - self._psi_wp) > 0.0
            if np.any(aligned):
                d2 = np.where(aligned, d2, np.inf)
            nearest_idx = int(np.argmin(d2))
        else:
            cands       = [(self._nearest_idx + k) % N for k in range(30)]
            nearest_idx = min(cands, key=lambda i: (wx[i] - p_x) ** 2 + (wy[i] - p_y) ** 2)
        self._nearest_idx = nearest_idx

        # ── 2. Adaptive lookahead ────────────────────────────────────────────
        ld_base = self.k_v * vx

        now_idxs   = [(nearest_idx + i) % N for i in range(_N_KAPPA_AVG)]
        kappa_now  = float(np.mean([abs(self.waypoints[i].kappa_radpm)
                                    for i in now_idxs]))

        corner_idx   = self._advance(nearest_idx, max(ld_base, self.ld_min),
                                     wx, wy, s_vals, s_total, N)
        corner_idxs  = [(corner_idx + i) % N for i in range(_N_KAPPA_AVG)]
        kappa_corner = float(np.mean([abs(self.waypoints[i].kappa_radpm)
                                      for i in corner_idxs]))

        # larger of the two: ld stays short both ON approach AND inside the corner
        kappa_eff = max(kappa_now, kappa_corner)
        ld = float(np.clip(
            ld_base / (1.0 + self.k_k * kappa_eff),
            self.ld_min, self.ld_max,
        ))

        # ── 3. Lookahead point ───────────────────────────────────────────────
        psi_n    = float(self.waypoints[nearest_idx].psi_rad)
        tnx, tny = math.cos(psi_n), math.sin(psi_n)
        extra    = ((p_x - float(wx[nearest_idx])) * tnx
                  + (p_y - float(wy[nearest_idx])) * tny)
        target_idx = self._advance(nearest_idx, max(0.0, ld + extra),
                                   wx, wy, s_vals, s_total, N)

        lx_w = float(wx[target_idx])
        ly_w = float(wy[target_idx])

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dtx  =  lx_w - p_x
        dty  =  ly_w - p_y
        lx_v =  dtx * cos_y + dty * sin_y
        ly_v = -dtx * sin_y + dty * cos_y
        dist = math.hypot(lx_v, ly_v)

        if dist < 1e-6:
            self._he_int   = 0.0
            self._he_prev  = None
            self._he_deriv = 0.0
            self._publish_drive(0.0, self.v_min)
            return

        # ── 4a. Pure-pursuit base, friction-circle limited ───────────────────
        # δ_geo = atan(L·kappa) computed directly so steering is valid at ALL
        # speeds (incl. standstill). Friction limit applied as a max curvature
        # kappa_max = a_total_max/vx² (only binds at high speed).
        kappa_pp  = 2.0 * ly_v / (dist * dist)
        v_safe    = max(vx, self.v_min_for_ik)
        kappa_max = self.a_total_max / (v_safe * v_safe)
        kappa_geo = float(np.clip(kappa_pp, -kappa_max, kappa_max))
        delta_geo = math.atan(self.wheelbase * kappa_geo)
        a_lat_pp  = vx * vx * kappa_pp
        a_lat_geo = vx * vx * kappa_geo

        # ── 4b. Understeer feedforward ───────────────────────────────────────
        delta_us = self.k_understeer * a_lat_geo

        # ── 4c. Yaw-rate damping (deviation from path-intended yaw rate) ──────
        yaw_rate_des = vx * kappa_geo
        delta_damp   = -self.k_yaw_damp * (omega_z - yaw_rate_des)

        # ── 4d. Heading-error PID (on the lookahead bearing) ─────────────────
        # e_psi = atan2(ly, lx) = angle between the car heading and the vector to
        # the lookahead point. δ_geo already steers toward it; the PID trims the
        # residual: Kp → faster convergence to the line, Kd → damps the resulting
        # overshoot/oscillation. The derivative is LOW-PASS FILTERED (d_tau) so
        # Kd is actually usable — raw 50 Hz de/dt is too noisy and forces Kd=0,
        # which leaves P alone → the hairpin oscillation seen earlier.
        e_psi = math.atan2(ly_v, lx_v)
        if self._he_prev is None:
            de_raw = 0.0
        else:
            de_raw = (e_psi - self._he_prev) / self._dt
        self._he_prev = e_psi
        # 1-pole low-pass on the derivative
        a_d = self._dt / (self.heading_d_tau + self._dt) if self.heading_d_tau > 0.0 else 1.0
        self._he_deriv += a_d * (de_raw - self._he_deriv)

        if vx >= self.v_min_for_ik:
            self._he_int += e_psi * self._dt
        else:
            self._he_int = 0.0
        i_term = float(np.clip(self.heading_ki * self._he_int,
                               -self.heading_i_max, self.heading_i_max))
        if self.heading_ki > 1e-9:
            self._he_int = i_term / self.heading_ki      # anti-windup back-calc

        delta_pid = (self.heading_kp * e_psi
                     + i_term
                     + self.heading_kd * self._he_deriv)

        # ── 4e. Combined steering ────────────────────────────────────────────
        delta_cmd = float(np.clip(delta_geo + delta_us + delta_damp + delta_pid,
                                  -self.delta_max, self.delta_max))

        # ── 5. Speed ─────────────────────────────────────────────────────────
        # Longitudinal accel budget from the raceline at the current point
        # (v_near matched to kappa_now → self-consistent, ≤ a_total_max).
        v_near        = float(self.waypoints[nearest_idx].vx_mps)
        a_lat_ref     = v_near * v_near * kappa_now
        a_long_budget = min(self.a_long_max,
                            math.sqrt(max(0.0, self.a_total_max ** 2 - a_lat_ref ** 2)))

        # Anticipatory v_ref: min over the braking window of (raceline speed,
        # curvature-capped speed √(a_lat_corner/|κ|)).
        s_nearest = float(s_vals[nearest_idx])
        s_ahead   = (s_vals - s_nearest) % s_total
        v_lhd     = vx ** 2 / (2.0 * self.a_brake_max) + self.ld_min
        v_win     = np.where(s_ahead <= v_lhd)[0]
        if len(v_win) > 0:
            v_ref = float(min(self.waypoints[i].vx_mps for i in v_win))
            kappa_win   = np.array([abs(float(self.waypoints[i].kappa_radpm))
                                    for i in v_win])
            v_curve_cap = float(np.min(np.sqrt(self.a_lat_corner
                                               / np.maximum(kappa_win, 1e-3))))
            v_ref = min(v_ref, v_curve_cap)
        else:
            v_ref = float(self.waypoints[target_idx].vx_mps)
        v_ref = max(v_ref, self.v_min)

        # Heading-alignment acceleration gate: no power until pointed along path.
        yaw_err = math.atan2(math.sin(yaw - psi_n), math.cos(yaw - psi_n))
        if self.yaw_gate_max > self.yaw_gate_min:
            align = (self.yaw_gate_max - abs(yaw_err)) / (self.yaw_gate_max - self.yaw_gate_min)
        else:
            align = 1.0 if abs(yaw_err) <= self.yaw_gate_max else 0.0
        align = float(np.clip(align, 0.0, 1.0))

        if v_ref > vx:
            v_cmd = min(v_ref, vx + a_long_budget * align * self.t_cmd_horizon)
        else:
            v_cmd = v_ref                            # braking: never gated
        v_cmd = float(np.clip(v_cmd, self.v_min, self.v_max))

        # ── publish ──────────────────────────────────────────────────────────
        self._publish_drive(delta_cmd, v_cmd)
        self._publish_lookahead(lx_w, ly_w, ld)
        self._publish_debug(a_lat_pp, a_lat_geo, delta_geo, delta_us, delta_damp,
                            delta_pid, e_psi, ld, v_ref, v_cmd, delta_cmd,
                            yaw_err, align)

    # ── publishers ───────────────────────────────────────────────────────────

    def _publish_drive(self, delta: float, speed: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.drive.steering_angle = delta
        msg.drive.speed          = speed
        self.drive_pub.publish(msg)

    def _publish_lookahead(self, x: float, y: float, ld: float) -> None:
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns, m.id        = 'pp_lookahead', 0
        m.type, m.action  = Marker.SPHERE, Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        s = float(np.clip(ld * 0.15, 0.1, 0.4))
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 1.0
        self.lookahead_pub.publish(m)

    def _publish_debug(self, a_lat_pp, a_lat_geo, delta_geo, delta_us, delta_damp,
                       delta_pid, e_psi, ld, v_ref, v_cmd, delta_cmd, yaw_err, align):
        d = [0.0] * _DEBUG_LEN
        d[_D_A_LAT_PP]   = a_lat_pp
        d[_D_A_LAT_GEO]  = a_lat_geo
        d[_D_DELTA_GEO]  = math.degrees(delta_geo)
        d[_D_DELTA_US]   = math.degrees(delta_us)
        d[_D_DELTA_DAMP] = math.degrees(delta_damp)
        d[_D_DELTA_PID]  = math.degrees(delta_pid)
        d[_D_E_PSI]      = e_psi
        d[_D_LD]         = ld
        d[_D_V_REF]      = v_ref
        d[_D_V_CMD]      = v_cmd
        d[_D_DELTA_CMD]  = math.degrees(delta_cmd)
        d[_D_YAW_ERR]    = yaw_err
        d[_D_ALIGN]      = align
        msg = Float64MultiArray()
        msg.data = d
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PPHeadingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
