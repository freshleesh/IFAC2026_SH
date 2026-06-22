"""Track data loader — CSV waypoints → CasADi spline LUTs.

Ported from `Nonlinear_MPC_node.py::preprocess_track_data` (ROS1).
ROS-free: pure numpy + casadi. The ROS2 wrapper hands the resulting
`TrackData` straight into `mpc.set_track_data(...)`.

Per-track CSV layout (all in one directory `<track_dir>/`):
    track<NAME>_centerline_waypoints.csv   x, y, ref_v   (ref_v scaled by vel_scale)
    track<NAME>_center_derivates.csv       dx, dy
    track<NAME>_right_waypoints.csv        x, y
    track<NAME>_left_waypoints.csv         x, y

Loop closure: the centerline / boundaries are duplicated by `extend_part`
so the spline LUT covers `s ∈ [0, L * (1 + 1/extend_part)]`. This gives
the MPC's prediction horizon something to query at the lap-rollover seam.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from casadi import interpolant


@dataclass
class TrackData:
    """All artefacts produced by `load_track`. One per track, immutable per run."""
    center_lane: np.ndarray              # (N_total, 2) — extended for loop overlap
    element_arc_lengths: np.ndarray      # (N_total,) cumulative s on extended grid
    element_arc_lengths_orig: np.ndarray # (N_orig,)  cumulative s on raw centerline
    center_lut_x: Any
    center_lut_y: Any
    center_lut_dx: Any
    center_lut_dy: Any
    right_lut_x: Any
    right_lut_y: Any
    left_lut_x: Any
    left_lut_y: Any
    lut_ref_v: Any
    ref_v: np.ndarray = field(default=None)  # raw speed profile (only from build_track_from_wpnts)
    center_point_angles: np.ndarray = field(default=None)
    # Raw (un-inflated, un-extended) boundary lanes for RViz viz publishers
    # /center_path /right_path /left_path. center_lane (above) is extended
    # for spline lookup; these are the original ROS1 layout (N_orig × 2).
    raw_center_lane: np.ndarray = field(default=None)
    raw_right_lane: np.ndarray = field(default=None)
    raw_left_lane: np.ndarray = field(default=None)


def _read_csv_xy(path: str, with_v: bool = False, vel_scale: float = 1.0):
    """CSV → np.ndarray. `with_v=True` returns (xy[N,2], v[N]) for centerline file."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"track CSV not found: {path}")
    with open(path) as f:
        rows = [tuple(line) for line in csv.reader(f, delimiter=',') if line]
    if with_v:
        xy = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=float)
        v = np.array([float(r[2]) * vel_scale for r in rows], dtype=float)
        return xy, v
    return np.array([[float(r[0]), float(r[1])] for r in rows], dtype=float)


def _arc_lengths(waypoints: np.ndarray) -> np.ndarray:
    d = np.diff(waypoints, axis=0)
    consecutive = np.sqrt(np.sum(d ** 2, axis=1))
    return np.insert(np.cumsum(consecutive), 0, 0.0)


def _inflate_boundary(center_lane: np.ndarray, side_lane: np.ndarray,
                      inflation_factor: float) -> np.ndarray:
    """Pull each side waypoint inward by inflation_factor * 0.8 m."""
    inflated = side_lane.copy()
    for idx in range(len(center_lane)):
        v = inflated[idx, :] - center_lane[idx, :]
        d = np.linalg.norm(v)
        if d < 1e-9:
            continue
        unit = v / d
        inflated[idx, :] -= unit * inflation_factor * 0.8
    return inflated


def _bspline_xy(label_x: str, label_y: str, pts: np.ndarray, arc_lengths: np.ndarray):
    return (interpolant(label_x, 'bspline', [arc_lengths], pts[:, 0]),
            interpolant(label_y, 'bspline', [arc_lengths], pts[:, 1]))


def _bspline_v(label: str, v_col: np.ndarray, arc_lengths: np.ndarray):
    return interpolant(label, 'bspline', [arc_lengths], v_col[:, 0])


def _build_track(center_lane: np.ndarray, center_deriv: np.ndarray,
                 right_lane: np.ndarray, left_lane: np.ndarray, ref_v: np.ndarray,
                 inflation_factor: float, extend_part: int) -> TrackData:
    """Shared geometry → spline LUT pipeline. CSV and wpnt-msg loaders both
    funnel through here so the resulting `TrackData` is identical-shape."""
    # Stash raw (pre-inflation) boundaries for RViz viz publishers.
    raw_center_lane = center_lane.copy()
    raw_right_lane = right_lane.copy()
    raw_left_lane = left_lane.copy()
    right_lane = _inflate_boundary(center_lane, right_lane, inflation_factor)
    left_lane = _inflate_boundary(center_lane, left_lane, inflation_factor)

    n_orig = center_lane.shape[0]
    overlap = max(1, int(n_orig / extend_part))
    cl_ext = np.row_stack((center_lane, center_lane[1:overlap, :]))
    rl_ext = np.row_stack((right_lane, right_lane[1:overlap, :]))
    ll_ext = np.row_stack((left_lane, left_lane[1:overlap, :]))
    cd_ext = np.row_stack((center_deriv, center_deriv[1:overlap, :]))

    end_v = max(1, int(ref_v.shape[0] / extend_part))
    ref_v_ext = np.row_stack((ref_v[:, np.newaxis], ref_v[1:end_v, np.newaxis]))

    s_orig = _arc_lengths(center_lane)
    s_ext = _arc_lengths(cl_ext)

    center_x_lut, center_y_lut = _bspline_xy('lut_center_x', 'lut_center_y', cl_ext, s_ext)
    center_dx_lut, center_dy_lut = _bspline_xy('lut_center_dx', 'lut_center_dy', cd_ext, s_ext)
    right_x_lut, right_y_lut = _bspline_xy('lut_right_x', 'lut_right_y', rl_ext, s_ext)
    left_x_lut, left_y_lut = _bspline_xy('lut_left_x', 'lut_left_y', ll_ext, s_ext)
    ref_v_lut = _bspline_v('lut_ref_v', ref_v_ext, s_ext)

    return TrackData(
        center_lane=cl_ext,
        element_arc_lengths=s_ext,
        element_arc_lengths_orig=s_orig,
        center_lut_x=center_x_lut, center_lut_y=center_y_lut,
        center_lut_dx=center_dx_lut, center_lut_dy=center_dy_lut,
        right_lut_x=right_x_lut, right_lut_y=right_y_lut,
        left_lut_x=left_x_lut, left_lut_y=left_y_lut,
        lut_ref_v=ref_v_lut,
        center_point_angles=np.arctan2(center_deriv[:, 1], center_deriv[:, 0]),
        raw_center_lane=raw_center_lane,
        raw_right_lane=raw_right_lane,
        raw_left_lane=raw_left_lane,
    )


def load_track(track_dir: str, track_name: str,
               vel_scale: float = 1.0,
               inflation_factor: float = 1.2,
               extend_part: int = 2) -> TrackData:
    """Load a track from `<track_dir>/track<track_name>/track<track_name>_*.csv`."""
    sub = os.path.join(track_dir, f"track{track_name}")
    base = os.path.join(sub, f"track{track_name}")

    center_lane, ref_v = _read_csv_xy(base + '_centerline_waypoints.csv',
                                      with_v=True, vel_scale=vel_scale)
    center_deriv = _read_csv_xy(base + '_center_derivates.csv')
    right_lane = _read_csv_xy(base + '_right_waypoints.csv')
    left_lane = _read_csv_xy(base + '_left_waypoints.csv')

    return _build_track(center_lane, center_deriv, right_lane, left_lane, ref_v,
                        inflation_factor=inflation_factor, extend_part=extend_part)


def corridor_speed_cap(width: float, v_full: float, v_floor: float = 0.0,
                       w_tight: float = 1.0, w_wide: float = 1.6) -> float:
    """Speed cap from corridor width (complements the curvature cap).

    A narrow-but-gently-curving section has low |κ|, so √(a_lat/|κ|) lets the
    car run ~v_max and clip the wall. This caps speed by available width:
    wide corridor → `v_full`; narrow → ramped linearly down to `v_floor`
    between `w_wide` and `w_tight`. `v_floor<=0` (or w_wide<=w_tight) disables
    it (returns `v_full`), so it is opt-in and a no-op by default.
    """
    if v_floor <= 0.0 or w_wide <= w_tight:
        return v_full
    if width >= w_wide:
        return v_full
    if width <= w_tight:
        return v_floor
    frac = (width - w_tight) / (w_wide - w_tight)
    return v_floor + (v_full - v_floor) * frac


def build_track_from_wpnts(wpnts, vel_scale: float = 1.0,
                           inflation_factor: float = 1.2,
                           extend_part: int = 2,
                           default_v: float = 5.0,
                           corridor_half_width: float = 0.0,
                           a_lat_max: float = 6.0,
                           a_long_max: float = 3.0,
                           corridor_v_floor: float = 0.0,
                           corridor_v_tight: float = 1.0,
                           corridor_v_wide: float = 1.6) -> TrackData:
    """Same TrackData as `load_track`, but built from a list of `f110_msgs/Wpnt`.

    Designed for race-stack `/centerline_waypoints` ingestion: wpnt fields
    `x_m, y_m, psi_centerline_rad, d_left, d_right, vx_mps` carry exactly the
    info the ROS1 CSV pipeline (centerline + derivative + L/R boundaries +
    ref_v) provides. `default_v` is used as fallback when wpnt.vx_mps is ≤ 0
    (centerline-only wpnts may not carry a velocity profile).
    """
    n = len(wpnts)
    if n < 4:
        raise ValueError(f"need ≥4 centerline wpnts, got {n}")

    center_lane = np.empty((n, 2), dtype=float)
    center_deriv = np.empty((n, 2), dtype=float)
    right_lane = np.empty((n, 2), dtype=float)
    left_lane = np.empty((n, 2), dtype=float)
    ref_v = np.empty(n, dtype=float)

    # `/centerline_waypoints` carries the centerline tangent in `psi_rad`
    # (the wpnts ARE the centerline). `psi_centerline_rad` is only filled
    # on raceline wpnts as a side-channel for d_left/d_right corridor
    # conversion — here it would be 0 and collapse all boundaries onto the
    # ±y axis. Fall back to psi_rad when psi_centerline_rad is unset.
    # ref_v: prefer wpnt.vx_mps; else κ-aware cap √(a_lat_max/|κ|), then
    # default_v cap. centerline-only wpnts ship vx=0, so this branch is
    # what's actually exercised under `/centerline_waypoints`.
    # corridor_half_width > 0: fixed-width MPC corridor (centerline ± half).
    #   mpc lateral search space cap, 좌우 대칭. 코너에서 raw d_left/d_right
    #   가 망가지지 않게 함. inflation 은 우회됨 (사용자 설정값이 곧 cap).
    use_fixed_corridor = corridor_half_width > 1e-3
    for i, w in enumerate(wpnts):
        x, y = w.x_m, w.y_m
        psi_c = getattr(w, 'psi_centerline_rad', 0.0)  # optional field; absent on car's Wpnt
        psi = psi_c if abs(psi_c) > 1e-9 else w.psi_rad
        c, s = np.cos(psi), np.sin(psi)
        center_lane[i] = (x, y)
        center_deriv[i] = (c, s)
        if use_fixed_corridor:
            d_r = d_l = corridor_half_width
        else:
            # Corridor bounds the car CENTER, but d_left/d_right are distances
            # to the track walls. Subtract a safety margin (car half-width
            # ~0.16 + buffer) on each side so the car body doesn't clip the
            # wall — without it the raceline (which hugs the inside edge at
            # apexes) leaves zero room and the car wedges (STUCK storm).
            _WALL_MARGIN = 0.18   # 2026-06-02: 0.25→0.18 (final 좁은 트랙: d=0.36m서 0.25면 corridor 0.11m→QP infeasible→fails폭증. car 반폭~0.15 기준 0.18이 최소안전)
            d_r = max(0.05, float(w.d_right) - _WALL_MARGIN)
            d_l = max(0.05, float(w.d_left) - _WALL_MARGIN)
        # right normal = (+sin(psi), -cos(psi)); left normal = (-sin(psi), +cos(psi))
        # NOTE: d_left/d_right are measured along the CENTERLINE normal, so this
        # projection is exact ONLY when psi == centerline tangent. Line 220
        # prefers psi_centerline_rad (the centerline tangent) for exactly this
        # reason. The psi_rad fallback (wpnts lacking centerline psi, e.g. a raw
        # raceline) would place the boundary off by 1/cos(Δψ) at points where the
        # wpnt heading differs from the centerline — feed centerline psi to avoid.
        right_lane[i] = (x + d_r * s,  y - d_r * c)
        left_lane[i]  = (x - d_l * s,  y + d_l * c)
        # Speed reference: cap to BOTH our top speed (default_v = v_max) AND the
        # κ-aware lateral-grip limit √(a_lat_max/|κ|). The IQP raceline ships
        # vx_mps optimized for a higher-grip/faster car (7-13 m/s here); used
        # raw (previous behaviour) it never slowed for apexes at our v=5 → the
        # car entered geometric apexes too hot → wedge (STUCK storm in raceline
        # mode). Taking the min slows it at high-curvature points per OUR limits
        # while still following the raceline GEOMETRY. Because the raceline is
        # min-curvature, its |κ| at corners is lower than the centerline's, so
        # the κ-cap permits MORE corner speed than centerline → faster lap.
        kappa = abs(float(w.kappa_radpm))
        v_kappa = np.sqrt(a_lat_max / kappa) if kappa > 1e-3 else default_v
        v_msg = float(w.vx_mps) * vel_scale
        v_src = v_msg if v_msg > 1e-3 else default_v
        # Corridor-width cap: the κ-cap is blind to width, so a narrow-but-
        # straight section (low |κ|) otherwise runs at v_max and clips the wall
        # (final-map s≈60 crash). d_r/d_l here are already wall-margin-subtracted,
        # so (d_r+d_l) is the usable corridor the car center must stay within.
        v_corr = corridor_speed_cap(d_r + d_l, default_v, corridor_v_floor,
                                    corridor_v_tight, corridor_v_wide)
        ref_v[i] = float(np.clip(min(v_src, v_kappa, default_v, v_corr),
                                 1.0, default_v))

    # ── Forward-backward velocity profile (TUM calc_vel_profile style) ──
    # The pointwise κ-cap above sets the APEX speed but does NOT tell the car to
    # brake BEFORE the apex. At v=5 with a real braking limit the car then
    # entered apexes too hot (late braking) → wedge in raceline mode. Propagate
    # the speed limits along the loop: a backward pass enforces "slow enough to
    # brake into the next point", a forward pass enforces "don't exceed what
    # acceleration out of the previous point allows". Result: a feasible profile
    # that decelerates AHEAD of corners. Loop-closed (the track is a circuit).
    # Applied only in the raceline/raw-corridor path (use_fixed_corridor=False);
    # C1 (2026-06-17): the BACKWARD (braking) pass now runs in fixed-corridor
    # mode too. Horizon is short (N=20·dT=0.04 = 0.8s ≈ 3m); without pre-corner
    # deceleration baked into ref_v the solver can't brake in time at high speed
    # (6→2.5 needs ~5m > horizon), which forced the external brake_anticip hack.
    # The pass only ever LOWERS ref_v (min), uses a_long = |A_MIN_DYN| (solver-
    # matched), so it can never make corner entry hotter — strictly safer.
    # The FORWARD (accel) pass stays raceline-only: the car accelerates far
    # faster than a_long(=3.0) (dyn_a_max=7.5), so capping exit ramp at 3.0
    # would needlessly throttle corner exits.
    if n >= 4:
        a_long = float(a_long_max)
        ds = np.empty(n, dtype=float)
        for i in range(n):
            j = (i + 1) % n
            ds[i] = float(np.hypot(center_lane[j, 0] - center_lane[i, 0],
                                   center_lane[j, 1] - center_lane[i, 1]))
        for _ in range(2):   # 2 wrap sweeps to converge across the start seam
            for i in range(n - 1, -1, -1):          # backward: braking (always)
                j = (i + 1) % n
                v_brake = float(np.sqrt(ref_v[j] ** 2 + 2.0 * a_long * ds[i]))
                if v_brake < ref_v[i]:
                    ref_v[i] = v_brake
            if not use_fixed_corridor:               # forward: accel (raceline only)
                for i in range(n):
                    k = (i - 1) % n
                    v_acc = float(np.sqrt(ref_v[k] ** 2 + 2.0 * a_long * ds[k]))
                    if v_acc < ref_v[i]:
                        ref_v[i] = v_acc
        ref_v = np.clip(ref_v, 1.0, default_v)

    # Fixed-width corridor already represents the mpc cap → skip inflation
    # (otherwise the boundary would be pulled inside the user-set width).
    eff_inflation = 0.0 if use_fixed_corridor else inflation_factor
    td = _build_track(center_lane, center_deriv, right_lane, left_lane, ref_v,
                      inflation_factor=eff_inflation, extend_part=extend_part)
    td.ref_v = ref_v.copy()  # stash raw (post-brake-profile) array for tests & diagnostics
    return td


def find_current_arc_length(track: TrackData, car_pos: np.ndarray,
                            arc_min_dist_tol: float = 1.0):
    """Project car (x,y) onto extended centerline; returns (current_s, nearest_idx).

    Uses dot-product projection on the segment between nearest waypoint and
    its neighbour. Wraps `current_s` modulo the original (unextended) arc
    length so MPC's solver-internal lap counter stays consistent.
    """
    cl = track.center_lane
    s_arr = track.element_arc_lengths
    L_orig = float(track.element_arc_lengths_orig[-1])

    distances = np.linalg.norm(cl - car_pos, axis=1)
    nearest = int(np.argmin(distances))
    min_dist = float(distances[nearest])

    if min_dist > arc_min_dist_tol:
        if nearest == 0:
            next_idx, prev_idx = 1, cl.shape[0] - 1
        elif nearest == cl.shape[0] - 1:
            next_idx, prev_idx = 0, cl.shape[0] - 2
        else:
            next_idx, prev_idx = nearest + 1, nearest - 1
        seg_back = cl[prev_idx] - cl[nearest]
        if np.dot(car_pos - cl[nearest], seg_back) > 0:
            actual = prev_idx
        else:
            actual = nearest
            nearest = next_idx
        seg_fwd = cl[nearest] - cl[actual]
        seg_norm = np.linalg.norm(seg_fwd)
        if seg_norm > 1e-9:
            projection = np.dot(car_pos - cl[actual], seg_fwd) / seg_norm
        else:
            projection = 0.0
        current_s = float(s_arr[actual]) + projection
    else:
        current_s = float(s_arr[nearest])

    # NOTE: do NOT force current_s=0 when nearest==0. When the car is just
    # BEHIND the start/finish seam the projection legitimately yields s≈L
    # (actual=last waypoint); clobbering it to 0 injected a full-lap
    # discontinuity at the line every lap. The wrap below handles s≥L_orig,
    # and the exactly-at-start case already reads s_arr[0]=0 via the else branch.
    if current_s >= L_orig:
        current_s = current_s % L_orig
    return current_s, nearest
