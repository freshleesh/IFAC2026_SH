"""ref_v forward-backward brake profile honesty — a_long must be the SOLVER
brake limit (|A_MIN_DYN|=3.0), not the lateral a_lat_max proxy (7.14 = 2.4×
optimistic → corner entry overspeed → the final2 3-corner stuck family).

Run:
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m pytest \
        src/nonlinear_mpc_acados/test/test_brake_profile.py -v
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from nonlinear_mpc_acados.track_loader import build_track_from_wpnts


def _mk_wpnts(n=60):
    """원 트랙 (r=10) + 마지막 10pt 급커브 표기(κ=2.0) — 코너 전 제동 검증용.
    Wpnt 메시지 대신 SimpleNamespace (필드 호환)."""
    wp = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        x, y = 10.0 * math.cos(ang), 10.0 * math.sin(ang)
        kappa = 2.0 if (n - 10) <= i else 0.01   # 마지막 10pt = 급코너
        wp.append(SimpleNamespace(
            x_m=x, y_m=y,
            psi_rad=ang + math.pi / 2.0, psi_centerline_rad=0.0,
            d_left=0.8, d_right=0.8, vx_mps=8.0, kappa_radpm=kappa))
    return wp


class TestBrakeProfileHonesty(unittest.TestCase):
    def _refv(self, a_long_max):
        td = build_track_from_wpnts(
            _mk_wpnts(), default_v=8.0, a_lat_max=7.1445,
            corridor_half_width=0.0,
            a_long_max=a_long_max)
        # td.ref_v is the raw speed array stored on TrackData by build_track_from_wpnts
        return np.asarray(td.ref_v, dtype=float)

    def test_honest_brakes_earlier_than_optimistic(self):
        v3 = self._refv(3.0)
        v7 = self._refv(7.1445)
        # 정직한 제동(3.0)은 낙관(7.14)보다 어디서도 빠르면 안 되고,
        self.assertTrue(np.all(v3 <= v7 + 1e-9))
        # 코너 앞 어딘가에서는 분명히 더 일찍(더 낮게) 감속해야 한다.
        self.assertTrue(np.any(v3 < v7 - 0.2),
                        "honest profile identical to optimistic")

    def test_brake_rate_within_solver_limit(self):
        v = self._refv(3.0)
        wp = _mk_wpnts()
        n = len(wp)
        for i in range(n):
            j = (i + 1) % n
            ds = math.hypot(wp[j].x_m - wp[i].x_m, wp[j].y_m - wp[i].y_m)
            if v[i % len(v)] > v[j % len(v)]:   # braking segment
                a_req = (v[i % len(v)] ** 2 - v[j % len(v)] ** 2) / (2.0 * max(ds, 1e-6))
                # backward-pass 구성상 정확히 ≤3.0 (부동소수 오차만 허용)
                self.assertLessEqual(a_req, 3.0 + 1e-6,
                                     f"i={i}: implied brake {a_req:.2f} > 3.0")


if __name__ == '__main__':
    unittest.main()
