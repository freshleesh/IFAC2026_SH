"""Grip single-source helpers — μ·g·η lateral limit + a_lat clamp + brake const.

2026-06-10 spec (friction-ellipse-mu-design): the controller must never plan on
more lateral grip than μ·g·η, and ref_v braking must use the SOLVER brake limit.

Run:
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest \
        nonlinear_mpc_acados.test.test_model_policy_grip -v
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.model_policy import (
    A_MIN_DYN, G_GRAV, clamp_a_lat_to_grip, grip_a_lat_limit)


class TestGripHelpers(unittest.TestCase):
    def test_brake_const_matches_solver(self):
        # acados_kinematic lbu[0] 와 단일 소스 — 솔버 제동한계 -3.0.
        self.assertEqual(A_MIN_DYN, -3.0)

    def test_grip_limit_value(self):
        # mu=0.6, η=0.95 → 0.6·9.81·0.95 = 5.5917
        self.assertAlmostEqual(grip_a_lat_limit(0.6, 0.95), 5.5917, places=4)
        self.assertAlmostEqual(G_GRAV, 9.81)

    def test_clamp_above_limit(self):
        # 요청 7.1445 > 한계 5.5917 → clamp + flag
        eff, clamped = clamp_a_lat_to_grip(7.1445, mu=0.6, ellipse_frac=0.95)
        self.assertAlmostEqual(eff, 5.5917, places=4)
        self.assertTrue(clamped)

    def test_clamp_below_limit_passthrough(self):
        # 고그립(mu=1.0489): 한계 9.777 > 요청 7.1445 → 그대로 (기존 동작 보존)
        eff, clamped = clamp_a_lat_to_grip(7.1445, mu=1.0489, ellipse_frac=0.95)
        self.assertAlmostEqual(eff, 7.1445, places=4)
        self.assertFalse(clamped)


if __name__ == '__main__':
    unittest.main()
