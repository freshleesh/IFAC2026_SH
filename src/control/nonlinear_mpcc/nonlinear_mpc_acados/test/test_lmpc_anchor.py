"""Track-anchored LMPC safe-set query arc length.

2026-06-14: the per-cycle SS query used to anchor at the previous solve's
predicted horizon-end state x_N. That created an anchor-less positive-feedback
loop — a lateral drift in x_N moved the query point, which moved the terminal
attractor onto the drifted point, which pulled x_N further out → ec grew
monotonically (0.22→0.47 m) until wedge (use_lmpc turned off, 3691fdb).

`lmpc_anchor_s` pins the query arc length to the TRACK (current s + one-horizon
look-ahead), independent of any predicted state — breaking the loop while
keeping the attractor one horizon ahead (the forward-progress carrot).

Run:
    cd src/nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_lmpc_anchor.py -q
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.model_policy import lmpc_anchor_s


class TestLmpcAnchorS(unittest.TestCase):
    def test_normal_lookahead(self):
        # s + v·N·dT, no wrap: 10 + 4·18·0.025 = 10 + 1.8 = 11.8
        self.assertAlmostEqual(
            lmpc_anchor_s(10.0, 4.0, 18, 0.025, 80.0), 11.8, places=6)

    def test_floor_applies_when_slow(self):
        # v≈0 → look-ahead collapses to the floor (1.0 m), so the anchor never
        # sits on top of the car (which would kill the forward pull).
        self.assertAlmostEqual(
            lmpc_anchor_s(10.0, 0.0, 18, 0.025, 80.0, lookahead_floor=1.0),
            11.0, places=6)

    def test_wraps_modulo_track_length(self):
        # 79 + 1.8 = 80.8 → wraps to 0.8 on an 80 m loop
        self.assertAlmostEqual(
            lmpc_anchor_s(79.0, 4.0, 18, 0.025, 80.0), 0.8, places=6)

    def test_lookahead_capped_at_half_loop(self):
        # absurd speed → look-ahead must not overtake half the loop (else the
        # anchor could land BEHIND the car after wrap).
        q = lmpc_anchor_s(0.0, 1000.0, 18, 0.025, 80.0)
        self.assertAlmostEqual(q, 40.0, places=6)

    def test_degenerate_track_length_returns_s(self):
        self.assertEqual(lmpc_anchor_s(5.0, 4.0, 18, 0.025, 0.0), 5.0)

    def test_result_always_in_loop(self):
        for s in (0.0, 40.0, 79.999):
            q = lmpc_anchor_s(s, 6.0, 18, 0.025, 80.0)
            self.assertGreaterEqual(q, 0.0)
            self.assertLess(q, 80.0)


if __name__ == '__main__':
    unittest.main()
