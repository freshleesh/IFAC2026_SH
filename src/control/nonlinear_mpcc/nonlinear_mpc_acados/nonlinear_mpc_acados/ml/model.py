"""MLP weight-scaler for the MPCC adaptive cost.

Input  (state, 5 dim):
    [kappa_abs, kappa_signed, v_actual, ref_v, v_max_cost]

Output (scale, 4 dim, clamped to [0.3, 3.0]):
    [q_cte_scale, q_lag_scale, q_v_scale, q_drate_scale]

These plug into mpc_core's `q_*_scale_live` (multiplier on the
corresponding cost residual). Default scale = 1.0 → no change vs yaml.

The model is intentionally small (~3k params) so that:
    - TorchScript inference takes <0.1 ms on CPU (fits 40 Hz MPC loop)
    - it can be retrained quickly from a single drive's CSV (~30s on laptop)
"""
from __future__ import annotations

import torch
import torch.nn as nn

INPUT_DIM = 5
OUTPUT_DIM = 4
SCALE_MIN = 0.3
SCALE_MAX = 5.0    # 3.0 → 5.0. 트랙 max κ=0.82 에서 강한 q_drate_scale 필요.


class WeightScaleMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUTPUT_DIM),
        )
        # Output scaling: sigmoid → [SCALE_MIN, SCALE_MAX]
        self.scale_min = SCALE_MIN
        self.scale_range = SCALE_MAX - SCALE_MIN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4) → (B, 4) in [SCALE_MIN, SCALE_MAX]
        raw = self.net(x)
        return self.scale_min + self.scale_range * torch.sigmoid(raw)


def build_model(hidden: int = 32) -> WeightScaleMLP:
    return WeightScaleMLP(hidden=hidden)
