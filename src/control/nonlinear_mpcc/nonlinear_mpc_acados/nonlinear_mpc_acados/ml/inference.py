"""Tiny runtime wrapper around the trained WeightScaleMLP TorchScript.

Used by mpc_node to convert (kappa_abs, kappa_signed, v_actual, ref_v,
v_max_cost) → (q_cte_scale, q_lag_scale, q_v_scale, q_drate_scale) every
control cycle.

torch.jit.load is used so this module does NOT depend on model.py — the
TorchScript file is self-contained.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import torch

# CPU thread thrash inside the 40Hz control loop was making the controller
# look "laggy" — cap to a single thread so torch can't fan out and contend
# with the rclpy executor / acados solver.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads can only be called before any torch op runs;
    # if something else already touched torch, just ignore.
    pass


class WeightScaleInference:
    def __init__(self, model_path: str):
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        self.model = torch.jit.load(str(path), map_location="cpu")
        self.model.eval()
        # Freeze + optimize the scripted graph so per-call overhead is just
        # the matmuls. freeze() inlines params, optimize_for_inference fuses.
        try:
            self.model = torch.jit.freeze(self.model)
            self.model = torch.jit.optimize_for_inference(self.model)
        except Exception:
            pass
        # Pre-allocated input buffer — avoid torch.tensor() per call (the
        # allocation + dtype check is the dominant cost for a ~3k-param MLP).
        self._buf = torch.zeros((1, 5), dtype=torch.float32)
        # Warm up so JIT codegen happens once, not on the first hot cycle.
        with torch.no_grad():
            self.model(self._buf)

    @torch.no_grad()
    def __call__(self, kappa_abs: float, kappa_signed: float,
                 v_actual: float, ref_v: float,
                 v_max_cost: float) -> tuple[float, float, float, float]:
        b = self._buf
        b[0, 0] = kappa_abs
        b[0, 1] = kappa_signed
        b[0, 2] = v_actual
        b[0, 3] = ref_v
        b[0, 4] = v_max_cost
        y = self.model(b)[0]
        return float(y[0]), float(y[1]), float(y[2]), float(y[3])


def default_model_path() -> str:
    return str(Path(__file__).resolve().parent / "saved" / "weight_scaler.pt")
