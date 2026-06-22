"""GP residual learning wrapper (Phase D, 2026-05-27).

Loads the offline-trained GP residual model and wraps the acados solver
with L4acados.controllers.ResidualLearningMPC. Exposes the same .set / .get
/ .solve / .reset API as AcadosOcpSolver so existing mpc_core code paths
keep working unchanged.

Pipeline:
  MPC.setup_MPC()          → acados ocp + AcadosOcpSolver (nominal model)
  wrap_solver_with_gp(...) → replaces self.mpc.solver with GPMPCAdapter

GP model:
  Trained offline via scripts/train_gp_residual.py.
  Input  5D: [vx, vy, r, delta, a_x]
  Output 3D: [d_vx, d_vy, d_r]

Acados side:
  B (8x3): identity on state rows [3, 4, 5] (vx, vy, r). x/y/psi/s/delta_prev
  not corrected — GP only learns dynamic-state residuals.

Feature selector:
  acados ocp's full (state, input) is 11D = 8 state + 3 input. We extract the
  5D feature vector (vx, vy, r, delta, a_x) by indexing into (state, input):
    state[3]=vx, state[4]=vy, state[5]=r
    input[1]=delta, input[0]=a_x

Loading dependencies (PYTHONPATH=$HOME/l4acados/src to import l4acados).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import gpytorch


# ────────────────────────────────────────────────────────────────
# Feature extraction: acados (state + input) → GP 5D input
# ────────────────────────────────────────────────────────────────
# acados state idx: 0=x, 1=y, 2=psi, 3=vx, 4=vy, 5=r, 6=s, 7=delta_prev
# acados input idx: 0=a_x, 1=delta, 2=p_v
_GP_FEATURE_IDX = [3, 4, 5, 8 + 1, 8 + 0]   # vx, vy, r, delta, a_x
                                              # (input idx 1, 0 shifted by 8)


def build_gp_model_from_state(ckpt_path: str | Path):
    """Reconstruct trained GP from saved state_dict."""
    try:
        from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
            BatchIndependentInducingPointGPModel,
        )
    except ImportError as e:
        raise ImportError(
            f"l4acados import failed ({e}). Set PYTHONPATH=$HOME/l4acados/src "
            "and re-launch."
        )

    ckpt = torch.load(str(ckpt_path), weights_only=False)
    n_train = ckpt["n_train"]
    inducing = ckpt["inducing"]
    in_dim = len(ckpt["input_keys"])
    out_dim = len(ckpt["output_keys"])

    # Likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=out_dim)
    likelihood = likelihood.double()
    likelihood.load_state_dict(ckpt["lik_state"])

    # Model — needs dummy train_x to reconstruct. The l4acados model selects
    # `inducing` rows FROM train_x when inducing_points is an int (it does
    # train_x[permuted_indices[:inducing]]). So dummy_x must have >= `inducing`
    # rows, otherwise the inducing_points parameter is allocated with the wrong
    # shape (e.g. [1, in_dim]) and load_state_dict fails with a size mismatch.
    n_dummy = max(inducing, 1)
    dummy_x = torch.zeros(n_dummy, in_dim, dtype=torch.double)
    dummy_y = torch.zeros(n_dummy, out_dim, dtype=torch.double)
    gp = BatchIndependentInducingPointGPModel(
        dummy_x, dummy_y, likelihood,
        inducing_points=inducing,
        use_ard=True,
        residual_dimension=out_dim,
    ).double()
    gp.load_state_dict(ckpt["gp_state"])
    gp.eval()
    likelihood.eval()

    return gp, likelihood, ckpt


# ────────────────────────────────────────────────────────────────
# Custom feature selector + normalizer
# ────────────────────────────────────────────────────────────────
class NormalizingResidualModel:
    """Wraps a GPyTorchResidualModel with input/output normalization +
    feature index selection, matching how the GP was trained."""

    def __init__(self, gp, likelihood, ckpt):
        from l4acados.models.pytorch_models.gpytorch_models.gpytorch_residual_model import (
            GPyTorchResidualModel,
        )
        from l4acados.models.pytorch_models.pytorch_feature_selector import (
            PyTorchFeatureSelector,
        )

        self.X_mean = ckpt["X_mean"].numpy().astype(np.float64)
        self.X_std  = ckpt["X_std"].numpy().astype(np.float64)
        self.Y_mean = ckpt["Y_mean"].numpy().astype(np.float64)
        self.Y_std  = ckpt["Y_std"].numpy().astype(np.float64)

        # FeatureSelector picks [vx, vy, r, delta, a_x] from (state+input)
        # and applies (x - mean) / std normalization.
        # Mask: True for selected dims, False else. acados full = 8 + 3 = 11.
        mask = [False] * 11
        for idx in _GP_FEATURE_IDX:
            mask[idx] = True
        # Note: feature selector picks columns in mask order. We need to
        # reorder to match training order [vx, vy, r, delta, a_x].
        # Build a permutation: post-selector dims → training order.
        # Simpler: just do raw indexing manually inside evaluate() wrapper.

        self.gp = gp
        self.likelihood = likelihood
        self._GPyTorchResidualModel = GPyTorchResidualModel
        # bypass feature selector (use our own indexing)
        self._inner = GPyTorchResidualModel(
            gp,
            feature_selector=PyTorchFeatureSelector(),  # identity
        )

    def _extract_features(self, y_full):
        """y_full: (N, 11) acados (state, input). Returns (N, 5) GP input."""
        y_arr = np.asarray(y_full, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[None, :]
        # Pick [vx, vy, r, delta, a_x]
        feats = y_arr[:, _GP_FEATURE_IDX].copy()
        # Normalize
        feats = (feats - self.X_mean) / self.X_std
        return feats

    def _denormalize_output(self, y_norm):
        """y_norm: (N, 3). Returns (N, 3) real-scale residual."""
        return y_norm * self.Y_std + self.Y_mean

    # Interface for ResidualLearningMPC
    def value_and_jacobian(self, y):
        """
        y: (N, 11) full (state, input) at each stage.
        Returns:
          value:    (N, 3) residual (real scale)
          jacobian: (N, 3, 11) ∂residual/∂(state, input)
        """
        feats = self._extract_features(y)  # (N, 5) normalized
        N = feats.shape[0]

        # Use the inner GPyTorchResidualModel for value + jacobian, but
        # feed it the already-extracted features (5D, normalized).
        val_norm, jac_norm = self._inner.value_and_jacobian(feats)
        # val_norm: (N, 3).
        # jac_norm: l4acados returns shape (residual_dim, N, feat_dim) = (3, N, 5)
        # (see PyTorchResidualModel.value_and_jacobian docstring). Reorder to
        # (N, 3, 5) so the denormalization broadcasts correctly.
        jac_norm = np.asarray(jac_norm, dtype=np.float64)
        if jac_norm.ndim == 3 and jac_norm.shape[0] == self.Y_std.shape[0]:
            jac_norm_5d = np.transpose(jac_norm, (1, 0, 2))  # (N, 3, 5)
        else:
            jac_norm_5d = jac_norm

        # Denormalize value
        val = val_norm * self.Y_std + self.Y_mean
        # Denormalize jacobian:
        #   value_real = value_norm * Y_std + Y_mean
        #   ∂value_real/∂feat_real = (Y_std / X_std) * ∂value_norm/∂feat_norm
        # (broadcasting: out_dim Y_std, in_dim X_std)
        jac_5d = jac_norm_5d * (self.Y_std[None, :, None] / self.X_std[None, None, :])

        # Expand jacobian to (N, 3, 11): zeros for unselected dims
        jac_11d = np.zeros((N, 3, 11), dtype=np.float64)
        for k, idx in enumerate(_GP_FEATURE_IDX):
            jac_11d[:, :, idx] = jac_5d[:, :, k]
        return val, jac_11d

    def evaluate(self, y):
        return self.value_and_jacobian(y)[0]

    def jacobian(self, y):
        return self.value_and_jacobian(y)[1]


# ────────────────────────────────────────────────────────────────
# Adapter: ResidualLearningMPC → AcadosOcpSolver-compatible API
# ────────────────────────────────────────────────────────────────
class GPMPCAdapter:
    """Make ResidualLearningMPC look like AcadosOcpSolver to existing code.

    Proxies set/get/reset to the underlying acados solver. solve() routes
    through the GP-corrected pipeline.
    """

    def __init__(self, residual_learning_mpc):
        self._wrap = residual_learning_mpc

    # Proxy direct attribute access (for status fields, etc.)
    def __getattr__(self, name):
        return getattr(self._wrap.ocp_solver, name)

    def set(self, *args, **kwargs):
        return self._wrap.ocp_solver.set(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self._wrap.ocp_solver.get(*args, **kwargs)

    def reset(self):
        return self._wrap.ocp_solver.reset()

    def solve(self):
        return self._wrap.solve(acados_sqp_mode=True)

    def options_set(self, *args, **kwargs):
        return self._wrap.ocp_solver.options_set(*args, **kwargs)


# ────────────────────────────────────────────────────────────────
# Top-level: wrap an MPC instance's solver with GP residual learning.
# ────────────────────────────────────────────────────────────────
def wrap_solver_with_gp(mpc, gp_ckpt_path: str | Path,
                         logger=None) -> bool:
    """
    Replace mpc.solver (AcadosOcpSolver) with a GPMPCAdapter wrapping
    ResidualLearningMPC.

    Returns True on success, False on failure (and leaves mpc.solver intact).

    Requires mpc.setup_MPC() to have been called (mpc.ocp + mpc.solver
    populated).
    """
    log = (logger.info if logger else print)
    log_warn = (logger.warning if logger else print)
    log_err = (logger.error if logger else print)

    if not os.path.isfile(gp_ckpt_path):
        log_warn(f"[gp_residual] no checkpoint at {gp_ckpt_path}, skipping")
        return False

    try:
        from l4acados.controllers import ResidualLearningMPC
    except ImportError as e:
        log_err(f"[gp_residual] l4acados import failed: {e}")
        log_err(f"[gp_residual] set PYTHONPATH=$HOME/l4acados/src and rebuild")
        return False

    try:
        log(f"[gp_residual] loading GP from {gp_ckpt_path}")
        gp, lik, ckpt = build_gp_model_from_state(gp_ckpt_path)
        log(f"[gp_residual] GP loaded: {ckpt['n_train']} train, "
            f"{ckpt['inducing']} inducing")

        residual_model = NormalizingResidualModel(gp, lik, ckpt)

        # B matrix: 8x3, identity on (vx, vy, r) = state idx [3, 4, 5]
        B = np.zeros((mpc.n_states, 3), dtype=np.float64)
        B[3, 0] = 1.0
        B[4, 1] = 1.0
        B[5, 2] = 1.0

        # NOTE: build_c_code=True triggers second codegen (~30s).
        log("[gp_residual] building ResidualLearningMPC (~30s codegen)...")
        residual_mpc = ResidualLearningMPC(
            ocp=mpc.ocp,
            B=B,
            residual_model=residual_model,
            use_cython=True,
            build_c_code=True,
            path_json_ocp="/tmp/acados_ocp_evompcc_gp.json",
            path_json_sim="/tmp/acados_sim_evompcc_gp.json",
        )

        mpc.solver = GPMPCAdapter(residual_mpc)
        mpc._gp_residual_active = True
        log("[gp_residual] solver wrapped — GP residual active")
        return True

    except Exception as e:
        log_err(f"[gp_residual] wrap failed: {e}")
        import traceback
        log_err(traceback.format_exc())
        return False
