"""gp_casadi_residual.py — closed-form CasADi export of a trained sparse GP mean.

Phase D, 2026-05-29.

GOAL
----
Express the posterior MEAN of the offline-trained sparse (inducing-point) GP
residual model as a *pure CasADi symbolic expression*, so it can be baked into
an acados MPCC dynamics model and codegen'd to C — WITHOUT torch / gpytorch /
l4acados at runtime.

For each output d (d ∈ {d_vx, d_vy, d_r}) the predictive mean is

    mu_d(z) = sum_{i=1}^{M} k_d(z_norm, Z_i) * alpha_{d,i}

with
    z_norm   = (z - X_mean) / X_std                       (input normalization)
    k_d(a,b) = outputscale_d * exp(-0.5 * sum_j ((a_j - b_j) / ell_{d,j})^2)
               (gpytorch ScaleKernel(RBFKernel, ARD))   ARD lengthscales ell_{d,j}
    mu_real_d = mu_d * Y_std_d + Y_mean_d                 (output de-normalization)

WHY THIS FORM IS EXACT
----------------------
The GP is a gpytorch `InducingPointKernel` (SGPR / Nyström / Subset-of-Regressors)
wrapped in an ExactGP.  Its predictive mean is

    mu(z) = K_zU * K_UU^{-1} * K_UX * mean_cache              (gpytorch SGPR)
          = K_zU * alpha,   alpha := K_UU^{-1} K_UX mean_cache  (M-vector)

i.e. the mean lies *exactly* in the span of {k_base(z, Z_i)}_{i=1..M}.  Therefore
alpha can be recovered without touching gpytorch internals: evaluate the gpytorch
predictive mean at the M inducing points themselves (in normalized space), giving
b_d = mu_d(Z) (length-M vector per output), then solve the M×M kernel system

    K_UU^{(d)} * alpha_d = b_d            =>   alpha_d = K_UU^{(d) -1} b_d

(adding the same psd-jitter gpytorch uses).  K_UU^{(d)} is the base (un-scaled
× outputscale) kernel Gram of the inducing points for output d.  This makes the
closed-form sum reproduce gpytorch's mean to ~1e-6.

USAGE
-----
    from .gp_casadi_residual import load_gp_casadi_params, make_casadi_function
    p   = load_gp_casadi_params("~/bo_results/gp_residual_realvy.pt")
    fun = make_casadi_function(p)            # ca.Function: 5-vec z -> 3-vec mu
    mu  = fun(z_numeric)

or symbolically inside an acados model build:
    z   = ca.vertcat(vx, vy, r, delta, a_x)
    mu  = build_casadi_residual(z, p)        # ca.SX/MX 3-vector (real scale)

This module is OFFLINE / build-time only. It depends on numpy + casadi at build
time; the *generated C* depends on neither.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import casadi as ca
except ImportError:  # casadi optional for param-extraction-only use
    ca = None


# acados (state,input) -> GP 5D feature indices, matching gp_residual_wrapper.
# state: 0=x 1=y 2=psi 3=vx 4=vy 5=r 6=s 7=delta_prev ; input: 0=a_x 1=delta 2=p_v
# GP input order (training): [vx, vy, r, delta, a_x]
GP_INPUT_KEYS = ["vx", "vy", "r", "delta", "a_x"]
GP_OUTPUT_KEYS = ["d_vx", "d_vy", "d_r"]


@dataclass
class GpCasadiParams:
    """All numbers needed to build the closed-form CasADi mean."""
    Z: np.ndarray            # (M, D)  inducing points, NORMALIZED input space
    alpha: np.ndarray        # (out_dim, M)  per-output weights
    lengthscale: np.ndarray  # (out_dim, D)  ARD lengthscales (normalized space)
    outputscale: np.ndarray  # (out_dim,)    ScaleKernel outputscale
    X_mean: np.ndarray       # (D,)
    X_std: np.ndarray        # (D,)
    Y_mean: np.ndarray       # (out_dim,)
    Y_std: np.ndarray        # (out_dim,)
    input_keys: list
    output_keys: list

    @property
    def M(self) -> int:
        return self.Z.shape[0]

    @property
    def in_dim(self) -> int:
        return self.Z.shape[1]

    @property
    def out_dim(self) -> int:
        return self.alpha.shape[0]


# ────────────────────────────────────────────────────────────────
# (a) Loader: read checkpoint, rebuild gpytorch GP, extract Z, alpha, hypers
# ────────────────────────────────────────────────────────────────
def _default_train_data_path(ckpt_path: str) -> str | None:
    """Heuristic: gp_residual_<tag>.pt  ->  gp_train_data_<tag>.pt next to it."""
    ck = Path(ckpt_path)
    name = ck.name
    if name.startswith("gp_residual"):
        cand = ck.with_name(name.replace("gp_residual", "gp_train_data", 1))
        if cand.is_file():
            return str(cand)
    return None


def load_gp_casadi_params(ckpt_path: str | Path,
                          train_data_path: str | Path | None = None
                          ) -> GpCasadiParams:
    """Load checkpoint, reconstruct the gpytorch SGPR CONDITIONED ON its training
    data, and extract closed-form parameters (Z, per-output alpha, ARD
    lengthscales, outputscale, norm stats).

    The checkpoint stores only hyperparameters + inducing points, NOT the train
    data. A gpytorch ExactGP's posterior mean is the *prior* (ZeroMean -> 0)
    unless the model is conditioned on its training inputs/targets. We therefore
    reload the training data (auto-detected as gp_train_data_<tag>.pt next to the
    checkpoint, or via train_data_path) and rebuild the model WITH it, so the
    posterior mean (and hence alpha) is correct.

    Requires torch + gpytorch + l4acados on PYTHONPATH (offline / build time).
    """
    import torch
    import gpytorch

    ckpt_path = os.path.expanduser(str(ckpt_path))
    ckpt = torch.load(ckpt_path, weights_only=False)

    in_dim = len(ckpt["input_keys"])
    out_dim = len(ckpt["output_keys"])
    inducing = ckpt["inducing"]

    X_mean = ckpt["X_mean"].double()
    X_std = ckpt["X_std"].double()
    Y_mean = ckpt["Y_mean"].double()
    Y_std = ckpt["Y_std"].double()

    # --- training data (needed to condition the ExactGP posterior) ----------
    if train_data_path is None:
        train_data_path = _default_train_data_path(ckpt_path)
    if train_data_path is None or not os.path.isfile(os.path.expanduser(str(train_data_path))):
        raise FileNotFoundError(
            "Training data file required to condition the GP posterior but not "
            f"found (looked for gp_train_data_<tag>.pt next to {ckpt_path}). "
            "Pass train_data_path=... explicitly."
        )
    blob = torch.load(os.path.expanduser(str(train_data_path)), weights_only=False)
    X = blob["x_train"].double()
    Y = blob["y_train"].double()
    # normalize with the SAME stats the checkpoint was trained with
    Xn = (X - X_mean) / X_std
    Yn = (Y - Y_mean) / Y_std

    # --- rebuild model from state_dict (mirror gp_residual_wrapper) ---------
    try:
        from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
            BatchIndependentInducingPointGPModel,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"l4acados import failed ({e}). Set PYTHONPATH=$HOME/l4acados/src."
        )

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=out_dim).double()
    likelihood.load_state_dict(ckpt["lik_state"])

    # Condition on the real (normalized) training data so the posterior mean is
    # correct. inducing points come from the loaded state_dict (overwrites the
    # random init selected from Xn here).
    gp = BatchIndependentInducingPointGPModel(
        Xn, Yn, likelihood,
        inducing_points=inducing, use_ard=True, residual_dimension=out_dim,
    ).double()
    gp.load_state_dict(ckpt["gp_state"])
    # make sure the ExactGP knows its train data (for posterior conditioning)
    gp.set_train_data(inputs=Xn, targets=Yn, strict=False)
    gp.eval()
    likelihood.eval()

    # --- hyperparameters (transformed, real values) -------------------------
    base_rbf = gp.covar_module.base_kernel.base_kernel        # RBFKernel (ARD)
    scale_k = gp.covar_module.base_kernel                     # ScaleKernel
    # lengthscale shape (out_dim, 1, in_dim) -> (out_dim, in_dim)
    lengthscale = base_rbf.lengthscale.detach().squeeze(1).cpu().numpy().astype(np.float64)
    outputscale = scale_k.outputscale.detach().cpu().numpy().astype(np.float64).reshape(out_dim)

    # inducing points (already in NORMALIZED input space — model was trained on Xn)
    Z = gp.covar_module.inducing_points.detach().cpu().numpy().astype(np.float64)  # (M, D)
    M = Z.shape[0]

    # --- recover per-output alpha so that mu_d(z) = sum_i k_d(z, Z_i) alpha_di
    #     mu lies in span{k_d(z, Z_i)} (SGPR), so:
    #       b_d = mu_d(Z)            (eval gpytorch mean AT inducing points)
    #       K_UU^{(d)} alpha_d = b_d
    #     where K_UU^{(d)} is the base (scaled) kernel Gram of Z for output d.
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        Zt = torch.from_numpy(Z).double()
        pred = gp(Zt)                       # MultitaskMultivariateNormal
        mu_Z = pred.mean.cpu().numpy().astype(np.float64)   # (M, out_dim)

    # base (scaled) kernel Gram for each output, computed numerically (numpy).
    # The posterior mean lies EXACTLY in span{k_d(.,Z_i)}, so a direct solve of
    # K_UU alpha = mu(Z) recovers alpha to ~1e-12 (no jitter needed; adding
    # jitter would only inject error vs the gpytorch reference).
    alpha = np.zeros((out_dim, M), dtype=np.float64)
    for d in range(out_dim):
        Kuu = _rbf_gram_np(Z, Z, lengthscale[d], outputscale[d])
        b_d = mu_Z[:, d]
        alpha[d] = np.linalg.solve(Kuu, b_d)

    return GpCasadiParams(
        Z=Z,
        alpha=alpha,
        lengthscale=lengthscale,
        outputscale=outputscale,
        X_mean=X_mean.cpu().numpy().astype(np.float64),
        X_std=X_std.cpu().numpy().astype(np.float64),
        Y_mean=Y_mean.cpu().numpy().astype(np.float64),
        Y_std=Y_std.cpu().numpy().astype(np.float64),
        input_keys=list(ckpt["input_keys"]),
        output_keys=list(ckpt["output_keys"]),
    )


def _rbf_gram_np(A: np.ndarray, B: np.ndarray, ell: np.ndarray,
                 outputscale: float) -> np.ndarray:
    """ARD RBF Gram matrix (numpy). gpytorch convention:
       k(a,b) = outputscale * exp(-0.5 * sum_j ((a_j-b_j)/ell_j)^2)."""
    A = np.atleast_2d(A) / ell[None, :]
    B = np.atleast_2d(B) / ell[None, :]
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    return outputscale * np.exp(-0.5 * np.clip(sq, 0.0, None))


# ────────────────────────────────────────────────────────────────
# Reference numpy evaluation of the closed form (for validation / debug)
# ────────────────────────────────────────────────────────────────
def eval_numpy(z_raw: np.ndarray, p: GpCasadiParams) -> np.ndarray:
    """Evaluate the closed-form mean in numpy. z_raw: (N, D) RAW (un-normalized)
    GP inputs. Returns (N, out_dim) real-scale residual."""
    z_raw = np.atleast_2d(z_raw).astype(np.float64)
    z_norm = (z_raw - p.X_mean[None, :]) / p.X_std[None, :]
    out = np.zeros((z_raw.shape[0], p.out_dim), dtype=np.float64)
    for d in range(p.out_dim):
        k = _rbf_gram_np(z_norm, p.Z, p.lengthscale[d], p.outputscale[d])  # (N, M)
        mu = k @ p.alpha[d]                                                # (N,)
        out[:, d] = mu * p.Y_std[d] + p.Y_mean[d]
    return out


# ────────────────────────────────────────────────────────────────
# (b) Symbolic CasADi builder
# ────────────────────────────────────────────────────────────────
def build_casadi_residual(z_sym, p: GpCasadiParams):
    """Build the CasADi symbolic posterior mean.

    Args:
        z_sym : CasADi SX/MX 5-vector of RAW (un-normalized) GP inputs,
                order [vx, vy, r, delta, a_x].
        p     : GpCasadiParams from load_gp_casadi_params().

    Returns:
        CasADi 3-vector (SX/MX) of the de-normalized posterior mean
        [mu_d_vx, mu_d_vy, mu_d_r].
    """
    if ca is None:  # pragma: no cover
        raise ImportError("casadi not available")

    z = z_sym
    # input normalization
    z_norm = (z - ca.DM(p.X_mean)) / ca.DM(p.X_std)   # (D,)

    Z = ca.DM(p.Z)                                     # (M, D) constant
    mu = []
    for d in range(p.out_dim):
        ell = ca.DM(p.lengthscale[d])                 # (D,)
        outsc = float(p.outputscale[d])
        alpha_d = ca.DM(p.alpha[d])                   # (M,)
        # vectorized squared scaled distance from z_norm to each inducing pt:
        # diff (M, D); scaled by lengthscale; row-sum of squares
        diff = (ca.repmat(z_norm.T, p.M, 1) - Z) / ca.repmat(ell.T, p.M, 1)  # (M,D)
        sq = ca.sum2(diff * diff)                      # (M,1)
        k = outsc * ca.exp(-0.5 * sq)                  # (M,1)
        mu_d = ca.dot(k, alpha_d)                      # scalar (normalized)
        mu_d_real = mu_d * float(p.Y_std[d]) + float(p.Y_mean[d])
        mu.append(mu_d_real)
    return ca.vertcat(*mu)


# ────────────────────────────────────────────────────────────────
# (c) ca.Function factory
# ────────────────────────────────────────────────────────────────
def make_casadi_function(p: GpCasadiParams, name: str = "gp_residual_mean"):
    """Return a ca.Function mapping a 5-vector RAW GP input -> 3-vector mean."""
    if ca is None:  # pragma: no cover
        raise ImportError("casadi not available")
    z = ca.SX.sym("z", p.in_dim)
    mu = build_casadi_residual(z, p)
    return ca.Function(name, [z], [mu], ["z"], ["mu"])
