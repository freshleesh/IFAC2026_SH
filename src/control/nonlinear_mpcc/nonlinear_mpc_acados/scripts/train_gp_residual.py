#!/usr/bin/env python3
"""train_gp_residual.py — Phase D step 2.

gp_train_data.pt (85K samples) → sparse Inducing-Point GP 학습.

전체 ExactGP 는 O(N³) = 6×10¹⁴ 로 불가능. SoR sparse approx 사용:
  - inducing points 200 (memory + inference < 5ms)
  - ARD (5D 각 차원별 lengthscale)
  - BatchIndependent 3 outputs (d_vx / d_vy / d_r 독립)

L4acados 의 BatchIndependentInducingPointGPModel 사용 →
PYTHONPATH=$HOME/l4acados/src 필요.

Usage:
  PYTHONPATH=$HOME/l4acados/src python3 train_gp_residual.py
  PYTHONPATH=$HOME/l4acados/src python3 train_gp_residual.py --inducing 500 --iters 300
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import gpytorch

from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentInducingPointGPModel,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path.home() / "bo_results" / "gp_train_data.pt"))
    p.add_argument("--out",  default=str(Path.home() / "bo_results" / "gp_residual.pt"))
    p.add_argument("--inducing", type=int, default=200,
                   help="inducing points for sparse approx (200 default)")
    p.add_argument("--iters", type=int, default=200, help="Adam epochs")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--subsample", type=int, default=15000,
                   help="random subsample size (0=all 85K, but slow)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # ── Load data ────────────────────────────────────────────────
    blob = torch.load(args.data, weights_only=False)
    X_all = blob["x_train"].double()
    Y_all = blob["y_train"].double()
    input_keys = blob["input_keys"]
    output_keys = blob["output_keys"]
    print(f"Loaded {X_all.shape[0]} samples (in={input_keys}, out={output_keys})")

    # ── Subsample ────────────────────────────────────────────────
    if args.subsample > 0 and X_all.size(0) > args.subsample:
        idx = torch.randperm(X_all.size(0))[: args.subsample]
        X = X_all[idx]
        Y = Y_all[idx]
        print(f"Subsampled to {X.shape[0]}")
    else:
        X = X_all
        Y = Y_all

    # ── Normalize ────────────────────────────────────────────────
    X_mean, X_std = X.mean(0), X.std(0).clamp_min(1e-6)
    Y_mean, Y_std = Y.mean(0), Y.std(0).clamp_min(1e-6)
    Xn = (X - X_mean) / X_std
    Yn = (Y - Y_mean) / Y_std

    print(f"X_mean={X_mean.tolist()}")
    print(f"X_std ={X_std.tolist()}")
    print(f"Y_std ={Y_std.tolist()}  (residual scale)")

    # ── Model setup ──────────────────────────────────────────────
    nout = Yn.size(-1)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout)
    likelihood = likelihood.double()
    gp = BatchIndependentInducingPointGPModel(
        Xn, Yn, likelihood,
        inducing_points=args.inducing,
        use_ard=True,
        residual_dimension=nout,
    ).double()

    print(f"Inducing points: {gp.num_inducing_points}")

    # ── Training ─────────────────────────────────────────────────
    gp.train(); likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    opt = torch.optim.Adam(gp.parameters(), lr=args.lr)

    t0 = time.time()
    losses = []
    for epoch in range(args.iters):
        opt.zero_grad()
        out = gp(Xn)
        loss = -mll(out, Yn).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == args.iters - 1:
            ls = (gp.covar_module.base_kernel.base_kernel.lengthscale
                  .squeeze().detach().mean(0).tolist())
            print(f"  [{epoch:3d}/{args.iters}] loss={loss.item():.3f}  "
                  f"avg_ls={[round(l,2) for l in ls]}  "
                  f"noise={likelihood.noise.mean().item():.4f}")

    train_time = time.time() - t0
    print(f"\nTraining took {train_time:.1f}s")

    # ── Eval mode + sanity check ─────────────────────────────────
    gp.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(gp(Xn[:1000]))
        mean = pred.mean
        rmse_norm = ((mean - Yn[:1000]) ** 2).mean(0).sqrt()
        rmse_real = rmse_norm * Y_std
        print(f"\nTrain RMSE (per output, real scale):")
        for k, name in enumerate(output_keys):
            print(f"  {name:8s}: rmse={rmse_real[k].item():.4f}  "
                  f"vs std={Y_std[k].item():.4f}  "
                  f"({rmse_real[k]/Y_std[k]*100:.1f}% of residual std)")

    # ── ARD lengthscales (real scale, per output) ────────────────
    ls = (gp.covar_module.base_kernel.base_kernel.lengthscale
          .squeeze().detach())   # (nout, ninput)
    print(f"\nFinal ARD lengthscales (normalized space):")
    for j, name in enumerate(output_keys):
        ls_str = ', '.join(f"{k}={ls[j,i].item():.2f}"
                            for i, k in enumerate(input_keys))
        print(f"  {name}: {ls_str}")

    # ── Save ─────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gp_state":    gp.state_dict(),
        "lik_state":   likelihood.state_dict(),
        "X_mean":      X_mean,
        "X_std":       X_std,
        "Y_mean":      Y_mean,
        "Y_std":       Y_std,
        "input_keys":  input_keys,
        "output_keys": output_keys,
        "inducing":    args.inducing,
        "n_train":     X.size(0),
        "losses":      losses,
        "rmse_real":   rmse_real.tolist(),
    }, str(out_path))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
