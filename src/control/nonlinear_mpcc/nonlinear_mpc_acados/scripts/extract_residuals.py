#!/usr/bin/env python3
"""extract_residuals.py — Phase D step 1.

mpc_logs/*.csv (BO trial logs) 에서 GP residual learning 학습 데이터 추출.

흐름:
  1. CSV 로드 (filter: post-tanh model, alive 구간, vx > 0.5)
  2. (vy, r) 를 positional derivative 로 추정
  3. a_x = dv_actual/dt 로 추정
  4. 매 row 마다 acados 모델 (tanh tire 8-state) 의 Euler 1-step 예측
  5. residual = state_{k+1}_actual - state_{k+1}_predicted
  6. 출력: gp_train_data.pt (x_train: 5D, y_train: 3D)

학습 input (5D)  : [vx, vy, r, delta, a_x]
학습 output (3D) : [residual_vx, residual_vy, residual_r]

Usage:
  python3 extract_residuals.py                  # 모든 post-tanh CSV
  python3 extract_residuals.py --csv <path>     # 단일 CSV (테스트)
  python3 extract_residuals.py --out custom.pt  # output path
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Model parameters (mirrors mpc_core/acados_kinematic.py)
L_WB = 0.307
M = 3.54
IZ = 0.05797
LF = 0.162
LR = 0.145
H_CG = 0.014
MU = 1.0
BF = BR = 10.0
DF = DR = 1.0           # tanh: F_y = mu * D * F_z * tanh(B * alpha)
G = 9.81
DT = 0.04               # control rate from yaml
V_B = 0.5               # dynamic/kinematic blend center
V_S = 0.3               # blend spread

MPC_LOGS = Path.home() / "mpc_logs"
# Post-tanh tire switch cutoff (2026-05-27 16:00:00 KST).
# 이후 모든 CSV (autoreg v=5/6/7/8 overnight 데이터 포함) 자동 매치.
POST_TANH_CUTOFF_EPOCH = 1779865200   # 2026-05-27 16:00 KST


# ────────────────────────────────────────────────────────────────
# Dynamics (numpy port of mpc_core's f_expl)
# ────────────────────────────────────────────────────────────────
def f_dynamic(state, u):
    """8-state dynamic bicycle (tanh tire) RHS. Returns xdot (8,)."""
    _, _, psi, vx, vy, r, _, delta_prev = state
    a_x, delta, p_v = u

    vx_safe = max(vx, 1e-3)
    alpha_f = np.arctan2(-vy - LF * r, vx_safe) + delta
    alpha_r = np.arctan2(-vy + LR * r, vx_safe)
    F_zf = M * (-a_x * H_CG + G * LR) / L_WB
    F_zr = M * (a_x * H_CG + G * LF) / L_WB
    F_yf = MU * DF * F_zf * np.tanh(BF * alpha_f)
    F_yr = MU * DR * F_zr * np.tanh(BR * alpha_r)

    return np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        r,
        a_x + (-F_yf * np.sin(delta)) / M + vy * r,
        (F_yr + F_yf * np.cos(delta)) / M - vx * r,
        (F_yf * LF * np.cos(delta) - F_yr * LR) / IZ,
        p_v,
        (delta - delta_prev) / DT,
    ])


def f_kinematic(state, u):
    """5-state kinematic limit (low-vx blend partner)."""
    _, _, psi, vx, vy, r, _, delta_prev = state
    a_x, delta, p_v = u

    beta_kin = np.arctan(LR * np.tan(delta) / L_WB)
    vy_tgt = vx * np.tan(beta_kin)
    r_tgt = (vx / L_WB) * np.tan(delta) * np.cos(beta_kin)
    tau_kin = 0.05
    return np.array([
        vx * np.cos(psi + beta_kin),
        vx * np.sin(psi + beta_kin),
        r_tgt,
        a_x,
        (vy_tgt - vy) / tau_kin,
        (r_tgt - r) / tau_kin,
        p_v,
        (delta - delta_prev) / DT,
    ])


def f_expl(state, u):
    """Blended dynamics (same as acados f_expl)."""
    vx = state[3]
    w = 0.5 * (1.0 + np.tanh((vx - V_B) / V_S))
    return w * f_dynamic(state, u) + (1.0 - w) * f_kinematic(state, u)


def euler_step(state, u, dt):
    return state + dt * f_expl(state, u)


# ────────────────────────────────────────────────────────────────
# Data extraction from CSV
# ────────────────────────────────────────────────────────────────
def process_csv(csv_path: Path,
                min_vx: float = 0.5,
                dt_tol: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_input [N, 5], y_residual [N, 3]) from one CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [skip] {csv_path.name}: {e}")
        return np.empty((0, 5)), np.empty((0, 3))

    needed = {"t", "v_actual", "steer_cmd", "car_x", "car_y", "car_yaw",
              "current_s", "mpcc_active", "feasible"}
    if not needed.issubset(df.columns):
        return np.empty((0, 5)), np.empty((0, 3))

    n = len(df)
    if n < 5:
        return np.empty((0, 5)), np.empty((0, 3))

    t = df["t"].to_numpy()
    vx = df["v_actual"].to_numpy()
    delta = df["steer_cmd"].to_numpy()
    x = df["car_x"].to_numpy()
    y = df["car_y"].to_numpy()
    yaw = np.unwrap(df["car_yaw"].to_numpy())   # unwrap to avoid 2π jumps
    s = df["current_s"].to_numpy()
    mpcc_active = df["mpcc_active"].to_numpy().astype(bool)
    feasible = df["feasible"].to_numpy().astype(bool) if "feasible" in df.columns else np.ones(len(df), bool)

    # Compute dt per step
    dt = np.diff(t)
    # Estimate r via yaw difference
    r_est = np.diff(yaw) / np.maximum(dt, 1e-6)
    # Estimate vx_world, vy_world via positional difference
    vxw = np.diff(x) / np.maximum(dt, 1e-6)
    vyw = np.diff(y) / np.maximum(dt, 1e-6)
    # Body frame
    yaw_mid = 0.5 * (yaw[:-1] + yaw[1:])
    vx_body_meas = np.cos(yaw_mid) * vxw + np.sin(yaw_mid) * vyw
    vy_body = -np.sin(yaw_mid) * vxw + np.cos(yaw_mid) * vyw
    # a_x = dv_actual / dt
    a_x_est = np.diff(vx) / np.maximum(dt, 1e-6)
    # delta_prev = previous delta
    delta_prev = np.roll(delta, 1)
    delta_prev[0] = 0.0

    # Build state arrays of length n-1 (using row k for state, k+1 for ref)
    # We need state[k] = [x[k], y[k], yaw[k], vx[k], vy_est[k], r_est[k], s[k], delta_prev[k]]
    # vy_est[k] / r_est[k] = midpoint estimate around k (use derivative at k, derived from k & k+1)
    # Use idx 0..n-2 as valid range (need k+1 for prediction).

    # For each row k in [0, n-2):
    #   state_k uses vy_body[k], r_est[k]  (forward derivative)
    #   u_k = [a_x_est[k], delta[k], p_v=v_cmd[k] (if exists else vx[k])]
    #   x_pred_k+1 = euler_step(state_k, u_k, dt[k])
    #   ref_k+1 = next row's full state (need vy_body[k+1], r_est[k+1] which requires k+2)
    # → valid range: k in [0, n-3)

    if n < 4:
        return np.empty((0, 5)), np.empty((0, 3))

    valid = []
    x_in_list = []
    y_out_list = []

    p_v_proxy = vx                          # rough: p_v ≈ vx (BO 결과 q_p 가 크니 acados 가 p_v ≈ v_max 로 채움 — 하지만 ṡ residual 안 쓸 거니 OK)

    for k in range(1, n - 2):
        # Filter: alive, post-warmup, dt normal, vx not too low
        # unicorn 트리: mpcc_active 미발행(전부0) → feasible 기반으로 우회
        # (v>min_vx 이고 feasible 이면 MPC 실제 제어 중)
        if not (feasible[k] and feasible[k+1]):
            continue
        if abs(dt[k] - DT) > dt_tol * DT:
            continue
        if vx[k] < min_vx:
            continue
        # Skip if positional derivative deviates too much (teleport from reset)
        if abs(vx_body_meas[k] - vx[k]) > 0.5:    # 0.5 m/s mismatch = teleport
            continue
        # Physical bounds: filter teleport/reset artefacts.
        if abs(a_x_est[k]) > 12.0:           # |a_x| > 12 m/s² = teleport jump
            continue
        if abs(r_est[k]) > 3.0:              # |r| > 3 rad/s = teleport/spin
            continue
        if abs(vy_body[k]) > 0.6:            # |vy| > 0.6 m/s = teleport drift
            continue
        # Also need k+1 sample to be sane (we use its r/vy as target)
        if abs(a_x_est[k+1]) > 12.0 or abs(r_est[k+1]) > 3.0:
            continue

        state_k = np.array([
            x[k], y[k], yaw[k], vx[k], vy_body[k], r_est[k],
            s[k], delta_prev[k]
        ])
        u_k = np.array([a_x_est[k], delta[k], p_v_proxy[k]])

        x_pred = euler_step(state_k, u_k, dt[k])
        # Reference (actual) next state — use derivative estimates at k+1
        state_kp1_actual = np.array([
            x[k+1], y[k+1], yaw[k+1], vx[k+1], vy_body[k+1], r_est[k+1],
            s[k+1], delta[k]
        ])

        # Residual: actual - predicted, only for dynamic states (vx, vy, r)
        res = state_kp1_actual[3:6] - x_pred[3:6]
        # Drop residual outliers (positional glitch / log timing artefact).
        if abs(res[0]) > 0.5 or abs(res[1]) > 0.5 or abs(res[2]) > 2.0:
            continue

        # GP input: 5D = [vx, vy_est, r_est, delta, a_x]
        x_in_list.append([vx[k], vy_body[k], r_est[k], delta[k], a_x_est[k]])
        y_out_list.append(res)
        valid.append(k)

    if not x_in_list:
        return np.empty((0, 5)), np.empty((0, 3))
    return np.array(x_in_list), np.array(y_out_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="단일 CSV 테스트")
    p.add_argument("--out", default=str(Path.home() / "bo_results" / "gp_train_data.pt"))
    p.add_argument("--min_vx", type=float, default=0.5)
    p.add_argument("--max_csvs", type=int, default=0, help="0 = 전체")
    args = p.parse_args()

    if args.csv:
        files = [Path(args.csv)]
    else:
        # mtime 기준 cutoff. autoreg overnight 데이터 자동 포함.
        all_csvs = sorted(MPC_LOGS.glob("mpc_*.csv"))
        files = [f for f in all_csvs
                 if f.stat().st_mtime >= POST_TANH_CUTOFF_EPOCH]
        print(f"Found {len(files)} CSVs after cutoff "
              f"(epoch {POST_TANH_CUTOFF_EPOCH})")
        if args.max_csvs > 0:
            files = files[:args.max_csvs]

    print(f"Processing {len(files)} CSV(s)...")

    X_all, Y_all = [], []
    for f in files:
        x, y = process_csv(f, min_vx=args.min_vx)
        if len(x) > 0:
            X_all.append(x)
            Y_all.append(y)
            if args.csv or args.max_csvs <= 5:
                print(f"  {f.name}: {len(x)} samples  "
                      f"vy std={np.std(x[:,1]):.3f}  r std={np.std(x[:,2]):.3f}  "
                      f"res_vx std={np.std(y[:,0]):.4f}")

    if not X_all:
        print("No valid samples. Check filters.")
        return

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)

    print(f"\nTotal: {len(X)} samples from {len(files)} CSVs")
    print(f"Input  (5D) range: [vx, vy, r, delta, a_x]")
    for i, name in enumerate(['vx', 'vy', 'r', 'delta', 'a_x']):
        print(f"  {name:8s}: min={X[:,i].min():.3f}  max={X[:,i].max():.3f}  "
              f"mean={X[:,i].mean():.3f}  std={X[:,i].std():.3f}")
    print(f"Residual (3D) [d_vx, d_vy, d_r]:")
    for i, name in enumerate(['d_vx', 'd_vy', 'd_r']):
        print(f"  {name:8s}: min={Y[:,i].min():.4f}  max={Y[:,i].max():.4f}  "
              f"mean={Y[:,i].mean():.4f}  std={Y[:,i].std():.4f}")

    # Sanity: residuals should be O(dt) for accurate model. Big residuals indicate
    # missing physics (= what GP will learn).
    res_norm = np.linalg.norm(Y, axis=1)
    print(f"\nResidual ‖·‖ percentiles: "
          f"p50={np.percentile(res_norm, 50):.4f} "
          f"p95={np.percentile(res_norm, 95):.4f} "
          f"max={res_norm.max():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'x_train': torch.tensor(X, dtype=torch.float32),
        'y_train': torch.tensor(Y, dtype=torch.float32),
        'input_keys': ['vx', 'vy', 'r', 'delta', 'a_x'],
        'output_keys': ['d_vx', 'd_vy', 'd_r'],
        'dt': DT,
        'n_csvs': len(files),
    }, str(out_path))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
