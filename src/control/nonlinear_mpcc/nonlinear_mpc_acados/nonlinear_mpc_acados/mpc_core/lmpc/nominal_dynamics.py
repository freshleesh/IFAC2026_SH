"""Numpy mirror of acados f_expl (8-state Pacejka tanh blend) + one-step
velocity residual. Single source of truth for B4' error regression.

Constants copied verbatim from scripts/extract_residuals.py:33-45. That script
keeps its own duplicate mirror; Task 5's cross-check guards against drift.
"""
import numpy as np

L_WB = 0.307
M = 3.54
IZ = 0.05797
LF = 0.162
LR = 0.145
H_CG = 0.014
MU = 1.0
BF = BR = 10.0
DF = DR = 1.0
G = 9.81
DT = 0.04
V_B = 0.5
V_S = 0.3


def f_dynamic(state, u):
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
    vx = state[3]
    w = 0.5 * (1.0 + np.tanh((vx - V_B) / V_S))
    return w * f_dynamic(state, u) + (1.0 - w) * f_kinematic(state, u)


def predict_next(state, u, dt):
    """One Euler step of the nominal blended dynamics. Returns (8,)."""
    return np.asarray(state, float) + dt * f_expl(np.asarray(state, float),
                                                  np.asarray(u, float))


def velocity_residual(state, u, next_state, dt):
    """actual_next[vx,vy,r] - nominal_predicted[vx,vy,r]. Returns (3,)."""
    pred = predict_next(state, u, dt)
    return np.asarray(next_state, float)[3:6] - pred[3:6]
