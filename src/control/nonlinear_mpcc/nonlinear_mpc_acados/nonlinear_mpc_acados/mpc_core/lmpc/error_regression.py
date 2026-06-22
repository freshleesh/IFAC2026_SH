"""B4' local error regression: Epanechnikov-weighted mean of SS-neighbour
velocity residuals -> a single affine correction e_corr (vx,vy,r).
Data-scarcity & magnitude safety built in (paper's nominal fallback)."""
import numpy as np


def epanechnikov_e_corr(residuals, distances, h=1.0, m_min=3, max_norm=3.0):
    """residuals: (K,3), distances: (K,) weighted-L2 to query.
    Returns (3,) correction, or zeros if fewer than m_min usable neighbours."""
    residuals = np.asarray(residuals, float).reshape(-1, 3)
    distances = np.asarray(distances, float).reshape(-1)
    if residuals.shape[0] < m_min or h <= 0:
        return np.zeros(3)
    u = distances / h
    w = np.maximum(0.0, 1.0 - u * u)            # Epanechnikov kernel
    sw = w.sum()
    if sw <= 1e-9:
        return np.zeros(3)
    e = (w[:, None] * residuals).sum(axis=0) / sw
    n = np.linalg.norm(e)
    if n > max_norm:
        e = e * (max_norm / n)
    return e
