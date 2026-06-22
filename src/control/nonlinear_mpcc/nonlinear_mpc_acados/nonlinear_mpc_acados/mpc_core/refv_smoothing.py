"""ref_v forward-curvature smoothing (R4).

The MPCC speed cap is v_cap = sqrt(a_lat / kappa_fwd), where kappa_fwd is the
"binding |kappa| ahead" over a lookahead window. The original implementation took
a HARD max over [s, s+lookahead]: a sharp corner crossing the window edge made
kappa_fwd (hence ref_v, hence the whole predicted trajectory) JUMP step-wise as
the car advanced -> visible trajectory shake at higher speed.

distance_tapered_forward_max keeps full kappa inside the window but TAPERS the
last `taper` samples to zero weight at the far edge, so a corner entering the
window ramps in continuously (no step) while early braking is preserved (full
kappa once the corner is past the taper region).
"""
import numpy as np


def distance_tapered_forward_max(abs_k, n_look, taper):
    """Continuous forward-window max of |kappa|.

    abs_k : (N,) non-negative curvature magnitude on the arc-length grid.
    n_look: window length in grid samples.
    taper : number of samples at the FAR edge over which the weight ramps 1->0.

    Returns (N,) where out[i] = max_d( w[d] * abs_k[i+d] ),
    w[d] = clip((n_look - d) / taper, 0, 1)  -> 1 for the near/interior part of
    the window, ramping linearly to 0 at the far edge (d = n_look).
    """
    abs_k = np.asarray(abs_k, dtype=float)
    n = abs_k.shape[0]
    n_look = int(n_look)
    taper = max(1, int(taper))
    out = np.empty(n, dtype=float)
    for i in range(n):
        j = min(i + n_look, n)
        seg = abs_k[i:j]
        if seg.size == 0:
            out[i] = 0.0
            continue
        d = np.arange(seg.size, dtype=float)
        w = np.clip((n_look - d) / float(taper), 0.0, 1.0)
        out[i] = float(np.max(w * seg))
    return out
