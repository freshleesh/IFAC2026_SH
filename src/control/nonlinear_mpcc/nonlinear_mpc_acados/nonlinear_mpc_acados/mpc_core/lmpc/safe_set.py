"""SafeSet kNN lookup for LMPC terminal cost.

Given a query state z_t (= MPC's predicted horizon-end x_N) and a v_bucket,
return the K nearest historical states from the LapDatabase along with their
cost-to-go values. These are fed to the acados terminal cost as parameters.

LMPC formulation (simplified for acados, Rosolia 2018 §IV.B):
    terminal_cost(x_N) = softmin_{(x*, q*) ∈ SS_K} ( q* + α · ‖x_N - x*‖²_W )
                       = -1/β · log( Σ exp(-β·(q* + α·d²)) )

We compute SS_K here on the CPU; acados gets the (x*_i, q*_i)_i=1..K
as flat parameter vector.

Distance metric: weighted L2 on (x, y, ψ, vx) — position dominant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .lap_database import LapDatabase, LapEntry, quantize_v


# Indices for our 8-state dynamic Pacejka model:
#   x = [px, py, psi, vx, vy, r, s, delta_prev]
_IDX_PX, _IDX_PY, _IDX_PSI, _IDX_VX = 0, 1, 2, 3
_IDX_S = 6


@dataclass
class SafeSetQuery:
    """Result of SafeSet lookup — ready to pack into acados parameters."""
    states: np.ndarray              # (K, n_state) — closest historical points
    cost_to_go: np.ndarray          # (K,) — Rosolia step-count cost
    residuals: np.ndarray           # (K, 3) — per-neighbour velocity residual
    distances: np.ndarray           # (K,) — measured distance to query (debug)
    K: int                          # effective K returned (may be < requested if SS small)
    used_buckets: List[float]       # which v_buckets contributed

    @property
    def is_ready(self) -> bool:
        return self.K > 0


class SafeSetLookup:
    """Wraps the LapDatabase + provides per-cycle kNN query.

    Per-cycle cost (CPU): build (M, n_state) candidate from K_laps × Tslice,
    then argpartition K nearest. For T~500 and K_laps=4 → M~2000 points,
    distance vectorized → < 1 ms.
    """

    def __init__(
        self,
        db: LapDatabase,
        K_points: int = 10,
        K_laps: int = 4,
        slice_window: int = 50,     # ± slice around predicted s — narrows search
        weights: Optional[np.ndarray] = None,
    ):
        self.db = db
        self.K_points = int(K_points)
        self.K_laps = int(K_laps)
        self.slice_window = int(slice_window)
        # Reviewer 2026-05-28 #4-D: ψ-weight 0.5 → 1.5 (heading 1 rad 차이 = px 1m
        # 와 비슷한 영향). vx weight 0.3 유지 (warm_transfer 는 #4-B 의 vx rescale
        # 로 mismatch 해결).
        if weights is None:
            self.W = np.array([1.0, 1.0, 1.5, 0.3, 0.05, 0.05, 0.05, 0.05])
        else:
            self.W = np.asarray(weights, dtype=float)

    # ----------------------------------------------------------
    def query(
        self,
        z_t: np.ndarray,
        v_bucket: float,
        s_curr: Optional[float] = None,
        track_length: Optional[float] = None,
    ) -> SafeSetQuery:
        """Return K nearest SS points + their cost-to-go for the given v_bucket.

        z_t       : (n_state,) predicted horizon-end state (or current state)
        v_bucket  : effective v_max
        s_curr    : current arc length (for window slicing). None = full lap.
        """
        z_t = np.asarray(z_t, dtype=float).flatten()
        used_buckets: List[float] = []

        # 1) Pull recent laps from this bucket (and warm-transfer neighbor if needed)
        laps = self.db.get_recent(v_bucket, K_laps=self.K_laps)
        used_buckets.append(quantize_v(v_bucket))
        # If too few, augment with the immediately-lower bucket (warm transfer)
        if len(laps) < 2:
            v_lower = quantize_v(v_bucket - 0.5)
            extra = self.db.get_recent(v_lower, K_laps=self.K_laps - len(laps))
            laps = laps + extra
            if extra:
                used_buckets.append(v_lower)

        if not laps:
            return SafeSetQuery(
                states=np.zeros((0, z_t.size)),
                cost_to_go=np.zeros(0),
                residuals=np.zeros((0, 3)),
                distances=np.zeros(0),
                K=0,
                used_buckets=[],
            )

        # 2) Concatenate candidate points across selected laps, with optional s-window slicing
        def _resid_of(e):
            r = getattr(e, 'residual', None)
            if r is None or r.shape[0] != e.state.shape[0]:
                return np.zeros((e.state.shape[0], 3))
            return r
        cand_states_list = []
        cand_cost_list = []
        cand_resid_list = []
        for e in laps:
            if s_curr is not None and self.slice_window > 0 and e.state.shape[1] > _IDX_S:
                s_arr = e.state[:, _IDX_S]
                # Reviewer 2026-05-28 #4-E: Frenet s discontinuity (lap rollover).
                # state[:,6]=s 가 0 으로 점프 — slice 가 s_curr ≈ L 근처에서 끊김.
                # Modular distance: signed_diff = ((s_i - s_curr + L/2) mod L) - L/2
                if track_length is not None and track_length > 0:
                    d = (s_arr - s_curr + 0.5 * track_length) % track_length - 0.5 * track_length
                else:
                    d = s_arr - s_curr
                idx_near = int(np.argmin(np.abs(d)))
                lo = max(0, idx_near - self.slice_window)
                hi = min(e.state.shape[0], idx_near + self.slice_window + 1)
                cand_states_list.append(e.state[lo:hi])
                cand_cost_list.append(e.cost_to_go[lo:hi])
                cand_resid_list.append(_resid_of(e)[lo:hi])
            else:
                cand_states_list.append(e.state)
                cand_cost_list.append(e.cost_to_go)
                cand_resid_list.append(_resid_of(e))

        cand_states = np.vstack(cand_states_list)
        cand_cost = np.concatenate(cand_cost_list)
        cand_resid = np.vstack(cand_resid_list) if cand_resid_list else np.zeros((0, 3))

        if cand_states.shape[0] == 0:
            return SafeSetQuery(
                states=np.zeros((0, z_t.size)),
                cost_to_go=np.zeros(0),
                residuals=np.zeros((0, 3)),
                distances=np.zeros(0),
                K=0,
                used_buckets=used_buckets,
            )

        # 3) Weighted L2 distance (broadcast)
        n_state = z_t.size
        W = self.W[:n_state] if self.W.size >= n_state else np.pad(self.W, (0, n_state - self.W.size), constant_values=0.1)
        diff = (cand_states - z_t[None, :]) * np.sqrt(W)[None, :]
        d2 = np.sum(diff * diff, axis=1)

        # 4) Pick K_points nearest by weighted L2 distance (feasibility: the
        #    attractor must be reachable from the query state).
        K = min(self.K_points, d2.size)
        if K == d2.size:
            near = np.argsort(d2)
        else:
            order_part = np.argpartition(d2, K)[:K]
            near = order_part[np.argsort(d2[order_part])]

        # 5) Re-order the K nearest by cost-to-go ASCENDING so states[0] is the
        #    lowest-time-to-go (furthest-along the lap) reachable point. The
        #    acados terminal cost uses ss_states[:,0] as the single attractor
        #    and ASSUMES it is the lowest-cost-to-go point (see acados_kinematic
        #    comment "caller sorts Q ascending"). Returning them distance-sorted
        #    made the attractor the spatially-nearest point (~the query state)
        #    → zero forward-progress pull → LMPC could not reduce lap time.
        #    Sorting by cost-to-go makes [:,0] the forward carrot among the
        #    reachable neighbours, restoring the Rosolia progress mechanism.
        order = near[np.argsort(cand_cost[near], kind='stable')]

        return SafeSetQuery(
            states=cand_states[order],
            cost_to_go=cand_cost[order],
            residuals=cand_resid[order],
            distances=np.sqrt(d2[order]),
            K=K,
            used_buckets=used_buckets,
        )

    # ----------------------------------------------------------
    @staticmethod
    def softmin_value(
        q_arr: np.ndarray,
        d2_arr: np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> float:
        """Reference smooth value: −(1/β)·log Σ exp(−β·(q + α·d²)).

        This is what acados will compute inside the cost expression; provided
        here for CPU-side debugging / monitoring.
        """
        if q_arr.size == 0:
            return float("inf")
        e = -beta * (q_arr + alpha * d2_arr)
        m = float(np.max(e))
        return -(1.0 / beta) * (m + float(np.log(np.sum(np.exp(e - m)))))
