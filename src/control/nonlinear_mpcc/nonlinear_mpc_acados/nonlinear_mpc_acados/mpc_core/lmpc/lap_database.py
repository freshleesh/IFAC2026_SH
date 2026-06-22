"""Lap trajectory database for LMPC.

Stores closed-loop trajectories per lap and per `v_bucket` (effective v_max).
Each lap entry contains state, input sequences plus a precomputed cost-to-go
in the LMPC sense (Rosolia 2018, eq. 13): `Cost[T-1]=0, Cost[i] = Cost[i+1]+1`.

User-facing concept:
  - `add_lap(...)` after a successful lap → SS for that v_bucket grows
  - `get_recent(v_bucket, K_laps)` → list of (state, cost) ready for SS lookup
  - `save_all(path)`, `load_all(path)` — npz persistence

`failed lap discard` rule:
  - We accept laps with `lap_time` finite and `n_resets <= max_resets`
  - bad laps (crash, runaway STUCK) skipped — protects SS from noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def quantize_v(v: float, step: float = 0.5) -> float:
    """Bucket a continuous v_max into discrete steps (5.0, 5.5, 6.0, ...)."""
    return round(round(v / step) * step, 1)


@dataclass
class LapEntry:
    """One lap of closed-loop data + cost-to-go."""
    v_bucket: float                # quantized v_max for this lap
    v_max_eff: float               # exact mpc.v_max during the lap
    state: np.ndarray              # (T, n_state)
    input: np.ndarray              # (T-1, n_input)
    time_step: np.ndarray          # (T,) wall-time in sec (relative to lap start)
    cost_to_go: np.ndarray         # (T,) Rosolia eq.13: T-1-t (step count to end)
    lap_time: float                # T * dt (or measured wall time)
    n_resets: int                  # number of safe_resets during this lap
    metadata: dict = field(default_factory=dict)
    residual: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (T,3) one-step velocity error

    @property
    def T(self) -> int:
        return self.state.shape[0]


class LapDatabase:
    """Per-v_bucket circular buffer of LapEntry."""

    def __init__(
        self,
        buffer_per_bucket: int = 10,
        max_resets_accept: int = 0,   # 2026-06-08: teleport된 랩(reset>0)은 safe-set 오염 → 기본 거부
        min_lap_steps: int = 50,
        # Reviewer 2026-05-28 #4-A: failed-lap 필터 강화
        max_lap_time_ratio: float = 1.5,    # > 1.5 × best_in_bucket → reject
        max_abs_ec_m: float = 1.0,           # max(|e_c|) > 이 값 → reject
        max_stuck_seconds: float = 5.0,     # cumulative STUCK time > 이 값 → reject
    ):
        self.buffer_per_bucket = int(buffer_per_bucket)
        self.max_resets_accept = int(max_resets_accept)
        self.min_lap_steps = int(min_lap_steps)
        self.max_lap_time_ratio = float(max_lap_time_ratio)
        self.max_abs_ec_m = float(max_abs_ec_m)
        self.max_stuck_seconds = float(max_stuck_seconds)
        # {v_bucket : list of LapEntry, newest at end}
        self._db: Dict[float, List[LapEntry]] = {}

    # ----------------------------------------------------------
    def add_lap(
        self,
        v_max_eff: float,
        state: np.ndarray,
        input_seq: np.ndarray,
        time_step: np.ndarray,
        lap_time: float,
        n_resets: int = 0,
        dt: float = 0.04,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Append lap if it passes acceptance filter. Returns True if stored.

        Reviewer 2026-05-28 #4-A: 다중 필터 — sim variance 견고화.
        - min steps, max resets, finite lap_time (기본)
        - lap_time < 1.5 × best_in_bucket  (wandering lap 제거)
        - max(|e_c|) < 1.0 m              (corridor 이탈한 lap state 는 SS 오염)
        - stuck_seconds < 5.0              (metadata 에 누적 STUCK 시간)
        """
        self.last_reject_reason = ""
        if state.shape[0] < self.min_lap_steps:
            self.last_reject_reason = f"min_steps {state.shape[0]}<{self.min_lap_steps}"
            return False
        if n_resets > self.max_resets_accept:
            self.last_reject_reason = f"n_resets {n_resets}>{self.max_resets_accept}"
            return False
        if not np.isfinite(lap_time) or lap_time <= 0:
            self.last_reject_reason = f"lap_time {lap_time} not positive finite"
            return False

        # filter: max |e_c| — state[:, 6] = s, e_c 는 별도 args 로 전달받을 수도.
        # metadata 에 있으면 사용.
        if metadata and "max_abs_ec" in metadata:
            mae = float(metadata["max_abs_ec"])
            if mae > self.max_abs_ec_m:
                self.last_reject_reason = f"max_abs_ec {mae:.2f}>{self.max_abs_ec_m:.2f}"
                return False

        # filter: stuck_seconds
        if metadata and "stuck_seconds" in metadata:
            ss = float(metadata["stuck_seconds"])
            if ss > self.max_stuck_seconds:
                self.last_reject_reason = f"stuck_seconds {ss:.2f}>{self.max_stuck_seconds:.2f}"
                return False

        # filter: lap_time > 1.5 × best_in_bucket  (단 첫 lap 은 통과 시켜야 best 형성)
        v_b = quantize_v(v_max_eff)
        existing_best = self.best_lap_time(v_b)
        if np.isfinite(existing_best) and lap_time > self.max_lap_time_ratio * existing_best:
            self.last_reject_reason = (
                f"lap_time {lap_time:.2f}>{self.max_lap_time_ratio:.1f}×best "
                f"({existing_best:.2f}) in bucket v={v_b:.1f}"
            )
            return False

        # Cost-to-go: Rosolia 식 — backward "step count to end"
        T = state.shape[0]
        cost_to_go = np.arange(T - 1, -1, -1, dtype=float)

        # B4' one-step velocity residual per transition (last row left 0 so the
        # residual aligns index-for-index with `state` for SS slicing).
        from .nominal_dynamics import velocity_residual
        residual = np.zeros((T, 3))
        _inp = np.asarray(input_seq)
        if _inp.ndim == 2 and _inp.shape[1] >= 3:
            for t in range(min(T - 1, _inp.shape[0])):
                residual[t] = velocity_residual(state[t], _inp[t], state[t + 1], dt)

        entry = LapEntry(
            v_bucket=v_b,
            v_max_eff=float(v_max_eff),
            state=np.asarray(state, dtype=float),
            input=np.asarray(input_seq, dtype=float),
            time_step=np.asarray(time_step, dtype=float),
            cost_to_go=cost_to_go,
            residual=residual,
            lap_time=float(lap_time),
            n_resets=int(n_resets),
            metadata=dict(metadata or {}),
        )

        if v_b not in self._db:
            self._db[v_b] = []
        self._db[v_b].append(entry)
        # Retain the BEST laps: when over capacity drop the WORST (highest
        # lap_time) entry, not the oldest. FIFO eviction discarded the fast
        # early laps, leaving only recent (slower) ones → the LMPC attractor
        # degraded and lap-time drifted UP. Keeping the best trajectories is
        # what makes Rosolia LMPC monotonically improve.
        if len(self._db[v_b]) > self.buffer_per_bucket:
            worst_i = max(range(len(self._db[v_b])),
                          key=lambda i: self._db[v_b][i].lap_time)
            self._db[v_b].pop(worst_i)
        return True

    # ----------------------------------------------------------
    def get_recent(self, v_bucket: float, K_laps: int = 4) -> List[LapEntry]:
        """The K BEST laps (lowest lap_time) for this bucket, best first.

        (Name kept for API compatibility.) Anchoring the safe set to the
        fastest trajectories — not the most recent — makes the LMPC attractor
        pull toward the best line achieved so far (monotonic improvement).
        Using the most-recent laps caused a positive-feedback drift to slower
        lap times as the safe set forgot the fast early laps.
        """
        v_b = quantize_v(v_bucket)
        if v_b not in self._db or not self._db[v_b]:
            return []
        return sorted(self._db[v_b], key=lambda e: e.lap_time)[:K_laps]

    # ----------------------------------------------------------
    def warm_transfer(self, v_from: float, v_to: float) -> bool:
        """Promote best lap from v_from bucket to v_to bucket as a slow seed.

        Used when ramp steps v_max up — bootstraps the new bucket with the
        proven trajectory from the lower v. cost_to_go is recomputed in-place.
        Returns True if a lap was promoted.
        """
        v_from_q = quantize_v(v_from)
        v_to_q = quantize_v(v_to)
        if v_from_q not in self._db or not self._db[v_from_q]:
            return False
        best = min(self._db[v_from_q], key=lambda e: e.lap_time)
        if v_to_q not in self._db:
            self._db[v_to_q] = []
        # If bucket already has laps, don't overwrite — transfer only seeds an
        # empty bucket. (caller decides when to call.)
        if self._db[v_to_q]:
            return False
        # Reviewer 2026-05-28 #4-B: warm_transfer 의 dynamic mismatch fix.
        # Seed 의 vx 가 v_from (예: 5.5) 인데 v_to (예: 6.0) 에서 사용하면
        # SafeSet 의 distance metric 이 항상 seed (vx=5.5) 쪽으로 끌어당김
        # → 차가 v=6 으로 안 가속됨. vx 컬럼 rescale: vx_new = vx_old × v_to/v_from.
        T = best.state.shape[0]
        state_rescaled = best.state.copy()
        if v_from_q > 1e-3:
            v_scale = float(v_to_q) / float(v_from_q)
            state_rescaled[:, 3] *= v_scale   # vx col
            # vy, r 는 그대로 (slip 정보 — 의미 보존)
        seed = LapEntry(
            v_bucket=v_to_q,
            v_max_eff=v_to,
            state=state_rescaled,
            input=best.input.copy(),
            time_step=best.time_step.copy(),
            cost_to_go=np.arange(T - 1, -1, -1, dtype=float),
            lap_time=best.lap_time * (v_from_q / max(v_to_q, 1e-3)),  # 시간도 ~ scale
            n_resets=best.n_resets,
            metadata={**best.metadata, "warm_from_v": v_from_q, "vx_rescaled": True},
        )
        self._db[v_to_q].append(seed)
        return True

    # ----------------------------------------------------------
    def seed_from_raceline(
        self,
        v_bucket: float,
        raceline_xy: np.ndarray,
        raceline_psi: np.ndarray,
        raceline_v: np.ndarray,
        raceline_s: np.ndarray,
        dt: float = 0.04,
    ) -> bool:
        """Inject an offline IQP raceline as a synthetic 'lap 0' for cold-start.

        Map-aware LMPC: instead of relying on a slow PID lap, we know the
        optimal centerline-minimum-curvature line and the kinematic feasible
        speed profile. Wrap them as a LapEntry so the SS has a viable seed
        from the first MPC cycle.

        cost-to-go is generated the same way (T-1-t step count), giving the
        synthetic lap a baseline lap_time estimate = T·dt.

        Note: the synthetic state is filled only in (px, py, psi, vx) — the
        unobserved dims (vy, r, s, δ) are set to 0 (raceline carries no slip
        info). SafeSet metric weights vy/r/δ near zero so they don't dominate.
        """
        T = raceline_xy.shape[0]
        if T < self.min_lap_steps:
            return False
        # 8-state: [px, py, psi, vx, vy, r, s, delta_prev]
        state = np.zeros((T, 8))
        state[:, 0] = raceline_xy[:, 0]
        state[:, 1] = raceline_xy[:, 1]
        state[:, 2] = raceline_psi
        state[:, 3] = raceline_v
        state[:, 6] = raceline_s
        # Approximate lap time = path length / mean v
        ds = np.linalg.norm(np.diff(raceline_xy, axis=0), axis=1)
        path_len = float(np.sum(ds))
        v_mean = float(np.mean(np.clip(raceline_v, 0.1, None)))
        lap_time = path_len / v_mean

        # Synthesize inputs as zero (no acceleration / steer rate info available
        # offline). They aren't read by the cost-to-go lookup.
        inp = np.zeros((T - 1, 2))
        t = np.linspace(0.0, T * dt, T)
        ok = self.add_lap(
            v_max_eff=v_bucket,
            state=state, input_seq=inp, time_step=t,
            lap_time=lap_time, n_resets=0,
            metadata={"synthetic": True, "source": "raceline_seed"},
        )
        # 2026-06-01 (rebuild Step 8): REMOVED the +1000 seed cost-to-go inflation.
        # The apex seed is FASTER (fewer steps → lower cost-to-go) so the joint-α
        # terminal (which minimizes Σα·Q) SHOULD prefer it — that is the whole point
        # of seeding apex. The old +1000 inverted this → α picked the real centerline
        # laps → car never cut apex (measured: ec stuck ~0.4 through Steps 2-7). The
        # original slip-safety concern is now handled by the SOFT anchor + HARD
        # corridor (joint-α), not by hiding the seed. Tiny tie-break so a genuinely
        # faster real lap can still win once achieved.
        if ok:
            v_b = quantize_v(v_bucket)
            laps = self._db.get(v_b, [])
            # Inflate the SEED's cost-to-go, not laps[-1]: add_lap may have
            # evicted the seed (if it was the worst lap in a full bucket), in
            # which case laps[-1] is a faster REAL lap that must not be penalized.
            seed = next((e for e in reversed(laps)
                         if e.metadata.get("source") == "raceline_seed"), None)
            if seed is not None:
                seed.cost_to_go = seed.cost_to_go + 5.0
        return ok

    def n_laps(self, v_bucket: float) -> int:
        v_b = quantize_v(v_bucket)
        return len(self._db.get(v_b, []))

    def n_real_laps(self, v_bucket: float) -> int:
        """Count non-synthetic laps (Reviewer 2026-05-28 #4-C — gate LMPC on)."""
        v_b = quantize_v(v_bucket)
        if v_b not in self._db:
            return 0
        return sum(1 for e in self._db[v_b]
                   if not e.metadata.get("synthetic", False))

    def best_lap_time(self, v_bucket: float) -> float:
        v_b = quantize_v(v_bucket)
        if v_b not in self._db or not self._db[v_b]:
            return float("inf")
        return min(e.lap_time for e in self._db[v_b])

    def buckets(self) -> List[float]:
        return sorted(self._db.keys())

    # ----------------------------------------------------------
    def save_all(self, path: str | Path) -> None:
        """Persist all laps to a single npz (per-lap-per-bucket arrays)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        meta = []
        for v_b, laps in self._db.items():
            for i, e in enumerate(laps):
                key = f"v{v_b:.1f}_lap{i}"
                data[f"{key}_state"] = e.state
                data[f"{key}_input"] = e.input
                data[f"{key}_t"]     = e.time_step
                data[f"{key}_q"]     = e.cost_to_go
                data[f"{key}_res"]   = e.residual   # B4' one-step velocity residual
                meta.append((v_b, e.v_max_eff, i, e.lap_time, e.n_resets))
        if meta:
            arr = np.array(meta, dtype=[
                ("v_bucket", "f8"), ("v_max_eff", "f8"),
                ("lap_idx", "i4"), ("lap_time", "f8"),
                ("n_resets", "i4"),
            ])
            data["_index"] = arr
        np.savez_compressed(str(path), **data)

    def load_all(self, path: str | Path) -> None:
        """Reload laps from npz produced by save_all()."""
        path = Path(path)
        if not path.exists():
            return
        z = np.load(str(path), allow_pickle=True)
        if "_index" not in z.files:
            return
        idx = z["_index"]
        self._db = {}
        for row in idx:
            v_b = float(row["v_bucket"])
            v_eff = float(row["v_max_eff"])
            lap_i = int(row["lap_idx"])
            lt = float(row["lap_time"])
            nr = int(row["n_resets"])
            base = f"v{v_b:.1f}_lap{lap_i}"
            try:
                # residual is optional for back-compat with pre-2026-06 npz files
                res_key = f"{base}_res"
                residual = z[res_key] if res_key in z.files else np.zeros((0, 3))
                e = LapEntry(
                    v_bucket=v_b, v_max_eff=v_eff,
                    state=z[f"{base}_state"], input=z[f"{base}_input"],
                    time_step=z[f"{base}_t"], cost_to_go=z[f"{base}_q"],
                    lap_time=lt, n_resets=nr, residual=residual,
                )
            except KeyError:
                continue
            self._db.setdefault(v_b, []).append(e)

    # ----------------------------------------------------------
    def summary(self) -> str:
        if not self._db:
            return "(empty database)"
        rows = []
        for v_b in sorted(self._db.keys()):
            laps = self._db[v_b]
            best = min(e.lap_time for e in laps)
            mean = sum(e.lap_time for e in laps) / len(laps)
            rows.append(f"  v={v_b:.1f}m/s : {len(laps):2d} laps  best={best:.2f}s  mean={mean:.2f}s")
        return "LapDatabase:\n" + "\n".join(rows)
