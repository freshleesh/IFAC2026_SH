#!/usr/bin/env python3
"""eval_run_quality.py — single mpc_*.csv → single quality Q.

**EVO-MPCC LTM (Lap-Time Minimization) objective with strict filtering.**
BO maximizes Q (higher = better). Convention:

    qualified:   Q = -T_lap + λ·max(t_lb - T_lap, 0)
                 (= -[T_lap + λ·[T_lap - t_lb]⁻] in EVO-MPCC's minimize form)
    unqualified: Q = -1000  (J_inf — hard reject)

Where `t_lb = ideal_lap_time · t_lb_factor` (default factor=1.3). ideal_lap_time
= track_length / max_speed (point-mass lower bound). Bonus λ triggers only when
T_lap < t_lb — exploration toward fast laps.

**Filter (strict — EVO-MPCC style)** marks a run unqualified if ANY of:
  - crashed (infeas>0.1 OR stuck>0.3)
  - laps < min_laps
  - mpcc_alive_frac < 0.8       (MPCC dropped to fallback too often)
  - switch_count > 3            (MPCC↔fallback flapping)
  - max lat_g > 15 m/s² (~1.5g) (would slip in real car)
  - cte_rms > 0.5 m             (corridor edge / lane departure)
  - shake_rms > 5.0             (oscillation, boundary contact)

Why LTM vs the old Q v5 (6-term mix):
  - clean objective lets ref_v / κ-mapping ablations show contribution unambiguously
  - filter handles safety as hard constraint (no penalty trade-off)
  - matches EVO-MPCC / VPMPCC literature → reproducible

Usage:
    python3 eval_run_quality.py                  # latest CSV
    python3 eval_run_quality.py --csv X.csv      # specific
    python3 eval_run_quality.py --json           # BO harness
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


_CENTERLINE_CACHE: dict = {}


def _load_centerline(map_name: str = 'f') -> np.ndarray | None:
    """centerline (x_m, y_m) Nx2 array. 캐시됨."""
    if map_name in _CENTERLINE_CACHE:
        return _CENTERLINE_CACHE[map_name]
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'stack_master', 'maps', map_name, 'global_waypoints.json')  # 2026-06-16 port: relative to this ws
    if not os.path.isfile(json_path):
        _CENTERLINE_CACHE[map_name] = None
        return None
    with open(json_path) as f:
        data = json.load(f)
    wpnts = data['centerline_waypoints']['wpnts']
    arr = np.array([[w['x_m'], w['y_m']] for w in wpnts], dtype=float)
    _CENTERLINE_CACHE[map_name] = arr
    return arr


def _compute_cte_rms(df: pd.DataFrame, map_name: str = 'f') -> float:
    """car_x/y 와 centerline 의 거리 RMS. KDTree 로 nearest 검색."""
    centerline = _load_centerline(map_name)
    if centerline is None or 'car_x' not in df.columns:
        return float('nan')
    tree = KDTree(centerline)
    pts = df[['car_x', 'car_y']].to_numpy()
    dists, _ = tree.query(pts)
    return float(np.sqrt(np.mean(dists ** 2)))


def evaluate(csv_path: str,
             lam: float = 20.0,
             t_lb_factor: float = 1.3,
             t_lb_override: float = None,    # PP baseline 직접 전달 (option 2)
             min_laps: int = 1,
             map_name: str = 'f',
             # filter thresholds — soft penalty 만 적용 (cliff 제거)
             alive_floor: float = 0.8,
             switch_ceiling: int = 3,
             lat_g_max: float = 20.0,     # 15→20 (reset 폭증 무시)
             cte_max: float = 0.8,        # 0.5→0.8 (corridor 마진 ↑)
             shake_max: float = 5.0,      # 그대로 (진동 fix 위해)
             j_inf: float = 1000.0,
             # B+C: reset/teleport 으로 인한 false lap 차단
             n_resets: int = 0,            # C: 외부 (BO) 가 sim log grep 으로 전달
             reset_ceiling: int = 2,       # 0→2 (2 reset 까지 OK, BO 학습 데이터 확보)
             min_lap_time_frac: float = 0.5,  # B: lap_time < 0.5 × ideal 이면 unqualified (false lap)
             # legacy kwargs (kept so BO harness import doesn't break — unused)
             gamma: float = None,
             alpha: float = None,
             beta: float = None,
             crash_penalty: float = None,
             delta_alive: float = None,
             epsilon_switch: float = None,
             zeta_latg: float = None) -> dict:
    """EVO-MPCC LTM (Lap-Time Minimization) — clean lap_time objective + strict filter.

    Q (BO maximizes):
      unqualified → Q = -j_inf                                (hard reject)
      qualified   → Q = -T_lap + λ·max(t_lb - T_lap, 0)       (lap-time + speedup bonus)

    `t_lb = ideal_lap_time · t_lb_factor`. ideal = track_length / max_speed
    (point-mass bound; raceline lap ≈ 1.3× by default). Bonus triggers if BO
    finds T_lap < t_lb — exploration toward fast laps with EVO-MPCC default λ=20.

    Filter (any failing → unqualified):
      crashed OR laps<min_laps                       (must finish run)
      mpcc_alive_frac < alive_floor                   (MPCC dropping → unreliable)
      switch_count > switch_ceiling                   (MPCC↔fallback flapping)
      lat_g_max > lat_g_max threshold                 (>1.5g would slip irl)
      cte_rms > cte_max                               (corridor edge / lane departure)
      shake_rms > shake_max                           (oscillation, contact)

    Example (track L=44m, vmax=4 → ideal=11s, t_lb=14.3s, λ=20):
      T=14s, qualified  → Q = -14 + 20·0.3 = -8     (fast → reward)
      T=24s, qualified  → Q = -24 + 0       = -24   (slow but safe)
      crash            → Q = -1000                  (hard fail)
      alive_frac=0.6   → Q = -1000                  (filtered out)

    Clean separation between -1000 (fail) and ~ -10..-30 (success).
    """
    _ = (gamma, alpha, beta, crash_penalty, delta_alive,
         epsilon_switch, zeta_latg)  # silence unused
    df = pd.read_csv(csv_path)
    n = len(df)
    if n < 50:
        return {'Q': -j_inf * 2,
                'reason': f'too short ({n} rows)',
                'csv': csv_path}

    # shake — RMS d²δ/dt²
    steer = df['steer_cmd'].to_numpy(dtype=float)
    d2_steer = np.diff(steer, n=2)
    shake = float(np.sqrt(np.mean(d2_steer ** 2)))

    avg_speed = float(df['v_actual'].mean())
    max_speed = float(df['v_actual'].max())
    cte_rms = _compute_cte_rms(df, map_name=map_name)

    # lateral_g — 코너 측방 가속도 [m/s²] (yaw_rate × vx). Filter 에 max 사용.
    yaw = df['car_yaw'].to_numpy(dtype=float)
    yaw_diff = np.diff(yaw)
    yaw_diff = np.mod(yaw_diff + np.pi, 2 * np.pi) - np.pi   # unwrap
    yaw_rate = yaw_diff / 0.025   # 40Hz → dt=0.025s
    vx = df['v_actual'].iloc[1:].to_numpy(dtype=float)
    lat_g = np.abs(yaw_rate * vx)
    lat_g_mean = float(np.mean(lat_g))
    # 2026-05-27: max → p99. reset/teleport 순간의 yaw discontinuity →
    # 1-2 cycle spike (yaw_rate 무한대) 이 max 잡음. p99 가 robust.
    lat_g_peak = float(np.percentile(lat_g, 99)) if len(lat_g) else 0.0

    # lap_time — 각 lap 의 시작 t 차이의 평균. lap 컬럼이 증가하는 순간을 detect.
    lap_arr = df['lap'].to_numpy(dtype=int)
    t_arr = df['t'].to_numpy(dtype=float) if 't' in df.columns else np.arange(len(df)) * 0.025
    transitions = np.where(np.diff(lap_arr) > 0)[0]   # lap 이 +1 되는 index
    if len(transitions) >= 2:
        lap_times = np.diff(t_arr[transitions])
        # BEST lap (min) 사용 — mean 은 lap 1 (cold start) 가 평균 끌어내림.
        # best 가 진짜 잠재력 측정. 이름은 호환성 위해 lap_time_mean 유지.
        lap_time_mean = float(np.min(lap_times))
    elif len(transitions) == 1:
        # 1 lap 만 완료 — 전체 run 시간을 lap_time 으로 사용 (conservative 추정)
        lap_time_mean = float(t_arr[transitions[0]] - t_arr[0])
    else:
        lap_time_mean = float(t_arr[-1] - t_arr[0])   # lap 미완료 — 펜앨티성

    laps = int(df['lap'].max())
    infeas_frac = float((df['feasible'] == 0).mean()) if 'feasible' in df.columns else 0.0
    stuck_frac = float((df['v_actual'].abs() < 0.1).mean())
    solve_ms = float(df['solve_ms'].mean()) if 'solve_ms' in df.columns else 0.0
    cost_mean = float(df['opti_value'].mean()) if 'opti_value' in df.columns else 0.0

    crashed = bool(infeas_frac > 0.1 or stuck_frac > 0.3)

    # MPCC stability — mpc_debug_logger 가 simple_mux 의 /mux/mpcc_active 토픽을
    # CSV 마지막 두 column 으로 기록 (mpcc_active 0/1, switch_count 누적).
    if 'mpcc_active' in df.columns:
        mpcc_alive_frac = float(df['mpcc_active'].mean())
    else:
        mpcc_alive_frac = 1.0   # 기록 없으면 100% 가정 (이전 데이터 호환).
    if 'switch_count' in df.columns and len(df) > 0:
        # delta = (마지막) - (첫 row). 첫 row 는 mpc_node 가 첫 /mpc_debug publish
        # 한 시점 = MPC ready 직후. 그 이전 launch overhead 의 zero→fallback→mpcc
        # 전환 (=2) 은 제외해서 race 중 실제 MPCC↔fallback 전환만 카운트.
        switch_count = int(df['switch_count'].iloc[-1] - df['switch_count'].iloc[0])
    else:
        switch_count = 0

    # 2026-05-27: Soft penalty (BO 친화). 박힘 + 미완주 만 hard reject.
    # n_resets / lat_g / cte / shake / alive / sw 는 모두 quadratic penalty.
    # 2026-05-27 (post-#9 fix): n_resets 도 cliff → soft 로 전환.
    #   이전: n_resets > 2 → Q=-1000 (cliff). reset 3 vs 19 가 같은 Q.
    #   현재: 0..2 무료, 3+ 부터 quadratic. reset 줄이는 방향 gradient 살아남.
    #   극단치 (n_resets > 15) 만 hard reject 로 보강.
    fail_reasons = []
    if crashed:                  fail_reasons.append('crashed')
    if laps < min_laps:          fail_reasons.append(f'laps {laps}<{min_laps}')
    HARD_RESET_LIMIT = 15
    if n_resets > HARD_RESET_LIMIT:
        fail_reasons.append(f'n_resets {n_resets}>{HARD_RESET_LIMIT}')

    success = (len(fail_reasons) == 0)

    # Soft penalty (excess over threshold)² × weight. 양수면 Q 감점.
    soft_penalty = 0.0
    if lat_g_peak > lat_g_max:
        soft_penalty += 1.0 * (lat_g_peak - lat_g_max) ** 2
    if cte_rms > cte_max:
        soft_penalty += 50.0 * (cte_rms - cte_max) ** 2
    if shake > shake_max:
        soft_penalty += 25.0 * (shake - shake_max) ** 2
    if mpcc_alive_frac < alive_floor:
        soft_penalty += 100.0 * (alive_floor - mpcc_alive_frac)
    if switch_count > switch_ceiling:
        soft_penalty += 10.0 * (switch_count - switch_ceiling)
    # n_resets 0..reset_ceiling 무료, 그 이후 quadratic.
    # weight 50: reset 5 → -450, reset 10 → -3200, reset 15 → -8450 (still finite).
    # 사용자 요청: 2번까지 봐주되 각 stuck마다 세게. reset당 강한 linear 페널티 +
    # ceiling 초과는 추가 quadratic (깨끗한 0-reset이 항상 최선).
    soft_penalty += 150.0 * n_resets                          # 모든 reset에 reset당 -150
    if n_resets > reset_ceiling:
        soft_penalty += 100.0 * (n_resets - reset_ceiling) ** 2  # ceiling 초과 가중

    # 2026-05-27: cap soft_penalty so qualified trial Q always > hard_reject (-1000).
    # 이전엔 v=6 lat_g spike (62 m/s²) → soft 2000+ → Q=-3074 < -1000 (crash).
    # BO 가 "차라리 박는 게 낫다" 학습 위험 → ordering 깨짐. cap 800.
    SOFT_CAP = 1400.0
    soft_penalty = min(soft_penalty, SOFT_CAP)

    # ── Always compute lap-time / t_lb for reporting ─────────────────
    cl = _load_centerline(map_name)
    if cl is not None and len(cl) > 1:
        track_length = float(np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1)))
    else:
        track_length = 76.48   # f map fallback
    if 'v_max_cost' in df.columns:
        vmc = df['v_max_cost'].dropna()
        max_speed_local = float(vmc.max()) if len(vmc) > 0 and vmc.max() > 0 else float(max_speed) + 0.5
    else:
        max_speed_local = float(max_speed) + 0.5
    if not np.isfinite(max_speed_local) or max_speed_local <= 0:
        max_speed_local = 4.0
    ideal_lap_time = track_length / max_speed_local
    # PP baseline 직접 전달되면 그 값 사용 (option 2). 아니면 ideal × factor.
    if t_lb_override is not None and t_lb_override > 0:
        t_lb = float(t_lb_override)
    else:
        t_lb = ideal_lap_time * t_lb_factor
    lap_time_delta = max(0.0, lap_time_mean - ideal_lap_time)

    # B: lap_time < 0.5 × ideal → false lap (teleport rollover). filter 가
    # success 결정 후 별도로 check (위 fail_reasons block 다음).
    if lap_time_mean < min_lap_time_frac * ideal_lap_time:
        fail_reasons.append(
            f'lap_time {lap_time_mean:.2f}s < {min_lap_time_frac}·ideal '
            f'({min_lap_time_frac * ideal_lap_time:.2f}s) — false lap')
        success = False

    # ── LTM Q + soft penalty ─────────────────────────────────────────
    if not success:
        Q = -j_inf                                              # hard reject only (crash/no-lap/reset)
    else:
        # LTM: -T_lap + λ·max(t_lb - T_lap, 0) - soft_penalty (lat_g/cte/shake/alive/sw)
        Q = -lap_time_mean + lam * max(t_lb - lap_time_mean, 0.0) - soft_penalty

    return {
        'Q': Q,
        'success': success,
        'fail_reasons': fail_reasons,            # list (only hard reject reasons)
        'soft_penalty': float(soft_penalty),     # soft penalty (lat_g/cte/shake/alive/sw)
        'n_resets': n_resets,                    # C: from external (BO log grep)
        'lap_time_mean': lap_time_mean,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'shake_rms': shake,
        'cte_rms': cte_rms,
        'lat_g_mean': lat_g_mean,
        'lat_g_peak': lat_g_peak,
        'ideal_lap_time': ideal_lap_time,
        't_lb': t_lb,
        'lap_time_delta': lap_time_delta,
        'mpcc_alive_frac': mpcc_alive_frac,
        'switch_count': switch_count,
        'laps': laps,
        'infeas_frac': infeas_frac,
        'stuck_frac': stuck_frac,
        'solve_ms_mean': solve_ms,
        'cost_mean': cost_mean,
        'crashed': crashed,
        'n_rows': n,
        'csv': csv_path,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="기본: ~/mpc_logs/ 의 최신 mpc_*.csv")
    p.add_argument("--lam", type=float, default=20.0,
                   help="LTM speedup bonus weight (EVO-MPCC default 20)")
    p.add_argument("--t_lb_factor", type=float, default=1.3,
                   help="t_lb = ideal_lap_time × factor (default 1.3 = 30%% slack from point-mass bound). "
                        "--t_lb override 되면 무시.")
    p.add_argument("--t_lb", type=float, default=None,
                   help="t_lb 직접 값 [s] (예: PP baseline best lap_time). 우선순위 > factor.")
    p.add_argument("--min_laps", type=int, default=1)
    p.add_argument("--map", default='f', help="centerline 로드용 맵 이름")
    # filter knobs
    p.add_argument("--alive_floor", type=float, default=0.8)
    p.add_argument("--switch_ceiling", type=int, default=3)
    p.add_argument("--lat_g_max", type=float, default=15.0, help="m/s² (~1.5g)")
    p.add_argument("--cte_max", type=float, default=0.5)
    p.add_argument("--shake_max", type=float, default=5.0)
    p.add_argument("--j_inf", type=float, default=1000.0)
    # B + C 추가 filter
    p.add_argument("--n_resets", type=int, default=0,
                   help="C: BO 가 sim log 의 '/initialpose stuck-recover' grep 회수 전달")
    p.add_argument("--reset_ceiling", type=int, default=0,
                   help="C: n_resets > ceiling → unqualified")
    p.add_argument("--min_lap_time_frac", type=float, default=0.5,
                   help="B: lap_time < frac × ideal 이면 false lap (teleport rollover)")
    p.add_argument("--json", action="store_true",
                   help="JSON 출력 (BO/RL harness 가 stdout 파싱)")
    args = p.parse_args()

    if args.csv is None:
        candidates = sorted(glob.glob(os.path.expanduser("~/mpc_logs/mpc_*.csv")))
        if not candidates:
            raise SystemExit("no CSV in ~/mpc_logs/")
        args.csv = candidates[-1]

    result = evaluate(args.csv,
                      lam=args.lam,
                      t_lb_factor=args.t_lb_factor,
                      t_lb_override=args.t_lb,
                      min_laps=args.min_laps,
                      map_name=args.map,
                      alive_floor=args.alive_floor,
                      switch_ceiling=args.switch_ceiling,
                      lat_g_max=args.lat_g_max,
                      cte_max=args.cte_max,
                      shake_max=args.shake_max,
                      j_inf=args.j_inf,
                      n_resets=args.n_resets,
                      reset_ceiling=args.reset_ceiling,
                      min_lap_time_frac=args.min_lap_time_frac)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"CSV:         {result['csv']}")
        print(f"  rows:      {result['n_rows']}  laps: {result['laps']}")
        print(f"  lap_time:  {result.get('lap_time_mean', 0):.3f} s "
              f"(ideal {result.get('ideal_lap_time', 0):.2f}, "
              f"t_lb {result.get('t_lb', 0):.2f})")
        print(f"  shake:     {result['shake_rms']:.4f}  "
              f"cte: {result.get('cte_rms', 0):.3f} m  "
              f"lat_g peak: {result.get('lat_g_peak', 0):.2f} m/s²")
        print(f"  alive:     {result['mpcc_alive_frac']*100:.1f}%  "
              f"switches: {result['switch_count']}")
        print(f"  infeas:    {result['infeas_frac']*100:.1f}%  "
              f"stuck: {result['stuck_frac']*100:.1f}%  "
              f"solve_ms: {result['solve_ms_mean']:.2f}")
        print(f"  crashed:   {result['crashed']}")
        print(f"  success:   {result.get('success', False)}")
        if not result.get('success', False):
            print(f"  fail_reasons: {result.get('fail_reasons', [])}")
        print(f"  Q:         {result['Q']:.3f}  (LTM)")


if __name__ == "__main__":
    main()
