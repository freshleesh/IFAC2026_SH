#!/usr/bin/env python3
"""Friction-circle μ 추정 — mpc_logs CSV의 (a_long, a_lat) 산점도에서 그립 한계 추정.

주행 데이터만으로 노면 μ를 측정한다 (별도 센서 불필요):
  a_long = d(vx)/dt  (Savitzky-Golay 미분, 노이즈 억제)
  a_lat  = vx · r    (구심가속도; vy 미분항은 작고 sim에선 vy=0)
포화 영역(타이어 한계)에서 점들이 원호를 그리므로, 방향별 빈으로 나눠
상위 분위수 반경을 모아 원을 적합 → μ = R/g.

★ 추정치는 "그 주행에서 실제로 쓴 그립"의 하한이다. 차가 한계까지 안 몰면
  μ를 과소추정한다 — 한계 주행 데이터(브레이크 테스트/정상상태 서클)와 병용 권장:
  · 종방향 μ: 직선 전속 → 풀브레이크, μ_x = |a_x|_max / g
  · 횡방향 μ: 일정 반경 R 서클 주행에서 미끄러지기 직전 v, μ_y = v²/(R·g)

사용:
  python3 friction_circle.py <csv> [--out fig.png] [--pct 98] [--vmin 1.0]
  python3 friction_circle.py latest          # ~/mpc_logs 최신 CSV
출력: μ 추정 수치 + friction circle 산점도 그림 (계기판 스타일).
실차: 같은 CSV 포맷(mpc_debug_logger)이므로 그대로 사용. dyn_mu 권장값도 출력.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

G = 9.81


def load_csv(path: str):
    if path == 'latest':
        logs = sorted(Path.home().glob('mpc_logs/*.csv'))
        if not logs:
            sys.exit('~/mpc_logs 에 CSV 없음')
        path = str(logs[-1])
    import csv as _csv
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    print(f'CSV: {path}  ({len(rows)} rows)')
    need = ('t', 'vx_odom', 'r_odom')
    if not rows or any(k not in rows[0] for k in need):
        sys.exit(f'필요 컬럼 {need} 없음 — 구버전 로그?')
    t = np.array([float(r['t']) for r in rows])
    vx = np.array([float(r['vx_odom']) for r in rows])
    r = np.array([float(r['r_odom']) for r in rows])
    return path, t, vx, r


def accel_series(t, vx, r, vmin: float):
    # 등간격 가정이 깨지는 구간(런 사이 점프) 제거
    dt = np.median(np.diff(t))
    ok = np.concatenate([[True], np.diff(t) < 5 * dt])
    t, vx, r = t[ok], vx[ok], r[ok]
    # Savitzky-Golay 1차 미분 (창 ~0.5s) — scipy 없으면 gradient+이동평균
    try:
        from scipy.signal import savgol_filter
        win = max(5, int(round(0.5 / dt)) | 1)
        a_long = savgol_filter(vx, win, 2, deriv=1, delta=dt)
        vx_s = savgol_filter(vx, win, 2)
        r_s = savgol_filter(r, win, 2)
    except ImportError:
        a_long = np.gradient(vx, t)
        k = max(5, int(round(0.5 / dt)))
        ker = np.ones(k) / k
        a_long = np.convolve(a_long, ker, 'same')
        vx_s, r_s = vx, np.convolve(r, ker, 'same')
    a_lat = vx_s * r_s
    drive = vx_s > vmin  # 정지/출발 노이즈 제거
    return a_long[drive], a_lat[drive]


def fit_circle(a_long, a_lat, pct: float):
    """방향별 빈 상위 분위수 반경 → 원 반경(= μ·g) 추정."""
    th = np.arctan2(a_lat, a_long)
    rad = np.hypot(a_long, a_lat)
    edges = np.linspace(-np.pi, np.pi, 25)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (th >= lo) & (th < hi)
        if m.sum() >= 20:
            pts.append(((lo + hi) / 2, np.percentile(rad[m], pct)))
    if not pts:
        sys.exit('데이터 부족 — 주행 시간이 너무 짧음')
    ths, rs = np.array(pts).T
    return ths, rs, float(np.max(rs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--out', default='/tmp/friction_circle.png')
    p.add_argument('--pct', type=float, default=98.0,
                   help='빈별 반경 분위수 (이상치 컷)')
    p.add_argument('--vmin', type=float, default=1.0)
    args = p.parse_args()

    path, t, vx, r = load_csv(args.csv)
    a_long, a_lat = accel_series(t, vx, r, args.vmin)
    ths, rs, R = fit_circle(a_long, a_lat, args.pct)

    mu_used = R / G
    mu_lat = float(np.percentile(np.abs(a_lat), args.pct)) / G
    mu_brake = float(np.percentile(np.maximum(-a_long, 0), args.pct)) / G
    print(f'사용 그립 한계 (이 주행에서 실제로 쓴 값 — 노면 μ의 하한):')
    print(f'  합성 |a|_max : {R:5.2f} m/s²  →  μ_used  = {mu_used:.3f}')
    print(f'  횡   |a_lat| : {mu_lat * G:5.2f} m/s²  →  μ_lat   = {mu_lat:.3f}')
    print(f'  제동 -a_long : {mu_brake * G:5.2f} m/s²  →  μ_brake = {mu_brake:.3f}')
    print(f'권장: dyn_mu ≳ {mu_used:.2f}  (한계 주행이었다면 = 노면 μ.')
    print(f'      여유 주행이었다면 브레이크 테스트로 μ_brake 를 따로 측정해 큰 쪽 사용)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a_lat, a_long, s=2, alpha=0.25, c=np.hypot(a_lat, a_long),
               cmap='viridis')
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(tt), R * np.sin(tt), 'r-', lw=2,
            label=f'fit |a|={R:.2f} m/s² (mu_used={mu_used:.2f})')
    for mu_ref, c in ((0.6, 'orange'), (1.0489, 'gray')):
        ax.plot(mu_ref * G * np.cos(tt), mu_ref * G * np.sin(tt), '--', c=c,
                lw=1, label=f'mu={mu_ref}')
    ax.plot(rs * np.sin(ths), rs * np.cos(ths), 'r.', ms=6)
    ax.set_xlabel('a_lat [m/s²]')
    ax.set_ylabel('a_long [m/s²]')
    ax.set_title(f'Friction circle — {Path(path).name}')
    ax.axhline(0, c='k', lw=0.3)
    ax.axvline(0, c='k', lw=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'그림: {args.out}')


if __name__ == '__main__':
    main()
