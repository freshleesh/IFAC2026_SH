#!/usr/bin/env python3
"""bo_sweep_turbo.py — BoTorch 기반 TuRBO + Constrained EI 로 MPCC bucket scale 학습.

기존 bo_sweep.py (skopt gp_minimize, gp_hedge acquisition) 의 SOTA 교체:
  - GP: BoTorch SingleTaskGP (Matern 5/2 + ARD lengthscale)
  - Trust Region: TuRBO (Eriksson 2019, NeurIPS) — high-dim BO (12D 에 효과적)
  - Acquisition: ConstrainedExpectedImprovement — crash 를 hard constraint 로
  - Warm start: 동일 max_speed 의 과거 best history entry (autoregressive 와 호환)

Output JSON / yaml override / objective 호출 흐름은 bo_sweep.py 와 100% 호환:
  autoregressive_bo.sh 가 BO=bo_sweep_turbo.py 로 가리키면 그대로 동작.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# BoTorch — SOTA BO library (Meta). 의존성: torch, gpytorch.
try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
    from botorch.acquisition.objective import ConstrainedMCObjective
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    BOTORCH_OK = True
except ImportError as e:
    print(f'BoTorch import 실패: {e}')
    print('설치: pip install botorch')
    BOTORCH_OK = False


# ───────────────────────────────────────────────────────────────────
# Paths / constants
# ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]   # → /home/hmcl/IFAC2026_SH
YAML = Path(__file__).resolve().parents[1] / 'config' / 'ddrx_unified_params.yaml'   # 2026-06-16 port: this ws nonlinear_mpc_acados/config (symlink-install → mpc_node reads edits)
LOG_DIR = Path(os.path.expanduser('~/mpc_logs'))
BO_DIR = Path(os.path.expanduser('~/bo_results'))
BO_DIR.mkdir(parents=True, exist_ok=True)
# 2026-06-02 'Q 최고점마다 기록': 새 best Q 갱신마다 한 줄씩 append (jsonl).
# 각 줄에 ts/max_speed 포함 → 여러 run 구분 가능. tail -f 로 실시간 추적.
PEAKS_PATH = BO_DIR / 'bo_q_peaks.jsonl'
EVAL_SCRIPT = Path(__file__).parent / 'eval_run_quality.py'
# 2026-06-10 mu 정직화: --dyn_mu 지정 시 매 trial launch 에 dyn_mu:=<v> 전달
# (codegen-time 상수 — mu-키잉된 codegen 디렉토리로 자동 분리됨).
DYN_MU: float | None = None

# B (VPMPCC simplify, 2026-05-26): 13D bucket + D_apex → 5D single weights.
# VPMPCC paper 의 search space (Table II): q_cte, q_lag, q_v, q_p (=γ), q_drate.
# bucket 폐기 (Switched MPCC), D_apex 폐기 (TUM AR), N_p 고정 (codegen 비용).
#
# 2026-05-27 review fixes:
#   #3: q_cte 하한 0.3 → 0.1 (per-dim). centerline 느슨 추종 자유도.
#   #5: log-uniform sampling. scale factor 는 곱셈 의미 → log space 가 자연.
#       GP 는 log10(scale) ∈ [-1.0, 0.70] 에서 작동, 평가 시 10^x 적용.
#   #6: q_psi 추가 → 6D. EVO-MPCC search space 와 정합.
#   #8: q_dv 추가 → 7D. a_x (longitudinal accel) penalty. EVO-MPCC Q_Δv 와 정합.
# 2026-06-01 Phase 1 roadmap: +a_lat_safe (corner grip) +D_apex (apex depth).
# corner-limited 상태에서 a_lat 이 직접 레버, D_apex 가 racing line 깊이.
# 둘은 scale factor 가 아니라 물리값이라 LINEAR (log 아님) per-dim.
DIM = 9
PARAM_KEYS = ['q_cte', 'q_lag', 'q_psi', 'q_v', 'q_p', 'q_drate', 'q_dv', 'a_lat_safe', 'D_apex']
PARAM_SCALE_TYPE = {'q_cte': 'log', 'q_lag': 'log', 'q_psi': 'log', 'q_v': 'log',
                    'q_p': 'log', 'q_drate': 'log', 'q_dv': 'log',
                    'a_lat_safe': 'linear', 'D_apex': 'linear'}
PER_DIM_LO = {'q_cte': 0.1, 'q_lag': 0.3, 'q_psi': 0.3, 'q_v': 0.3, 'q_p': 0.3, 'q_drate': 0.3, 'q_dv': 0.3,
              'a_lat_safe': 3.0, 'D_apex': 0.0}   # 2026-06-02 a_lat LO 7→3: "박으면안됨" = 느린-코너 clean 영역 탐색 허용
# q_v hi widened 5.0->8.0 (2026-06-01, EVO/RVP analysis): deploy q_v=0.317 sat
# at the bottom of [0.3,5] — found on the OLD objective (min-lap, lat_g=50) that
# favored loose ref_v tracking. EVO's edge is a brake-aware RVP tracked strongly;
# on the hardened (median-lap = consistency) objective, stronger q_v should win.
# Give the search room upward to test it (log-uniform → ~half the samples q_v>1.6).
# a_lat_safe [7,11]: 9 가 v=5 최적였으나 v=7 헤드룸 다름 — 탐색. D_apex [0,1.0]: 0=centerline, 1.0=깊은 apex(코리도어 ±1.25 안).
PER_DIM_HI = {'q_cte': 5.0, 'q_lag': 5.0, 'q_psi': 5.0, 'q_v': 8.0, 'q_p': 5.0, 'q_drate': 30.0, 'q_dv': 5.0,
              'a_lat_safe': 16.0, 'D_apex': 1.0}   # 2026-06-07: HI 11→16 (16s 도전, 새 final 맵 코너 grip 캡 들어올림; ConstrainedEI가 슬립/접촉 페널티)
# Linear bounds (보고용 / yaml 작성용)
BOUNDS = torch.tensor(
    [[PER_DIM_LO[k] for k in PARAM_KEYS],
     [PER_DIM_HI[k] for k in PARAM_KEYS]],
    dtype=torch.double,
)
# Log-uniform bounds (log dim 만 사용; linear dim 은 0 하한이라 clamp 필수)
LOG_BOUNDS = torch.log10(torch.clamp(BOUNDS, min=1e-6))
SCALE_LO, SCALE_HI = 0.1, 5.0  # legacy print only
_IS_LOG = torch.tensor([PARAM_SCALE_TYPE[k] == 'log' for k in PARAM_KEYS])


def x_norm_to_scale(x_norm: torch.Tensor) -> torch.Tensor:
    """[0,1]^d 정규화된 BO 점 → 실제 값. log dim 은 log-uniform, linear dim 은 선형."""
    log_x = LOG_BOUNDS[0] + (LOG_BOUNDS[1] - LOG_BOUNDS[0]) * x_norm
    out_log = 10.0 ** log_x
    out_lin = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_norm
    return torch.where(_IS_LOG.to(x_norm.device), out_log, out_lin)


def scale_to_x_norm(scale: torch.Tensor) -> torch.Tensor:
    """실제 값 → [0,1]^d 정규화 (warm start 용). per-dim log/linear."""
    log_x = torch.log10(scale.clamp(min=1e-6))
    norm_log = (log_x - LOG_BOUNDS[0]) / (LOG_BOUNDS[1] - LOG_BOUNDS[0])
    norm_lin = (scale - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    return torch.where(_IS_LOG.to(scale.device), norm_log, norm_lin)
# back-compat
N_SCALES = DIM
SCALE_KEYS = PARAM_KEYS
BOUNDS_LO, BOUNDS_HI = SCALE_LO, SCALE_HI


# ───────────────────────────────────────────────────────────────────
# yaml override helper (bo_sweep.py 와 동일)
# ───────────────────────────────────────────────────────────────────
def sed_yaml_override(params: dict, mode: str = 'off'):
    """yaml 의 q_*_scale_live 값 in-place 갱신.

    B 적용 후: bucket 모드 폐기. PARAM_KEYS = q_cte, q_lag, q_v, q_p, q_drate
    → yaml 의 q_cte_scale_live, q_lag_scale_live, q_v_scale_live, q_p_scale_live,
    q_drate_scale_live 직접 변경. override_mode 는 'off' (bucket scale 비활성).
    """
    txt = YAML.read_text()
    replacements = {'override_mode': f"'off'", 'override_scales': 'false'}
    if params:
        # q_* → q_*_scale_live (scale factor). a_lat_safe/D_apex → *_live (물리값, scale 아님).
        _DIRECT_LIVE = {'a_lat_safe', 'D_apex'}
        for k, v in params.items():
            suffix = '_live' if k in _DIRECT_LIVE else '_scale_live'
            replacements[f'{k}{suffix}'] = f'{v:.4f}'
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for key, val in replacements.items():
            if stripped.startswith(f'{key}:'):
                indent = line[:len(line) - len(stripped)]
                lines[i] = f'{indent}{key}: {val}'
                break
    YAML.write_text('\n'.join(lines) + '\n')


def read_yaml_max_speed() -> float:
    try:
        for line in YAML.read_text().splitlines():
            s = line.strip()
            if s.startswith('max_speed:') and not s.startswith('max_speed_p'):
                return float(s.split(':', 1)[1].strip().split('#')[0].strip())
    except Exception:
        pass
    return -1.0


# ───────────────────────────────────────────────────────────────────
# Sim runner (bo_sweep.py 와 동일 인터페이스)
# ───────────────────────────────────────────────────────────────────
def _poll_lap_count() -> int:
    """ros2 topic echo --once /mpc/lap_count → int."""
    try:
        r = subprocess.run(
            ['timeout', '2', 'ros2', 'topic', 'echo', '--once', '/mpc/lap_count'],
            capture_output=True, text=True,
        )
        for ln in r.stdout.splitlines():
            ln = ln.strip()
            if ln.startswith('data:'):
                return int(ln.split(':', 1)[1].strip())
    except Exception:
        pass
    return -1


def run_one_sim(map_name: str, n_laps: int, wall_timeout: int, stuck_timeout: int) -> dict:
    """sim 한 번 돌려서 termination reason + 최신 CSV 반환."""
    import signal
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    # CYCLONEDDS_URI 명시 — bash -lc 의 non-interactive 환경에선 .bashrc 안 load
    # 됨 → ~/cyclonedds.xml 의 MaxAutoParticipantIndex 120 안 적용 → mpc_node
    # DDS slot fail. 명시 전달로 강제.
    env.setdefault('CYCLONEDDS_URI', f'file://{os.path.expanduser("~/cyclonedds_laptop.xml")}')
    # 2026-06-01: a_lat_safe 가 BO dim 이 되며 codegen-baked hard cap(a_lat_max=
    # a_lat_safe+1) 도 eval 마다 달라져야 함. codegen 캐시를 매 eval 지워 재생성
    # (~30s/eval) → a_lat_safe 가 soft+hard cap 모두에 정확히 반영. (캐시 두면 hard
    # cap 이 첫 eval 값에 고정돼 a_lat 탐색 편향.)
    cmd = ['bash', '-c',
           f'source $HOME/unicorn_ws/src/unicorn-racing-stack/unicorn.sh >/dev/null 2>&1 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
           f'export CYCLONEDDS_URI=file://$HOME/cyclonedds_laptop.xml && '
           f'rm -rf /tmp/acados_codegen_evompcc* /tmp/acados_ocp_evompcc* ; '
           f'ros2 launch stack_master low_level.launch.xml sim:=true map:={map_name} use_estop:=false >/tmp/bo_lowlevel.log 2>&1 & sleep 16; ros2 run nonlinear_mpc_acados mpc_debug_logger >/tmp/bo_logger.log 2>&1 & ros2 launch nonlinear_mpc_acados mpcc.launch.xml map:={map_name} sim:=true detector:=false'
           + (f' dyn_mu:={DYN_MU}' if DYN_MU is not None else '')]
    # stderr 를 file 로 redirect (debug 용 — startup fail 원인 파악).
    log_path = f'/tmp/bo_trial_sim.log'
    log_fh = open(log_path, 'w')
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid, env=env)
    t0 = time.time()
    # 이전 trial 의 latched /mpc/lap_count (TRANSIENT_LOCAL) 가 DDS 에 남아
    # 새 trial 첫 폴링이 stale 값을 즉시 읽어 "reached N laps" 오판하는 버그
    # (unicorn DDS, 좀비 rviz). sim 이 실제로 뜬 뒤(~40s) 의 lap 을 baseline
    # 으로 잡고, 그 이상으로 올라간 만큼만 센다.
    time.sleep(40)   # sim + acados codegen 뜨는 시간 (그 후 lap_count=0 갱신)
    lap_base = _poll_lap_count()
    if lap_base < 0:
        lap_base = 0
    last_lap = -1
    last_lap_time = t0
    reason = 'wall timeout'
    try:
        while True:
            elapsed = time.time() - t0
            lap = _poll_lap_count() - lap_base
            if lap > last_lap:
                print(f'  [{int(elapsed):5d}s] lap={lap}')
                last_lap = lap
                last_lap_time = time.time()
            if lap >= n_laps:
                reason = f'reached {n_laps} laps'
                break
            if elapsed > wall_timeout:
                reason = f'wall timeout {wall_timeout}s'
                break
            no_progress = time.time() - last_lap_time
            # lap >= 1 이면 정상 stuck (race 중 멈춤).
            # lap == 0 + no_progress > stuck_timeout 이면 startup 자체 실패 (mpc_node 죽음
            # 또는 가속 못함). BO 가 무한 대기하지 않게 즉시 abort.
            if lap >= 1 and no_progress > stuck_timeout:
                reason = f'stuck ({no_progress:.0f}s no progress at lap={lap})'
                break
            if lap < 1 and elapsed > 90 and no_progress > stuck_timeout:
                reason = f'startup fail ({elapsed:.0f}s, lap=0)'
                break
            time.sleep(1.0)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            time.sleep(3)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        # 자식 좀비 정리. 매 trial mpc lap_count topic stale 방지 위해 강력하게.
        # iter 2+ 의 "[ 0s] lap=N" 즉시 = 이전 sim 의 lap_count 가 남아있음 → 새 sim
        # 안 떠도 이전 값 받음. 충분히 sleep (8s) + ros2 daemon kill.
        subprocess.run(
            'pkill -KILL -f "gym_bridge\\|state_machine\\|spliner\\|controller_manager\\|'
            'global_republisher\\|frenet_conversion\\|frenet_odom_republisher\\|'
            'static_obstacle_manager\\|fake_topic_relay\\|simple_mux\\|'
            'ego_robot_state_publisher\\|rviz2\\|mpc_node\\|mpc_debug_logger\\|'
            'pp_fallback\\|ftg_fallback\\|joy_node\\|robot_state_publisher\\|'
            'obstacle_publisher\\|waypoint_publisher\\|nonlinear_mpc"',
            shell=True, check=False,
        )
        # NOTE: ros2.*launch 제외 — bo_train.launch.py 가 부모면 자기 죽임.
        # bo_sweep_turbo 의 trial launch 는 proc.pid 의 process group 으로 별도 kill.
        subprocess.run('pkill -KILL -f "ros2.*daemon"', shell=True, check=False)
        time.sleep(8)

    csvs = sorted(LOG_DIR.glob('mpc_*.csv'), key=lambda p: p.stat().st_mtime)
    latest = csvs[-1] if csvs else None
    # C: log 안의 "[stuck-recover] /initialpose" 등장 횟수 = trial 의 reset 회수.
    # mpc_node 가 stuck 감지 시 발행 → gym env.reset() → teleport. eval filter 에 전달.
    n_resets = 0
    try:
        with open(log_path, 'r', errors='ignore') as f:
            n_resets = sum(1 for line in f if 'stuck-recover' in line and '/initialpose' in line)
    except Exception:
        pass
    return {'reason': reason, 'csv': str(latest) if latest else None,
            'n_resets': n_resets}


def find_pp_baseline_t_lb(map_name: str, v: float) -> float | None:
    """PP baseline json 자동 찾기 — best_lap_time 반환. 없으면 None.

    Pattern: ~/bo_results/pp_baseline_v{v}_*.json. JSON 의 'map' field 가
    map_name 와 일치하는 가장 최근 파일의 best_lap_time 사용.
    """
    pattern = os.path.expanduser(f'~/bo_results/pp_baseline_v{v}_*.json')
    files = sorted(glob.glob(pattern), reverse=True)  # 최근 파일 우선
    for fp in files:
        try:
            d = json.load(open(fp))
            if d.get('map') == map_name and d.get('laps', 0) >= 1:
                blt = d.get('best_lap_time')
                if blt is not None and blt > 0 and blt == blt:   # not NaN
                    return float(blt)
        except Exception:
            pass
    return None


def evaluate_csv(csv_path: str, n_resets: int = 0,
                 t_lb_override: float | None = None,
                 map_name: str = 'f') -> dict:
    cmd = ['python3', str(EVAL_SCRIPT), '--csv', csv_path, '--json',
           '--n_resets', str(n_resets), '--map', map_name]
    if t_lb_override is not None:
        cmd += ['--t_lb', f'{t_lb_override:.4f}']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {'Q': -1e6, 'crashed': True, 'reason': 'eval failed'}
    return json.loads(r.stdout)


# ───────────────────────────────────────────────────────────────────
# TuRBO state (Eriksson 2019 NeurIPS)
# ───────────────────────────────────────────────────────────────────
class TurboState:
    """Trust region 의 길이 length 를 success/failure 카운터로 자동 조정.

    - length_init = 0.8 (정규화 [0,1]^d 기준)
    - length_min  = 0.5^7  (너무 작아지면 restart)
    - length_max  = 1.6
    - success: best 갱신 → length *= 2 (more exploration in larger region)
    - failure: best 갱신 못 함 → length /= 2 (focus on smaller region)
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.length = 0.8
        self.length_min = 0.5 ** max(7, dim)   # 2026-06-01: dim 7→9 (a_lat,D_apex). hardcoded 7→dim
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = max(4, int(dim / 1))   # dim=12 → 12 회 실패 → 축소
        self.success_counter = 0
        self.success_tolerance = 3                       # 3 회 연속 success → 확장
        self.best_value = -float('inf')
        self.restart_triggered = False


def update_turbo(state: TurboState, y_new: float):
    """Y_new 한 sample 결과로 state 갱신."""
    if y_new > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, y_new)
    if state.length < state.length_min:
        state.restart_triggered = True


def generate_candidates(state: TurboState, model: SingleTaskGP, X_norm: torch.Tensor,
                        Y: torch.Tensor, n_candidates: int = 2000):
    """Trust region 안에서 Sobol-perturbed candidates 생성 + best 1 개 선택.

    weight 는 GP 의 ARD lengthscale 로부터 (각 차원별 importance).
    """
    dim = X_norm.shape[-1]
    # GP lengthscales 로 trust region 의 차원별 weighting.
    # BoTorch 버전마다 covar_module 구조 다름:
    #   구버전: ScaleKernel(MaternKernel) — .base_kernel.lengthscale
    #   신버전: RBFKernel 직접     — .lengthscale
    covar = model.covar_module
    if hasattr(covar, 'base_kernel'):
        lengthscale = covar.base_kernel.lengthscale
    else:
        lengthscale = covar.lengthscale
    weights = lengthscale.squeeze().detach()
    # ARD 가 아닐 수도 (scalar lengthscale) → uniform weight 로 fallback
    if weights.dim() == 0 or weights.numel() == 1:
        weights = torch.ones(dim, dtype=X_norm.dtype)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / dim))
    # best 점 중심으로 trust region 박스
    x_center = X_norm[Y.argmax(), :].clone()
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    # Sobol sample within trust region
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=X_norm.dtype)
    pert = tr_lb + (tr_ub - tr_lb) * pert
    # Perturbation mask — 일부 차원만 perturb (TuRBO trick)
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim, size=(len(ind),))] = 1
    # Combine: 마스크된 차원만 perturbed, 나머지는 center 값
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    return X_cand


def noise_aware_incumbent(model: SingleTaskGP, X_norm: torch.Tensor,
                          Y: torch.Tensor,
                          crash_model: SingleTaskGP | None = None) -> float:
    """Noise-aware EI incumbent (Shahriari et al. 2016, BO review p.161).

    Q (lap time 기반) 는 noisy (ms-stochastic) → raw best observed value 를
    EI incumbent 으로 쓰면 "운좋은 빠른 lap" 한 번이 incumbent 를 오염시켜
    over-exploitation 유발. 대신 GP posterior MEAN 을 관측점들에서 평가한 뒤
    그 max 를 incumbent (best_f, maximization convention) 로 사용한다.

    crash_model 이 있으면 feasible (P(feasible) > 0.5, 즉 Crashed 예측 < 0.5)
    관측점들에 대해서만 max 를 취한다. feasible 점이 하나도 없으면 전체 관측점
    posterior mean max 로 fallback, 그것마저 실패하면 raw Y.max() 로 fallback.
    """
    try:
        with torch.no_grad():
            post_mean = model.posterior(X_norm).mean.squeeze(-1)  # (n,)
            if crash_model is not None:
                crash_mean = crash_model.posterior(X_norm).mean.squeeze(-1)
                feasible = crash_mean < 0.5
                if bool(feasible.any()):
                    return float(post_mean[feasible].max().item())
            return float(post_mean.max().item())
    except Exception:
        # robust fallback: raw best observed (maximization).
        return float(Y.max().item())


def constrained_ei_select(model: SingleTaskGP, X_cand: torch.Tensor,
                          best_f: float, crash_model: SingleTaskGP | None) -> torch.Tensor:
    """ConstrainedEI 로 best candidate 1 개 선택.

    EI(x) * P(feasible | x). crash_model 이 None 이면 plain EI.
    """
    # BoTorch 버전마다 LogEI 위치/이름 다름 → ExpectedImprovement 사용 (warning suppress).
    import warnings
    from botorch.acquisition.analytic import ExpectedImprovement
    try:
        from botorch.exceptions.warnings import NumericsWarning
        warnings.filterwarnings('ignore', category=NumericsWarning)
    except ImportError:
        pass
    ei = ExpectedImprovement(model, best_f=best_f, maximize=True)
    with torch.no_grad():
        acq = ei(X_cand.unsqueeze(1))   # (n, 1, d) → (n,)
        if crash_model is not None:
            mean = crash_model.posterior(X_cand.unsqueeze(1)).mean.squeeze()
            p_feasible = torch.clamp(1.0 - mean, 0.0, 1.0)
            acq = acq * p_feasible
    return X_cand[acq.argmax()]


# ───────────────────────────────────────────────────────────────────
# Main BO loop
# ───────────────────────────────────────────────────────────────────
def main():
    if not BOTORCH_OK:
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument('--n_calls', type=int, default=30)
    p.add_argument('--n_initial', type=int, default=8)   # Sobol initial sample
    p.add_argument('--n_laps', type=int, default=3)
    p.add_argument('--wall_timeout', type=int, default=180)
    p.add_argument('--stuck_timeout', type=int, default=60)
    p.add_argument('--map', default='f', help='단일 map (back-compat). --maps 와 함께 쓰면 무시.')
    p.add_argument('--maps', default=None,
                   help='multi-map BO 학습용 공백/콤마 구분 map 목록.')
    p.add_argument('--map_mode', default='alternate', choices=['alternate', 'mean'],
                   help='alternate (default): iter 마다 round-robin 으로 1 map (2× 빠름). '
                        'mean: 매 iter 모든 map 평가 후 평균 Q (오버피팅 더 강하게 방지).')
    p.add_argument('--mode', default='bucketed', choices=['bucketed'])
    p.add_argument('--isotropic', action='store_true',
                   help='GP kernel 을 isotropic (single lengthscale) 로. '
                        '20-trial 등 데이터 적을 때 ARD 가 noise 만 학습할 위험 ↓.')
    p.add_argument('--dyn_mu', type=float, default=None,
                   help='모델 그립 mu — 매 trial launch 에 dyn_mu:=<v> 전달 (예: 0.6)')
    p.add_argument('--x0', default=None,
                   help=f'warm start ({DIM} float JSON list 또는 CSV).')
    args = p.parse_args()

    global DYN_MU
    DYN_MU = args.dyn_mu
    if DYN_MU is not None:
        print(f'  dyn_mu: {DYN_MU} (매 trial launch 에 전달)')

    # map list 결정 — --maps 우선, 없으면 단일 --map.
    if args.maps:
        maps_list = [m for m in args.maps.replace(',', ' ').split() if m]
    else:
        maps_list = [args.map]
    print(f'  maps: {maps_list}  mode: {args.map_mode}')

    # Warm start
    x0 = None
    if args.x0:
        try:
            parsed = json.loads(args.x0) if args.x0.lstrip().startswith('[') \
                     else [float(v) for v in args.x0.split(',')]
            if len(parsed) != DIM:
                raise ValueError(f'x0 길이 {len(parsed)} ≠ DIM {DIM}')
            x0 = torch.tensor(parsed, dtype=torch.double)
            print(f'  warm start (--x0): {[f"{v:.3f}" for v in parsed]}')
        except Exception as e:
            print(f'  warm start parse 실패: {e}')

    print(f'\n=== BO sweep start (BoTorch TuRBO + ConstrainedEI) ===')
    print(f'  map={args.map}  n_calls={args.n_calls}  n_initial={args.n_initial}')
    print(f'  dim={DIM}D, scale ∈ [{BOUNDS_LO}, {BOUNDS_HI}]')

    # Initial samples — Sobol for space coverage, 그리고 warm start 가 있으면 추가.
    sobol = torch.quasirandom.SobolEngine(DIM, scramble=True, seed=int(time.time()) % 100000)
    X_norm_init = sobol.draw(args.n_initial)    # [0,1]^d (in log-uniform space)
    if x0 is not None:
        x0_norm = scale_to_x_norm(x0)
        X_norm_init = torch.cat([x0_norm.unsqueeze(0), X_norm_init], dim=0)

    history = []
    X_norm = torch.empty(0, DIM, dtype=torch.double)
    Y_neg_Q = torch.empty(0, 1, dtype=torch.double)
    Crashed = torch.empty(0, 1, dtype=torch.double)

    # ── helper: x_norm 한 점 평가 (multi-map 평균) ────────────────────
    def evaluate_point(x_norm_pt: torch.Tensor, iter_no: int) -> tuple[float, bool, dict]:
        x = x_norm_to_scale(x_norm_pt).tolist()
        params = dict(zip(PARAM_KEYS, x))
        head = ', '.join(f'{k}={v:.2f}' for k, v in params.items())
        print(f'\n[iter {iter_no}] scales=' + head)
        sed_yaml_override(params, mode='bucketed')

        per_map = []
        Qs = []
        any_crash = False
        # alternate: iter 마다 round-robin 으로 1 map. mean: 매 iter 전체.
        if args.map_mode == 'alternate':
            iter_maps = [maps_list[(iter_no - 1) % len(maps_list)]]
        else:
            iter_maps = maps_list
        for mn in iter_maps:
            label = f'[{mn}]' if len(maps_list) > 1 else ''
            sim_info = run_one_sim(mn, args.n_laps, args.wall_timeout, args.stuck_timeout)
            print(f'  {label} sim done: {sim_info["reason"]}')
            if sim_info['csv'] is None:
                m = {'Q': -1e6, 'reason': 'no csv', 'crashed': True, 'success': False}
            else:
                # Option 2: PP baseline json 자동 찾기 → t_lb 로 사용.
                v_max_yaml = read_yaml_max_speed()
                t_lb_pp = find_pp_baseline_t_lb(mn, v_max_yaml)
                m = evaluate_csv(sim_info['csv'],
                                 n_resets=sim_info.get('n_resets', 0),
                                 t_lb_override=t_lb_pp,
                                 map_name=mn)
                ok_tag = 'OK' if m.get('success', False) else 'FAIL'
                fail = ','.join(m.get('fail_reasons', []))
                print(f'  {label} Q={m["Q"]:.2f} [{ok_tag}]  '
                      f'lap={m.get("lap_time_mean", 0):.3f}s '
                      f'(ideal {m.get("ideal_lap_time", 0):.2f}, '
                      f't_lb {m.get("t_lb", 0):.2f})  '
                      f'v_avg={m.get("avg_speed", 0):.2f}  '
                      f'v_max={m.get("max_speed", 0):.2f}  '
                      f'lat_g_peak={m.get("lat_g_peak", 0):.2f}  '
                      f'shake={m.get("shake_rms", 0):.4f}  '
                      f'cte={m.get("cte_rms", 0):.3f}  '
                      f'laps={m.get("laps", 0)}  '
                      f'alive={m.get("mpcc_alive_frac", 0)*100:.0f}%  '
                      f'sw={m.get("switch_count", 0)}'
                      + (f'  → {fail}' if fail else ''))
            per_map.append({'map': mn, 'sim': sim_info, 'metrics': m})
            Qs.append(float(m['Q']))
            any_crash = any_crash or (not bool(m.get('success', False)))

        mean_Q = float(np.mean(Qs))
        if len(iter_maps) > 1:
            print(f'  → mean Q across {len(iter_maps)} maps: {mean_Q:.2f}  '
                  f'(per-map: {[round(q,1) for q in Qs]})')
        elif len(maps_list) > 1:
            print(f'  → [alternate] Q@{iter_maps[0]} = {mean_Q:.2f}')
        history.append({
            'scales': dict(params),  # 9D all keys (incl a_lat_safe, D_apex)
            'params': params,
            'per_map': per_map,
            'mean_Q': mean_Q,
            'crashed': any_crash,
        })
        # ── Q 최고점마다 기록 (2026-06-02 user req) ──
        # non-crashed 중 새 best 갱신 시 peaks jsonl 에 append.
        try:
            _valid = [h['mean_Q'] for h in history if not h['crashed']]
            if (not any_crash) and _valid and mean_Q >= max(_valid):
                _rec = {
                    'ts': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'iter': len(history),
                    'Q': round(mean_Q, 3),
                    'max_speed': read_yaml_max_speed(),
                    'params': {k: round(float(v), 4) for k, v in params.items()},
                }
                # 대표 metric(첫 map) 함께 — lap/속도 추적용
                try:
                    _m0 = per_map[0]['metrics']
                    _rec['lap_time_mean'] = round(float(_m0.get('lap_time_mean', 0)), 3)
                    _rec['v_max'] = round(float(_m0.get('max_speed', 0)), 2)
                    _rec['v_avg'] = round(float(_m0.get('avg_speed', 0)), 2)
                except Exception:
                    pass
                with open(PEAKS_PATH, 'a') as _pf:
                    _pf.write(json.dumps(_rec, default=str) + '\n')
                print(f'  ★ NEW Q-PEAK Q={mean_Q:.2f} (lap={_rec.get("lap_time_mean","?")}s, '
                      f'v_max={_rec.get("v_max","?")}) → {PEAKS_PATH.name}')
        except Exception:
            pass
        return mean_Q, any_crash, {'Q': mean_Q, 'crashed': any_crash}

    # ── Phase 1: initial samples (Sobol + warm) ─────────────────────
    print(f'\n--- Phase 1: {len(X_norm_init)} initial samples (Sobol + warm)')
    for i, x_norm_pt in enumerate(X_norm_init):
        Q, crashed, _ = evaluate_point(x_norm_pt, i + 1)
        X_norm = torch.cat([X_norm, x_norm_pt.unsqueeze(0).to(torch.double)], dim=0)
        Y_neg_Q = torch.cat([Y_neg_Q, torch.tensor([[Q]], dtype=torch.double)], dim=0)
        Crashed = torch.cat([Crashed, torch.tensor([[float(crashed)]], dtype=torch.double)], dim=0)

    # ── Phase 2: TuRBO + ConstrainedEI loop ─────────────────────────
    state = TurboState(dim=DIM)
    state.best_value = float(Y_neg_Q.max().item())
    print(f'\n--- Phase 2: TuRBO + ConstrainedEI (remaining {args.n_calls - len(X_norm)} iters)')
    print(f'  initial best Q={state.best_value:.2f}, TR length={state.length:.3f}')

    iter_no = len(X_norm)
    while iter_no < args.n_calls and not state.restart_triggered:
        # Fit GP on (X, Y_neg_Q). Constraint GP on (X, Crashed).
        # 2026-05-27 review #7: --isotropic 면 single lengthscale (ARD 끔).
        if args.isotropic:
            iso_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=None))
            gp = SingleTaskGP(X_norm, Y_neg_Q,
                              likelihood=GaussianLikelihood(),
                              covar_module=iso_kernel)
        else:
            gp = SingleTaskGP(X_norm, Y_neg_Q,
                              likelihood=GaussianLikelihood())
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        # 2026-05-27 review: ARD lengthscale 진단 출력. 정규화 [0,1]^5 기준
        # lengthscale > 10 이면 해당 차원 "정보 없음" (GP 가 학습 못함).
        try:
            ls = gp.covar_module.base_kernel.lengthscale.squeeze().detach().tolist()
        except AttributeError:
            ls = gp.covar_module.lengthscale.squeeze().detach().tolist()
        if isinstance(ls, float):
            ls = [ls] * DIM
        ls_str = ', '.join(f'{k}={l:.2f}' for k, l in zip(PARAM_KEYS, ls))
        print(f'  GP ARD lengthscale: {ls_str}')
        # Crash classifier — GP for binary 라 over-simplified. 그래도 P(crashed) 추정.
        if Crashed.sum().item() > 0 and Crashed.sum().item() < len(Crashed):
            crash_gp = SingleTaskGP(X_norm, Crashed)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(crash_gp.likelihood, crash_gp))
        else:
            crash_gp = None

        # Generate candidates in trust region + select via ConstrainedEI
        X_cand = generate_candidates(state, gp, X_norm, Y_neg_Q.squeeze(-1),
                                     n_candidates=2000)
        # Improvement A (Shahriari 2016 p.161): noisy objective 에서는 EI incumbent
        # 을 raw best observed (Y_neg_Q.max()) 가 아니라 GP posterior mean 의 max
        # (feasible 관측점 한정) 로. 운좋은 빠른 lap 의 incumbent 오염 방지.
        best_f_incumbent = noise_aware_incumbent(gp, X_norm, Y_neg_Q,
                                                 crash_model=crash_gp)
        x_next = constrained_ei_select(gp, X_cand,
                                       best_f=best_f_incumbent,
                                       crash_model=crash_gp)
        # Evaluate
        iter_no += 1
        Q, crashed, _ = evaluate_point(x_next, iter_no)
        X_norm = torch.cat([X_norm, x_next.unsqueeze(0)], dim=0)
        Y_neg_Q = torch.cat([Y_neg_Q, torch.tensor([[Q]], dtype=torch.double)], dim=0)
        Crashed = torch.cat([Crashed, torch.tensor([[float(crashed)]], dtype=torch.double)], dim=0)
        update_turbo(state, Q)
        print(f'  TR: length={state.length:.3f}  success={state.success_counter}  '
              f'failure={state.failure_counter}  best_Q={state.best_value:.2f}')
        if state.restart_triggered:
            print('  TR restart triggered (length < min) — stopping.')
            break

    # ── Phase 3: result save ────────────────────────────────────────
    # raw best observed (참조용 로깅 — 운좋은 lap 포함 가능).
    raw_best_idx = int(Y_neg_Q.argmax().item())
    raw_best_x = x_norm_to_scale(X_norm[raw_best_idx]).tolist()
    raw_best_Q = float(Y_neg_Q[raw_best_idx].item())
    raw_best_params = dict(zip(PARAM_KEYS, raw_best_x))

    # Improvement B (Shahriari 2016 Eq.21 p.156, simple-regret regime):
    # 최종 추천 config 는 empirical best 가 아니라 GP-posterior-best (noise 제거).
    # 평가된 관측점들 중 final GP posterior mean 이 최대인 feasible 점을 고른다.
    # GP fit / posterior 가 실패하면 raw best 로 안전 fallback.
    best_idx = raw_best_idx
    selection_rule = 'gp_posterior_best'
    try:
        if args.isotropic:
            _iso = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=None))
            final_gp = SingleTaskGP(X_norm, Y_neg_Q,
                                    likelihood=GaussianLikelihood(),
                                    covar_module=_iso)
        else:
            final_gp = SingleTaskGP(X_norm, Y_neg_Q,
                                    likelihood=GaussianLikelihood())
        fit_gpytorch_mll(ExactMarginalLogLikelihood(final_gp.likelihood, final_gp))
        if Crashed.sum().item() > 0 and Crashed.sum().item() < len(Crashed):
            final_crash_gp = SingleTaskGP(X_norm, Crashed)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(
                final_crash_gp.likelihood, final_crash_gp))
        else:
            final_crash_gp = None
        with torch.no_grad():
            post_mean = final_gp.posterior(X_norm).mean.squeeze(-1)  # (n,)
            cand_mask = torch.ones_like(post_mean, dtype=torch.bool)
            if final_crash_gp is not None:
                crash_mean = final_crash_gp.posterior(X_norm).mean.squeeze(-1)
                feas = crash_mean < 0.5
                if bool(feas.any()):
                    cand_mask = feas
            masked = post_mean.clone()
            masked[~cand_mask] = -float('inf')
            best_idx = int(masked.argmax().item())
    except Exception as e:
        print(f'  [warn] GP-posterior-best 선택 실패 ({e}) → raw best 로 fallback')
        selection_rule = 'raw_best_observed (fallback)'

    best_x_norm = X_norm[best_idx]
    best_x = x_norm_to_scale(best_x_norm).tolist()
    best_Q = float(Y_neg_Q[best_idx].item())   # 해당 점의 관측 Q
    best_params = dict(zip(PARAM_KEYS, best_x))
    best_scales = dict(best_params)

    print(f'\n=== BO sweep done ===')
    print(f'  selection rule: {selection_rule}')
    print(f'  recommended config observed Q: {best_Q:.3f}')
    print(f'  (raw best observed Q: {raw_best_Q:.3f})')
    for k, v in best_params.items():
        print(f'    {k} = {v:.3f}')

    # 결과 자동반영 (2026-06-18): best_params 를 yaml 에 write-back.
    # 기존 sed_yaml_override({}) 는 override 만 끄고 q_*_scale_live 에는 '마지막 후보값'
    # 이 남았음 → best_params 로 덮어써 "BO 결과 바로 반영". (override_mode 도 off)
    sed_yaml_override(best_params, mode='off')
    print(f'  ✅ best_params → {YAML.name} 자동반영 (q_*_scale_live 갱신)')

    # 결과 저장
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = BO_DIR / f'bo_turbo_{ts}.json'
    out_path.write_text(json.dumps({
        'args': vars(args),
        'mode': args.mode,
        'algorithm': 'BoTorch_TuRBO_ConstrainedEI',
        'max_speed': read_yaml_max_speed(),
        'maps': maps_list,
        'history': history,
        'best_scales': best_scales,
        'best_params': best_params,
        'best_Q': best_Q,
        # Improvement B: 최종 추천은 GP-posterior-best (Shahriari 2016 p.156).
        'selection_rule': selection_rule,
        # 참조용 raw empirical best (운좋은 lap 포함 가능).
        'raw_best_scales': raw_best_params,
        'raw_best_params': raw_best_params,
        'raw_best_Q': raw_best_Q,
    }, indent=2, default=str))
    print(f'  saved → {out_path}')


if __name__ == '__main__':
    main()
