# IFAC2026 MPCC Pipeline

F1TENTH MPCC + Bayesian Optimization 의 전 파이프라인. 트랙 생성부터 BO 학습, 디버깅, 알려진 함정까지.

기준 일자: 2026-05-28  ·  Target: 4 m/s VPMPCC + Liniger baseline + EVO-MPCC BO

---

## 목차

0. [Quick Start](#0-quick-start)
1. [Track Generation](#1-track-generation)
2. [IQP Raceline (PP baseline 용)](#2-iqp-raceline-pp-baseline-용)
3. [PP Baseline 측정](#3-pp-baseline-측정)
4. [MPCC Algorithm](#4-mpcc-algorithm)
5. [BO Learning](#5-bo-learning)
6. [Operations & Cleanup](#6-operations--cleanup)
7. [참고 논문](#7-참고-논문)
8. [Fix 연대기](#8-fix-연대기)
9. [디렉터리 구조](#9-디렉터리-구조)
10. [알려진 한계 / Phase 2-3 계획](#10-알려진-한계--phase-2-3-계획)
11. [디버깅 체크리스트](#11-디버깅-체크리스트)

---

## 0. Quick Start

```bash
# 환경
cd ~/IFAC2026_SH
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"

# 1) Track generation (centerline + boundaries + start_pose + 4 mpc CSV)
python3 src/nonlinear_mpc_acados/scripts/gen_random_track.py \
    --name rand_a --preset race

# 2) IQP raceline 생성 (PP baseline 측정 전용; mpcc 자체는 centerline 사용)
ros2 launch global_planner create_path.launch.xml map:=rand_a

# 3) PP baseline (raceline 추종 → 목표 lap_time = t_lb)
python3 src/nonlinear_mpc_acados/scripts/pp_baseline.py \
    --v 4.0 --map rand_a --n_laps 3

# 4) MPCC 단독 검증 (manual)
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=rand_a

# 5) BO 학습 (Algorithm 1 Shahriari, BoTorch TuRBO)
ros2 launch nonlinear_mpc_acados bo_train.launch.py map:=rand_a n_calls:=20
```

---

## 1. Track Generation

### 1.1 스크립트
`src/nonlinear_mpc_acados/scripts/gen_random_track.py`

### 1.2 PRESETS
```python
PRESETS['race'] = dict(
    n_points     = 22,   # control point 수
    n_regions    = 9,    # Voronoi region 수
    max_bound    = 30,   # bounding box [m]
    width        = 2.0,  # track width [m]
    smoothing    = ...,  # Catmull-Rom / cubic spline
)
```
결과: 약 80m 길이, hairpin 1-2 개 포함. f-map (TUM) 복잡도와 유사.

### 1.3 출력 파일 (모두 `src/stack_master/maps/<name>/` 아래)

| 파일 | 용도 | 포맷 |
|------|------|------|
| `<name>.png` | gym occupancy 맵 | grayscale PNG |
| `<name>.yaml` | ROS map_server 메타 | resolution / origin |
| `global_waypoints.json` | centerline (+ 후속 IQP raceline) | TUM wpnts 포맷 |
| `centerline.csv` | flat CSV | x, y, w_tr_right, w_tr_left |
| `ot_sectors.yaml` | overtaking sectors (default 0) | sector_idx / start_s / end_s |
| `speed_scaling.yaml` | sector 별 v scale (1.0) | dict |
| `start_pose.yaml` | spawn pose | sx, sy, stheta |

추가로 mpc 가 직접 읽는 CSV 4개가 `share/nonlinear_mpc_acados/tracks/track<name>/` 에 install 됨 (colcon build 시 setup.py 의 `_track_entries` 가 자동 감지).

### 1.4 중요한 보정 처리
- **centerline rolling**: index 0 을 (0,0) 최근접점으로 roll → spawn 과 시작점 일치.
- **start_pose.stheta = psi[0]**: centerline 접선 = spawn yaw (90° 미스매치 fix).
- **Heilmeier forward-backward velocity profile**: brake-aware ref_v (코너 전 미리 감속).

### 1.5 트랙 변경 시
`gen_random_track.py` 의 PRESETS dict 수정 또는 새 preset 추가 (`PRESETS['name']=dict(n_points,n_regions,max_bound,...)`).

---

## 2. IQP Raceline (PP baseline 용)

### 2.1 Launch
`src/global_planner/launch/create_path.launch.xml` → `trajectory_optimizer.py`

### 2.2 알고리즘
TUM `trajectory_planning_helpers` IQP min-curvature optimizer. Cost=곡률² 적분, constraint=track 내부+width, 출력=raceline wpnts + 점별 vx_mps (a_lat/a_lon 한계).

### 2.3 Option α (centerline 보존)
`'centerline_waypoints': existing_cl_wpnts or _traj_to_wpnts(traj_iqp)` — 기존 `centerline_waypoints` 보존해 mpc 가 원본 825-pt centerline 유지 (이전 버그: IQP 가 centerline 덮어써 raceline 을 centerline 으로 오인).

### 2.4 결과
`global_waypoints.json` 에 `raceline_waypoints` + `speed_profile` 추가. **PP baseline 전용**, MPCC 는 centerline 추종 (VPMPCC).

### 2.5 IQP 가 실패할 때
`inflation_factor` 큼 → corridor 음수 → solve fail. spike/loop centerline → smoothing 강화. 해결: `centerline.csv` width 칼럼을 raw boundary 거리로 재계산.

---

## 3. PP Baseline 측정

### 3.1 스크립트
`src/nonlinear_mpc_acados/scripts/pp_baseline.py`

### 3.2 동작
`full_sim.launch.py mode:=pp` 백그라운드 → PP + IQP raceline + speed_profile 추종 → N lap 후 kill → `/mpc/lap_count`,`/mpc/lap_time` 구독 → JSON 저장.

### 3.3 결과
`~/bo_results/pp_baseline_v<v>_<map>_<ts>.json`:
```json
{
  "map": "rand_a",
  "v": 4.0,
  "laps": 3,
  "lap_times": [27.987, 22.639, 23.121],
  "best_lap_time": 22.639
}
```

### 3.4 현재 baseline (rand_a)
| v_max | PP best | 측정 일자 | json |
|-------|---------|----------|------|
| 4.0 | **22.639s** | 2026-05-27 | `pp_baseline_v4.0_*.json` |
| 5.0 | **18.460s** | 2026-05-27 | `pp_baseline_v5.0_*.json` |
| 6.0 | **15.659s** | 2026-05-27 | `pp_baseline_v6.0_*.json` |

- 이 값이 BO 의 `t_lb` 로 사용됨. MPCC 가 이보다 빠르면 보너스 +20·(t_lb - T_lap)
- BO 가 `find_pp_baseline_t_lb(map, v)` 로 자동 검출. v 별 별도 측정 필수.
- 단축률: v=4→5 (-18.5%), v=5→6 (-15.2%). 코너 cap 으로 점점 감소.

### 3.5 PP baseline 재측정 trigger
- 트랙 변경 (PRESETS 수정 or 새 맵 생성)
- v_max 변경 (4 → 5 등)
- IQP 알고리즘 / weight 변경

---

## 4. MPCC Algorithm

### 4.1 State / Model

**Dynamic 8-state Pacejka** (default; kinematic 5-state 옵션):
```
x_state(dyn, 8) = [x, y, ψ, vx, vy, r, s, δ_prev]
x_state(kin, 5) = [x, y, ψ, s, δ_prev]
u_input         = [v|a_x, δ, p_v]   # kin: 속도 v / dyn: 가속 a_x ‖ 조향각 δ ‖ virtual progress velocity
```
- `δ` 는 **직접 제어**(state 아님). `δ_prev` = ZOH 추적 state → steer-rate cost = (δ − δ_prev).
- `vy` (lateral velocity), `r` (yaw rate): tire slip 모델링 필수 (dynamic only).
- `s`: progress along centerline (cumulative arc length).
- `p_v`: virtual progress speed (MPCC trick — progress 를 cost 화).

**Tire model** (`dyn_tire_model` yaml): `tanh` ✅ 현재 (2026-05-27 #9, saturation). 옵션 `linear`, `pacejka`(full Magic Formula).

**Horizon**: `N_horizon=25`, `dT=0.04s` → 1.0s lookahead (2026-06-01: 50→25, solve 10.5→5.18ms). acados `SQP_RTI + HPIPM`, 1 SQP iter.

### 4.2 Cost (Liniger + VPMPCC, NONLINEAR_LS)

매 stage 의 residual vector `y_expr`:

| # | Residual | Weight | 의미 |
|---|----------|--------|------|
| 1 | `sqrt(q_cte_scale) · sqrt(att) · (e_c - e_c_ref)` | q_cte (baked 15) | contour error (centerline 거리) |
| 2 | `sqrt(q_lag_scale) · sqrt(att) · e_l` | q_lag (baked 80) | lag error (s 방향) |
| 3 | `sqrt(q_psi_scale) · yaw_err` | q_psi | heading 일치 |
| 4 | `sqrt(q_v_scale) · (vx - ref_v_expr)` | q_v (VPMPCC RVP) | reference velocity 추종 |
| 5 | `sqrt(q_dd_scale) · δ` | q_dd | steering magnitude |
| 6 | `sqrt(q_p_scale) · (p_v - vmax)` | q_p (baked 1) | progress maximization |
| 7 | `side_term` | 0 | VPMPCC 에서는 비활성 (장애물 회피) |
| 8 | `sqrt(q_drate_scale) · (δ - δ_prev)` | q_drate | steer rate (oscillation) |
| 9 | `sqrt(q_dv_scale) · a_x` | q_dv (baked 15) | longitudinal accel penalty (2026-05-27 #8) |

**Cost = ½ · ‖y_expr · √W‖²**. W 는 codegen 시 baked, scale_p 는 live param (rqt / BO).

**B 단순화 (2026-05-22)**: `attenuation=1`·`att_kappa=1` (장애물/κ attenuation off — corner 에서도 centerline tracking), `side_term=0` (장애물 회피 off). 이전 `_old` 변수는 2026-05-27 삭제. ★ **apex bias 는 2026-06-01 복원** — 현재 `e_c_ref = -D_apex·tanh(signed_κ/0.20)` 활성 (D_apex_live=0.63). 단 κ-cap 이 centerline-κ 기반이라 apex 가 더 빠른 ref_v 를 못 받음 → 라인만 바뀌고 속도이득 無 (§8 구조적 한계).

### 4.3 ref_v_expr (VPMPCC + κ-cap)

```python
ref_v_track = forward_backward_vel_at_s(s)  # Heilmeier profile
v_cap_kappa = ca.sqrt(a_lat_safe / (kappa_at_s + 1e-6))
ref_v_expr  = smooth_min(ref_v_track, v_cap_kappa)  # 2-way smooth min
```
- κ-cap 은 ref_v 의 상한만 잡음. Cost 는 RVP `q_v·(vx - ref_v)` 하나만 → **이중 cost 없음**
- CiMPCC g(κ) = vmax·exp(-α·κ²) 는 alpha=10, 2 모두 stuck → reverted

### 4.4 Constraints

| 변수 | lbx | ubx | 의미 |
|------|-----|-----|------|
| v (u[0], kin) / a_x (dyn) | 0 / -3 | max_speed / +4 | 속도 직접 제어 (kin) 또는 가속 (dyn) |
| δ (u[1]) | -0.45 | +0.45 | mpc_max_steering. hairpin 통과 (2026-06-02 0.3→0.45) |
| p_v (u[2]) | 0 | max_speed | progress speed cap |
| vx (state, dyn) | 0 | `v_max + 0.5` | dynamic 절대 cap. IPM 마진 +0.5 |

**Corridor (path constraint)**: `|e_c| ≤ mpc_corridor_half_width` (0.75m). track width 2m → boundary half 1.0m, R_car 0.15m → 마진 0.10m. `inflation_factor=0.0` 필수 (0.1+ 면 corridor 음수 → solve fail).

### 4.5 Live parameters (rqt / BO 가 갱신)

mpc_node 가 cycle 마다 yaml `<key>_live` 읽어서 `p_sym` 에 주입:
```
q_cte_live, q_lag_live, q_d_delta_live, R_safe_live, M_slack_live,
a_lat_safe_live, D_detour_live, D_apex_live, R_car_live,
commit_dist_live, cost_spike_thr_live, alpha_steer_live,
q_cte_scale_live, q_lag_scale_live, q_psi_scale_live, q_v_scale_live,
q_dd_scale_live, q_p_scale_live, q_drate_scale_live
```
**BO 가 수정하는 건 7개 (§ 5.2)**: `q_cte, q_lag, q_psi, q_v, q_p, q_drate, q_dv` 의 `_scale_live`.

### 4.6 ROS wrapper (`mpc_node.py`)

#### Speed command
```python
cmd.drive.speed = max(traj[-1, 3], v_now + 0.5)
v_cap            = mpc.v_max
cmd.drive.speed  = min(cmd.drive.speed, v_cap)
```
under-actuation 방지 (predicted vx 가 너무 작으면 가속 명령 약함).

#### STUCK 복구 (gym in_collision latch escape)
**문제**: f1tenth_gym `base_classes.py:288` 의 `in_collision=True` latch 되면 reverse 무시.
**해결** (`mpc_node.py` persistent stuck detect): 속도≈0 + cost spike + 지속 시 `cmd.speed=-0.5` (후진) + `_publish_safe_reset(x0)` + 5s cooldown. `_publish_safe_reset` = 최근접 s → 전방 2m 좌표+접선 yaw → `PoseWithCovarianceStamped` publish to **`/sim/initialpose`** (gym 전용, `/initialpose` 아님). gym_bridge 가 차 reset.

### 4.7 yaml 핵심 값 (⚠️ rand_a centerline 시절 — 2026-06-01; **현재 deploy(final/raceline/dynamic 22s)는 §12 참조**)
```yaml
max_speed: 5.0
mpc_max_steering: 0.3
N_horizon: 25                   # 2026-06-01: 50→25 (lap 17.68 best·solve 절반, STUCK=0). 미커밋
dT: 0.04
mpc_corridor_half_width: 0.75
inflation_factor: 0.0
auto_tune: false                # S1
override_mode: 'off'            # S1 영구
use_dynamic: true
dyn_tire_model: tanh            # 2026-05-27 #9
track_source: centerline        # VPMPCC
use_lmpc: false                 # B-mode sustained 가 최선 (§ 8)

a_lat_safe_live: 9.0            # 2026-06-01 #3: 6→9 (real vy/r → a_lat 정확, 실측 grip 9.7)
D_apex_live: 0.0               # 2026-05-27 #1: dead path (e_c_ref=0)
cost_spike_thr_live: 8000.0    # 2026-05-28 #11

# Deploy weights (B-mode racing line, 2026-05-30 BO 재최적화 → lap 17.84/N=50, 17.68/N=25):
#   q_cte=0.591 (low = apex 자유) · q_drate=3.594 (high = smooth) 가 핵심.
#   v=6 retry/v=4 Phase A 참고 weight 는 § 5.9 / § 5.10.
q_cte_scale_live: 0.591
q_drate_scale_live: 3.594
# (q_lag/q_psi/q_v/q_p/q_dv 는 BO best 값 — live yaml 참조)
```
**주의**: BO 가 `sed_yaml_override` 로 매 trial 이 yaml in-place 덮어씀. dead BO 가 나쁜 weight 남기는 회귀 있음 (§ 8 2026-05-29). BO 전후 `git diff <commit> -- ddrx_unified_params.yaml` 확인 필수.

---

## 5. BO Learning

### 5.1 Algorithm — Shahriari 2016 Algorithm 1

`scripts/bo_sweep_turbo.py`:

```
1. n_initial=5 Sobol samples (in [0,1]^d log-uniform space) → 실험 → (X, Y)
2. for iter = 1..n_calls:
       fit GP(X, Y)                          # SingleTaskGP + Matern 5/2 + ARD (or isotropic)
       print ARD lengthscale per dim         # 2026-05-27 #4: 진단
       acq = ConstrainedExpectedImprovement
       x_next = argmax acq(x) over TR        # TuRBO trust region
       scale_next = 10 ^ (LOG_BOUNDS · x_next)  # 2026-05-27 #5: log → linear
       y_next = experiment(scale_next)        # full sim + Q 계산
       X ← X ∪ {x_next};  Y ← Y ∪ {y_next}
       update TR length (success → ×2, fail → ÷2)
3. return argmax Y
```

- **GP**: SingleTaskGP, Matern 5/2 + ARD, GaussianLikelihood, MAP via `fit_gpytorch_mll`.
- **Acquisition**: `ConstrainedExpectedImprovement` (crash hard-constraint × EI), best_f = qualified max Q.
- **TuRBO TR** (Eriksson 2019): length_init=0.8, length_min=0.5^7 (restart), length_max=1.6, success_tol=3 (×2), failure_tol=max(4,dim)=5.

### 5.2 Search Space (7D, log-uniform)

2026-05-27 review fix #5/#6/#8 적용 후:
```python
DIM = 7
PARAM_KEYS = ['q_cte', 'q_lag', 'q_psi', 'q_v', 'q_p', 'q_drate', 'q_dv']
PER_DIM_LO = {'q_cte': 0.1, ..., others: 0.3}     # q_cte 만 0.1 (느슨 추종 자유)
PER_DIM_HI = {all: 5.0}
LOG_BOUNDS = log10(BOUNDS)                         # GP 는 log space 에서 작동
```

**Log-uniform 의미** (`x_norm_to_scale`):
- x_norm = 0   →  scale = 10^LOG_LO  (q_cte = 0.1, 나머지 = 0.3)
- x_norm = 0.5 →  scale = geomean(LO, HI)         (q_cte = 0.71, 나머지 = 1.22)
- x_norm = 1   →  scale = 10^LOG_HI = 5.0
- **이유**: scale 은 곱셈 의미 (0.5×W vs 2×W 가 대칭) → log space 가 prior 로 자연

**왜 7D**: 이전 13D (4w×3 buckets+D_apex) 는 bucket switching 으로 cost surface 불연속. 5D 는 q_psi/q_dv 빠져 yaw/longitudinal 학습 못함. 7D 가 EVO-MPCC search space 와 정합 (q_Δp_v 만 빠짐, 새 state 필요). n_calls ≥ 70 권장 (Shahriari 10·dim).

**Isotropic kernel** (`--isotropic`): 데이터 부족 시 ARD noise overfit 위험 ↓ (single lengthscale 안정화, 비교 실험용).

### 5.3 sed_yaml_override

```python
def sed_yaml_override(params, mode='off'):
    # yaml 의 q_*_scale_live: <value> 만 in-place rewrite
    # override_mode 는 항상 'off' 고정
```
mpc_node 가 별도 codegen 없이 매 cycle 새 값 읽음. BO iter 간 launch 재시작은 sim_bridge 한정 (mpc node 는 codegen 결과 캐시 유지).

### 5.4 Objective Q — EVO-MPCC LTM + Soft Penalty

`scripts/eval_run_quality.py`:

**Hard reject** (`Q = -1000`):
- `crashed = True` (collision)
- `laps < 1` (lap 못 끝냄)
- `n_resets > 2` (reset 으로 false lap 차단; reset 1-2 회까지는 OK)
- `lap_time < 0.5 × ideal_lap_time` (teleport false lap)

**Soft penalty** (qualified 일 때, BO 가 cliff 가 아닌 gradient 학습):
```python
soft = (
    1.0  * max(lat_g_peak - 15, 0) ** 2     # 측방 가속도 [m/s²]  (2026-06-01 #2: 50→15, real vy/r 로 정확)
  + 50.0 * max(cte_rms    - 0.8, 0) ** 2     # corridor 이탈 [m]
  +  5.0 * max(shake_rms  - 5,  0) ** 2      # high-freq 진동 [m/s²]
  +100.0 * max(0.8 - alive_frac, 0)          # mpc dropout %
  + 10.0 * max(switch_count - 3, 0)          # fallback flap 횟수
)
Q = -T_lap + 20.0 * max(t_lb - T_lap, 0) - soft
```
- `-T_lap` 빠를수록 보상, `+20·max(t_lb-T_lap)` PP 돌파 보너스, `-soft` 안전/부드러움 연속 penalty.
- **lat_g_max 변천**: 2026-05-28 #12 에 15/20→50 (당시 yaw finite-diff noise 로 정상 corner p99 39-44 가 학습 막아서). 2026-06-01 #2 에 real vy/r 노출로 a_lat 정확해져 50→15 복귀. argparse↔함수 default 정렬 (이전 default 불일치로 더 빡빡했음).

### 5.5 PP baseline 자동 연동
```python
def find_pp_baseline_t_lb(map_name, v):
    # ~/bo_results/pp_baseline_v<v>_<map>_*.json 중 가장 최근
    # data.get('map') == map_name and data.get('laps') >= 1
    return data['best_lap_time']
```
`evaluate_csv()` 가 `--t_lb <PP best>` 로 eval 에 전달.

### 5.6 launch wrapper

`launch/bo_train.launch.py`:
```bash
ros2 launch nonlinear_mpc_acados bo_train.launch.py \
    map:=rand_a n_calls:=20 n_initial:=5 n_laps:=3
```
내부적으로 `scripts/run_bo.sh` 호출 → leftover pkill + sleep 5 + python3 bo_sweep_turbo.

### 5.7 Cleanup pkill 주의
```bash
pkill -9 -f "gym_bridge|mpc_node|mpc_debug|state_machine|frenet|simple_mux|rviz2|fake_topic|pp_fallback|ftg_fallback|global_republisher|joy_node|robot_state|obstacle"
```
- `ros2.*launch` 와 `nonlinear_mpc` 패턴 **제외** — 부모 launch 자기 죽임 ("Killed")
- 구체 node 이름만 매칭

### 5.8 BO log 포맷
```
[iter 1] scales=q_cte=4.00, q_lag=2.00, q_psi=1.00, q_v=1.50, q_p=1.00, q_drate=3.00, q_dv=1.50
  [   36s] lap=1
  [   60s] lap=2
  [   83s] lap=3
   sim done: reached 3 laps
   Q=-478.66 [OK]  lap=22.159s (ideal 20.59, t_lb 22.64)
   v_avg=3.44  v_max=4.00  lat_g_peak=19.01
   shake=0.1153  cte=0.387  laps=3  alive=100%  sw=1
  GP ARD lengthscale: q_cte=1.19, q_lag=1.07, q_psi=1.10, q_v=0.38, q_p=0.15, q_drate=1.07, q_dv=0.97
  TR: length=0.800  success=1  failure=0  best_Q=-478.66
```

### 5.9 BO Phase A 결과 (2026-05-27, rand_a, v=4)

> 참고용 (현 deploy 는 2026-05-30 B-mode racing line, § 8). 핵심만 남김.

- **best**: 67 iter, Q=9.17, lap 20.96s (PP 22.64 -7.4%, ideal 20.59 +1.8%, 거의 물리한계). OK 59/67 (88%, soft penalty 효과).
- **best params**: `q_cte=3.665, q_lag=0.397, q_psi=0.597, q_v=0.313, q_p=4.282, q_drate=3.533, q_dv=1.675`.
- **Lengthscale**: q_lag(12.5)·q_psi(10.8) "정보 없음" (제거 가능), q_v(0.27)·q_drate(0.29)·q_dv(0.32) 매우 sensitive.
- **발견**: VPMPCC RVP 거의 무용 (q_v=0.31, baked의 1/3만 사용 → centerline 모드선 progress 직접 최대화가 효율). q_p 지배적 (4.28 수렴). q_drate/q_dv 가 진동·longitudinal 평탄화 핵심.

### 5.10 BO v=6 retry (2026-05-28, post-#13 cliff softening)

> 참고용. #13 (n_resets cliff→linear) 효과 입증 — 1차는 모든 OK 가 cap 800 직격으로 학습 실패, #13 후 gradient 복원.

- **best**: 30 iter, Q=-47.57, lap 14.32s (PP 15.66 -8.6%), n_resets=2, OK 25/30, cap 직격 0/25.
- **best scales**: `q_cte=4.956 q_lag=0.502 q_psi=0.462 q_v=1.378 q_p=0.417 q_drate=4.992 q_dv=1.085`.
- **v=4 ↔ v=6 weight 반전 ★**: q_p 4.28→0.42 (1/10), q_cte 3.67→4.96, q_drate 3.53→4.99, q_v 0.31→1.38. v=4 는 "progress 압도+자유 corridor", v=6 는 "centerline 단단히+osc 억제". **속도가 빨라지면 weight 전략 자체가 바뀜** → single-v static tune 한계 증명, multi-v BO sweep 가치 입증.

---

## 6. Operations & Cleanup

### 6.1 빌드
```bash
cd ~/IFAC2026_SH
colcon build --symlink-install                # 전체
colcon build --packages-select nonlinear_mpc_acados --symlink-install  # 단일
```
`--symlink-install` 은 python 파일 수정 시 rebuild 불필요 (acados codegen 만 예외).

### 6.2 acados codegen 재실행
yaml 의 다음 값을 바꾸면 codegen 다시 (~30s):
- `N_horizon`
- `dT`
- `dyn_tire_model`
- `max_speed` (cost expression 의 baked vmax 사용 시)
- `q_*_def` (baked weights — BO_params_LTM.json)

scale_live 만 바꾸는 건 codegen 불필요.

### 6.3 CycloneDDS
`~/cyclonedds.xml` 의 `<Interfaces>` 에 `<NetworkInterface name="lo"/>` 필수 — 빠지면 local sim discovery 깨짐.

### 6.4 환경 격리
**절대 동시 source 금지**: `~/IFAC2026_SH/install/setup.bash` ↔ `~/creating_autonomous_car_ws/install/setup.bash`. 두 ws 가 `frenet_conversion`/`f110_msgs` 를 다르게 patch → Wpnt 메시지 정의 충돌.

### 6.5 Background 정리 (수동)
```bash
pkill -9 -f "gym_bridge|mpc_node|mpc_debug|state_machine|frenet|simple_mux|rviz2|pp_fallback|ftg_fallback|global_republisher|obstacle"
```
정적 obstacle_manager 가 며칠 살아있을 수 있음 → `ps aux | grep obstacle` 후 PID kill.

### 6.6 mpc_logs / bo_results 정리
- `~/mpc_logs/mpc_*.csv`: launch 마다 누적 (>1GB). `~/bo_results/bo_*.log`,`*.json`: BO trace. `archive_pre_*` 는 큰 step 전 백업 (수동 삭제 OK).

---

## 7. 참고 논문

| 영역 | 논문 | 적용 위치 |
|------|------|---------|
| MPCC 기본 | Liniger et al. 2015 (Optimization-Based Autonomous Racing of 1:43 Scale RC Cars) | `acados_kinematic.py` cost 1-3, 6 |
| Velocity profile MPCC | VPMPCC (Vázquez 2020) | `q_v · (vx - ref_v)` (cost 4) |
| Brake-aware ref_v | Heilmeier 2019 (Minimum Curvature Trajectory Planning) | `forward_backward_vel` in `gen_random_track.py` |
| Apex bias | TUM AR | 시도 후 폐기 (B 단순화) |
| κ-mapping cap | CiMPCC | 시도 후 폐기 (alpha=10, 2 모두 stuck) |
| LTM objective | EVO-MPCC | `Q = -T_lap + λ·max(t_lb - T_lap, 0)` in `eval_run_quality.py` |
| BO 알고리즘 | Shahriari et al. 2016 (Taking the Human Out of the Loop, Algorithm 1) | `bo_sweep_turbo.py` 전체 구조 |
| TuRBO | Eriksson et al. 2019 NeurIPS (Scalable Global Optimization via LCB Trust Regions) | TR length adaptive |
| ConstrainedEI | Gardner et al. 2014 | `ConstrainedExpectedImprovement` from BoTorch |
| GP residual (Phase 3) | Kabzan et al. 2019 (Learning-Based MPCC) | future: L4acados 통합 |

---

## 8. Fix 연대기

> **2026-05-29 이전 항목은 한 줄로 압축됨** (날짜+무엇+결과). 상세 reasoning 은 코드 주석/memory 참조. 2026-05-29 이후는 full 보존.

| Date | Fix (한 줄 요약) |
|------|-----|
| 2026-05-22 | B 단순화 (apex/κ-att/obstacle off, BO 13D→5D); S1 (auto_tune off, override off, D_apex=0); S-clip (`ubx[0]=v_max+0.5`, corridor 0.95→0.75, IPM 발산 방지); N 50→40 (6.4m lookahead 충분) |
| 2026-05-26 | false-lap+n_resets filter (reset teleport 카운트 차단); `/initialpose`→`/sim/initialpose` (gym latch escape); `inflation_factor` 1.2→0.0 (corridor 붕괴); trajectory_optimizer Option α (centerline 보존); track_name=map_name override ('f' 맵 오로딩 fix); centerline rolling + stheta=psi[0] (spawn 90° fix) |
| 2026-05-27 | Soft penalty (cliff 제거, BO gradient 복원); `evaluate_csv(map_name)` (cte=5.25m 오계산 fix); cleanup (_old 변수·bucket/poly 16 keys·dead BO 4개); Review fix #1-9: D_apex_live=0, shake 5→25, q_cte lower 0.3→0.1, ARD LS 출력, log-uniform space, BO 5D→6D(q_psi), `--isotropic`, 9th residual q_dv·a_x(6D→7D), tire linear→tanh; a_x_input scope fix; n_resets cliff→soft; BO archive + centerline warm-start |
| 2026-05-27 | **BO Phase A 완료**: 67 iter, Q=9.17, lap 20.96s (PP -7.4%), best `[3.665,0.397,0.597,0.313,4.282,3.533,1.675]`; v=5 동일 weight generalize 17.12s; PP baseline v=5/6 측정 (18.46/15.66s) |
| 2026-05-27 | Stuck recovery `v_cmd_prev > 2.0`→`>1.5` (seed_v=2.0 충돌 fix); lat_g spike max→p99; soft penalty cap 800; Phase D Step 1-3 (extract/train/wrap GP, 85K samples RMSE 96/72/62%); autoreg_speed_bo.sh (v=5/6/7/8, N 50/60/70/80) |
| 2026-05-28 | 실차 포팅 (`creating_autonomous_car_ws/.../nonlinear_mpc_acados/`, v=5 weights, nuc5 baked-path rebuild fix); autoreg v=6 plateau Q=-796 진단 (root=cost_spike fallback 무한 cycle, thr=1500 vs 실측 119K) |
| 2026-05-28 #10-13 | v_floor v 비례화 (`max(0.5·v_max, v_est+1)`); cost_spike_thr 1500→8000; **lat_g_max 15/20→50** (★ 정상 corner p99 39-44 가 BO 학습 막던 근본 fix, argparse default 15 도 정렬); n_resets penalty quadratic 50→linear 30, HARD_RESET_LIMIT 15→30 |
| 2026-05-28 | **BO v=6 retry 완료**: 30 iter, Q=-47.57, lap 14.32s (PP -8.6%, ideal +4.3%), n_resets=2, OK 25/30, cap 직격 0/25. v=4↔v=6 weight 반전 (q_p 4.28→0.42, q_cte 3.67→4.96) — multi-v sweep 가치 증명. § 5.10 상세 |
| 2026-05-28 #14-15 | ftg_fallback `range_offset` 0.0(float)→30 (slice TypeError fix, fallback path 활성화); STUCK detect `v_cmd_prev > 1.5`→`>0.0` (자포자기 vcmd≈0.02 미감지 fix). rand_a 곡률 분석: κ_max=0.44, R_min=2.26m, v_cap@a_lat30=8.24 → 박힘은 물리한계 아님 (STUCK detect bug) |
| 2026-05-29 ★★ | **BO config-clobber 회귀 발견+복원** (커밋 `b554f25`) | 모든 "SQP_RTI 흔들림"·wedging·backward버그의 근본원인 = 죽은 BO(18:02 iter7)가 live `ddrx_unified_params.yaml`에 남긴 나쁜 q_*_scale_live (q_cte 3.79→0.45, q_drate 2.63→0.42). committed 값 복원 → **shake 0.32→0.028, MINSTEP 23→0, STUCK 50→0, lap 47s→19s.** `sed_yaml_override`가 매 trial yaml in-place 덮어쓰고 종료 trap 없어서 kill 시 leftover. → `run_bo.sh`에 INT/TERM/EXIT trap으로 yaml 복원 추가. BO 전후 `git diff <commit> -- ddrx...yaml` 확인 필수 |
| 2026-05-29 | **backward-creep 버그 fix** (`acados_kinematic.py` `STUCK_REVERSE_DIST`) | 박힌 뒤 빠져나오고도 살짝 더 뒤로 가다 출발 = 고정 20-cycle 강제후진에 early-exit 없음. wedge점에서 0.20m 후진해 분리되면 즉시 종료 + teleport 시 중단 |
| 2026-05-29 | **LMPC functional fix** (`mpc_node.py`) | LMPC가 no-op이었음: `/mpc/lap_count` 구독이 `auto_step_enabled` 안에만 생성 → auto_step off(기본)면 lap-end 훅 미호출 → SS에 랩 0 buffer. fix: `auto_step OR use_lmpc`로 구독. 부작용(auto_step 램프 무조건 작동)은 `if not enabled: return` 가드. + SS query를 예측 horizon-end로(전진 attractor) + cost-to-go 정렬 + best-lap retention. (커밋 `ae83fc5`) |
| 2026-05-30 ★ | **RACING LINE: MPCC B-mode** (커밋 `ca58446`→`95f1496`) | centerline-hug(q_cte=3.79, 19.1s)에서 **낮은 contouring(q_cte↓)+높은 progress(q_p)+corridor=racing line**(Liniger MPCC). q_cte=0.8 손튜닝 18.87 → **BO 재최적화(B-mode warm-start, LTM) → q_cte=0.591/q_drate=3.594 등 = 일관 18.62s, STUCK=0 (best 18.44 < PP 18.46!).** low q_cte(apex 자유)+high q_drate(smooth)가 핵심. **현재 배포 config.** |
| 2026-05-30 | **LMPC softmin (proximity-weighted cost-to-go)** (커밋 `eb11a34`) | nearest-point attractor(추종만→drift)→ `cog=Σ softmax_j(-β d_j²)·Q_j`, r=√(w·cog+reg·d²). NLS 유지(EXTERNAL_COST 불필요), **QP 완벽안정**(옛 log-sum-exp softmin 불안정 회피). 초반 laps2-10 mean **18.48**(B-mode보다↑, PP 돌파 다수). 단 **drift 잔존**(laps11+ 19.27, soft attractor라 monotonic 보장X) → full Rosolia 하드 safe-set 제약이 다음. β=0.01 안정(1.0은 chatter). use_lmpc=false(B-mode가 sustained 최선) |
| 2026-05-30 | **raceline 데이터/min-time 진단** | track_source=raceline wedge 근본 = rand_a IQP raceline **불량 데이터**(d_left/d_right=0.5 vs 실제 1.25, vx 7-13 v=5에 못씀). 컨트롤러 버그 아님. min-time 재생성 시도 → `opt_mintime_traj` **모듈 미설치**(ModuleNotFoundError). idea2(offline raceline)는 ①모듈 설치 ②raceline-tracking 기하 wedge 둘 다 필요. track_loader에 forward-backward 속도프로파일 추가(raceline-only guard) + ref_v κ/v_max cap + corridor margin |

| 2026-06-01 ★ | **#3 real vy/r → a_lat 6→9 = 17.84s** (커밋 `6e83b12`) | gym_bridge가 real vy(v·sinβ)+참 yaw rate 노출 → a_lat=vx·r 정확. 실측 그립 9.7@slip5°(미포화) → a_lat_safe 6→9, 코너 ref_v √9/κ → **18.62→17.84 (PP 18.46 깸 -3.4%, ideal갭 +13%→+8.3%)**. apex bias(D_apex) 복원했으나 현 cost(κ-cap=centerline)론 느려져(18.9) D_apex=0 유지. |
| 2026-06-01 | **감사 fix 배치** (커밋 `486537d`) | 전체 코드감사: ① objective hardening(eval_run_quality): lat_g_max 50→15(real vy/r로 정확), argparse↔함수 default 정렬, lap_time min→cold-start제외 median(단일운좋은랩 BO익스플로잇 차단) ② a_lat 인버전 fix: hard cap 8→`a_lat_safe+1`=10 (√(9/κ) 추종시 steady a_lat=9>8 매코너 위반하던 silent slack 제거) ③ steer EMA clip, LMPC stuck_accum reset ④ gym_bridge copy.py 삭제. 검증 17.84유지·Q+53.6·lat_g_peak14<15. |
| 2026-06-01 ★ | **N_horizon 50→25** (미커밋) | N-sweep@deploy weight: N=50→35→30→25→20 전부 STUCK=0. **N=25 = lap 17.68(best)·solve 10.5→5.18ms(절반)·이론바닥(코너1개+제동0.18s 여유)**. 과거 "N25 wedge"=오염weight artifact(거짓 판명). lat_g_peak~16(늦은제동)이 유일흔적. EVO/RVP 분석: 우리 ref_v(forward-backward κ-cap)=EVO RVP, q_v=0.317로 약추종. MPPI 강건성=속도불변 물리제약(latacc/slip)에 의존·reference-free. dt=0.04 균일이라 변수 더↓는 pyramidal multi-dt 필요(#15). |
| 2026-06-01 | **LMPC apex 시도 — 미완(중요 진단)** | 목표=hairpin을 centerline κ=0.84(v 3.3) 대신 raceline κ=0.236(v 5)로. ① **hard safe-set terminal 제약(con_h_expr_e) = SQP_RTI(1-iter) 비양립** → lap2 활성 즉시 wedge. revert. **2논문(TC-LMPC WEVJ2023, Berkeley Racing-LMPC-ROS2) 둘다 terminal=SOFT(convex-combo+slack), 절대 hard 아님** 확인 [[lmpc-mpcc-references]]. ② **soft TC-LMPC**(softmin cost + IQP apex seed grip-clamp(min(v_max,√(a_lat/κ_raceline))) + B-mode q_cte0.3): **lap2=17.44 천장 첫 돌파!** 단 **지속실패**: 바깥drift(max_ec 0.5→1.07)+wall clip. ③ max_ec=0.6 필터 → 나쁜랩 SS거부 작동하나 **drift는 컨트롤러 자체**(SS얼려도 차가 바깥drift)→여전히 baseline보다 나쁨. ④ **terminal e_c가 raw+W_e[0]=q_cte_def×5 고정** → B-mode가 terminal엔 안먹혀 x_N을 센터라인에 묶음(RViz 예측선 끝점=센터라인 확인) → `sqrt_q_cte_scale·(e_c−e_c_ref)`로 fix(미테스트). **상태: 증분 접근 소진 — 근본진단 완료, 다음은 full 재구축 (포기 아님, 방법 전환).** PoC로 apex가 baseline 돌파(17.44<17.68) 검증됨(개념 맞음). 그러나 **지속화 실패 — 일관된 바깥drift→wall clip**. 시도/결과:
- hard terminal 제약 → SQP_RTI 비양립 wedge (revert).
- soft softmin + apex seed + terminal e_c/e_l/yaw scale fix(RViz 끝점=센터라인 관찰로 발견) → drift 줄었으나 **여전히 어느 순간부터 벽 연속 충돌**.
- max_ec=0.6 필터(나쁜랩 SS거부 작동) + lmpc_w↑(0.1→1.5) → drift 안 멈춤.
- ★ **근본원인 확정**: centerline 추종(e_c)을 줄여 apex 허용 = **벽에서 차를 잡아주던 힘 제거**. 그 자리를 apex Q가 메워야 하나 **우리 Q(고정 softmax, β=0.01≈uniform)가 너무 약해 라인을 못 붙듦** → 바깥 drift → 벽. 가중치 손튜닝(β/lmpc_w/q_cte)으로 균형 못 맞춤.
- ⚠️ **테스트 하니스 함정 2개 발견**(시간 낭비함): ① `_poll_lap_count` stale latch → 새 sim 첫 poll에 이전 lap=16 읽고 즉시종료(1랩만 돎). ② probe 자기 KILL 패턴 self-match + pkill exit1→set-e 중단. → 고정시간 주행+CSV 분석(`lmpc_probe2.py`)으로 우회.
- **다음 (제대로) = full TC-LMPC: optimized convex α (decision var, 외부 mini-QP=deep-dive Phase2) — 강하고 정확한 terminal 끌림.** 고정-softmax는 약한 고리. + 완전 reference-free + forward 점선택. Berkeley Racing-LMPC-ROS2 이식. **multi-day 전용 작업**(증분 X). 상세 [[lmpc-mpcc-references]].
- 인프라(gated off, deploy 무해): lmpc_raceline_json param·grip-clamp IQP seed·terminal scale fix·soft softmin. use_lmpc=false. |

**현재 best (2026-06-01)**: **N=25, 17.68s, solve 5.18ms, STUCK=0** (deploy q_cte=0.591 등, use_lmpc=false, a_lat_safe=9, max_speed=5, centerline corridor). 17.84(N=50)보다 N축소로 미세↑+solve절반. **N=25는 검증됐으나 미커밋** — 다음에 커밋 권장. ideal 16.47 갭 +7.3%.
**다음 (천장 17.68 돌파 / ideal 16.47)**: (1) **q_v-aware BO**(상한5→8, hardened objective, warm-start 必 — 안하면 random crash) → N=25용 weight 재최적화 (#14). (2) **full TC-LMPC**(soft terminal Q + reference-free + optimized α + apex seed; Berkeley ROS2 이식) — apex로 hairpin 3.3→5, 단 multi-day (#a). (3) **max_speed 5→6/7** — 직선 직접이득(코너 grip제한). (4) **pyramidal multi-dt** stage-dt cost 수정 → lookahead 유지하며 변수↓ (#15). (5) staged BO objective(느린영역=박힘횟수 채점, #16). (6) GP residual: velocity state만·feature에서 vy 제외(TC-LMPC/우리 gym vy=0 일치). 상세 [[lmpc-mpcc-references]] [[horizon-reduction-evo]].

---

## 9. 디렉터리 구조

```
~/IFAC2026_SH/
├── PIPELINE.md                              # 이 파일
├── MIGRATION.md                             # ROS1 → ROS2 변경점
├── README.md
├── cyclonedds.xml -> ~/cyclonedds.xml
├── install/  build/  log/
└── src/
    ├── nonlinear_mpc_acados/                # MPCC 본체
    │   ├── nonlinear_mpc_acados/
    │   │   ├── mpc_core/
    │   │   │   ├── acados_kinematic.py      # MPC class (cost/constraint/codegen)
    │   │   │   └── dyn_tire_model/
    │   │   ├── mpc_node.py                   # ROS wrapper + STUCK 복구
    │   │   ├── track_loader.py               # share/tracks/<name>/ 로드
    │   │   ├── mpc_debug_logger.py           # CSV logger
    │   │   ├── ftg_fallback_node.py          # Follow-The-Gap fallback
    │   │   ├── pp_fallback_node.py
    │   │   └── ml/                           # MLP weight scaler (use_ml_scale)
    │   ├── config/
    │   │   ├── ddrx_unified_params.yaml      # 단일 source of truth
    │   │   ├── mpc/BO_params_LTM.json        # codegen baked weights
    │   │   └── tire/
    │   ├── scripts/
    │   │   ├── gen_random_track.py           # § 1
    │   │   ├── pp_baseline.py                # § 3
    │   │   ├── bo_sweep_turbo.py             # § 5 (Algorithm 1)
    │   │   ├── eval_run_quality.py           # Q (LTM + soft penalty)
    │   │   ├── run_bo.sh                     # BO bash wrapper
    │   │   ├── auto_speed_collect.sh         # ML 학습 데이터 수집
    │   │   ├── eval_run_quality.py
    │   │   └── install_acados.sh
    │   ├── launch/
    │   │   └── bo_train.launch.py            # § 5.6
    │   └── setup.py                          # _track_entries 자동
    ├── global_planner/                       # § 2 (IQP raceline)
    │   ├── global_planner/trajectory_optimizer.py
    │   └── launch/create_path.launch.xml
    ├── stack_master/
    │   ├── maps/<name>/                      # 트랙 산출물 (§ 1.3)
    │   ├── config/SIM/sim.yaml               # sim 전용 override
    │   ├── launch/full_sim.launch.py         # 메인 sim launch
    │   └── stack_master/simple_mux_node.py   # cmd_vel multiplexer
    ├── controller/                           # pure pursuit etc.
    ├── f1tenth_gym_ros/                      # gym bridge (in_collision latch)
    │   └── f1tenth_gym_ros/gym_bridge.py
    └── ...
```

### 9.1 핵심 entry points
- `mpc_node = nonlinear_mpc_acados.mpc_node:main`
- `mpc_debug_logger = nonlinear_mpc_acados.mpc_debug_logger:main`
- `ftg_fallback_node`, `pp_fallback_node`

---

## 9.2 실차 포팅 (2026-05-28)

위치: `~/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpc_acados/` (unicorn-racing-stack 차 PC 워크스페이스에 신규 패키지). ROS2 + acados + BO weights.

- **포팅 파일**: `nonlinear_mpc_acados/mpc_node.py` (실차 topic `/vesc/odom`, `/vesc/high_level/ackermann_cmd`), `mpc_core/*` (acados_kinematic, gp_residual_wrapper, ipopt_kinematic, _ros_compat), track_loader, mpc_debug_logger, `config/ddrx_unified_params.yaml` (v=5 weights, max_speed=4 안전치), BO_params_LTM.json, `launch/mpcc.launch.xml`, scripts (pp_baseline/bo_sweep_turbo/eval_run_quality/extract_residuals/train_gp_residual), requirements/README.
- **빌드 함정 (해결됨)**: 이전 install/ 이 user `nuc5` 환경 baked path (`f110_msgs` hook) → 전체 rebuild (`rm -rf build install log` 후 colcon). vesc/urg/cartographer fail 은 무관.
- **옮기기**: A git (`HMCL-UNIST/creating_autonomous_car` push/pull), B rsync `-av --ignore-existing` (SSH 시 ★ 권장), C USB/tar, D scp.

---

## 10. 알려진 한계 / Phase 2-3 계획

### 10.1 Polynomial / fixed / bucketed mode
- yaml 의 bucket/poly 키는 2026-05-27 cleanup 으로 제거
- `mpc_node.py` 의 코드 경로는 남아 있음 (declare default + if 분기) — `override_mode='off'` 영구이므로 dead path
- 완전 제거하면 surgery 크니 일단 유지

### 10.2 dyn_tire_model
- ✅ `tanh` 전환 (2026-05-27 #9). `F_y = μ·D·F_z·tanh(B·α)` — 작은 α linear, 큰 α saturation. lat_g≈24 hairpin 에서 linear force 무한 증가 fix. (전환 후 scale_live 재튜닝 필요)
- Future: `pacejka` (full Magic Formula B/C/D/E)

### 10.2b BO search space (q_Δv 완료, q_Δp_v 미적용)
- ✅ 9th residual `q_dv·a_x` 연결, BO 6D→7D (2026-05-27 #8). y_expr/W 9 entries, p_sym slot 17, `q_dv_scale_live` 라이브 추가. codegen 재실행 필요.
- ⏸ q_Δp_v (progress velocity rate): 새 state `p_v_prev` 필요 → 별도 phase

### 10.3 GP residual learning (Phase D)

**목표**: 실측과 acados 예측의 차이를 GP 로 학습 → dynamics 보정. Kabzan 2019 + L4acados (clone `~/l4acados/`, PYTHONPATH import). 의존성 torch/gpytorch/casadi 이미 user-local.

#### 진행 상태
| Step | 파일 | 상태 |
|------|------|------|
| 1 | `scripts/extract_residuals.py` | ✅ 85K samples (vx [0.5, 4.0]) |
| 2 | `scripts/train_gp_residual.py` | ✅ gp_residual.pt (200 inducing, 300 iter, RMSE 96/72/62%) |
| 3 | `mpc_core/gp_residual_wrapper.py` + `mpc_node.py` | ✅ ResidualLearningMPC wrap + GPMPCAdapter |
| 4 | sim sanity test | ⏸ (autoreg 후 multi-v 데이터로 재학습) |

#### 통합 (mpc_node.py, setup_MPC 직후 use_gp_residual=true 시 자동)
`wrap_solver_with_gp(mpc, ckpt)` → `NormalizingResidualModel` (5D feature+normalize) + `B[3,0]=B[4,1]=B[5,2]=1` (vx/vy/r 채널) → `ResidualLearningMPC(ocp, B, model)` → `mpc.solver = GPMPCAdapter(...)`.

#### 데이터 흐름 (Phase A→D 통합)
```
autoreg CSV 누적 (post-tanh epoch ≥ 1779865200)
 → extract_residuals.py: alive=100%·vx>0.5 필터, (vy,r) positional-derivative 추정,
   acados 1-step predict → residual, outlier filter → gp_train_data.pt (5D in, 3D out)
 → train_gp_residual.py (PYTHONPATH=$HOME/l4acados/src): BatchIndependentInducingPointGPModel
   (sparse 200-300 inducing, ARD) → gp_residual.pt
 → mpc_node.py 자동 wrap (use_gp_residual=true): codegen ~30s, inference ~1-3ms/cycle,
   import 실패 시 plain acados fallback
```

#### Global GP 전략
**Single GP for all v ∈ [4, 8]**: autoreg 가 v=5/6/7/8 BO 돌리며 CSV 누적 → 코너 ref_v cap 으로 1 lap 이 vx [4, v_max] 분포 포함 → single GP 가 vx [0.5, 8] cover → deploy v_max 무관 작동.

#### Feature 선택
```python
# GP input 5D: vx, vy, r, delta, a_x  (나머지 6D 위치/누적 변수는 잔차 무관)
_GP_FEATURE_IDX = [3, 4, 5, 9, 8]   # state[3,4,5] + input[1,0]
```

#### Logger 한계 + 향후
- mpc_debug CSV 에 `vy`,`r` 없음 → positional derivative 추정 (sim 노이즈 적어 OK). 실차 transfer 시 logger 에 state estimator 출력 직접 추가 권장.
- 향후 옵션: per-v GP, online learning (`record_datapoint()`), ZeroOrderGPMPC, state estimator 통합.

### 10.3b L4acados API 핵심 (참고)
- `ResidualLearningMPC(ocp, B, residual_model, build_c_code, use_cython)` — `.ocp_solver`, `.solve(acados_sqp_mode)`, `.get_solution()`.
- `GPyTorchResidualModel.value_and_jacobian(y)` (auto finite-diff), `.record_datapoint(x,y,t)` (online).

### 10.4 장애물 회피
- B 단순화에서 비활성 (`side_term = 0`, `attenuation = 1`)
- IFAC 12-week plan: model-based 회피 분리 (RL 은 weight scheduling 만)

### 10.5 Multi-map BO
- 현재 single map (rand_a) 검증 중
- 향후: rand_a + rand_b + ... alternate mode → 일반화 검증 (anti-overfit)

---

## 11. 디버깅 체크리스트

### 11.1 차가 안 움직임
- [ ] CycloneDDS lo 있는가? (`grep lo ~/cyclonedds.xml`)
- [ ] `colcon build` 후 `source install/local_setup.bash` 했는가?
- [ ] auto_engage timer 5s (mpc_disable) / 40s (full) 기다렸는가?
- [ ] `/sim/initialpose` 가 spawn 위치와 맞는가?
- [ ] `ros2 topic echo /drive` 에서 cmd 발행되고 있는가?

### 11.2 차가 박힘 / stuck
- [ ] `cost_spike_thr_live` 너무 낮은가? (현재 8000; 1500 은 정상 운전도 fallback 시켜 #11 에서 상향)
- [ ] `mpc_corridor_half_width = 0.75` 가 track width 의 absolute half 보다 작은가?
- [ ] `inflation_factor = 0.0` 인가? (1.2 이면 corridor 음수)
- [ ] STUCK 복구 동작하는가? (`mpc_debug` 로그에서 'STUCK' / 'safe_reset' 검색)
- [ ] gym `in_collision` latch — `gym_bridge.py:288` 직접 확인

### 11.3 BO 가 Q=-1000 만 뱉음
- [ ] `n_resets > 2` 조건 — sim 중 reset 빈도 점검
- [ ] **`--map` 인자 전달되는가?** (2026-05-27 bug fix; `bo_sweep_turbo.py:236`)
- [ ] `crashed = True` flag 검사 — eval 의 collision detection
- [ ] `~/mpc_logs/*.csv` 직접 열어서 `car_x/car_y` vs centerline 거리 측정
- [ ] PP baseline json 이 존재하는가? (`find_pp_baseline_t_lb` 검색 경로)

### 11.3b GP ARD lengthscale 진단 (2026-05-27 #4)
매 BO iter 마다 출력되는 `GP ARD lengthscale: q_cte=X, q_lag=Y, ...` 해석:
| lengthscale | 의미 | 조치 |
|------------|------|------|
| < 1.0 | 해당 차원 sensitive — GP 가 잘 학습 중 | OK |
| 1.0 ~ 5.0 | moderate | OK |
| 5.0 ~ 10.0 | 약한 신호. 데이터 더 필요 | n_calls 증가 |
| > 10.0 | "정보 없음" — 차원 무시됨 | 해당 param 제거 / 또는 `--isotropic` 시도 |

여러 차원이 동시에 lengthscale > 10 이면 → 데이터 자체가 noise dominated. shake/cte penalty 가중치 재조정 검토.

### 11.4 cte 가 비정상적으로 큼
- [ ] eval 의 `--map` 인자 — 잘못된 centerline 매칭이 가장 흔한 원인
- [ ] centerline rolling 됐는가? (index 0 ≈ spawn 위치)
- [ ] s wraparound 처리 — KDTree 가 nearest 만 보므로 looping 시 정상

### 11.5 Solver / codegen 실패
- [ ] `dyn_tire_model` 변경 후 codegen 안 재실행?
- [ ] `N_horizon`, `dT` 변경 후 codegen 안 재실행?
- [ ] acados/c_generated_code/ 캐시 삭제 후 재실행
- [ ] HPIPM solve_ms 가 50ms 넘는 경우 — `cost_spike_thr_live` 트리거 → fallback

### 11.6 Launch "Killed"
- [ ] `run_bo.sh` 의 pkill 에서 `ros2.*launch` 패턴 들어가 있나? (들어가면 자기 죽임)
- [ ] `nonlinear_mpc` 패턴도 빠져야 함

### 11.7 NVIDIA driver mismatch
- 2026-05-22 발생 → reboot 으로 해결. 재발 시 `nvidia-smi` 출력 확인.

### 11.8 IFAC 환경 contamination
- 두 워크스페이스 동시 source 됐다면 `unset AMENT_PREFIX_PATH` 후 새 셸 열기

---

## 12. Final 시험맵 — dynamic 22s clean (2026-06-03)

시험맵 `final` (외곽루프+중앙island, 코너 R≈0.87m, **총폭 min 0.89m**·차폭 0.20m, L=76.5m, IQP raceline 추종)에서 **dynamic 모델로 22초 무접촉 7랩 지속** 달성. PP baseline(22.56s 단일랩 후 wedge)을 빠르고·깨끗하고·지속적으로 추월.

### 12.1 최종 결과 (검증, 7랩 재현)
- lap **22초 일관**, **접촉 0**, MINSTEP 0, steer RMS 0.15, v_avg 3.56, v_max 6.0(=max_speed cap).
- 커밋 `07fd33b`. config: `use_dynamic=true, track_source=raceline, max_speed=6, a_lat_safe=7.14, q_cte=0.15, q_lag=0.36, q_psi=0.79, q_v=0.53, q_p=2.70, q_drate=9.0, q_dv=0.35, D_apex=0.63`.
- (옵션) max_speed=8 → ~21초지만 접촉 2회. "박으면 안된다" 우선이면 max_speed=6 유지.

### 12.2 핵심 수정 (왜 이전엔 dynamic 이 안 됐나) — 커밋 `9439cf8`
1. **저속 블렌드/regularization 파라미터** (MINSTEP 주범). 형제 코드 `ifac_mpcc`(ICRA·타이트 검증)와 대조:
   - `dyn_v_eps` 0.5→**1.0** (ifac 주석: 0.5는 IPM step collapse, 1.0 안정)
   - `dyn_v_s` 0.1→**0.3** (블렌드 폭 너무 좁아 tanh 급전환 → stiff Hessian)
   - `dyn_v_b` 0.3→**0.5**
   - → final dynamic **MINSTEP 122 → 0**. (타이어모델은 이미 `linear`, Pacejka 아님)
2. **q_drate (steer-rate 페널티)** dynamic 엔 너무 약함 → steer 지그재그 RMS 0.36(kinematic 0.035의 10배) → wander → 16접촉/랩. `q_drate_scale_live` 0.6→**15** → RMS 0.15·접촉 16→2/랩·40s→28s. (ifac dynamic 도 q_d_rate=80 강함)
3. **per-stage corner-speed cap** (커밋 `af2df87`): 각 예측 stage 의 속도를 √(a_lat_safe/|κ(s_k)|) 로 제한. kinematic=ubu[v] 하드, dynamic=ubx[vx] generous ×1.6 (하드면 infeasible→MINSTEP). 직선=v_max, 코너만 grip속도.
4. **kinematic STUCK 자동복구** (커밋 `af2df87`): `/sim/initialpose` teleport-escape 가 dynamic 브랜치에만 있어 kinematic 모드서 벽 접촉 시 gym in_collision latch 영구 wedge. mode-agnostic 복구 블록 추가.

### 12.3 dynamic vs kinematic on final
- dynamic BO best: **22s / 0접촉** (a_lat_safe=7.14 — 실 grip 사용).
- kinematic BO best: 27s / 0.5접촉 (a_lat=3 바닥 — slip 미모델링이라 무접촉 위해 코너 극저속).
- **dynamic 의 slip 모델링이 고속 코너 정밀추종 → 실 grip(a_lat~7) 으로 빠르게.** 저속 타이트맵이라 kinematic 이 낫다는 가설은 **틀렸음**(블렌드 파라미터 버그였음).

### 12.4 BO objective 변경 (접촉수 우선) — 커밋 `9439cf8`
- `eval_run_quality.py`: `reset_ceiling` 2→**0** (무료 접촉 0, "박으면 안된다"). n_resets = 로그 `[stuck-recover] /initialpose` 카운트.
- `bo_sweep_turbo.py`: `a_lat_safe` LO 7→**3** (느린-코너 clean 탐색), `q_drate` HI 5→**30** (dynamic 강한 steer-rate 필요).

### 12.5 교훈 (리뷰 포인트)
- dynamic 이 안 될 때 **형제 코드(ifac_mpcc)와 파라미터 1:1 대조** 가 결정적이었음. "모델이 이 맵에 안 맞아" 성급한 단정 2회 → 실제론 내 블렌드 파라미터 버그.
- **단일 noisy run 결론 금지**: 이번 세션 초반 오판 다수(없는 폭문제 / 안움직인 run을 "0접촉" / 운 좋은 단일랩). fixed-window 다중랩 + s-rollover + steer-RMS 측정 필수.
- 다음: max_speed 8 + 재BO(직선 v↑ 반영), 실차 검증(dynamic blend 파라미터는 sim 기준 — 실차 grip GP residual 필요), multi-map 과적합 확인.
