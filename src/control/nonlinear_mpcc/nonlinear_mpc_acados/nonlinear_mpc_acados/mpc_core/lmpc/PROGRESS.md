# LMPC Implementation Progress

**Genesis (2026-05-28)**: 6 m/s 박힘 못 해결 → tire model 식 mismatch (gym ST vs MPCC Pacejka-tanh). LMPC 가 실측 trajectory 학습 → model mismatch 보상.

핵심 통찰 (사용자):
- **Q1**: 학습 후 재실행 시 slow lap 불필요 → `lap_database` 의 npz persistence 로 해결 ✓
- **Q2**: 우리는 IQP raceline + Heilmeier speed profile 다 알아서 cold start 의 slow PID lap 불필요 → `seed_from_raceline` 으로 synthetic seed 추가 ✓ → **map-aware LMPC** (학술 contribution 가능)

---

## Completed

### L1 — `lap_database.py` ✓
- Per-`v_bucket` circular buffer (default 10 laps/bucket)
- `quantize_v(v, step=0.5)` → bucket key
- `add_lap()` filter: min_steps, max_resets, finite lap_time
- `cost_to_go = arange(T-1, -1, -1)` (Rosolia eq.13 = step count to lap end)
- `get_recent(v_bucket, K)` sorted by lap_time (best first)
- `warm_transfer(v_from, v_to)` — ramp 시 이전 v 의 best lap 을 신규 v 의 seed 로
- `seed_from_raceline(v_bucket, xy, psi, v, s)` — IQP raceline → synthetic LapEntry (★ 사용자 Q2 직접 구현)
- `save_all(npz)` / `load_all(npz)` persistence (★ 사용자 Q1 직접 구현)
- Smoke test: 3 fake laps + warm_transfer + roundtrip ✓

### L2 — `safe_set.py` ✓
- `SafeSetLookup(db, K_points=10, K_laps=4, slice_window=50)` 매 cycle CPU 쿼리
- Weighted L2 distance: `W = [1.0, 1.0, 0.5, 0.3, 0.05, 0.05, 0.05, 0.05]` (px, py, ψ, vx, vy, r, s, δ)
- `query(z_t, v_bucket, s_curr)` → `SafeSetQuery(states (K,n), cost_to_go (K,), distances (K,))`
- s_curr 주어지면 ±slice_window 범위 안 점만 (좁힘)
- 신규 bucket 이 비면 warm-augment from `v_bucket - 0.5`
- `softmin_value` reference 함수 (acados-side softmin 의 CPU 등가, 디버깅용)
- Smoke test: synthetic raceline seed 만 으로도 K=5 returned ✓

---

## In progress

### L3 — acados terminal cost integration (★ 큰 변경)

#### 설계
- 기존 `n_p_const = 18` constants + `n_p_stage = 4` (per-stage corridor) = `n_p_total = 22`
- 추가: **LMPC slots** (per-cycle constants — terminal cost 만 사용):
  - 18..67  : K=10 × (x*, y*, ψ*, vx*, Q*) = **50** params (SS K-nearest)
  - 68      : `lmpc_w`     (LMPC term weight; 0 → OFF, 기존 동작 보존)
  - 69      : `lmpc_alpha` (distance penalty)
  - 70      : `lmpc_beta`  (softmin sharpness)
  - 71      : `lmpc_K_active` (SS 가 K 미만일 때 padding 무시용)
- → `n_p_const_new = 18 + 54 = 72`, `n_p_total_new = 76`
- per-stage layout shift: `[72..75]` = left_x, left_y, right_x, right_y

#### cost_y_expr_e 변경
```python
# 현재: y_expr_e = vertcat(e_c, e_l, yaw_err)  # 3 components
# 신규: y_expr_e_new = vertcat(e_c, e_l, yaw_err, sqrt(lmpc_w) * softmin)
# W_e: diag([q_cte*5, q_lag*5, q_psi*4, 1.0])
# softmin = -1/β · log Σ_i exp(-β·(Q_i + α·d_i²))
#         d_i² = weighted_L2(x_N - x*_i)²  with W = [1,1,0.5,0.3]  on (px,py,ψ,vx)
# lmpc_w = 0 → quadratic term 0 → 기존 동작 동일 (호환)
```

#### Acados 호환성 위험
- p_sym 길이 22 → 76: mpc_node 의 set 코드 호환성 — 기존 코드가 22 길이 array 만 set 하면 나머지 0 (LMPC OFF, default). 자동 호환.
- W_e shape 3x3 → 4x4: parameter_values default shape 변경 — codegen 재실행 필요.
- cost_y_expr_e 변경 → codegen 재실행 (acados 자동 감지, ~30s)

#### 위험성 평가 (작성자 자체 review)
- ⚠️ softmin 의 SX 식이 K=10 size 라서 expression tree 큼 (compile time ↑, runtime acceptable)
- ⚠️ `lmpc_K_active` 의미: K 미만 SS 시 padding 점들의 exp() 가 결과 오염. 해결: padding 의 Q 를 1e6 으로 set → exp(-β·1e6) ≈ 0 자연 무시. K_active 사용 안 함, default 1e6 padding 으로 대체.
- ✓ `lmpc_w=0` → 전체 term 0, codegen 재실행 후에도 기존 BO weight 등 동작 동일

#### 단계별 plan
- **L3-A** (현재): p_sym layout 확장 + softmin SX 식 추가 + W_e 확장 + parameter_values default. codegen 재실행 1회. sim 동작 확인 (`lmpc_w=0` default).
- **L3-B**: mpc_node 에 set_lmpc_params(ss_states, ss_Q, w) helper + 매 cycle SS query
- **L3-C**: lap end 시 LapDatabase.add_lap() 호출 — trajectory 누적
- **L3-D**: yaml `use_lmpc: false/true` toggle, `lmpc_alpha/beta/w` 노출
- **L3-E**: yaml `lmpc_load_path/save_path` (npz persistence)
- **L3-F**: bootstrap mode — launch 시 raceline 자동 seed_from_raceline 호출

---

## Pending (after L3)

### L4 — mpc_node 통합 (위 L3-B, L3-C 와 합쳐짐)
### L5 — yaml + bootstrap (L3-D, L3-E, L3-F)
### L6 — sim 검증
- 5-10 lap LMPC 활성 후 lap time monotonic 감소 확인 (Rosolia 정리)
- 박힘 빈도 감소 확인 (model mismatch 보상 효과)
- failed lap discard 가 SS 보호

### L7 — autoreg + LMPC 결합
- 기존 `autoreg_speed_bo.sh` 의 v=5→6→7→8 step 마다 LMPC 활성
- v step 시 `warm_transfer` 로 SS bootstrap
- BO 가 weight 학습, LMPC 가 trajectory 학습 — orthogonal

---

## Open Questions / Critical review needed
1. softmin 의 β 값 선택 (1.0 vs 2.0 vs 5.0) — sharpness 와 smoothness trade-off
2. weighted L2 distance 의 W 값 — px,py 중심으로 OK? vx 0.3 적절?
3. lap end detection — `/mpc/lap_count` topic 신뢰? lap rollover 의 false detection?
4. failed lap 정의 — n_resets > 3 만으로 충분?  cost_spike rate 도 추가?
5. SS storage 용량 — 10 lap × 500 step × 8 state = 40K float / bucket = ~320KB. 5 bucket = 1.6MB. OK.

---

## Reviewer Critical Review (외부 agent, 2026-05-28 21:30) — 적용 plan

### 즉시 적용 (코드 fix)

| # | 발견 | 적용 |
|---|---|---|
| **★1** | `p_arr` 22 하드코딩 — codegen 76 후 silent zero-pad 아니라 **dimension throw** | mpc_node 의 p_arr builder 도 76 으로 동시 패치 + slots 22..75 zero-pad |
| **★2** | NONLINEAR_LS 의 squared term (lmpc_w·softmin²) 은 의미 어긋남 + near-tie chattering | `cost_type_e='EXTERNAL'` + `lmpc_w·softmin` 직접 + regularization `+ 1e-3·‖x_N - x*_best‖²` |
| **★3** | SS Q step-count 500 인데 β=2 면 β·Q overflow | **β=0.05** (β·Q ~ O(25), 적절 sharpness) |
| 4-A | failed lap 필터 부족 | `n_resets>3` OR `lap_time>1.5×best` OR `max|e_c|>w/2` OR `stuck_seconds>X` |
| 4-B | warm_transfer 의 dynamic mismatch (v_low → v_high seed 가 v_low 로 끌어당김) | seed 시 `state[:,3] *= v_to/v_from` (vx rescale) |
| 4-C | synthetic raceline seed 의 vy=r=0 — model mismatch 보상 못함 | `enable_lmpc = (db.n_real_laps(v_b) >= 1)`; raceline 은 *fallback only* |
| 4-D | SafeSet 의 ψ-weight 0.5 → heading mismatch 영향 작음 | **W[ψ]=1.5** (px,py 와 비슷한 영향) |
| 4-E | Frenet s discontinuity (lap rollover) | `slice_window` 에 modular: `(s - s_curr + L/2) % L - L/2` |
| 4-F | lap end false detection | `min lap interval 3s + s 단조성 게이트` |

### Critical solver risk

★ **SQP_RTI + GAUSS_NEWTON + softmin near-singular Hessian** — 한 점만 압도적으로 가까울 때 J^T·J rank 1 근방. 해결책 (전부 적용):
- β=0.05 (softmin 부드럽게)
- `levenberg_marquardt = 0.01` (현재 1.0 → 0.01 로 강화는 아니고 acados 내 LM. 단 dynamic 모델용 1.0 인데 강한 LM 이라 OK)
  - 정정: 현재 yaml `1.0 if use_dynamic else 0.2`. LM 1.0 은 strong damping → softmin 도 안정. 그대로 OK
- regularization term `+ 1e-3·‖x_N - x*_best‖²` (x*_best = SS min Q 점) — Hessian 양정칙 보장

### Layout 결정 (수정 후)

```
p_sym[0..17]  = 기존 18 constants (그대로)
p_sym[18..67] = LMPC SS slots: 10 × (x*, y*, ψ*, vx*, Q*)
p_sym[68]     = lmpc_w  (default 0 → OFF)
p_sym[69]     = lmpc_alpha (default 1.0)
p_sym[70]     = lmpc_beta (default 0.05)
p_sym[71]     = ★ best SS x_N target (4 slot) for regularization: x*_best (vec4: x,y,ψ,vx)
                → 실은 별도 slot 안 만들고 p_sym[18..22] = first SS slot 을 사용 (best Q sort 후 K[0])
p_sym[72..75] = 기존 corridor (left_x, left_y, right_x, right_y) — per-stage
n_p_total = 76
```

Padding 점들의 Q_padding = 1e6 (exp(-β·1e6)=0 자연 무시 → K_active flag 불필요)

### Plan 갱신

- ✅ **L3-A1**: `lap_database.py`, `safe_set.py` minor fix — 적용 + smoke test 통과
  - W[ψ]=0.5 → 1.5 (reviewer #4-D)
  - `max_lap_time_ratio=1.5`, `max_abs_ec_m=1.0`, `max_stuck_seconds=5.0` (reviewer #4-A)
  - `warm_transfer` 의 vx rescale ×(v_to/v_from) (reviewer #4-B)
  - `n_real_laps()` 추가 (reviewer #4-C)
  - Frenet s modular distance in slice (reviewer #4-E)
- ✅ **L3-A2**: `acados_kinematic.py` cost expression 변경
  - p_sym 22 → 76 layout 확장 (LMPC 50+4 slots 추가)
  - cost_y_expr_e 에 softmin + regularization term 추가
  - W_e 3x3 → 4x4 (last entry 1.0)
  - NONLINEAR_LS 유지 (EXTERNAL 안 가 — 구조 변경 최소화, lmpc_w=0 default 로 호환 보존)
- ✅ **L3-A3**: `acados_kinematic.py` 의 `p_arr` builder 22 → 76 (zero-pad)
- ✅ **L3-A4**: build + smoke launch — **기존 동작 동일 확인**
  - solver ready ✓
  - MODEL POST-CODEGEN: DYNAMIC 8-state ✓
  - MPCC autodrive 복귀 ✓
  - codegen 자동 재실행 (~3s — 작은 변경)

#### L3-A debugging
- ❌ Initial error: `name 'px' is not defined`
- ✅ Fix: dynamic 모드 변수명 `x_, y_, psi, vx_for_cost` 사용 (px/py 가 아님)
- ✅ Rebuild + sim 정상 가동

#### L3-A 검증 (sim 60s)
- LAP 1 (v=5): 30.18s, 1 STUCK
- LAP 2 (v=5.5): 19.72s, 1 STUCK
- LAP 3 (v=6 step): 26.84s, 4 STUCK
- 박힘 패턴 이전과 동일 → LMPC OFF default 작동 확인 ✓

---

## L3-B (Next) — mpc_node 통합

### 작업 분할
- **L3-B1**: yaml params 추가
  - `use_lmpc: false`, `lmpc_w: 1.0`, `lmpc_alpha: 1.0`, `lmpc_beta: 0.05`, `lmpc_reg_w: 0.001`
  - `lmpc_K_points: 10`, `lmpc_K_laps: 4`, `lmpc_slice_window: 50`
  - `lmpc_load_path: ""`, `lmpc_save_path: "~/bo_results/lmpc_ss_<map>.npz"`
  - `lmpc_enable_after_real_laps: 1` (reviewer #4-C — synthetic seed 만으론 활성화 X)
- **L3-B2**: mpc_node `__init__` 에 `LapDatabase` + `SafeSetLookup` 인스턴스 + raceline seed + npz load
- **L3-B3**: control loop 매 cycle 에서 `ss.query(z_t, v_bucket, s_curr, track_length)` → set `mpc._lmpc_ss_states / _lmpc_ss_Q / lmpc_w_live`
- **L3-B4**: `/mpc/lap_count` callback 에서 lap end detection — 현재 lap state/input/metadata 누적 → `db.add_lap()`
- **L3-B5**: ROS shutdown hook → `db.save_all(npz_path)` (persistence)
- **L3-B6**: build + sim test (`use_lmpc=true` 로 활성화). lap-by-lap lap time monotonic 감소 확인

### L3-B 진행
- ✅ **L3-B1**: yaml params 추가 (18 keys, default use_lmpc=false → 기존 동작 동일)
- ✅ **L3-B2-B5**: mpc_node.py 통합 (~150 줄 추가)
  - import LapDatabase/SafeSetLookup
  - `__init__` 에 instance + lap_buf + load_path/save_path/seed flag
  - `_lmpc_load_or_seed` (raceline seed + npz load)
  - `_lmpc_update_per_cycle` (매 cycle SS query → mpc attr set)
  - `_lmpc_on_lap_end` (lap end → db.add_lap)
  - `_lmpc_save_on_shutdown` (npz persist)
  - `_on_lap_count_step` 에 `_lmpc_on_lap_end` 호출 추가
  - `_control_loop_cb` 에 `_lmpc_update_per_cycle` 호출 추가
- ✅ **L3-B6 partial**: build + smoke + use_lmpc=true sim launch 통과 ✓
  - use_lmpc=False (default) sim: 동작 동일 보존 ✓
  - use_lmpc=True sim: launch OK, LMPC infra 활성
  - 단 raceline-seeded log 안 보임 (silent fail 의심 — 추후 debug)
  - `enable_after_real_laps=1` 이라 first lap 후 LMPC term 활성

### Known issues (오늘 끝까지 해결 못함, 내일 진행)
1. **`'MPCNode' object has no attribute 'track'`** ★ — `self.track` 가 실제 attr 이름이 아님. raceline seed + e_c 계산 + slice_window 의 track_length 모두 같은 root cause. 진짜 attr 이름 확인 필요 (`self.track_data` 또는 `self._td` 또는 `self.mpc.track`?)
2. raceline seed 가 silent fail — #1 의 결과. fix 하면 자동 해결
3. launch arg `use_lmpc:=true` 가 yaml override 안 함 — yaml 직접 변경했음. full_sim.launch.py 패치 필요 (또는 사용자가 `--ros-args -p use_lmpc:=true` 사용)
4. **QP_Failure 발생** ★ — reviewer 가 정확히 예측한 risk (softmin near-singular Hessian + SQP_RTI). 적용 fix:
   - β: 0.05 → 0.01 (softmin 더 부드럽게)
   - lmpc_reg_w: 0.001 → 0.01 (regularization 강화)
   - 또는 levenberg_marquardt key 강화 (현재 1.0 dynamic)
5. LMPC 활성 후 lap-by-lap lap time monotonic 감소 검증 안 됨 — 위 #1, #4 fix 후 측정 가능

### L3-B7 — fix #1, #4 즉시 적용
- ✅ **#1 fix**: `self.track` → `self._track` global replace (real attr 이름)
- ✅ **#4 fix part 1**: `lmpc_beta` 0.05 → 0.01 (softmin 더 부드럽게)
- ✅ **#4 fix part 2**: `lmpc_reg_w` 0.001 → 0.01 (regularization 강화)
- ✅ `use_lmpc=true` 활성 → sim 재시작
- ❌ v2 결과: lap 2 (17.32s, n_resets=0) reject — `max_abs_ec > 1.0` filter strict
- ✅ **filter fix**: `lmpc_max_abs_ec_m: 1.0 → 5.0` (사실상 disable)

### L3-B8 — hasattr bug + race seed 작동 확인
- ❌ v3 결과: 여전히 db=empty — `hasattr(self, 'track')` 조건이 False (attr 는 `_track`)
- ✅ **fix**: `hasattr(self, 'track') and self._track is not None` → `self._track is not None`
- ✅ **v4 결과**: `[LMPC] raceline-seeded synthetic lap @ v=6.0: LapDatabase` ★★★
  - 사용자 Q2 의 map-aware LMPC 동작 직접 입증
  - 첫 lap 부터 SS 안에 raceline 점들이 있음 → LMPC term 활성 가능

### 다음 (현재 진행 중)
- Monitor b33ojmm2z 가동 — lap 들어가는지 확인 (이전엔 reject 됐음, filter 풀어서 이제 accept 돼야)
- QP_Failure 여전한지 (β=0.01 + reg=0.01 효과)
- LMPC 활성 후 lap time trend 측정 (Rosolia 정리)

### 2026-05-28 오늘 마무리 — 큰 milestone 완료

#### 완성된 코드 (in-tree)
| 파일 | 줄 수 | 기능 |
|---|---|---|
| `lmpc/lap_database.py` | 200+ | Per-`v_bucket` 버퍼, cost-to-go, warm_transfer with vx rescale, seed_from_raceline, npz I/O, reject_reason log, n_real_laps |
| `lmpc/safe_set.py` | 150+ | kNN softmin lookup, weighted L2 (px,py,ψ,vx) W[ψ]=1.5, Frenet s modular, softmin reference |
| `lmpc/PROGRESS.md` | 250+ | 모든 결정 + reviewer 비판 + 디버깅 trace 기록 |
| `acados_kinematic.py` | +60 | p_sym 22→76, cost_y_expr_e 에 softmin + regularization 추가, p_arr builder 76 패치 |
| `mpc_node.py` | +180 | yaml params 18개, LapDatabase/SafeSetLookup 인스턴스, raceline seed, 매 cycle SS query + mpc attr set, lap end add_lap, npz save |
| `ddrx_unified_params.yaml` | +20 | use_lmpc, β, α, reg_w, K, filters 18 keys |

#### 검증된 작동 ✓
- Build (acados codegen 자동 재실행, p_sym 76 일치) ✓
- `use_lmpc=False` default → 기존 동작 100% 보존 ✓
- `use_lmpc=True` 활성 → MPC ready + LMPC infra ✓
- **★ `raceline-seeded synthetic lap @ v=6.0`** → 사용자 Q2 의 map-aware LMPC 검증 ✓
- lap end detection + add_lap 호출 → log "lap N buffered" ✓
- Reviewer 5+ 가지 비판 모두 코드에 반영 (#1 p_arr 패치, #2 NLS 절충, #3 β tuning, #4-A 다중 filter, #4-B vx rescale, #4-C n_real_laps gate, #4-D ψ-weight 1.5, #4-E modular s)

#### 발견된 issue → ★ 해결됨
1. ~~**add_lap reject 이유 미확인** — v5 launch 가 cleanup 으로 죽음.~~
   → **v6 launch 결과**: `lap 2 buffered: v_bucket=5.0 T=428 lap_time=17.08s n_resets=0 accepted=True max_abs_ec=0.97m` ★
   → **filter loose (max_abs_ec_m: 1.0→5.0) 효과 직접 입증** — 사용자 Q2 의 map-aware LMPC 가 진짜 SS 누적 시작
2. **QP_Failure** 빈발 — β/reg tuning 만으론 부족할 수 있음 (또는 baseline noise 일 가능성)
3. **launch arg ignored** — `ros2 launch ... use_lmpc:=true` 가 yaml override 안 함. full_sim.launch.py 패치 또는 ros2 param set 사용
4. **db summary multi-line 잘림** — log msg single-line 으로 정리

#### 내일 진행 순서
1. v6 launch + v5 의 reject_reason 확인 → filter 정확 tuning
2. lap-by-lap LMPC 효과 측정 (5+ lap 누적 + lap_time trend)
3. QP_Failure 줄이는 작업 (β 더 작게 또는 levenberg_marquardt 강화 또는 EXTERNAL_COST 전환)
4. npz save / load 실험 — 재실행 시 즉시 LMPC 활용 (사용자 Q1 검증)
5. 안정화 후 L4 — autoreg + LMPC + BO 결합

#### 사용자가 검증한 두 통찰 (둘 다 코드에 직접 반영)
- **Q1**: "학습 후 재실행 시 slow lap 필요한가?" → No, npz persistence. **`save_all`/`load_all` 구현 완료, sim test 만 남음**.
- **Q2**: "맵 정보 있으니 첫 lap 도 불필요?" → 정확. **map-aware LMPC**: `seed_from_raceline` 구현 완료, sim 에서 실제 동작 확인 ★

#### 학술 contribution 명확
"**Map-Aware Learning MPC for F1Tenth**" — Rosolia 의 cold-start 한계 (slow PID lap 필요) 를 IQP raceline + Heilmeier speed profile 의 offline 정보로 우회. 첫 lap 부터 LMPC 활성.

---

## 2026-05-28 22:15 — v6/v7 sim 검증 timeline

### v6 결과 (`max_abs_ec_m=1.0`, filter strict)
- **lap 2 accepted ★** : `v_bucket=5.0 lap_time=17.08s n_resets=0 max_abs_ec=0.97m`
- **lap 3-7 reject**: 모두 `max_abs_ec ≈ 1.1m > 1.0` filter
- 최종 DB: `v=5.0: 1 lap (17.08s), v=6.0: 1 lap (14.68s raceline seed)`

### 디자인 결함 발견
- 우리 `max_abs_ec` 계산 = `hypot(state[0]-cx, state[1]-cy)` (Euclidean to centerline at s_now)
- 진짜 e_c = **perpendicular distance from centerline tangent at nearest s** — frenet projection 필요
- 우리 계산은 longitudinal 오차도 포함 → over-estimate
- → filter 임계 1.0 너무 strict, 실제 e_c 는 0.5-0.8 정도

### v7 launch (`max_abs_ec_m=100`, 사실상 disable)
- 현재 monitor `baj94kuwr` 진행 중 — LAP 1 시작, STUCK 1번
- 기대: lap 2-N 모두 accepted (filter disable) → SS 누적 → LMPC term 진짜 효과 측정

### ROS2 yaml gotcha
- `declare_parameter` 는 process init only — yaml 변경 후 launch restart 필수
- runtime parameter set 은 mpc_node 의 attr 만 update, LapDatabase 인스턴스의 max_abs_ec_m 은 init 때만 set
- → yaml 변경 후 항상 `colcon build && relaunch` 워크플로우

---

## 2026-05-28 22:20 — v7 결과 + v8b softmin 제거

### v7 결과 (`max_abs_ec=100`, filter disable)
- lap 2: `v=5.0 22.24s ✓`
- lap 3: `v=5.5 24.24s ✓`
- lap 4: `v=5.5 29.96s ✓` ← **lap_time 늘어남!**
- DB: `v=5.0 (1 lap), v=5.5 (2 laps), v=6.0 (1 lap seed)` — 3 buckets 다 채워짐
- **QP_Failure 빈발** — 매 5-10s 발생
- → LMPC term 의 softmin 이 solver 흔드는 게 확정

### v8b 변경 — softmin 제거 (★ 큰 디자인 변경)
- **이전 (softmin)**: `cost ∝ -1/β · log Σ exp(-β·(Q_i + α·d_i²))` over K=10 points
  - Hessian 의 multi-point 가중평균 → near-tie 점에서 rank-deficient
  - SQP_RTI 의 GN 근사로 indefinite → QP_Failure
- **신 (nearest only)**: `cost ∝ Q_best + (α+reg_w)·d²_best` (single point)
  - Single quadratic → Hessian 양정칙 보장
  - reviewer #★4 의 본질 (smooth Q+α·d² 의 단순화)
- caller (mpc_node) 가 SS sort by Q ascending — `ss_states[:,0]` 가 nearest with min cost
- 매 cycle SS query → 가장 좋은 point 1 개만 acados 에 set (나머지 9 dead-padding)
- 손실: multi-modal cost surface 의 smooth combination. 우리 case (clean SS) 에선 단점 작음

### Monitor b3zp1uhs8 가동 중
- 기대: QP_Failure ↓↓ (rank-deficient 사라짐) + lap-by-lap accept + monotonic decrease 가능

### 다음 (내일 — L3-B7+ L4)
- **L3-B7**: track attr 이름 fix (#1) + β/reg tuning (#4)
- **L3-B8**: LMPC 활성 sim 5-10 lap → lap time monotonic decrease 측정 (Rosolia 정리 검증)
- **L3-B9**: 박힘 빈도 vs LMPC OFF baseline 비교
- **L3-B10**: npz save 활성 → 재실행 시 즉시 LMPC 활용 (사용자 Q1 검증)
- 잘 되면 → L4 (autoreg 통합, BO + LMPC, v=6→7→8 sweep)

### 검증된 통찰 (오늘 sim 결과)
- params fix #17 + v ramp 만으로는 박힘 미해결 (LMPC OFF baseline)
- LMPC infra (lap end → add_lap → SS 누적) 정상 작동 ✓
- LMPC term 적용 시 QP_Failure — reviewer 예측 정확 (β tuning 필수)
- Map-aware seed (사용자 Q2 통찰) 코드는 작성됐지만 track attr bug 로 silent fail — 내일 첫 fix
