# nonlinear_mpc_acados

VPMPCC + EVO-MPCC LTM acados controller, ported from IFAC2026_SH.

`controller/PP.py` 와 같은 역할이지만 nonlinear MPC (acados SQP_RTI + HPIPM).
센터라인 추종이 아닌 **progress maximization** + corridor 안에서 자유 plan.
Sim BO Phase B 의 v=5 best weights 미리 적용 (`config/ddrx_unified_params.yaml`).

## Quick Start (그들 ppc.launch.xml 와 동일 패턴)

### Sim
```bash
# Terminal 1: low_level (sim mode)
ros2 launch stack_master low_level.launch.xml sim:=true map:=<your_map>

# Terminal 2: MPCC
ros2 launch nonlinear_mpc_acados mpcc.launch.xml map:=<your_map> sim:=true
```

### Real Car
```bash
# Terminal 1: middle_level (cartographer localization + EKF + LiDAR + VESC)
ros2 launch stack_master middle_level.launch.xml map:=<your_map>

# Terminal 2: MPCC
ros2 launch nonlinear_mpc_acados mpcc.launch.xml map:=<your_map>
#   sim arg omitted → default false → /vesc/odom auto-remaps to /car_state/odom
#   enable_sim_reset 자동 false → /sim/initialpose publish skip (real car 안전)

# Terminal 3 (optional): record
ros2 bag record -o lap_mpcc /vesc/odom /car_state/odom \
    /vesc/high_level/ackermann_cmd /global_waypoints /tf
```

### 안전
- `config/ddrx_unified_params.yaml` 의 `max_speed: 4.0` 부터 시작
- 첫 lap: joy HUMAN 으로 천천히 → AUTO 전환 → LB 버튼 즉시 복귀 준비
- `enable_sim_reset: false` (real) → 박힘 감지해도 sim reset 안 함 (실차엔 무의미)

## Install (acados 차에 없으면)

```bash
# requirements.txt 참고
# 핵심: acados manual build, l4acados PYTHONPATH (optional GP).
```

## 동작 흐름
```
waypoint_publisher (planner pkg)
   → /global_waypoints (WpntArray, latched)
   → /centerline_waypoints/wpnts (WpntArray, latched)
                ↓
            mpc_node (this pkg)
   ←──── /vesc/odom (또는 /car_state/odom remap on real)
                ↓
   → /vesc/high_level/ackermann_cmd (AckermannDriveStamped)
   → /mpc/* topics (debug / viz)
                ↓
       Simple_mux + VESC → motor + servo
```

## yaml 핵심 param (`config/ddrx_unified_params.yaml`)

| param | 값 | 의미 |
|-------|-----|------|
| max_speed | 4.0 | upper cap (실차 안전) |
| N_horizon | 40 | 1.6s lookahead @ v=4 |
| dT | 0.04 | 25 Hz control |
| dyn_tire_model | tanh | F_y = μ·D·F_z·tanh(B·α) |
| track_source | centerline | 'raceline' 도 가능 (IQP line 추종) |
| a_lat_safe_live | 15.0 | corner cap √(15/κ) ≈ 5.9 @ κ=0.43 |
| q_*_scale_live | (v=5 best) | sim BO 결과 |
| use_gp_residual | false | 활성화 시 PYTHONPATH=$HOME/l4acados/src |
| enable_sim_reset | (launch arg) | 실차 false (안전) |

## 실차 BO 사이클 (권장)
1. 첫 launch 그대로 (v=4, sim weight) → 완주 확인
2. lap_time 기록 → `pp_baseline.py` 와 비교
3. **실차 BO** (`scripts/autoreg_speed_bo.sh`) — 3-5 iter 정도, 안전 cap v=4 고정
4. 결과 weights → yaml 적용 → 다시 검증
5. (선택) GP residual data 수집 → `extract_residuals.py` → `train_gp_residual.py`

## 참고
- 원본: `~/IFAC2026_SH/src/control/nonlinear_mpc_acados/` (IFAC paper 코드)
- 자세한 알고리즘: `nonlinear_mpc_acados/mpc_core/PIPELINE.md`
- BO + GP 학습 plan: 동 PIPELINE.md 의 § 5, § 10.3
