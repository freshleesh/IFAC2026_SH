#!/bin/bash
# autoreg_speed_bo.sh — autoregressive speed-stepped BO.
#
# v=5 부터 시작해서 N 단계 BO 자동 진행:
#   각 단계 마다:
#     1) yaml max_speed / max_speed_p 갱신
#     2) PP baseline 측정 (이미 있으면 skip)
#     3) 이전 단계 best 를 warm-start x0 로 BO 실행
#     4) 결과 best 를 다음 단계 x0 로 stash
#
# Usage:
#   bash autoreg_speed_bo.sh "5 6 7" rand_a 50 8
#   bash autoreg_speed_bo.sh "5 6"    rand_a 30 8   # 짧게 sweep
#
# Args:
#   1 v_list (공백 구분, default "5 6 7")
#   2 map    (default rand_a)
#   3 n_calls per stage (default 50)
#   4 n_initial per stage (default 8)
#
# 출력: ~/bo_results/autoreg_v<v>_<ts>.log (단계별)
#       last_x0.txt — 마지막 best 저장 (다음 sweep 의 seed)

set -e

V_LIST=${1:-"5 6 7"}
MAP=${2:-rand_a}
N_CALLS=${3:-50}
N_INITIAL=${4:-8}

WS=$HOME/IFAC2026_SH
YAML=$WS/src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml
PP=$WS/src/nonlinear_mpc_acados/scripts/pp_baseline.py
BO_RESULTS=$HOME/bo_results
STASH=$BO_RESULTS/autoreg_last_x0.txt

# Seed warm-start 우선순위:
#   1) ~/bo_results/autoreg_last_x0.txt   (이전 autoreg sweep 결과)
#   2) ~/bo_results/bo_turbo_*.json 최신  (수동 BO 후 자동 적용)
#   3) BO Phase A best (v=4 hardcoded baseline)
LATEST_BO=$(ls -t $BO_RESULTS/bo_turbo_*.json 2>/dev/null | head -1)
if [ -f "$STASH" ]; then
    X0=$(cat "$STASH")
    echo "[autoreg] resuming from STASH: $X0"
elif [ -n "$LATEST_BO" ]; then
    X0=$(python3 -c "
import json
d = json.load(open('$LATEST_BO'))
keys = ['q_cte','q_lag','q_psi','q_v','q_p','q_drate','q_dv']
print(','.join(f\"{d['best_scales'][k]:.4f}\" for k in keys))
")
    echo "[autoreg] cold start from latest bo_turbo_*.json: $X0"
    echo "          (source: $LATEST_BO)"
else
    X0="3.665,0.397,0.597,0.313,4.282,3.533,1.675"
    echo "[autoreg] cold start from hardcoded v=4 BO Phase A best"
fi

cd "$WS"
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"

# v 별 N_horizon mapping (lookahead = N × dT × v ≈ 10-26m).
# 코너 라인 plan 위해 v 빠를수록 N 키움. dT=0.04 고정.
declare -A N_FOR_V
N_FOR_V[4]=40   # 6.4m  (v=4 baseline)
N_FOR_V[5]=50   # 10m
N_FOR_V[6]=60   # 14.4m
N_FOR_V[7]=70   # 19.6m
N_FOR_V[8]=80   # 25.6m

for V in $V_LIST; do
    N=${N_FOR_V[$V]:-40}    # default 40 if v not in map
    TS=$(date +%Y%m%d_%H%M%S)
    STAGE_LOG=$BO_RESULTS/autoreg_v${V}_${TS}.log
    echo "================================================================"
    echo "[autoreg v=$V N=$N] start at $TS, log → $STAGE_LOG"
    echo "================================================================"

    # 1) yaml 업데이트 (max_speed + N_horizon → codegen 자동 재실행)
    sed -i "s/^    max_speed:.*/    max_speed: ${V}.0/" "$YAML"
    sed -i "s/^    max_speed_p:.*/    max_speed_p: ${V}.0/" "$YAML"
    sed -i "s/^    N_horizon:.*/    N_horizon: ${N}                  # autoreg v=${V}: lookahead $(python3 -c "print(${N}*0.04*${V})") m/" "$YAML"
    echo "[autoreg v=$V] yaml: max_speed=$V, N_horizon=$N"

    # 2) PP baseline (없으면 측정)
    if ls $BO_RESULTS/pp_baseline_v${V}.0_*.json >/dev/null 2>&1; then
        EXISTING=$(ls -t $BO_RESULTS/pp_baseline_v${V}.0_*.json | head -1)
        echo "[autoreg v=$V] PP baseline 이미 존재: $EXISTING"
    else
        echo "[autoreg v=$V] measuring PP baseline..."
        python3 "$PP" --v ${V}.0 --map "$MAP" --n_laps 3 2>&1 | tee -a "$STAGE_LOG"
        # PP 끝나면 잠시 대기 (sim 완전 종료)
        sleep 5
        pkill -9 -f gym_bridge 2>/dev/null || true
        pkill -9 -f mpc_node 2>/dev/null || true
        sleep 2
    fi

    # 3) BO 실행
    echo "[autoreg v=$V] BO launch (n_calls=$N_CALLS, x0=$X0)"
    ros2 launch nonlinear_mpc_acados bo_train.launch.py \
        map:=$MAP \
        n_calls:=$N_CALLS \
        n_initial:=$N_INITIAL \
        n_laps:=3 \
        x0:="$X0" 2>&1 | tee -a "$STAGE_LOG"

    # 4) 끝나면 best 추출 → 다음 단계 x0 로 stash
    LATEST_JSON=$(ls -t $BO_RESULTS/bo_turbo_*.json 2>/dev/null | head -1)
    if [ -z "$LATEST_JSON" ]; then
        echo "[autoreg v=$V] ERROR: no bo_turbo_*.json found. Stopping."
        exit 1
    fi
    NEW_X0=$(python3 -c "
import json
d = json.load(open('$LATEST_JSON'))
keys = ['q_cte','q_lag','q_psi','q_v','q_p','q_drate','q_dv']
print(','.join(f'{d[\"best_scales\"][k]:.4f}' for k in keys))
")
    echo "$NEW_X0" > "$STASH"
    echo "[autoreg v=$V] DONE. best Q=$(python3 -c "import json; print(json.load(open('$LATEST_JSON'))['best_Q'])")"
    echo "[autoreg v=$V] new x0 stashed: $NEW_X0"

    X0=$NEW_X0

    # 다음 단계 전에 cleanup
    pkill -9 -f bo_sweep_turbo 2>/dev/null || true
    pkill -9 -f gym_bridge 2>/dev/null || true
    pkill -9 -f mpc_node 2>/dev/null || true
    pkill -9 -f mpc_debug 2>/dev/null || true
    pkill -9 -f simple_mux 2>/dev/null || true
    pkill -9 -f rviz2 2>/dev/null || true
    pkill -9 -f frenet 2>/dev/null || true
    pkill -9 -f state_machine 2>/dev/null || true
    pkill -9 -f ftg_fallback 2>/dev/null || true
    pkill -9 -f pp_fallback 2>/dev/null || true
    pkill -9 -f global_republisher 2>/dev/null || true
    pkill -9 -f fake_topic 2>/dev/null || true
    sleep 5
done

echo "================================================================"
echo "[autoreg] ALL STAGES DONE. final best x0 in $STASH"
cat "$STASH"
echo "================================================================"
