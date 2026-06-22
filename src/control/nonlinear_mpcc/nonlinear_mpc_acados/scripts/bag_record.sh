#!/usr/bin/env bash
# 실차 rosbag 기록 — MPCC 딥디버깅용 토픽 셋.
# 차에서 실행:  ./bag_record.sh [출력이름]
# 기본 출력: ~/bags/mpcc_YYYYmmdd_HHMMSS
#
# 존재하지 않는 토픽(예: mac에선 nuc 드라이브 토픽)은 그냥 비어있게
# 기록되므로 양쪽 차 공용으로 써도 무해.
set -u
OUT="${1:-$HOME/bags/mpcc_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$OUT")"
echo "[bag_record] → $OUT  (Ctrl-C로 종료)"
exec ros2 bag record -o "$OUT" \
  /scan \
  /map \
  /tf /tf_static \
  /car_state/odom /vesc/odom \
  /global_waypoints \
  /external_obstacles \
  /mux/mpcc_active \
  /joy \
  /drive \
  /vesc/high_level/ackermann_cmd_mux/input/nav_1 \
  /mpc_trajectory /mpc/prediction /mpc_debug \
  /mpc/cost /mpc/solve_time /mpc/is_feasible /mpc/lap_count \
  /boundary_marker /center_path /left_path /right_path /reference_path
