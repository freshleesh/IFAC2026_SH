#!/usr/bin/env bash
# 백파일 리플레이 + RViz 딥디버깅.
# 사용:  ./bag_replay.sh <bag경로> [--rate 0.5] [--loop]
#
# 키 조작 (ros2 bag play 터미널에서):
#   Space = 일시정지/재개,  → = 일시정지 중 한 메시지씩 스텝
#
# 사고 직전 구간만 반복해 보려면:
#   ./bag_replay.sh <bag> --start-offset 42 --rate 0.25
set -u
BAG="${1:?사용법: bag_replay.sh <bag경로> [ros2 bag play 추가옵션...]}"
shift || true
RVIZ_CFG="$(dirname "$0")/bag_replay.rviz"

# RViz를 sim-time으로 (bag의 /clock 기준 시계)
rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!
trap 'kill $RVIZ_PID 2>/dev/null' EXIT

sleep 2
ros2 bag play "$BAG" --clock 100 "$@"
