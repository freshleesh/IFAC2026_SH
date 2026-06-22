#!/usr/bin/env python3
"""pp_baseline.py — global line (raceline) 추종 PP 의 lap_time baseline 측정.

BO 시작 전 한 번 호출. MPCC 가 능가해야 하는 reference time.

흐름:
1. sim launch (mpc_disable=true + pp_wpnts_topic=/global_waypoints + pp_max_speed=v)
2. /car_state/odom_frenet 의 s (raceline arc length) 모니터
3. s rollover 감지 → lap_time 측정
4. 3 lap 완료 또는 wall_timeout → kill sim → best lap_time JSON 저장

Usage:
  python3 pp_baseline.py --v 4.0 --map f
  → ~/bo_results/pp_baseline_v4.0_<timestamp>.json
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


BO_DIR = Path(os.path.expanduser('~/bo_results'))
BO_DIR.mkdir(parents=True, exist_ok=True)


class LapTimer(Node):
    """odom_frenet 의 s rollover 감지 → lap_time 누적."""

    def __init__(self, max_laps: int = 3):
        super().__init__('pp_baseline_lap_timer')
        self.max_laps = max_laps
        self.last_s = None
        self.lap_count = 0
        self.lap_times: list[float] = []
        self.lap_start_t = time.time()
        self.done = False
        self.create_subscription(Odometry, '/car_state/odom_frenet', self._cb, 10)
        self.get_logger().info(
            f'[pp_baseline] LapTimer up. target n_laps={max_laps}, listening /car_state/odom_frenet')

    def _cb(self, msg: Odometry):
        s = float(msg.pose.pose.position.x)
        if self.last_s is not None and (self.last_s - s) > 30.0:
            # s rollover (예: 76 → 0) → lap 완료
            now = time.time()
            lap_t = now - self.lap_start_t
            self.lap_start_t = now
            self.lap_count += 1
            self.lap_times.append(lap_t)
            self.get_logger().info(
                f'[pp_baseline] lap {self.lap_count}: {lap_t:.3f}s '
                f'(s rollover {self.last_s:.1f} → {s:.1f})')
            if self.lap_count >= self.max_laps:
                self.done = True
        self.last_s = s


def run_pp_baseline(v: float, map_name: str, n_laps: int, wall_timeout: int) -> dict:
    """sim launch + lap_time 측정 + 결과 dict."""
    cmd = [
        'bash', '-c',
        f'source /opt/ros/jazzy/setup.bash && '
        f'source ~/IFAC2026_SH/install/local_setup.bash && '
        f'export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml && '
        f'ros2 launch stack_master full_sim.launch.py '
        f'mode:=mpcc map:={map_name} '
        f'mpc_disable:=true '
        f'pp_wpnts_topic:=/global_waypoints '
        f'pp_max_speed:={v}',
    ]
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('CYCLONEDDS_URI', f'file://{os.path.expanduser("~/cyclonedds.xml")}')
    # ROS_DOMAIN_ID 격리 — PP baseline = 19, BO trial = 18. DDS state 충돌 방지.
    env['ROS_DOMAIN_ID'] = '19'
    # LapTimer 도 같은 도메인 사용 — rclpy.init 전에 set.
    os.environ['ROS_DOMAIN_ID'] = '19'
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid, env=env)
    print(f'[pp_baseline] sim launched (v={v}, map={map_name}, n_laps={n_laps})')

    # rclpy 별도 init/spin — sim 외부에서 odom subscribe
    rclpy.init()
    timer = LapTimer(max_laps=n_laps)
    t0 = time.time()
    reason = 'wall timeout'
    try:
        while not timer.done:
            rclpy.spin_once(timer, timeout_sec=0.5)
            if time.time() - t0 > wall_timeout:
                reason = f'wall timeout {wall_timeout}s'
                break
        if timer.done:
            reason = f'reached {n_laps} laps'
    finally:
        timer.destroy_node()
        rclpy.shutdown()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            time.sleep(3)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        # 좀비 정리
        # 좀비 정리 — frenet_conversion 등 잔존 시 다음 BO trial 의 mpc_node 가
        # DDS participant slot 못 받아 startup fail. 강하게 + 충분히 대기.
        for _ in range(2):
            subprocess.run(
                'pkill -KILL -f "gym_bridge\\|state_machine\\|spliner\\|controller_manager\\|'
                'global_republisher\\|frenet_conversion\\|frenet_odom_republisher\\|'
                'static_obstacle_manager\\|fake_topic_relay\\|simple_mux\\|'
                'ego_robot_state_publisher\\|rviz2\\|mpc_node\\|mpc_debug_logger\\|'
                'pp_fallback\\|ftg_fallback\\|joy_node\\|robot_state_publisher"',
                shell=True, check=False,
            )
            time.sleep(1)
        subprocess.run('pkill -KILL -f "ros2.*daemon"', shell=True, check=False)
        time.sleep(20)  # DDS slot 회수 (10→20)

    laps = timer.lap_count
    lap_times = timer.lap_times
    best_lap = float(min(lap_times)) if lap_times else float('nan')
    mean_lap = float(sum(lap_times) / len(lap_times)) if lap_times else float('nan')
    return {
        'v': v,
        'map': map_name,
        'reason': reason,
        'laps': laps,
        'lap_times': lap_times,
        'best_lap_time': best_lap,
        'mean_lap_time': mean_lap,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v', type=float, required=True, help='PP max_speed (BO 의 v 와 동일)')
    p.add_argument('--map', default='f')
    p.add_argument('--n_laps', type=int, default=3)
    p.add_argument('--wall_timeout', type=int, default=180)
    args = p.parse_args()

    print(f'\n=== PP baseline 측정 (v={args.v}) ===')
    result = run_pp_baseline(args.v, args.map, args.n_laps, args.wall_timeout)
    print(f'  reason: {result["reason"]}')
    print(f'  laps:   {result["laps"]}')
    if result['lap_times']:
        print(f'  lap_times: {[f"{t:.3f}s" for t in result["lap_times"]]}')
        print(f'  best:   {result["best_lap_time"]:.3f}s')
        print(f'  mean:   {result["mean_lap_time"]:.3f}s')
    else:
        print('  (lap 완주 실패)')

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = BO_DIR / f'pp_baseline_v{args.v}_{ts}.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f'  saved → {out_path}')

    # exit code: 0 if lap 완주, 1 if 실패
    sys.exit(0 if result['laps'] >= args.n_laps else 1)


if __name__ == '__main__':
    main()
