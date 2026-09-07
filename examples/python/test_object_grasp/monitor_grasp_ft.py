#!/usr/bin/env python3
"""Real-Time Force/Torque & Object Grasp Monitor.

Run this script in a second terminal alongside teleop_autohome.py (or any other teleop script).
It connects to the robot in read-only mode, streams 6-axis FT sensor data from both wrists,
and displays a live dashboard indicating contact and whether an object is grasped.

Usage:
    /mnt/ssd/rby1-sdk/.venv/bin/python /mnt/ssd/rby1-sdk/examples/python/test_object_grasp/monitor_grasp_ft.py --address 192.168.30.1:50051 --model a
"""

from __future__ import annotations

import argparse
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
import numpy as np

import rby1_sdk as rby


class KeyboardReader:
    """Non-blocking keyboard reader for interactive tare/quit."""

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self) -> str | None:
        if sys.stdin.isatty():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
            if rlist:
                return sys.stdin.read(1)
        return None


def main():
    parser = argparse.ArgumentParser(description="Live Grasp & FT Monitor")
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot IP:port")
    parser.add_argument("--model", default="a", help="Robot model (default: a)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Grasp delta force threshold in N (default: 3.0)")
    parser.add_argument("--rate", type=float, default=25.0, help="Update rate in Hz (default: 25)")
    args = parser.parse_args()

    print(f"Connecting to RB-Y1 robot at {args.address} (Read-Only Monitor)...")
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        print(f"Error: Failed to connect to robot at {args.address}")
        return 1

    print("Connected! Initializing state stream...")

    state_lock = threading.Lock()
    ft_r = np.zeros(3)
    ft_l = np.zeros(3)
    tau_r = np.zeros(3)
    tau_l = np.zeros(3)
    baseline_r = None
    baseline_l = None
    first_data_event = threading.Event()

    def state_callback(s):
        nonlocal ft_r, ft_l, tau_r, tau_l
        with state_lock:
            ft_r = s.ft_sensor_right.force.copy()
            tau_r = s.ft_sensor_right.torque.copy()
            ft_l = s.ft_sensor_left.force.copy()
            tau_l = s.ft_sensor_left.torque.copy()
        first_data_event.set()

    robot.start_state_update(state_callback, rate=args.rate)
    first_data_event.wait(timeout=3.0)

    def tare_baseline():
        nonlocal baseline_r, baseline_l
        samples_r, samples_l = [], []
        for _ in range(15):
            with state_lock:
                samples_r.append(ft_r.copy())
                samples_l.append(ft_l.copy())
            time.sleep(0.02)
        baseline_r = np.mean(samples_r, axis=0)
        baseline_l = np.mean(samples_l, axis=0)

    print("Calibrating resting baseline (Tare)...")
    tare_baseline()
    print("Baseline locked! Starting dashboard...\n")
    time.sleep(0.5)

    def force_bar(val: float, max_val: float = 20.0, width: int = 15) -> str:
        ratio = min(max(val / max_val, 0.0), 1.0)
        bars = int(ratio * width)
        return "[" + "#" * bars + "-" * (width - bars) + "]"

    try:
        with KeyboardReader() as kbd:
            last_render = 0.0
            tare_message_until = 0.0

            while True:
                now = time.monotonic()
                key = kbd.get_key()
                if key:
                    key = key.lower()
                    if key in ("t", " "):
                        tare_baseline()
                        tare_message_until = now + 1.5
                    elif key in ("q", "\x03"):
                        break

                if now - last_render >= (1.0 / args.rate):
                    with state_lock:
                        cur_r = ft_r.copy()
                        cur_l = ft_l.copy()
                        cur_tr = tau_r.copy()
                        cur_tl = tau_l.copy()

                    df_r = float(np.linalg.norm(cur_r - baseline_r)) if baseline_r is not None else 0.0
                    df_l = float(np.linalg.norm(cur_l - baseline_l)) if baseline_l is not None else 0.0
                    norm_r = float(np.linalg.norm(cur_r))
                    norm_l = float(np.linalg.norm(cur_l))

                    grasped_r = df_r >= args.threshold
                    grasped_l = df_l >= args.threshold

                    badge_r = "\033[92;1m*** GRASPED / CONTACT ***\033[0m" if grasped_r else "\033[90mFREE (No Contact)\033[0m       "
                    badge_l = "\033[92;1m*** GRASPED / CONTACT ***\033[0m" if grasped_l else "\033[90mFREE (No Contact)\033[0m       "

                    bar_r = force_bar(df_r)
                    bar_l = force_bar(df_l)

                    tare_notify = "\033[93;1m [FT TARE RE-ZEROED!] \033[0m" if now < tare_message_until else "                       "

                    # ANSI terminal rendering (clear to top left)
                    sys.stdout.write("\033[H")
                    sys.stdout.write("=" * 76 + "\n")
                    sys.stdout.write(f"  RB-Y1 FORCE / TORQUE & GRASP MONITOR   {tare_notify}\n")
                    sys.stdout.write("=" * 76 + "\n")
                    sys.stdout.write(f"{' [RIGHT WRIST SENSOR]':<38} | {' [LEFT WRIST SENSOR]':<38}\n")
                    sys.stdout.write(f" Status : {badge_r:<39} | Status : {badge_l:<39}\n")
                    sys.stdout.write(f" ΔForce : \033[1m{df_r:5.2f} N\033[0m {bar_r:<17} | ΔForce : \033[1m{df_l:5.2f} N\033[0m {bar_l:<17}\n")
                    sys.stdout.write(f" |F_raw|: {norm_r:5.2f} N                       | |F_raw|: {norm_l:5.2f} N\n")
                    sys.stdout.write("-" * 76 + "\n")
                    sys.stdout.write(
                        f" Fx,Fy,Fz [N]:  {cur_r[0]:+5.1f}, {cur_r[1]:+5.1f}, {cur_r[2]:+5.1f}   | "
                        f"Fx,Fy,Fz [N]:  {cur_l[0]:+5.1f}, {cur_l[1]:+5.1f}, {cur_l[2]:+5.1f}\n"
                    )
                    sys.stdout.write(
                        f" Tx,Ty,Tz [Nm]: {cur_tr[0]:+5.2f}, {cur_tr[1]:+5.2f}, {cur_tr[2]:+5.2f}  | "
                        f"Tx,Ty,Tz [Nm]: {cur_tl[0]:+5.2f}, {cur_tl[1]:+5.2f}, {cur_tl[2]:+5.2f}\n"
                    )
                    sys.stdout.write("=" * 76 + "\n")
                    sys.stdout.write(f" Threshold: {args.threshold:.1f} N  |  Press [T] to Re-Tare baseline  |  Press [Q] to Quit\n")
                    sys.stdout.flush()

                    last_render = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n\nStopping monitor...\n")
        sys.stdout.flush()
        robot.stop_state_update()
        robot.disconnect()
        print("Monitor stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
