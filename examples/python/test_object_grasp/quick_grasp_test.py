#!/usr/bin/env python3
"""Quick Robot Hand Grasp & Contact Test (Approach A: Finger Stall Detection).

This script demonstrates Approach A:
- Detects object grasping by checking if the fingers stall/stop against an object
  while being commanded to close.
- Also monitors the wrist 6-axis FT sensor for table contact and lift force.

Usage:
    python quick_grasp_test.py --address 192.168.30.1:50051 --model a
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
import numpy as np

import rby1_sdk as rby

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

leader_example = importlib.import_module("35_leader_arm_teleop_with_monitor")


class EnhancedGripper(leader_example.Gripper):
    """Gripper with real-time finger position tracking for Approach A grasp detection."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.present_q = np.array([0.0, 0.0])

    def loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            with self.lock:
                if self.target_q is not None:
                    self.bus.group_sync_write_send_position(
                        [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                    )
                rv = self.bus.group_fast_sync_read_encoder([0, 1])
                if rv is not None:
                    for dev_id, enc in rv:
                        self.present_q[dev_id] = enc
            time.sleep(0.05)

    def get_closed_ratio(self) -> np.ndarray:
        """Returns normalized closed ratio: 0.0 = fully open, 1.0 = fully closed."""
        with self.lock:
            if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
                return np.array([0.0, 0.0])
            span = np.maximum(self.max_q - self.min_q, 1e-4)
            closed_ratio = (self.present_q - self.min_q) / span
            return np.clip(closed_ratio, 0.0, 1.0)


class KeyboardReader:
    """Non-blocking keyboard input."""

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
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                return sys.stdin.read(1)
        return None


def main():
    parser = argparse.ArgumentParser(description="Quick Grasp Test (Approach A)")
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot IP:port")
    parser.add_argument("--model", default="a", help="Robot model (default: a)")
    parser.add_argument("--ft-threshold", type=float, default=3.0, help="Wrist FT contact threshold [N] (default: 3.0)")
    args = parser.parse_args()

    print(f"Connecting to robot at {args.address}...")
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        print(f"Failed to connect to robot at {args.address}")
        return 1

    # 1. Power on 12V tool flange
    print("Powering on 12V tool flange...")
    robot.set_tool_flange_output_voltage("right", 12)
    robot.set_tool_flange_output_voltage("left", 12)
    time.sleep(0.5)

    # 2. Initialize and home gripper
    print("Initializing gripper...")
    gripper = EnhancedGripper()
    if not gripper.initialize():
        print("Failed to initialize gripper. Check /dev/rby1_gripper.")
        return 1

    print("Homing gripper (calibrating travel limits and opening fingers)...")
    gripper.homing()
    gripper.start()

    # Target: 0.0 = Open, 1.0 = Closed
    target_close = np.array([0.0, 0.0])
    gripper.set_target(target_close)
    print("Gripper homed and opened!")

    # 3. Setup live FT sensor state stream
    state_lock = threading.Lock()
    ft_right = np.zeros(3)
    ft_left = np.zeros(3)
    baseline_r = None
    baseline_l = None

    def state_callback(s):
        nonlocal ft_right, ft_left
        with state_lock:
            ft_right = s.ft_sensor_right.force.copy()
            ft_left = s.ft_sensor_left.force.copy()

    robot.start_state_update(state_callback, rate=30)
    time.sleep(0.5)

    def do_tare():
        nonlocal baseline_r, baseline_l
        samples_r, samples_l = [], []
        for _ in range(15):
            with state_lock:
                samples_r.append(ft_right.copy())
                samples_l.append(ft_left.copy())
            time.sleep(0.02)
        baseline_r = np.mean(samples_r, axis=0)
        baseline_l = np.mean(samples_l, axis=0)

    print("Taring wrist FT baseline...")
    do_tare()

    print("\n" + "=" * 76)
    print("  OBJECT GRASP TEST (APPROACH A: FINGER STALL DETECTION)")
    print("  Controls:")
    print("    [O] : Open Gripper (0% closed)")
    print("    [C] : Close Gripper onto Object (Target: 100% closed)")
    print("    [T] : Re-Tare Wrist FT Sensor")
    print("    [Q] : Quit")
    print("=" * 76 + "\n")

    try:
        with KeyboardReader() as kbd:
            last_display_time = 0.0
            while True:
                now = time.monotonic()
                key = kbd.get_key()
                if key:
                    key = key.lower()
                    if key == "o":
                        target_close = np.array([0.0, 0.0])
                        gripper.set_target(target_close)
                    elif key == "c":
                        target_close = np.array([1.0, 1.0])
                        gripper.set_target(target_close)
                    elif key == "t":
                        do_tare()
                    elif key in ("q", "\x03"):
                        break

                if now - last_display_time >= 0.1:
                    with state_lock:
                        cur_r = ft_right.copy()
                        cur_l = ft_left.copy()

                    df_r = np.linalg.norm(cur_r - baseline_r) if baseline_r is not None else 0.0
                    df_l = np.linalg.norm(cur_l - baseline_l) if baseline_l is not None else 0.0

                    # Read actual finger position from encoders
                    actual_closed = gripper.get_closed_ratio()
                    r_act = actual_closed[0]
                    l_act = actual_closed[1]

                    # Approach A Detection:
                    # Target is closed (> 0.4), but fingers stalled on object (actual_closed < 0.88)
                    stall_gap_r = target_close[0] - r_act
                    is_grasped_r = (target_close[0] > 0.4) and (r_act < 0.88) and (stall_gap_r > 0.12)

                    stall_gap_l = target_close[1] - l_act
                    is_grasped_l = (target_close[1] > 0.4) and (l_act < 0.88) and (stall_gap_l > 0.12)

                    # Determine status
                    if is_grasped_r:
                        status_r = f"\033[92;1m*** GRASPED (Object: {int((1-r_act)*100)}% wide) ***\033[0m"
                    elif target_close[0] > 0.4 and r_act >= 0.88:
                        status_r = "\033[93mCLOSED / EMPTY (No Object)\033[0m   "
                    else:
                        status_r = f"\033[94mOPEN ({int((1-r_act)*100)}% open)\033[0m              "

                    table_r = f" [Table Contact: {df_r:.1f}N]" if df_r >= args.ft_threshold else ""

                    print(
                        f"\rRight: {status_r} | Pos: {int(r_act*100)}% closed | Wrist FT: ΔF={df_r:4.1f}N{table_r:<22}",
                        end="",
                        flush=True,
                    )
                    last_display_time = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nStopping test...")
        gripper.stop()
        robot.stop_state_update()
        robot.disconnect()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
