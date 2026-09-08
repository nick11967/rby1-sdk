#!/usr/bin/env python3
"""Standalone Real-Time Grasp & Contact Detector.

Run this script in a SECOND terminal alongside your original tele-op command:
    Terminal 1 (Teleop):
        /mnt/ssd/rby1-sdk/.venv/bin/python /mnt/ssd/rby1-sdk/examples/python/leader_arm_mobile_record_replay/teleop_autohome.py --address 192.168.30.1:50051 --model a --mode position

    Terminal 2 (Grasp Detector):
        /mnt/ssd/rby1-sdk/.venv/bin/python /mnt/ssd/rby1-sdk/examples/python/test_object_grasp/detector.py --address 192.168.30.1:50051 --model a

Features:
- Subscribes to teleop trigger state from /dev/shm (zero-latency shared memory from teleop_autohome).
- Reads wrist 6-axis FT sensors (Right & Left) for table contact and lifted object weight.
- Reads gripper finger position stall (Approach A) when holding an object.
- 100% read-only: does not interfere with or command the robot.
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

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
LEADER_DIR = EXAMPLES_DIR / "leader_arm_mobile_record_replay"

for p in [str(EXAMPLES_DIR), str(LEADER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from camera_io import TeleopStateSubscriber
except ImportError:
    TeleopStateSubscriber = None


class KeyboardReader:
    """Non-blocking keyboard reader."""

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


def bar(val: float, max_val: float = 1.0, width: int = 14) -> str:
    ratio = min(max(val / max_val, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    parser = argparse.ArgumentParser(description="Standalone Grasp Detector")
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot IP:port")
    parser.add_argument("--model", default="a", help="Robot model (default: a)")
    parser.add_argument("--threshold", type=float, default=2.5, help="Contact/lift force threshold in N (default: 2.5)")
    parser.add_argument("--rate", type=float, default=25.0, help="Display refresh rate in Hz (default: 25)")
    args = parser.parse_args()

    print(f"Connecting to robot at {args.address} (Read-Only Listener)...")
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        print(f"Failed to connect to {args.address}")
        return 1

    # Initialize Teleop SHM subscriber (reads live trigger from teleop_autohome)
    teleop_sub = TeleopStateSubscriber() if TeleopStateSubscriber is not None else None

    # Try connecting to Gripper Dynamixel bus to read encoders
    gripper_bus = None
    try:
        gripper_bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        if not gripper_bus.open_port():
            gripper_bus = None
        else:
            gripper_bus.set_baud_rate(2_000_000)
    except Exception:
        gripper_bus = None

    # Stream FT sensor state
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

    print("Calibrating initial resting baseline (Tare)...")
    tare_baseline()
    print("Baseline set! Starting Live Detector Dashboard...\n")
    time.sleep(0.5)

    # Gripper travel limits (calibrated defaults for RB-Y1 parallel gripper)
    # min_q ~ 3.15 rad (closed), max_q ~ 7.66 rad (open)
    min_q = np.array([3.15, 3.11])
    max_q = np.array([7.66, 7.66])

    force_threshold = args.threshold

    try:
        with KeyboardReader() as kbd:
            last_render = 0.0
            tare_message_until = 0.0
            prev_act_r = 0.0
            prev_act_l = 0.0
            stall_start_r = None
            stall_start_l = None
            q_open_r = 3.1600
            q_open_l = -3.1661

            while True:
                now = time.monotonic()
                key = kbd.get_key()
                if key:
                    key = key.lower()
                    if key in ("t", " "):
                        tare_baseline()
                        tare_message_until = now + 1.5
                    elif key in ("+", "="):
                        force_threshold += 0.5
                    elif key in ("-", "_"):
                        force_threshold = max(0.5, force_threshold - 0.5)
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

                    # 1. Read trigger command from teleop_autohome via SHM
                    trig_raw_r = 1.0
                    trig_raw_l = 1.0
                    if teleop_sub is not None:
                        shm_data = teleop_sub.read(max_age_s=0.5)
                        if shm_data is not None:
                            trig_raw_r = float(shm_data[0][0])
                            trig_raw_l = float(shm_data[0][1])

                    # In teleop_autohome, trigger is 1000/1000 = 1.0 when RELEASED (open)
                    # and drops towards 0.0 when SQUEEZED (closed).
                    cmd_closed_r = float(np.clip(1.0 - trig_raw_r, 0.0, 1.0))
                    cmd_closed_l = float(np.clip(1.0 - trig_raw_l, 0.0, 1.0))

                    # 2. Read finger encoders
                    act_closed_r = 0.0
                    act_closed_l = 0.0
                    has_finger_data = False
                    if gripper_bus is not None:
                        try:
                            rv = gripper_bus.group_fast_sync_read_encoder([0, 1])
                            if rv is not None and len(rv) == 2:
                                encs = {dev_id: val for dev_id, val in rv}
                                if 0 in encs and 1 in encs:
                                    # Auto-refine open resting angle when triggers are released
                                    if cmd_closed_r <= 0.05:
                                        q_open_r = encs[0]
                                    if cmd_closed_l <= 0.05:
                                        q_open_l = encs[1]

                                    # Stroke is ~4.50 rad for both parallel grippers
                                    # Displacement: |q - q_open| / stroke
                                    stroke = 4.50
                                    act_closed_r = float(np.clip(abs(encs[0] - q_open_r) / stroke, 0.0, 1.0))
                                    act_closed_l = float(np.clip(abs(encs[1] - q_open_l) / stroke, 0.0, 1.0))
                                    has_finger_data = True
                        except Exception:
                            pass

                    # 3. Motion & Velocity Tracking
                    # Estimate finger velocity (% stroke per second) to prevent false GRASPED during air travel
                    dt = max(now - last_render, 1e-4) if last_render > 0 else (1.0 / args.rate)
                    vel_r = abs(act_closed_r - prev_act_r) / dt
                    vel_l = abs(act_closed_l - prev_act_l) / dt
                    prev_act_r = act_closed_r
                    prev_act_l = act_closed_l

                    is_moving_r = (vel_r > 0.12)  # moving faster than 12% closure per sec
                    is_moving_l = (vel_l > 0.12)

                    # 4. Grasp Detection Logic (Approach A - Velocity-Filtered & Debounced)
                    EMPTY_THRESHOLD = 0.80  # Touching fingers in empty air reach > 0.80
                    stall_r = cmd_closed_r - act_closed_r
                    stall_l = cmd_closed_l - act_closed_l

                    # Candidate for holding an object: trigger squeezed, fingers stalled partway
                    stall_candidate_r = (cmd_closed_r > 0.35) and (act_closed_r < EMPTY_THRESHOLD) and (stall_r > 0.12)
                    if stall_candidate_r and not is_moving_r:
                        if stall_start_r is None:
                            stall_start_r = now
                        stall_dur_r = now - stall_start_r
                    else:
                        stall_start_r = None
                        stall_dur_r = 0.0

                    stall_candidate_l = (cmd_closed_l > 0.35) and (act_closed_l < EMPTY_THRESHOLD) and (stall_l > 0.12)
                    if stall_candidate_l and not is_moving_l:
                        if stall_start_l is None:
                            stall_start_l = now
                        stall_dur_l = now - stall_start_l
                    else:
                        stall_start_l = None
                        stall_dur_l = 0.0

                    # Right Hand Badge
                    if has_finger_data:
                        if act_closed_r >= EMPTY_THRESHOLD:
                            badge_r = "\033[93mCLOSED / EMPTY (No Object)\033[0m   "
                        elif stall_candidate_r and stall_dur_r >= 0.15:
                            # Stalled solidly against object for at least 150 ms
                            badge_r = f"\033[92;1m*** GRASPED ({int((1-act_closed_r)*100)}% wide) ***\033[0m"
                        elif is_moving_r:
                            if cmd_closed_r > act_closed_r + 0.03:
                                badge_r = "\033[94mCLOSING...                    \033[0m"
                            elif cmd_closed_r < act_closed_r - 0.03:
                                badge_r = "\033[94mOPENING...                    \033[0m"
                            else:
                                badge_r = "\033[94mCLOSING...                    \033[0m"
                        elif act_closed_r <= 0.25:
                            badge_r = "\033[90mOPEN / FREE                   \033[0m"
                        else:
                            badge_r = f"\033[96mOPEN ({int((1-act_closed_r)*100)}% wide)              \033[0m"
                    else:
                        if cmd_closed_r > 0.10:
                            badge_r = f"\033[94mSQUEEZING ({int(cmd_closed_r*100)}% cmd)\033[0m      "
                        else:
                            badge_r = "\033[90mOPEN / FREE                   \033[0m"

                    # Left Hand Badge
                    if has_finger_data:
                        if act_closed_l >= EMPTY_THRESHOLD:
                            badge_l = "\033[93mCLOSED / EMPTY (No Object)\033[0m   "
                        elif stall_candidate_l and stall_dur_l >= 0.15:
                            # Stalled solidly against object for at least 150 ms
                            badge_l = f"\033[92;1m*** GRASPED ({int((1-act_closed_l)*100)}% wide) ***\033[0m"
                        elif is_moving_l:
                            if cmd_closed_l > act_closed_l + 0.03:
                                badge_l = "\033[94mCLOSING...                    \033[0m"
                            elif cmd_closed_l < act_closed_l - 0.03:
                                badge_l = "\033[94mOPENING...                    \033[0m"
                            else:
                                badge_l = "\033[94mCLOSING...                    \033[0m"
                        elif act_closed_l <= 0.25:
                            badge_l = "\033[90mOPEN / FREE                   \033[0m"
                        else:
                            badge_l = f"\033[96mOPEN ({int((1-act_closed_l)*100)}% wide)              \033[0m"
                    else:
                        if cmd_closed_l > 0.10:
                            badge_l = f"\033[94mSQUEEZING ({int(cmd_closed_l*100)}% cmd)\033[0m      "
                        else:
                            badge_l = "\033[90mOPEN / FREE                   \033[0m"

                    tare_notify = "\033[93;1m [FT TARE RE-ZEROED!] \033[0m" if now < tare_message_until else "                       "

                    # Render terminal HUD
                    sys.stdout.write("\033[H")
                    sys.stdout.write("=" * 76 + "\n")
                    sys.stdout.write(f"  RB-Y1 REAL-TIME GRASP & CONTACT DETECTOR  {tare_notify}\n")
                    sys.stdout.write("=" * 76 + "\n")
                    sys.stdout.write(f"{' [RIGHT HAND]':<38} | {' [LEFT HAND]':<38}\n")
                    sys.stdout.write(f" Status : {badge_r:<39} | Status : {badge_l:<39}\n")
                    sys.stdout.write(
                        f" Trigger: {bar(cmd_closed_r)} {int(cmd_closed_r*100):3d}% squeeze  | "
                        f"Trigger: {bar(cmd_closed_l)} {int(cmd_closed_l*100):3d}% squeeze\n"
                    )
                    if has_finger_data:
                        sys.stdout.write(
                            f" Fingers: {bar(act_closed_r)} {int(act_closed_r*100):3d}% closed ({int((1-act_closed_r)*100):3d}% wide) | "
                            f"Fingers: {bar(act_closed_l)} {int(act_closed_l*100):3d}% closed ({int((1-act_closed_l)*100):3d}% wide)\n"
                        )
                    sys.stdout.write(
                        f" WristFT: ΔF={df_r:4.1f}N {bar(df_r, 15.0)}           | "
                        f"WristFT: ΔF={df_l:4.1f}N {bar(df_l, 15.0)}\n"
                    )
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
                    sys.stdout.write(f" Threshold: {force_threshold:.1f} N  |  [T] Re-Tare  |  [+] / [-] Threshold  |  [Q] Quit\n")
                    sys.stdout.flush()

                    last_render = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n\nStopping detector...\n")
        sys.stdout.flush()
        robot.stop_state_update()
        robot.disconnect()
        print("Detector stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
