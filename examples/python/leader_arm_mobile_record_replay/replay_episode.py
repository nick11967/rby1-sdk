#!/usr/bin/env python3
"""Replay a recorded RBY1 episode (.npz) on the robot hardware with Auto-Homing.

This script:
1. Loads a recorded episode .npz file (e.g. recordings/episode_0001.npz).
2. Connects to the robot and starts the state stream.
3. Automatically and smoothly homes the robot from its CURRENT pose to the trajectory's initial pose.
4. Streams the recorded 100 Hz joint trajectory (and base velocity commands) to the robot.
5. Safely cleans up on completion or Ctrl+C without causing a Major Fault.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import logging
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np
import rby1_sdk as rby

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
leader_example = importlib.import_module("35_leader_arm_teleop_with_monitor")
Settings = leader_example.Settings
Pose = leader_example.Pose
joint_position_command_builder = leader_example.joint_position_command_builder


def load_episode_data(path: Path) -> dict[str, np.ndarray]:
    """Load and validate an episode .npz file."""
    if not path.is_file():
        raise FileNotFoundError(f"Recording file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        recording = {name: data[name].copy() for name in data.files}

    if "position" not in recording or "time_s" not in recording:
        raise ValueError(f"Recording file {path} missing required fields 'position' or 'time_s'")

    sample_count = len(recording["time_s"])
    if sample_count == 0:
        raise ValueError(f"Recording {path} contains 0 samples")

    return recording


def s_curve_quintic(t: float, total_time: float) -> float:
    """Quintic polynomial smoothstep for minimum-jerk trajectory."""
    if total_time <= 0.0:
        return 1.0
    tau = np.clip(t / total_time, 0.0, 1.0)
    return float(10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5))


def _build_mobile_command(base_command: np.ndarray, args: argparse.Namespace):
    return (
        rby.SE2VelocityCommandBuilder()
        .set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(args.mobile_hold_time)
        )
        .set_minimum_time(args.mobile_ramp_time)
        .set_acceleration_limit(
            np.array([args.linear_acceleration, args.linear_acceleration]),
            float(args.angular_acceleration),
        )
        .set_velocity(base_command[:2], float(base_command[2]))
    )


def _send_base_stop(mobility_stream, args: argparse.Namespace) -> None:
    if mobility_stream is None:
        return
    try:
        mobility_stream.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_mobility_command(
                    _build_mobile_command(np.zeros(3, dtype=np.float64), args)
                )
            )
        )
    except Exception as exc:
        logging.warning("Failed to send stop command to mobility base: %s", exc)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Path to episode .npz file (e.g. recordings/episode_0001.npz)")
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot address (default: 192.168.30.1:50051)")
    parser.add_argument("--model", default="a", help="Robot model (default: a)")
    parser.add_argument("--power", default=".*", help="Power-device regex")
    parser.add_argument("--servo", default=".*", help="Servo regex")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed scale (default: 1.0 = real-time, 0.5 = half-speed)")
    parser.add_argument("--home-time", type=float, default=4.0, help="Duration to smoothly home robot to initial trajectory pose [s]")
    parser.add_argument("--priority", type=int, default=1, help="Robot command priority")
    parser.add_argument("--mobility-priority", type=int, default=10, help="Mobility command priority")
    parser.add_argument("--mobile-hold-time", type=float, default=0.75, help="Mobility hold time [s]")
    parser.add_argument("--mobile-ramp-time", type=float, default=0.10, help="Mobility ramp time [s]")
    parser.add_argument("--linear-acceleration", type=float, default=0.50, help="Base linear acceleration [m/s^2]")
    parser.add_argument("--angular-acceleration", type=float, default=1.0, help="Base angular acceleration [rad/s^2]")
    parser.add_argument("--skip-mobility", action="store_true", help="Skip mobile base command replay (joint arms only)")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load trajectory
    recording = load_episode_data(args.recording)
    positions = recording["position"]  # (N, 24)
    time_s = recording["time_s"]
    num_samples = len(time_s)
    total_duration = time_s[-1]
    has_base_command = "base_command" in recording and not args.skip_mobility
    base_commands = recording["base_command"] if has_base_command else None

    logging.info("=" * 60)
    logging.info("Loaded Episode: %s", args.recording)
    logging.info("  Total Samples:  %d", num_samples)
    logging.info("  Duration:       %.2f s (Replay duration: %.2f s at %.1fx)", total_duration, total_duration / args.speed, args.speed)
    logging.info("  Base Commands:  %s", "YES" if has_base_command else "NO (joint arms only)")
    logging.info("=" * 60)

    stop_event = threading.Event()
    robot = None
    arm_stream = None
    mobility_stream = None

    def request_stop(sig=None, frame=None):
        logging.info("\nReplay stop requested by user.")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        # 1. Connect to Robot
        logging.info("Connecting to robot at %s...", args.address)
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            logging.error("Failed to connect to robot at %s!", args.address)
            return 1

        model = robot.model()

        # 2. Power and Servo Setup
        logging.info("Ensuring robot power and servos are enabled...")
        robot.power_on(args.power)
        for arm in ["right", "left"]:
            robot.set_tool_flange_output_voltage(arm, 12)
        robot.servo_on(args.servo)
        robot.enable_control_manager()
        robot.set_parameter("joint_position_command.cutoff_frequency", "5")

        # 3. Read Current Robot State
        robot_position = None
        state_lock = threading.Lock()

        def on_robot_state(state):
            nonlocal robot_position
            with state_lock:
                robot_position = state.position.copy()

        robot.start_state_update(on_robot_state, 100.0)
        deadline = time.monotonic() + 3.0
        while robot_position is None and time.monotonic() < deadline:
            time.sleep(0.01)

        if robot_position is None:
            raise RuntimeError("Timed out waiting for robot state update.")

        # 4. Smooth Auto-Homing to Initial Trajectory Pose
        target_start_pos = positions[0]
        start_torso_q = target_start_pos[model.torso_idx]
        start_right_q = target_start_pos[model.right_arm_idx]
        start_left_q = target_start_pos[model.left_arm_idx]

        initial_pose = Pose(
            toros=start_torso_q,
            right_arm=start_right_q,
            left_arm=start_left_q,
        )

        logging.info("=" * 60)
        logging.info("Auto-Homing robot to trajectory start pose (%.1f s)...", args.home_time)
        home_feedback = robot.send_command(
            joint_position_command_builder(initial_pose, minimum_time=args.home_time),
            priority=args.priority,
        ).get()

        if home_feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Auto-homing to start pose failed: {home_feedback.finish_code}")

        logging.info("Robot arrived at initial trajectory pose successfully.")
        logging.info("=" * 60)
        time.sleep(0.5)

        # 5. Create Command Streams
        arm_stream = robot.create_command_stream(priority=args.priority)
        if has_base_command:
            mobility_stream = robot.create_command_stream(priority=args.mobility_priority)

        # 6. Stream Replay Trajectory
        logging.info("Starting trajectory replay at %.1fx speed...", args.speed)
        start_mono = time.monotonic()
        replay_start_time = start_mono
        dt_target = 0.01 / max(0.1, args.speed)  # 100 Hz step scaled by speed

        for i in range(num_samples):
            if stop_event.is_set():
                break

            target_pos = positions[i]
            cur_pose = Pose(
                toros=target_pos[model.torso_idx],
                right_arm=target_pos[model.right_arm_idx],
                left_arm=target_pos[model.left_arm_idx],
            )

            # Send Body Joint Command
            cmd = joint_position_command_builder(cur_pose, minimum_time=0.02, control_hold_time=0.1)
            arm_stream.send_command(cmd)

            # Send Mobility Base Command (if present)
            if has_base_command and mobility_stream is not None:
                base_cmd = base_commands[i]
                mobility_stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_mobility_command(
                            _build_mobile_command(base_cmd, args)
                        )
                    )
                )

            # Progress logging every 1.0s
            elapsed_real = time.monotonic() - replay_start_time
            progress_pct = (i + 1) / num_samples * 100.0
            if (i % 100) == 0 or i == num_samples - 1:
                sys.stdout.write(f"\r[REPLAYING] Progress: {progress_pct:5.1f}% | Time: {time_s[i]:5.2f} / {total_duration:5.2f}s | Real: {elapsed_real:5.2f}s")
                sys.stdout.flush()

            # Precise timing
            next_tick = replay_start_time + (time_s[i] / max(0.1, args.speed))
            now = time.monotonic()
            if next_tick > now:
                time.sleep(next_tick - now)

        print("\n")
        logging.info("=" * 60)
        logging.info("Trajectory replay completed successfully!")
        logging.info("=" * 60)

    except Exception:
        logging.exception("Exception during trajectory replay!")
        return 1
    finally:
        logging.info("Initiating graceful replay shutdown...")
        stop_event.set()

        # Stop mobility base
        _send_base_stop(mobility_stream, args)

        # Cancel command streams cleanly (maintains control manager without Major Fault)
        if arm_stream is not None:
            try:
                arm_stream.cancel()
            except Exception:
                pass
        if mobility_stream is not None:
            try:
                mobility_stream.cancel()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.stop_state_update()
            except Exception:
                pass
            try:
                robot.cancel_control()
            except Exception:
                pass

        logging.info("Replay cleanup complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
