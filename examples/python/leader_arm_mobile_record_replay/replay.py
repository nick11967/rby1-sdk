#!/usr/bin/env python3
"""Replay an arm, gripper, and two-wheel mobile-base trajectory."""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import rby1_sdk as rby


EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
leader_example = importlib.import_module("35_leader_arm_teleop_with_monitor")
Gripper = leader_example.Gripper


def load_recording(path: Path) -> dict[str, np.ndarray]:
    required = {
        "schema_version",
        "model_name",
        "joint_names",
        "time_s",
        "position",
        "odometry",
        "base_command",
        "gripper_command",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Recording is missing fields: {sorted(missing)}")
        recording = {name: data[name].copy() for name in data.files}

    version = int(recording["schema_version"].item())
    if version not in (1, 2):
        raise ValueError(f"Unsupported recording schema version: {version}")
    sample_count = len(recording["time_s"])
    for name in ("position", "odometry", "base_command", "gripper_command"):
        if len(recording[name]) != sample_count:
            raise ValueError(f"Inconsistent sample count in {name}")
    if sample_count == 0:
        raise ValueError("Recording contains no samples")
    if recording["time_s"].ndim != 1:
        raise ValueError("time_s must be a one-dimensional array")
    if recording["position"].ndim != 2:
        raise ValueError("position must have shape (samples, robot_dof)")
    if recording["odometry"].shape[1:] != (3, 3):
        raise ValueError("odometry must have shape (samples, 3, 3)")
    if recording["base_command"].shape[1:] != (3,):
        raise ValueError("base_command must have shape (samples, 3)")
    if recording["gripper_command"].shape[1:] != (2,):
        raise ValueError("gripper_command must have shape (samples, 2)")
    for name in ("time_s", "position", "odometry", "base_command", "gripper_command"):
        if not np.isfinite(recording[name]).all():
            raise ValueError(f"Recording contains non-finite values in {name}")
    if np.any(np.diff(recording["time_s"]) < 0):
        raise ValueError("Recording timestamps are not monotonic")
    return recording


def build_body_command(body_position, minimum_time, args):
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(
                    args.control_hold_time
                )
            )
            .set_position(body_position)
            .set_minimum_time(minimum_time)
        )
    )


def build_mobile_command(base_command, args):
    mobility = (
        rby.SE2VelocityCommandBuilder()
        .set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(
                args.mobile_hold_time
            )
        )
        .set_minimum_time(args.mobile_ramp_time)
        .set_acceleration_limit(
            np.array([args.linear_acceleration, args.linear_acceleration]),
            args.angular_acceleration,
        )
        .set_velocity(base_command[:2], float(base_command[2]))
    )
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_mobility_command(mobility)
    )


def build_component_command(body_position, base_command, minimum_time, args):
    """Build a combined command for offline compatibility/tests."""
    body = (
        rby.JointPositionCommandBuilder()
        .set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(args.control_hold_time)
        )
        .set_position(body_position)
        .set_minimum_time(minimum_time)
    )
    mobility = (
        rby.SE2VelocityCommandBuilder()
        .set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(
                args.mobile_hold_time
            )
        )
        .set_minimum_time(args.mobile_ramp_time)
        .set_acceleration_limit(
            np.array([args.linear_acceleration, args.linear_acceleration]),
            args.angular_acceleration,
        )
        .set_velocity(base_command[:2], float(base_command[2]))
    )
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(body)
        .set_mobility_command(mobility)
    )


def send_base_stop(stream, args) -> None:
    if stream is None:
        return
    try:
        stop = (
            rby.SE2VelocityCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(
                    args.mobile_hold_time
                )
            )
            .set_minimum_time(0.05)
            .set_acceleration_limit(
                np.array([args.linear_acceleration, args.linear_acceleration]),
                args.angular_acceleration,
            )
            .set_velocity(np.zeros(2), 0.0)
        )
        stream.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_mobility_command(stop)
            )
        )
        time.sleep(args.mobile_hold_time)
    except Exception as exc:
        logging.warning("Failed to send final mobile-base stop: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="NPZ file produced by record.py")
    parser.add_argument("--address", required=True, help="Robot address")
    parser.add_argument("--model", default="a", help="Robot model (this example requires A)")
    parser.add_argument("--power", default=".*", help="Power-device regex")
    parser.add_argument("--servo", default=".*", help="Servo regex including wheels")
    parser.add_argument("--priority", type=int, default=1, help="Robot command priority")
    parser.add_argument(
        "--mobility-priority",
        type=int,
        default=10,
        help="Mobility command priority (matches example 37)",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Safety delay before any replay motion [s]",
    )
    parser.add_argument("--ready-time", type=float, default=5.0, help="Time to move to first recorded pose [s]")
    parser.add_argument("--control-hold-time", type=float, default=0.10, help="Command hold time [s]")
    parser.add_argument(
        "--mobile-hold-time",
        type=float,
        default=0.75,
        help="Mobility command hold time [s]",
    )
    parser.add_argument(
        "--mobile-ramp-time",
        type=float,
        default=0.10,
        help="Minimum mobility velocity-profile ramp time [s]",
    )
    parser.add_argument(
        "--mobile-refresh-time",
        type=float,
        default=0.40,
        help="Interval for refreshing an unchanged mobility command [s]",
    )
    parser.add_argument("--linear-acceleration", type=float, default=0.50, help="Base linear acceleration limit [m/s^2]")
    parser.add_argument("--angular-acceleration", type=float, default=1.0, help="Base angular acceleration limit [rad/s^2]")
    parser.add_argument(
        "--skip-gripper",
        action="store_true",
        help="Replay arm/base only and do not initialize the gripper",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.start_delay < 0:
        parser.error("--start-delay must be non-negative")
    if args.ready_time <= 0 or args.control_hold_time <= 0:
        parser.error("--ready-time and --control-hold-time must be positive")
    if args.mobile_hold_time <= 0 or args.mobile_ramp_time <= 0:
        parser.error("Mobile hold and ramp times must be positive")
    if not 0 < args.mobile_refresh_time < args.mobile_hold_time:
        parser.error("Mobile refresh time must be positive and shorter than hold time")
    if args.linear_acceleration <= 0 or args.angular_acceleration <= 0:
        parser.error("Acceleration limits must be positive")

    stop_event = threading.Event()
    robot = None
    gripper = None
    arm_stream = None
    mobility_stream = None

    def request_stop(signum=None, frame=None):
        stop_event.set()

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        recording = load_recording(args.recording)
        recorded_model = str(recording["model_name"].item())
        if recorded_model != "A":
            raise RuntimeError(f"Recording model is {recorded_model}, expected A")

        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            raise RuntimeError(f"Failed to connect to robot at {args.address}")
        model = robot.model()
        if model.model_name != recorded_model or len(model.mobility_idx) != 2:
            raise RuntimeError(
                f"Connected model {model.model_name} does not match recording model {recorded_model}"
            )
        if list(recording["joint_names"]) != list(model.robot_joint_names):
            raise RuntimeError("Recorded joint layout does not match the connected robot")
        if recording["position"].shape[1] != len(model.robot_joint_names):
            raise RuntimeError("Recorded joint-position width is invalid")

        if not robot.is_power_on(args.power) and not robot.power_on(args.power):
            raise RuntimeError(f"Failed to power on devices matching {args.power!r}")
        if not robot.is_servo_on(args.servo) and not robot.servo_on(args.servo):
            raise RuntimeError(f"Failed to servo on devices matching {args.servo!r}")
        robot.reset_fault_control_manager()
        if not robot.enable_control_manager():
            raise RuntimeError("Failed to enable the control manager")

        logging.warning(
            "Replay will move the arms and mobile base. Starting in %.1f seconds; "
            "press Ctrl+C to abort.",
            args.start_delay,
        )
        if stop_event.wait(args.start_delay):
            logging.info("Replay aborted before motion")
            return 0

        if not args.skip_gripper:
            for arm in ("right", "left"):
                if not robot.set_tool_flange_output_voltage(arm, 12):
                    raise RuntimeError(
                        f"Failed to enable 12 V tool power on {arm} arm"
                    )
            gripper = Gripper()
            if not gripper.initialize():
                raise RuntimeError("Failed to initialize the gripper")
            gripper.homing()
            gripper.start()

        body_indices = list(model.body_idx)
        first_body = recording["position"][0, body_indices]
        ready_command = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(
                        args.control_hold_time
                    )
                )
                .set_position(first_body)
                .set_minimum_time(args.ready_time)
            )
        )
        ready_feedback = robot.send_command(
            ready_command, priority=args.priority
        ).get()
        if ready_feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(
                f"Failed to move to first recorded pose: {ready_feedback.finish_code}"
            )

        arm_stream = robot.create_command_stream(priority=args.priority)
        mobility_stream = robot.create_command_stream(
            priority=args.mobility_priority
        )
        timestamps = recording["time_s"]
        logging.info("Replaying %d samples (Ctrl+C to stop)", len(timestamps))
        next_deadline = time.monotonic()
        last_mobile_command = np.full(3, np.nan, dtype=np.float64)
        last_mobile_send_time = 0.0
        for index in range(len(timestamps)):
            if stop_event.is_set():
                break
            if index + 1 < len(timestamps):
                dt = max(float(timestamps[index + 1] - timestamps[index]), 0.001)
            elif index > 0:
                dt = max(float(timestamps[index] - timestamps[index - 1]), 0.001)
            else:
                dt = 0.01

            arm_stream.send_command(
                build_body_command(
                    recording["position"][index, body_indices],
                    max(dt, 0.01),
                    args,
                )
            )
            base_command = recording["base_command"][index]
            now = time.monotonic()
            if (
                not np.array_equal(base_command, last_mobile_command)
                or now - last_mobile_send_time >= args.mobile_refresh_time
            ):
                mobility_stream.send_command(
                    build_mobile_command(base_command, args)
                )
                last_mobile_command = base_command.copy()
                last_mobile_send_time = now
            if gripper is not None:
                gripper.set_target(recording["gripper_command"][index])

            next_deadline += dt
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                stop_event.wait(remaining)

        logging.info("Replay %s", "interrupted" if stop_event.is_set() else "complete")
        return 0
    except Exception:
        logging.exception("Replay failed")
        return 1
    finally:
        stop_event.set()
        send_base_stop(mobility_stream, args)
        if robot is not None:
            try:
                robot.cancel_control()
            except Exception:
                pass
            try:
                robot.disable_control_manager()
            except Exception:
                pass
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass
        if robot is not None and not args.skip_gripper:
            try:
                robot.power_off("12v")
            except Exception:
                pass
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
