#!/usr/bin/env python3
"""Leader-arm teleoperation with LeaderArm Auto-Homing and mobile base control.

This script enhances the mobile base teleoperation workflow by automatically
homing the LeaderArm to the robot's actual joint angles (READY_POSE) before
entering the active teleoperation loop, eliminating initial snapping or trajectory jumps.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Optional, Tuple

import numpy as np
import rby1_sdk as rby

from camera_io import (
    CameraSessionProcess,
    DEFAULT_CAMERA_PYTHON,
    DEFAULT_CAMERA_STACK_ROOT,
    DEFAULT_ZED_RECORD_PROFILE,
    ZED_RECORD_PROFILES,
    ZED_SHM_MODES,
    camera_sidecar_path,
    TeleopStatePublisher,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

# Import drivers and settings from SDK example 35
leader_example = importlib.import_module("35_leader_arm_teleop_with_monitor")
LeaderArm = leader_example.LeaderArm
Gripper = leader_example.Gripper
READY_POSE = leader_example.READY_POSE
Settings = leader_example.Settings
joint_position_command_builder = leader_example.joint_position_command_builder


# Self-contained LeaderArm Autohome & Utility Functions
HOMING_PID_P = 2800
HOMING_PID_I = 300
HOMING_PID_D = 4000
NUM_LEADER_MOTORS = 14


def s_curve_quintic(t: float, total_time: float) -> float:
    """5th-order polynomial S-curve for minimum-jerk trajectory."""
    if total_time <= 0.0:
        return 1.0
    tau = np.clip(t / total_time, 0.0, 1.0)
    return float(10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5))


def read_joint_positions(bus: rby.DynamixelBus, motor_ids: list[int] = list(range(NUM_LEADER_MOTORS))) -> np.ndarray:
    """Read present positions from Dynamixel bus."""
    ms_list = bus.get_motor_states(motor_ids)
    if not ms_list:
        raise RuntimeError("Failed to read leader arm joint positions")
    sorted_states = sorted(ms_list, key=lambda x: x[0])
    return np.array([mstate.position for _, mstate in sorted_states], dtype=np.float64)


def auto_home_leader_arm(
    leader_arm: LeaderArm,
    target_pose: np.ndarray,
    duration: float = 4.0,
    tolerance: float = 0.05,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Smoothly auto-home LeaderArm to target joint angles."""
    bus = leader_arm.bus
    motor_ids = list(range(NUM_LEADER_MOTORS))
    homing_torque_limits = [3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2

    for dev_id in motor_ids:
        bus.set_position_pid_gain(dev_id, p_gain=HOMING_PID_P, i_gain=HOMING_PID_I, d_gain=HOMING_PID_D)

    bus.group_sync_write_torque_enable(motor_ids, 0)
    bus.group_sync_write_operating_mode([(i, rby.DynamixelBus.CurrentBasedPositionControlMode) for i in motor_ids])
    bus.group_sync_write_torque_enable(motor_ids, 1)
    bus.group_sync_write_send_torque([(i, homing_torque_limits[i]) for i in motor_ids])

    q_start = read_joint_positions(bus, motor_ids)
    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= duration:
            break
        s = s_curve_quintic(elapsed, duration)
        q_des = q_start + s * (target_pose - q_start)
        bus.group_sync_write_send_position([(i, q_des[i]) for i in motor_ids])
        time.sleep(0.01)

    bus.group_sync_write_send_position([(i, target_pose[i]) for i in motor_ids])
    time.sleep(0.15)

    q_final = read_joint_positions(bus, motor_ids)
    errors = np.abs(q_final - target_pose)
    success = bool(np.all(errors < tolerance))
    return success, q_final, errors


class ArrowKeyController:
    """Read arrow keys from a TTY and expose a dead-man base velocity command."""

    _DIRECTIONS = {
        b"\x1b[A": (1.0, 0.0, "FORWARD"),
        b"\x1bOA": (1.0, 0.0, "FORWARD"),
        b"w": (1.0, 0.0, "FORWARD"),
        b"\x1b[B": (-1.0, 0.0, "BACKWARD"),
        b"\x1bOB": (-1.0, 0.0, "BACKWARD"),
        b"s": (-1.0, 0.0, "BACKWARD"),
        b"\x1b[D": (0.0, -1.0, "TURN LEFT"),
        b"\x1bOD": (0.0, -1.0, "TURN LEFT"),
        b"a": (0.0, -1.0, "TURN LEFT"),
        b"\x1b[C": (0.0, 1.0, "TURN RIGHT"),
        b"\x1bOC": (0.0, 1.0, "TURN RIGHT"),
        b"d": (0.0, 1.0, "TURN RIGHT"),
    }

    def __init__(
        self,
        linear_speed: float,
        angular_speed: float,
        timeout_s: float,
        stop_event: threading.Event,
        debug: bool = False,
    ) -> None:
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.timeout_s = timeout_s
        self.stop_event = stop_event
        self.debug = debug
        self._lock = threading.Lock()
        self._command = np.zeros(3, dtype=np.float64)  # vx, vy, wz
        self._expires_at = 0.0
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._old_termios = None
        self._active_label = "STOP"

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Arrow-key control requires an interactive terminal (TTY).")
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        logging.info("Keyboard input attached to %s", os.ttyname(self._fd))
        self._thread = threading.Thread(target=self._run, name="arrow-keys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.set_stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._fd is not None and self._old_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        self._fd = None
        self._old_termios = None

    def set_stop(self) -> None:
        with self._lock:
            was_moving = np.any(self._command)
            self._command.fill(0.0)
            self._expires_at = 0.0
            self._active_label = "STOP"
        if was_moving:
            logging.info("Base input: STOP")

    def get_command(self) -> np.ndarray:
        with self._lock:
            if time.monotonic() >= self._expires_at:
                was_moving = np.any(self._command)
                self._command.fill(0.0)
                self._active_label = "STOP"
                if was_moving:
                    logging.info("Base input timed out: STOP")
            return self._command.copy()

    def _set_direction(
        self, linear_sign: float, angular_sign: float, label: str
    ) -> None:
        with self._lock:
            changed = label != self._active_label
            self._command[:] = [
                linear_sign * self.linear_speed,
                0.0,
                angular_sign * self.angular_speed,
            ]
            self._expires_at = time.monotonic() + self.timeout_s
            self._active_label = label
        if changed:
            logging.info("Base input: %s", label)

    def _run(self) -> None:
        assert self._fd is not None
        pending = b""
        while not self.stop_event.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            data = os.read(self._fd, 32)
            if not data:
                self.stop_event.set()
                break
            pending += data
            if self.debug:
                logging.info("Raw keyboard bytes: %r", data)

            if b"q" in pending.lower():
                self.set_stop()
                self.stop_event.set()
                break
            if b" " in pending:
                self.set_stop()

            consumed_direction = False
            for sequence, direction in self._DIRECTIONS.items():
                if sequence in pending:
                    self._set_direction(*direction)
                    pending = pending.replace(sequence, b"")
                    consumed_direction = True

            if not consumed_direction:
                possible_prefixes = (b"\x1b", b"\x1b[", b"\x1bO")
                pending = next(
                    (prefix for prefix in possible_prefixes if pending.endswith(prefix)),
                    b"",
                )
            elif len(pending) > 2:
                pending = pending[-2:]


class TrajectoryRecorder:
    """Collect synchronized robot, leader, gripper, and mobility samples."""

    def __init__(self, output: Path, model) -> None:
        self.output = output
        self.model_name = model.model_name
        self.joint_names = list(model.robot_joint_names)
        self.started_at_ns = time.monotonic_ns()
        self.started_unix_ns = time.time_ns()
        self._wall_minus_monotonic_ns = self.started_unix_ns - self.started_at_ns
        self._lock = threading.Lock()
        self._time_s: list[float] = []
        self._monotonic_ns: list[int] = []
        self._unix_ns: list[int] = []
        self._position: list[np.ndarray] = []
        self._odometry: list[np.ndarray] = []
        self._base_command: list[np.ndarray] = []
        self._gripper_command: list[np.ndarray] = []
        self._leader_position: list[np.ndarray] = []

    def append(
        self,
        robot_position: np.ndarray,
        odometry: np.ndarray,
        base_command: np.ndarray,
        gripper_command: np.ndarray,
        leader_position: np.ndarray,
    ) -> None:
        monotonic_ns = time.monotonic_ns()
        with self._lock:
            self._time_s.append((monotonic_ns - self.started_at_ns) / 1e9)
            self._monotonic_ns.append(monotonic_ns)
            self._unix_ns.append(monotonic_ns + self._wall_minus_monotonic_ns)
            self._position.append(robot_position.copy())
            self._odometry.append(odometry.copy())
            self._base_command.append(base_command.copy())
            self._gripper_command.append(gripper_command.copy())
            self._leader_position.append(leader_position.copy())

    def save(self) -> int:
        with self._lock:
            count = len(self._time_s)
            if count == 0:
                logging.warning("No trajectory samples were recorded; no file was written.")
                return 0
            self.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.output,
                schema_version=np.array(2, dtype=np.int64),
                model_name=np.array(self.model_name),
                joint_names=np.asarray(self.joint_names),
                started_monotonic_ns=np.array(self.started_at_ns, dtype=np.uint64),
                started_unix_ns=np.array(self.started_unix_ns, dtype=np.uint64),
                time_s=np.asarray(self._time_s, dtype=np.float64),
                monotonic_ns=np.asarray(self._monotonic_ns, dtype=np.uint64),
                unix_ns=np.asarray(self._unix_ns, dtype=np.uint64),
                position=np.asarray(self._position, dtype=np.float64),
                odometry=np.asarray(self._odometry, dtype=np.float64),
                base_command=np.asarray(self._base_command, dtype=np.float64),
                gripper_command=np.asarray(self._gripper_command, dtype=np.float64),
                leader_position=np.asarray(self._leader_position, dtype=np.float64),
            )
        logging.info("Saved %d samples to %s", count, self.output)
        return count


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


def run(args: argparse.Namespace, record_path: Optional[Path] = None) -> int:
    record_cameras = record_path is not None and not args.no_camera_recording
    camera_output = camera_sidecar_path(record_path) if record_cameras else None
    if camera_output is not None and camera_output.exists():
        logging.error("Camera output already exists: %s", camera_output)
        return 2
    if record_cameras or args.visualize_cameras:
        if args.camera_hz <= 0 or args.camera_preview_hz <= 0:
            logging.error("Camera and preview rates must be positive")
            return 2
        if args.camera_max_frame_age_s <= 0 or args.camera_init_timeout_s <= 0:
            logging.error("Camera age and initialization timeout must be positive")
            return 2
        if args.camera_preview_panel_width < 100:
            logging.error("Camera preview panel width must be at least 100")
            return 2
        if not 0 <= args.camera_preview_port <= 65535:
            logging.error("Camera preview port must be between 0 and 65535")
            return 2
        if not 1 <= args.camera_jpeg_quality <= 100:
            logging.error("Camera JPEG quality must be between 1 and 100")
            return 2

    stop_event = threading.Event()
    keyboard = ArrowKeyController(
        args.linear_speed,
        args.angular_speed,
        args.key_timeout,
        stop_event,
        debug=args.keyboard_debug,
    )
    robot = None
    gripper = None
    leader_arm = None
    arm_stream = None
    mobility_stream = None
    recorder = None
    camera_session = None
    state_lock = threading.Lock()
    robot_position = None
    robot_odometry = np.eye(3, dtype=np.float64)
    robot_velocity = None
    robot_target_velocity = None
    robot_is_ready = None
    exit_code = 0

    previous_handlers = {}

    def request_stop(signum=None, frame=None):
        keyboard.set_stop()
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        # 1. Connect to Robot & Initialize Subsystems
        logging.info("Connecting to robot at %s...", args.address)
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            raise RuntimeError(f"Failed to connect to robot at {args.address}")

        model = robot.model()
        if model.model_name != "A" or len(model.mobility_idx) != 2:
            raise RuntimeError(
                f"This example targets the two-wheel A model; connected model is {model.model_name}."
            )

        dyn_model = robot.get_dynamics()
        dyn_state = dyn_model.make_state([], model.robot_joint_names)
        robot_max_q = dyn_model.get_limit_q_upper(dyn_state)
        robot_min_q = dyn_model.get_limit_q_lower(dyn_state)
        robot_max_qdot = dyn_model.get_limit_qdot_upper(dyn_state)
        robot_max_qddot = dyn_model.get_limit_qddot_upper(dyn_state)
        if args.mode == "impedance":
            robot_max_qdot[model.right_arm_idx[-1]] *= 10
            robot_max_qdot[model.left_arm_idx[-1]] *= 10
        position_mode = args.mode == "position"

        if not robot.is_power_on(args.power) and not robot.power_on(args.power):
            raise RuntimeError(f"Failed to power on devices matching {args.power!r}")
        if not robot.is_servo_on(args.servo) and not robot.servo_on(args.servo):
            raise RuntimeError(f"Failed to servo on devices matching {args.servo!r}")
        robot.reset_fault_control_manager()
        if not robot.enable_control_manager():
            raise RuntimeError("Failed to enable the control manager")
        for arm in ("right", "left"):
            if not robot.set_tool_flange_output_voltage(arm, 12):
                raise RuntimeError(f"Failed to enable 12 V tool power on {arm} arm")

        robot.set_parameter("joint_position_command.cutoff_frequency", "3")

        # 2. Start State Update Stream
        def robot_state_callback(state):
            nonlocal robot_position, robot_odometry
            nonlocal robot_velocity, robot_target_velocity, robot_is_ready
            with state_lock:
                robot_position = state.position.copy()
                robot_odometry = state.odometry.copy()
                robot_velocity = state.velocity.copy()
                robot_target_velocity = state.target_velocity.copy()
                robot_is_ready = state.is_ready.copy()

        robot.start_state_update(
            robot_state_callback, 1.0 / Settings.leader_arm_loop_period
        )
        state_deadline = time.monotonic() + 2.0
        while robot_is_ready is None and time.monotonic() < state_deadline:
            time.sleep(0.01)
        if robot_is_ready is None:
            raise RuntimeError("Timed out waiting for the first robot state update")

        # 3. Handle Initial Robot Pose (Current pose by default, or move to READY_POSE if requested)
        if args.move_to_ready:
            logging.info("Moving robot to predefined READY_POSE (5.0 s)...")
            ready_feedback = robot.send_command(
                joint_position_command_builder(
                    READY_POSE[model.model_name], minimum_time=5.0
                ),
                priority=args.priority,
            ).get()
            if ready_feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                raise RuntimeError(
                    f"Failed to move to the ready pose: {ready_feedback.finish_code}"
                )
            time.sleep(0.1)

        with state_lock:
            curr_torso_q = robot_position[model.torso_idx].copy()
            curr_robot_right_q = robot_position[model.right_arm_idx].copy()
            curr_robot_left_q = robot_position[model.left_arm_idx].copy()

        current_robot_pose = leader_example.Pose(
            toros=curr_torso_q,
            right_arm=curr_robot_right_q,
            left_arm=curr_robot_left_q,
        )
        target_leader_q = np.concatenate([curr_robot_right_q, curr_robot_left_q])

        logging.info("=" * 60)
        logging.info("Robot starting pose detected:")
        logging.info("  Right arm (deg): %s", np.round(np.rad2deg(curr_robot_right_q), 2).tolist())
        logging.info("  Left arm  (deg): %s", np.round(np.rad2deg(curr_robot_left_q), 2).tolist())
        logging.info("=" * 60)

        # 4. Initialize Gripper
        gripper = Gripper()
        if not gripper.initialize():
            raise RuntimeError("Failed to initialize the gripper")
        gripper.homing()
        gripper.start()

        # 5. Initialize LeaderArm
        logging.info("Initializing LeaderArm hardware...")
        leader_arm = LeaderArm(control_period=Settings.leader_arm_loop_period)
        leader_arm.initialize(verbose=args.verbose)
        if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
            raise RuntimeError(
                "Leader-arm device count mismatch: "
                f"expected {leader_arm.DEVICE_COUNT}, got {len(leader_arm.active_ids)}"
            )

        # 6. LeaderArm Auto-Homing to Robot's Current Pose
        if not args.skip_autohome:
            logging.info("=" * 60)
            logging.info("Starting LeaderArm Auto-Homing to Robot's Current Pose...")
            success, final_q, errors = auto_home_leader_arm(
                leader_arm=leader_arm,
                target_pose=target_leader_q,
                duration=args.autohome_time,
                tolerance=args.autohome_tolerance,
            )
            if not success:
                logging.warning("LeaderArm Auto-Homing finished with errors exceeding tolerance. Proceeding with caution.")
            else:
                logging.info("LeaderArm Auto-Homing completed successfully!")
            logging.info("=" * 60)
        else:
            logging.info("Skipping LeaderArm Auto-Homing as requested (--skip-autohome).")

        # Set initial reference poses matching the target
        right_q = target_leader_q[0:7].copy()
        left_q = target_leader_q[7:14].copy()

        # 7. Start Cameras (if enabled)
        if record_cameras or args.visualize_cameras:
            camera_session = CameraSessionProcess(
                camera_python=args.camera_python,
                camera_stack_root=args.camera_stack_root,
                output=camera_output,
                camera_hz=args.camera_hz,
                zed_shm_mode=args.zed_shm_mode,
                zed_record_profile=args.zed_record_profile,
                max_frame_age_s=args.camera_max_frame_age_s,
                init_timeout_s=args.camera_init_timeout_s,
                visualize=args.visualize_cameras,
                preview_hz=args.camera_preview_hz,
                panel_width=args.camera_preview_panel_width,
                preview_mode=args.camera_preview_mode,
                preview_host=args.camera_preview_host,
                preview_port=args.camera_preview_port,
                storage_format=args.camera_storage,
                jpeg_quality=args.camera_jpeg_quality,
                compression=args.camera_compression,
            )
            camera_session.start()

        if record_path is not None:
            recorder = TrajectoryRecorder(record_path, model)

        ma_q_limit_barrier = 0.5
        ma_min_q = np.deg2rad(
            [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
        )
        ma_max_q = np.deg2rad(
            [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
        )
        ma_torque_limit = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
        ma_viscous_gain = np.array(
            [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2
        )
        right_minimum_time = 1.0
        left_minimum_time = 1.0
        last_collision_log_time = 0.0
        last_reported_base_command = np.full(3, np.nan, dtype=np.float64)
        last_mobile_send_time = 0.0
        last_mobile_state_log_time = 0.0

        with state_lock:
            mobility_ready = robot_is_ready[model.mobility_idx].copy()
            mobility_velocity = robot_velocity[model.mobility_idx].copy()
        mobility_names = [model.robot_joint_names[i] for i in model.mobility_idx]
        logging.info(
            "Mobility joints: %s, ready=%s, velocity=%s",
            mobility_names,
            mobility_ready.tolist(),
            np.round(mobility_velocity, 4).tolist(),
        )
        if not np.all(mobility_ready):
            raise RuntimeError(
                "Mobility wheel servos are not ready. Check --servo and the wheel drive state."
            )

        arm_stream = robot.create_command_stream(priority=args.priority)
        mobility_stream = robot.create_command_stream(
            priority=args.mobility_priority
        )
        arm_stream.send_command(
            joint_position_command_builder(
                current_robot_pose,
                minimum_time=1.0,
                control_hold_time=1e6,
                position_mode=position_mode,
            )
        )

        teleop_publisher = TeleopStatePublisher()

        def leader_arm_control_loop(state):
            nonlocal right_q, left_q
            nonlocal right_minimum_time, left_minimum_time
            nonlocal last_collision_log_time
            nonlocal last_reported_base_command
            nonlocal last_mobile_send_time, last_mobile_state_log_time

            if stop_event.is_set():
                return None
            if right_q is None:
                right_q = state.q_joint[0:7].copy()
            if left_q is None:
                left_q = state.q_joint[7:14].copy()

            gripper_command = np.array(
                [state.button_right.trigger, state.button_left.trigger],
                dtype=np.float64,
            ) / 1000.0
            gripper.set_target(gripper_command)

            ma_input = LeaderArm.ControlInput()
            torque = (
                state.gravity_term
                + ma_q_limit_barrier
                * (
                    np.maximum(ma_min_q - state.q_joint, 0)
                    + np.minimum(ma_max_q - state.q_joint, 0)
                )
                + ma_viscous_gain * state.qvel_joint
            )
            torque = np.clip(torque, -ma_torque_limit, ma_torque_limit)

            if state.button_right.button == 1:
                ma_input.target_operating_mode[0:7].fill(
                    rby.DynamixelBus.CurrentControlMode
                )
                ma_input.target_torque[0:7] = torque[0:7] * 0.6
                right_q = state.q_joint[0:7].copy()
            else:
                ma_input.target_operating_mode[0:7].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_torque[0:7] = ma_torque_limit[0:7]
                ma_input.target_position[0:7] = right_q

            if state.button_left.button == 1:
                ma_input.target_operating_mode[7:14].fill(
                    rby.DynamixelBus.CurrentControlMode
                )
                ma_input.target_torque[7:14] = torque[7:14] * 0.6
                left_q = state.q_joint[7:14].copy()
            else:
                ma_input.target_operating_mode[7:14].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_torque[7:14] = ma_torque_limit[7:14]
                ma_input.target_position[7:14] = left_q

            with state_lock:
                if robot_position is None:
                    return ma_input
                q = robot_position.copy()
                odometry = robot_odometry.copy()

            q_for_collision = q.copy()
            q_for_collision[model.right_arm_idx] = right_q
            q_for_collision[model.left_arm_idx] = left_q
            dyn_state.set_q(q_for_collision)
            dyn_model.compute_forward_kinematics(dyn_state)
            nearest = dyn_model.detect_collisions_or_nearest_links(dyn_state, 1)[0]
            is_collision = nearest.distance < args.collision_distance
            if is_collision and (state.button_right.button or state.button_left.button):
                now = time.monotonic()
                if now - last_collision_log_time >= 1.0:
                    logging.warning(
                        "Arm command blocked by self-collision limit (%.4f m)",
                        nearest.distance,
                    )
                    last_collision_log_time = now

            body_builder = rby.BodyComponentBasedCommandBuilder()
            has_body_command = False

            if state.button_right.button and not is_collision:
                right_minimum_time = max(
                    right_minimum_time - Settings.leader_arm_loop_period,
                    Settings.leader_arm_loop_period * 1.01,
                )
                right_builder = (
                    rby.JointPositionCommandBuilder()
                    if position_mode
                    else rby.JointImpedanceControlCommandBuilder()
                )
                (
                    right_builder.set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                    )
                    .set_position(
                        np.clip(
                            right_q,
                            robot_min_q[model.right_arm_idx],
                            robot_max_q[model.right_arm_idx],
                        )
                    )
                    .set_velocity_limit(robot_max_qdot[model.right_arm_idx])
                    .set_acceleration_limit(robot_max_qddot[model.right_arm_idx] * 30)
                    .set_minimum_time(right_minimum_time)
                )
                if not position_mode:
                    (
                        right_builder.set_stiffness(
                            [Settings.impedance_stiffness] * len(model.right_arm_idx)
                        )
                        .set_damping_ratio(Settings.impedance_damping_ratio)
                        .set_torque_limit(
                            [Settings.impedance_torque_limit] * len(model.right_arm_idx)
                        )
                    )
                body_builder.set_right_arm_command(right_builder)
                has_body_command = True
            else:
                right_minimum_time = 0.8

            if state.button_left.button and not is_collision:
                left_minimum_time = max(
                    left_minimum_time - Settings.leader_arm_loop_period,
                    Settings.leader_arm_loop_period * 1.01,
                )
                left_builder = (
                    rby.JointPositionCommandBuilder()
                    if position_mode
                    else rby.JointImpedanceControlCommandBuilder()
                )
                (
                    left_builder.set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                    )
                    .set_position(
                        np.clip(
                            left_q,
                            robot_min_q[model.left_arm_idx],
                            robot_max_q[model.left_arm_idx],
                        )
                    )
                    .set_velocity_limit(robot_max_qdot[model.left_arm_idx])
                    .set_acceleration_limit(robot_max_qddot[model.left_arm_idx] * 30)
                    .set_minimum_time(left_minimum_time)
                )
                if not position_mode:
                    (
                        left_builder.set_stiffness(
                            [Settings.impedance_stiffness] * len(model.left_arm_idx)
                        )
                        .set_damping_ratio(Settings.impedance_damping_ratio)
                        .set_torque_limit(
                            [Settings.impedance_torque_limit] * len(model.left_arm_idx)
                        )
                    )
                body_builder.set_left_arm_command(left_builder)
                has_body_command = True
            else:
                left_minimum_time = 0.8

            base_command = keyboard.get_command()
            teleop_publisher.publish(gripper_command, base_command, state.q_joint)
            if has_body_command:
                arm_stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_body_command(
                            body_builder
                        )
                    )
                )

            now = time.monotonic()
            command_changed = not np.array_equal(
                base_command, last_reported_base_command
            )
            refresh_due = now - last_mobile_send_time >= args.mobile_refresh_time
            if command_changed or refresh_due:
                feedback = mobility_stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_mobility_command(
                            _build_mobile_command(base_command, args)
                        )
                    )
                )
                last_mobile_send_time = now

            if command_changed:
                logging.info(
                    "Mobile command sent: vx=%+.3f m/s, wz=%+.3f rad/s "
                    "(status=%s, finish=%s)",
                    base_command[0],
                    base_command[2],
                    feedback.status,
                    feedback.finish_code,
                )
                last_reported_base_command = base_command.copy()

            if np.any(base_command) and now - last_mobile_state_log_time >= 0.5:
                with state_lock:
                    wheel_velocity = robot_velocity[model.mobility_idx].copy()
                    wheel_target = robot_target_velocity[model.mobility_idx].copy()
                    wheel_ready = robot_is_ready[model.mobility_idx].copy()
                logging.info(
                    "Wheel state: ready=%s, velocity=%s rad/s, target=%s rad/s",
                    wheel_ready.tolist(),
                    np.round(wheel_velocity, 4).tolist(),
                    np.round(wheel_target, 4).tolist(),
                )
                last_mobile_state_log_time = now

            if recorder is not None:
                recorder.append(
                    q,
                    odometry,
                    base_command,
                    gripper_command,
                    state.q_joint,
                )
            return ma_input

        def safety_function(state):
            logging.error(
                "Leader-arm safety shutdown; failed IDs: %s",
                sorted(set(state.fault_ids) | set(state.tool_fault_ids)),
            )
            keyboard.set_stop()
            stop_event.set()
            try:
                leader_arm.DisableTorque()
            except Exception:
                pass
            try:
                robot.power_off("12v")
            except Exception:
                pass

        if not leader_arm.start_control(
            leader_arm_control_loop, safety_function=safety_function
        ):
            raise RuntimeError("Failed to start leader-arm control")

        keyboard.start()
        logging.info("=" * 60)
        logging.info("Teleoperation Ready & Synchronized!")
        logging.info("Leader Arm: Hold Right/Left trigger button to move arm")
        logging.info("Gripper: Squeeze trigger to close gripper")
        logging.info("Mobile Base: ↑/W forward, ↓/S backward, ←/A left, →/D right")
        logging.info("Press Space to stop base; press Q or Ctrl+C to exit")
        logging.info("=" * 60)
        if recorder is not None:
            logging.info("Recording to %s", record_path)
            if camera_output is not None:
                logging.info(
                    "Camera recording to %s at %.1f Hz (ZED SHM mode: %s)",
                    camera_output,
                    args.camera_hz,
                    args.zed_shm_mode,
                )

        while not stop_event.is_set() and leader_arm.ctrl_session_active:
            if camera_session is not None:
                camera_session.check_health()
            time.sleep(0.1)

    except Exception:
        logging.exception("Teleoperation failed")
        exit_code = 1
    finally:
        logging.info("Initiating graceful teleoperation shutdown...")
        stop_event.set()

        # 1. Stop keyboard input
        try:
            keyboard.stop()
        except Exception as exc:
            logging.warning("Keyboard/terminal cleanup failed: %s", exc)

        # 2. Stop camera recording session
        if camera_session is not None:
            try:
                camera_session.stop()
            except Exception as exc:
                logging.warning("Camera cleanup failed: %s", exc)

        # 3. Stop mobility base
        _send_base_stop(mobility_stream, args)

        # 4. Gracefully ramp down LeaderArm torques to prevent sudden drop
        if leader_arm is not None:
            try:
                logging.info("Gently ramping down LeaderArm motor torques (1.2 s)...")
                leader_arm.stop_control(torque_disable=False)
                motor_ids = list(range(14))
                steps = 24
                default_torque_limits = np.array([3.5, 3.5, 3.5, 2.5, 1.5, 1.5, 1.5] * 2, dtype=np.float64)
                for alpha in np.linspace(1.0, 0.0, steps):
                    leader_arm.bus.group_sync_write_send_torque([(i, default_torque_limits[i] * alpha) for i in motor_ids])
                    time.sleep(1.2 / steps)
                leader_arm.DisableTorque()
                logging.info("LeaderArm torques safely released.")
            except Exception as exc:
                logging.warning("Leader-arm cleanup failed: %s", exc)
                try:
                    leader_arm.DisableTorque()
                except Exception:
                    pass

        # 5. Stop Gripper
        if gripper is not None:
            try:
                gripper.stop()
            except Exception as exc:
                logging.warning("Gripper cleanup failed: %s", exc)

        # 6. Cleanly cancel robot command streams and active control (preserves Control Manager without Major Fault)
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

        # 7. Save trajectory recording if enabled
        if recorder is not None:
            try:
                recorder.save()
            except Exception:
                logging.exception("Failed to save recording to %s", record_path)
                exit_code = 1

        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

        logging.info("Teleoperation shutdown complete.")

    return exit_code


def create_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--address", required=True, help="Robot address, e.g. 192.168.30.1:50051")
    parser.add_argument("--model", default="a", help="Robot model (this example requires A)")
    parser.add_argument("--power", default=".*", help="Power-device regex")
    parser.add_argument(
        "--servo",
        default=".*",
        help="Servo regex; default includes the wheel servos",
    )
    parser.add_argument(
        "--mode",
        choices=("position", "impedance"),
        default="position",
        help="Robot arm control mode",
    )
    parser.add_argument("--linear-speed", type=float, default=0.10, help="Forward/backward speed [m/s]")
    parser.add_argument("--angular-speed", type=float, default=0.30, help="Turn speed [rad/s]")
    parser.add_argument("--linear-acceleration", type=float, default=0.50, help="Base linear acceleration limit [m/s^2]")
    parser.add_argument("--angular-acceleration", type=float, default=1.0, help="Base angular acceleration limit [rad/s^2]")
    parser.add_argument(
        "--key-timeout",
        type=float,
        default=0.35,
        help="Stop if an arrow-key repeat is not received within this time [s]",
    )
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
        help="Minimum velocity-profile ramp time [s]",
    )
    parser.add_argument(
        "--mobile-refresh-time",
        type=float,
        default=0.40,
        help="Interval for refreshing an unchanged mobility command [s]",
    )
    parser.add_argument("--collision-distance", type=float, default=0.02, help="Arm self-collision command block distance [m]")
    parser.add_argument("--priority", type=int, default=1, help="Robot command priority")
    parser.add_argument(
        "--mobility-priority",
        type=int,
        default=10,
        help="Mobility command priority (matches example 37)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable leader-arm device logs")
    parser.add_argument(
        "--keyboard-debug",
        action="store_true",
        help="Print raw terminal bytes to diagnose keyboard input",
    )
    # Auto-Homing Arguments
    parser.add_argument(
        "--move-to-ready",
        action="store_true",
        help="Move robot to predefined READY_POSE before starting (by default, robot stays at its current pose)",
    )
    parser.add_argument(
        "--autohome-time",
        type=float,
        default=4.0,
        help="Leader-arm auto-homing S-curve duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--autohome-current",
        type=float,
        default=1.0,
        help="Leader-arm auto-homing torque/current scale (default: 1.0 = standard limits: 3.5 Nm shoulder, 2.5 Nm elbow, 1.5 Nm wrist)",
    )
    parser.add_argument(
        "--autohome-tolerance",
        type=float,
        default=0.05,
        help="Leader-arm auto-homing convergence tolerance in radians (default: 0.05 rad ≈ 2.86 deg)",
    )
    parser.add_argument(
        "--skip-autohome",
        action="store_true",
        help="Skip leader-arm auto-homing step before starting teleoperation",
    )
    # Camera arguments
    parser.add_argument(
        "--camera-stack-root",
        type=Path,
        default=DEFAULT_CAMERA_STACK_ROOT,
        help="Repository root containing camera_stack",
    )
    parser.add_argument(
        "--camera-python",
        type=Path,
        default=DEFAULT_CAMERA_PYTHON,
        help="Python environment containing h5py and OpenCV",
    )
    parser.add_argument(
        "--camera-hz",
        type=float,
        default=30.0,
        help="Camera recording rate [Hz]",
    )
    parser.add_argument(
        "--zed-shm-mode",
        choices=ZED_SHM_MODES,
        default="rgb-only",
        help=(
            "Must match run_cameras.sh --zed-mode; default is the 30 Hz "
            "RGB-only collection mode"
        ),
    )
    parser.add_argument(
        "--zed-record-profile",
        choices=tuple(ZED_RECORD_PROFILES),
        default=DEFAULT_ZED_RECORD_PROFILE,
        help=(
            "Head ZED HDF5 resolution: full=1920x1200, "
            "record=960x600, fast=640x400 (stereo-rgbd only)"
        ),
    )
    parser.add_argument(
        "--camera-max-frame-age-s",
        type=float,
        default=1.0,
        help="Reject SHM frames older than this age [s]",
    )
    parser.add_argument(
        "--camera-init-timeout-s",
        type=float,
        default=30.0,
        help="Wait this long for all three SHM cameras [s]",
    )
    parser.add_argument(
        "--camera-storage",
        choices=("jpeg", "raw"),
        default="jpeg",
        help="HDF5 image storage; JPEG is recommended for 30 Hz recording",
    )
    parser.add_argument(
        "--camera-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality used by the default camera storage",
    )
    parser.add_argument(
        "--camera-compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="HDF5 compression used only with --camera-storage raw",
    )
    parser.add_argument(
        "--visualize-cameras",
        action="store_true",
        help="Show the three live camera images in an OpenCV window",
    )
    parser.add_argument(
        "--camera-preview-hz",
        type=float,
        default=10.0,
        help="Live camera preview refresh rate [Hz]",
    )
    parser.add_argument(
        "--camera-preview-panel-width",
        type=int,
        default=400,
        help="Width of each preview panel [pixels]",
    )
    parser.add_argument(
        "--camera-preview-mode",
        choices=("auto", "window", "browser"),
        default="auto",
        help="Preview backend; auto falls back to a browser server",
    )
    parser.add_argument(
        "--camera-preview-host",
        default="127.0.0.1",
        help="Browser preview bind address",
    )
    parser.add_argument(
        "--camera-preview-port",
        type=int,
        default=8765,
        help="Browser preview port (0 chooses an available port)",
    )
    parser.add_argument(
        "--no-camera-recording",
        action="store_true",
        help="With record.py, record robot data only",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(create_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
