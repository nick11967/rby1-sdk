#!/usr/bin/env python3
"""Leader-arm teleoperation with Object Grasp Detection via 6-Axis Force/Torque Sensors.

This script extends teleop_autohome to monitor the wrist 6-axis FT sensors (right & left)
in real time. It enables:
1. Powering the tool flange (12V) so the robot gripper initializes, homes, and opens.
2. Auto-homing the LeaderArm to prevent trajectory snaps.
3. Auto-taring the FT sensors at startup (with an on-the-fly 't' key re-tare option).
4. Real-time Grasp Detection: Checks if the trigger is squeezed AND external contact force
   exceeds the threshold to confirm that an object is grasped.
5. Real-time visual status HUD in terminal.

Usage:
    python teleop_grasp_test.py --address 192.168.30.1:50051 --model a
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
from typing import Optional, Tuple

import numpy as np
import rby1_sdk as rby

# Resolve path hierarchy
SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
LEADER_DIR = EXAMPLES_DIR / "leader_arm_mobile_record_replay"

for p in [str(EXAMPLES_DIR), str(LEADER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import drivers and utilities from SDK examples
leader_example = importlib.import_module("35_leader_arm_teleop_with_monitor")
LeaderArm = leader_example.LeaderArm
READY_POSE = leader_example.READY_POSE
Settings = leader_example.Settings
joint_position_command_builder = leader_example.joint_position_command_builder


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


Gripper = EnhancedGripper

HOMING_PID_P = 2800
HOMING_PID_I = 300
HOMING_PID_D = 4000
NUM_LEADER_MOTORS = 14


def s_curve_quintic(t: float, total_time: float) -> float:
    """5th-order polynomial S-curve for minimum-jerk trajectory."""
    if total_time <= 0:
        return 1.0
    tau = np.clip(t / total_time, 0.0, 1.0)
    return float(10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)


def read_joint_positions(bus: rby.DynamixelBus, motor_ids: list[int]) -> np.ndarray:
    ms_list = bus.group_fast_sync_read_encoder(motor_ids)
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
    """Read arrow keys for mobile base and 't' for FT sensor tare."""

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
    ):
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.timeout_s = timeout_s
        self.stop_event = stop_event
        self._command = np.zeros(3, dtype=np.float64)
        self._expires_at = 0.0
        self._active_label = "STOP"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._old_termios = None
        self.tare_requested = False

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        new_termios = termios.tcgetattr(self._fd)
        new_termios[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(self._fd, termios.TCSANOW, new_termios)
        self._thread = threading.Thread(target=self._run, name="keyboard-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.set_stop()
        if self._fd is not None and self._old_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        self._fd = None
        self._old_termios = None

    def set_stop(self) -> None:
        with self._lock:
            self._command.fill(0.0)
            self._expires_at = 0.0
            self._active_label = "STOP"

    def get_command(self) -> np.ndarray:
        with self._lock:
            if time.monotonic() >= self._expires_at:
                self._command.fill(0.0)
                self._active_label = "STOP"
            return self._command.copy()

    def check_tare(self) -> bool:
        with self._lock:
            if self.tare_requested:
                self.tare_requested = False
                return True
            return False

    def _set_direction(self, linear_sign: float, angular_sign: float, label: str) -> None:
        with self._lock:
            self._command[:] = [
                linear_sign * self.linear_speed,
                0.0,
                angular_sign * self.angular_speed,
            ]
            self._expires_at = time.monotonic() + self.timeout_s
            self._active_label = label

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

            if b"q" in pending.lower():
                self.set_stop()
                self.stop_event.set()
                break
            if b" " in pending:
                self.set_stop()
            if b"t" in pending.lower():
                with self._lock:
                    self.tare_requested = True

            matched = False
            for seq, (lin, ang, label) in self._DIRECTIONS.items():
                if seq in pending:
                    self._set_direction(lin, ang, label)
                    matched = True
                    break
            if matched or len(pending) > 8:
                pending = pending[-2:]


def _build_mobile_command(base_command: np.ndarray, args: argparse.Namespace):
    return (
        rby.SE2VelocityCommandBuilder()
        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(args.mobile_hold_time))
        .set_minimum_time(args.mobile_ramp_time)
        .set_acceleration_limit(
            np.array([args.linear_acceleration, args.linear_acceleration]),
            float(args.angular_acceleration),
        )
        .set_velocity(base_command[:2], float(base_command[2]))
    )


def run_teleop(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stop_event = threading.Event()
    previous_handlers = {}

    def _signal_handler(signum, frame):
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, _signal_handler)

    robot = None
    leader_arm = None
    gripper = None
    keyboard = ArrowKeyController(
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
        timeout_s=args.key_timeout,
        stop_event=stop_event,
    )

    exit_code = 0
    try:
        logging.info("Connecting to RB-Y1 robot at %s...", args.address)
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            raise RuntimeError(f"Failed to connect to robot at {args.address}")
        model = robot.model()
        dyn_model = robot.get_dynamics()
        dyn_state = dyn_model.make_state()
        position_mode = args.mode == "position"

        if not robot.is_power_on(args.power) and not robot.power_on(args.power):
            raise RuntimeError("Failed to power on")
        if not robot.is_servo_on(args.servo) and not robot.servo_on(args.servo):
            raise RuntimeError("Failed to servo on")
        robot.reset_fault_control_manager()
        if not robot.enable_control_manager():
            raise RuntimeError("Failed to enable the control manager")

        # 1. Power on tool flange (12V) so Gripper Dynamixels have power
        logging.info("Enabling 12V tool flange power on right and left wrists...")
        for arm in ("right", "left"):
            if not robot.set_tool_flange_output_voltage(arm, 12):
                raise RuntimeError(f"Failed to enable 12 V tool power on {arm} arm")

        robot.set_parameter("joint_position_command.cutoff_frequency", "3")

        # 2. State update stream & FT sensor storage
        state_lock = threading.Lock()
        robot_position = None
        robot_velocity = None
        robot_odometry = None
        robot_is_ready = None
        robot_ft_right = np.zeros(3)
        robot_ft_left = np.zeros(3)
        robot_tau_right = np.zeros(3)
        robot_tau_left = np.zeros(3)

        def robot_state_callback(state):
            nonlocal robot_position, robot_odometry, robot_velocity, robot_is_ready
            nonlocal robot_ft_right, robot_ft_left, robot_tau_right, robot_tau_left
            with state_lock:
                robot_position = state.position.copy()
                robot_odometry = state.odometry.copy()
                robot_velocity = state.velocity.copy()
                robot_is_ready = state.is_ready.copy()
                robot_ft_right = state.ft_sensor_right.force.copy()
                robot_tau_right = state.ft_sensor_right.torque.copy()
                robot_ft_left = state.ft_sensor_left.force.copy()
                robot_tau_left = state.ft_sensor_left.torque.copy()

        robot.start_state_update(robot_state_callback, 1.0 / Settings.leader_arm_loop_period)

        state_deadline = time.monotonic() + 2.0
        while robot_is_ready is None and time.monotonic() < state_deadline:
            time.sleep(0.01)
        if robot_is_ready is None:
            raise RuntimeError("Timed out waiting for robot state update")

        # 3. Optional move to READY_POSE
        if args.move_to_ready:
            logging.info("Moving robot to READY_POSE (5.0s)...")
            ready_feedback = robot.send_command(
                joint_position_command_builder(READY_POSE[model.model_name], minimum_time=5.0),
                priority=args.priority,
            ).get()
            if ready_feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                raise RuntimeError(f"Failed to move to ready pose: {ready_feedback.finish_code}")
            time.sleep(0.1)

        # 4. Initialize and Home Gripper (Hand opens!)
        logging.info("Initializing gripper...")
        gripper = Gripper()
        if not gripper.initialize():
            raise RuntimeError("Failed to initialize gripper")
        logging.info("Homing gripper (calibrating travel limits and opening fingers)...")
        gripper.homing()
        gripper.start()
        gripper.set_target(np.array([0.0, 0.0]))  # Open hands
        logging.info("Gripper ready and open!")

        # 5. Measure resting FT sensor baseline (Tare)
        logging.info("Calibrating FT sensor resting baseline (tare) for 1.0s...")
        tare_r, tare_l = [], []
        for _ in range(30):
            with state_lock:
                tare_r.append(robot_ft_right.copy())
                tare_l.append(robot_ft_left.copy())
            time.sleep(0.03)
        baseline_ft_right = np.mean(tare_r, axis=0)
        baseline_ft_left = np.mean(tare_l, axis=0)
        logging.info(
            "FT Baseline calibrated:\n  Right: %s N (magnitude %.1f N)\n  Left:  %s N (magnitude %.1f N)",
            np.round(baseline_ft_right, 2),
            np.linalg.norm(baseline_ft_right),
            np.round(baseline_ft_left, 2),
            np.linalg.norm(baseline_ft_left),
        )

        # 6. Initialize Leader Arm & Auto-Home
        logging.info("Initializing LeaderArm driver...")
        leader_arm = LeaderArm(dev_name=rby.upc.resolve_leader_arm_device_name())
        if not leader_arm.initialize():
            raise RuntimeError("Failed to initialize LeaderArm")

        with state_lock:
            robot_q_ready = robot_position.copy()

        target_leader_q = np.concatenate([
            robot_q_ready[model.right_arm_idx],
            robot_q_ready[model.left_arm_idx],
        ])

        if not args.skip_autohome:
            logging.info("Auto-homing LeaderArm to current robot pose (%.1f s)...", args.autohome_time)
            success, q_final, errors = auto_home_leader_arm(
                leader_arm, target_leader_q, duration=args.autohome_time, tolerance=args.autohome_tolerance
            )
            if not success:
                max_err = float(np.max(errors))
                logging.warning("Leader-arm autohome had max error %.3f rad (tolerance=%.3f)", max_err, args.autohome_tolerance)
            else:
                logging.info("LeaderArm successfully auto-homed!")

        # 7. Start Teleoperation Loop
        arm_stream = robot.create_command_stream(priority=args.priority)
        mobility_stream = robot.create_command_stream(priority=args.mobility_priority)

        right_q = None
        left_q = None
        right_minimum_time = 0.8
        left_minimum_time = 0.8
        last_hud_log_time = 0.0
        last_mobile_send_time = 0.0
        last_reported_base_command = np.zeros(3)

        def leader_arm_control_loop(state):
            nonlocal right_q, left_q, right_minimum_time, left_minimum_time
            nonlocal last_hud_log_time, last_mobile_send_time, last_reported_base_command
            nonlocal baseline_ft_right, baseline_ft_left

            if stop_event.is_set():
                return None

            if right_q is None:
                right_q = state.q_joint[0:7].copy()
            if left_q is None:
                left_q = state.q_joint[7:14].copy()

            # Gripper control (squeezing trigger closes gripper)
            gripper_command = np.array(
                [state.button_right.trigger, state.button_left.trigger],
                dtype=np.float64,
            ) / 1000.0
            gripper.set_target(gripper_command)

            # Arm teleoperation input
            ma_input = LeaderArm.ControlInput()
            torque = (
                state.gravity_term
                + Settings.ma_viscous_gain * state.qvel_joint
            )
            torque = np.clip(torque, -Settings.ma_torque_limit, Settings.ma_torque_limit)

            if state.button_right.button == 1:
                ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentControlMode)
                ma_input.target_torque[0:7] = torque[0:7] * 0.6
                right_q = state.q_joint[0:7].copy()
            else:
                ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                ma_input.target_torque[0:7] = Settings.ma_torque_limit[0:7]
                ma_input.target_position[0:7] = right_q

            if state.button_left.button == 1:
                ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentControlMode)
                ma_input.target_torque[7:14] = torque[7:14] * 0.6
                left_q = state.q_joint[7:14].copy()
            else:
                ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                ma_input.target_torque[7:14] = Settings.ma_torque_limit[7:14]
                ma_input.target_position[7:14] = left_q

            # Send arm command
            body_builder = rby.BodyComponentBasedCommandBuilder()
            has_body_command = False

            if state.button_right.button == 1:
                right_builder = (
                    rby.JointPositionCommandBuilder()
                    if position_mode
                    else rby.JointImpedanceControlCommandBuilder()
                )
                (
                    right_builder.set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.04))
                    .set_position(right_q)
                    .set_velocity_limit(robot.model().joint_velocity_limit[model.right_arm_idx])
                    .set_acceleration_limit(robot.model().joint_acceleration_limit[model.right_arm_idx] * 30)
                    .set_minimum_time(right_minimum_time)
                )
                if not position_mode:
                    (
                        right_builder.set_stiffness([Settings.impedance_stiffness] * len(model.right_arm_idx))
                        .set_damping_ratio(Settings.impedance_damping_ratio)
                        .set_torque_limit([Settings.impedance_torque_limit] * len(model.right_arm_idx))
                    )
                body_builder.set_right_arm_command(right_builder)
                has_body_command = True
            else:
                right_minimum_time = 0.8

            if state.button_left.button == 1:
                left_builder = (
                    rby.JointPositionCommandBuilder()
                    if position_mode
                    else rby.JointImpedanceControlCommandBuilder()
                )
                (
                    left_builder.set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.04))
                    .set_position(left_q)
                    .set_velocity_limit(robot.model().joint_velocity_limit[model.left_arm_idx])
                    .set_acceleration_limit(robot.model().joint_acceleration_limit[model.left_arm_idx] * 30)
                    .set_minimum_time(left_minimum_time)
                )
                if not position_mode:
                    (
                        left_builder.set_stiffness([Settings.impedance_stiffness] * len(model.left_arm_idx))
                        .set_damping_ratio(Settings.impedance_damping_ratio)
                        .set_torque_limit([Settings.impedance_torque_limit] * len(model.left_arm_idx))
                    )
                body_builder.set_left_arm_command(left_builder)
                has_body_command = True
            else:
                left_minimum_time = 0.8

            if has_body_command:
                arm_stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_body_command(body_builder)
                    )
                )

            # Mobile base command
            base_command = keyboard.get_command()
            now = time.monotonic()
            if not np.array_equal(base_command, last_reported_base_command) or (now - last_mobile_send_time >= args.mobile_refresh_time):
                mobility_stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder().set_mobility_command(
                            _build_mobile_command(base_command, args)
                        )
                    )
                )
                last_mobile_send_time = now
                last_reported_base_command = base_command.copy()

            # Check if keyboard requested re-tare
            if keyboard.check_tare():
                with state_lock:
                    baseline_ft_right = robot_ft_right.copy()
                    baseline_ft_left = robot_ft_left.copy()
                logging.info("FT sensors RE-TARED to current wrist orientation!")

            # Grasp & Force/Torque Detection HUD
            if now - last_hud_log_time >= args.hud_interval:
                with state_lock:
                    cur_ft_r = robot_ft_right.copy()
                    cur_ft_l = robot_ft_left.copy()

                df_r = np.linalg.norm(cur_ft_r - baseline_ft_right)
                df_l = np.linalg.norm(cur_ft_l - baseline_ft_left)

                trig_r = gripper_command[0]
                trig_l = gripper_command[1]

                # Approach A: Check actual finger position stall vs commanded trigger
                act_closed = gripper.get_closed_ratio()
                r_act = act_closed[0]
                l_act = act_closed[1]

                stall_r = trig_r - r_act
                stall_l = trig_l - l_act

                # Grasp confirmed if trigger is closing (>0.35) and fingers stalled on object (<0.88 closed)
                is_grasping_r = (trig_r > 0.35) and (r_act < 0.88) and (stall_r > 0.12)
                is_grasping_l = (trig_l > 0.35) and (l_act < 0.88) and (stall_l > 0.12)

                if is_grasping_r:
                    status_r = f"*** GRASPED ({int((1-r_act)*100)}% wide) ***"
                elif trig_r > 0.35 and r_act >= 0.88:
                    status_r = "CLOSED / EMPTY"
                elif trig_r > 0.15:
                    status_r = "CLOSING"
                else:
                    status_r = "OPEN"

                if is_grasping_l:
                    status_l = f"*** GRASPED ({int((1-l_act)*100)}% wide) ***"
                elif trig_l > 0.35 and l_act >= 0.88:
                    status_l = "CLOSED / EMPTY"
                elif trig_l > 0.15:
                    status_l = "CLOSING"
                else:
                    status_l = "OPEN"

                contact_r = f" [Table Contact: {df_r:.1f}N]" if df_r >= args.grasp_threshold else ""
                contact_l = f" [Table Contact: {df_l:.1f}N]" if df_l >= args.grasp_threshold else ""

                logging.info(
                    "[GRASP HUD] R: Trig:%3.0f%% Pos:%3.0f%% [%-22s] ΔF=%4.1fN%s",
                    trig_r * 100, r_act * 100, status_r, df_r, contact_r,
                )
                last_hud_log_time = now

            return ma_input

        def safety_function(state):
            logging.error("Leader-arm safety shutdown triggered!")
            keyboard.set_stop()
            stop_event.set()
            try:
                leader_arm.DisableTorque()
            except Exception:
                pass

        if not leader_arm.start_control(leader_arm_control_loop, safety_function=safety_function):
            raise RuntimeError("Failed to start leader-arm control")

        keyboard.start()
        logging.info("=" * 65)
        logging.info("  TELEOPERATION & OBJECT GRASP DETECTION ACTIVE")
        logging.info("  - LeaderArm: Hold button on arm to drive robot arm")
        logging.info("  - Gripper:   Squeeze trigger to grasp object")
        logging.info("  - Re-Tare:   Press 'T' on keyboard to zero force baseline")
        logging.info("  - Mobile:    W/A/S/D or Arrow keys (Space = Stop)")
        logging.info("  - Quit:      Press 'Q' or Ctrl+C to exit")
        logging.info("=" * 65)

        while not stop_event.is_set() and leader_arm.ctrl_session_active:
            time.sleep(0.1)

    except Exception:
        logging.exception("Teleoperation failed")
        exit_code = 1
    finally:
        logging.info("Initiating graceful shutdown...")
        stop_event.set()
        keyboard.stop()

        if leader_arm is not None:
            try:
                leader_arm.stop_control(torque_disable=True)
            except Exception:
                pass

        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass

        if robot is not None:
            try:
                robot.stop_state_update()
                robot.cancel_control()
                # Safely turn off tool flange power
                robot.set_tool_flange_output_voltage("right", 0)
                robot.set_tool_flange_output_voltage("left", 0)
                robot.disconnect()
            except Exception:
                pass

        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

        logging.info("Shutdown complete.")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Teleop Grasp & FT Sensor Test", allow_abbrev=False)
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot IP:port")
    parser.add_argument("--model", default="a", help="Robot model name (default: a)")
    parser.add_argument(
        "--mode",
        choices=("position", "impedance"),
        default="position",
        help="Robot arm control mode (default: position)",
    )
    parser.add_argument("--power", default=".*", help="Power device regex")
    parser.add_argument("--servo", default=".*", help="Servo regex")
    parser.add_argument("--priority", type=int, default=1, help="Arm command priority")
    parser.add_argument("--mobility-priority", type=int, default=10, help="Mobility priority")
    parser.add_argument("--linear-speed", type=float, default=0.10, help="Base linear speed [m/s]")
    parser.add_argument("--angular-speed", type=float, default=0.30, help="Base turn speed [rad/s]")
    parser.add_argument("--linear-acceleration", type=float, default=0.50, help="Base linear accel [m/s^2]")
    parser.add_argument("--angular-acceleration", type=float, default=1.0, help="Base angular accel [rad/s^2]")
    parser.add_argument("--key-timeout", type=float, default=0.35, help="Arrow key timeout [s]")
    parser.add_argument("--mobile-hold-time", type=float, default=0.75, help="Mobile hold time [s]")
    parser.add_argument("--mobile-ramp-time", type=float, default=0.10, help="Mobile ramp time [s]")
    parser.add_argument("--mobile-refresh-time", type=float, default=0.40, help="Mobile refresh time [s]")
    parser.add_argument("--move-to-ready", action="store_true", help="Move to READY_POSE at startup")
    parser.add_argument("--autohome-time", type=float, default=4.0, help="Autohome S-curve duration [s]")
    parser.add_argument("--autohome-tolerance", type=float, default=0.05, help="Autohome tolerance [rad]")
    parser.add_argument("--skip-autohome", action="store_true", help="Skip leader-arm autohome")
    parser.add_argument("--grasp-threshold", type=float, default=3.0, help="Force delta threshold in N to confirm grasp (default: 3.0)")
    parser.add_argument("--hud-interval", type=float, default=0.3, help="HUD log interval in seconds (default: 0.3)")

    args = parser.parse_args()
    return run_teleop(args)


if __name__ == "__main__":
    sys.exit(main())
