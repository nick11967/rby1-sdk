#!/usr/bin/env python3
"""Interactive Multi-Episode Data Collection Session for RB-Y1.

Seamlessly integrates:
1. One-shot auto-reset via sample_and_move_to_ready.py (isolated venv subprocess)
2. Smooth LeaderArm Auto-Homing to the perturbed ready pose
3. Long-lived Teleoperation and Multi-Camera recording session (zero restart latency)
4. Terminal interactive state machine:
     [s] : Start recording current episode (kinematics NPZ + camera H5)
     [e] : Stop recording -> save episode -> auto-release gripper
     [c] : Continue to next episode (auto-reset -> auto-home -> standby)
     [d] : Discard current/recent episode
     [q] : Safe shutdown (LeaderArm torque ramp-down)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import logging
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Dict, List, Optional, Tuple

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
Settings = leader_example.Settings
joint_position_command_builder = leader_example.joint_position_command_builder

# Teleoperation and record session constants

DEFAULT_RESET_PYTHON = Path.home() / "arpa_h_demo_robot_side" / ".venv" / "bin" / "python"
DEFAULT_RESET_SCRIPT = Path.home() / "arpa_h_demo_robot_side" / "sample_and_move_to_ready.py"


def find_next_episode_idx(output_dir: Path, prefix: str) -> int:
    """Find the next unused episode index in output_dir."""
    if not output_dir.exists():
        return 0
    existing = list(output_dir.glob(f"{prefix}_*"))
    max_idx = -1
    for p in existing:
        name = p.name
        if not name.startswith(f"{prefix}_"):
            continue
        try:
            suffix_part = name[len(prefix) + 1 :]
            idx_str = suffix_part.split(".")[0]
            idx = int(idx_str)
            if idx > max_idx:
                max_idx = idx
        except (ValueError, IndexError):
            pass
    return max_idx + 1


class UnifiedTerminalController:
    """Handles both base mobility (arrow keys) and episode FSM (s, e, c, d, q) in a single TTY listener."""

    _ARROW_DIRECTIONS = {
        b"\x1b[A": (1.0, 0.0, "FORWARD"),
        b"\x1bOA": (1.0, 0.0, "FORWARD"),
        b"\x1b[B": (-1.0, 0.0, "BACKWARD"),
        b"\x1bOB": (-1.0, 0.0, "BACKWARD"),
        b"\x1b[D": (0.0, -1.0, "TURN LEFT"),
        b"\x1bOD": (0.0, -1.0, "TURN LEFT"),
        b"\x1b[C": (0.0, 1.0, "TURN RIGHT"),
        b"\x1bOC": (0.0, 1.0, "TURN RIGHT"),
    }

    def __init__(
        self,
        key_queue: queue.Queue[str],
        stop_event: threading.Event,
        linear_speed: float = 0.10,
        angular_speed: float = 0.30,
        timeout_s: float = 0.25,
    ) -> None:
        self.key_queue = key_queue
        self.stop_event = stop_event
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.timeout_s = timeout_s

        self._lock = threading.Lock()
        self._command = np.zeros(3, dtype=np.float64)  # vx, vy, wz
        self._expires_at = 0.0
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._old_termios = None
        self._active_label = "STOP"

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive session requires a TTY terminal.")
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="unified-tty-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.set_stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._fd is not None and self._old_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
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
                if np.any(self._command):
                    self._command.fill(0.0)
                    self._active_label = "STOP"
            return self._command.copy()

    def _set_direction(self, linear_sign: float, angular_sign: float, label: str) -> None:
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

            if 3 in data:  # Ctrl+C
                self.set_stop()
                self.stop_event.set()
                break

            pending += data

            # Space stops base
            if b" " in pending:
                self.set_stop()
                pending = pending.replace(b" ", b"")

            # Check for arrow key sequences
            consumed_direction = False
            for seq, (lin, ang, label) in self._ARROW_DIRECTIONS.items():
                if seq in pending:
                    self._set_direction(lin, ang, label)
                    pending = pending.replace(seq, b"")
                    consumed_direction = True

            # Process single character action keys
            remaining = b""
            i = 0
            while i < len(pending):
                if pending[i : i + 1] == b"\x1b" or pending[i : i + 2] in (b"\x1b[", b"\x1bO"):
                    remaining = pending[i:]
                    break
                byte_val = pending[i]
                char = chr(byte_val).lower()
                if char in ("s", "e", "c", "d", "q", "r"):
                    self.key_queue.put(char)
                i += 1

            if consumed_direction and len(remaining) > 3:
                pending = b""
            else:
                pending = remaining


class EpisodeTrajectoryRecorder:
    """Collects synchronized robot state, leader positions, and commands for one episode."""

    def __init__(self, model) -> None:
        self.model_name = model.model_name
        self.joint_names = list(model.robot_joint_names)
        self._lock = threading.Lock()
        self.is_recording = False
        self.started_at_ns = 0
        self.started_unix_ns = 0
        self._wall_minus_monotonic_ns = 0

        self._time_s: List[float] = []
        self._monotonic_ns: List[int] = []
        self._unix_ns: List[int] = []
        self._position: List[np.ndarray] = []
        self._velocity: List[np.ndarray] = []
        self._odometry: List[np.ndarray] = []
        self._base_command: List[np.ndarray] = []
        self._gripper_command: List[np.ndarray] = []
        self._leader_position: List[np.ndarray] = []

    def start(self) -> None:
        with self._lock:
            self.started_at_ns = time.monotonic_ns()
            self.started_unix_ns = time.time_ns()
            self._wall_minus_monotonic_ns = self.started_unix_ns - self.started_at_ns
            self._time_s.clear()
            self._monotonic_ns.clear()
            self._unix_ns.clear()
            self._position.clear()
            self._velocity.clear()
            self._odometry.clear()
            self._base_command.clear()
            self._gripper_command.clear()
            self._leader_position.clear()
            self.is_recording = True

    def append(
        self,
        robot_position: np.ndarray,
        robot_velocity: Optional[np.ndarray],
        odometry: np.ndarray,
        base_command: np.ndarray,
        gripper_command: np.ndarray,
        leader_position: np.ndarray,
    ) -> None:
        if not self.is_recording:
            return
        now_ns = time.monotonic_ns()
        with self._lock:
            if not self.is_recording:
                return
            self._time_s.append((now_ns - self.started_at_ns) / 1e9)
            self._monotonic_ns.append(now_ns)
            self._unix_ns.append(now_ns + self._wall_minus_monotonic_ns)
            self._position.append(robot_position.copy())
            if robot_velocity is not None:
                self._velocity.append(robot_velocity.copy())
            self._odometry.append(odometry.copy())
            self._base_command.append(base_command.copy())
            self._gripper_command.append(gripper_command.copy())
            self._leader_position.append(leader_position.copy())

    def stop_and_save(self, output_path: Path) -> int:
        with self._lock:
            self.is_recording = False
            count = len(self._time_s)
            if count == 0:
                logging.warning("No samples collected; file not saved.")
                return 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            kwargs = {
                "schema_version": np.array(2, dtype=np.int64),
                "model_name": np.array(self.model_name),
                "joint_names": np.asarray(self.joint_names),
                "started_monotonic_ns": np.array(self.started_at_ns, dtype=np.uint64),
                "started_unix_ns": np.array(self.started_unix_ns, dtype=np.uint64),
                "time_s": np.asarray(self._time_s, dtype=np.float64),
                "monotonic_ns": np.asarray(self._monotonic_ns, dtype=np.uint64),
                "unix_ns": np.asarray(self._unix_ns, dtype=np.uint64),
                "position": np.asarray(self._position, dtype=np.float64),
                "odometry": np.asarray(self._odometry, dtype=np.float64),
                "base_command": np.asarray(self._base_command, dtype=np.float64),
                "gripper_command": np.asarray(self._gripper_command, dtype=np.float64),
                "leader_position": np.asarray(self._leader_position, dtype=np.float64),
            }
            if self._velocity:
                kwargs["velocity"] = np.asarray(self._velocity, dtype=np.float64)

            np.savez_compressed(output_path, **kwargs)
            return count

    def discard(self) -> None:
        with self._lock:
            self.is_recording = False
            self._time_s.clear()
            self._monotonic_ns.clear()
            self._unix_ns.clear()
            self._position.clear()
            self._velocity.clear()
            self._odometry.clear()
            self._base_command.clear()
            self._gripper_command.clear()
            self._leader_position.clear()


def execute_auto_reset(args: argparse.Namespace) -> bool:
    """Run sample_and_move_to_ready.py via subprocess with isolated venv."""
    python_bin = Path(args.reset_python).expanduser()
    script_path = Path(args.reset_script).expanduser()
    if not python_bin.is_file():
        logging.error("Reset python executable not found: %s", python_bin)
        return False
    if not script_path.is_file():
        logging.error("Reset script not found: %s", script_path)
        return False

    cmd = [
        str(python_bin),
        str(script_path),
        "--address", args.address,
        "--model", args.model,
        "--noise-deg", str(args.noise_deg),
        "--noise-mode", args.noise_mode,
        "--minimum-time", str(args.reset_time),
        "--yes",
    ]
    logging.info("[RESET] Executing: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logging.info("[RESET] Auto-reset to perturbed ready pose successful.")
        time.sleep(0.5)  # Required safe interval for robot SDK session release
        return True
    except subprocess.CalledProcessError as exc:
        logging.error("[RESET] Auto-reset failed with exit code %d", exc.returncode)
        return False
    except Exception as exc:
        logging.error("[RESET] Exception during auto-reset: %s", exc)
        return False


def release_gripper_fully(gripper: Gripper) -> None:
    """Command gripper to open position and wait briefly."""
    logging.info("[GRIPPER] Auto-releasing gripper (fully open)...")
    try:
        gripper.set_target(np.array([0.0, 0.0], dtype=np.float64))
        time.sleep(0.3)
    except Exception as exc:
        logging.warning("[GRIPPER] Failed to release gripper: %s", exc)


DEFAULT_SERVO_PATTERN = "torso_.*|right_arm_.*|left_arm_.*|head_.*"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="192.168.30.1:50051", help="RB-Y1 address (default: %(default)s)")
    parser.add_argument("--model", default="a", help="RB-Y1 model (default: %(default)s)")
    parser.add_argument("--power", default=".*", help="Power pattern (default: %(default)s)")
    parser.add_argument("--servo", default=DEFAULT_SERVO_PATTERN, help="Servo pattern (default: upper-body servos)")
    parser.add_argument("--priority", type=int, default=1, help="Arm stream priority (default: %(default)s)")
    parser.add_argument("--mobility-priority", type=int, default=1, help="Mobility stream priority (default: %(default)s)")
    parser.add_argument("--mode", choices=("position", "impedance"), default="position", help="Arm control mode")
    parser.add_argument("--collision-distance", type=float, default=0.015, help="Self-collision threshold [m]")

    # Auto-Reset Subprocess Arguments
    parser.add_argument("--reset-python", type=Path, default=DEFAULT_RESET_PYTHON, help="Python binary for reset script")
    parser.add_argument("--reset-script", type=Path, default=DEFAULT_RESET_SCRIPT, help="sample_and_move_to_ready.py path")
    parser.add_argument("--noise-deg", type=float, default=3.0, help="Perturbation noise magnitude in degrees")
    parser.add_argument("--noise-mode", choices=("uniform", "normal"), default="uniform", help="Perturbation mode")
    parser.add_argument("--reset-time", type=float, default=4.0, help="Reset motion duration in seconds")

    # LeaderArm Auto-Homing Arguments
    parser.add_argument("--autohome-time", type=float, default=3.0, help="LeaderArm auto-homing duration [s]")
    parser.add_argument("--autohome-tolerance", type=float, default=0.05, help="Auto-homing convergence tolerance [rad]")

    # Episode Recording Arguments
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"), help="Output directory for episodes")
    parser.add_argument("--prefix", default="episode", help="Episode file prefix (e.g. episode_0000.npz)")

    # Camera Recording Arguments
    parser.add_argument("--no-cameras", action="store_true", help="Disable camera recording")
    parser.add_argument("--camera-stack-root", type=Path, default=DEFAULT_CAMERA_STACK_ROOT, help="Camera stack root")
    parser.add_argument("--camera-python", type=Path, default=DEFAULT_CAMERA_PYTHON, help="Camera Python environment")
    parser.add_argument("--camera-hz", type=float, default=30.0, help="Camera recording rate [Hz]")
    parser.add_argument("--zed-shm-mode", choices=ZED_SHM_MODES, default="rgb-only", help="ZED SHM mode")
    parser.add_argument("--zed-record-profile", choices=tuple(ZED_RECORD_PROFILES.keys()), default=DEFAULT_ZED_RECORD_PROFILE, help="ZED profile")
    parser.add_argument("--camera-max-frame-age-s", type=float, default=1.0, help="Max frame age [s]")
    parser.add_argument("--camera-init-timeout-s", type=float, default=3.0, help="Camera init timeout [s]")
    parser.add_argument("--camera-storage", choices=("jpeg", "raw"), default="jpeg", help="Camera storage format")
    parser.add_argument("--camera-jpeg-quality", type=int, default=90, help="Camera JPEG quality")
    parser.add_argument("--camera-compression", choices=("lzf", "gzip", "none"), default="lzf", help="HDF5 compression")
    parser.add_argument("--visualize-cameras", action="store_true", help="Enable camera preview")
    parser.add_argument("--camera-preview-hz", type=float, default=10.0, help="Preview rate [Hz]")
    parser.add_argument("--camera-preview-panel-width", type=int, default=400, help="Preview panel width")
    parser.add_argument("--camera-preview-mode", choices=("auto", "window", "browser"), default="auto", help="Preview mode")
    parser.add_argument("--camera-preview-host", default="127.0.0.1", help="Browser preview host")
    parser.add_argument("--camera-preview-port", type=int, default=8765, help="Browser preview port")

    # Mobile Base Arguments
    parser.add_argument("--linear-speed", type=float, default=0.10, help="Linear speed limit [m/s]")
    parser.add_argument("--angular-speed", type=float, default=0.30, help="Angular speed limit [rad/s]")
    parser.add_argument("--linear-acceleration", type=float, default=1.0, help="Linear acceleration [m/s^2]")
    parser.add_argument("--angular-acceleration", type=float, default=2.0, help="Angular acceleration [rad/s^2]")
    parser.add_argument("--mobile-ramp-time", type=float, default=0.20, help="Mobile command ramp time [s]")
    parser.add_argument("--mobile-hold-time", type=float, default=0.60, help="Mobile command hold time [s]")
    parser.add_argument("--mobile-refresh-time", type=float, default=0.40, help="Mobile refresh time [s]")
    parser.add_argument("--key-timeout", type=float, default=0.25, help="Key repeat timeout [s]")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_episode_idx = find_next_episode_idx(args.output_dir, args.prefix)

    stop_event = threading.Event()
    key_queue: queue.Queue[str] = queue.Queue()

    # Unified terminal controller for base mobility (arrows) & FSM action keys (s, e, c, d, q)
    terminal_controller = UnifiedTerminalController(
        key_queue=key_queue,
        stop_event=stop_event,
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
        timeout_s=args.key_timeout,
    )

    robot = None
    gripper = None
    leader_arm = None
    arm_stream = None
    camera_session: Optional[CameraSessionProcess] = None
    current_camera_output: Optional[Path] = None
    current_npz_output: Optional[Path] = None

    state_lock = threading.Lock()
    robot_position = None
    robot_velocity = None
    robot_odometry = np.eye(3, dtype=np.float64)
    robot_is_ready = None

    # Teleop state variables
    teleop_active = False
    right_q = None
    left_q = None
    right_minimum_time = 1.0
    left_minimum_time = 1.0
    last_collision_log_time = 0.0

    def request_stop(signum=None, frame=None):
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    print("\n" + "=" * 70)
    print("  RB-Y1 Interactive Multi-Episode Teleoperation & Recording Session ")
    print("=" * 70)
    print("Workflow:")
    print("  1. Auto-Reset (sample_and_move_to_ready.py with noise)")
    print("  2. Auto-Home LeaderArm to perturbed initial pose")
    print("  3. Teleop Ready & Standby")
    print("  [s] : START recording episode")
    print("  [e] : STOP & SAVE episode -> Auto-open gripper")
    print("  [c] : CONTINUE to next episode (trigger auto-reset)")
    print("  [d] : DISCARD current/recent episode")
    print("  [q] : QUIT session")
    print("=" * 70 + "\n")

    try:
        # 1. Connect to Robot
        logging.info("Connecting to RB-Y1 at %s...", args.address)
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            raise RuntimeError(f"Failed to connect to robot at {args.address}")

        model = robot.model()
        dyn_model = robot.get_dynamics()
        dyn_state = dyn_model.make_state([], model.robot_joint_names)
        robot_max_q = dyn_model.get_limit_q_upper(dyn_state)
        robot_min_q = dyn_model.get_limit_q_lower(dyn_state)
        robot_max_qdot = dyn_model.get_limit_qdot_upper(dyn_state)
        robot_max_qddot = dyn_model.get_limit_qddot_upper(dyn_state)
        position_mode = args.mode == "position"

        if not robot.is_power_on(args.power) and not robot.power_on(args.power):
            logging.warning("Power on check for %r returned False (continuing if devices are powered)", args.power)

        if not robot.is_servo_on(args.servo):
            if not robot.servo_on(args.servo):
                logging.warning(
                    "Failed to servo on %r (wheel drives may be off), falling back to upper body: %s",
                    args.servo,
                    DEFAULT_SERVO_PATTERN,
                )
                if not robot.is_servo_on(DEFAULT_SERVO_PATTERN) and not robot.servo_on(DEFAULT_SERVO_PATTERN):
                    raise RuntimeError(f"Failed to servo on upper body devices ({DEFAULT_SERVO_PATTERN})")
                else:
                    logging.info("Upper body servos (torso/arms/head) are ON.")
            else:
                logging.info("Servos (%s) successfully turned ON.", args.servo)
        else:
            logging.info("Servos (%s) are already ON.", args.servo)

        robot.reset_fault_control_manager()
        if not robot.enable_control_manager():
            raise RuntimeError("Failed to enable control manager")
        for arm in ("right", "left"):
            robot.set_tool_flange_output_voltage(arm, 12)
        time.sleep(1.5)  # Wait for tool flange / gripper Dynamixels to boot up
        robot.set_parameter("joint_position_command.cutoff_frequency", "3")

        # 2. Start State Update Stream (100 Hz)
        def on_robot_state(state):
            nonlocal robot_position, robot_velocity, robot_odometry, robot_is_ready
            with state_lock:
                robot_position = state.position.copy()
                robot_velocity = state.velocity.copy()
                robot_odometry = state.odometry.copy()
                robot_is_ready = state.is_ready.copy()

        robot.start_state_update(on_robot_state, 1.0 / Settings.leader_arm_loop_period)
        deadline = time.monotonic() + 3.0
        while robot_is_ready is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if robot_is_ready is None:
            raise RuntimeError("Timed out waiting for robot state updates")

        # 3. Initialize Gripper
        logging.info("Initializing Gripper hardware...")
        gripper = Gripper()
        if not gripper.initialize():
            raise RuntimeError("Failed to initialize gripper")
        gripper.homing()
        gripper.start()

        # 4. Initialize LeaderArm
        logging.info("Initializing LeaderArm hardware...")
        leader_arm = LeaderArm(control_period=Settings.leader_arm_loop_period)
        leader_arm.initialize(verbose=args.verbose)

        # 5. Stationary table manipulation mode (mobile base excluded)
        logging.info("Stationary table manipulation mode (mobile base disabled).")
        teleop_publisher = TeleopStatePublisher()
        recorder = EpisodeTrajectoryRecorder(model)

        ma_q_limit_barrier = 0.5
        ma_min_q = np.deg2rad([-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360])
        ma_max_q = np.deg2rad([360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360])
        ma_torque_limit = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
        ma_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)

        # LeaderArm Modes: "HOLD" (calm stationary hold during reset), "HOMING" (100 Hz S-curve), "TELEOP"
        leader_mode = "HOLD"
        leader_hold_q: Optional[np.ndarray] = None
        homing_start_q: Optional[np.ndarray] = None
        homing_target_q: Optional[np.ndarray] = None
        homing_start_time = 0.0
        homing_done_event = threading.Event()
        homing_torque_limits = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2, dtype=np.float64)

        def s_curve_quintic(t: float, total_time: float) -> float:
            if total_time <= 0.0:
                return 1.0
            tau = np.clip(t / total_time, 0.0, 1.0)
            return float(10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5))

        def leader_arm_control_loop(state):
            nonlocal right_q, left_q, right_minimum_time, left_minimum_time
            nonlocal last_collision_log_time
            nonlocal leader_hold_q

            if stop_event.is_set():
                return None

            # Mode 1: HOLD - Rock-solid steady hold (prevents jitter/shaking during reset)
            if leader_mode == "HOLD":
                if leader_hold_q is None:
                    leader_hold_q = state.q_joint.copy()
                ma_input = LeaderArm.ControlInput()
                ma_input.target_operating_mode.fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                ma_input.target_torque.fill(2.0)
                ma_input.target_position = leader_hold_q
                return ma_input

            # Mode 2: HOMING - Smooth 100 Hz S-curve auto-homing in the control thread (zero bus collisions)
            if leader_mode == "HOMING":
                elapsed = time.monotonic() - homing_start_time
                s = s_curve_quintic(elapsed, args.autohome_time)
                assert homing_start_q is not None and homing_target_q is not None
                q_des = homing_start_q + s * (homing_target_q - homing_start_q)

                ma_input = LeaderArm.ControlInput()
                ma_input.target_operating_mode.fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                ma_input.target_torque = homing_torque_limits
                ma_input.target_position = q_des

                if elapsed >= args.autohome_time:
                    err = np.max(np.abs(state.q_joint - homing_target_q))
                    if err < args.autohome_tolerance or elapsed >= args.autohome_time + 0.5:
                        homing_done_event.set()
                return ma_input

            # Mode 3: TELEOP - Normal leader arm teleoperation

            if right_q is None:
                right_q = state.q_joint[0:7].copy()
            if left_q is None:
                left_q = state.q_joint[7:14].copy()

            gripper_command = np.array([state.button_right.trigger, state.button_left.trigger], dtype=np.float64) / 1000.0
            gripper.set_target(gripper_command)

            ma_input = LeaderArm.ControlInput()
            torque = (
                state.gravity_term
                + ma_q_limit_barrier * (np.maximum(ma_min_q - state.q_joint, 0) + np.minimum(ma_max_q - state.q_joint, 0))
                + ma_viscous_gain * state.qvel_joint
            )
            torque = np.clip(torque, -ma_torque_limit, ma_torque_limit)

            if state.button_right.button == 1:
                ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentControlMode)
                ma_input.target_torque[0:7] = torque[0:7] * 0.6
                right_q = state.q_joint[0:7].copy()
            else:
                ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                ma_input.target_torque[0:7] = ma_torque_limit[0:7]
                ma_input.target_position[0:7] = right_q

            if state.button_left.button == 1:
                ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentControlMode)
                ma_input.target_torque[7:14] = torque[7:14] * 0.6
                left_q = state.q_joint[7:14].copy()
            else:
                ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
                ma_input.target_torque[7:14] = ma_torque_limit[7:14]
                ma_input.target_position[7:14] = left_q

            with state_lock:
                if robot_position is None:
                    return ma_input
                q = robot_position.copy()
                q_dot = robot_velocity.copy() if robot_velocity is not None else None
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
                    logging.warning("Arm motion blocked by self-collision limit (%.4fm)", nearest.distance)
                    last_collision_log_time = now

            body_builder = rby.BodyComponentBasedCommandBuilder()
            has_body_command = False

            if state.button_right.button and not is_collision:
                right_minimum_time = max(right_minimum_time - Settings.leader_arm_loop_period, Settings.leader_arm_loop_period * 1.01)
                right_builder = rby.JointPositionCommandBuilder() if position_mode else rby.JointImpedanceControlCommandBuilder()
                (
                    right_builder.set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                    .set_position(np.clip(right_q, robot_min_q[model.right_arm_idx], robot_max_q[model.right_arm_idx]))
                    .set_velocity_limit(robot_max_qdot[model.right_arm_idx])
                    .set_acceleration_limit(robot_max_qddot[model.right_arm_idx] * 30)
                    .set_minimum_time(right_minimum_time)
                )
                body_builder.set_right_arm_command(right_builder)
                has_body_command = True
            else:
                right_minimum_time = 0.8

            if state.button_left.button and not is_collision:
                left_minimum_time = max(left_minimum_time - Settings.leader_arm_loop_period, Settings.leader_arm_loop_period * 1.01)
                left_builder = rby.JointPositionCommandBuilder() if position_mode else rby.JointImpedanceControlCommandBuilder()
                (
                    left_builder.set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                    .set_position(np.clip(left_q, robot_min_q[model.left_arm_idx], robot_max_q[model.left_arm_idx]))
                    .set_velocity_limit(robot_max_qdot[model.left_arm_idx])
                    .set_acceleration_limit(robot_max_qddot[model.left_arm_idx] * 30)
                    .set_minimum_time(left_minimum_time)
                )
                body_builder.set_left_arm_command(left_builder)
                has_body_command = True
            else:
                left_minimum_time = 0.8

            base_command = np.zeros(3, dtype=np.float64)
            teleop_publisher.publish(gripper_command, base_command, state.q_joint)
            if has_body_command and arm_stream is not None:
                try:
                    arm_stream.send_command(
                        rby.RobotCommandBuilder().set_command(
                            rby.ComponentBasedCommandBuilder().set_body_command(body_builder)
                        )
                    )
                except Exception as exc:
                    logging.warning("Arm stream send error (stream may need recreate): %s", exc)

            # Record synchronized step if recording is active
            recorder.append(q, q_dot, odometry, base_command, gripper_command, state.q_joint)
            return ma_input

        def safety_function(state):
            logging.error("Leader-arm safety shutdown; faults: %s", sorted(set(state.fault_ids) | set(state.tool_fault_ids)))
            stop_event.set()

        if not leader_arm.start_control(leader_arm_control_loop, safety_function=safety_function):
            raise RuntimeError("Failed to start leader-arm control")

        # Start unified terminal controller (handles arrow keys & action keys)
        terminal_controller.start()

        # FSM Helper: Prepare Episode (Reset -> In-loop S-Curve Auto-home -> Standby)
        def prepare_episode(ep_idx: int) -> bool:
            nonlocal leader_mode, leader_hold_q, homing_start_q, homing_target_q, homing_start_time
            nonlocal arm_stream, right_q, left_q

            # 1. Switch LeaderArm to calm HOLD mode (freezes current angle so it does NOT vibrate)
            leader_hold_q = None  # Will latch current joint angles on next tick
            leader_mode = "HOLD"

            # Release active arm stream before subprocess takes control
            if arm_stream is not None:
                try:
                    arm_stream.cancel()
                except Exception:
                    pass
                arm_stream = None

            print(f"\n[Episode #{ep_idx:04d}] >>> Step 1: Moving robot to perturbed initial pose...")
            if not execute_auto_reset(args):
                print(f"[!] Auto-reset failed for Episode #{ep_idx:04d}. Press 'c' to retry or 'q' to quit.")
                return False

            # 2. Read live robot arm angles
            with state_lock:
                curr_right = robot_position[model.right_arm_idx].copy()
                curr_left = robot_position[model.left_arm_idx].copy()
                curr_torso = robot_position[model.torso_idx].copy()
            target_leader = np.concatenate([curr_right, curr_left])

            # 3. Smooth in-loop S-Curve auto-homing without serial port conflict
            print(f"[Episode #{ep_idx:04d}] >>> Step 2: Auto-Homing LeaderArm to robot pose (duration={args.autohome_time:.1f}s)...")
            homing_done_event.clear()
            homing_start_q = leader_hold_q.copy() if leader_hold_q is not None else target_leader.copy()
            homing_target_q = target_leader.copy()
            homing_start_time = time.monotonic()
            leader_mode = "HOMING"

            if not homing_done_event.wait(timeout=args.autohome_time + 2.0):
                logging.warning("LeaderArm auto-homing timeout reached; proceeding to teleop.")

            right_q = target_leader[0:7].copy()
            left_q = target_leader[7:14].copy()

            # 4. Reconnect arm stream with hold command
            try:
                arm_stream = robot.create_command_stream(priority=args.priority)
                curr_pose = leader_example.Pose(
                    toros=curr_torso,
                    right_arm=curr_right,
                    left_arm=curr_left,
                )
                arm_stream.send_command(
                    joint_position_command_builder(curr_pose, minimum_time=1.0, control_hold_time=1e6, position_mode=position_mode)
                )
            except Exception as exc:
                logging.error("Failed to re-create arm command stream: %s", exc)

            leader_mode = "TELEOP"
            print(f"\n>>> [Episode #{ep_idx:04d}] 로봇 준비 완료 (Teleop ACTIVE).")
            print("    [s] : 데이터 기록 시작 (START)")
            print("    [c] : 다시 리셋 (Re-reset)")
            print("    [q] : 종료 (Quit)")
            return True

        # Run initial episode preparation
        fsm_state = "PREPARING"
        if prepare_episode(current_episode_idx):
            fsm_state = "STANDBY"

        # Main Interactive Event Loop
        while not stop_event.is_set():
            # Check camera session health
            if camera_session is not None and fsm_state == "RECORDING":
                camera_session.check_health()

            # Process keyboard events
            try:
                char = key_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if char == "q":
                print("\n[QUIT] Exiting interactive session...")
                stop_event.set()
                break

            elif char == "s":
                if fsm_state == "RECORDING":
                    print(f"\n[!] Already recording Episode #{current_episode_idx:04d}!")
                    continue
                if fsm_state != "STANDBY":
                    print(f"\n[!] Cannot start recording in state {fsm_state}. Press 'c' to reset first.")
                    continue

                current_npz_output = args.output_dir / f"{args.prefix}_{current_episode_idx:04d}.npz"
                if not args.no_cameras:
                    current_camera_output = camera_sidecar_path(current_npz_output)
                    try:
                        camera_session = CameraSessionProcess(
                            camera_python=args.camera_python,
                            camera_stack_root=args.camera_stack_root,
                            output=current_camera_output,
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
                    except Exception as exc:
                        logging.error("Failed to start camera session: %s", exc)
                        camera_session = None

                recorder.start()
                fsm_state = "RECORDING"
                print("\n" + "=" * 60)
                print(f" >>> [RECORDING] Episode #{current_episode_idx:04d} STARTED!")
                print("     파지를 수행하세요. 파지 완료 후 [e]를 누르세요.")
                print("     취소는 [d] (Discard)")
                print("=" * 60)

            elif char == "e":
                if fsm_state != "RECORDING":
                    print("\n[!] Not recording. Press 's' to start recording first.")
                    continue

                # Stop camera session
                if camera_session is not None:
                    try:
                        camera_session.stop()
                    except Exception as exc:
                        logging.warning("Camera session cleanup error: %s", exc)
                    camera_session = None

                # Save kinematics NPZ
                count = recorder.stop_and_save(current_npz_output)
                fsm_state = "STOPPED"

                # Auto-release gripper
                release_gripper_fully(gripper)

                print("\n" + "=" * 60)
                print(f" >>> [STOPPED] Episode #{current_episode_idx:04d} 저장 완료 (총 {count} samples).")
                print("     그리퍼를 열었습니다.")
                print("     [c] : 다음 에피소드로 이동 (Auto-reset)")
                print("     [d] : 방금 에피소드 삭제 (Discard)")
                print("     [q] : 전체 세션 종료 (Quit)")
                print("=" * 60)

            elif char == "d":
                if fsm_state == "RECORDING":
                    print(f"\n>>> [DISCARD] Cancelling recording of Episode #{current_episode_idx:04d}...")
                    recorder.discard()
                    if camera_session is not None:
                        try:
                            camera_session.stop()
                        except Exception:
                            pass
                        camera_session = None
                    if current_camera_output and current_camera_output.exists():
                        try:
                            current_camera_output.unlink()
                        except Exception:
                            pass
                    release_gripper_fully(gripper)
                    fsm_state = "STANDBY"
                    print(">>> 에피소드가 취소되었습니다. 다시 파지하려면 [s], 리셋하려면 [c]를 누르세요.")

                elif fsm_state == "STOPPED":
                    print(f"\n>>> [DISCARD] Deleting saved files for Episode #{current_episode_idx:04d}...")
                    if current_npz_output and current_npz_output.exists():
                        current_npz_output.unlink(missing_ok=True)
                    if current_camera_output and current_camera_output.exists():
                        current_camera_output.unlink(missing_ok=True)
                    print(f">>> Episode #{current_episode_idx:04d} 파일이 삭제되었습니다. [c]를 눌러 다시 진행하세요.")

            elif char == "c":
                if fsm_state == "RECORDING":
                    print("\n[!] Episode is currently recording. Press 'e' to stop first.")
                    continue

                if fsm_state == "STOPPED":
                    current_episode_idx += 1

                fsm_state = "PREPARING"
                if prepare_episode(current_episode_idx):
                    fsm_state = "STANDBY"

    except Exception:
        logging.exception("Interactive record session failed")
    finally:
        logging.info("Shutting down interactive session...")
        stop_event.set()

        try:
            terminal_controller.stop()
        except Exception:
            pass

        if camera_session is not None:
            try:
                camera_session.stop()
            except Exception:
                pass


        if arm_stream is not None:
            try:
                arm_stream.cancel()
            except Exception:
                pass

        # Graceful torque ramp-down
        if leader_arm is not None:
            try:
                logging.info("Gently ramping down LeaderArm torques (1.2s)...")
                leader_arm.stop_control(torque_disable=False)
                motor_ids = list(range(14))
                steps = 24
                torques = np.array([3.5, 3.5, 3.5, 2.5, 1.5, 1.5, 1.5] * 2, dtype=np.float64)
                for a in np.linspace(1.0, 0.0, steps):
                    leader_arm.bus.group_sync_write_send_torque([(i, torques[i] * a) for i in motor_ids])
                    time.sleep(1.2 / steps)
                leader_arm.DisableTorque()
                logging.info("LeaderArm torques safely released.")
            except Exception:
                pass

        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
