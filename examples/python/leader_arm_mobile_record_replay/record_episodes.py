#!/usr/bin/env python3
"""Interactive Multi-Episode Data Recorder for RBY1 with 3-Camera Support.

This script runs as a completely independent, read-only data logger:
- Absolutely NO robot actuation or control commands are sent.
- Connects to the robot state update stream (100 Hz).
- Records 3 synchronized cameras (Head ZED, Right Wrist, Left Wrist) into HDF5 sidecars.
- Provides an interactive terminal interface for episode-based data collection:
    [s] : Start recording an episode (Kinematics NPZ + Cameras H5)
    [e] : Stop recording and save episode (e.g. recordings/episode_0000.npz & .cameras.h5)
    [x] : Discard/cancel current episode without saving
    [q] : Quit recorder (or Ctrl+C)
"""

from __future__ import annotations

import argparse
from datetime import datetime
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
from typing import Dict, List, Optional

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
    TeleopStateSubscriber,
)


def find_next_episode_idx(output_dir: Path, prefix: str) -> int:
    """Find the next unused episode index in output_dir (checks both .npz and .cameras.h5)."""
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


class KeyboardListener:
    """Non-blocking interactive keyboard input handler for terminal."""

    def __init__(self, key_callback, stop_event: threading.Event) -> None:
        self.key_callback = key_callback
        self.stop_event = stop_event
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._old_termios = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive recording requires a TTY terminal.")
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="key-listener", daemon=True)
        self._thread.start()

    def flush(self) -> None:
        """Flush any pending unread characters in the stdin buffer."""
        if self._fd is not None:
            try:
                termios.tcflush(self._fd, termios.TCIFLUSH)
            except Exception:
                pass

    def stop(self) -> None:
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

    def _run(self) -> None:
        assert self._fd is not None
        while not self.stop_event.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            data = os.read(self._fd, 16)
            if not data:
                self.stop_event.set()
                break
            for byte_val in data:
                char = chr(byte_val).lower()
                if byte_val == 3:  # Ctrl+C
                    self.stop_event.set()
                    return
                self.key_callback(char)


def check_camera_shm_status() -> Dict[str, bool]:
    """Check if camera shared memory segments exist in /dev/shm."""
    shm_dir = Path("/dev/shm")
    status = {
        "head": (shm_dir / "head_frame_shm").exists(),
        "right_wrist": (shm_dir / "right_wrist_frame_shm").exists(),
        "left_wrist": (shm_dir / "left_wrist_frame_shm").exists(),
    }
    return status


class EpisodeBuffer:
    """Thread-safe buffer holding samples for a single recording episode."""

    def __init__(self, model_name: str, joint_names: List[str]) -> None:
        self.model_name = model_name
        self.joint_names = joint_names
        self.started_monotonic_ns = 0
        self.started_unix_ns = 0
        self.is_recording = False
        self.teleop_sub = TeleopStateSubscriber()
        self._lock = threading.Lock()

        self.time_s: List[float] = []
        self.monotonic_ns: List[int] = []
        self.unix_ns: List[int] = []
        self.position: List[np.ndarray] = []
        self.velocity: List[np.ndarray] = []
        self.current: List[np.ndarray] = []
        self.torque: List[np.ndarray] = []
        self.target_position: List[np.ndarray] = []
        self.target_velocity: List[np.ndarray] = []
        self.odometry: List[np.ndarray] = []
        self.odometry_pose: List[np.ndarray] = []
        self.gripper_command: List[np.ndarray] = []
        self.base_command: List[np.ndarray] = []
        self.leader_position: List[np.ndarray] = []

    def start(self) -> None:
        with self._lock:
            self.started_monotonic_ns = time.monotonic_ns()
            self.started_unix_ns = time.time_ns()
            self.time_s.clear()
            self.monotonic_ns.clear()
            self.unix_ns.clear()
            self.position.clear()
            self.velocity.clear()
            self.current.clear()
            self.torque.clear()
            self.target_position.clear()
            self.target_velocity.clear()
            self.odometry.clear()
            self.odometry_pose.clear()
            self.gripper_command.clear()
            self.base_command.clear()
            self.leader_position.clear()
            self.is_recording = True

    def append_sample(self, state: rby.RobotState) -> None:
        with self._lock:
            if not self.is_recording:
                return
            now_mono = time.monotonic_ns()
            elapsed_s = (now_mono - self.started_monotonic_ns) / 1e9
            wall_ns = self.started_unix_ns + (now_mono - self.started_monotonic_ns)

            self.time_s.append(elapsed_s)
            self.monotonic_ns.append(now_mono)
            self.unix_ns.append(wall_ns)
            self.position.append(state.position.copy())
            self.velocity.append(state.velocity.copy())
            self.current.append(state.current.copy())
            self.torque.append(state.torque.copy())
            self.target_position.append(state.target_position.copy())
            self.target_velocity.append(state.target_velocity.copy())
            self.odometry.append(state.odometry.copy())

            # SE2 pose representation [x, y, theta] from odometry transformation matrix
            odom_mat = state.odometry
            x = odom_mat[0, 2]
            y = odom_mat[1, 2]
            theta = np.arctan2(odom_mat[1, 0], odom_mat[0, 0])
            self.odometry_pose.append(np.array([x, y, theta], dtype=np.float64))

            # Read live gripper triggers, mobile base command, and leader arm joint angles from SHM
            grip_cmd, base_cmd, leader_q = self.teleop_sub.read()
            self.gripper_command.append(grip_cmd.copy())
            self.base_command.append(base_cmd.copy())
            self.leader_position.append(leader_q.copy())

    def sample_count(self) -> int:
        with self._lock:
            return len(self.time_s)

    def elapsed_time(self) -> float:
        with self._lock:
            if not self.is_recording or not self.time_s:
                return 0.0
            return self.time_s[-1]

    def stop_and_save(self, file_path: Path) -> int:
        with self._lock:
            self.is_recording = False
            count = len(self.time_s)
            if count == 0:
                logging.warning("No samples collected; file not saved.")
                return 0

            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                file_path,
                schema_version=np.array(2, dtype=np.int64),
                model_name=np.array(self.model_name),
                joint_names=np.asarray(self.joint_names),
                started_monotonic_ns=np.array(self.started_monotonic_ns, dtype=np.uint64),
                started_unix_ns=np.array(self.started_unix_ns, dtype=np.uint64),
                time_s=np.asarray(self.time_s, dtype=np.float64),
                monotonic_ns=np.asarray(self.monotonic_ns, dtype=np.uint64),
                unix_ns=np.asarray(self.unix_ns, dtype=np.uint64),
                position=np.asarray(self.position, dtype=np.float64),
                velocity=np.asarray(self.velocity, dtype=np.float64),
                current=np.asarray(self.current, dtype=np.float64),
                torque=np.asarray(self.torque, dtype=np.float64),
                target_position=np.asarray(self.target_position, dtype=np.float64),
                target_velocity=np.asarray(self.target_velocity, dtype=np.float64),
                odometry=np.asarray(self.odometry, dtype=np.float64),
                odometry_pose=np.asarray(self.odometry_pose, dtype=np.float64),
                gripper_command=np.asarray(self.gripper_command, dtype=np.float64),
                base_command=np.asarray(self.base_command, dtype=np.float64),
                leader_position=np.asarray(self.leader_position, dtype=np.float64),
            )
            return count

    def discard(self) -> None:
        with self._lock:
            self.is_recording = False
            self.time_s.clear()
            self.monotonic_ns.clear()
            self.unix_ns.clear()
            self.position.clear()
            self.velocity.clear()
            self.current.clear()
            self.torque.clear()
            self.target_position.clear()
            self.target_velocity.clear()
            self.odometry.clear()
            self.odometry_pose.clear()
            self.gripper_command.clear()
            self.base_command.clear()
            self.leader_position.clear()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot address (default: 192.168.30.1:50051)")
    parser.add_argument("--model", default="a", help="Robot model (default: a)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory to save episode .npz files (default: recordings/)",
    )
    parser.add_argument(
        "--prefix",
        default="episode",
        help="Episode file prefix (default: 'episode' -> episode_0000.npz)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Robot state polling rate in Hz (default: 100.0 Hz)",
    )

    # Camera Arguments
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Disable camera recording (records robot kinematics NPZ only)",
    )
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
        help="Must match run_cameras.sh --zed-mode; default is the 30 Hz RGB-only collection mode",
    )
    parser.add_argument(
        "--zed-record-profile",
        choices=tuple(ZED_RECORD_PROFILES.keys()),
        default=DEFAULT_ZED_RECORD_PROFILE,
        help="Head ZED HDF5 resolution (stereo-rgbd only)",
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
        default=3.0,
        help="Wait this long for all three SHM cameras (default: 3.0s)",
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
        help="Show the three live camera images in an OpenCV window / preview server",
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
        help="Browser preview port",
    )

    return parser


def main() -> int:
    args = create_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    current_episode_idx = find_next_episode_idx(output_dir, args.prefix)

    stop_event = threading.Event()
    state_received_event = threading.Event()
    robot = None
    listener = None
    camera_session: Optional[CameraSessionProcess] = None
    current_camera_output: Optional[Path] = None

    print("\n" + "=" * 65)
    print("  RBY1 Interactive Episode Data Recorder (with 3 Cameras)     ")
    print("=" * 65)
    print(f"Cameras Enabled: {'NO (--no-cameras)' if args.no_cameras else f'YES (30 Hz, mode={args.zed_shm_mode})'}")

    # Camera SHM Diagnostics
    if not args.no_cameras:
        shm_status = check_camera_shm_status()
        shm_all_ok = all(shm_status.values())
        print("Camera Shared Memory Status:")
        for cam_name, is_ok in shm_status.items():
            print(f"  - {cam_name:12s} (/dev/shm/{cam_name}_frame_shm): {'[OK]' if is_ok else '[MISSING - not running]'}")
        if not shm_all_ok:
            print("\n[!] WARNING: One or more camera SHM streams are missing.")
            print("    Please run: /home/nvidia/arpa_h_demo_robot_side/camera_stack/run_cameras.sh --zed-mode rgb-only")
            print("    Or run with --no-cameras to record robot kinematics only.")

    print("\nControls:")
    print("  [s] : START recording episode")
    print("  [e] : STOP and SAVE current episode")
    print("  [x] : CANCEL / DISCARD current episode (do not save)")
    print("  [q] : QUIT recorder (or Ctrl+C)")
    print("=" * 65 + "\n")

    try:
        logging.info("Connecting to robot state stream at %s (Read-Only mode)...", args.address)
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            logging.error("Failed to connect to robot at %s!", args.address)
            return 1

        model = robot.model()
        buffer = EpisodeBuffer(model.model_name, list(model.robot_joint_names))

        def on_robot_state(state):
            state_received_event.set()
            buffer.append_sample(state)

        # Passive 100 Hz state update subscription (NO robot actuation or commands sent)
        robot.start_state_update(on_robot_state, args.rate)

        if not state_received_event.wait(timeout=3.0):
            logging.error("Timed out waiting for robot state stream from %s!", args.address)
            return 1

        logging.info("Connected to robot state stream successfully (%d joints, %.1f Hz).",
                     len(model.robot_joint_names), args.rate)

        # State machine variables
        state = "IDLE"  # "IDLE" or "RECORDING"

        def print_idle_prompt():
            cam_str = " [+ Cameras H5]" if not args.no_cameras else ""
            print(f"\n>>> [IDLE] Ready for Episode #{current_episode_idx:04d}{cam_str}")
            print(f"    Press 's' to start recording, 'q' to quit.")

        def on_key(char: str):
            nonlocal state, current_episode_idx, camera_session, current_camera_output

            if char == "q":
                print("\n[QUIT] Exiting recorder...")
                stop_event.set()

            elif char == "s":
                if state == "RECORDING":
                    print(f"\n[!] Already recording Episode #{current_episode_idx:04d}!")
                    return

                file_name = f"{args.prefix}_{current_episode_idx:04d}.npz"
                save_path = output_dir / file_name

                # Start Camera Session if enabled
                if not args.no_cameras:
                    current_camera_output = camera_sidecar_path(save_path)
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
                        if listener is not None:
                            listener.flush()
                        print("\n[!] ERROR: Camera initialization failed! Recording was CANCELLED.")
                        print("    Please check if run_cameras.sh is running, or add --no-cameras.")
                        state = "IDLE"
                        print_idle_prompt()
                        return

                state = "RECORDING"
                if listener is not None:
                    listener.flush()
                buffer.start()
                cam_msg = " + 3 Cameras" if camera_session is not None else ""
                print(f"\n>>> [● RECORDING START] Episode #{current_episode_idx:04d} recording started{cam_msg}!")
                print("    Press 'e' to stop and save, 'x' to discard.")

            elif char == "e":
                if state != "RECORDING":
                    return

                # Stop camera session
                if camera_session is not None:
                    try:
                        camera_session.stop()
                    except Exception as exc:
                        logging.warning("Camera stop error: %s", exc)
                    camera_session = None

                file_name = f"{args.prefix}_{current_episode_idx:04d}.npz"
                save_path = output_dir / file_name
                count = buffer.stop_and_save(save_path)
                dur = buffer.elapsed_time()

                if count > 0:
                    cam_info = ""
                    if current_camera_output and current_camera_output.exists():
                        cam_size_mb = current_camera_output.stat().st_size / (1024 * 1024)
                        cam_info = f" + Cameras ({cam_size_mb:.1f} MB)"

                    print(f"\n>>> [✔ SAVED] Episode #{current_episode_idx:04d} saved to {save_path} ({count} samples, {dur:.2f}s){cam_info}")
                    current_episode_idx += 1
                else:
                    print(f"\n>>> [!] Episode #{current_episode_idx:04d} stopped with 0 samples. (File not saved)")

                state = "IDLE"
                if listener is not None:
                    listener.flush()
                print_idle_prompt()

            elif char == "x":
                if state != "RECORDING":
                    return

                # Discard camera session and sidecar file
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

                buffer.discard()
                print(f"\n>>> [✖ DISCARDED] Episode #{current_episode_idx:04d} discarded without saving.")
                state = "IDLE"
                if listener is not None:
                    listener.flush()
                print_idle_prompt()

        listener = KeyboardListener(on_key, stop_event)
        listener.start()

        print_idle_prompt()

        # Main loop: display live status while recording
        last_display_time = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if state == "RECORDING":
                if camera_session is not None:
                    try:
                        camera_session.check_health()
                    except Exception as exc:
                        logging.warning("Camera session health warning: %s", exc)
                if now - last_display_time >= 0.5:
                    samples = buffer.sample_count()
                    dur = buffer.elapsed_time()
                    cam_tag = " [Cam: REC]" if camera_session is not None else ""
                    sys.stdout.write(f"\r    [● REC #{current_episode_idx:04d}{cam_tag}] Duration: {dur:5.1f}s | Samples: {samples:5d}  (Press 'e' to save, 'x' to discard)")
                    sys.stdout.flush()
                    last_display_time = now
            time.sleep(0.05)

    except Exception:
        logging.exception("Exception in Episode Recorder!")
        return 1
    finally:
        print("\nCleaning up recorder...")
        stop_event.set()
        if listener is not None:
            listener.stop()
        if camera_session is not None:
            try:
                camera_session.stop()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.stop_state_update()
            except Exception:
                pass
        logging.info("Recorder shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
