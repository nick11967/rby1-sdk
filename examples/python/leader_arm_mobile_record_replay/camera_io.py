#!/usr/bin/env python3
"""Record and preview the three externally managed camera SHM streams.

The public ``CameraSessionProcess`` class is imported by teleop.py under the
RBY1 SDK environment.  The worker is launched with the OpenPI environment,
which provides h5py and OpenCV without adding them to the SDK environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
from multiprocessing import resource_tracker, shared_memory
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np


DEFAULT_CAMERA_STACK_ROOT = Path("/home/nvidia/arpa_h_demo_robot_side")
DEFAULT_CAMERA_PYTHON = Path("/home/nvidia/openpi_rby1/.venv/bin/python")
CAMERA_ROLES = ("head", "right_wrist", "left_wrist")
WRIST_FRAME_SHAPE = (480, 640, 3)
ZED_RGB_ONLY_SHAPE = (480, 640, 3)
ZED_SOURCE_SHAPE = (1200, 1920, 3)
ZED_SHM_MODES = ("rgb-only", "stereo-rgbd")
ZED_RECORD_PROFILES = {
    "full": ZED_SOURCE_SHAPE[:2],
    "record": (600, 960),
    "fast": (400, 640),
}
DEFAULT_ZED_RECORD_PROFILE = "record"


def camera_sidecar_path(robot_recording: Path) -> Path:
    """Return ``session.cameras.h5`` for ``session.npz``."""
    return robot_recording.with_suffix(".cameras.h5")


import struct

TELEOP_SHM_PATH = Path("/dev/shm/rby1_teleop_state")
TELEOP_STRUCT_FMT = "=4sQ2d3d14d"
TELEOP_STRUCT_SIZE = struct.calcsize(TELEOP_STRUCT_FMT)


class TeleopStatePublisher:
    """Publish 100 Hz leader arm triggers, base velocity, and leader joints to /dev/shm."""

    def __init__(self, path: Path = TELEOP_SHM_PATH):
        self.path = path

    def publish(
        self,
        gripper_command: np.ndarray,
        base_command: np.ndarray,
        leader_q: np.ndarray,
    ) -> None:
        try:
            r_g = float(gripper_command[0]) if len(gripper_command) > 0 else 0.0
            l_g = float(gripper_command[1]) if len(gripper_command) > 1 else 0.0
            vx = float(base_command[0]) if len(base_command) > 0 else 0.0
            vy = float(base_command[1]) if len(base_command) > 1 else 0.0
            wz = float(base_command[2]) if len(base_command) > 2 else 0.0
            lq = [float(v) for v in leader_q[:14]] if len(leader_q) >= 14 else [0.0] * 14

            data = struct.pack(
                TELEOP_STRUCT_FMT,
                b"RBY1",
                time.monotonic_ns(),
                r_g,
                l_g,
                vx,
                vy,
                wz,
                *lq,
            )
            with open(self.path, "wb") as f:
                f.write(data)
        except Exception:
            pass


class TeleopStateSubscriber:
    """Subscribe to 100 Hz teleoperation state from /dev/shm."""

    def __init__(self, path: Path = TELEOP_SHM_PATH):
        self.path = path

    def read(self, max_age_s: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            if self.path.exists():
                with open(self.path, "rb") as f:
                    data = f.read(TELEOP_STRUCT_SIZE)
                if len(data) == TELEOP_STRUCT_SIZE:
                    unpacked = struct.unpack(TELEOP_STRUCT_FMT, data)
                    magic, ts_ns, r_g, l_g, vx, vy, wz = unpacked[:7]
                    lq = unpacked[7:]
                    now_ns = time.monotonic_ns()
                    if magic == b"RBY1" and (now_ns - ts_ns) < int(max_age_s * 1e9):
                        return (
                            np.array([r_g, l_g], dtype=np.float64),
                            np.array([vx, vy, wz], dtype=np.float64),
                            np.array(lq, dtype=np.float64),
                        )
        except Exception:
            pass
        return (
            np.zeros(2, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(14, dtype=np.float64),
        )


def _load_camera_readers(camera_stack_root: Path):
    root = camera_stack_root.expanduser().resolve()
    if not (root / "camera_stack").is_dir():
        raise RuntimeError(f"camera_stack package not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from camera_stack.camera_reader import CameraShmError, CameraShmReader
    from camera_stack import zed_stereo_shm

    return CameraShmError, CameraShmReader, zed_stereo_shm


@dataclass(frozen=True)
class ZedLeftSnapshot:
    monotonic_timestamp_ns: int
    camera_timestamp_ns: int
    left_bgr: np.ndarray


class ZedLeftShmReader:
    """Read only the ZED left image while honoring the stereo sequence lock."""

    def __init__(
        self,
        zed_shm,
        *,
        max_snapshot_age_s: float,
        read_attempts: int = 10,
        retry_delay_s: float = 0.0002,
    ) -> None:
        self._zed = zed_shm
        self._max_age_ns = int(max_snapshot_age_s * 1e9)
        self._read_attempts = read_attempts
        self._retry_delay_s = retry_delay_s
        self._left_shm = None
        self._meta_shm = None
        self._left = None
        self._closed = False

    @staticmethod
    def _untrack(shm) -> None:
        private_name = getattr(shm, "_name", None)
        if private_name:
            resource_tracker.unregister(private_name, "shared_memory")

    def _attach(self) -> None:
        left_shm = None
        meta_shm = None
        try:
            left_shm = shared_memory.SharedMemory(
                name=self._zed.LEFT_SHM_NAME, create=False
            )
            self._untrack(left_shm)
            meta_shm = shared_memory.SharedMemory(
                name=self._zed.META_SHM_NAME, create=False
            )
            self._untrack(meta_shm)
            if left_shm.size != self._zed.RGB_NBYTES:
                raise self._zed.ZedStereoShmError(
                    f"{self._zed.LEFT_SHM_NAME} has size {left_shm.size}, "
                    f"expected {self._zed.RGB_NBYTES}"
                )
            if meta_shm.size != self._zed.META_NBYTES:
                raise self._zed.ZedStereoShmError(
                    f"{self._zed.META_SHM_NAME} has size {meta_shm.size}, "
                    f"expected {self._zed.META_NBYTES}"
                )
            self._left_shm = left_shm
            self._meta_shm = meta_shm
            self._left = np.ndarray(
                self._zed.RGB_SHAPE,
                dtype=self._zed.RGB_DTYPE,
                buffer=left_shm.buf,
            )
        except Exception:
            if meta_shm is not None:
                meta_shm.close()
            if left_shm is not None:
                left_shm.close()
            raise

    def _detach(self) -> None:
        self._left = None
        for shm in (self._meta_shm, self._left_shm):
            if shm is not None:
                shm.close()
        self._meta_shm = None
        self._left_shm = None

    def _read_attached(self) -> Optional[ZedLeftSnapshot]:
        for attempt in range(self._read_attempts):
            before = self._zed.META_STRUCT.unpack_from(self._meta_shm.buf, 0)
            sequence = int(before[4])
            if (
                int(before[0]) != self._zed.META_MAGIC
                or int(before[1]) != self._zed.META_VERSION
                or int(before[3]) == 0
                or sequence == 0
            ):
                return None
            if sequence & 1:
                if attempt + 1 < self._read_attempts:
                    time.sleep(self._retry_delay_s)
                continue

            left = self._left.copy()
            after = self._zed.META_STRUCT.unpack_from(self._meta_shm.buf, 0)
            if before[:5] != after[:5] or int(after[4]) & 1:
                if attempt + 1 < self._read_attempts:
                    time.sleep(self._retry_delay_s)
                continue

            timestamp_ns = int(after[5])
            age_ns = time.monotonic_ns() - timestamp_ns
            if (
                int(after[7]) != self._zed.WIDTH
                or int(after[8]) != self._zed.HEIGHT
                or timestamp_ns <= 0
                or age_ns < 0
                or age_ns > self._max_age_ns
            ):
                return None
            return ZedLeftSnapshot(
                monotonic_timestamp_ns=timestamp_ns,
                camera_timestamp_ns=int(after[6]),
                left_bgr=left,
            )
        return None

    def get_latest_snapshot(self) -> ZedLeftSnapshot:
        if self._closed:
            raise self._zed.ZedStereoShmError("ZED left reader is closed")
        last_error = None
        for _ in range(2):
            try:
                if self._left is None:
                    self._attach()
                snapshot = self._read_attached()
                if snapshot is not None:
                    return snapshot
                last_error = "snapshot is stale, updating, or inconsistent"
            except (FileNotFoundError, BufferError, OSError) as exc:
                last_error = str(exc)
            self._detach()
        raise self._zed.ZedStereoShmError(
            f"ZED left snapshot unavailable: {last_error}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._detach()


class CameraHdf5Writer:
    """Incrementally write synchronized RGB frames without retaining them in RAM."""

    def __init__(
        self,
        path: Path,
        *,
        camera_hz: float,
        frame_shapes: dict[str, tuple[int, int, int]],
        head_source_shape: tuple[int, int, int],
        zed_shm_mode: str,
        zed_record_profile: str,
        storage_format: str,
        jpeg_quality: int,
        compression: str,
        capacity_block: int = 120,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is unavailable in the camera worker environment"
            ) from exc

        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._h5 = h5py.File(path, "w")
        self._count = 0
        self._capacity = 0
        self._capacity_block = capacity_block
        self._target_hz = camera_hz
        self._first_monotonic_ns = None
        self._last_monotonic_ns = None
        self._storage_format = storage_format
        self._jpeg_quality = jpeg_quality
        compression_value = None if compression == "none" else compression
        if storage_format == "jpeg":
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV is required for JPEG camera storage"
                ) from exc
            self._cv2 = cv2_module
        else:
            self._cv2 = None

        meta = self._h5.create_group("meta")
        meta.attrs["schema_version"] = 2
        meta.attrs["camera_hz"] = float(camera_hz)
        meta.attrs["color_order"] = "RGB"
        meta.attrs["camera_roles"] = CAMERA_ROLES
        meta.attrs["source"] = "camera_stack_shared_memory"
        meta.attrs["zed_shm_mode"] = zed_shm_mode
        meta.attrs["zed_record_profile"] = zed_record_profile
        meta.attrs["head_source_shape"] = head_source_shape
        meta.attrs["storage_format"] = storage_format
        meta.attrs["jpeg_quality"] = jpeg_quality if storage_format == "jpeg" else -1
        meta.attrs["raw_compression"] = compression if storage_format == "raw" else "n/a"
        meta.attrs["started_monotonic_ns"] = time.monotonic_ns()
        meta.attrs["started_unix_ns"] = time.time_ns()

        timestamps = self._h5.create_group("timestamps")
        self._monotonic_ns = timestamps.create_dataset(
            "monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(capacity_block,),
        )
        self._unix_ns = timestamps.create_dataset(
            "unix_ns",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(capacity_block,),
        )
        self._head_source_monotonic_ns = timestamps.create_dataset(
            "head_source_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(capacity_block,),
        )
        self._head_camera_ns = timestamps.create_dataset(
            "head_camera_ns",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(capacity_block,),
        )

        cameras = self._h5.create_group("cameras")
        self._images = {}
        self._frame_shapes = dict(frame_shapes)
        for role in CAMERA_ROLES:
            frame_shape = self._frame_shapes[role]
            group = cameras.create_group(role)
            group.attrs["source_role"] = role
            group.attrs["frame_shape"] = frame_shape
            group.attrs["storage_format"] = storage_format
            if storage_format == "jpeg":
                self._images[role] = group.create_dataset(
                    "jpeg",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                    chunks=(capacity_block,),
                )
            else:
                self._images[role] = group.create_dataset(
                    "rgb",
                    shape=(0, *frame_shape),
                    maxshape=(None, *frame_shape),
                    dtype="uint8",
                    chunks=(1, *frame_shape),
                    compression=compression_value,
                    shuffle=False,
                )
        self._last_flush = time.monotonic()

    @property
    def count(self) -> int:
        return self._count

    @property
    def effective_hz(self) -> float:
        if (
            self._count < 2
            or self._first_monotonic_ns is None
            or self._last_monotonic_ns <= self._first_monotonic_ns
        ):
            return 0.0
        return (self._count - 1) / (
            (self._last_monotonic_ns - self._first_monotonic_ns) / 1e9
        )

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        capacity = (
            (required + self._capacity_block - 1) // self._capacity_block
        ) * self._capacity_block
        self._monotonic_ns.resize((capacity,))
        self._unix_ns.resize((capacity,))
        self._head_source_monotonic_ns.resize((capacity,))
        self._head_camera_ns.resize((capacity,))
        for role, dataset in self._images.items():
            if self._storage_format == "jpeg":
                dataset.resize((capacity,))
            else:
                dataset.resize((capacity, *self._frame_shapes[role]))
        self._capacity = capacity

    def append(
        self,
        frames_bgr: dict[str, np.ndarray],
        *,
        monotonic_ns: int,
        unix_ns: int,
        head_source_monotonic_ns: int,
        head_camera_ns: int,
    ) -> None:
        self._ensure_capacity(self._count + 1)
        index = self._count
        self._monotonic_ns[index] = monotonic_ns
        self._unix_ns[index] = unix_ns
        self._head_source_monotonic_ns[index] = head_source_monotonic_ns
        self._head_camera_ns[index] = head_camera_ns
        if self._first_monotonic_ns is None:
            self._first_monotonic_ns = monotonic_ns
        self._last_monotonic_ns = monotonic_ns
        for role in CAMERA_ROLES:
            frame = frames_bgr[role]
            expected_shape = self._frame_shapes[role]
            if frame.shape != expected_shape or frame.dtype != np.uint8:
                raise ValueError(
                    f"{role} frame must be uint8 {expected_shape}, got "
                    f"{frame.dtype} {frame.shape}"
                )
            if self._storage_format == "jpeg":
                success, encoded = self._cv2.imencode(
                    ".jpg",
                    frame,
                    [self._cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if not success:
                    raise RuntimeError(f"failed to JPEG-encode {role} frame")
                self._images[role][index] = encoded.reshape(-1)
            else:
                self._images[role][index] = np.ascontiguousarray(frame[:, :, ::-1])
        self._count += 1
        now = time.monotonic()
        if now - self._last_flush >= 1.0:
            self._h5.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._h5 is None:
            return
        self._monotonic_ns.resize((self._count,))
        self._unix_ns.resize((self._count,))
        self._head_source_monotonic_ns.resize((self._count,))
        self._head_camera_ns.resize((self._count,))
        for role, dataset in self._images.items():
            if self._storage_format == "jpeg":
                dataset.resize((self._count,))
            else:
                dataset.resize((self._count, *self._frame_shapes[role]))
        self._h5["meta"].attrs["frame_count"] = self._count
        self._h5["meta"].attrs["effective_hz"] = self.effective_hz
        self._h5["meta"].attrs["duration_s"] = (
            (self._last_monotonic_ns - self._first_monotonic_ns) / 1e9
            if self._count > 1
            else 0.0
        )
        self._h5.flush()
        self._h5.close()
        self._h5 = None


def _make_preview_panel(cv2, image: np.ndarray, title: str, width: int) -> np.ndarray:
    title_height = 36
    image_height, image_width = image.shape[:2]
    display_height = int(round(width * 3.0 / 4.0))
    scale = min(width / image_width, display_height / image_height)
    resized_size = (
        max(1, int(round(image_width * scale))),
        max(1, int(round(image_height * scale))),
    )
    resized = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
    panel = np.zeros((title_height + display_height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = title_height + (display_height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(
        panel,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _resize_head_frame(cv2, frame: np.ndarray, profile: str) -> np.ndarray:
    target_height, target_width = ZED_RECORD_PROFILES[profile]
    if frame.shape[:2] == (target_height, target_width):
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(
        cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    )


class BrowserPreviewServer:
    """Serve the latest three-camera panel without requiring OpenCV HighGUI."""

    def __init__(self, host: str, port: int, refresh_hz: float) -> None:
        self.host = host
        self.port = port
        self.refresh_hz = refresh_hz
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        owner = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    interval_ms = max(20, int(round(1000.0 / owner.refresh_hz)))
                    page = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>RB-Y1 cameras</title>
<style>body{{margin:0;background:#111;color:#eee;font-family:sans-serif}}
h2{{margin:12px}}img{{display:block;width:100%;height:auto}}</style></head>
<body><h2>RB-Y1 Teleop Cameras</h2><img id=\"camera\" alt=\"waiting for frames\">
<script>const image=document.getElementById('camera');
setInterval(()=>{{image.src='/frame.jpg?t='+Date.now()}}, {interval_ms});</script>
</body></html>""".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if path == "/frame.jpg":
                    with owner._lock:
                        frame = owner._latest_jpeg
                    if frame is None:
                        self.send_error(503, "Waiting for the first camera frame")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.send_header("Cache-Control", "no-store, no-cache")
                    self.end_headers()
                    self.wfile.write(frame)
                    return
                self.send_error(404)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((host, port), PreviewHandler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera-browser-preview",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def update(self, cv2, panel_bgr: np.ndarray) -> None:
        success, encoded = cv2.imencode(
            ".jpg", panel_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if not success:
            raise RuntimeError("failed to encode browser preview frame")
        with self._lock:
            self._latest_jpeg = encoded.tobytes()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def camera_worker(args: argparse.Namespace) -> int:
    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    wrist_reader = None
    zed_reader = None
    writer = None
    cv2 = None
    browser_preview = None
    preview_enabled = bool(args.visualize)
    preview_backend = None
    window_name = "RB-Y1 Teleop Cameras"
    frame_count = 0
    last_error_log = 0.0
    next_preview = 0.0
    unavailable_since = None
    last_head_source_monotonic_ns = None
    output_path = Path(args.output).resolve() if args.output else None

    try:
        (
            CameraShmError,
            CameraShmReader,
            zed_shm,
        ) = _load_camera_readers(Path(args.camera_stack_root))
        ZedStereoShmError = zed_shm.ZedStereoShmError
        wrist_reader = CameraShmReader(max_frame_age_s=args.max_frame_age_s)
        if args.zed_shm_mode == "stereo-rgbd":
            zed_reader = ZedLeftShmReader(
                zed_shm,
                max_snapshot_age_s=args.max_frame_age_s,
            )

        try:
            import cv2 as cv2_module

            cv2 = cv2_module
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is unavailable in the camera worker environment"
            ) from exc

        if preview_enabled:
            has_display = bool(
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )
            gui_lines = [
                line.strip()
                for line in cv2.getBuildInformation().splitlines()
                if line.strip().startswith("GUI:")
            ]
            has_highgui = bool(gui_lines and not gui_lines[0].endswith("NONE"))
            if (
                args.preview_mode in ("auto", "window")
                and has_display
                and has_highgui
            ):
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    preview_backend = "window"
                except Exception as exc:
                    logging.warning(
                        "OpenCV window preview unavailable (%s)", exc
                    )
            elif args.preview_mode == "window":
                logging.warning(
                    "OpenCV window preview requested but a display or HighGUI "
                    "backend is unavailable"
                )

            if preview_backend is None and args.preview_mode in ("auto", "browser"):
                try:
                    browser_preview = BrowserPreviewServer(
                        args.preview_host,
                        args.preview_port,
                        args.preview_hz,
                    )
                    browser_preview.start()
                    preview_backend = "browser"
                    logging.info(
                        "Browser camera preview: http://%s:%d/ "
                        "(forward this port when using SSH)",
                        args.preview_host,
                        browser_preview.port,
                    )
                except Exception as exc:
                    logging.warning("Browser camera preview unavailable: %s", exc)

            if preview_backend is None:
                if output_path is None:
                    raise RuntimeError("failed to initialize camera preview")
                logging.warning("Camera recording continues without a preview")
                preview_enabled = False

        period = 1.0 / args.camera_hz
        preview_period = 1.0 / args.preview_hz
        next_sample = time.monotonic()
        first_sample = True
        while not stop_event.is_set():
            if output_path is None and not preview_enabled:
                stop_event.wait(0.1)
                continue
            now = time.monotonic()
            if now < next_sample:
                stop_event.wait(min(next_sample - now, 0.005))
                continue
            next_sample = max(next_sample + period, now)

            try:
                if args.zed_shm_mode == "rgb-only":
                    head_snapshot = wrist_reader.get_head_snapshot()
                    head_bgr = head_snapshot.frame
                    head_source_monotonic_ns = (
                        head_snapshot.monotonic_timestamp_ns
                    )
                    head_camera_ns = 0
                else:
                    zed = zed_reader.get_latest_snapshot()
                    head_bgr = zed.left_bgr
                    head_source_monotonic_ns = zed.monotonic_timestamp_ns
                    head_camera_ns = zed.camera_timestamp_ns
                frames = {
                    "head": head_bgr,
                    "right_wrist": wrist_reader.get_right_wrist_frame(),
                    "left_wrist": wrist_reader.get_left_wrist_frame(),
                }
            except (CameraShmError, ZedStereoShmError) as exc:
                error_time = time.monotonic()
                if unavailable_since is None:
                    unavailable_since = error_time
                timeout_s = (
                    args.init_timeout_s
                    if first_sample
                    else max(2.0, args.max_frame_age_s * 2.0)
                )
                if error_time - unavailable_since >= timeout_s:
                    stage = "initialization" if first_sample else "recording"
                    raise RuntimeError(
                        f"camera SHM unavailable for {timeout_s:.1f}s during "
                        f"{stage} (ZED mode {args.zed_shm_mode}): {exc}"
                    )
                if error_time - last_error_log >= 2.0:
                    logging.warning("Waiting for camera SHM: %s", exc)
                    last_error_log = error_time
                continue
            unavailable_since = None

            # The worker clock may run slightly faster than the producer. Do
            # not serialize the same head publication twice as a new sample.
            if head_source_monotonic_ns == last_head_source_monotonic_ns:
                continue
            last_head_source_monotonic_ns = head_source_monotonic_ns

            captured_monotonic_ns = time.monotonic_ns()
            captured_unix_ns = time.time_ns()
            if output_path is not None:
                recorded_frames = dict(frames)
                if args.zed_shm_mode == "stereo-rgbd":
                    recorded_frames["head"] = _resize_head_frame(
                        cv2, frames["head"], args.zed_record_profile
                    )
                if writer is None:
                    if args.zed_shm_mode == "rgb-only":
                        head_shape = ZED_RGB_ONLY_SHAPE
                        zed_record_profile = "rgb-only"
                    else:
                        head_height, head_width = ZED_RECORD_PROFILES[
                            args.zed_record_profile
                        ]
                        head_shape = (head_height, head_width, 3)
                        zed_record_profile = args.zed_record_profile
                    writer = CameraHdf5Writer(
                        output_path,
                        camera_hz=args.camera_hz,
                        frame_shapes={
                            "head": head_shape,
                            "right_wrist": WRIST_FRAME_SHAPE,
                            "left_wrist": WRIST_FRAME_SHAPE,
                        },
                        head_source_shape=(
                            ZED_RGB_ONLY_SHAPE
                            if args.zed_shm_mode == "rgb-only"
                            else ZED_SOURCE_SHAPE
                        ),
                        zed_shm_mode=args.zed_shm_mode,
                        zed_record_profile=zed_record_profile,
                        storage_format=args.storage_format,
                        jpeg_quality=args.jpeg_quality,
                        compression=args.compression,
                    )
                writer.append(
                    recorded_frames,
                    monotonic_ns=captured_monotonic_ns,
                    unix_ns=captured_unix_ns,
                    head_source_monotonic_ns=head_source_monotonic_ns,
                    head_camera_ns=head_camera_ns,
                )
            frame_count += 1

            if preview_enabled and time.monotonic() >= next_preview:
                panels = [
                    _make_preview_panel(
                        cv2, frames["right_wrist"], "Right wrist", args.panel_width
                    ),
                    _make_preview_panel(
                        cv2, frames["head"], "Head", args.panel_width
                    ),
                    _make_preview_panel(
                        cv2, frames["left_wrist"], "Left wrist", args.panel_width
                    ),
                ]
                canvas = np.hstack(panels)
                if preview_backend == "window":
                    cv2.imshow(window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        logging.info("Camera preview closed; recording continues")
                        cv2.destroyWindow(window_name)
                        preview_enabled = False
                        preview_backend = None
                elif preview_backend == "browser":
                    browser_preview.update(cv2, canvas)
                next_preview = time.monotonic() + preview_period

            if first_sample:
                Path(args.ready_file).write_text("ready\n", encoding="ascii")
                logging.info(
                    "Camera SHM ready (%s): head, right_wrist, left_wrist%s",
                    args.zed_shm_mode,
                    f"; recording to {output_path}" if output_path else "",
                )
                first_sample = False

        return 0
    except Exception as exc:
        logging.exception("Camera worker failed")
        try:
            Path(args.error_file).write_text(str(exc), encoding="utf-8")
        except Exception:
            pass
        return 1
    finally:
        if writer is not None:
            try:
                writer.close()
                logging.info(
                    "Camera recording saved: %s (%d synchronized frames, %.2f Hz)",
                    output_path,
                    writer.count,
                    writer.effective_hz,
                )
                if writer.effective_hz < args.camera_hz * 0.9:
                    logging.warning(
                        "Camera recording rate %.2f Hz is below the %.2f Hz target",
                        writer.effective_hz,
                        args.camera_hz,
                    )
            except Exception:
                logging.exception("Failed to finalize camera HDF5")
        if zed_reader is not None:
            zed_reader.close()
        if wrist_reader is not None:
            wrist_reader.close()
        if browser_preview is not None:
            try:
                browser_preview.close()
            except Exception:
                logging.exception("Failed to stop browser camera preview")
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass


class CameraSessionProcess:
    """Manage the camera worker running in a dependency-complete environment."""

    def __init__(
        self,
        *,
        camera_python: Path = DEFAULT_CAMERA_PYTHON,
        camera_stack_root: Path = DEFAULT_CAMERA_STACK_ROOT,
        output: Optional[Path] = None,
        camera_hz: float = 30.0,
        zed_shm_mode: str = "rgb-only",
        zed_record_profile: str = DEFAULT_ZED_RECORD_PROFILE,
        max_frame_age_s: float = 1.0,
        init_timeout_s: float = 30.0,
        visualize: bool = False,
        preview_hz: float = 10.0,
        panel_width: int = 400,
        preview_mode: str = "auto",
        preview_host: str = "127.0.0.1",
        preview_port: int = 8765,
        storage_format: str = "jpeg",
        jpeg_quality: int = 90,
        compression: str = "lzf",
    ) -> None:
        self.camera_python = Path(camera_python).expanduser()
        self.camera_stack_root = Path(camera_stack_root).expanduser()
        self.output = Path(output) if output is not None else None
        self.camera_hz = camera_hz
        self.zed_shm_mode = zed_shm_mode
        self.zed_record_profile = zed_record_profile
        self.max_frame_age_s = max_frame_age_s
        self.init_timeout_s = init_timeout_s
        self.visualize = visualize
        self.preview_hz = preview_hz
        self.panel_width = panel_width
        self.preview_mode = preview_mode
        self.preview_host = preview_host
        self.preview_port = preview_port
        self.storage_format = storage_format
        self.jpeg_quality = jpeg_quality
        self.compression = compression
        self._process: Optional[subprocess.Popen] = None
        self._control_dir: Optional[Path] = None
        self._error_file: Optional[Path] = None

    def start(self) -> None:
        if not self.camera_python.is_file():
            raise RuntimeError(
                f"camera Python environment not found: {self.camera_python}"
            )
        if self.output is None and not self.visualize:
            return
        self._control_dir = Path(tempfile.mkdtemp(prefix="rby1_camera_"))
        ready_file = self._control_dir / "ready"
        error_file = self._control_dir / "error"
        self._error_file = error_file
        command = [
            str(self.camera_python),
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--camera-stack-root",
            str(self.camera_stack_root),
            "--camera-hz",
            str(self.camera_hz),
            "--zed-shm-mode",
            self.zed_shm_mode,
            "--zed-record-profile",
            self.zed_record_profile,
            "--max-frame-age-s",
            str(self.max_frame_age_s),
            "--init-timeout-s",
            str(self.init_timeout_s),
            "--preview-hz",
            str(self.preview_hz),
            "--panel-width",
            str(self.panel_width),
            "--preview-mode",
            self.preview_mode,
            "--preview-host",
            self.preview_host,
            "--preview-port",
            str(self.preview_port),
            "--storage-format",
            self.storage_format,
            "--jpeg-quality",
            str(self.jpeg_quality),
            "--compression",
            self.compression,
            "--ready-file",
            str(ready_file),
            "--error-file",
            str(error_file),
        ]
        if self.output is not None:
            command.extend(("--output", str(self.output.resolve())))
        if self.visualize:
            command.append("--visualize")

        self._process = subprocess.Popen(command)
        deadline = time.monotonic() + self.init_timeout_s + 1.0
        while time.monotonic() < deadline:
            if ready_file.exists():
                return
            if self._process.poll() is not None:
                message = (
                    error_file.read_text(encoding="utf-8")
                    if error_file.exists()
                    else f"camera worker exited with code {self._process.returncode}"
                )
                self.stop()
                raise RuntimeError(message)
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(
            f"camera initialization timed out after {self.init_timeout_s:.1f}s; "
            "start camera_stack/run_cameras.sh with matching --zed-mode "
            f"{self.zed_shm_mode} first"
        )

    def check_health(self) -> None:
        """Raise if a worker that was started has exited unexpectedly."""
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is None:
            return
        message = (
            self._error_file.read_text(encoding="utf-8")
            if self._error_file is not None and self._error_file.exists()
            else f"camera worker exited with code {return_code}"
        )
        raise RuntimeError(message)

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
        self._process = None
        if self._control_dir is not None:
            shutil.rmtree(self._control_dir, ignore_errors=True)
        self._control_dir = None
        self._error_file = None


def create_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--camera-stack-root", type=Path, required=True)
    parser.add_argument("--camera-hz", type=float, default=30.0)
    parser.add_argument(
        "--zed-shm-mode", choices=ZED_SHM_MODES, default="rgb-only"
    )
    parser.add_argument(
        "--zed-record-profile",
        choices=tuple(ZED_RECORD_PROFILES),
        default=DEFAULT_ZED_RECORD_PROFILE,
    )
    parser.add_argument("--max-frame-age-s", type=float, default=1.0)
    parser.add_argument("--init-timeout-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--preview-hz", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument(
        "--preview-mode", choices=("auto", "window", "browser"), default="auto"
    )
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument(
        "--storage-format", choices=("jpeg", "raw"), default="jpeg"
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--error-file", type=Path, required=True)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [camera] %(message)s",
    )
    args = create_worker_parser().parse_args()
    if not args.worker:
        raise SystemExit("camera_io.py is launched by teleop.py/record.py")
    if args.camera_hz <= 0 or args.preview_hz <= 0:
        raise SystemExit("camera and preview rates must be positive")
    if args.max_frame_age_s <= 0 or args.init_timeout_s <= 0:
        raise SystemExit("camera age and initialization timeout must be positive")
    if args.panel_width < 100:
        raise SystemExit("preview panel width must be at least 100")
    if not 0 <= args.preview_port <= 65535:
        raise SystemExit("preview port must be between 0 and 65535")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("JPEG quality must be between 1 and 100")
    return camera_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
