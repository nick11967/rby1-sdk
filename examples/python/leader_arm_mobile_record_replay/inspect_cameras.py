#!/usr/bin/env python3
"""Inspect a camera HDF5 sidecar and export a three-view contact sheet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


DEFAULT_CAMERA_PYTHON = Path("/home/nvidia/openpi_rby1/.venv/bin/python")
CAMERA_ROLES = ("right_wrist", "head", "left_wrist")


def _load_dependencies():
    try:
        import cv2
        import h5py

        return cv2, h5py
    except ImportError as exc:
        target = DEFAULT_CAMERA_PYTHON.expanduser().absolute()
        current = Path(sys.executable).absolute()
        if current == target or not target.is_file():
            raise RuntimeError(
                f"h5py/OpenCV unavailable and camera Python was not usable: {target}"
            ) from exc
        os.execv(
            str(target),
            [str(target), "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        )
        raise AssertionError("os.execv returned unexpectedly")


def _read_rgb(cv2, h5, role: str, index: int) -> np.ndarray:
    group = h5[f"cameras/{role}"]
    if "rgb" in group:
        return np.asarray(group["rgb"][index], dtype=np.uint8)
    if "jpeg" in group:
        encoded = np.asarray(group["jpeg"][index], dtype=np.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"failed to decode {role} JPEG at index {index}")
        return np.ascontiguousarray(frame_bgr[:, :, ::-1])
    raise ValueError(f"camera group {role!r} has neither rgb nor jpeg data")


def _frame_count(h5, role: str) -> int:
    group = h5[f"cameras/{role}"]
    for name in ("rgb", "jpeg"):
        if name in group:
            return len(group[name])
    raise ValueError(f"camera group {role!r} has no supported image dataset")


def _default_contact_sheet(recording: Path) -> Path:
    suffix = ".cameras.h5"
    if recording.name.endswith(suffix):
        return recording.with_name(recording.name[: -len(suffix)] + ".contact_sheet.jpg")
    return recording.with_suffix(".contact_sheet.jpg")


def _make_panel(cv2, frame_rgb: np.ndarray, title: str) -> np.ndarray:
    width = 640
    image_height = 400
    title_height = 46
    frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    scale = min(width / frame_bgr.shape[1], image_height / frame_bgr.shape[0])
    resized = cv2.resize(
        frame_bgr,
        (
            max(1, int(round(frame_bgr.shape[1] * scale))),
            max(1, int(round(frame_bgr.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((title_height + image_height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = title_height + (image_height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(
        panel,
        title,
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="*.cameras.h5 file")
    parser.add_argument(
        "--index",
        type=int,
        help="Frame index to export (default: middle frame; negative values allowed)",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="Output JPG path (default: <session>.contact_sheet.jpg)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Also export the selected frame from each camera as a separate JPG",
    )
    args = parser.parse_args()

    cv2, h5py = _load_dependencies()
    recording = args.recording.expanduser().resolve()
    if not recording.is_file():
        parser.error(f"recording not found: {recording}")

    with h5py.File(recording, "r") as h5:
        counts = {role: _frame_count(h5, role) for role in CAMERA_ROLES}
        if len(set(counts.values())) != 1:
            raise ValueError(f"camera frame counts do not match: {counts}")
        count = next(iter(counts.values()))
        if count == 0:
            raise ValueError("camera recording contains no frames")

        index = count // 2 if args.index is None else args.index
        if index < 0:
            index += count
        if not 0 <= index < count:
            parser.error(f"--index must resolve to 0..{count - 1}")

        timestamps = np.asarray(h5["timestamps/monotonic_ns"][:], dtype=np.uint64)
        duration_s = (
            float(timestamps[-1] - timestamps[0]) / 1e9 if count > 1 else 0.0
        )
        effective_hz = (count - 1) / duration_s if duration_s > 0 else 0.0
        meta = h5["meta"].attrs
        storage = str(meta.get("storage_format", "raw"))
        zed_shm_mode = str(meta.get("zed_shm_mode", "legacy/unknown"))
        print(f"file: {recording}")
        print(
            f"frames: {count}, duration: {duration_s:.3f}s, "
            f"effective rate: {effective_hz:.2f} Hz, storage: {storage}, "
            f"ZED SHM mode: {zed_shm_mode}"
        )

        frames_rgb = {}
        for role in CAMERA_ROLES:
            frame = _read_rgb(cv2, h5, role, index)
            frames_rgb[role] = frame
            print(
                f"{role}: {frame.shape[1]}x{frame.shape[0]}, "
                f"min={int(frame.min())}, max={int(frame.max())}, "
                f"mean={float(frame.mean()):.2f}, std={float(frame.std()):.2f}"
            )

    output = (
        args.contact_sheet.expanduser().resolve()
        if args.contact_sheet is not None
        else _default_contact_sheet(recording)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet = np.hstack(
        [_make_panel(cv2, frames_rgb[role], role) for role in CAMERA_ROLES]
    )
    if not cv2.imwrite(str(output), contact_sheet):
        raise RuntimeError(f"failed to write {output}")
    print(f"contact sheet: {output}")

    if args.export_dir is not None:
        export_dir = args.export_dir.expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        for role, frame_rgb in frames_rgb.items():
            frame_path = export_dir / f"{role}_{index:06d}.jpg"
            if not cv2.imwrite(
                str(frame_path), np.ascontiguousarray(frame_rgb[:, :, ::-1])
            ):
                raise RuntimeError(f"failed to write {frame_path}")
            print(f"frame: {frame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
