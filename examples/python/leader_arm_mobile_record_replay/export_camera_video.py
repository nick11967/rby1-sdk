#!/usr/bin/env python3
"""Convert a camera HDF5 sidecar into a labeled three-view MP4 video."""

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


def _image_dataset(group):
    if "jpeg" in group:
        return "jpeg", group["jpeg"]
    if "rgb" in group:
        return "rgb", group["rgb"]
    raise ValueError(f"camera group {group.name!r} has no jpeg or rgb dataset")


def _read_bgr(cv2, group, index: int) -> np.ndarray:
    storage, dataset = _image_dataset(group)
    if storage == "jpeg":
        encoded = np.asarray(dataset[index], dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(
                f"failed to decode JPEG at {group.name}, frame {index}"
            )
        return frame
    frame_rgb = np.asarray(dataset[index], dtype=np.uint8)
    return np.ascontiguousarray(frame_rgb[:, :, ::-1])


def _make_panel(
    cv2,
    frame_bgr: np.ndarray,
    title: str,
    *,
    width: int,
    image_height: int,
) -> np.ndarray:
    title_height = 40
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
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _default_output(recording: Path) -> Path:
    suffix = ".cameras.h5"
    if recording.name.endswith(suffix):
        return recording.with_name(
            recording.name[: -len(suffix)] + ".camera_preview.mp4"
        )
    return recording.with_suffix(".mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="*.cameras.h5 file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path (default: <session>.camera_preview.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Output FPS (default: effective rate calculated from timestamps)",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        help="Exclusive end frame index (default: end of recording)",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=480,
        help="Width of each of the three video panels",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC video codec (default: mp4v)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output video",
    )
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index <= args.start_index:
        parser.error("--end-index must be greater than --start-index")
    if args.panel_width < 160:
        parser.error("--panel-width must be at least 160")
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")

    cv2, h5py = _load_dependencies()
    recording = args.recording.expanduser().resolve()
    if not recording.is_file():
        parser.error(f"recording not found: {recording}")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output(recording)
    )
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output} (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(recording, "r") as h5:
        datasets = {
            role: _image_dataset(h5[f"cameras/{role}"])[1]
            for role in CAMERA_ROLES
        }
        counts = {role: len(dataset) for role, dataset in datasets.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"camera frame counts do not match: {counts}")
        total_count = next(iter(counts.values()))
        end_index = total_count if args.end_index is None else args.end_index
        if not 0 <= args.start_index < end_index <= total_count:
            parser.error(f"frame range must be within 0..{total_count}")

        timestamps = np.asarray(
            h5["timestamps/monotonic_ns"][args.start_index:end_index],
            dtype=np.uint64,
        )
        frame_count = len(timestamps)
        duration_s = (
            float(timestamps[-1] - timestamps[0]) / 1e9
            if frame_count > 1
            else 0.0
        )
        measured_fps = (
            (frame_count - 1) / duration_s if duration_s > 0 else 30.0
        )
        output_fps = args.fps if args.fps is not None else measured_fps

        image_height = int(round(args.panel_width * 3.0 / 4.0))
        video_size = (args.panel_width * 3, image_height + 40)
        video = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*args.codec),
            output_fps,
            video_size,
        )
        if not video.isOpened():
            raise RuntimeError(
                f"failed to open video writer for {output} with codec {args.codec}"
            )

        print(
            f"source: {recording}\n"
            f"frames: {frame_count}, recorded duration: {duration_s:.3f}s, "
            f"output FPS: {output_fps:.3f}\n"
            f"video: {video_size[0]}x{video_size[1]}, codec: {args.codec}"
        )
        progress_interval = max(1, frame_count // 10)
        try:
            first_timestamp = int(timestamps[0])
            for output_index, source_index in enumerate(
                range(args.start_index, end_index)
            ):
                elapsed_s = (int(timestamps[output_index]) - first_timestamp) / 1e9
                panels = []
                for role in CAMERA_ROLES:
                    frame = _read_bgr(
                        cv2, h5[f"cameras/{role}"], source_index
                    )
                    title = role
                    if role == "head":
                        title = f"head | t={elapsed_s:7.3f}s"
                    panels.append(
                        _make_panel(
                            cv2,
                            frame,
                            title,
                            width=args.panel_width,
                            image_height=image_height,
                        )
                    )
                video.write(np.hstack(panels))
                completed = output_index + 1
                if completed % progress_interval == 0 or completed == frame_count:
                    print(f"progress: {completed}/{frame_count}")
        finally:
            video.release()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"video output was not created correctly: {output}")
    print(f"saved: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
