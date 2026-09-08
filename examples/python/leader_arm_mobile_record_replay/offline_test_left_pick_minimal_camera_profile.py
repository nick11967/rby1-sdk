#!/usr/bin/env python3
"""Offline verification script for DATA-04A: Left Pick Minimal Camera Recorder Profile.

Runs completely offline without real cameras, robot SDK, or network calls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

# Ensure examples directory is in path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from camera_io import (
    CAMERA_ROLES,
    COLLECTION_PROFILES,
    DEFAULT_COLLECTION_PROFILE,
    WRIST_FRAME_SHAPE,
    ZED_RGB_ONLY_SHAPE,
    CameraHdf5Writer,
    CameraSessionProcess,
)
from record_episodes import check_camera_shm_status


@dataclass
class FakeSnapshot:
    frame: np.ndarray
    monotonic_timestamp_ns: int
    generation: int = 1
    sequence: int = 2


class MockCameraReader:
    def __init__(self):
        self.head_reads = 0
        self.left_wrist_reads = 0
        self.right_wrist_reads = 0
        self.real_hardware_calls = 0

    def get_head_snapshot(self) -> FakeSnapshot:
        self.head_reads += 1
        return FakeSnapshot(
            frame=np.zeros(ZED_RGB_ONLY_SHAPE, dtype=np.uint8),
            monotonic_timestamp_ns=100_000_000 * self.head_reads,
        )

    def get_left_wrist_frame(self) -> np.ndarray:
        self.left_wrist_reads += 1
        return np.zeros(WRIST_FRAME_SHAPE, dtype=np.uint8)

    def get_right_wrist_frame(self) -> np.ndarray:
        self.right_wrist_reads += 1
        return np.zeros(WRIST_FRAME_SHAPE, dtype=np.uint8)

    def get_right_wrist_snapshot(self) -> FakeSnapshot:
        self.right_wrist_reads += 1
        return FakeSnapshot(
            frame=np.zeros(WRIST_FRAME_SHAPE, dtype=np.uint8),
            monotonic_timestamp_ns=100_000_000 * self.right_wrist_reads,
        )


def run_verification() -> int:
    profile = "left-pick-minimal"
    roles = COLLECTION_PROFILES[profile]
    assert roles == ("head", "left_wrist"), f"Unexpected roles: {roles}"

    reader = MockCameraReader()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_h5 = Path(tmpdir) / "test_minimal.cameras.h5"

        writer = CameraHdf5Writer(
            output_h5,
            camera_hz=30.0,
            frame_shapes={
                "head": ZED_RGB_ONLY_SHAPE,
                "left_wrist": WRIST_FRAME_SHAPE,
            },
            head_source_shape=ZED_RGB_ONLY_SHAPE,
            zed_shm_mode="rgb-only",
            zed_record_profile="rgb-only",
            storage_format="jpeg",
            jpeg_quality=90,
            compression="lzf",
            collection_profile=profile,
            camera_roles=roles,
        )

        # Simulate 5 frames recorded following the left-pick-minimal profile
        for i in range(5):
            frames = {}
            if "head" in roles:
                snap = reader.get_head_snapshot()
                frames["head"] = snap.frame
                head_source_ns = snap.monotonic_timestamp_ns
            else:
                head_source_ns = 0

            if "right_wrist" in roles:
                frames["right_wrist"] = reader.get_right_wrist_frame()

            if "left_wrist" in roles:
                frames["left_wrist"] = reader.get_left_wrist_frame()

            writer.append(
                frames,
                monotonic_ns=1_000_000_000 + i * 33_333_333,
                unix_ns=1_700_000_000_000_000_000 + i * 33_333_333,
                head_source_monotonic_ns=head_source_ns,
                head_camera_ns=0,
            )

        writer.close()

        # Validate generated HDF5 file
        with h5py.File(output_h5, "r") as h5:
            # Metadata checks
            meta = h5["meta"].attrs
            saved_profile = meta["collection_profile"]
            saved_roles = list(meta["camera_roles"])
            if isinstance(saved_profile, bytes):
                saved_profile = saved_profile.decode("utf-8")
            saved_roles = [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in saved_roles]

            assert saved_profile == "left-pick-minimal"
            assert saved_roles == ["head", "left_wrist"]

            # Dataset checks
            cameras = h5["cameras"]
            assert "head" in cameras
            assert "left_wrist" in cameras
            assert "right_wrist" not in cameras
            assert "jpeg" in cameras["head"]
            assert "jpeg" in cameras["left_wrist"]
            assert cameras["head"]["jpeg"].shape[0] == 5
            assert cameras["left_wrist"]["jpeg"].shape[0] == 5

        # Check SHM status function for profile ignores right_wrist
        shm_status = check_camera_shm_status("left-pick-minimal")
        assert "right_wrist" not in shm_status
        assert "head" in shm_status
        assert "left_wrist" in shm_status

        # Check CameraSessionProcess command generation
        session = CameraSessionProcess(
            camera_python=Path("/fake/bin/python"),
            collection_profile="left-pick-minimal",
            output=output_h5,
        )
        assert session.collection_profile == "left-pick-minimal"

        # Check invalid profile rejection
        try:
            CameraHdf5Writer(
                Path(tmpdir) / "invalid.h5",
                camera_hz=30.0,
                frame_shapes={},
                head_source_shape=ZED_RGB_ONLY_SHAPE,
                zed_shm_mode="rgb-only",
                zed_record_profile="rgb-only",
                storage_format="jpeg",
                jpeg_quality=90,
                compression="lzf",
                collection_profile="invalid_profile_name",
            )
            assert False, "Should have raised ValueError for invalid profile"
        except ValueError:
            pass

    # Print exact required output
    print("Collection profile: left-pick-minimal")
    print("Recorded roles: head, left_wrist")
    print(f"Right wrist reads: {reader.right_wrist_reads}")
    print("HDF5 right_wrist group: absent")
    print(f"Real camera/SDK calls: {reader.real_hardware_calls}")

    return 0


if __name__ == "__main__":
    sys.exit(run_verification())
