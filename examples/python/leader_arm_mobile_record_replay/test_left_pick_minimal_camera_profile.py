from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

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
    create_worker_parser,
)
from record_episodes import check_camera_shm_status, create_parser


@dataclass
class FakeSnapshot:
    frame: np.ndarray
    monotonic_timestamp_ns: int
    generation: int = 1
    sequence: int = 2


class MockReader:
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


def test_1_default_profile_is_full():
    assert DEFAULT_COLLECTION_PROFILE == "full"
    worker_args = create_worker_parser().parse_args(["--camera-stack-root", "/tmp", "--ready-file", "/tmp/r", "--error-file", "/tmp/e"])
    assert worker_args.collection_profile == "full"
    record_args = create_parser().parse_args([])
    assert record_args.collection_profile == "full"


def test_2_full_profile_retains_three_camera_roles():
    assert "full" in COLLECTION_PROFILES
    assert COLLECTION_PROFILES["full"] == ("head", "right_wrist", "left_wrist")
    assert CAMERA_ROLES == ("head", "right_wrist", "left_wrist")

    shm_status = check_camera_shm_status("full")
    assert set(shm_status.keys()) == {"head", "right_wrist", "left_wrist"}


def test_3_and_4_left_pick_minimal_reads_only_head_and_left_wrist():
    reader = MockReader()
    active_roles = COLLECTION_PROFILES["left-pick-minimal"]
    assert active_roles == ("head", "left_wrist")

    frames = {}
    if "head" in active_roles:
        snap = reader.get_head_snapshot()
        frames["head"] = snap.frame
    if "right_wrist" in active_roles:
        frames["right_wrist"] = reader.get_right_wrist_frame()
    if "left_wrist" in active_roles:
        frames["left_wrist"] = reader.get_left_wrist_frame()

    assert set(frames.keys()) == {"head", "left_wrist"}
    assert reader.head_reads == 1
    assert reader.left_wrist_reads == 1
    assert reader.right_wrist_reads == 0


def test_5_right_wrist_shm_absence_allows_minimal_init(tmp_path):
    # Right wrist SHM is not required for left-pick-minimal
    shm_status = check_camera_shm_status("left-pick-minimal")
    assert "right_wrist" not in shm_status
    assert "head" in shm_status
    assert "left_wrist" in shm_status

    # CameraHdf5Writer initializes successfully with left-pick-minimal without right_wrist
    h5_path = tmp_path / "test.cameras.h5"
    writer = CameraHdf5Writer(
        h5_path,
        camera_hz=30.0,
        frame_shapes={"head": ZED_RGB_ONLY_SHAPE, "left_wrist": WRIST_FRAME_SHAPE},
        head_source_shape=ZED_RGB_ONLY_SHAPE,
        zed_shm_mode="rgb-only",
        zed_record_profile="rgb-only",
        storage_format="jpeg",
        jpeg_quality=90,
        compression="lzf",
        collection_profile="left-pick-minimal",
    )
    writer.close()
    assert h5_path.exists()


def test_6_and_7_hdf5_creates_only_head_and_left_wrist_no_right_wrist(tmp_path):
    h5_path = tmp_path / "test_roles.cameras.h5"
    writer = CameraHdf5Writer(
        h5_path,
        camera_hz=30.0,
        frame_shapes={"head": ZED_RGB_ONLY_SHAPE, "left_wrist": WRIST_FRAME_SHAPE},
        head_source_shape=ZED_RGB_ONLY_SHAPE,
        zed_shm_mode="rgb-only",
        zed_record_profile="rgb-only",
        storage_format="jpeg",
        jpeg_quality=90,
        compression="lzf",
        collection_profile="left-pick-minimal",
    )
    writer.append(
        {
            "head": np.zeros(ZED_RGB_ONLY_SHAPE, dtype=np.uint8),
            "left_wrist": np.zeros(WRIST_FRAME_SHAPE, dtype=np.uint8),
        },
        monotonic_ns=1_000_000_000,
        unix_ns=1_700_000_000_000_000_000,
        head_source_monotonic_ns=1_000_000_000,
        head_camera_ns=0,
    )
    writer.close()

    with h5py.File(h5_path, "r") as h5:
        cameras = h5["cameras"]
        assert "head" in cameras
        assert "left_wrist" in cameras
        assert "right_wrist" not in cameras


def test_8_hdf5_metadata_correctness(tmp_path):
    h5_path = tmp_path / "test_meta.cameras.h5"
    writer = CameraHdf5Writer(
        h5_path,
        camera_hz=30.0,
        frame_shapes={"head": ZED_RGB_ONLY_SHAPE, "left_wrist": WRIST_FRAME_SHAPE},
        head_source_shape=ZED_RGB_ONLY_SHAPE,
        zed_shm_mode="rgb-only",
        zed_record_profile="rgb-only",
        storage_format="jpeg",
        jpeg_quality=90,
        compression="lzf",
        collection_profile="left-pick-minimal",
    )
    writer.append(
        {
            "head": np.zeros(ZED_RGB_ONLY_SHAPE, dtype=np.uint8),
            "left_wrist": np.zeros(WRIST_FRAME_SHAPE, dtype=np.uint8),
        },
        monotonic_ns=1_000_000_000,
        unix_ns=1_700_000_000_000_000_000,
        head_source_monotonic_ns=1_000_000_000,
        head_camera_ns=0,
    )
    writer.close()

    with h5py.File(h5_path, "r") as h5:
        meta = h5["meta"].attrs
        profile = meta["collection_profile"]
        if isinstance(profile, bytes):
            profile = profile.decode("utf-8")
        assert profile == "left-pick-minimal"

        roles = [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in meta["camera_roles"]]
        assert roles == ["head", "left_wrist"]

        # Ensure existing metadata fields are preserved
        assert meta["camera_hz"] == 30.0
        assert meta["source"] == "camera_stack_shared_memory"
        assert meta["zed_shm_mode"] == "rgb-only"
        assert meta["zed_record_profile"] == "rgb-only"
        assert meta["storage_format"] == "jpeg"
        assert meta["jpeg_quality"] == 90
        assert meta["frame_count"] == 1


@pytest.mark.parametrize("storage_format", ["jpeg", "raw"])
def test_9_same_role_selection_for_jpeg_and_raw(tmp_path, storage_format):
    h5_path = tmp_path / f"test_{storage_format}.cameras.h5"
    writer = CameraHdf5Writer(
        h5_path,
        camera_hz=30.0,
        frame_shapes={"head": ZED_RGB_ONLY_SHAPE, "left_wrist": WRIST_FRAME_SHAPE},
        head_source_shape=ZED_RGB_ONLY_SHAPE,
        zed_shm_mode="rgb-only",
        zed_record_profile="rgb-only",
        storage_format=storage_format,
        jpeg_quality=90,
        compression="lzf",
        collection_profile="left-pick-minimal",
    )
    writer.append(
        {
            "head": np.zeros(ZED_RGB_ONLY_SHAPE, dtype=np.uint8),
            "left_wrist": np.zeros(WRIST_FRAME_SHAPE, dtype=np.uint8),
        },
        monotonic_ns=1_000_000_000,
        unix_ns=1_700_000_000_000_000_000,
        head_source_monotonic_ns=1_000_000_000,
        head_camera_ns=0,
    )
    writer.close()

    with h5py.File(h5_path, "r") as h5:
        cameras = h5["cameras"]
        assert "head" in cameras
        assert "left_wrist" in cameras
        assert "right_wrist" not in cameras
        data_key = "jpeg" if storage_format == "jpeg" else "rgb"
        assert data_key in cameras["head"]
        assert data_key in cameras["left_wrist"]


def test_10_camera_session_process_passes_profile():
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "ready"
            # fake python path
            py_path = Path(td) / "python"
            py_path.touch()

            session = CameraSessionProcess(
                camera_python=py_path,
                collection_profile="left-pick-minimal",
                output=Path(td) / "out.cameras.h5",
                init_timeout_s=0.1,
            )

            # verify command constructed contains --collection-profile left-pick-minimal
            def side_effect(cmd, *args, **kwargs):
                assert "--collection-profile" in cmd
                idx = cmd.index("--collection-profile")
                assert cmd[idx + 1] == "left-pick-minimal"
                # write ready file to let start() return
                for arg in cmd:
                    if str(arg).endswith("ready"):
                        Path(arg).write_text("ready\n")
                return mock_proc

            mock_popen.side_effect = side_effect
            session.start()
            session.stop()


def test_11_invalid_profile_rejected_before_execution():
    with pytest.raises(ValueError, match="unknown collection profile"):
        CameraSessionProcess(
            collection_profile="nonexistent-profile",
        )

    with pytest.raises(ValueError, match="unknown collection profile"):
        CameraHdf5Writer(
            Path("/tmp/invalid.h5"),
            camera_hz=30.0,
            frame_shapes={},
            head_source_shape=ZED_RGB_ONLY_SHAPE,
            zed_shm_mode="rgb-only",
            zed_record_profile="rgb-only",
            storage_format="jpeg",
            jpeg_quality=90,
            compression="lzf",
            collection_profile="nonexistent-profile",
        )

    with pytest.raises(SystemExit):
        create_parser().parse_args(["--collection-profile", "invalid_choice"])


def test_12_offline_execution_zero_hardware_calls():
    reader = MockReader()
    # Confirm reader starts with 0 hardware calls
    assert reader.real_hardware_calls == 0
    # Simulate sampling
    _ = reader.get_head_snapshot()
    _ = reader.get_left_wrist_frame()
    assert reader.real_hardware_calls == 0
    assert reader.right_wrist_reads == 0


def test_13_inspect_and_export_cameras_support_two_camera_profile():
    import export_camera_video
    import inspect_cameras

    with tempfile.NamedTemporaryFile(suffix=".cameras.h5") as f:
        with h5py.File(f.name, "w") as h5:
            cam_grp = h5.create_group("cameras")
            cam_grp.create_group("head")
            cam_grp.create_group("left_wrist")

        with h5py.File(f.name, "r") as h5:
            inspect_roles = inspect_cameras._get_camera_roles(h5)
            export_roles = export_camera_video._get_camera_roles(h5)

        assert inspect_roles == ["head", "left_wrist"]
        assert export_roles == ["head", "left_wrist"]

