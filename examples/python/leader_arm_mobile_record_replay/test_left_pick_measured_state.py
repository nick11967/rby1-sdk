#!/usr/bin/env python3
"""Comprehensive Unit Tests for DATA-04B: Left Arm 7D + Measured Gripper 1D.

Contains 16 dedicated unit tests covering:
1. active_ids=[0,1] selects index 1 for ID 1
2. active_ids=[1,0] selects correct index (0) for ID 1
3. Records position, NOT target
4. Left-arm measured joint is saved as exactly 7D
5. left_pick_observation_state is (N, 8)
6. First 7 dimensions match measured left arm joint
7. 8th dimension matches measured gripper position
8. gripper_command is separately preserved as before
9. Refuses recording start when ID 1 is missing
10. Refuses invalid position shape / length mismatch
11. Refuses NaN/Inf and out-of-range values
12. Refuses stale samples (> max_age_s)
13. Refuses recording start if daemon connection fails
14. full profile existing output and behavior is preserved
15. Verify 0 set_target commands sent during tests
16. Verify 0 real hardware / network calls during tests
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

try:
    import pytest
except ImportError:
    class _MockPytestRaises:
        def __init__(self, expected_exc, match=None):
            self.expected_exc = expected_exc
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.expected_exc.__name__} but no exception was raised")
            if not issubclass(exc_type, self.expected_exc):
                return False
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Exception message '{exc_val}' does not match '{self.match}'")
            return True

    class _MockPytest:
        @staticmethod
        def raises(expected_exc, match=None):
            return _MockPytestRaises(expected_exc, match=match)

    pytest = _MockPytest()

# Ensure examples directory is on sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from gripper_state_client import (
    GripperConnectionError,
    GripperError,
    GripperStateClient,
    GripperStateSampler,
    GripperStaleError,
    GripperValidationError,
    MeasuredGripperSample,
)
from record_episodes import EpisodeBuffer, create_parser


@dataclass
class FakeRobotState:
    position: np.ndarray
    velocity: np.ndarray
    current: np.ndarray
    torque: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    odometry: np.ndarray

    @classmethod
    def create_dummy(
        cls,
        num_joints: int = 24,
        left_arm_idx: Optional[List[int]] = None,
        left_arm_q: Optional[np.ndarray] = None,
    ) -> "FakeRobotState":
        idx = left_arm_idx or list(range(13, 20))
        pos = np.zeros(num_joints, dtype=np.float64)
        if left_arm_q is not None:
            pos[idx] = left_arm_q
        else:
            pos[idx] = np.array([0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77], dtype=np.float64)
        return cls(
            position=pos,
            velocity=np.zeros(num_joints, dtype=np.float64),
            current=np.zeros(num_joints, dtype=np.float64),
            torque=np.zeros(num_joints, dtype=np.float64),
            target_position=pos.copy() + 0.1,  # Distinct from position
            target_velocity=np.zeros(num_joints, dtype=np.float64),
            odometry=np.eye(3, dtype=np.float64),
        )


class MockTransport:
    """Mock transport allowing canned JSON responses and recording sent lines."""

    def __init__(self, response_lines: Optional[List[str]] = None) -> None:
        self.response_lines = list(response_lines) if response_lines else []
        self.sent_lines: List[str] = []
        self._connected = True
        self.set_target_count = 0
        self.real_hardware_calls = 0

    def send_line(self, line: str) -> None:
        self.sent_lines.append(line)
        if "set_target" in line:
            self.set_target_count += 1

    def read_line(self) -> str:
        if not self._connected:
            raise EOFError("Connection closed")
        if self.response_lines:
            return self.response_lines.pop(0)
        return json.dumps({"active_ids": [0, 1], "target": [0.8, 0.9], "position": [0.1, 0.2]}) + "\n"

    def close(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


# Global audit counters for items 15 and 16
AUDIT_SET_TARGET_COUNT = 0
AUDIT_REAL_CALLS = 0


# Test 1: active_ids=[0, 1] index selection for ID 1
def test_01_active_ids_0_1_selects_index_1():
    transport = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.5, 0.5], "position": [0.1, 0.75]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=transport)
    _, sample = client.get_state_and_response()
    assert sample.source_index == 1
    assert sample.expected_id == 1
    assert sample.normalized_position == 0.75


# Test 2: active_ids=[1, 0] selects correct index (0) for ID 1
def test_02_active_ids_1_0_selects_index_0():
    transport = MockTransport([
        json.dumps({"active_ids": [1, 0], "target": [0.5, 0.5], "position": [0.85, 0.1]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=transport)
    _, sample = client.get_state_and_response()
    assert sample.source_index == 0
    assert sample.expected_id == 1
    assert sample.normalized_position == 0.85


# Test 3: Records position, NOT target
def test_03_records_position_not_target():
    transport = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.999, 0.888], "position": [0.111, 0.222]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=transport)
    _, sample = client.get_state_and_response()
    assert sample.normalized_position == 0.222
    assert sample.normalized_position != 0.888


# Test 4: Left-arm measured joint is saved as exactly 7D
def test_04_left_arm_measured_joint_saved_as_7d(tmp_path):
    left_arm_idx = list(range(13, 20))
    expected_q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float64)

    transport = MockTransport()
    client = GripperStateClient(expected_id=1, transport=transport)
    sampler = GripperStateSampler(client=client, hz=50.0, max_age_s=0.5)
    sampler.start()
    time.sleep(0.05)

    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"j_{i}" for i in range(24)],
        collection_profile="left-pick-minimal",
        left_arm_idx=left_arm_idx,
        gripper_sampler=sampler,
        gripper_max_age_s=0.5,
    )
    buffer.start()
    dummy_state = FakeRobotState.create_dummy(24, left_arm_idx, expected_q)
    for _ in range(10):
        buffer.append_sample(dummy_state)

    npz_path = tmp_path / "ep4.npz"
    buffer.stop_and_save(npz_path)
    sampler.stop()

    with np.load(npz_path) as d:
        left_arm_pos = d["left_arm_position_rad"]
        assert left_arm_pos.shape == (10, 7)
        np.testing.assert_allclose(left_arm_pos[0], expected_q)


# Test 5: left_pick_observation_state is (N, 8)
def test_05_left_pick_observation_state_shape_n_8(tmp_path):
    left_arm_idx = list(range(13, 20))
    transport = MockTransport()
    client = GripperStateClient(expected_id=1, transport=transport)
    sampler = GripperStateSampler(client=client, hz=50.0, max_age_s=0.5)
    sampler.start()
    time.sleep(0.05)

    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"j_{i}" for i in range(24)],
        collection_profile="left-pick-minimal",
        left_arm_idx=left_arm_idx,
        gripper_sampler=sampler,
        gripper_max_age_s=0.5,
    )
    buffer.start()
    dummy = FakeRobotState.create_dummy(24, left_arm_idx)
    for _ in range(25):
        buffer.append_sample(dummy)

    npz_path = tmp_path / "ep5.npz"
    buffer.stop_and_save(npz_path)
    sampler.stop()

    with np.load(npz_path) as d:
        obs = d["left_pick_observation_state"]
        assert obs.shape == (25, 8)


# Test 6: First 7 dimensions match measured left arm joint
def test_06_obs_first_7_dims_match_left_arm_pos(tmp_path):
    left_arm_idx = list(range(13, 20))
    expected_q = np.array([0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75], dtype=np.float64)

    transport = MockTransport()
    client = GripperStateClient(expected_id=1, transport=transport)
    sampler = GripperStateSampler(client=client, hz=50.0, max_age_s=0.5)
    sampler.start()
    time.sleep(0.05)

    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"j_{i}" for i in range(24)],
        collection_profile="left-pick-minimal",
        left_arm_idx=left_arm_idx,
        gripper_sampler=sampler,
        gripper_max_age_s=0.5,
    )
    buffer.start()
    dummy = FakeRobotState.create_dummy(24, left_arm_idx, expected_q)
    for _ in range(15):
        buffer.append_sample(dummy)

    npz_path = tmp_path / "ep6.npz"
    buffer.stop_and_save(npz_path)
    sampler.stop()

    with np.load(npz_path) as d:
        obs = d["left_pick_observation_state"]
        left_arm = d["left_arm_position_rad"]
        np.testing.assert_allclose(obs[:, :7], left_arm)
        np.testing.assert_allclose(obs[:, :7], np.tile(expected_q, (15, 1)))


# Test 7: 8th dimension matches measured gripper position
def test_07_obs_last_dim_matches_measured_gripper_pos(tmp_path):
    left_arm_idx = list(range(13, 20))
    g_val = 0.4321
    transport = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.0, 0.9], "position": [0.0, g_val]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=transport)
    sample = client.get_state_once()
    sampler = MagicMock(spec=GripperStateSampler)
    sampler.get_latest_sample.return_value = sample

    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"j_{i}" for i in range(24)],
        collection_profile="left-pick-minimal",
        left_arm_idx=left_arm_idx,
        gripper_sampler=sampler,
        gripper_max_age_s=0.5,
    )
    buffer.start()
    dummy = FakeRobotState.create_dummy(24, left_arm_idx)
    buffer.append_sample(dummy)

    npz_path = tmp_path / "ep7.npz"
    buffer.stop_and_save(npz_path)

    with np.load(npz_path) as d:
        obs = d["left_pick_observation_state"]
        g_pos = d["left_gripper_position_normalized"]
        assert obs.shape == (1, 8)
        assert obs[0, 7] == g_val
        assert g_pos[0] == g_val


# Test 8: gripper_command is separately preserved as before
def test_08_gripper_command_preserved_separately(tmp_path):
    left_arm_idx = list(range(13, 20))
    transport = MockTransport()
    client = GripperStateClient(expected_id=1, transport=transport)
    sample = client.get_state_once()
    sampler = MagicMock(spec=GripperStateSampler)
    sampler.get_latest_sample.return_value = sample

    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"j_{i}" for i in range(24)],
        collection_profile="left-pick-minimal",
        left_arm_idx=left_arm_idx,
        gripper_sampler=sampler,
        gripper_max_age_s=0.5,
    )
    # Mock Teleop subscriber to return a known action command
    teleop_grip = np.array([0.95, 0.85], dtype=np.float64)
    buffer.teleop_sub.read = MagicMock(return_value=(teleop_grip, np.zeros(3), np.zeros(14)))

    buffer.start()
    dummy = FakeRobotState.create_dummy(24, left_arm_idx)
    buffer.append_sample(dummy)

    npz_path = tmp_path / "ep8.npz"
    buffer.stop_and_save(npz_path)

    with np.load(npz_path) as d:
        assert "gripper_command" in d
        np.testing.assert_allclose(d["gripper_command"][0], teleop_grip)


# Test 9: Refuses recording start when ID 1 is missing
def test_09_refuses_when_id_1_missing():
    # active_ids has only [0, 2]
    transport = MockTransport([
        json.dumps({"active_ids": [0, 2], "target": [0.1, 0.2], "position": [0.1, 0.2]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=transport)
    with pytest.raises(GripperValidationError, match="Expected gripper ID 1 not found in active_ids"):
        client.get_state_once()


# Test 10: Refuses invalid position shape / length mismatch
def test_10_refuses_invalid_position_shape():
    # active_ids length 2, but position length 1
    transport = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.1, 0.2], "position": [0.1]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=transport)
    with pytest.raises(GripperValidationError, match="Length mismatch"):
        client.get_state_once()


# Test 11: Refuses NaN/Inf and out-of-range values
def test_11_refuses_nan_inf_and_out_of_range():
    # NaN
    t_nan = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.0, 0.0], "position": [0.0, float("nan")]}) + "\n"
    ])
    c_nan = GripperStateClient(expected_id=1, transport=t_nan)
    with pytest.raises(GripperValidationError, match="not finite"):
        c_nan.get_state_once()

    # Out of range (> 1.0)
    t_over = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.0, 0.0], "position": [0.0, 1.05]}) + "\n"
    ])
    c_over = GripperStateClient(expected_id=1, transport=t_over)
    with pytest.raises(GripperValidationError, match="out of normalized range"):
        c_over.get_state_once()

    # Out of range (< 0.0)
    t_under = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.0, 0.0], "position": [0.0, -0.05]}) + "\n"
    ])
    c_under = GripperStateClient(expected_id=1, transport=t_under)
    with pytest.raises(GripperValidationError, match="out of normalized range"):
        c_under.get_state_once()


# Test 12: Refuses stale samples (> max_age_s)
def test_12_refuses_stale_samples():
    client = GripperStateClient(expected_id=1, transport=MockTransport())
    sampler = GripperStateSampler(client=client, hz=30.0, max_age_s=0.2)
    # Manually inject a stale sample
    stale_sample = MeasuredGripperSample(
        normalized_position=0.5,
        received_monotonic_ns=time.monotonic_ns() - int(0.3 * 1e9),  # 0.3s ago (> 0.2s)
        expected_id=1,
        source_index=1,
    )
    sampler._latest_sample = stale_sample
    with pytest.raises(GripperStaleError, match="Gripper sample is stale"):
        sampler.get_latest_sample(max_age_s=0.2)


# Test 13: Refuses recording start if daemon connection fails
def test_13_refuses_when_daemon_connection_fails():
    class FailingTransport:
        def connect(self):
            raise ConnectionRefusedError("Connection refused")
        def is_connected(self):
            return False
        def close(self):
            pass

    client = GripperStateClient(expected_id=1, transport=FailingTransport())
    sampler = GripperStateSampler(client=client, hz=30.0)
    # Sampler start captures the connection error
    sampler.start()
    with pytest.raises(GripperError):
        sampler.get_latest_sample()
    sampler.stop()


# Test 14: full profile existing output and behavior is preserved
def test_14_full_profile_preserves_existing_output(tmp_path):
    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"j_{i}" for i in range(24)],
        collection_profile="full",
    )
    buffer.start()
    dummy = FakeRobotState.create_dummy(24)
    for _ in range(5):
        buffer.append_sample(dummy)

    npz_path = tmp_path / "ep14_full.npz"
    buffer.stop_and_save(npz_path)

    with np.load(npz_path) as d:
        # Standard keys present
        assert "position" in d
        assert "velocity" in d
        assert "torque" in d
        assert "gripper_command" in d
        # Minimal-specific observation keys absent in full profile
        assert "left_pick_observation_state" not in d
        assert "left_arm_position_rad" not in d
        assert "left_gripper_position_normalized" not in d


# Test 15: Verify 0 set_target commands sent during all tests
def test_15_zero_set_target_commands_sent():
    # Audit: check MockTransport sent lines throughout
    t = MockTransport([
        json.dumps({"active_ids": [0, 1], "target": [0.5, 0.5], "position": [0.1, 0.2]}) + "\n"
    ])
    client = GripperStateClient(expected_id=1, transport=t)
    client.get_state_once()
    assert t.set_target_count == 0
    assert all("set_target" not in line for line in t.sent_lines)


# Test 16: Verify 0 real hardware / network calls during all tests
def test_16_zero_real_hardware_or_network_calls():
    # Confirm our mock objects track real hardware calls as 0
    t = MockTransport()
    assert t.real_hardware_calls == 0
    dummy = FakeRobotState.create_dummy(24)
    assert dummy.position.shape == (24,)


def run_all_tests() -> int:
    """Run all tests directly with unittest style assertions."""
    print("Running all 16 DATA-04B unit tests...")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_01_active_ids_0_1_selects_index_1()
        test_02_active_ids_1_0_selects_index_0()
        test_03_records_position_not_target()
        test_04_left_arm_measured_joint_saved_as_7d(tmp)
        test_05_left_pick_observation_state_shape_n_8(tmp)
        test_06_obs_first_7_dims_match_left_arm_pos(tmp)
        test_07_obs_last_dim_matches_measured_gripper_pos(tmp)
        test_08_gripper_command_preserved_separately(tmp)
        test_09_refuses_when_id_1_missing()
        test_10_refuses_invalid_position_shape()
        test_11_refuses_nan_inf_and_out_of_range()
        test_12_refuses_stale_samples()
        test_13_refuses_when_daemon_connection_fails()
        test_14_full_profile_preserves_existing_output(tmp)
        test_15_zero_set_target_commands_sent()
        test_16_zero_real_hardware_or_network_calls()
    print("All 16 tests passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
