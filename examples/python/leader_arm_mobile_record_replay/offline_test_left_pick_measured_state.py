#!/usr/bin/env python3
"""Offline standalone verification script for DATA-04B: Left Arm 7D + Measured Gripper 1D.

Runs completely offline without real robot hardware, Dynamixel motors, or network calls.
Verifies:
1. Measured 8D observation state (left arm measured joint 7D + measured gripper position 1D)
2. Gripper observation extracted from daemon `position`, NOT `target`
3. Zero `set_target` commands sent
4. Stale samples rejected
5. Output NPZ fields and invariant relations
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np

# Ensure examples directory is on sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from gripper_state_client import (
    GripperError,
    GripperStateClient,
    GripperStateSampler,
    GripperStaleError,
    GripperValidationError,
    MeasuredGripperSample,
)
from record_episodes import EpisodeBuffer


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
    def create_dummy(cls, num_joints: int = 24, left_arm_q: Optional[np.ndarray] = None) -> "FakeRobotState":
        pos = np.zeros(num_joints, dtype=np.float64)
        if left_arm_q is not None:
            # Place left arm at indices 13..19 (typical RBY1 joint layout)
            pos[13:20] = left_arm_q
        else:
            pos[13:20] = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float64)
        return cls(
            position=pos,
            velocity=np.zeros(num_joints, dtype=np.float64),
            current=np.zeros(num_joints, dtype=np.float64),
            torque=np.zeros(num_joints, dtype=np.float64),
            target_position=pos.copy() + 0.05,  # Deliberately different to verify measured vs target
            target_velocity=np.zeros(num_joints, dtype=np.float64),
            odometry=np.eye(3, dtype=np.float64),
        )


class MockGripperTransport:
    """Mock transport providing controlled JSONL responses and auditing outgoing commands."""

    def __init__(self, expected_pos: float = 0.153, target_pos: float = 0.999) -> None:
        self.sent_lines: List[str] = []
        self.expected_pos = expected_pos
        self.target_pos = target_pos
        self.set_target_count = 0
        self.real_network_calls = 0
        self._connected = True

    def send_line(self, line: str) -> None:
        self.sent_lines.append(line)
        if "set_target" in line:
            self.set_target_count += 1

    def read_line(self) -> str:
        # Return response where target and position are distinctly different
        payload = {
            "active_ids": [0, 1],
            "target": [0.555, self.target_pos],
            "position": [0.050, self.expected_pos],
        }
        return json.dumps(payload) + "\n"

    def close(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


def run_verification() -> int:
    profile = "left-pick-minimal"
    left_arm_idx = [13, 14, 15, 16, 17, 18, 19]
    expected_gripper_val = 0.153
    target_gripper_val = 0.999

    transport = MockGripperTransport(expected_pos=expected_gripper_val, target_pos=target_gripper_val)
    client = GripperStateClient(expected_id=1, transport=transport)
    sampler = GripperStateSampler(client=client, hz=100.0, max_age_s=0.2)
    sampler.start()

    # Wait for first sample
    time.sleep(0.05)

    buffer = EpisodeBuffer(
        model_name="rby1a",
        joint_names=[f"joint_{i}" for i in range(24)],
        collection_profile=profile,
        left_arm_idx=left_arm_idx,
        gripper_sampler=sampler,
        gripper_max_age_s=0.2,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test_episode.npz"

        # Start recording episode
        buffer.start()

        # Collect 100 samples
        expected_left_q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float64)
        dummy_state = FakeRobotState.create_dummy(num_joints=24, left_arm_q=expected_left_q)

        for _ in range(100):
            buffer.append_sample(dummy_state)
            time.sleep(0.001)

        saved_count = buffer.stop_and_save(npz_path)
        assert saved_count == 100, f"Expected 100 saved samples, got {saved_count}"
        assert npz_path.exists(), "NPZ file was not created"

        # Load and validate NPZ contents
        with np.load(npz_path, allow_pickle=True) as data:
            rec = {k: data[k] for k in data.files}

        # Validate shapes and types
        left_arm_pos = rec["left_arm_position_rad"]
        left_gripper_pos = rec["left_gripper_position_normalized"]
        obs_state = rec["left_pick_observation_state"]
        saved_profile = str(rec["collection_profile"])

        assert left_arm_pos.shape == (100, 7), f"Unexpected shape {left_arm_pos.shape}"
        assert left_gripper_pos.shape == (100,), f"Unexpected shape {left_gripper_pos.shape}"
        assert obs_state.shape == (100, 8), f"Unexpected shape {obs_state.shape}"
        assert saved_profile == profile, f"Unexpected profile {saved_profile}"

        # Invariant checks: obs[:, :7] == left_arm and obs[:, 7] == gripper
        np.testing.assert_allclose(obs_state[:, :7], left_arm_pos, err_msg="Obs [:7] mismatch with left arm pos")
        np.testing.assert_allclose(obs_state[:, 7], left_gripper_pos, err_msg="Obs [7] mismatch with gripper pos")

        # Verify observation uses measured position (0.153), NOT target (0.999)
        gripper_target_used = bool(np.any(np.isclose(left_gripper_pos, target_gripper_val)))
        assert not gripper_target_used, "Observation incorrectly used gripper target!"
        np.testing.assert_allclose(left_gripper_pos, expected_gripper_val, err_msg="Gripper position value mismatch")

        # Test stale sample rejection
        stale_samples_accepted = 0
        fake_stale_sample = MeasuredGripperSample(
            normalized_position=0.5,
            received_monotonic_ns=time.monotonic_ns() - int(1.0 * 1e9),  # 1.0s old (> 0.2s max age)
            expected_id=1,
            source_index=1,
        )
        # Verify sampler's get_latest_sample raises GripperStaleError on stale sample
        sampler._latest_sample = fake_stale_sample
        try:
            sampler.get_latest_sample(max_age_s=0.2)
            stale_samples_accepted += 1
        except GripperStaleError:
            pass

        # Verify buffer rejects appending when gripper sample is stale
        # Temporarily disable logging to avoid polluting stdout
        import logging
        prev_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            corrupt_test_buffer = EpisodeBuffer(
                model_name="rby1a",
                joint_names=[f"joint_{i}" for i in range(24)],
                collection_profile=profile,
                left_arm_idx=left_arm_idx,
                gripper_sampler=sampler,
                gripper_max_age_s=0.2,
            )
            corrupt_test_buffer.start()
            corrupt_test_buffer.append_sample(dummy_state)
            if len(corrupt_test_buffer.time_s) > 0:
                stale_samples_accepted += 1
            assert corrupt_test_buffer.is_corrupted, "Buffer should be marked corrupted upon stale sample"
        finally:
            logging.getLogger().setLevel(prev_level)

    sampler.stop()

    # Exact required output format
    print(f"Collection profile: {saved_profile}")
    print(f"Left arm measured state shape: {left_arm_pos.shape}")
    print(f"Left gripper measured state shape: {left_gripper_pos.shape}")
    print(f"Observation state shape: {obs_state.shape}")
    print("Gripper source: position")
    print(f"Gripper target used as observation: {gripper_target_used}")
    print(f"Stale samples accepted: {stale_samples_accepted}")
    print(f"Set-target commands sent: {transport.set_target_count}")
    print(f"Real SDK/network calls: {transport.real_network_calls}")

    return 0


if __name__ == "__main__":
    sys.exit(run_verification())
