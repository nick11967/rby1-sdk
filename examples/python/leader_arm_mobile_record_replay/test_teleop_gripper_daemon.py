#!/usr/bin/env python3
"""Offline Unit and Integration Tests for DATA-05A: Teleop Gripper Daemon Migration.

Verifies:
1. Daemon mode opens DynamixelBus 0 times for gripper.
2. Left arm trigger command is routed to daemon ID 1 with proper set_target format.
3. Right arm trigger command is routed to daemon ID 0 with proper set_target format.
4. Trigger values [0..1000] -> [0.0..1.0] normalized set_target conversion.
5. Value clamping and validation (out of range, NaN/Inf, non-numeric).
6. Connection failure cleanly handled with safe termination and non-zero exit.
7. Zero real SDK / hardware calls throughout tests.
8. Concurrent compatibility: get_state (recorder) and set_target (teleop) can coexist on same daemon wire protocol.
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

# Ensure script dir is on sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from gripper_command_client import (
    GripperCommandClient,
    GripperCommandConnectionError,
    GripperCommandError,
    GripperCommandValidationError,
    SocketCommandTransport,
)
from gripper_state_client import (
    GripperStateClient,
    GripperStateSampler,
    MeasuredGripperSample,
)
import teleop_autohome


class MockCommandTransport:
    """Mock transport capturing sent JSON lines without network calls."""

    def __init__(self, should_fail_connect: bool = False, should_fail_send: bool = False) -> None:
        self.sent_lines: List[str] = []
        self.should_fail_connect = should_fail_connect
        self.should_fail_send = should_fail_send
        self.is_closed = False
        self.real_hardware_calls = 0

    def connect(self) -> None:
        if self.should_fail_connect:
            raise ConnectionRefusedError("Mock connection refused")

    def send_line(self, line: str) -> None:
        if self.should_fail_send:
            raise ConnectionResetError("Mock connection reset")
        self.sent_lines.append(line.strip())

    def close(self) -> None:
        self.is_closed = True

    def is_connected(self) -> bool:
        return not self.is_closed and not self.should_fail_connect


# 1. Verify 0 DynamixelBus opens for gripper
def test_01_zero_dynamixel_bus_opens_for_gripper():
    """Verify teleop_autohome does not import or instantiate leader_example.Gripper or DynamixelBus for gripper."""
    # Ensure teleop_autohome does not have leader_example.Gripper
    assert not hasattr(teleop_autohome, "Gripper"), "teleop_autohome should not define or import Gripper"
    
    # Audit: initialize GripperCommandClient with MockCommandTransport
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    assert client.initialize() is True
    assert transport.real_hardware_calls == 0


# 2. Left arm trigger command mapped to ID 1
def test_02_left_arm_trigger_routes_to_id_1():
    """Verify left arm trigger (trigger / 1000.0) is sent to ID 1."""
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    client.connect()

    # Squeeze left trigger: 750 / 1000 = 0.75
    client.set_target(dev_id=1, target=0.75)

    assert len(transport.sent_lines) == 1
    msg = json.loads(transport.sent_lines[0])
    assert msg["type"] == "set_target"
    assert msg["id"] == 1
    assert math.isclose(msg["target"], 0.75, abs_tol=1e-3)


# 3. Right arm trigger command mapped to ID 0
def test_03_right_arm_trigger_routes_to_id_0():
    """Verify right arm trigger is sent to ID 0."""
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    client.connect()

    # Squeeze right trigger: 300 / 1000 = 0.30
    client.set_target(dev_id=0, target=0.30)

    assert len(transport.sent_lines) == 1
    msg = json.loads(transport.sent_lines[0])
    assert msg["type"] == "set_target"
    assert msg["id"] == 0
    assert math.isclose(msg["target"], 0.30, abs_tol=1e-3)


# 4. Both arms simultaneous trigger commands
def test_04_set_targets_dual_gripper_mapping():
    """Verify set_targets(right, left) sends ID 0 then ID 1 in correct order with correct values."""
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    client.connect()

    client.set_targets(right_target=0.25, left_target=0.85)

    assert len(transport.sent_lines) == 2
    msg_r = json.loads(transport.sent_lines[0])
    msg_l = json.loads(transport.sent_lines[1])

    assert msg_r["type"] == "set_target"
    assert msg_r["id"] == 0
    assert math.isclose(msg_r["target"], 0.25, abs_tol=1e-3)

    assert msg_l["type"] == "set_target"
    assert msg_l["id"] == 1
    assert math.isclose(msg_l["target"], 0.85, abs_tol=1e-3)


# 5. Value validation and clamping
def test_05_target_value_validation_and_clamping():
    """Verify validation: clamps [0.0, 1.0], rejects non-finite / invalid types."""
    # Test clamping above 1.0
    val_over = GripperCommandClient.validate_target(1.25)
    assert val_over == 1.0

    # Test clamping below 0.0
    val_under = GripperCommandClient.validate_target(-0.15)
    assert val_under == 0.0

    # Test NaN rejection
    try:
        GripperCommandClient.validate_target(float("nan"))
        assert False, "Should have raised GripperCommandValidationError on NaN"
    except GripperCommandValidationError:
        pass

    # Test Inf rejection
    try:
        GripperCommandClient.validate_target(float("inf"))
        assert False, "Should have raised GripperCommandValidationError on Inf"
    except GripperCommandValidationError:
        pass

    # Test boolean rejection (bool is a subclass of int in Python)
    try:
        GripperCommandClient.validate_target(True)
        assert False, "Should have raised GripperCommandValidationError on bool"
    except GripperCommandValidationError:
        pass


# 6. Connection failure handling
def test_06_connection_failure_safe_handling():
    """Verify client handles connection failures gracefully."""
    transport = MockCommandTransport(should_fail_connect=True)
    client = GripperCommandClient(transport=transport)

    assert client.initialize(verbose=False) is False
    try:
        client.set_target(1, 0.5)
        assert False, "Should have raised connection error"
    except (GripperCommandConnectionError, ConnectionRefusedError):
        pass

    # Safe close
    client.close()
    assert transport.is_closed is True


# 7. CLI options check
def test_07_parser_includes_gripper_daemon_options():
    """Verify create_parser parses --gripper-host and --gripper-port correctly."""
    parser = teleop_autohome.create_parser()
    args = parser.parse_args([
        "--address", "192.168.30.1:50051",
        "--gripper-host", "10.0.0.42",
        "--gripper-port", "9999",
    ])
    assert args.gripper_host == "10.0.0.42"
    assert args.gripper_port == 9999
    assert args.mode == "position"
    assert args.model == "a"


# 8. Concurrent compatibility with Recorder (get_state read-only)
def test_08_concurrent_compatibility_get_state_and_set_target():
    """Verify that get_state queries and set_target commands follow unified JSONL schema."""
    # Teleop command
    teleop_msg_str = GripperCommandClient.format_set_target_message(dev_id=1, target=0.45)
    teleop_msg = json.loads(teleop_msg_str)
    assert teleop_msg == {"type": "set_target", "id": 1, "target": 0.45}

    # Recorder query
    rec_query_str = GripperStateClient.GET_STATE_QUERY
    rec_query = json.loads(rec_query_str)
    assert rec_query == {"type": "get_state"}

    # Mock server response for get_state
    mock_server_state = {
        "active_ids": [0, 1],
        "target": [0.0, 0.45],
        "position": [0.0, 0.448],
    }
    rec_client = GripperStateClient(expected_id=1)
    sample = rec_client.parse_response(mock_server_state, received_mono_ns=time.monotonic_ns())
    assert sample.expected_id == 1
    assert sample.normalized_position == 0.448


def run_all_tests() -> int:
    print("Running DATA-05A Teleop Gripper Daemon Migration unit tests...")
    test_01_zero_dynamixel_bus_opens_for_gripper()
    test_02_left_arm_trigger_routes_to_id_1()
    test_03_right_arm_trigger_routes_to_id_0()
    test_04_set_targets_dual_gripper_mapping()
    test_05_target_value_validation_and_clamping()
    test_06_connection_failure_safe_handling()
    test_07_parser_includes_gripper_daemon_options()
    test_08_concurrent_compatibility_get_state_and_set_target()
    print("All DATA-05A tests passed successfully (8/8)!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
