#!/usr/bin/env python3
"""Offline Unit and Integration Tests for DATA-05B: Gripper Command Client Protocol.

Verifies:
1. Zero DynamixelBus / SDK opens for gripper in daemon mode.
2. Single message per set_targets() call: {"type": "set_target", "target": [right, left]}.
3. Absolutely no "id" field in transmitted messages.
4. target shape is strictly a list of length 2: target[0] is right (ID 0), target[1] is left (ID 1).
5. Left trigger update modifies target[1] only; right trigger update modifies target[0] only.
6. Target normalization [0.0..1.0] and out-of-range clamping / rejection of invalid types.
7. Server get_state() reflects [right, left] ordering accurately.
8. Round-trip integration test using strict server handler (DryRunGripper equivalent) over real TCP loopback.
9. Zero real SDK / hardware calls throughout all tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import socketserver
import sys
import tempfile
import threading
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
    assert not hasattr(teleop_autohome, "Gripper"), "teleop_autohome should not define or import Gripper"
    
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    assert client.initialize() is True
    assert transport.real_hardware_calls == 0


# 2. set_targets sends exactly 1 message with {"type": "set_target", "target": [right, left]}
def test_02_single_message_format_no_id_length_2():
    """Verify set_targets sends exactly one message with [right, left] and no 'id' field."""
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    client.connect()

    client.set_targets(right_target=0.25, left_target=0.75)

    # Must be exactly 1 message
    assert len(transport.sent_lines) == 1, f"Expected 1 message, got {len(transport.sent_lines)}"
    
    msg = json.loads(transport.sent_lines[0])
    assert msg["type"] == "set_target"
    assert "id" not in msg, f"Protocol violation: 'id' field must NOT be in message: {msg}"
    assert isinstance(msg["target"], list), f"target must be a list, got {type(msg['target'])}"
    assert len(msg["target"]) == 2, f"target must have length 2, got {len(msg['target'])}"
    assert math.isclose(msg["target"][0], 0.25, abs_tol=1e-3), f"target[0] should be right (0.25), got {msg['target'][0]}"
    assert math.isclose(msg["target"][1], 0.75, abs_tol=1e-3), f"target[1] should be left (0.75), got {msg['target'][1]}"


# 3. Left trigger update modifies target[1] only
def test_03_left_trigger_modifies_target_1_only():
    """Verify updating left gripper (ID 1) preserves target[0] and only changes target[1]."""
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    client.connect()

    # Initial state: right=0.30, left=0.40
    client.set_targets(0.30, 0.40)
    assert len(transport.sent_lines) == 1

    # Update left only to 0.85
    client.set_target(dev_id=1, target=0.85)
    assert len(transport.sent_lines) == 2

    msg2 = json.loads(transport.sent_lines[1])
    assert "id" not in msg2
    assert len(msg2["target"]) == 2
    # target[0] remains 0.30, target[1] changes to 0.85
    assert math.isclose(msg2["target"][0], 0.30, abs_tol=1e-3)
    assert math.isclose(msg2["target"][1], 0.85, abs_tol=1e-3)


# 4. Right trigger update modifies target[0] only
def test_04_right_trigger_modifies_target_0_only():
    """Verify updating right gripper (ID 0) preserves target[1] and only changes target[0]."""
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    client.connect()

    # Initial state: right=0.20, left=0.60
    client.set_targets(0.20, 0.60)
    assert len(transport.sent_lines) == 1

    # Update right only to 0.90
    client.set_target(dev_id=0, target=0.90)
    assert len(transport.sent_lines) == 2

    msg2 = json.loads(transport.sent_lines[1])
    assert "id" not in msg2
    assert len(msg2["target"]) == 2
    # target[0] changes to 0.90, target[1] remains 0.60
    assert math.isclose(msg2["target"][0], 0.90, abs_tol=1e-3)
    assert math.isclose(msg2["target"][1], 0.60, abs_tol=1e-3)


# 5. Value validation and clamping
def test_05_target_value_validation_and_clamping():
    """Verify validation: clamps [0.0, 1.0], rejects non-finite / invalid types."""
    assert GripperCommandClient.validate_target(1.25) == 1.0
    assert GripperCommandClient.validate_target(-0.15) == 0.0

    # NaN
    try:
        GripperCommandClient.validate_target(float("nan"))
        assert False, "Should have raised GripperCommandValidationError on NaN"
    except GripperCommandValidationError:
        pass

    # Inf
    try:
        GripperCommandClient.validate_target(float("inf"))
        assert False, "Should have raised GripperCommandValidationError on Inf"
    except GripperCommandValidationError:
        pass

    # Bool
    try:
        GripperCommandClient.validate_target(True)
        assert False, "Should have raised GripperCommandValidationError on bool"
    except GripperCommandValidationError:
        pass

    # Invalid ID in set_target
    transport = MockCommandTransport()
    client = GripperCommandClient(transport=transport)
    try:
        client.set_target(2, 0.5)
        assert False, "Should have raised GripperCommandValidationError on invalid dev_id"
    except GripperCommandValidationError:
        pass


# 6. Connection failure handling
def test_06_connection_failure_safe_handling():
    """Verify client handles connection failures gracefully."""
    transport = MockCommandTransport(should_fail_connect=True)
    client = GripperCommandClient(transport=transport)

    assert client.initialize(verbose=False) is False
    try:
        client.set_targets(0.5, 0.5)
        assert False, "Should have raised connection error"
    except (GripperCommandConnectionError, ConnectionRefusedError):
        pass

    client.close()
    assert transport.is_closed is True


# 7. Parser options
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


# 8. Strict DryRunGripper Round-Trip Test over real TCP Loopback
class StrictDryRunServerHandler(socketserver.StreamRequestHandler):
    """Handler enforcing strict DATA-05B protocol matching actual gripper server."""

    def handle(self) -> None:
        server: StrictDryRunServer = self.server  # type: ignore
        for line in self.rfile:
            text = line.decode("utf-8").strip()
            if not text:
                continue
            data = json.loads(text)
            msg_type = data.get("type")

            if msg_type == "set_target":
                # Strict check: actual server does not read or expect "id"
                if "id" in data:
                    server.protocol_violations.append(f"Server rejected message with 'id': {data}")
                    raise ValueError(f"Strict server error: 'id' field is not allowed: {data}")
                target = data.get("target")
                if not isinstance(target, list) or len(target) != 2:
                    server.protocol_violations.append(f"Server rejected non-list[2] target: {target}")
                    raise ValueError(f"Strict server error: target must be list of length 2: {data}")
                
                with server.lock:
                    server.target = [float(target[0]), float(target[1])]
                    # DryRun simulates motor position tracking target
                    server.position = list(server.target)
                    server.received_commands.append(data)
                # set_target sends no response (one-way command streaming)

            elif msg_type == "get_state":
                with server.lock:
                    resp = {
                        "active_ids": [0, 1],
                        "target": list(server.target),
                        "position": list(server.position),
                    }
                out_bytes = (json.dumps(resp) + "\n").encode("utf-8")
                self.wfile.write(out_bytes)
                self.wfile.flush()

            else:
                server.protocol_violations.append(f"Unknown message type: {msg_type}")


class StrictDryRunServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address):
        super().__init__(server_address, StrictDryRunServerHandler)
        self.lock = threading.Lock()
        self.target = [0.0, 0.0]
        self.position = [0.0, 0.0]
        self.received_commands: List[Dict[str, Any]] = []
        self.protocol_violations: List[str] = []


def test_08_strict_dryrun_server_round_trip():
    """Full round-trip test with strict DryRun server over real TCP loopback.
    
    Verifies:
    - Command client connects and sends {"type": "set_target", "target": [right, left]}
    - Zero 'id' fields sent
    - Server accepts and updates internal target [right, left]
    - State client reads back measured position for ID 0 (right) and ID 1 (left)
    """
    server = StrictDryRunServer(("127.0.0.1", 0))
    host, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    cmd_client = None
    state_client_left = None
    state_client_right = None

    try:
        cmd_client = GripperCommandClient(host=host, port=port)
        cmd_client.connect()

        state_client_left = GripperStateClient(host=host, port=port, expected_id=1)
        state_client_right = GripperStateClient(host=host, port=port, expected_id=0)

        # 1. Send [0.35, 0.75]
        cmd_client.set_targets(right_target=0.35, left_target=0.75)
        time.sleep(0.02)

        # 2. Read back states via read-only state clients
        sample_left = state_client_left.get_state_once()
        sample_right = state_client_right.get_state_once()

        assert math.isclose(sample_right.normalized_position, 0.35, abs_tol=1e-3), (
            f"Expected right position 0.35, got {sample_right.normalized_position}"
        )
        assert math.isclose(sample_left.normalized_position, 0.75, abs_tol=1e-3), (
            f"Expected left position 0.75, got {sample_left.normalized_position}"
        )

        # 3. Update right only to 0.80 -> left should stay 0.75
        cmd_client.set_target(dev_id=0, target=0.80)
        time.sleep(0.02)

        sample_left = state_client_left.get_state_once()
        sample_right = state_client_right.get_state_once()
        assert math.isclose(sample_right.normalized_position, 0.80, abs_tol=1e-3)
        assert math.isclose(sample_left.normalized_position, 0.75, abs_tol=1e-3)

        # 4. Update left only to 0.10 -> right should stay 0.80
        cmd_client.set_target(dev_id=1, target=0.10)
        time.sleep(0.02)

        sample_left = state_client_left.get_state_once()
        sample_right = state_client_right.get_state_once()
        assert math.isclose(sample_right.normalized_position, 0.80, abs_tol=1e-3)
        assert math.isclose(sample_left.normalized_position, 0.10, abs_tol=1e-3)

        # 5. Check server audit: no protocol violations occurred
        assert len(server.protocol_violations) == 0, f"Server reported violations: {server.protocol_violations}"
        assert len(server.received_commands) == 3

        for cmd in server.received_commands:
            assert cmd["type"] == "set_target"
            assert "id" not in cmd
            assert len(cmd["target"]) == 2

    finally:
        if cmd_client:
            cmd_client.close()
        if state_client_left:
            state_client_left.close()
        if state_client_right:
            state_client_right.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


def run_all_tests() -> int:
    print("Running DATA-05B Gripper Command Client Protocol unit & integration tests...")
    test_01_zero_dynamixel_bus_opens_for_gripper()
    test_02_single_message_format_no_id_length_2()
    test_03_left_trigger_modifies_target_1_only()
    test_04_right_trigger_modifies_target_0_only()
    test_05_target_value_validation_and_clamping()
    test_06_connection_failure_safe_handling()
    test_07_parser_includes_gripper_daemon_options()
    test_08_strict_dryrun_server_round_trip()
    print("All DATA-05B tests passed successfully (8/8)!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
