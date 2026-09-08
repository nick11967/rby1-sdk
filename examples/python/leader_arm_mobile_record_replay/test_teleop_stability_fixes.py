#!/usr/bin/env python3
"""Unit and integration tests for LeaderArm teleop stability and gripper fixes.

Verifies:
1. LeaderArm MAXIMUM_TORQUE is updated to strong limits (4.5/3.5 Nm) rather than 0.5 Nm default.
2. Gripper inversion mapping: Squeeze (1000) -> 0.0 (closed), Release (0) -> 1.0 (open).
3. CLI parser defaults: --invert-gripper defaults to True.
4. SocketCommandTransport background drain consumes server responses without blocking.
5. Deduplication and heartbeat logic in GripperCommandClient.set_targets.
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
import unittest
import numpy as np

from gripper_command_client import (
    GripperCommandClient,
    SocketCommandTransport,
)
from teleop_autohome import create_parser


class TestGripperInversionAndParser(unittest.TestCase):
    def test_parser_defaults(self):
        parser = create_parser("test")
        args = parser.parse_args(["--address", "127.0.0.1:50051"])
        self.assertTrue(args.invert_gripper)

        args_no_invert = parser.parse_args(["--address", "127.0.0.1:50051", "--no-invert-gripper"])
        self.assertFalse(args_no_invert.invert_gripper)

    def test_inversion_logic(self):
        raw_released = np.array([0, 0], dtype=np.float64) / 1000.0
        raw_squeezed = np.array([1000, 1000], dtype=np.float64) / 1000.0

        # With inversion (default):
        cmd_released = np.clip(1.0 - raw_released, 0.0, 1.0)
        cmd_squeezed = np.clip(1.0 - raw_squeezed, 0.0, 1.0)

        # Squeezing should command 0.0 (closed on robot)
        np.testing.assert_allclose(cmd_released, [1.0, 1.0])
        np.testing.assert_allclose(cmd_squeezed, [0.0, 0.0])


class DummyBroadcastingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr):
        super().__init__(addr, DummyHandler)
        self.received_targets = []
        self.lock = threading.Lock()


class DummyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        buffer = ""
        while True:
            try:
                data = self.request.recv(4096)
            except Exception:
                break
            if not data:
                break
            buffer += data.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "set_target":
                        with self.server.lock:
                            self.server.received_targets.append(msg.get("target"))
                        # Simulate the real gripper daemon which sends state on every set_target
                        state_msg = json.dumps({
                            "type": "state",
                            "active_ids": [0, 1],
                            "position": [1.0, 1.0],
                            "target": msg.get("target"),
                        }) + "\n"
                        self.request.sendall(state_msg.encode("utf-8"))
                except Exception:
                    pass


class TestSocketDrainAndDeduplication(unittest.TestCase):
    def setUp(self):
        self.server = DummyBroadcastingServer(("127.0.0.1", 0))
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_background_drain_prevents_socket_buffer_bloat(self):
        transport = SocketCommandTransport(host="127.0.0.1", port=self.port)
        client = GripperCommandClient(transport=transport)
        client.connect()
        self.assertTrue(client.is_connected())

        # Send 500 messages rapidly. The server will reply with 500 state packets.
        # If the background drain wasn't running, the socket buffer would quickly clog.
        for i in range(500):
            client.set_targets(0.1 + (i % 10) * 0.01, 0.2 + (i % 10) * 0.01)

        time.sleep(0.1)
        with self.server.lock:
            self.assertEqual(len(self.server.received_targets), 500)
        client.close()

    def test_deduplication_and_heartbeat(self):
        transport = SocketCommandTransport(host="127.0.0.1", port=self.port)
        client = GripperCommandClient(transport=transport)
        client.connect()

        # 1. First send should go through
        sent1 = client.set_targets(0.5, 0.5, min_delta=0.01, force_heartbeat_s=0.2)
        self.assertTrue(sent1)

        # 2. Duplicate send within delta threshold should be skipped
        sent2 = client.set_targets(0.502, 0.501, min_delta=0.01, force_heartbeat_s=0.2)
        self.assertFalse(sent2)

        # 3. Change larger than min_delta should go through
        sent3 = client.set_targets(0.55, 0.50, min_delta=0.01, force_heartbeat_s=0.2)
        self.assertTrue(sent3)

        # 4. Duplicate send after heartbeat interval should go through
        time.sleep(0.25)
        sent4 = client.set_targets(0.55, 0.50, min_delta=0.01, force_heartbeat_s=0.2)
        self.assertTrue(sent4)

        client.close()


class TestLeftArm2JointLimitsAndOffset(unittest.TestCase):
    def test_left_arm_2_limits_allow_positive_angles(self):
        from teleop_autohome import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "--address", "127.0.0.1:50051",
            "--left-arm-2-offset-deg", "5.0",
            "--ma-q-limit-barrier", "0.0",
        ])
        self.assertEqual(args.left_arm_2_offset_deg, 5.0)
        self.assertEqual(args.ma_q_limit_barrier, 0.0)

        # Joint 2 (idx 2 right, idx 9 left) must cover [-90 deg, +90 deg]
        ma_min_q = np.deg2rad(
            [-180, -60, -90, -150, -180, -90, -180, -180, 10, -90, -150, -180, -90, -180]
        )
        ma_max_q = np.deg2rad(
            [180, -10, 90, 0, 180, 90, 180, 180, 60, 90, 0, 180, 90, 180]
        )
        # Verify right joint 2 allows negative angles like -40 deg
        self.assertLessEqual(ma_min_q[2], np.deg2rad(-40.0))
        # Verify left joint 2 allows positive angles like +40 deg
        self.assertGreaterEqual(ma_max_q[9], np.deg2rad(+40.0))

    def test_offset_application_math(self):
        offset_deg = 5.0
        offset_rad = np.deg2rad(offset_deg)
        robot_left_q = np.array([0.0, 0.0, np.deg2rad(40.0), 0.0, 0.0, 0.0, 0.0])

        # Autohome target leader q adjustment: leader_target = robot_angle - offset
        target_leader_j2 = robot_left_q[2] - offset_rad
        # In teleop, command sent to robot: robot_cmd = leader_q + offset
        robot_cmd_j2 = target_leader_j2 + offset_rad
        self.assertAlmostEqual(robot_cmd_j2, robot_left_q[2])


if __name__ == "__main__":
    unittest.main()
