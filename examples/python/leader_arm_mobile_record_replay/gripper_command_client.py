#!/usr/bin/env python3
"""Gripper Command Client for RBY1 Teleoperation (DATA-05B).

This module provides a dedicated client for sending gripper actuation commands
(`set_target`) to the standalone Gripper Daemon over TCP JSONL.
- Does NOT access /dev/rby1_gripper or DynamixelBus directly.
- Transmits strictly `{"type": "set_target", "target": [right, left]}` queries.
- Array order is always [ID 0 (right), ID 1 (left)].
- No `id` field is sent over the wire.
- Validates input targets to normalized range [0.0, 1.0].
- Handles connection lifecycle, reconnects, and graceful shutdown.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union


class GripperCommandError(Exception):
    """Base exception for gripper command client errors."""


class GripperCommandConnectionError(GripperCommandError):
    """Raised when connection to gripper daemon fails or drops."""


class GripperCommandValidationError(GripperCommandError):
    """Raised when command parameters fail validation."""


class GripperCommandTransport(Protocol):
    """Abstract transport protocol to allow dependency injection and mocking."""

    def send_line(self, line: str) -> None:
        ...

    def close(self) -> None:
        ...

    def is_connected(self) -> bool:
        ...


class SocketCommandTransport:
    """TCP JSONL transport for sending commands to the gripper daemon."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888, timeout_s: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None
        self._wfile = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            if self._sock is not None and self._wfile is not None:
                return
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock
                self._wfile = sock.makefile("w", encoding="utf-8")
            except Exception as exc:
                self._close_locked()
                raise GripperCommandConnectionError(
                    f"Failed to connect to gripper daemon at {self.host}:{self.port}: {exc}"
                ) from exc

    def send_line(self, line: str) -> None:
        with self._lock:
            if self._wfile is None or self._sock is None:
                raise GripperCommandConnectionError("Transport is not connected")
            try:
                formatted = line if line.endswith("\n") else line + "\n"
                self._wfile.write(formatted)
                self._wfile.flush()
            except Exception as exc:
                self._close_locked()
                raise GripperCommandConnectionError(
                    f"Failed to send command line to gripper daemon: {exc}"
                ) from exc

    def is_connected(self) -> bool:
        with self._lock:
            return self._sock is not None and self._wfile is not None

    def _close_locked(self) -> None:
        if self._wfile is not None:
            try:
                self._wfile.close()
            except Exception:
                pass
            self._wfile = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()


class GripperCommandClient:
    """Client for controlling gripper positions via Gripper Daemon.

    Strict protocol:
    {"type": "set_target", "target": [right, left]}
    """

    RIGHT_GRIPPER_ID = 0
    LEFT_GRIPPER_ID = 1

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
        timeout_s: float = 2.0,
        transport: Optional[GripperCommandTransport] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.transport = transport or SocketCommandTransport(host=host, port=port, timeout_s=timeout_s)
        self._lock = threading.Lock()
        # Internal state tracking: [right (ID 0), left (ID 1)]
        self._current_targets: Dict[int, float] = {
            self.RIGHT_GRIPPER_ID: 0.0,
            self.LEFT_GRIPPER_ID: 0.0,
        }

    def connect(self) -> None:
        """Establish connection to the gripper daemon."""
        if hasattr(self.transport, "connect"):
            getattr(self.transport, "connect")()

    def close(self) -> None:
        """Close connection to the gripper daemon."""
        self.transport.close()

    def is_connected(self) -> bool:
        """Check if connected to the gripper daemon."""
        return self.transport.is_connected()

    @staticmethod
    def validate_target(target: Union[float, int]) -> float:
        """Validate and normalize a target gripper value within [0.0, 1.0]."""
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise GripperCommandValidationError(f"Target position must be numeric, got {type(target)}")
        target_val = float(target)
        if not math.isfinite(target_val):
            raise GripperCommandValidationError(f"Target position must be finite, got {target_val}")
        return float(min(max(target_val, 0.0), 1.0))

    @staticmethod
    def format_set_targets_message(right_target: float, left_target: float) -> str:
        """Create JSON string for set_target command: {"type": "set_target", "target": [right, left]}.

        Strict requirements:
        - No 'id' field is present.
        - 'target' is a list of exactly length 2.
        - target[0] is right gripper (ID 0), target[1] is left gripper (ID 1).
        """
        valid_right = GripperCommandClient.validate_target(right_target)
        valid_left = GripperCommandClient.validate_target(left_target)
        return json.dumps({
            "type": "set_target",
            "target": [round(valid_right, 4), round(valid_left, 4)],
        })

    def set_targets(self, right_target: float, left_target: float) -> None:
        """Send target commands for both right (ID 0) and left (ID 1) grippers in a single message."""
        msg = self.format_set_targets_message(right_target, left_target)
        with self._lock:
            if not self.is_connected():
                self.connect()
            self.transport.send_line(msg)
            self._current_targets[self.RIGHT_GRIPPER_ID] = GripperCommandClient.validate_target(right_target)
            self._current_targets[self.LEFT_GRIPPER_ID] = GripperCommandClient.validate_target(left_target)

    def set_target(self, dev_id: int, target: float) -> None:
        """Update target for one gripper ID and send full [right, left] message without 'id' field."""
        if dev_id not in (self.RIGHT_GRIPPER_ID, self.LEFT_GRIPPER_ID):
            raise GripperCommandValidationError(
                f"Device ID must be {self.RIGHT_GRIPPER_ID} (right) or {self.LEFT_GRIPPER_ID} (left), got {dev_id}"
            )
        with self._lock:
            right = target if dev_id == self.RIGHT_GRIPPER_ID else self._current_targets[self.RIGHT_GRIPPER_ID]
            left = target if dev_id == self.LEFT_GRIPPER_ID else self._current_targets[self.LEFT_GRIPPER_ID]
            msg = self.format_set_targets_message(right, left)
            if not self.is_connected():
                self.connect()
            self.transport.send_line(msg)
            self._current_targets[self.RIGHT_GRIPPER_ID] = GripperCommandClient.validate_target(right)
            self._current_targets[self.LEFT_GRIPPER_ID] = GripperCommandClient.validate_target(left)

    def set_command_array(self, command: Sequence[float]) -> None:
        """Send target array [right_target, left_target]."""
        if len(command) < 2:
            raise GripperCommandValidationError(f"Expected at least 2 command values, got {len(command)}")
        self.set_targets(float(command[0]), float(command[1]))

    @property
    def current_targets(self) -> Tuple[float, float]:
        """Return the current [right, left] targets."""
        with self._lock:
            return (
                self._current_targets[self.RIGHT_GRIPPER_ID],
                self._current_targets[self.LEFT_GRIPPER_ID],
            )

    # Compatibility methods mimicking leader_example.Gripper interface
    def initialize(self, verbose: bool = True) -> bool:
        """Initialize connection to daemon."""
        try:
            self.connect()
            if verbose:
                logging.info(f"Connected to Gripper Daemon at {self.host}:{self.port}")
            return True
        except Exception as exc:
            if verbose:
                logging.error(f"Failed to connect to Gripper Daemon at {self.host}:{self.port}: {exc}")
            return False

    def homing(self) -> bool:
        """No-op in daemon mode: homing is handled by the standalone daemon."""
        return True

    def start(self) -> None:
        """No-op in daemon mode: daemon is already running."""
        pass

    def stop(self) -> None:
        """Close connection on teleop shutdown."""
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gripper Command Client CLI / Diagnostic Tool")
    parser.add_argument("--host", default="127.0.0.1", help="Gripper daemon host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8888, help="Gripper daemon port (default: 8888)")
    parser.add_argument("--right", type=float, default=None, help="Target for right gripper (ID 0) [0.0..1.0]")
    parser.add_argument("--left", type=float, default=None, help="Target for left gripper (ID 1) [0.0..1.0]")
    parser.add_argument("--id", type=int, default=None, help="Gripper device ID (0=right, 1=left)")
    parser.add_argument("--target", type=float, default=None, help="Target normalized position [0.0..1.0]")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = GripperCommandClient(host=args.host, port=args.port)
    try:
        client.connect()
        if args.right is not None or args.left is not None:
            r = args.right if args.right is not None else 0.0
            l = args.left if args.left is not None else 0.0
            logging.info(f"Setting targets: right={r:.3f}, left={l:.3f}...")
            client.set_targets(r, l)
        elif args.id is not None and args.target is not None:
            logging.info(f"Setting target for ID {args.id} to {args.target:.3f}...")
            client.set_target(args.id, args.target)
        elif args.target is not None:
            logging.info(f"Setting both targets to {args.target:.3f}...")
            client.set_targets(args.target, args.target)
        else:
            logging.error("Please specify --right/--left or --id/--target")
            return 1
        logging.info("Command successfully sent to daemon.")
        return 0
    except Exception as exc:
        logging.error(f"Error: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
