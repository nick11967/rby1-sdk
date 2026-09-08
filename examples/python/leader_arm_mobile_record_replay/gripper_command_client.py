#!/usr/bin/env python3
"""Gripper Command Client for RBY1 Teleoperation (DATA-05A).

This module provides a dedicated client for sending gripper actuation commands
(`set_target`) to the standalone Gripper Daemon over TCP JSONL.
- Does NOT access /dev/rby1_gripper or DynamixelBus directly.
- Formats and sends `{"type": "set_target", "id": <id>, "target": <target>}` queries.
- Maps Left arm gripper to ID 1 and Right arm gripper to ID 0.
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
    """Client for controlling gripper positions via Gripper Daemon."""

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
        self._last_targets: Dict[int, float] = {}

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
    def format_set_target_message(dev_id: int, target: float) -> str:
        """Create JSON string for set_target command."""
        if isinstance(dev_id, bool) or not isinstance(dev_id, int) or dev_id < 0:
            raise GripperCommandValidationError(f"Device ID must be non-negative integer, got {dev_id}")
        valid_target = GripperCommandClient.validate_target(target)
        return json.dumps({"type": "set_target", "id": dev_id, "target": round(valid_target, 4)})

    def set_target(self, dev_id: int, target: float) -> None:
        """Send a set_target command for a specific gripper motor ID."""
        msg = self.format_set_target_message(dev_id, target)
        with self._lock:
            if not self.is_connected():
                self.connect()
            self.transport.send_line(msg)
            self._last_targets[dev_id] = float(target)

    def set_targets(self, right_target: float, left_target: float) -> None:
        """Send target commands for both right (ID 0) and left (ID 1) grippers."""
        self.set_target(self.RIGHT_GRIPPER_ID, right_target)
        self.set_target(self.LEFT_GRIPPER_ID, left_target)

    def set_command_array(self, command: Sequence[float]) -> None:
        """Send target array [right_target, left_target]."""
        if len(command) < 2:
            raise GripperCommandValidationError(f"Expected at least 2 command values, got {len(command)}")
        self.set_targets(float(command[0]), float(command[1]))

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
    parser.add_argument("--id", type=int, default=1, help="Gripper device ID (0=right, 1=left)")
    parser.add_argument("--target", type=float, required=True, help="Target normalized position [0.0..1.0]")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = GripperCommandClient(host=args.host, port=args.port)
    try:
        client.connect()
        logging.info(f"Setting target for ID {args.id} to {args.target:.3f}...")
        client.set_target(args.id, args.target)
        logging.info("Command successfully sent to daemon.")
        return 0
    except Exception as exc:
        logging.error(f"Error: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
