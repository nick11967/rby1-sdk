#!/usr/bin/env python3
"""Read-only Gripper State Client for RBY1 Left-Pick-Minimal Recorder (DATA-04B).

This module provides a strictly read-only client and background sampler for the
gripper daemon.
- Absolutely NO control commands (such as `set_target`) are ever sent.
- ONLY `{"type":"get_state"}` queries are transmitted over TCP JSONL.
- Parses and validates measured `position` from the daemon response, explicitly
  ignoring `target`.
- Enforces fail-closed validation: checks active_ids, position dimension, finite
  float within [0.0, 1.0], and freshness (< max_age_s).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass(frozen=True)
class MeasuredGripperSample:
    """Immutable sample representing measured gripper state."""

    normalized_position: float
    received_monotonic_ns: int
    expected_id: int
    source_index: int


class GripperError(Exception):
    """Base exception for gripper client errors."""


class GripperConnectionError(GripperError):
    """Raised when connection to gripper daemon fails."""


class GripperValidationError(GripperError):
    """Raised when gripper response fails schema or value validation."""


class GripperStaleError(GripperError):
    """Raised when the latest gripper sample exceeds allowable age."""


class GripperTransport(Protocol):
    """Abstract transport protocol to allow dependency injection for testing."""

    def send_line(self, line: str) -> None:
        ...

    def read_line(self) -> str:
        ...

    def close(self) -> None:
        ...

    def is_connected(self) -> bool:
        ...


class SocketTransport:
    """TCP JSONL transport implementing persistent connection to gripper daemon."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888, timeout_s: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None
        self._rfile = None
        self._wfile = None

    def connect(self) -> None:
        if self.is_connected():
            return
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            self._rfile = sock.makefile("r", encoding="utf-8")
            self._wfile = sock.makefile("w", encoding="utf-8")
        except Exception as exc:
            self.close()
            raise GripperConnectionError(
                f"Failed to connect to gripper daemon at {self.host}:{self.port}: {exc}"
            ) from exc

    def send_line(self, line: str) -> None:
        if self._wfile is None:
            raise GripperConnectionError("Transport is not connected")
        try:
            formatted = line if line.endswith("\n") else line + "\n"
            self._wfile.write(formatted)
            self._wfile.flush()
        except Exception as exc:
            self.close()
            raise GripperConnectionError(f"Failed to send line to gripper daemon: {exc}") from exc

    def read_line(self) -> str:
        if self._rfile is None:
            raise GripperConnectionError("Transport is not connected")
        try:
            line = self._rfile.readline()
            if not line:
                raise GripperConnectionError("Gripper daemon closed the connection (EOF)")
            return line
        except Exception as exc:
            self.close()
            if isinstance(exc, GripperError):
                raise
            raise GripperConnectionError(f"Failed to read line from gripper daemon: {exc}") from exc

    def is_connected(self) -> bool:
        return self._sock is not None and self._rfile is not None and self._wfile is not None

    def close(self) -> None:
        if self._wfile is not None:
            try:
                self._wfile.close()
            except Exception:
                pass
            self._wfile = None
        if self._rfile is not None:
            try:
                self._rfile.close()
            except Exception:
                pass
            self._rfile = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class GripperStateClient:
    """Read-only client for querying gripper daemon state."""

    GET_STATE_QUERY = json.dumps({"type": "get_state"})

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
        expected_id: int = 1,
        timeout_s: float = 2.0,
        transport: Optional[GripperTransport] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.expected_id = expected_id
        self.timeout_s = timeout_s
        self.transport = transport or SocketTransport(host=host, port=port, timeout_s=timeout_s)

    def connect(self) -> None:
        if hasattr(self.transport, "connect"):
            getattr(self.transport, "connect")()

    def close(self) -> None:
        self.transport.close()

    def is_connected(self) -> bool:
        return self.transport.is_connected()

    def parse_response(
        self, raw_data: Dict[str, Any], received_mono_ns: int
    ) -> MeasuredGripperSample:
        """Validate daemon JSON response and extract measured gripper observation."""
        if not isinstance(raw_data, dict):
            raise GripperValidationError(f"Expected JSON object response, got {type(raw_data)}")

        active_ids = raw_data.get("active_ids")
        if not isinstance(active_ids, list):
            raise GripperValidationError(
                f"Missing or invalid 'active_ids' in gripper response: {active_ids}"
            )

        if self.expected_id not in active_ids:
            raise GripperValidationError(
                f"Expected gripper ID {self.expected_id} not found in active_ids {active_ids}"
            )

        source_idx = active_ids.index(self.expected_id)

        positions = raw_data.get("position")
        if not isinstance(positions, list):
            raise GripperValidationError(
                f"Missing or invalid 'position' in gripper response: {positions}"
            )

        if len(positions) != len(active_ids):
            raise GripperValidationError(
                f"Length mismatch: len(position)={len(positions)} != len(active_ids)={len(active_ids)}"
            )

        raw_pos = positions[source_idx]
        if isinstance(raw_pos, bool) or not isinstance(raw_pos, (int, float)):
            raise GripperValidationError(
                f"Gripper position for ID {self.expected_id} (idx {source_idx}) is not numeric: {raw_pos}"
            )

        pos_val = float(raw_pos)
        if not math.isfinite(pos_val):
            raise GripperValidationError(
                f"Gripper position for ID {self.expected_id} is not finite: {pos_val}"
            )

        if pos_val < 0.0 or pos_val > 1.0:
            raise GripperValidationError(
                f"Gripper position {pos_val} out of normalized range [0.0, 1.0] (no clamping permitted)"
            )

        return MeasuredGripperSample(
            normalized_position=pos_val,
            received_monotonic_ns=received_mono_ns,
            expected_id=self.expected_id,
            source_index=source_idx,
        )

    def get_state_and_response(self) -> Tuple[Dict[str, Any], MeasuredGripperSample]:
        """Send get_state query and return raw response dict along with validated sample."""
        if not self.is_connected():
            self.connect()

        self.transport.send_line(self.GET_STATE_QUERY)
        recv_mono_ns = time.monotonic_ns()
        line = self.transport.read_line()

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GripperValidationError(f"Malformed JSON from gripper daemon: {line}") from exc

        sample = self.parse_response(data, recv_mono_ns)
        return data, sample

    def get_state_once(self) -> MeasuredGripperSample:
        """Fetch a single validated MeasuredGripperSample."""
        _, sample = self.get_state_and_response()
        return sample


class GripperStateSampler:
    """Threaded sampler continuously querying gripper daemon at fixed rate."""

    def __init__(
        self,
        client: GripperStateClient,
        hz: float = 30.0,
        max_age_s: float = 0.2,
    ) -> None:
        self.client = client
        self.hz = hz
        self.max_age_s = max_age_s
        self._period_s = 1.0 / hz if hz > 0 else 1.0 / 30.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_sample: Optional[MeasuredGripperSample] = None
        self._last_error: Optional[Exception] = None
        self._is_running = False

    def start(self) -> None:
        with self._lock:
            if self._is_running:
                return
            self._stop_event.clear()
            self._is_running = True
            # Attempt initial connection
            try:
                self.client.connect()
            except Exception as exc:
                self._last_error = exc

            self._thread = threading.Thread(
                target=self._run_loop, name="gripper-sampler", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        with self._lock:
            self._is_running = False
        try:
            self.client.close()
        except Exception:
            pass

    def is_connected(self) -> bool:
        return self.client.is_connected()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            t_start = time.monotonic()
            try:
                _, sample = self.client.get_state_and_response()
                with self._lock:
                    self._latest_sample = sample
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = exc
                    # On connection failure, close so next call can re-connect
                    if isinstance(exc, GripperConnectionError):
                        self.client.close()

            elapsed = time.monotonic() - t_start
            remaining = self._period_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def get_latest_sample(self, max_age_s: Optional[float] = None) -> MeasuredGripperSample:
        """Return the latest fresh MeasuredGripperSample or raise fail-closed error."""
        allowed_max_age = self.max_age_s if max_age_s is None else max_age_s

        with self._lock:
            sample = self._latest_sample
            err = self._last_error

        if sample is None:
            if err is not None:
                raise GripperError(f"No gripper sample available; daemon error: {err}") from err
            raise GripperError("No gripper sample received yet")

        now_mono_ns = time.monotonic_ns()
        age_s = (now_mono_ns - sample.received_monotonic_ns) / 1e9

        if age_s > allowed_max_age:
            detail = f" (last sampler error: {err})" if err is not None else ""
            raise GripperStaleError(
                f"Gripper sample is stale: age {age_s:.4f}s > {allowed_max_age:.4f}s{detail}"
            )

        return sample

    def check_health(self) -> None:
        """Check that the sampler is running, connected, and has a fresh sample."""
        if not self.is_connected():
            raise GripperConnectionError("Gripper daemon is not connected")
        self.get_latest_sample()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Gripper State Client / Diagnostic Tool (DATA-04B)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Gripper daemon host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8888, help="Gripper daemon port (default: 8888)")
    parser.add_argument(
        "--expected-id", type=int, default=1, help="Expected gripper ID to sample (default: 1)"
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="Connection timeout in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch a single gripper sample and exit",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Sampling rate in Hz for continuous monitoring (default: 30.0)",
    )
    args = parser.parse_args()

    client = GripperStateClient(
        host=args.host,
        port=args.port,
        expected_id=args.expected_id,
        timeout_s=args.timeout,
    )

    if args.once:
        try:
            client.connect()
            resp, sample = client.get_state_and_response()
            now_mono_ns = time.monotonic_ns()
            sample_age_ms = (now_mono_ns - sample.received_monotonic_ns) / 1e6
            print(f"active_ids: {resp.get('active_ids')}")
            print(f"expected_id: {sample.expected_id}")
            print(f"source_index: {sample.source_index}")
            print(f"measured_position: {sample.normalized_position:.3f}")
            print(f"sample_age_ms: {sample_age_ms:.1f}")
            return 0
        except Exception as exc:
            print(f"Error querying gripper daemon at {args.host}:{args.port}: {exc}", file=sys.stderr)
            return 1
        finally:
            client.close()

    # Continuous monitor
    print(f"Connecting to gripper daemon at {args.host}:{args.port} (expected_id: {args.expected_id})...")
    sampler = GripperStateSampler(client, hz=args.rate)
    sampler.start()

    try:
        while True:
            time.sleep(1.0 / args.rate)
            try:
                sample = sampler.get_latest_sample()
                age_ms = (time.monotonic_ns() - sample.received_monotonic_ns) / 1e6
                sys.stdout.write(
                    f"\rID: {sample.expected_id} | Index: {sample.source_index} | "
                    f"Pos: {sample.normalized_position:6.4f} | Age: {age_ms:5.1f}ms"
                )
                sys.stdout.flush()
            except GripperError as exc:
                sys.stdout.write(f"\r[!] {exc}")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopping sampler...")
    finally:
        sampler.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
