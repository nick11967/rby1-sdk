"""Grasp Detector for RB-Y1 UPC Gripper.

Evaluates grasp status based on:
- desired_closed (0.0=open, 1.0=closed)
- actual_closed (0.0=open, 1.0=closed)
- finger velocity (% stroke / s)
- stall duration (stalled against object for >= min_duration_s)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any


@dataclass(frozen=True)
class GraspResult:
    status: str  # "open" | "moving" | "grasped" | "closed_empty"
    grasped: bool
    desired_closed: float
    actual_closed: float
    object_width: float  # Fraction remaining open (1.0 - actual_closed)
    timestamp: float
    stall_duration: float = 0.0
    velocity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraspDetector:
    """Stateful detector evaluating real-time grasp and contact state."""

    def __init__(
        self,
        *,
        empty_threshold: float = 0.80,
        min_squeeze: float = 0.35,
        stall_diff: float = 0.12,
        velocity_threshold: float = 0.12,
        min_duration_s: float = 0.15,
    ) -> None:
        self.empty_threshold = empty_threshold
        self.min_squeeze = min_squeeze
        self.stall_diff = stall_diff
        self.velocity_threshold = velocity_threshold
        self.min_duration_s = min_duration_s

        self._last_time: float | None = None
        self._prev_actual: float | None = None
        self._stall_start_time: float | None = None

    def reset(self) -> None:
        self._last_time = None
        self._prev_actual = None
        self._stall_start_time = None

    def update(
        self,
        desired_closed: float,
        actual_closed: float,
        timestamp: float | None = None,
    ) -> GraspResult:
        now = timestamp if timestamp is not None else time.monotonic()

        # Clamp normalized inputs
        desired = float(min(max(desired_closed, 0.0), 1.0))
        actual = float(min(max(actual_closed, 0.0), 1.0))

        # 1. Velocity calculation
        velocity = 0.0
        if self._last_time is not None and self._prev_actual is not None:
            dt = max(now - self._last_time, 1e-4)
            velocity = abs(actual - self._prev_actual) / dt

        self._last_time = now
        self._prev_actual = actual

        is_moving = velocity > self.velocity_threshold

        # 2. Stall detection
        # Squeezing, partway closed, and actual lagging behind desired
        stall = desired - actual
        stall_candidate = (
            (desired > self.min_squeeze)
            and (actual < self.empty_threshold)
            and (stall > self.stall_diff)
            and (not is_moving)
        )

        stall_dur = 0.0
        if stall_candidate:
            if self._stall_start_time is None:
                self._stall_start_time = now
            stall_dur = now - self._stall_start_time
        else:
            self._stall_start_time = None

        # 3. Status determination
        grasped = False
        if actual >= self.empty_threshold:
            status = "closed_empty"
        elif stall_candidate and (stall_dur >= self.min_duration_s):
            status = "grasped"
            grasped = True
        elif is_moving:
            status = "moving"
        elif actual <= 0.25 and desired <= 0.25:
            status = "open"
        else:
            status = "open" if actual <= 0.25 else "moving"

        object_width = max(0.0, 1.0 - actual)

        return GraspResult(
            status=status,
            grasped=grasped,
            desired_closed=desired,
            actual_closed=actual,
            object_width=object_width,
            timestamp=now,
            stall_duration=stall_dur,
            velocity=velocity,
        )
