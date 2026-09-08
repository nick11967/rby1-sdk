"""Unit tests for GraspDetector (TQM-06a)."""

from pathlib import Path
import sys
import unittest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grasp_detector import GraspDetector, GraspResult


class TestGraspDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GraspDetector(
            empty_threshold=0.80,
            min_squeeze=0.35,
            stall_diff=0.12,
            velocity_threshold=0.12,
            min_duration_s=0.15,
        )

    def test_open_state(self):
        """When gripper is open and not commanded, status should be 'open'."""
        res = self.detector.update(desired_closed=0.0, actual_closed=0.0, timestamp=1.0)
        self.assertIsInstance(res, GraspResult)
        self.assertEqual(res.status, "open")
        self.assertFalse(res.grasped)
        self.assertAlmostEqual(res.object_width, 1.0)

    def test_closed_empty_air(self):
        """When gripper closes beyond empty threshold (0.80), detect closed_empty."""
        self.detector.update(desired_closed=1.0, actual_closed=0.50, timestamp=1.0)
        res = self.detector.update(desired_closed=1.0, actual_closed=0.85, timestamp=1.1)
        self.assertEqual(res.status, "closed_empty")
        self.assertFalse(res.grasped)

    def test_moving_state(self):
        """When fingers are closing fast, status is 'moving'."""
        self.detector.update(desired_closed=1.0, actual_closed=0.10, timestamp=1.0)
        # Fast movement: 0.10 -> 0.30 in 0.05s = 4.0 /s
        res = self.detector.update(desired_closed=1.0, actual_closed=0.30, timestamp=1.05)
        self.assertEqual(res.status, "moving")
        self.assertFalse(res.grasped)

    def test_successful_object_grasp(self):
        """When fingers stall on object partway (< 0.80) for >= 0.15s, detect grasped."""
        t = 1.0
        # Start closing: moving from 0.10 to 0.45
        self.detector.update(desired_closed=1.0, actual_closed=0.10, timestamp=t)
        t += 0.05
        # Reached object at actual=0.45, but was moving during this dt (0.35 / 0.05 = 7.0/s)
        res = self.detector.update(desired_closed=1.0, actual_closed=0.45, timestamp=t)
        self.assertFalse(res.grasped)

        # Still at 0.45 at t=1.10 (velocity drops to 0, stall start time recorded at 1.10)
        t += 0.05
        res = self.detector.update(desired_closed=1.0, actual_closed=0.45, timestamp=t)
        self.assertFalse(res.grasped)

        # At t=1.20 (stall duration = 0.10s < 0.15s)
        t += 0.10
        res = self.detector.update(desired_closed=1.0, actual_closed=0.45, timestamp=t)
        self.assertFalse(res.grasped)

        # At t=1.26 (stall duration = 0.16s >= 0.15s)
        t += 0.06
        res = self.detector.update(desired_closed=1.0, actual_closed=0.45, timestamp=t)
        self.assertTrue(res.grasped)
        self.assertEqual(res.status, "grasped")
        self.assertAlmostEqual(res.actual_closed, 0.45)
        self.assertAlmostEqual(res.object_width, 0.55)
        self.assertAlmostEqual(res.desired_closed, 1.0)

    def test_insufficient_squeeze_not_grasped(self):
        """When desired squeeze is weak (<= 0.35), should not trigger grasp."""
        t = 1.0
        self.detector.update(desired_closed=0.30, actual_closed=0.10, timestamp=t)
        t += 0.20
        res = self.detector.update(desired_closed=0.30, actual_closed=0.10, timestamp=t)
        self.assertFalse(res.grasped)


if __name__ == "__main__":
    unittest.main()
