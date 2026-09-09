from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from interactive_record_session import (
    UnifiedTerminalController,
    create_parser,
    execute_auto_reset,
    read_joint_positions,
    release_gripper_fully,
    s_curve_quintic,
)


def test_1_cli_parser_defaults_and_custom_options():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.address == "192.168.30.1:50051"
    assert args.model == "a"
    assert args.noise_deg == 3.0
    assert args.noise_mode == "uniform"
    assert args.arm_only is True
    assert args.left_arm_2_offset_deg == -40.0
    assert args.gripper_host == "127.0.0.1"
    assert args.gripper_port == 8888
    assert args.gripper_id == 1
    assert args.invert_gripper is True
    assert args.collection_profile == "left-pick-minimal"
    assert args.reset_python == Path("/home/nvidia/arpa_h_demo_robot_side/.venv/bin/python")
    assert args.reset_script == Path("/home/nvidia/arpa_h_demo_robot_side/sample_and_move_to_ready.py")


def test_2_cli_parser_custom_overrides():
    parser = create_parser()
    custom_cmd = (
        "--output-dir /mnt/ssd/rby1_data/custom_test "
        "--prefix custom "
        "--noise-deg 5.0 "
        "--noise-mode normal "
        "--no-arm-only "
        "--left-arm-2-offset-deg -35.0 "
        "--gripper-port 9999 "
        "--gripper-id 0 "
        "--visualize-cameras "
        "--skip-initial-reset"
    )
    args = parser.parse_args(custom_cmd.split())
    assert args.output_dir == Path("/mnt/ssd/rby1_data/custom_test")
    assert args.prefix == "custom"
    assert args.noise_deg == 5.0
    assert args.noise_mode == "normal"
    assert args.arm_only is False
    assert args.left_arm_2_offset_deg == -35.0
    assert args.gripper_port == 9999
    assert args.gripper_id == 0
    assert args.visualize_cameras is True
    assert args.skip_initial_reset is True


def test_3_s_curve_quintic_properties():
    assert s_curve_quintic(0.0, 3.0) == 0.0
    assert s_curve_quintic(3.0, 3.0) == 1.0
    assert s_curve_quintic(4.0, 3.0) == 1.0
    assert s_curve_quintic(-1.0, 3.0) == 0.0

    # Monotonicity test
    ts = np.linspace(0, 3.0, 50)
    vals = [s_curve_quintic(t, 3.0) for t in ts]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1]


def test_4_terminal_controller_command_and_expiration():
    key_q = queue.Queue()
    stop_ev = threading.Event()
    controller = UnifiedTerminalController(
        key_queue=key_q,
        stop_event=stop_ev,
        linear_speed=0.20,
        angular_speed=0.40,
        timeout_s=0.10,
    )

    # Initially at rest
    cmd = controller.get_command()
    assert np.all(cmd == 0.0)

    # Set forward direction
    controller._set_direction(1.0, 0.0, "FORWARD")
    cmd = controller.get_command()
    assert np.isclose(cmd[0], 0.20)
    assert np.isclose(cmd[2], 0.0)

    # Immediate stop
    controller.set_stop()
    cmd = controller.get_command()
    assert np.all(cmd == 0.0)


def test_5_execute_auto_reset_subprocess_invocation():
    mock_args = argparse.Namespace(
        reset_python=Path("/bin/true"),
        reset_script=Path("/bin/true"),
        address="192.168.30.1:50051",
        model="a",
        noise_deg=3.0,
        noise_mode="uniform",
        reset_time=4.0,
        arm_only=True,
    )

    with patch("subprocess.run") as mock_run, patch("time.sleep"):
        mock_run.return_value = MagicMock(returncode=0)
        success = execute_auto_reset(mock_args)
        assert success is True

        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == str(Path("/bin/true").resolve())
        assert called_cmd[1] == str(Path("/bin/true").resolve())
        assert "--address" in called_cmd
        assert "192.168.30.1:50051" in called_cmd
        assert "--noise-deg" in called_cmd
        assert "3.0" in called_cmd
        assert "--arm-only" in called_cmd
        assert "--yes" in called_cmd


def test_6_release_gripper_fully():
    mock_client = MagicMock()
    with patch("time.sleep"):
        release_gripper_fully(mock_client, invert=True)
    mock_client.set_targets.assert_called_once_with(1.0, 1.0)

    mock_client.reset_mock()
    with patch("time.sleep"):
        release_gripper_fully(mock_client, invert=False)
    mock_client.set_targets.assert_called_once_with(0.0, 0.0)


def test_7_gripper_trigger_mapping():
    # Released (ADC 0) -> raw 0.0
    raw_released = np.array([0, 0], dtype=np.float64) / 1000.0
    # Squeezed (ADC 1000) -> raw 1.0
    raw_squeezed = np.array([1000, 1000], dtype=np.float64) / 1000.0

    # Inverted (Default: True) -> Squeezing gives 0.0 (closed on robot), released gives 1.0 (open on robot)
    assert np.allclose(np.clip(1.0 - raw_released, 0.0, 1.0), [1.0, 1.0])
    assert np.allclose(np.clip(1.0 - raw_squeezed, 0.0, 1.0), [0.0, 0.0])

    # Non-inverted
    assert np.allclose(np.clip(raw_released, 0.0, 1.0), [0.0, 0.0])
    assert np.allclose(np.clip(raw_squeezed, 0.0, 1.0), [1.0, 1.0])


def test_8_read_joint_positions_mock():
    mock_bus = MagicMock()
    mstate_0 = MagicMock()
    mstate_0.position = 0.5
    mstate_1 = MagicMock()
    mstate_1.position = -0.3
    mock_bus.get_motor_states.return_value = [(1, mstate_1), (0, mstate_0)]
    q = read_joint_positions(mock_bus, [0, 1])
    assert np.allclose(q, [0.5, -0.3])


