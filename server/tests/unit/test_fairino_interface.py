#!/usr/bin/env python3
##
# @file test_fairino_interface.py
#
# @brief Unit tests for FairinoInterface movej and movel offset behaviour.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import unittest
from unittest.mock import MagicMock, patch

# Internal library
from devices.fairino.interface import FairinoInterface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_iface(mock_robot_module):
    """Open a non-debug FairinoInterface backed by the given mock module."""
    mock_robot = MagicMock()
    mock_robot_module.RPC.return_value = mock_robot
    mock_robot.MoveJ.return_value = 0
    mock_robot.MoveL.return_value = 0
    iface = FairinoInterface(ip="192.168.58.2", debug=False)
    iface.open()
    return iface, mock_robot


# ---------------------------------------------------------------------------
# OPEN / CLOSE
# ---------------------------------------------------------------------------

class TestFairinoInterfaceOpenClose(unittest.TestCase):
    """Tests for open() and close() lifecycle."""

    @patch("devices.fairino.interface.Robot")
    def test_open_debug_does_not_call_rpc(self, mock_robot_module):
        iface = FairinoInterface(debug=True)
        iface.open()
        mock_robot_module.RPC.assert_not_called()
        self.assertTrue(iface.is_opened())

    @patch("devices.fairino.interface.Robot")
    def test_open_non_debug_calls_rpc_and_enable(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot_module.RPC.assert_called_once_with("192.168.58.2")
        mock_robot.RobotEnable.assert_called_once_with(1)
        self.assertTrue(iface.is_opened())

    @patch("devices.fairino.interface.Robot")
    def test_open_idempotent(self, mock_robot_module):
        iface, _ = _make_iface(mock_robot_module)
        iface.open()
        self.assertEqual(mock_robot_module.RPC.call_count, 1)

    def test_close_when_not_opened_is_safe(self):
        iface = FairinoInterface(debug=True)
        iface.close()  # should not raise
        self.assertFalse(iface.is_opened())

    def test_close_debug_sets_is_opened_false(self):
        iface = FairinoInterface(debug=True)
        iface.open()
        iface.close()
        self.assertFalse(iface.is_opened())

    @patch("devices.fairino.interface.Robot")
    def test_close_non_debug_calls_close_rpc(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        iface.close()
        mock_robot.CloseRPC.assert_called_once()
        self.assertFalse(iface.is_opened())


# ---------------------------------------------------------------------------
# MOVEJ
# ---------------------------------------------------------------------------

class TestFairinoInterfaceMovej(unittest.TestCase):
    """Unit tests for movej() offset behaviour."""

    @patch("devices.fairino.interface.time")
    def test_movej_debug_returns_true(self, mock_time):
        iface = FairinoInterface(debug=True)
        iface.open()
        result = iface.movej(1, 2, 3, 4, 5, 6)
        self.assertTrue(result)

    @patch("devices.fairino.interface.Robot")
    def test_movej_no_offset_calls_sdk_with_offset_flag_0(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        iface.movej(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        mock_robot.MoveJ.assert_called_once_with(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0, 0,
            vel=20.0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

    @patch("devices.fairino.interface.Robot")
    def test_movej_tool_offset_calls_sdk_with_offset_flag_2(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        offset = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        iface.movej(0, 0, 0, 0, 0, 0, tool_offset=offset)
        _, kwargs = mock_robot.MoveJ.call_args
        self.assertEqual(kwargs["offset_flag"], 2)
        self.assertEqual(kwargs["offset_pos"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    @patch("devices.fairino.interface.Robot")
    def test_movej_base_offset_calls_sdk_with_offset_flag_1(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        offset = [7.0, 8.0, 9.0, 0.0, 0.0, 0.0]
        iface.movej(0, 0, 0, 0, 0, 0, base_offset=offset)
        _, kwargs = mock_robot.MoveJ.call_args
        self.assertEqual(kwargs["offset_flag"], 1)
        self.assertEqual(kwargs["offset_pos"], [7.0, 8.0, 9.0, 0.0, 0.0, 0.0])

    @patch("devices.fairino.interface.Robot")
    def test_movej_both_offsets_raises_value_error(self, mock_robot_module):
        iface, _ = _make_iface(mock_robot_module)
        with self.assertRaises(ValueError):
            iface.movej(0, 0, 0, 0, 0, 0,
                        tool_offset=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        base_offset=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    @patch("devices.fairino.interface.Robot")
    def test_movej_returns_true_on_success(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot.MoveJ.return_value = 0
        self.assertTrue(iface.movej(0, 0, 0, 0, 0, 0))

    @patch("devices.fairino.interface.Robot")
    def test_movej_returns_false_on_sdk_error(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot.MoveJ.return_value = -1
        self.assertFalse(iface.movej(0, 0, 0, 0, 0, 0))

    @patch("devices.fairino.interface.Robot")
    def test_movej_returns_false_on_exception(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot.MoveJ.side_effect = Exception("connection lost")
        self.assertFalse(iface.movej(0, 0, 0, 0, 0, 0))

    @patch("devices.fairino.interface.Robot")
    def test_movej_custom_vel_passed_to_sdk(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        iface.movej(0, 0, 0, 0, 0, 0, vel=50.0)
        _, kwargs = mock_robot.MoveJ.call_args
        self.assertEqual(kwargs["vel"], 50.0)


# ---------------------------------------------------------------------------
# MOVEL
# ---------------------------------------------------------------------------

class TestFairinoInterfaceMovel(unittest.TestCase):
    """Unit tests for movel() offset behaviour."""

    @patch("devices.fairino.interface.time")
    def test_movel_debug_returns_true(self, mock_time):
        iface = FairinoInterface(debug=True)
        iface.open()
        result = iface.movel(1, 2, 3, 4, 5, 6)
        self.assertTrue(result)

    @patch("devices.fairino.interface.Robot")
    def test_movel_no_offset_calls_sdk_with_offset_flag_0(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        iface.movel(10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
        mock_robot.MoveL.assert_called_once_with(
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0], 0, 0,
            vel=20.0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

    @patch("devices.fairino.interface.Robot")
    def test_movel_tool_offset_calls_sdk_with_offset_flag_2(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        offset = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        iface.movel(0, 0, 0, 0, 0, 0, tool_offset=offset)
        _, kwargs = mock_robot.MoveL.call_args
        self.assertEqual(kwargs["offset_flag"], 2)
        self.assertEqual(kwargs["offset_pos"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    @patch("devices.fairino.interface.Robot")
    def test_movel_base_offset_calls_sdk_with_offset_flag_1(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        offset = [7.0, 8.0, 9.0, 0.0, 0.0, 0.0]
        iface.movel(0, 0, 0, 0, 0, 0, base_offset=offset)
        _, kwargs = mock_robot.MoveL.call_args
        self.assertEqual(kwargs["offset_flag"], 1)
        self.assertEqual(kwargs["offset_pos"], [7.0, 8.0, 9.0, 0.0, 0.0, 0.0])

    @patch("devices.fairino.interface.Robot")
    def test_movel_both_offsets_raises_value_error(self, mock_robot_module):
        iface, _ = _make_iface(mock_robot_module)
        with self.assertRaises(ValueError):
            iface.movel(0, 0, 0, 0, 0, 0,
                        tool_offset=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        base_offset=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    @patch("devices.fairino.interface.Robot")
    def test_movel_returns_true_on_success(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot.MoveL.return_value = 0
        self.assertTrue(iface.movel(0, 0, 0, 0, 0, 0))

    @patch("devices.fairino.interface.Robot")
    def test_movel_returns_false_on_sdk_error(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot.MoveL.return_value = -1
        self.assertFalse(iface.movel(0, 0, 0, 0, 0, 0))

    @patch("devices.fairino.interface.Robot")
    def test_movel_returns_false_on_exception(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        mock_robot.MoveL.side_effect = Exception("timeout")
        self.assertFalse(iface.movel(0, 0, 0, 0, 0, 0))

    @patch("devices.fairino.interface.Robot")
    def test_movel_custom_vel_passed_to_sdk(self, mock_robot_module):
        iface, mock_robot = _make_iface(mock_robot_module)
        iface.movel(0, 0, 0, 0, 0, 0, vel=75.0)
        _, kwargs = mock_robot.MoveL.call_args
        self.assertEqual(kwargs["vel"], 75.0)


if __name__ == "__main__":
    unittest.main()
