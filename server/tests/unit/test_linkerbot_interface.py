#!/usr/bin/env python3
##
# @file test_linkerbot_interface.py
#
# @brief Unit tests for the LinkerbotInterface class.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/18.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import unittest
from unittest.mock import patch, MagicMock

# Internal library
from devices.linkerbot.interface import LinkerbotInterface


class TestLinkerbotInterface(unittest.TestCase):
    """! Unit tests for LinkerbotInterface."""

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def test_init_defaults(self):
        iface = LinkerbotInterface()
        self.assertEqual(iface._hand_type, "left")
        self.assertEqual(iface._hand_joint, "L10")
        self.assertEqual(iface._modbus, "COM3")
        self.assertFalse(iface._debug)
        self.assertFalse(iface._is_opened)
        self.assertIsNone(iface._hand)

    def test_init_custom_params(self):
        iface = LinkerbotInterface(hand_type="right", hand_joint="L25", modbus="/dev/ttyUSB0", debug=True)
        self.assertEqual(iface._hand_type, "right")
        self.assertEqual(iface._hand_joint, "L25")
        self.assertEqual(iface._modbus, "/dev/ttyUSB0")
        self.assertTrue(iface._debug)

    def test_is_opened_initially_false(self):
        iface = LinkerbotInterface()
        self.assertFalse(iface.is_opened())

    # =========================================================================
    # OPEN / CLOSE LIFECYCLE
    # =========================================================================

    def test_open_debug_mode_does_not_instantiate_api(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertTrue(iface.is_opened())
        self.assertIsNone(iface._hand)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_open_calls_linker_hand_api(self, mock_api_class):
        iface = LinkerbotInterface(hand_type="left", hand_joint="L10", modbus="COM3", debug=False)
        iface.open()
        mock_api_class.assert_called_once_with(hand_type="left", hand_joint="L10", modbus="COM3")
        self.assertIs(iface._hand, mock_api_class.return_value)
        self.assertTrue(iface.is_opened())

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_open_when_already_opened_is_idempotent(self, mock_api_class):
        iface = LinkerbotInterface(debug=False)
        iface.open()
        iface.open()
        mock_api_class.assert_called_once()

    def test_close_when_not_opened(self):
        iface = LinkerbotInterface()
        iface.close()
        self.assertFalse(iface.is_opened())

    def test_close_after_open_debug_mode(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        iface.close()
        self.assertFalse(iface.is_opened())

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_close_after_open_non_debug(self, mock_api_class):
        iface = LinkerbotInterface(debug=False)
        iface.open()
        iface.close()
        self.assertFalse(iface.is_opened())
        self.assertIsNone(iface._hand)

    # =========================================================================
    # MOVE
    # =========================================================================

    def test_move_when_not_opened_returns_false(self):
        iface = LinkerbotInterface()
        self.assertFalse(iface.move([0] * 10))

    def test_move_debug_mode_returns_true(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertTrue(iface.move([0] * 10))

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_move_calls_finger_move(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.move([10] * 10)
        mock_hand.finger_move.assert_called_once_with([10] * 10)
        self.assertTrue(result)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_move_returns_false_on_exception(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.finger_move.side_effect = Exception("CAN error")
        iface = LinkerbotInterface(debug=False)
        iface.open()
        self.assertFalse(iface.move([0] * 10))

    # =========================================================================
    # SET_SPEED
    # =========================================================================

    def test_set_speed_when_not_opened_returns_false(self):
        iface = LinkerbotInterface()
        self.assertFalse(iface.set_speed([100] * 5))

    def test_set_speed_debug_mode_returns_true(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertTrue(iface.set_speed([100] * 5))

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_set_speed_calls_api(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.set_speed([100] * 5)
        mock_hand.set_speed.assert_called_once_with([100] * 5)
        self.assertTrue(result)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_set_speed_returns_false_on_exception(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.set_speed.side_effect = Exception("error")
        iface = LinkerbotInterface(debug=False)
        iface.open()
        self.assertFalse(iface.set_speed([100] * 5))

    # =========================================================================
    # SET_TORQUE
    # =========================================================================

    def test_set_torque_when_not_opened_returns_false(self):
        iface = LinkerbotInterface()
        self.assertFalse(iface.set_torque([180] * 5))

    def test_set_torque_debug_mode_returns_true(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertTrue(iface.set_torque([180] * 5))

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_set_torque_calls_api(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.set_torque([180] * 5)
        mock_hand.set_torque.assert_called_once_with([180] * 5)
        self.assertTrue(result)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_set_torque_returns_false_on_exception(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.set_torque.side_effect = Exception("error")
        iface = LinkerbotInterface(debug=False)
        iface.open()
        self.assertFalse(iface.set_torque([180] * 5))

    # =========================================================================
    # GET_STATE
    # =========================================================================

    def test_get_state_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_state())

    def test_get_state_debug_mode_returns_mock_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_state(), [0] * 10)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_state_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_state.return_value = [5] * 10
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_state()
        mock_hand.get_state.assert_called_once()
        self.assertEqual(result, [5] * 10)

    # =========================================================================
    # GET_SPEED
    # =========================================================================

    def test_get_speed_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_speed())

    def test_get_speed_debug_mode_returns_mock_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_speed(), [100] * 10)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_speed_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_speed.return_value = [50] * 10
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_speed()
        mock_hand.get_speed.assert_called_once()
        self.assertEqual(result, [50] * 10)

    # =========================================================================
    # GET_TORQUE
    # =========================================================================

    def test_get_torque_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_torque())

    def test_get_torque_debug_mode_returns_mock_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_torque(), [180] * 10)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_torque_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_torque.return_value = [200] * 10
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_torque()
        mock_hand.get_torque.assert_called_once()
        self.assertEqual(result, [200] * 10)

    # =========================================================================
    # GET_TOUCH
    # =========================================================================

    def test_get_touch_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_touch())

    def test_get_touch_debug_mode_returns_mock_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_touch(), [0] * 6)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_touch_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_touch.return_value = [1, 0, 1, 0, 1, 0]
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_touch()
        mock_hand.get_touch.assert_called_once()
        self.assertEqual(result, [1, 0, 1, 0, 1, 0])

    # =========================================================================
    # GET_FORCE
    # =========================================================================

    def test_get_force_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_force())

    def test_get_force_debug_mode_returns_nested_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_force(), [[0] * 5, [0] * 5, [0] * 5, [0] * 5])

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_force_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        expected = [[1] * 5, [2] * 5, [3] * 5, [4] * 5]
        mock_hand.get_force.return_value = expected
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_force()
        mock_hand.get_force.assert_called_once()
        self.assertEqual(result, expected)

    # =========================================================================
    # GET_VERSION
    # =========================================================================

    def test_get_version_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_version())

    def test_get_version_debug_mode_returns_string(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_version(), "debug-version")

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_version_calls_get_embedded_version(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_embedded_version.return_value = "V1.2.3"
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_version()
        mock_hand.get_embedded_version.assert_called_once()
        self.assertEqual(result, "V1.2.3")

    # =========================================================================
    # GET_SERIAL_NUMBER
    # =========================================================================

    def test_get_serial_number_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_serial_number())

    def test_get_serial_number_debug_mode_returns_string(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_serial_number(), "debug-serial")

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_serial_number_calls_api(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_serial_number.return_value = "SN-12345"
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_serial_number()
        mock_hand.get_serial_number.assert_called_once()
        self.assertEqual(result, "SN-12345")

    # =========================================================================
    # GET_TEMPERATURE
    # =========================================================================

    def test_get_temperature_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_temperature())

    def test_get_temperature_debug_mode_returns_mock_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_temperature(), [25] * 10)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_temperature_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_temperature.return_value = [30] * 10
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_temperature()
        mock_hand.get_temperature.assert_called_once()
        self.assertEqual(result, [30] * 10)

    # =========================================================================
    # GET_FAULT
    # =========================================================================

    def test_get_fault_when_not_opened_returns_none(self):
        iface = LinkerbotInterface()
        self.assertIsNone(iface.get_fault())

    def test_get_fault_debug_mode_returns_mock_list(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.get_fault(), [0] * 10)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_get_fault_calls_api_and_returns_value(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.get_fault.return_value = [0] * 10
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.get_fault()
        mock_hand.get_fault.assert_called_once()
        self.assertEqual(result, [0] * 10)

    # =========================================================================
    # CLEAR_FAULTS
    # =========================================================================

    def test_clear_faults_when_not_opened_returns_empty_list(self):
        iface = LinkerbotInterface()
        self.assertEqual(iface.clear_faults(), [])

    def test_clear_faults_debug_mode_returns_zeros(self):
        iface = LinkerbotInterface(debug=True)
        iface.open()
        self.assertEqual(iface.clear_faults(), [0] * 5)

    @patch("devices.linkerbot.interface.LinkerHandApi")
    def test_clear_faults_calls_api(self, mock_api_class):
        mock_hand = mock_api_class.return_value
        mock_hand.clear_faults.return_value = [0] * 5
        iface = LinkerbotInterface(debug=False)
        iface.open()
        result = iface.clear_faults()
        mock_hand.clear_faults.assert_called_once()
        self.assertEqual(result, [0] * 5)


if __name__ == "__main__":
    unittest.main()
