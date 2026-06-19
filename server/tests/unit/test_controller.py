#!/usr/bin/env python3
##
# @file test_controller.py
#
# @brief Unit tests for the Controller class.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import unittest
from unittest.mock import MagicMock, patch

# Internal library
from app.controller import Controller


_DEVICE_CONFIG = {
    "left_arm": {"type": "fairino", "params": {"ip": "192.168.58.2", "debug": False}},
    "left_hand": {
        "type": "linkerbot",
        "params": {"hand_type": "left", "hand_joint": "L10", "modbus": "COM3", "debug": False},
    },
    "camera": {"type": "orbbec", "params": {"device_index": 0}},
}


class TestControllerRegistry(unittest.TestCase):
    """! Tests for _build_registry error-handling paths."""

    def _make_controller(self, devices_cfg: dict) -> Controller:
        mock_config = MagicMock()
        mock_config.get.return_value = devices_cfg
        with patch("app.controller.Logger"):
            return Controller(mock_config)

    # =========================================================================
    # REGISTRY BUILD
    # =========================================================================

    @patch.dict("app.controller._DEVICE_FACTORIES", {"fairino": MagicMock()})
    def test_registry_skips_entry_with_missing_type(self):
        ctrl = self._make_controller({"arm": {"params": {}}})
        self.assertEqual(ctrl.list_devices(), [])

    @patch.dict("app.controller._DEVICE_FACTORIES", {})
    def test_registry_skips_entry_with_unknown_type(self):
        ctrl = self._make_controller({"arm": {"type": "unknown_xyz", "params": {}}})
        self.assertEqual(ctrl.list_devices(), [])

    def test_registry_skips_entry_with_bad_params(self):
        bad_factory = MagicMock(side_effect=TypeError("unexpected keyword argument 'bad_param'"))
        with patch.dict("app.controller._DEVICE_FACTORIES", {"fairino": bad_factory}):
            ctrl = self._make_controller(
                {"arm": {"type": "fairino", "params": {"bad_param": 99}}}
            )
        self.assertEqual(ctrl.list_devices(), [])

    def test_registry_empty_devices_config(self):
        ctrl = self._make_controller({})
        self.assertEqual(ctrl.list_devices(), [])


class TestController(unittest.TestCase):
    """! Tests for Controller public methods with all device interfaces mocked."""

    # =========================================================================
    # SETUP / TEARDOWN
    # =========================================================================

    def setUp(self):
        self._fairino_instance = MagicMock()
        self._linkerbot_instance = MagicMock()
        self._orbbec_instance = MagicMock()

        self._factories_patcher = patch.dict(
            "app.controller._DEVICE_FACTORIES",
            {
                "fairino": MagicMock(return_value=self._fairino_instance),
                "linkerbot": MagicMock(return_value=self._linkerbot_instance),
                "orbbec": MagicMock(return_value=self._orbbec_instance),
            },
        )
        self._factories_patcher.start()

        self._logger_patcher = patch("app.controller.Logger")
        self._logger_patcher.start()

        mock_config = MagicMock()
        mock_config.get.return_value = _DEVICE_CONFIG
        self._ctrl = Controller(mock_config)

    def tearDown(self):
        self._factories_patcher.stop()
        self._logger_patcher.stop()

    # =========================================================================
    # LIST DEVICES
    # =========================================================================

    def test_list_devices_returns_all_registered_names(self):
        self.assertEqual(set(self._ctrl.list_devices()), {"left_arm", "left_hand", "camera"})

    # =========================================================================
    # OPEN
    # =========================================================================

    def test_open_calls_open_on_fairino(self):
        self._ctrl.open("left_arm")
        self._fairino_instance.open.assert_called_once()

    def test_open_calls_open_on_linkerbot(self):
        self._ctrl.open("left_hand")
        self._linkerbot_instance.open.assert_called_once()

    def test_open_calls_start_on_orbbec(self):
        self._ctrl.open("camera")
        self._orbbec_instance.start.assert_called_once()
        self._orbbec_instance.open.assert_not_called()

    def test_open_returns_true_on_success(self):
        self.assertTrue(self._ctrl.open("left_arm"))

    def test_open_returns_false_for_unknown_device(self):
        self.assertFalse(self._ctrl.open("does_not_exist"))

    def test_open_returns_false_when_device_raises(self):
        self._fairino_instance.open.side_effect = RuntimeError("connection refused")
        self.assertFalse(self._ctrl.open("left_arm"))

    # =========================================================================
    # CLOSE
    # =========================================================================

    def test_close_calls_close_on_fairino(self):
        self._ctrl.close("left_arm")
        self._fairino_instance.close.assert_called_once()

    def test_close_calls_close_on_linkerbot(self):
        self._ctrl.close("left_hand")
        self._linkerbot_instance.close.assert_called_once()

    def test_close_calls_stop_on_orbbec(self):
        self._ctrl.close("camera")
        self._orbbec_instance.stop.assert_called_once()
        self._orbbec_instance.close.assert_not_called()

    def test_close_returns_true_on_success(self):
        self.assertTrue(self._ctrl.close("left_arm"))

    def test_close_returns_false_for_unknown_device(self):
        self.assertFalse(self._ctrl.close("does_not_exist"))

    def test_close_returns_false_when_device_raises(self):
        self._fairino_instance.close.side_effect = RuntimeError("disconnect error")
        self.assertFalse(self._ctrl.close("left_arm"))

    # =========================================================================
    # OPEN ALL / CLOSE ALL
    # =========================================================================

    def test_open_all_returns_success_map(self):
        result = self._ctrl.open_all()
        self.assertEqual(set(result.keys()), {"left_arm", "left_hand", "camera"})
        self.assertTrue(all(result.values()))

    def test_close_all_returns_success_map(self):
        result = self._ctrl.close_all()
        self.assertEqual(set(result.keys()), {"left_arm", "left_hand", "camera"})
        self.assertTrue(all(result.values()))

    def test_close_all_continues_after_partial_failure(self):
        self._fairino_instance.close.side_effect = RuntimeError("error")
        result = self._ctrl.close_all()
        self.assertFalse(result["left_arm"])
        self.assertTrue(result["left_hand"])
        self.assertTrue(result["camera"])

    # =========================================================================
    # STATUS
    # =========================================================================

    def test_status_returns_none_for_unknown_device(self):
        self.assertIsNone(self._ctrl.status("does_not_exist"))

    def test_status_calls_is_opened_for_fairino(self):
        self._fairino_instance.is_opened.return_value = True
        result = self._ctrl.status("left_arm")
        self._fairino_instance.is_opened.assert_called_once()
        self.assertTrue(result["is_opened"])

    def test_status_calls_is_opened_for_linkerbot(self):
        self._linkerbot_instance.is_opened.return_value = False
        result = self._ctrl.status("left_hand")
        self._linkerbot_instance.is_opened.assert_called_once()
        self.assertFalse(result["is_opened"])

    def test_status_calls_is_alive_for_orbbec(self):
        self._orbbec_instance.is_alive.return_value = True
        result = self._ctrl.status("camera")
        self._orbbec_instance.is_alive.assert_called_once()
        self._orbbec_instance.is_opened.assert_not_called()
        self.assertTrue(result["is_opened"])

    def test_status_dict_contains_name_and_type(self):
        result = self._ctrl.status("left_arm")
        self.assertEqual(result["name"], "left_arm")
        self.assertEqual(result["type"], "fairino")

    def test_status_returns_false_when_probe_raises(self):
        self._fairino_instance.is_opened.side_effect = RuntimeError("probe error")
        result = self._ctrl.status("left_arm")
        self.assertFalse(result["is_opened"])

    # =========================================================================
    # EXECUTE
    # =========================================================================

    def test_execute_dispatches_method_and_returns_result(self):
        self._fairino_instance.tpos.return_value = (0, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        result = self._ctrl.execute("left_arm", "tpos")
        self._fairino_instance.tpos.assert_called_once()
        self.assertEqual(result, (0, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))

    def test_execute_passes_positional_args(self):
        pose = [0] * 10
        self._linkerbot_instance.move.return_value = True
        result = self._ctrl.execute("left_hand", "move", pose)
        self._linkerbot_instance.move.assert_called_once_with(pose)
        self.assertTrue(result)

    def test_execute_passes_keyword_args(self):
        self._fairino_instance.movej.return_value = True
        self._ctrl.execute("left_arm", "movej", j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0)
        self._fairino_instance.movej.assert_called_once_with(
            j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0
        )

    def test_execute_raises_key_error_for_unknown_device(self):
        with self.assertRaises(KeyError):
            self._ctrl.execute("does_not_exist", "move")

    def test_execute_raises_attribute_error_for_private_method(self):
        with self.assertRaises(AttributeError):
            self._ctrl.execute("left_arm", "_connect")

    def test_execute_raises_attribute_error_for_missing_method(self):
        self._fairino_instance.nonexistent_method_xyz = None
        with self.assertRaises(AttributeError):
            self._ctrl.execute("left_arm", "nonexistent_method_xyz")

    def test_execute_reraises_device_exception(self):
        self._fairino_instance.tpos.side_effect = RuntimeError("hardware fault")
        with self.assertRaises(RuntimeError):
            self._ctrl.execute("left_arm", "tpos")


if __name__ == "__main__":
    unittest.main()
