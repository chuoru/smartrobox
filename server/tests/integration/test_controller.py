#!/usr/bin/env python3
##
# @file test_controller.py
#
# @brief Integration test for Controller running the full device workflow:
#        open left_arm → open left_hand → control index finger →
#        open camera → take picture → close camera →
#        close left_hand → retrieve left_arm pose → close left_arm.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import time
import tempfile
import unittest

# External library
import yaml

# Internal library
from app.config import Config
from app.controller import Controller


# L10 hand has 10 joints; joint index 1 is the index-finger curl joint.
_INDEX_FINGER_CURL = [0, 100, 0, 0, 0, 0, 0, 0, 0, 0]
_INDEX_FINGER_EXTEND = [0] * 10
_MOVEMENT_DELAY = 1.5  # seconds — allow physical movement to complete


class TestControllerIntegration(unittest.TestCase):
    """! Full device workflow through Controller against real hardware."""

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        device_cfg = {
            "devices": {
                "left_arm": {
                    "type": "fairino",
                    "params": {"ip": "192.168.57.2"},
                },
                "left_hand": {
                    "type": "linkerbot",
                    "params": {
                        "hand_type": "left",
                        "hand_joint": "L10",
                        "modbus": "/dev/ttyUSB0",
                    },
                },
                "camera": {
                    "type": "orbbec",
                    "params": {"device_index": 0},
                },
            }
        }
        fd, cls._cfg_path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(device_cfg, f)
        cls._ctrl = Controller(Config(cls._cfg_path))

    @classmethod
    def tearDownClass(cls):
        cls._ctrl.close_all()
        os.unlink(cls._cfg_path)

    # =========================================================================
    # WORKFLOW TESTS
    # =========================================================================

    def test_01_open_left_arm(self):
        result = self._ctrl.open("left_arm")
        if not result:
            self.skipTest("left_arm (fairino at 192.168.57.2) not available")
        self.assertTrue(self._ctrl.status("left_arm")["is_opened"])

    def test_02_open_left_hand(self):
        if not self._ctrl.status("left_arm")["is_opened"]:
            self.skipTest("left_arm not open — skipping dependent step")
        result = self._ctrl.open("left_hand")
        if not result:
            self.skipTest("left_hand (linkerbot on /dev/ttyUSB0) not available")
        self.assertTrue(self._ctrl.status("left_hand")["is_opened"])

    def test_03_control_index_finger(self):
        if not self._ctrl.status("left_hand")["is_opened"]:
            self.skipTest("left_hand not open — skipping dependent step")
        result = self._ctrl.execute("left_hand", "move", _INDEX_FINGER_CURL)
        self.assertTrue(result, "index finger curl failed")
        time.sleep(_MOVEMENT_DELAY)

        result = self._ctrl.execute("left_hand", "move", _INDEX_FINGER_EXTEND)
        self.assertTrue(result, "index finger extend failed")
        time.sleep(_MOVEMENT_DELAY)

    def test_04_open_camera(self):
        result = self._ctrl.open("camera")
        if not result:
            self.skipTest("camera (orbbec device_index=0) not available")
        self.assertTrue(self._ctrl.status("camera")["is_opened"])

    def test_05_take_picture(self):
        if not self._ctrl.status("camera")["is_opened"]:
            self.skipTest("camera not open — skipping dependent step")
        frame = self._ctrl.execute("camera", "get_color_frame")
        self.assertIsNotNone(frame, "get_color_frame() returned None")
        self.assertEqual(len(frame.shape), 3, "color frame should be H×W×3")

    def test_06_close_camera(self):
        result = self._ctrl.close("camera")
        self.assertTrue(result)
        self.assertFalse(self._ctrl.status("camera")["is_opened"])

    def test_07_close_left_hand(self):
        if not self._ctrl.status("left_hand")["is_opened"]:
            self.skipTest("left_hand was not open")
        result = self._ctrl.close("left_hand")
        self.assertTrue(result)
        self.assertFalse(self._ctrl.status("left_hand")["is_opened"])

    def test_08_retrieve_left_arm_pose(self):
        if not self._ctrl.status("left_arm")["is_opened"]:
            self.skipTest("left_arm not open — skipping dependent step")
        error, pose = self._ctrl.execute("left_arm", "tpos")
        self.assertEqual(error, 0, f"tpos() returned error code {error}")
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 6)

    def test_09_close_left_arm(self):
        if not self._ctrl.status("left_arm")["is_opened"]:
            self.skipTest("left_arm was not open")
        result = self._ctrl.close("left_arm")
        self.assertTrue(result)
        self.assertFalse(self._ctrl.status("left_arm")["is_opened"])


if __name__ == "__main__":
    unittest.main()
