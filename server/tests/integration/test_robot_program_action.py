#!/usr/bin/env python3
##
# @file test_robot_program_action.py
#
# @brief Integration test for RobotProgramAction running sample.txt
#        against a real Fairino arm.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import tempfile
import unittest

# External library
import yaml

# Internal library
from actions.base import ActionState
from actions.robot_program import RobotProgramAction, deserialize
from app.config import Config
from app.controller import Controller


_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_PROGRAM_NAME = "sample.txt"
_DEVICE_NAME = "left_arm"
_ARM_IP = "192.168.58.2"


class TestRobotProgramActionIntegration(unittest.TestCase):
    """! Integration test: run sample.txt on a real Fairino arm via RobotProgramAction."""

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        device_cfg = {
            "devices": {
                _DEVICE_NAME: {
                    "type": "fairino",
                    "params": {"ip": _ARM_IP},
                }
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
        result = self._ctrl.open(_DEVICE_NAME)
        if not result:
            self.skipTest(f"{_DEVICE_NAME} (fairino at {_ARM_IP}) not available")
        self.assertTrue(self._ctrl.status(_DEVICE_NAME)["is_opened"])

    def test_02_sample_program_runs_to_completion(self):
        if not self._ctrl.status(_DEVICE_NAME)["is_opened"]:
            self.skipTest(f"{_DEVICE_NAME} not open — skipping dependent step")
        action = RobotProgramAction(
            self._ctrl, _PROGRAM_NAME, _DEVICE_NAME, _DATA_FOLDER
        )
        action.start()
        finished = action.wait(timeout=60.0)
        self.assertTrue(finished, f"action did not complete (state={action.state()}, error={action.error()})")
        self.assertEqual(action.state(), ActionState.DONE)

    def test_03_result_matches_step_count(self):
        if not self._ctrl.status(_DEVICE_NAME)["is_opened"]:
            self.skipTest(f"{_DEVICE_NAME} not open — skipping dependent step")
        program_path = os.path.join(_DATA_FOLDER, "robot_program", _PROGRAM_NAME)
        with open(program_path, "r") as f:
            expected_steps = len(deserialize(f.read()))
        action = RobotProgramAction(
            self._ctrl, _PROGRAM_NAME, _DEVICE_NAME, _DATA_FOLDER
        )
        action.start()
        action.wait(timeout=60.0)
        self.assertEqual(action.result(), expected_steps)

    def test_04_parameters_reflect_configuration(self):
        action = RobotProgramAction(
            self._ctrl, _PROGRAM_NAME, _DEVICE_NAME, _DATA_FOLDER
        )
        params = action.parameters()
        self.assertEqual(params["program_name"], _PROGRAM_NAME)
        self.assertEqual(params["device_name"], _DEVICE_NAME)

    def test_05_close_left_arm(self):
        if not self._ctrl.status(_DEVICE_NAME)["is_opened"]:
            self.skipTest(f"{_DEVICE_NAME} was not open")
        result = self._ctrl.close(_DEVICE_NAME)
        self.assertTrue(result)
        self.assertFalse(self._ctrl.status(_DEVICE_NAME)["is_opened"])


if __name__ == "__main__":
    unittest.main()
