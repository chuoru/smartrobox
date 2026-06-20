#!/usr/bin/env python3
##
# @file test_grasp_action.py
#
# @brief Integration tests for GraspAction against real LinkerBot O6 hands.
#        Tests are skipped automatically when a hand is unavailable.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import os
import tempfile
import time
import unittest

# External library
import yaml

# Internal library
from actions.base import ActionState
from actions.grasp import GraspAction
from app.config import Config
from app.controller import Controller


_OPEN_POSE = [255] * 6
_TORQUE_LIMIT = 180
_MOVEMENT_DELAY = 2.0
_GRASP_TIMEOUT = 10.0


class _GraspActionTestBase:
    """! Shared test logic for GraspAction against a single LinkerBot O6 hand.

    Subclasses must define _DEVICE_NAME, _HAND_TYPE, and _MODBUS as class
    attributes.  Each subclass gets its own Controller and temp config so the
    two hand suites are fully independent.
    """

    _DEVICE_NAME: str = ""
    _HAND_TYPE: str = ""
    _MODBUS: str = ""

    _ctrl: Controller = None
    _cfg_path: str = None

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls) -> None:
        """! Build a temporary config and create the Controller."""
        device_cfg = {
            "devices": {
                cls._DEVICE_NAME: {
                    "type": "linkerbot",
                    "params": {
                        "hand_type": cls._HAND_TYPE,
                        "hand_joint": "O6",
                        "modbus": cls._MODBUS,
                    },
                }
            }
        }
        fd, cls._cfg_path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(device_cfg, f)
        cls._ctrl = Controller(Config(cls._cfg_path))

    @classmethod
    def tearDownClass(cls) -> None:
        """! Close all devices and remove the temporary config file."""
        cls._ctrl.close_all()
        os.unlink(cls._cfg_path)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _is_open(self) -> bool:
        """! Return True when the hand device is currently open.

        @return<bool>: True if the device reports is_opened.
        """
        return self._ctrl.status(self._DEVICE_NAME)["is_opened"]

    def _reset_hand(self) -> None:
        """! Send an open-pose command and wait for physical motion to settle."""
        self._ctrl.execute(self._DEVICE_NAME, "move", _OPEN_POSE)
        time.sleep(_MOVEMENT_DELAY)

    # =========================================================================
    # WORKFLOW TESTS
    # =========================================================================

    def test_01_open_hand(self) -> None:
        """! Open the hand via Controller; skip the suite if hardware is absent."""
        result = self._ctrl.open(self._DEVICE_NAME)
        if not result:
            self.skipTest(f"{self._DEVICE_NAME} (linkerbot on {self._MODBUS}) not available")
        self.assertTrue(self._is_open())

    def test_02_grasp_level_1_runs_to_completion(self) -> None:
        """! A level-1 grasp (index finger only) completes and reports DONE."""
        if not self._is_open():
            self.skipTest(f"{self._DEVICE_NAME} not open — skipping dependent step")
        action = GraspAction(self._ctrl, self._DEVICE_NAME, grasp_level=1, torque_limit=_TORQUE_LIMIT)
        action.start()
        finished = action.wait(timeout=_GRASP_TIMEOUT)
        self._reset_hand()
        self.assertTrue(finished, f"action did not complete (state={action.state()}, error={action.error()})")
        self.assertEqual(action.state(), ActionState.DONE)
        self.assertTrue(action.result())

    def test_03_grasp_level_4_runs_to_completion(self) -> None:
        """! A full-hand grasp (level 4, all fingers) completes and reports DONE."""
        if not self._is_open():
            self.skipTest(f"{self._DEVICE_NAME} not open — skipping dependent step")
        action = GraspAction(self._ctrl, self._DEVICE_NAME, grasp_level=4, torque_limit=_TORQUE_LIMIT)
        action.start()
        finished = action.wait(timeout=_GRASP_TIMEOUT)
        self._reset_hand()
        self.assertTrue(finished, f"full-hand grasp did not complete (state={action.state()}, error={action.error()})")
        self.assertEqual(action.state(), ActionState.DONE)

    def test_04_cancel_stops_action_in_terminal_state(self) -> None:
        """! cancel() issued immediately after start() reaches a terminal state without error.

        Over a live RS485 bus the cancel may land between the two checkpoints
        (producing CANCELLED) or after both phases complete (producing DONE).
        Both are valid; FAILED or a hang are not.
        """
        if not self._is_open():
            self.skipTest(f"{self._DEVICE_NAME} not open — skipping dependent step")
        action = GraspAction(self._ctrl, self._DEVICE_NAME, grasp_level=1, torque_limit=_TORQUE_LIMIT)
        action.start()
        action.cancel()
        action.wait(timeout=_GRASP_TIMEOUT)
        self._reset_hand()
        self.assertIn(
            action.state(),
            (ActionState.CANCELLED, ActionState.DONE),
            f"unexpected terminal state: {action.state()}",
        )
        self.assertIsNone(action.error())

    def test_05_parameters_reflect_configuration(self) -> None:
        """! parameters() returns the exact values passed to the constructor."""
        action = GraspAction(self._ctrl, self._DEVICE_NAME, grasp_level=2, torque_limit=_TORQUE_LIMIT)
        params = action.parameters()
        self.assertEqual(params["device_name"], self._DEVICE_NAME)
        self.assertEqual(params["grasp_level"], 2)
        self.assertEqual(params["torque_limit"], _TORQUE_LIMIT)
        self.assertEqual(set(params.keys()), {"device_name", "grasp_level", "torque_limit"})

    def test_06_close_hand(self) -> None:
        """! close() via Controller succeeds and reports the device as closed."""
        if not self._is_open():
            self.skipTest(f"{self._DEVICE_NAME} was not open")
        result = self._ctrl.close(self._DEVICE_NAME)
        self.assertTrue(result)
        self.assertFalse(self._is_open())


class TestGraspActionLeftHandIntegration(_GraspActionTestBase, unittest.TestCase):
    """! GraspAction integration tests for the left hand (/dev/ttyUSB1)."""

    _DEVICE_NAME = "left_hand"
    _HAND_TYPE = "left"
    _MODBUS = "/dev/ttyUSB1"


class TestGraspActionRightHandIntegration(_GraspActionTestBase, unittest.TestCase):
    """! GraspAction integration tests for the right hand (/dev/ttyUSB0)."""

    _DEVICE_NAME = "right_hand"
    _HAND_TYPE = "right"
    _MODBUS = "/dev/ttyUSB0"


if __name__ == "__main__":
    unittest.main()
