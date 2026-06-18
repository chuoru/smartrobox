#!/usr/bin/env python3
##
# @file test_fairino_interface.py
#
# @brief Integration tests for FairinoInterface against real robots at
#        192.168.57.2 and 192.168.57.3. Tests are skipped automatically
#        when the target device is unavailable.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/19.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# ========================
# Standard library
# ========================
import unittest

# ========================
# Internal library
# ========================
from devices.fairino.interface import FairinoInterface


class TestFairinoRobot1Integration(unittest.TestCase):
    """! Integration tests for FairinoInterface with robot 1 (192.168.57.2)."""

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        cls._iface = FairinoInterface(ip="192.168.57.2")
        try:
            cls._iface.open()
        except Exception as exc:
            raise unittest.SkipTest(f"Robot 1 not available at 192.168.57.2: {exc}")
        if not cls._iface.is_opened():
            raise unittest.SkipTest("Robot 1 failed to open at 192.168.57.2")

    @classmethod
    def tearDownClass(cls):
        cls._iface.close()

    # =========================================================================
    # TESTS
    # =========================================================================

    def test_is_opened_after_connection(self):
        self.assertTrue(self._iface.is_opened())

    def test_tpos_returns_valid_pose(self):
        error, pose = self._iface.tpos()
        self.assertEqual(error, 0)
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 6)
        for value in pose:
            self.assertIsInstance(value, float)

    def test_movej_moves_to_current_joint_pose(self):
        error, pose = self._iface.tpos()
        self.assertEqual(error, 0, "tpos() failed before movej test")
        j1, j2, j3, j4, j5, j6 = self._iface.get_inverse_kinematics(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        )
        result = self._iface.movej(j1, j2, j3, j4, j5, j6)
        self.assertTrue(result)

    def test_movel_moves_to_current_pose(self):
        error, pose = self._iface.tpos()
        self.assertEqual(error, 0, "tpos() failed before movel test")
        result = self._iface.movel(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        )
        self.assertTrue(result)


class TestFairinoRobot2Integration(unittest.TestCase):
    """! Integration tests for FairinoInterface with robot 2 (192.168.57.3)."""

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        cls._iface = FairinoInterface(ip="192.168.57.3")
        try:
            cls._iface.open()
        except Exception as exc:
            raise unittest.SkipTest(f"Robot 2 not available at 192.168.57.3: {exc}")
        if not cls._iface.is_opened():
            raise unittest.SkipTest("Robot 2 failed to open at 192.168.57.3")

    @classmethod
    def tearDownClass(cls):
        cls._iface.close()

    # =========================================================================
    # TESTS
    # =========================================================================

    def test_is_opened_after_connection(self):
        self.assertTrue(self._iface.is_opened())

    def test_tpos_returns_valid_pose(self):
        error, pose = self._iface.tpos()
        self.assertEqual(error, 0)
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 6)
        for value in pose:
            self.assertIsInstance(value, float)

    def test_movej_moves_to_current_joint_pose(self):
        error, pose = self._iface.tpos()
        self.assertEqual(error, 0, "tpos() failed before movej test")
        j1, j2, j3, j4, j5, j6 = self._iface.get_inverse_kinematics(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        )
        result = self._iface.movej(j1, j2, j3, j4, j5, j6)
        self.assertTrue(result)

    def test_movel_moves_to_current_pose(self):
        error, pose = self._iface.tpos()
        self.assertEqual(error, 0, "tpos() failed before movel test")
        result = self._iface.movel(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
