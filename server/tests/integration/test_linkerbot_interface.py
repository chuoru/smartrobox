#!/usr/bin/env python3
##
# @file test_linkerbot_interface.py
#
# @brief Integration tests for LinkerbotInterface against real left/right hands.
#        Tests are skipped automatically when the target device is unavailable.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/18.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# ========================
# Standard library
# ========================
import time
import unittest

# ========================
# Internal library
# ========================
from devices.linkerbot.interface import LinkerbotInterface


_OPEN_POSE = [0] * 6    # fingers fully extended
_CLOSE_POSE = [100] * 6  # fingers partially curled
_MOVEMENT_DELAY = 1.5    # seconds to allow physical movement to complete


class TestLinkerbotLeftHandIntegration(unittest.TestCase):
    """! Integration tests for LinkerbotInterface with the left hand (/dev/ttyUSB0)."""

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        cls._iface = LinkerbotInterface(
            hand_type="left",
            hand_joint="L6",
            modbus="/dev/ttyUSB0",
        )
        try:
            cls._iface.open()
        except Exception as exc:
            raise unittest.SkipTest(f"Left hand not available on /dev/ttyUSB0: {exc}")
        if not cls._iface.is_opened():
            raise unittest.SkipTest("Left hand failed to open on /dev/ttyUSB0")

    @classmethod
    def tearDownClass(cls):
        cls._iface.close()

    # =========================================================================
    # TESTS
    # =========================================================================

    def test_is_opened_after_connection(self):
        self.assertTrue(self._iface.is_opened())

    def test_open_and_close_hand_pose(self):
        # Open hand — fingers extended
        result = self._iface.move(_OPEN_POSE)
        self.assertTrue(result, "move() to open pose failed")
        time.sleep(_MOVEMENT_DELAY)

        # Close hand — fingers curled
        result = self._iface.move(_CLOSE_POSE)
        self.assertTrue(result, "move() to close pose failed")
        time.sleep(_MOVEMENT_DELAY)


class TestLinkerbotRightHandIntegration(unittest.TestCase):
    """! Integration tests for LinkerbotInterface with the right hand (/dev/ttyUSB1)."""

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        cls._iface = LinkerbotInterface(
            hand_type="right",
            hand_joint="L6",
            modbus="/dev/ttyUSB1",
        )
        try:
            cls._iface.open()
        except Exception as exc:
            raise unittest.SkipTest(f"Right hand not available on /dev/ttyUSB1: {exc}")
        if not cls._iface.is_opened():
            raise unittest.SkipTest("Right hand failed to open on /dev/ttyUSB1")

    @classmethod
    def tearDownClass(cls):
        cls._iface.close()

    # =========================================================================
    # TESTS
    # =========================================================================

    def test_is_opened_after_connection(self):
        self.assertTrue(self._iface.is_opened())

    def test_open_and_close_hand_pose(self):
        # Open hand — fingers extended
        result = self._iface.move(_OPEN_POSE)
        self.assertTrue(result, "move() to open pose failed")
        time.sleep(_MOVEMENT_DELAY)

        # Close hand — fingers curled
        result = self._iface.move(_CLOSE_POSE)
        self.assertTrue(result, "move() to close pose failed")
        time.sleep(_MOVEMENT_DELAY)


if __name__ == "__main__":
    unittest.main()
