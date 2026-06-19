#!/usr/bin/env python3
##
# @file test_orbbec_interface.py
#
# @brief Integration tests for OrbbecInterface against a real Orbbec camera.
#        Tests are skipped automatically when no camera is connected.
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
from devices.orbbec_interface import OrbbecInterface


class TestOrbbecInterfaceIntegration(unittest.TestCase):
    """! Integration tests for OrbbecInterface using a physically connected camera.

    Opens the camera once for the entire suite in setUpClass and closes it in
    tearDownClass. All tests are skipped when no Orbbec device is detected.
    """

    _FRAME_TIMEOUT = 5.0  # seconds to wait for the first live frame

    # =========================================================================
    # SUITE SETUP / TEARDOWN
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        cls.iface = OrbbecInterface(device_index=0)
        try:
            cls.iface.start()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"No Orbbec camera available: {exc}")

    @classmethod
    def tearDownClass(cls):
        cls.iface.stop()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _wait_for_frame(self, timeout: float = _FRAME_TIMEOUT) -> bool:
        """! Poll until the first color frame is delivered by the SDK callback.

        @param timeout<float>: Maximum seconds to wait.
        @return<bool>: True if a frame arrived within the deadline.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.iface.get_color_frame() is not None:
                return True
            time.sleep(0.1)
        return False

    # =========================================================================
    # TESTS
    # =========================================================================

    def test_camera_is_alive_after_start(self):
        # Assert
        self.assertTrue(self.iface.is_alive())
        self.assertIsNotNone(self.iface._fx)
        self.assertIsNotNone(self.iface._fy)
        self.assertIsNotNone(self.iface._cx)
        self.assertIsNotNone(self.iface._cy)
        self.assertGreater(self.iface._fx, 0.0)
        self.assertGreater(self.iface._fy, 0.0)

    def test_capture_returns_color_and_depth_frames(self):
        # Arrange — wait for the SDK callback to deliver the first frame
        got_frame = self._wait_for_frame()
        self.assertTrue(got_frame, "Timed out waiting for first frame from camera")

        # Act
        color = self.iface.get_color_frame()
        depth = self.iface.get_depth_frame()

        # Assert — color: BGR uint8 H×W×3
        self.assertIsNotNone(color)
        self.assertEqual(color.ndim, 3)
        self.assertEqual(color.shape[2], 3)
        self.assertEqual(str(color.dtype), "uint8")

        # Assert — depth: uint16 H×W
        self.assertIsNotNone(depth)
        self.assertEqual(depth.ndim, 2)
        self.assertEqual(str(depth.dtype), "uint16")

    def test_pixel_to_world_returns_plausible_3d_point(self):
        # Arrange — wait for the SDK callback to deliver the first frame
        got_frame = self._wait_for_frame()
        self.assertTrue(got_frame, "Timed out waiting for first frame from camera")

        depth = self.iface.get_depth_frame()
        self.assertIsNotNone(depth)
        h, w = depth.shape
        u, v = w // 2, h // 2

        # Act
        result = self.iface.pixel_to_world(u, v)

        # Assert — centre pixel should have a valid, positive depth
        self.assertIsNotNone(result, "pixel_to_world returned None at image centre")
        _x, _y, z = result
        self.assertGreater(z, 0.0)


if __name__ == "__main__":
    unittest.main()
