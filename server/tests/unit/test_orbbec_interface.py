#!/usr/bin/env python3
##
# @file test_orbbec_interface.py
#
# @brief Unit tests for the OrbbecInterface class.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/18.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import sys
import unittest
from unittest.mock import MagicMock, patch, call

# External library
import numpy as np

# ── mock pyorbbecsdk and cv2 BEFORE importing the module under test ───────────
_mock_pyorbbecsdk = MagicMock()
_mock_pyorbbecsdk.OBFormat.MJPG = "MJPG"
_mock_pyorbbecsdk.OBFormat.RGB = "RGB"
_mock_pyorbbecsdk.OBFormat.BGR = "BGR"
sys.modules["pyorbbecsdk"] = _mock_pyorbbecsdk
sys.modules["cv2"] = MagicMock()

# Internal library
from devices.orbbec_interface import OrbbecInterface  # noqa: E402


class TestOrbbecInterface(unittest.TestCase):
    """! Unit tests for OrbbecInterface."""

    def _make_started_interface(self) -> OrbbecInterface:
        """! Return an OrbbecInterface in the running state without calling start()."""
        iface = OrbbecInterface(device_index=0)
        iface.running = True
        iface._pipeline = MagicMock()
        iface._fail_count = 0
        iface._fx = 500.0
        iface._fy = 500.0
        iface._cx = 320.0
        iface._cy = 240.0
        iface._depth_scale = 0.001
        return iface

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def test_init_default_device_index(self):
        # Arrange / Act
        iface = OrbbecInterface()
        # Assert
        self.assertEqual(iface.device_index, 0)
        self.assertFalse(iface.running)
        self.assertEqual(iface._fail_count, 0)
        self.assertIsNone(iface._context)
        self.assertIsNone(iface._pipeline)
        self.assertIsNone(iface._thread)

    def test_init_custom_device_index(self):
        # Arrange / Act
        iface = OrbbecInterface(device_index=2)
        # Assert
        self.assertEqual(iface.device_index, 2)

    def test_init_intrinsics_and_frames_none(self):
        # Arrange / Act
        iface = OrbbecInterface()
        # Assert
        self.assertIsNone(iface._fx)
        self.assertIsNone(iface._fy)
        self.assertIsNone(iface._cx)
        self.assertIsNone(iface._cy)
        self.assertIsNone(iface._depth_scale)
        self.assertIsNone(iface._color_frame)
        self.assertIsNone(iface._depth_frame)

    # =========================================================================
    # START LIFECYCLE
    # =========================================================================

    @patch("devices.orbbec_interface.Context")
    def test_start_when_already_running_is_idempotent(self, mock_context):
        # Arrange
        iface = OrbbecInterface()
        iface.running = True
        # Act
        iface.start()
        # Assert
        mock_context.assert_not_called()
        self.assertTrue(iface.running)

    @patch("devices.orbbec_interface.Context")
    def test_start_raises_if_no_devices_found(self, mock_context):
        # Arrange
        mock_device_list = MagicMock()
        mock_device_list.get_count.return_value = 0
        mock_context.return_value.query_devices.return_value = mock_device_list
        iface = OrbbecInterface()
        # Act & Assert
        with self.assertRaises(RuntimeError) as ctx:
            iface.start()
        self.assertIn("No Orbbec devices found", str(ctx.exception))
        self.assertIsNone(iface._context)

    @patch("devices.orbbec_interface.Context")
    def test_start_raises_if_device_index_out_of_range(self, mock_context):
        # Arrange
        mock_device_list = MagicMock()
        mock_device_list.get_count.return_value = 1
        mock_context.return_value.query_devices.return_value = mock_device_list
        iface = OrbbecInterface(device_index=2)
        # Act & Assert
        with self.assertRaises(RuntimeError) as ctx:
            iface.start()
        self.assertIn("out of range", str(ctx.exception))
        self.assertIsNone(iface._context)

    @patch("devices.orbbec_interface.threading")
    @patch("devices.orbbec_interface.Config")
    @patch("devices.orbbec_interface.Pipeline")
    @patch("devices.orbbec_interface.Context")
    def test_start_success_sets_running_true(
        self, mock_context, mock_pipeline, mock_config, mock_threading
    ):
        # Arrange
        mock_device_list = MagicMock()
        mock_device_list.get_count.return_value = 1
        mock_context.return_value.query_devices.return_value = mock_device_list
        mock_intr = MagicMock()
        mock_intr.fx = 600.0
        mock_intr.fy = 600.0
        mock_intr.cx = 320.0
        mock_intr.cy = 240.0
        mock_pipeline.return_value.get_camera_param.return_value.rgb_intrinsic = mock_intr
        mock_threading.Thread.return_value = MagicMock()
        mock_threading.Lock = unittest.mock.MagicMock(return_value=MagicMock())
        iface = OrbbecInterface()
        # Act
        iface.start()
        # Assert
        self.assertTrue(iface.running)
        self.assertEqual(iface._fail_count, 0)

    @patch("devices.orbbec_interface.threading")
    @patch("devices.orbbec_interface.Config")
    @patch("devices.orbbec_interface.Pipeline")
    @patch("devices.orbbec_interface.Context")
    def test_start_sets_intrinsics_from_camera_param(
        self, mock_context, mock_pipeline, mock_config, mock_threading
    ):
        # Arrange
        mock_device_list = MagicMock()
        mock_device_list.get_count.return_value = 1
        mock_context.return_value.query_devices.return_value = mock_device_list
        mock_intr = MagicMock()
        mock_intr.fx = 600.0
        mock_intr.fy = 601.0
        mock_intr.cx = 320.0
        mock_intr.cy = 240.0
        mock_pipeline.return_value.get_camera_param.return_value.rgb_intrinsic = mock_intr
        mock_threading.Thread.return_value = MagicMock()
        iface = OrbbecInterface()
        # Act
        iface.start()
        # Assert
        self.assertEqual(iface._fx, 600.0)
        self.assertEqual(iface._fy, 601.0)
        self.assertEqual(iface._cx, 320.0)
        self.assertEqual(iface._cy, 240.0)

    @patch("devices.orbbec_interface.threading")
    @patch("devices.orbbec_interface.Config")
    @patch("devices.orbbec_interface.Pipeline")
    @patch("devices.orbbec_interface.Context")
    def test_start_creates_pipeline_with_correct_device(
        self, mock_context, mock_pipeline, mock_config, mock_threading
    ):
        # Arrange
        mock_device_list = MagicMock()
        mock_device_list.get_count.return_value = 1
        mock_device = MagicMock()
        mock_device_list.get_device_by_index.return_value = mock_device
        mock_context.return_value.query_devices.return_value = mock_device_list
        mock_intr = MagicMock()
        mock_intr.fx = mock_intr.fy = mock_intr.cx = mock_intr.cy = 1.0
        mock_pipeline.return_value.get_camera_param.return_value.rgb_intrinsic = mock_intr
        mock_threading.Thread.return_value = MagicMock()
        iface = OrbbecInterface()
        # Act
        iface.start()
        # Assert
        mock_pipeline.assert_called_once_with(mock_device)
        mock_pipeline.return_value.start.assert_called_once()
        mock_pipeline.return_value.enable_frame_sync.assert_called_once()

    @patch("devices.orbbec_interface.threading")
    @patch("devices.orbbec_interface.Config")
    @patch("devices.orbbec_interface.Pipeline")
    @patch("devices.orbbec_interface.Context")
    def test_start_spawns_daemon_thread(
        self, mock_context, mock_pipeline, mock_config, mock_threading
    ):
        # Arrange
        mock_device_list = MagicMock()
        mock_device_list.get_count.return_value = 1
        mock_context.return_value.query_devices.return_value = mock_device_list
        mock_intr = MagicMock()
        mock_intr.fx = mock_intr.fy = mock_intr.cx = mock_intr.cy = 1.0
        mock_pipeline.return_value.get_camera_param.return_value.rgb_intrinsic = mock_intr
        mock_thread_instance = MagicMock()
        mock_threading.Thread.return_value = mock_thread_instance
        iface = OrbbecInterface()
        # Act
        iface.start()
        # Assert
        _, kwargs = mock_threading.Thread.call_args
        self.assertTrue(kwargs.get("daemon"))
        self.assertIn("Orbbec-0", kwargs.get("name", ""))
        mock_thread_instance.start.assert_called_once()

    # =========================================================================
    # STOP
    # =========================================================================

    def test_stop_sets_running_false(self):
        # Arrange
        iface = self._make_started_interface()
        # Act
        iface.stop()
        # Assert
        self.assertFalse(iface.running)

    def test_stop_clears_pipeline_and_context(self):
        # Arrange
        iface = self._make_started_interface()
        # Act
        iface.stop()
        # Assert
        self.assertIsNone(iface._pipeline)
        self.assertIsNone(iface._context)

    def test_stop_clears_frames_under_lock(self):
        # Arrange
        iface = self._make_started_interface()
        iface._color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        iface._depth_frame = np.zeros((480, 640), dtype=np.uint16)
        # Act
        iface.stop()
        # Assert
        self.assertIsNone(iface._color_frame)
        self.assertIsNone(iface._depth_frame)

    def test_stop_calls_pipeline_stop_and_swallows_exception(self):
        # Arrange
        iface = self._make_started_interface()
        iface._pipeline.stop.side_effect = Exception("camera error")
        # Act — must not raise
        iface.stop()
        # Assert
        self.assertIsNone(iface._pipeline)

    def test_stop_when_not_started_is_safe(self):
        # Arrange
        iface = OrbbecInterface()
        # Act — must not raise
        iface.stop()
        # Assert
        self.assertFalse(iface.running)
        self.assertIsNone(iface._pipeline)

    # =========================================================================
    # IS_ALIVE
    # =========================================================================

    def test_is_alive_false_when_not_running(self):
        # Arrange
        iface = OrbbecInterface()
        iface.running = False
        iface._pipeline = MagicMock()
        iface._fail_count = 0
        # Assert
        self.assertFalse(iface.is_alive())

    def test_is_alive_false_when_pipeline_is_none(self):
        # Arrange
        iface = OrbbecInterface()
        iface.running = True
        iface._pipeline = None
        iface._fail_count = 0
        # Assert
        self.assertFalse(iface.is_alive())

    def test_is_alive_false_when_fail_count_at_max(self):
        # Arrange
        iface = OrbbecInterface()
        iface.running = True
        iface._pipeline = MagicMock()
        iface._fail_count = OrbbecInterface._MAX_FAIL  # 10
        # Assert
        self.assertFalse(iface.is_alive())

    def test_is_alive_true_when_all_conditions_met(self):
        # Arrange
        iface = self._make_started_interface()
        # Assert — fail_count=0
        self.assertTrue(iface.is_alive())
        # Secondary: fail_count just below threshold also returns True
        iface._fail_count = OrbbecInterface._MAX_FAIL - 1
        self.assertTrue(iface.is_alive())

    # =========================================================================
    # GET_COLOR_FRAME
    # =========================================================================

    def test_get_color_frame_returns_none_when_not_alive(self):
        # Arrange
        iface = OrbbecInterface()
        # Assert
        self.assertIsNone(iface.get_color_frame())

    def test_get_color_frame_returns_none_when_frame_is_none(self):
        # Arrange
        iface = self._make_started_interface()
        iface._color_frame = None
        # Assert
        self.assertIsNone(iface.get_color_frame())

    def test_get_color_frame_returns_copy_of_frame(self):
        # Arrange
        iface = self._make_started_interface()
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        iface._color_frame = original
        # Act
        result = iface.get_color_frame()
        # Assert
        self.assertIsNotNone(result)
        self.assertTrue(np.array_equal(result, original))
        self.assertIsNot(result, original)

    # =========================================================================
    # GET_DEPTH_FRAME
    # =========================================================================

    def test_get_depth_frame_returns_none_when_not_alive(self):
        # Arrange
        iface = OrbbecInterface()
        # Assert
        self.assertIsNone(iface.get_depth_frame())

    def test_get_depth_frame_returns_none_when_frame_is_none(self):
        # Arrange
        iface = self._make_started_interface()
        iface._depth_frame = None
        # Assert
        self.assertIsNone(iface.get_depth_frame())

    def test_get_depth_frame_returns_copy_of_frame(self):
        # Arrange
        iface = self._make_started_interface()
        original = np.ones((480, 640), dtype=np.uint16) * 1000
        iface._depth_frame = original
        # Act
        result = iface.get_depth_frame()
        # Assert
        self.assertIsNotNone(result)
        self.assertTrue(np.array_equal(result, original))
        self.assertIsNot(result, original)

    # =========================================================================
    # PIXEL_TO_WORLD
    # =========================================================================

    def test_pixel_to_world_returns_none_when_fx_is_none(self):
        # Arrange
        iface = self._make_started_interface()
        iface._fx = None
        iface._depth_frame = np.ones((480, 640), dtype=np.uint16) * 1000
        # Assert
        self.assertIsNone(iface.pixel_to_world(320, 240))

    def test_pixel_to_world_returns_none_when_depth_scale_is_none(self):
        # Arrange
        iface = self._make_started_interface()
        iface._depth_scale = None
        iface._depth_frame = np.ones((480, 640), dtype=np.uint16) * 1000
        # Assert
        self.assertIsNone(iface.pixel_to_world(320, 240))

    def test_pixel_to_world_returns_none_when_depth_frame_is_none(self):
        # Arrange
        iface = self._make_started_interface()
        iface._depth_frame = None
        # Assert
        self.assertIsNone(iface.pixel_to_world(320, 240))

    def test_pixel_to_world_returns_none_when_pixel_out_of_bounds(self):
        # Arrange
        iface = self._make_started_interface()
        iface._depth_frame = np.ones((480, 640), dtype=np.uint16) * 1000
        # Assert — negative u, u at width boundary, v at height boundary
        self.assertIsNone(iface.pixel_to_world(-1, 240))
        self.assertIsNone(iface.pixel_to_world(640, 240))
        self.assertIsNone(iface.pixel_to_world(320, 480))

    def test_pixel_to_world_returns_none_when_depth_is_zero(self):
        # Arrange
        iface = self._make_started_interface()
        iface._depth_frame = np.zeros((480, 640), dtype=np.uint16)
        # Assert
        self.assertIsNone(iface.pixel_to_world(320, 240))

    def test_pixel_to_world_computes_correct_xyz(self):
        # Arrange — fx=fy=500, cx=320, cy=240, scale=0.001
        iface = self._make_started_interface()
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[290, 420] = 2000  # depth[v, u] = 2000 → Z = 2.0 m
        iface._depth_frame = depth
        # Act
        result = iface.pixel_to_world(420, 290)
        # Assert: X = (420-320)*2.0/500 = 0.4, Y = (290-240)*2.0/500 = 0.2, Z = 2.0
        self.assertIsNotNone(result)
        X, Y, Z = result
        self.assertAlmostEqual(Z, 2.0, places=6)
        self.assertAlmostEqual(X, 0.4, places=6)
        self.assertAlmostEqual(Y, 0.2, places=6)

    def test_pixel_to_world_at_principal_point_returns_zero_xy(self):
        # Arrange
        iface = self._make_started_interface()
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[240, 320] = 1000  # Z = 1.0 m
        iface._depth_frame = depth
        # Act
        result = iface.pixel_to_world(320, 240)
        # Assert
        self.assertIsNotNone(result)
        X, Y, Z = result
        self.assertAlmostEqual(X, 0.0, places=6)
        self.assertAlmostEqual(Y, 0.0, places=6)
        self.assertAlmostEqual(Z, 1.0, places=6)

    # =========================================================================
    # _DECODE_COLOR
    # =========================================================================

    def _make_color_frame_mock(self, fmt: str, h: int = 4, w: int = 4) -> MagicMock:
        """! Build a minimal ColorFrame mock for _decode_color tests."""
        frame = MagicMock()
        frame.get_height.return_value = h
        frame.get_width.return_value = w
        frame.get_format.return_value = fmt
        frame.get_data.return_value = bytes(h * w * 3)
        return frame

    @patch("devices.orbbec_interface.cv2")
    def test_decode_color_mjpg_calls_imdecode(self, mock_cv2):
        # Arrange
        frame = self._make_color_frame_mock("MJPG")
        expected = np.zeros((4, 4, 3), dtype=np.uint8)
        mock_cv2.imdecode.return_value = expected
        # Act
        result = OrbbecInterface._decode_color(frame)
        # Assert
        mock_cv2.imdecode.assert_called_once()
        self.assertEqual(result.shape, (4, 4, 3))

    @patch("devices.orbbec_interface.cv2")
    def test_decode_color_mjpg_raises_on_imdecode_failure(self, mock_cv2):
        # Arrange
        frame = self._make_color_frame_mock("MJPG")
        mock_cv2.imdecode.return_value = None
        # Act & Assert
        with self.assertRaises(ValueError):
            OrbbecInterface._decode_color(frame)

    @patch("devices.orbbec_interface.cv2")
    def test_decode_color_rgb_calls_cvtcolor_rgb2bgr(self, mock_cv2):
        # Arrange
        frame = self._make_color_frame_mock("RGB")
        mock_cv2.COLOR_RGB2BGR = 4
        mock_cv2.cvtColor.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        # Act
        OrbbecInterface._decode_color(frame)
        # Assert
        mock_cv2.cvtColor.assert_called_once()
        _, second_arg = mock_cv2.cvtColor.call_args[0]
        self.assertEqual(second_arg, mock_cv2.COLOR_RGB2BGR)

    @patch("devices.orbbec_interface.cv2")
    def test_decode_color_bgr_returns_reshaped_copy(self, mock_cv2):
        # Arrange
        frame = MagicMock()
        frame.get_height.return_value = 4
        frame.get_width.return_value = 4
        frame.get_format.return_value = "BGR"
        frame.get_data.return_value = bytes(range(48))
        # Act
        result = OrbbecInterface._decode_color(frame)
        # Assert — cv2 must not be touched for the BGR path
        mock_cv2.imdecode.assert_not_called()
        mock_cv2.cvtColor.assert_not_called()
        self.assertEqual(result.shape, (4, 4, 3))
        self.assertEqual(result.dtype, np.uint8)

    @patch("devices.orbbec_interface.cv2")
    def test_decode_color_fallback_calls_cvtcolor(self, mock_cv2):
        # Arrange
        frame = self._make_color_frame_mock("YUY2")
        mock_cv2.COLOR_RGB2BGR = 4
        mock_cv2.cvtColor.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        # Act
        OrbbecInterface._decode_color(frame)
        # Assert — fallback must use cvtColor, never imdecode
        mock_cv2.imdecode.assert_not_called()
        mock_cv2.cvtColor.assert_called_once()

    # =========================================================================
    # _DECODE_DEPTH
    # =========================================================================

    def _make_depth_frame_mock(self, h: int = 4, w: int = 4) -> MagicMock:
        """! Build a minimal DepthFrame mock for _decode_depth tests."""
        frame = MagicMock()
        frame.get_height.return_value = h
        frame.get_width.return_value = w
        frame.get_data.return_value = bytes(h * w * 2)  # uint16 = 2 bytes/pixel
        return frame

    def test_decode_depth_returns_correct_shape_and_dtype(self):
        # Arrange
        frame = self._make_depth_frame_mock(h=4, w=4)
        # Act
        result = OrbbecInterface._decode_depth(frame)
        # Assert
        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result.dtype, np.uint16)

    def test_decode_depth_returns_independent_copy(self):
        # Arrange
        frame = self._make_depth_frame_mock(h=4, w=4)
        # Act
        result = OrbbecInterface._decode_depth(frame)
        # Assert — result owns its data (is not a view)
        self.assertTrue(result.flags.owndata)


if __name__ == "__main__":
    unittest.main()
