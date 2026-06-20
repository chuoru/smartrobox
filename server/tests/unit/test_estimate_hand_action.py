#!/usr/bin/env python3
##
# @file test_estimate_hand_action.py
#
# @brief Unit tests for EstimateHandAction.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import unittest
from unittest.mock import MagicMock, patch

# External library
import numpy as np

# Internal library
from actions.base import ActionState
from actions.estimate_hand import EstimateHandAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_KP_21 = [[float(i), float(i)] for i in range(21)]
_CONF_21 = [0.9] * 21


def _make_hand_result(kp_xy_list, kp_conf_list, boxes_xyxy=None, boxes_conf=None):
    """Return a mock YOLO Results list with detected hands."""
    r = MagicMock()
    r.keypoints.xy.tolist.return_value = kp_xy_list
    r.keypoints.conf.tolist.return_value = kp_conf_list
    r.boxes.xyxy.tolist.return_value = boxes_xyxy or [[0.0, 0.0, 100.0, 100.0]] * len(kp_xy_list)
    r.boxes.conf.tolist.return_value = boxes_conf or [0.9] * len(kp_xy_list)
    return [r]


def _make_empty_result():
    r = MagicMock()
    r.keypoints = None
    r.boxes = None
    return [r]


class _EstimateHandTestMixin:
    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._yolo_patcher = patch("actions.estimate_hand.YOLO")
        self._mock_yolo_cls = self._yolo_patcher.start()
        self._mock_model = MagicMock()
        self._mock_yolo_cls.return_value = self._mock_model
        self._mock_model.return_value = _make_hand_result([_KP_21], [_CONF_21])
        self._controller = MagicMock()
        self._controller.execute.side_effect = self._default_execute

    def _default_execute(self, device, method, *args, **kwargs):
        if method == "get_color_frame":
            return _ZERO_FRAME
        return True

    def tearDown(self):
        self._logger_patcher.stop()
        self._yolo_patcher.stop()

    def _make(self, **overrides):
        defaults = dict(
            controller=self._controller,
            device_name="camera",
        )
        defaults.update(overrides)
        return EstimateHandAction(**defaults)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestEstimateHandActionInit(_EstimateHandTestMixin, unittest.TestCase):
    """Tests for EstimateHandAction construction."""

    def test_initial_state_is_idle(self):
        self.assertEqual(self._make().state(), ActionState.IDLE)

    def test_yolo_loaded_at_construction(self):
        self._make(model_name="yolo11n-hand-pose.pt")
        self._mock_yolo_cls.assert_called_once_with("yolo11n-hand-pose.pt")

    def test_default_model_name(self):
        action = self._make()
        self.assertEqual(action.parameters()["model_name"], "yolo11n-hand-pose.pt")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestEstimateHandActionParameters(_EstimateHandTestMixin, unittest.TestCase):
    """Tests for EstimateHandAction.parameters()."""

    def test_parameters_device_name(self):
        self.assertEqual(self._make(device_name="cam").parameters()["device_name"], "cam")

    def test_parameters_warmup_timeout(self):
        self.assertEqual(
            self._make(warmup_timeout=5.0).parameters()["warmup_timeout"], 5.0
        )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestEstimateHandActionResult(_EstimateHandTestMixin, unittest.TestCase):
    """Tests for result structure and content."""

    def _run_and_get_result(self, **overrides):
        action = self._make(**overrides)
        action.start()
        action.wait(timeout=3.0)
        return action.result()

    def test_result_is_list(self):
        self.assertIsInstance(self._run_and_get_result(), list)

    def test_single_hand_result_has_required_keys(self):
        result = self._run_and_get_result()
        self.assertEqual(len(result), 1)
        hand = result[0]
        self.assertSetEqual(set(hand.keys()), {"keypoints", "keypoint_conf", "bbox", "conf"})

    def test_keypoints_length_is_21(self):
        result = self._run_and_get_result()
        self.assertEqual(len(result[0]["keypoints"]), 21)

    def test_keypoint_conf_length_is_21(self):
        result = self._run_and_get_result()
        self.assertEqual(len(result[0]["keypoint_conf"]), 21)

    def test_bbox_length_is_4(self):
        result = self._run_and_get_result()
        self.assertEqual(len(result[0]["bbox"]), 4)

    def test_multiple_hands_returned(self):
        self._mock_model.return_value = _make_hand_result(
            [_KP_21, _KP_21], [_CONF_21, _CONF_21]
        )
        result = self._run_and_get_result()
        self.assertEqual(len(result), 2)

    def test_no_detection_returns_empty_list(self):
        self._mock_model.return_value = _make_empty_result()
        result = self._run_and_get_result()
        self.assertEqual(result, [])

    def test_state_is_done_on_success(self):
        action = self._make()
        action.start()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.DONE)


# ---------------------------------------------------------------------------
# Camera warmup
# ---------------------------------------------------------------------------

class TestEstimateHandActionWarmup(_EstimateHandTestMixin, unittest.TestCase):
    """Tests for warmup frame polling."""

    def test_no_frame_raises_runtime_error(self):
        self._controller.execute.side_effect = lambda d, m, *a, **k: None
        action = self._make(warmup_timeout=0.05)
        action.start()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)

    def test_delayed_frame_eventually_succeeds(self):
        call_count = [0]

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                call_count[0] += 1
                return _ZERO_FRAME if call_count[0] >= 3 else None
            return True

        self._controller.execute.side_effect = _execute
        action = self._make(warmup_timeout=1.0)
        action.start()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.DONE)


if __name__ == "__main__":
    unittest.main()
