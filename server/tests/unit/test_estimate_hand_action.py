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
_KP_21 = [[float(i * 10), float(i * 5)] for i in range(21)]
_VIS_21 = [0.9] * 21


def _make_landmark(x: float, y: float, visibility: float = 0.9):
    lm = MagicMock()
    lm.x = x / 640.0
    lm.y = y / 480.0
    lm.visibility = visibility
    return lm


def _make_mp_result(kp_lists=None, vis_lists=None, hand_scores=None):
    """Return a mock MediaPipe Hands result."""
    result = MagicMock()
    if not kp_lists:
        result.multi_hand_landmarks = None
        result.multi_handedness = None
        return result

    hand_landmarks_list = []
    for i, kps in enumerate(kp_lists):
        vis = vis_lists[i] if vis_lists else [0.9] * len(kps)
        hl = MagicMock()
        hl.landmark = [_make_landmark(kps[j][0], kps[j][1], vis[j]) for j in range(len(kps))]
        hand_landmarks_list.append(hl)

    result.multi_hand_landmarks = hand_landmarks_list
    result.multi_handedness = [
        MagicMock(classification=[MagicMock(score=(hand_scores[i] if hand_scores else 0.9))])
        for i in range(len(kp_lists))
    ]
    return result


class _EstimateHandTestMixin:
    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._mp_patcher = patch("actions.estimate_hand.mp")
        self._mock_mp = self._mp_patcher.start()
        self._mock_hands_instance = MagicMock()
        self._mock_mp.solutions.hands.Hands.return_value = self._mock_hands_instance
        self._mock_hands_instance.process.return_value = _make_mp_result([_KP_21])
        self._controller = MagicMock()
        self._controller.execute.side_effect = self._default_execute

    def _default_execute(self, device, method, *args, **kwargs):
        if method == "get_color_frame":
            return _ZERO_FRAME
        return True

    def tearDown(self):
        self._logger_patcher.stop()
        self._mp_patcher.stop()

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

    def test_mp_hands_initialized_at_construction(self):
        self._make(max_num_hands=1, min_detection_confidence=0.7)
        self._mock_mp.solutions.hands.Hands.assert_called_once_with(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
        )

    def test_default_max_num_hands(self):
        self.assertEqual(self._make().parameters()["max_num_hands"], 2)

    def test_default_min_detection_confidence(self):
        self.assertEqual(self._make().parameters()["min_detection_confidence"], 0.5)


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

    def test_parameters_max_num_hands(self):
        self.assertEqual(self._make(max_num_hands=1).parameters()["max_num_hands"], 1)

    def test_parameters_min_detection_confidence(self):
        self.assertEqual(
            self._make(min_detection_confidence=0.8).parameters()["min_detection_confidence"],
            0.8,
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
        self._mock_hands_instance.process.return_value = _make_mp_result(
            [_KP_21, _KP_21]
        )
        result = self._run_and_get_result()
        self.assertEqual(len(result), 2)

    def test_no_detection_returns_empty_list(self):
        self._mock_hands_instance.process.return_value = _make_mp_result()
        result = self._run_and_get_result()
        self.assertEqual(result, [])

    def test_hands_sorted_by_confidence_descending(self):
        self._mock_hands_instance.process.return_value = _make_mp_result(
            [_KP_21, _KP_21], hand_scores=[0.6, 0.9]
        )
        result = self._run_and_get_result()
        self.assertGreaterEqual(result[0]["conf"], result[1]["conf"])

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
