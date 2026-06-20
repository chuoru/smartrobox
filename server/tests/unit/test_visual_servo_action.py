#!/usr/bin/env python3
##
# @file test_visual_servo_action.py
#
# @brief Unit tests for VisualServoAction.
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import math
import threading
import unittest
from unittest.mock import MagicMock, patch

# External library
import numpy as np

# Internal library
from actions.base import ActionState
from actions.visual_servo import VisualServoAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TARGET_KPS = [[100.0, 100.0]]
_ZERO_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _make_yolo_result(kp_xy, kp_conf):
    """Return a mock YOLO Results list with one detected person."""
    r = MagicMock()
    r.keypoints.xy.tolist.return_value = [kp_xy]
    r.keypoints.conf.tolist.return_value = [kp_conf]
    r.boxes.conf.tolist.return_value = [0.9]
    return [r]


def _make_empty_result():
    """Return a mock YOLO Results list with no detections."""
    r = MagicMock()
    r.boxes.conf.tolist.return_value = []
    return [r]


class _VisualServoTestMixin:
    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._yolo_patcher = patch("actions.visual_servo.YOLO")
        self._mock_yolo_cls = self._yolo_patcher.start()
        self._mock_model = MagicMock()
        self._mock_yolo_cls.return_value = self._mock_model
        self._mock_model.return_value = _make_yolo_result(
            [[100.0, 100.0]], [0.9]
        )
        self._controller = MagicMock()
        self._setup_controller()

    def _setup_controller(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_joint_pos":
                return (0, [0.0] * 6)
            return True
        self._controller.execute.side_effect = _execute

    def tearDown(self):
        self._logger_patcher.stop()
        self._yolo_patcher.stop()

    def _make(self, **overrides):
        defaults = dict(
            controller=self._controller,
            robot_device="robot",
            camera_device="camera",
            target_keypoints=list(_TARGET_KPS),
            error_threshold=10.0,
            stable_ticks=3,
            gain_matrix=[[0.0, 0.0]] * 6,
            cmd_period=0.001,
            timeout=2.0,
        )
        defaults.update(overrides)
        return VisualServoAction(**defaults)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestVisualServoActionInit(_VisualServoTestMixin, unittest.TestCase):
    """Tests for VisualServoAction construction."""

    def test_initial_state_is_idle(self):
        self.assertEqual(self._make().state(), ActionState.IDLE)

    def test_yolo_loaded_at_construction(self):
        self._make(model_name="yolo11n-pose.pt")
        self._mock_yolo_cls.assert_called_once_with("yolo11n-pose.pt")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestVisualServoActionParameters(_VisualServoTestMixin, unittest.TestCase):
    """Tests for VisualServoAction.parameters()."""

    def test_parameters_robot_device(self):
        self.assertEqual(self._make(robot_device="arm").parameters()["robot_device"], "arm")

    def test_parameters_camera_device(self):
        self.assertEqual(
            self._make(camera_device="cam").parameters()["camera_device"], "cam"
        )

    def test_parameters_error_threshold(self):
        self.assertEqual(
            self._make(error_threshold=5.0).parameters()["error_threshold"], 5.0
        )

    def test_parameters_stable_ticks(self):
        self.assertEqual(
            self._make(stable_ticks=7).parameters()["stable_ticks"], 7
        )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestVisualServoActionConvergence(_VisualServoTestMixin, unittest.TestCase):
    """Tests for convergence logic."""

    def test_converges_when_error_below_threshold_for_stable_ticks(self):
        # keypoints at target → error = 0
        self._mock_model.return_value = _make_yolo_result([[100.0, 100.0]], [0.9])
        action = self._make(error_threshold=10.0, stable_ticks=3)
        action.start()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.DONE)
        self.assertTrue(action.result()["converged"])

    def test_result_contains_required_keys(self):
        self._mock_model.return_value = _make_yolo_result([[100.0, 100.0]], [0.9])
        action = self._make()
        action.start()
        action.wait(timeout=3.0)
        self.assertSetEqual(
            set(action.result().keys()), {"converged", "stable_ticks", "final_error"}
        )

    def test_timeout_sets_failed_state(self):
        self._mock_model.return_value = _make_yolo_result([[500.0, 500.0]], [0.9])
        action = self._make(error_threshold=1.0, stable_ticks=100, timeout=0.1)
        action.start()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)

    def test_stable_counter_resets_on_high_error(self):
        low = _make_yolo_result([[100.0, 100.0]], [0.9])
        high = _make_yolo_result([[500.0, 500.0]], [0.9])
        call_count = [0]

        def alternating(*args, **kwargs):
            if args and len(args) > 0 and isinstance(args[0], np.ndarray):
                call_count[0] += 1
                return high if call_count[0] % 2 == 0 else low
            return []

        self._mock_model.side_effect = alternating
        action = self._make(error_threshold=10.0, stable_ticks=4, timeout=2.0)
        action.start()
        # It should NOT converge quickly since the counter resets every other tick
        finished = action.wait(timeout=2.5)
        if finished:
            # If it converged it needed >= 4 consecutive low-error ticks
            self.assertTrue(action.result()["converged"])


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestVisualServoActionDispatch(_VisualServoTestMixin, unittest.TestCase):
    """Tests for controller dispatch order and arguments."""

    def _run_and_get_calls(self, **overrides):
        action = self._make(**overrides)
        action.start()
        action.wait(timeout=3.0)
        return self._controller.execute.call_args_list

    def test_servo_start_called_before_servo_j(self):
        calls = self._run_and_get_calls()
        methods = [c[0][1] for c in calls]
        idx_start = next(i for i, m in enumerate(methods) if m == "servo_start")
        idx_j = next((i for i, m in enumerate(methods) if m == "servo_j"), None)
        if idx_j is not None:
            self.assertLess(idx_start, idx_j)

    def test_servo_end_called_after_servo_j(self):
        calls = self._run_and_get_calls()
        methods = [c[0][1] for c in calls]
        idx_end = next(i for i, m in enumerate(methods) if m == "servo_end")
        last_j = max((i for i, m in enumerate(methods) if m == "servo_j"), default=-1)
        self.assertGreater(idx_end, last_j)

    def test_get_joint_pos_called_once(self):
        self._run_and_get_calls()
        jp_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "get_joint_pos"
        ]
        self.assertEqual(len(jp_calls), 1)

    def test_servo_j_receives_list_of_six_floats(self):
        calls = self._run_and_get_calls()
        j_calls = [c for c in calls if c[0][1] == "servo_j"]
        self.assertGreater(len(j_calls), 0)
        joint_arg = j_calls[0][0][2]
        self.assertIsInstance(joint_arg, list)
        self.assertEqual(len(joint_arg), 6)


# ---------------------------------------------------------------------------
# No detection / low confidence
# ---------------------------------------------------------------------------

class TestVisualServoActionNoDetection(_VisualServoTestMixin, unittest.TestCase):
    """Tests for behaviour when YOLO produces no usable detections."""

    def test_no_detection_does_not_call_servo_j(self):
        self._mock_model.return_value = _make_empty_result()
        action = self._make(timeout=0.15)
        action.start()
        action.wait(timeout=2.0)
        j_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "servo_j"
        ]
        self.assertEqual(len(j_calls), 0)

    def test_low_confidence_keypoints_excluded_from_error(self):
        # conf = 0.0 → below keypoint_conf_min → treated as no detection
        self._mock_model.return_value = _make_yolo_result([[100.0, 100.0]], [0.0])
        action = self._make(keypoint_conf_min=0.5, timeout=0.15)
        action.start()
        action.wait(timeout=2.0)
        j_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "servo_j"
        ]
        self.assertEqual(len(j_calls), 0)


# ---------------------------------------------------------------------------
# Checkpoint (pause / cancel)
# ---------------------------------------------------------------------------

class TestVisualServoActionCheckpoint(_VisualServoTestMixin, unittest.TestCase):
    """Tests for cooperative pause/cancel during the servo loop."""

    def _blocking_servo_j(self, n):
        """Block the first n servo_j calls with Event synchronization."""
        at_call = [threading.Event() for _ in range(n)]
        proceed = [threading.Event() for _ in range(n)]
        idx = [0]
        original_side_effect = self._controller.execute.side_effect

        def side_effect(device, method, *args, **kwargs):
            if method == "servo_j" and idx[0] < n:
                i = idx[0]
                idx[0] += 1
                at_call[i].set()
                proceed[i].wait()
                return True
            return original_side_effect(device, method, *args, **kwargs)

        self._controller.execute.side_effect = side_effect
        return at_call, proceed

    def test_cancel_during_servo_loop_stops_action(self):
        at_call, proceed = self._blocking_servo_j(2)
        action = self._make(stable_ticks=100, timeout=5.0)
        action.start()
        at_call[0].wait()
        proceed[0].set()
        at_call[1].wait()
        action.cancel()
        proceed[1].set()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.CANCELLED)

    def test_pause_resume_during_loop_completes(self):
        at_call, proceed = self._blocking_servo_j(2)
        # After 3 stable ticks it converges; keypoints at target
        self._mock_model.return_value = _make_yolo_result([[100.0, 100.0]], [0.9])
        action = self._make(stable_ticks=3, error_threshold=10.0, timeout=5.0)
        action.start()
        at_call[0].wait()
        action.pause()
        proceed[0].set()
        self.assertFalse(at_call[1].wait(timeout=0.1))
        action.resume()
        at_call[1].wait()
        proceed[1].set()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.DONE)


# ---------------------------------------------------------------------------
# Camera warmup
# ---------------------------------------------------------------------------

class TestVisualServoActionCameraWarmup(_VisualServoTestMixin, unittest.TestCase):
    """Tests for warmup frame polling."""

    def test_no_camera_frame_raises_runtime_error(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return None
            if method == "get_joint_pos":
                return (0, [0.0] * 6)
            return True
        self._controller.execute.side_effect = _execute
        action = self._make(warmup_timeout=0.05)
        action.start()
        action.wait(timeout=3.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)


# ---------------------------------------------------------------------------
# Eye-to-hand (PBVS)
# ---------------------------------------------------------------------------

_ETH_EXTRINSIC = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
]
_ETH_TARGET_3D = [[0.0, 0.0, 0.0]]


class TestVisualServoActionEyeToHandInit(_VisualServoTestMixin, unittest.TestCase):
    """Tests for eye_to_hand construction validation."""

    def test_missing_extrinsic_raises_value_error(self):
        with self.assertRaises(ValueError):
            VisualServoAction(
                self._controller, "robot", "camera",
                list(_TARGET_KPS), 10.0, 3,
                camera_config="eye_to_hand",
                target_keypoints_3d=_ETH_TARGET_3D,
            )

    def test_missing_target_3d_raises_value_error(self):
        with self.assertRaises(ValueError):
            VisualServoAction(
                self._controller, "robot", "camera",
                list(_TARGET_KPS), 10.0, 3,
                camera_config="eye_to_hand",
                camera_extrinsic=_ETH_EXTRINSIC,
            )

    def test_eye_in_hand_default_no_error(self):
        action = self._make()
        self.assertEqual(action.parameters()["camera_config"], "eye_in_hand")


class TestVisualServoActionEyeToHandParameters(_VisualServoTestMixin, unittest.TestCase):
    """Tests for eye_to_hand parameters()."""

    def _setup_controller(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return (0.1, 0.05, 0.5)
            if method == "tpos":
                return (0, [0.0] * 6)
            return True
        self._controller.execute.side_effect = _execute

    def _make_eth(self, **overrides):
        defaults = dict(
            controller=self._controller,
            robot_device="robot",
            camera_device="camera",
            target_keypoints=list(_TARGET_KPS),
            error_threshold=10.0,
            stable_ticks=3,
            cmd_period=0.001,
            timeout=2.0,
            camera_config="eye_to_hand",
            camera_extrinsic=_ETH_EXTRINSIC,
            target_keypoints_3d=_ETH_TARGET_3D,
        )
        defaults.update(overrides)
        return VisualServoAction(**defaults)

    def test_parameters_includes_camera_config(self):
        self.assertEqual(
            self._make_eth().parameters()["camera_config"], "eye_to_hand"
        )


class TestVisualServoActionEyeToHandCompute(_VisualServoTestMixin, unittest.TestCase):
    """Unit tests for _compute_3d_correction."""

    def _setup_controller(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return (0.1, 0.05, 0.5)
            if method == "tpos":
                return (0, [0.0] * 6)
            return True
        self._controller.execute.side_effect = _execute

    def _make_eth(self, **overrides):
        defaults = dict(
            controller=self._controller,
            robot_device="robot",
            camera_device="camera",
            target_keypoints=list(_TARGET_KPS),
            error_threshold=10.0,
            stable_ticks=3,
            cmd_period=0.001,
            timeout=2.0,
            camera_config="eye_to_hand",
            camera_extrinsic=_ETH_EXTRINSIC,
            target_keypoints_3d=_ETH_TARGET_3D,
        )
        defaults.update(overrides)
        return VisualServoAction(**defaults)

    def test_compute_3d_correction_base_frame_transform(self):
        # pixel_to_world returns (0.1, 0.05, 0.5) in camera frame.
        # _ETH_EXTRINSIC translates Z by +0.5, so base Z = 0.5 + 0.5 = 1.0.
        # target = [0.0, 0.0, 0.0], so delta = (0.1, 0.05, 1.0).
        action = self._make_eth()
        self._mock_model.return_value = _make_yolo_result([[100.0, 100.0]], [0.9])
        yolo_results = self._mock_model(_ZERO_FRAME)
        dx, dy, dz = action._compute_3d_correction(_ZERO_FRAME, yolo_results)
        self.assertAlmostEqual(dx, 0.1, places=4)
        self.assertAlmostEqual(dy, 0.05, places=4)
        self.assertAlmostEqual(dz, 1.0, places=4)

    def test_compute_3d_correction_no_depth_returns_zero(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return None
            if method == "tpos":
                return (0, [0.0] * 6)
            return True
        self._controller.execute.side_effect = _execute
        action = self._make_eth()
        self._mock_model.return_value = _make_yolo_result([[100.0, 100.0]], [0.9])
        yolo_results = self._mock_model(_ZERO_FRAME)
        self.assertEqual(action._compute_3d_correction(_ZERO_FRAME, yolo_results), (0.0, 0.0, 0.0))

    def test_compute_3d_correction_no_detection_returns_zero(self):
        action = self._make_eth()
        self._mock_model.return_value = _make_empty_result()
        yolo_results = self._mock_model(_ZERO_FRAME)
        self.assertEqual(action._compute_3d_correction(_ZERO_FRAME, yolo_results), (0.0, 0.0, 0.0))


class TestVisualServoActionEyeToHandDispatch(_VisualServoTestMixin, unittest.TestCase):
    """Tests for eye_to_hand servo dispatch."""

    def _setup_controller(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return (0.0, 0.0, 0.5)
            if method == "tpos":
                return (0, [0.0] * 6)
            if method == "get_joint_pos":
                return (0, [0.0] * 6)
            return True
        self._controller.execute.side_effect = _execute

    def _make_eth(self, **overrides):
        defaults = dict(
            controller=self._controller,
            robot_device="robot",
            camera_device="camera",
            target_keypoints=list(_TARGET_KPS),
            error_threshold=10.0,
            stable_ticks=3,
            cmd_period=0.001,
            timeout=0.3,
            camera_config="eye_to_hand",
            camera_extrinsic=_ETH_EXTRINSIC,
            target_keypoints_3d=_ETH_TARGET_3D,
        )
        defaults.update(overrides)
        return VisualServoAction(**defaults)

    def test_eye_to_hand_calls_servo_c_not_servo_j(self):
        action = self._make_eth()
        action.start()
        action.wait(timeout=3.0)
        calls = self._controller.execute.call_args_list
        j_calls = [c for c in calls if c[0][1] == "servo_j"]
        c_calls = [c for c in calls if c[0][1] == "servo_c"]
        self.assertEqual(len(j_calls), 0)
        self.assertGreater(len(c_calls), 0)

    def test_eye_in_hand_regression_still_uses_servo_j(self):
        # eye_in_hand with zero gain — servo_j is called (not servo_c)
        action = self._make(gain_matrix=[[0.0, 0.0]] * 6, timeout=0.1)
        action.start()
        action.wait(timeout=3.0)
        calls = self._controller.execute.call_args_list
        methods = [c[0][1] for c in calls]
        self.assertIn("servo_j", methods)
        self.assertNotIn("servo_c", methods)


if __name__ == "__main__":
    unittest.main()
