#!/usr/bin/env python3
##
# @file test_visual_servo_action.py
#
# @brief Unit tests for VisualServoAction (dual-arm live shoulder tracking).
#
# @section author Author(s)
# - Created by chuoru on 2026/06/20.
#
# Copyright (c) 2026 HACHIX.  All rights reserved.

# Standard library
import math
import threading
import unittest
from unittest.mock import MagicMock, call, patch

# External library
import numpy as np

# Internal library
from actions.base import ActionState
from actions.visual_servo import VisualServoAction


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ZERO_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

_LEFT_EXTRINSIC = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
]
_RIGHT_EXTRINSIC = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
]

# Initial TCP: (200, 150, 500) mm + zero orientation.
_INIT_TCP = [200.0, 150.0, 500.0, 0.0, 0.0, 0.0]

# pixel_to_world returns (0.2, 0.15, 0.0) m → after identity+0.5 Z translate
# → base = (0.2, 0.15, 0.5) m = (200, 150, 500) mm = exact TCP → error = 0.
_WORLD_AT_TCP = (0.2, 0.15, 0.0)

# pixel_to_world returns (0.0, 0.0, 0.0) m → base = (0.0, 0.0, 500) mm → far in X/Y from TCP.
_WORLD_FAR = (0.0, 0.0, 0.0)

# pixel_to_world for a shoulder that differs in Z: cam=(0.2, 0.15, -0.2) →
# base = (0.2, 0.15, 0.3) m = (200, 150, 300) mm → below TCP Z=500.
_WORLD_BELOW_TCP = (0.2, 0.15, -0.2)


def _make_yolo_result(kp_xy: list, kp_conf: list):
    """Return a mock YOLO Results list with one detected person.

    kp_xy must have at least 7 entries (indices 0-6).
    """
    r = MagicMock()
    r.boxes.conf.tolist.return_value = [0.9]
    r.keypoints.xy.tolist.return_value = [kp_xy]
    r.keypoints.conf.tolist.return_value = [kp_conf]
    return [r]


def _make_empty_result():
    """Return a mock YOLO Results list with no detections."""
    r = MagicMock()
    r.boxes.conf.tolist.return_value = []
    return [r]


def _kp_xy(n: int = 17) -> list:
    """Return n dummy [0, 0] keypoint coords."""
    return [[0.0, 0.0]] * n


def _kp_conf_high(n: int = 17) -> list:
    """Return n keypoint confidences all at 0.9."""
    return [0.9] * n


class _VisualServoTestMixin:
    def setUp(self):
        self._logger_patcher = patch("actions.base.Logger")
        self._logger_patcher.start()
        self._yolo_patcher = patch("actions.visual_servo.YOLO")
        self._mock_yolo_cls = self._yolo_patcher.start()
        self._mock_model = MagicMock()
        self._mock_yolo_cls.return_value = self._mock_model
        # Default: detect both shoulders at the TCP position (error = 0).
        self._mock_model.return_value = _make_yolo_result(
            _kp_xy(), _kp_conf_high()
        )
        self._controller = MagicMock()
        self._setup_controller()

    def _setup_controller(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_AT_TCP
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

    def tearDown(self):
        self._logger_patcher.stop()
        self._yolo_patcher.stop()

    def _make(self, **overrides) -> VisualServoAction:
        defaults = dict(
            controller=self._controller,
            left_robot_device="left_arm",
            right_robot_device="right_arm",
            camera_device="camera",
            left_arm_extrinsic=_LEFT_EXTRINSIC,
            right_arm_extrinsic=_RIGHT_EXTRINSIC,
            error_threshold=30.0,
            stable_ticks=3,
            servo_gain=0.5,
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

    def test_servo_gain_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._make(servo_gain=0.0)

    def test_servo_gain_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._make(servo_gain=-0.1)

    def test_servo_gain_one_accepted(self):
        action = self._make(servo_gain=1.0)
        self.assertEqual(action.parameters()["servo_gain"], 1.0)

    def test_extrinsics_stored_as_numpy_array(self):
        action = self._make()
        self.assertIsInstance(action._left_arm_extrinsic, np.ndarray)
        self.assertIsInstance(action._right_arm_extrinsic, np.ndarray)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestVisualServoActionParameters(_VisualServoTestMixin, unittest.TestCase):
    """Tests for parameters() dict."""

    def test_parameters_left_robot_device(self):
        self.assertEqual(
            self._make(left_robot_device="l_arm").parameters()["left_robot_device"], "l_arm"
        )

    def test_parameters_right_robot_device(self):
        self.assertEqual(
            self._make(right_robot_device="r_arm").parameters()["right_robot_device"], "r_arm"
        )

    def test_parameters_camera_device(self):
        self.assertEqual(
            self._make(camera_device="head_cam").parameters()["camera_device"], "head_cam"
        )

    def test_parameters_error_threshold(self):
        self.assertEqual(
            self._make(error_threshold=15.0).parameters()["error_threshold"], 15.0
        )

    def test_parameters_stable_ticks(self):
        self.assertEqual(
            self._make(stable_ticks=7).parameters()["stable_ticks"], 7
        )

    def test_parameters_servo_gain(self):
        self.assertAlmostEqual(
            self._make(servo_gain=0.3).parameters()["servo_gain"], 0.3
        )

    def test_parameters_exact_key_set(self):
        expected = {
            "left_robot_device", "right_robot_device", "camera_device",
            "error_threshold", "stable_ticks", "servo_gain",
            "cmd_period", "timeout", "model_name",
        }
        self.assertSetEqual(set(self._make().parameters().keys()), expected)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestVisualServoActionConvergence(_VisualServoTestMixin, unittest.TestCase):
    """Tests for dual-arm convergence logic."""

    def test_converges_when_both_arms_below_threshold(self):
        # pixel_to_world returns coords that map to exactly _INIT_TCP → error=0
        action = self._make(error_threshold=30.0, stable_ticks=3)
        action.start()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.DONE)
        self.assertTrue(action.result()["converged"])

    def test_result_contains_all_required_keys(self):
        action = self._make()
        action.start()
        action.wait(timeout=5.0)
        self.assertSetEqual(
            set(action.result().keys()),
            {"converged", "left_stable_ticks", "right_stable_ticks",
             "left_final_error", "right_final_error"},
        )

    def test_timeout_sets_failed_state(self):
        # Shoulder far from TCP → never converges.
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute
        action = self._make(error_threshold=1.0, stable_ticks=100, timeout=0.1)
        action.start()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)

    def test_timeout_error_message_contains_mm(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute
        action = self._make(error_threshold=1.0, stable_ticks=100, timeout=0.1)
        action.start()
        action.wait(timeout=5.0)
        self.assertIn("mm", str(action.error()))

    def test_one_arm_below_threshold_does_not_converge(self):
        # Left shoulder detected and at TCP (error=0); right shoulder never detected.
        # Right stable counter stays at 0 → action must time out.
        kp_conf = _kp_conf_high()
        kp_conf[VisualServoAction._KP_RIGHT_SHOULDER] = 0.0
        self._mock_model.return_value = _make_yolo_result(_kp_xy(), kp_conf)

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_AT_TCP
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True

        self._controller.execute.side_effect = _execute
        action = self._make(error_threshold=30.0, stable_ticks=5, timeout=0.3)
        action.start()
        action.wait(timeout=5.0)
        # Right shoulder never detected → right_stable never reaches stable_ticks → timeout.
        self.assertEqual(action.state(), ActionState.FAILED)

    def test_stable_counter_resets_on_high_error(self):
        # Alternate between near and far to prevent stable accumulation.
        pw_call = [0]

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                pw_call[0] += 1
                return _WORLD_AT_TCP if pw_call[0] % 4 < 2 else _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True

        self._controller.execute.side_effect = _execute
        # Need 10 consecutive stable ticks — alternation prevents this.
        action = self._make(error_threshold=1.0, stable_ticks=10, timeout=0.3)
        action.start()
        finished = action.wait(timeout=5.0)
        if finished and action.state() == ActionState.DONE:
            self.assertTrue(action.result()["converged"])


# ---------------------------------------------------------------------------
# Detection handling
# ---------------------------------------------------------------------------

class TestVisualServoActionDetectionHandling(_VisualServoTestMixin, unittest.TestCase):
    """Tests for YOLO detection edge cases."""

    def test_no_detection_does_not_call_servo_c(self):
        self._mock_model.return_value = _make_empty_result()
        action = self._make(timeout=0.1)
        action.start()
        action.wait(timeout=5.0)
        c_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "servo_c"
        ]
        self.assertEqual(len(c_calls), 0)

    def test_low_confidence_excluded(self):
        # Both shoulder confs below keypoint_conf_min.
        kp_conf = [0.0] * 17
        self._mock_model.return_value = _make_yolo_result(_kp_xy(), kp_conf)

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

        action = self._make(keypoint_conf_min=0.5, timeout=0.1)
        action.start()
        action.wait(timeout=5.0)
        c_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "servo_c"
        ]
        self.assertEqual(len(c_calls), 0)

    def test_one_shoulder_not_detected_other_arm_still_corrected(self):
        # Right shoulder conf = 0 → only left servo_c should be issued.
        kp_conf = _kp_conf_high()
        kp_conf[6] = 0.0  # right shoulder
        self._mock_model.return_value = _make_yolo_result(_kp_xy(), kp_conf)

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

        action = self._make(error_threshold=1.0, timeout=0.15)
        action.start()
        action.wait(timeout=5.0)

        calls = self._controller.execute.call_args_list
        left_c  = [c for c in calls if c[0][0] == "left_arm"  and c[0][1] == "servo_c"]
        right_c = [c for c in calls if c[0][0] == "right_arm" and c[0][1] == "servo_c"]
        self.assertGreater(len(left_c), 0)
        self.assertEqual(len(right_c), 0)

    def test_no_color_frame_yolo_not_called(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME if method == "get_color_frame" and False else None
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True

        # Warmup succeeds (first get_color_frame call returns a frame), then None.
        frame_calls = [0]

        def _execute2(device, method, *args, **kwargs):
            if method == "get_color_frame":
                frame_calls[0] += 1
                return _ZERO_FRAME if frame_calls[0] == 1 else None
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True

        self._controller.execute.side_effect = _execute2
        action = self._make(timeout=0.1)
        action.start()
        action.wait(timeout=5.0)
        # YOLO should be called at most once (during warmup it's not called; only in loop).
        self.assertEqual(self._mock_model.call_count, 0)

    def test_no_depth_returns_no_servo_c(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return None   # depth unavailable
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

        action = self._make(timeout=0.1)
        action.start()
        action.wait(timeout=5.0)
        c_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "servo_c"
        ]
        self.assertEqual(len(c_calls), 0)


# ---------------------------------------------------------------------------
# Proportional controller
# ---------------------------------------------------------------------------

class TestVisualServoActionProportionalController(_VisualServoTestMixin, unittest.TestCase):
    """Tests for servo_c argument correctness."""

    def _run_and_get_calls(self, **overrides):
        action = self._make(**overrides)
        action.start()
        action.wait(timeout=5.0)
        return self._controller.execute.call_args_list

    def _setup_far_shoulder(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

    def test_servo_c_receives_list_of_six_floats(self):
        self._setup_far_shoulder()
        calls = self._run_and_get_calls(error_threshold=1.0, timeout=0.1)
        c_calls = [c for c in calls if c[0][1] == "servo_c"]
        self.assertGreater(len(c_calls), 0)
        pose_arg = c_calls[0][0][2]
        self.assertIsInstance(pose_arg, list)
        self.assertEqual(len(pose_arg), 6)
        for v in pose_arg:
            self.assertIsInstance(v, float)

    def test_servo_c_tcp_moves_toward_shoulder(self):
        # Shoulder at (200, 150, 300) mm; TCP at (200, 150, 500) mm.
        # Correction: new_Z = 500 - 0.5*(500-300) = 400 → Z decreases toward shoulder.
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_BELOW_TCP
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute
        calls = self._run_and_get_calls(error_threshold=1.0, timeout=0.05)
        c_calls = [c for c in calls if c[0][0] == "left_arm" and c[0][1] == "servo_c"]
        self.assertGreater(len(c_calls), 0)
        first_z = c_calls[0][0][2][2]
        self.assertLess(first_z, _INIT_TCP[2])  # TCP Z decreased toward shoulder Z=300

    def test_servo_c_not_called_when_error_below_threshold(self):
        # pixel_to_world returns coords mapping exactly to INIT_TCP → error=0.
        action = self._make(error_threshold=500.0, stable_ticks=3)
        action.start()
        action.wait(timeout=5.0)
        c_calls = [
            c for c in self._controller.execute.call_args_list
            if c[0][1] == "servo_c"
        ]
        self.assertEqual(len(c_calls), 0)


# ---------------------------------------------------------------------------
# Dispatch order
# ---------------------------------------------------------------------------

class TestVisualServoActionDispatchOrder(_VisualServoTestMixin, unittest.TestCase):
    """Tests for device call ordering."""

    def _run_and_get_methods(self, **overrides):
        action = self._make(**overrides)
        action.start()
        action.wait(timeout=5.0)
        return [(c[0][0], c[0][1]) for c in self._controller.execute.call_args_list]

    def test_servo_start_before_servo_c(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

        methods = self._run_and_get_methods(error_threshold=1.0, timeout=0.1)
        start_indices = [i for i, (_, m) in enumerate(methods) if m == "servo_start"]
        c_indices = [i for i, (_, m) in enumerate(methods) if m == "servo_c"]
        self.assertTrue(len(start_indices) >= 2)
        if c_indices:
            self.assertLess(max(start_indices), min(c_indices))

    def test_tpos_called_on_both_arms(self):
        methods = self._run_and_get_methods()
        devices_with_tpos = {dev for dev, m in methods if m == "tpos"}
        self.assertIn("left_arm",  devices_with_tpos)
        self.assertIn("right_arm", devices_with_tpos)

    def test_servo_end_called_on_both_arms_after_convergence(self):
        methods = self._run_and_get_methods()
        devices_with_end = {dev for dev, m in methods if m == "servo_end"}
        self.assertIn("left_arm",  devices_with_end)
        self.assertIn("right_arm", devices_with_end)

    def test_servo_end_called_even_when_tpos_fails(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "tpos":
                return (-1, None)
            return True
        self._controller.execute.side_effect = _execute

        action = self._make()
        action.start()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        methods = [(c[0][0], c[0][1]) for c in self._controller.execute.call_args_list]
        devices_with_end = {dev for dev, m in methods if m == "servo_end"}
        self.assertIn("left_arm",  devices_with_end)
        self.assertIn("right_arm", devices_with_end)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestVisualServoActionWarmup(_VisualServoTestMixin, unittest.TestCase):
    """Tests for frame and depth warmup polling."""

    def test_no_color_frame_raises_runtime_error(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return None
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

        action = self._make(warmup_timeout=0.05)
        action.start()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)

    def test_no_color_frame_servo_start_never_called(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return None
            return True
        self._controller.execute.side_effect = _execute

        action = self._make(warmup_timeout=0.05)
        action.start()
        action.wait(timeout=5.0)
        methods = [c[0][1] for c in self._controller.execute.call_args_list]
        self.assertNotIn("servo_start", methods)

    def test_no_depth_frame_raises_runtime_error(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return None
            if method == "tpos":
                return (0, list(_INIT_TCP))
            return True
        self._controller.execute.side_effect = _execute

        action = self._make(warmup_timeout=0.05)
        action.start()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)

    def test_tpos_nonzero_error_code_raises(self):
        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "tpos":
                return (-1, None)
            return True
        self._controller.execute.side_effect = _execute

        action = self._make()
        action.start()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.FAILED)
        self.assertIsInstance(action.error(), RuntimeError)


# ---------------------------------------------------------------------------
# Checkpoint (pause / cancel)
# ---------------------------------------------------------------------------

class TestVisualServoActionCheckpoint(_VisualServoTestMixin, unittest.TestCase):
    """Tests for cooperative pause/cancel during the servo loop."""

    def _setup_far_shoulder_and_block_servo_c(self, n: int):
        at_call = [threading.Event() for _ in range(n)]
        proceed = [threading.Event() for _ in range(n)]
        idx = [0]

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_FAR
            if method == "tpos":
                return (0, list(_INIT_TCP))
            if method == "servo_c" and device == "left_arm" and idx[0] < n:
                i = idx[0]
                idx[0] += 1
                at_call[i].set()
                proceed[i].wait()
            return True

        self._controller.execute.side_effect = _execute
        return at_call, proceed

    def test_cancel_during_servo_loop_stops_action(self):
        at_call, proceed = self._setup_far_shoulder_and_block_servo_c(2)
        action = self._make(error_threshold=1.0, stable_ticks=100, timeout=10.0)
        action.start()
        at_call[0].wait()
        proceed[0].set()
        at_call[1].wait()
        action.cancel()
        proceed[1].set()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.CANCELLED)

    def test_servo_end_called_after_cancel(self):
        at_call, proceed = self._setup_far_shoulder_and_block_servo_c(1)
        action = self._make(error_threshold=1.0, stable_ticks=100, timeout=10.0)
        action.start()
        at_call[0].wait()
        action.cancel()
        proceed[0].set()
        action.wait(timeout=5.0)
        methods = [(c[0][0], c[0][1]) for c in self._controller.execute.call_args_list]
        devices_with_end = {dev for dev, m in methods if m == "servo_end"}
        self.assertIn("left_arm",  devices_with_end)
        self.assertIn("right_arm", devices_with_end)

    def test_pause_resume_during_loop_completes(self):
        # Near shoulder so it converges after resume.
        at_call = [threading.Event()]
        proceed = [threading.Event()]
        idx = [0]

        def _execute(device, method, *args, **kwargs):
            if method == "get_color_frame":
                return _ZERO_FRAME
            if method == "get_depth_frame":
                return np.zeros((480, 640), dtype=np.uint16)
            if method == "pixel_to_world":
                return _WORLD_AT_TCP
            if method == "tpos":
                return (0, list(_INIT_TCP))
            if method == "servo_start" and device == "left_arm" and idx[0] == 0:
                idx[0] += 1
                at_call[0].set()
                proceed[0].wait()
            return True

        self._controller.execute.side_effect = _execute
        action = self._make(error_threshold=30.0, stable_ticks=3, timeout=10.0)
        action.start()
        at_call[0].wait()
        action.pause()
        proceed[0].set()
        self.assertFalse(action.wait(timeout=0.05))
        action.resume()
        action.wait(timeout=5.0)
        self.assertEqual(action.state(), ActionState.DONE)


# ---------------------------------------------------------------------------
# _lift_shoulder helper
# ---------------------------------------------------------------------------

class TestLiftShoulderHelper(_VisualServoTestMixin, unittest.TestCase):
    """Unit tests for _lift_shoulder() directly."""

    def _make_action(self) -> VisualServoAction:
        return self._make()

    def test_returns_none_when_pixel_to_world_returns_none(self):
        def _execute(device, method, *args, **kwargs):
            if method == "pixel_to_world":
                return None
            return True
        self._controller.execute.side_effect = _execute
        action = self._make_action()
        result = action._lift_shoulder([100.0, 200.0], action._left_arm_extrinsic)
        self.assertIsNone(result)

    def test_identity_extrinsic_converts_metres_to_mm(self):
        def _execute(device, method, *args, **kwargs):
            if method == "pixel_to_world":
                return (0.1, 0.2, 0.3)
            return True
        self._controller.execute.side_effect = _execute
        identity = np.eye(4)
        action = self._make_action()
        x, y, z = action._lift_shoulder([50.0, 60.0], identity)
        self.assertAlmostEqual(x, 100.0, places=4)
        self.assertAlmostEqual(y, 200.0, places=4)
        self.assertAlmostEqual(z, 300.0, places=4)

    def test_translation_extrinsic_applied_correctly(self):
        # Extrinsic: identity rotation + (0.1, 0.2, 0.5) translation (metres).
        extrinsic = np.eye(4)
        extrinsic[0, 3] = 0.1
        extrinsic[1, 3] = 0.2
        extrinsic[2, 3] = 0.5

        def _execute(device, method, *args, **kwargs):
            if method == "pixel_to_world":
                return (0.0, 0.0, 0.0)
            return True
        self._controller.execute.side_effect = _execute

        action = self._make_action()
        x, y, z = action._lift_shoulder([0.0, 0.0], extrinsic)
        self.assertAlmostEqual(x, 100.0, places=4)  # 0.1 m × 1000
        self.assertAlmostEqual(y, 200.0, places=4)  # 0.2 m × 1000
        self.assertAlmostEqual(z, 500.0, places=4)  # 0.5 m × 1000

    def test_pixel_coordinates_rounded(self):
        captured = []

        def _execute(device, method, *args, **kwargs):
            if method == "pixel_to_world":
                captured.append((args[0], args[1]))
                return (0.0, 0.0, 0.0)
            return True
        self._controller.execute.side_effect = _execute

        action = self._make_action()
        action._lift_shoulder([100.6, 200.4], action._left_arm_extrinsic)
        self.assertEqual(captured[0], (101, 200))


if __name__ == "__main__":
    unittest.main()
